from typing import Any

from src.cache.cache import Cache

# from src.aiconfigurator_lib.estimator import (
#     build_session,
#     get_meta,
#     run_static_inference,
# )
from src.eroors.errors import PrefillError
from src.hardware.hardware import GPUHardwareSpec
from src.logger import LOG_INSTANCE, log, should_log
from src.model.model import Model
from src.request.request import DownloadRequest, Request, UploadRequest
from src.scheduler.bandwidth_scheduler import BandwidthScheduler
from src.utils.utils import calculate_flops, calculate_memory


class PrefillInstance:
    node_id: int
    hardware: GPUHardwareSpec
    queue: list[tuple[Request, float]]
    upload_queue: list[tuple[UploadRequest, float]]
    background_upload_queue: list[UploadRequest]
    download_queue: list[tuple[DownloadRequest, float]]
    background_download_queue: list[DownloadRequest]
    cache: Cache | None
    scheduler: BandwidthScheduler | None

    model: Model
    session: Any
    max_batch_size: int

    def __init__(
        self,
        node_id: int,
        hardware: GPUHardwareSpec,
        model: Model,
        max_batch_size: int = 10,
    ):
        self.node_id = node_id
        self.hardware = hardware
        self.queue = []
        self.upload_queue = []
        self.background_upload_queue = []
        self.download_queue = []
        self.background_download_queue = []
        self.cache = None
        self.scheduler = None
        self.model = model
        self.max_batch_size = max_batch_size

        # system_name, backend_version = get_meta(
        #     backend_version="",
        #     mem_bw=self.hardware.gpu_bw,
        #     mem_capacity=self.hardware.gpu_mem,
        #     bfloat16_tc_flops=self.hardware.flops,
        # )
        # self.session = build_session(
        #     model_name=self.model.name,
        #     system_name=system_name,
        #     backend_name="vllm",
        #     backend_version=backend_version,
        #     database_mode="SOL",
        # )

    def set_cache(self, cache: Cache):
        self.cache = cache

    def set_scheduler(self, scheduler: BandwidthScheduler):
        self.scheduler = scheduler

    def _global_time_ms(self) -> float:
        """Return the scheduler's global time, or 0.0 if unavailable."""
        if self.scheduler is None:
            return 0.0
        return float(self.scheduler.time_ms)

    def add_request(self, request: Request):
        assert self.cache is not None, "Cache must be set before adding requests"
        assert self.scheduler is not None, (
            "Scheduler must be set before adding requests"
        )
        now = self._global_time_ms()
        request.prefill_download_start_ms = now
        dr = self.cache.download_kv(self.node_id, request)
        if dr.active_legs:
            self.scheduler.register(dr)
            self.download_queue.append((dr, -1))
        else:
            request.prefill_download_end_ms = now
            request.prefill_queue_start_ms = now
            if request.initial_prefilled_tokens is None:
                request.initial_prefilled_tokens = request.prefilled_tokens
            if not self.queue:
                request.prefill_start_ms = now
            # Eagerly compute the remaining prefill time so the router can sum
            # accurate remaining times for queued requests rather than merging
            # them into one approximate mega-request.
            if request.remaining_prefill_time_ms == -1:
                request.remaining_prefill_time_ms = self.calculate_prefill_time(request)
            self.queue.append((request, -1))

    def time_to_next_completion(self) -> float:
        """Return the remaining time until the active prefill finishes.

        Transfer completion times are handled globally by the
        ``BandwidthScheduler``; this method only reports compute events.

        If the head download in the queue has no active legs (e.g. a zero-byte
        download for a prefix that is already local), return 0 so the event
        loop drains it immediately.
        """
        if self.download_queue and self.download_queue[0][0].is_download_done():
            return 0.0
        if self.queue:
            request = self.queue[0][0]
            if request.remaining_prefill_time_ms == -1:
                request.remaining_prefill_time_ms = self.calculate_prefill_time(request)
            return request.remaining_prefill_time_ms
        return float("inf")

    def process_queue(self, time_ms: float) -> list[Request]:
        # Allow the prefill queue to grow to a multiple of the decode batch size
        # plus a small headroom; small batch sizes keep the original floor of 10.
        queue_limit = max(10, self.max_batch_size * 2)
        if len(self.queue) >= queue_limit:
            head = self.queue[0][0]
            sample_size = min(5, len(self.queue))
            sample_info = ", ".join(
                f"r{req.id}: isl={req.isl}, osl={req.osl}, "
                f"prefilled={req.prefilled_tokens}"
                for req, _ in self.queue[:sample_size]
            )
            raise PrefillError(
                f"Too many requests in prefill queue for node {self.node_id}: "
                f"{len(self.queue)} requests (limit={queue_limit}). "
                f"Head request r{head.id}: isl={head.isl}, osl={head.osl}, "
                f"prefilled={head.prefilled_tokens}. "
                f"Sample requests (first {sample_size}): {sample_info}"
            )
        assert self.cache is not None, "Cache must be set before processing queue"
        assert self.scheduler is not None, (
            "Scheduler must be set before processing queue"
        )
        assert time_ms >= 0, "Time to process queue should be non-negative"

        now = self._global_time_ms()
        if should_log(LOG_INSTANCE):
            log(
                LOG_INSTANCE,
                f"[t={now:.3f} ms] Processing prefill queue for node {self.node_id} "
                f"with time_ms: {time_ms}",
            )

        finished_requests: list[Request] = []

        # Drain completed prefill downloads. Only the data tracks (not the
        # background eviction tracks) gate prefill start; evictions are kept in
        # the scheduler and finish asynchronously.
        while self.download_queue and self.download_queue[0][0].is_download_done():
            download_request, _ = self.download_queue.pop(0)
            request = download_request.request
            request.prefill_download_end_ms = now
            request.prefill_download_active_ms = (
                download_request.download_active_duration_ms()
            )
            request.prefill_queue_start_ms = now
            if request.initial_prefilled_tokens is None:
                # Capture the cached prefix length after the download has merged
                # available KV into local RAM. For fresh requests this is zero;
                # for cache hits it equals the effective downloaded prefix.
                request.initial_prefilled_tokens = request.prefilled_tokens
            self.queue.append((request, 0))
            # Keep the DownloadRequest around so the full background eviction
            # duration can be captured once every track is exhausted.
            self.background_download_queue.append(download_request)

        # Drain completed background downloads and record their eviction duration.
        still_running: list[DownloadRequest] = []
        for download_request in self.background_download_queue:
            if download_request.is_complete():
                request = download_request.request
                request.prefill_download_background_active_ms = (
                    download_request.download_background_active_duration_ms()
                )
            else:
                still_running.append(download_request)
        self.background_download_queue = still_running

        # Drain completed prefill uploads.  The actual upload is the last track;
        # once it finishes the request is considered uploaded and can move on,
        # while any eviction tracks keep running in the background.
        while self.upload_queue and self.upload_queue[0][0].is_upload_done():
            upload_request, _ = self.upload_queue.pop(0)
            request = upload_request.request
            request.prefill_upload_end_ms = now
            request.prefill_upload_active_ms = (
                upload_request.upload_active_duration_ms()
            )
            finished_requests.append(request)
            # Keep the UploadRequest around so the full background eviction
            # duration can be captured once every track is exhausted.
            self.background_upload_queue.append(upload_request)

        # Drain completed background uploads and record their eviction duration.
        still_running: list[UploadRequest] = []
        for upload_request in self.background_upload_queue:
            if upload_request.is_complete():
                request = upload_request.request
                request.prefill_upload_background_active_ms = (
                    upload_request.background_active_duration_ms()
                )
            else:
                still_running.append(upload_request)
        self.background_upload_queue = still_running

        # Active prefill
        if self.queue:
            request, _ = self.queue[0]
            if request.prefill_start_ms is None:
                request.prefill_start_ms = now
            if request.remaining_prefill_time_ms == -1:
                request.remaining_prefill_time_ms = self.calculate_prefill_time(request)
            request.remaining_prefill_time_ms -= time_ms
            if request.remaining_prefill_time_ms <= 0:
                request.remaining_prefill_time_ms = 0
                request.prefilled_tokens += request.remaining_tokens_prefill
                request.decoded_tokens = 1
                request.prefill_end_ms = now
                if should_log(LOG_INSTANCE):
                    log(
                        LOG_INSTANCE,
                        f"[t={now:.3f} ms] Finished prefill for request with id: "
                        f"{request.id}",
                    )
                self.queue.pop(0)
                ur = self.cache.upload_kv(self.node_id, request)
                if ur.active_legs:
                    request.prefill_upload_start_ms = now
                    self.scheduler.register(ur)
                    self.upload_queue.append((ur, 0))
                else:
                    # The full KV was already cached locally; nothing to upload.
                    request.prefill_upload_start_ms = now
                    request.prefill_upload_end_ms = now
                    finished_requests.append(request)

                # The next request in line (if any) starts prefill service now.
                if self.queue:
                    self.queue[0][0].prefill_start_ms = now

        return finished_requests

    def calculate_prefill_time(self, request: Request) -> int:
        flops = calculate_flops(self.model, [(request, 0)], "prefill")
        memory = calculate_memory(self.model, [(request, 0)], "prefill")

        time_ms: int = int(
            max(
                float(flops) / self.hardware.flops,
                float(memory) / self.hardware.gpu_bw,
            )
            * 1000
        )
        if should_log(LOG_INSTANCE):
            log(
                LOG_INSTANCE,
                f"Calculated prefill time for request with id: {request.id} : {time_ms} ms",
            )
        return time_ms

    # def calculate_prefill_time(self, request: Request) -> float:
    #     result = run_static_inference(
    #         mode="prefill",
    #         built_session=self.session,
    #         isl=request.isl,
    #         osl=1,
    #         prefix=request.prefilled_tokens,
    #     )
    #     if result is None or "prefill_latency_ms" not in result:
    #         raise ValueError(
    #             f"Prefill latency not found in result for request with id: {request.id}, result: {result}, hardware: {self.hardware}, model: {self.model}"
    #         )
    #     time_ms = result["prefill_latency_ms"]
    #     log(
    #         LOG_INSTANCE,
    #         f"Calculated prefill time for request with id: {request.id} : {time_ms} ms",
    #     )
    #     return time_ms

    def log(self):
        if should_log(LOG_INSTANCE):
            log(
                LOG_INSTANCE,
                f"Prefill instance state: {len(self.queue)} requests in queue, "
                f"{len(self.upload_queue)} requests in upload queue, "
                f"{len(self.background_upload_queue)} background uploads",
            )
            for request, _ in self.queue:
                log(
                    LOG_INSTANCE,
                    f"Request id: {request.id}, prefilled tokens: {request.prefilled_tokens}, "
                    f"remaining tokens prefill: {request.remaining_tokens_prefill}, "
                    f"prefill time ms: {request.prefill_time_ms}, "
                    f"remaining prefill time ms: {request.remaining_prefill_time_ms}",
                )
