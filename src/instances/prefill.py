from typing import Any

from src.cache.cache import Cache

# from src.aiconfigurator_lib.estimator import (
#     build_session,
#     get_meta,
#     run_static_inference,
# )
from src.eroors.errors import PrefillError
from src.hardware.hardware import GPUHardwareSpec
from src.logger import LOG_INSTANCE, log
from src.model.model import Model
from src.request.request import DownloadRequest, Request, UploadRequest
from src.scheduler.bandwidth_scheduler import BandwidthScheduler
from src.utils.utils import calculate_flops, calculate_memory


class PrefillInstance:
    node_id: int
    hardware: GPUHardwareSpec
    queue: list[tuple[Request, float]]
    upload_queue: list[tuple[UploadRequest, float]]
    download_queue: list[tuple[DownloadRequest, float]]
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
        self.download_queue = []
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
            if not self.queue:
                request.prefill_start_ms = now
            self.queue.append((request, -1))

    def time_to_next_completion(self) -> float:
        """Return the remaining time until the active prefill finishes.

        Transfer completion times are handled globally by the
        ``BandwidthScheduler``; this method only reports compute events.
        """
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
            raise PrefillError(
                f"Too many requests in prefill queue for node {self.node_id}: "
                f"{len(self.queue)} requests (limit={queue_limit})"
            )
        assert self.cache is not None, "Cache must be set before processing queue"
        assert self.scheduler is not None, (
            "Scheduler must be set before processing queue"
        )
        assert time_ms >= 0, "Time to process queue should be non-negative"

        now = self._global_time_ms()
        log(
            LOG_INSTANCE,
            f"[t={now:.3f} ms] Processing prefill queue for node {self.node_id} "
            f"with time_ms: {time_ms}",
        )

        finished_requests: list[Request] = []

        # Drain completed prefill downloads. The scheduler already advanced the
        # transfer; we just record the end timestamp and active duration.
        while self.download_queue and not self.download_queue[0][0].active_legs:
            download_request, _ = self.download_queue.pop(0)
            request = download_request.request
            request.prefill_download_end_ms = now
            request.prefill_download_active_ms = (
                download_request.active_transfer_duration_ms
            )
            request.prefill_queue_start_ms = now
            self.queue.append((request, 0))

        # Drain completed prefill uploads.
        while self.upload_queue and not self.upload_queue[0][0].active_legs:
            upload_request, _ = self.upload_queue.pop(0)
            request = upload_request.request
            request.prefill_upload_end_ms = now
            request.prefill_upload_active_ms = (
                upload_request.active_transfer_duration_ms
            )
            finished_requests.append(request)

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
                log(
                    LOG_INSTANCE,
                    f"[t={now:.3f} ms] Finished prefill for request with id: "
                    f"{request.id}",
                )
                self.queue.pop(0)
                ur = self.cache.upload_kv(self.node_id, request)
                assert ur.active_legs, (
                    f"Prefill upload for request {request.id} (user {request.user_id}, "
                    f"session {request.session_id}) on node {self.node_id} has no active legs"
                )
                request.prefill_upload_start_ms = now
                self.scheduler.register(ur)
                self.upload_queue.append((ur, 0))

                # The next request in line (if any) starts prefill service now.
                if self.queue:
                    self.queue[0][0].prefill_start_ms = now

        return finished_requests

    def calculate_prefill_time(self, request: Request) -> int:
        flops = calculate_flops(self.model, [request], "prefill")
        memory = calculate_memory(self.model, [request], "prefill")

        time_ms: int = int(
            float(flops) / self.hardware.flops * 1000
            + float(memory) / self.hardware.gpu_bw * 1000
        )
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
        log(
            LOG_INSTANCE,
            f"Prefill instance state: {len(self.queue)} requests in queue, "
            f"{len(self.upload_queue)} requests in upload queue",
        )
        for request, _ in self.queue:
            log(
                LOG_INSTANCE,
                f"Request id: {request.id}, prefilled tokens: {request.prefilled_tokens}, "
                f"remaining tokens prefill: {request.remaining_tokens_prefill}, "
                f"prefill time ms: {request.prefill_time_ms}, "
                f"remaining prefill time ms: {request.remaining_prefill_time_ms}",
            )
