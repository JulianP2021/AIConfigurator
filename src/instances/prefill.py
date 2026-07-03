from typing import Any

from src.aiconfigurator_lib.estimator import (
    build_session,
    get_meta,
    run_static_inference,
)
from src.cache.cache import Cache
from src.hardware.hardware import GPUHardwareSpec
from src.logger import LOG_INSTANCE, log
from src.model.model import Model
from src.request.request import DownloadRequest, Request, UploadRequest
from src.scheduler.bandwidth_scheduler import BandwidthScheduler


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

    def __init__(self, node_id: int, hardware: GPUHardwareSpec, model: Model):
        self.node_id = node_id
        self.hardware = hardware
        self.queue = []
        self.upload_queue = []
        self.download_queue = []
        self.cache = None
        self.scheduler = None
        self.model = model

        system_name, backend_version = get_meta(
            backend_version="",
            mem_bw=self.hardware.gpu_bw,
            mem_capacity=self.hardware.gpu_mem,
            bfloat16_tc_flops=self.hardware.flops,
        )
        self.session = build_session(
            model_name=self.model.name,
            system_name=system_name,
            backend_name="vllm",
            backend_version=backend_version,
            database_mode="SOL",
        )

    def set_cache(self, cache: Cache):
        self.cache = cache

    def set_scheduler(self, scheduler: BandwidthScheduler):
        self.scheduler = scheduler

    def add_request(self, request: Request):
        assert self.cache is not None, "Cache must be set before adding requests"
        assert self.scheduler is not None, (
            "Scheduler must be set before adding requests"
        )
        dr = self.cache.download_kv(self.node_id, request)
        if dr.active_legs:
            self.scheduler.register(dr)
            self.download_queue.append((dr, -1))
        else:
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
        assert len(self.queue) < 100, "Too many requests in prefill queue"
        assert self.cache is not None, "Cache must be set before processing queue"
        assert self.scheduler is not None, (
            "Scheduler must be set before processing queue"
        )
        assert time_ms >= 0, "Time to process queue should be non-negative"

        log(
            LOG_INSTANCE,
            f"Processing prefill queue for node {self.node_id} with time_ms: {time_ms}",
        )

        finished_requests: list[Request] = []

        # Accumulate total time for every request currently in a prefill-side
        # transfer queue.  Active time will be reconciled when the transfer is
        # popped, leaving wait = total - active.
        for download_request, _ in self.download_queue:
            download_request.request.prefill_download_total_ms += time_ms
        for upload_request, _ in self.upload_queue:
            upload_request.request.prefill_upload_total_ms += time_ms

        # Requests in the prefill compute queue but not at the head are waiting.
        for request, _ in self.queue[1:]:
            request.prefill_wait_ms += time_ms

        # Drain transfers that the global scheduler has fully completed.  The
        # transfer objects are shared between the instance queue and the
        # scheduler; a finished transfer has no active leg left.
        while self.download_queue and not self.download_queue[0][0].active_legs:
            download_request, _ = self.download_queue.pop(0)
            request = download_request.request
            request.prefill_download_active_ms = (
                download_request.active_transfer_duration_ms
            )
            request.prefill_download_wait_ms = (
                request.prefill_download_total_ms - request.prefill_download_active_ms
            )
            self.queue.append((request, 0))

        while self.upload_queue and not self.upload_queue[0][0].active_legs:
            upload_request, _ = self.upload_queue.pop(0)
            request = upload_request.request
            request.prefill_upload_active_ms = (
                upload_request.active_transfer_duration_ms
            )
            request.prefill_upload_wait_ms = (
                request.prefill_upload_total_ms - request.prefill_upload_active_ms
            )
            # Keep backward-compatible totals.
            request.kv_upload_time_ms += request.prefill_upload_active_ms
            finished_requests.append(request)

        # Also handle the case where the head upload finished in the same step
        # that the prefill itself finished.  In that situation the upload
        # object is still at the head of upload_queue and has no active leg.
        while self.upload_queue and not self.upload_queue[0][0].active_legs:
            upload_request, _ = self.upload_queue.pop(0)
            request = upload_request.request
            request.prefill_upload_active_ms = (
                upload_request.active_transfer_duration_ms
            )
            request.prefill_upload_wait_ms = (
                request.prefill_upload_total_ms - request.prefill_upload_active_ms
            )
            request.kv_upload_time_ms += request.prefill_upload_active_ms
            finished_requests.append(request)

        # Active prefill
        if self.queue:
            request, _ = self.queue[0]
            if request.remaining_prefill_time_ms == -1:
                request.remaining_prefill_time_ms = self.calculate_prefill_time(request)
            request.prefill_time_ms += time_ms
            request.remaining_prefill_time_ms -= time_ms
            if request.remaining_prefill_time_ms <= 0:
                request.remaining_prefill_time_ms = 0
                request.prefilled_tokens += request.remaining_tokens_prefill
                request.decoded_tokens = 1
                log(
                    LOG_INSTANCE,
                    f"Finished prefill for request with id: {request.id} after "
                    f"{request.prefill_time_ms / 1000} seconds + process queue",
                )
                self.queue.pop(0)
                ur = self.cache.upload_kv(self.node_id, request)
                assert ur.active_legs, (
                    f"Prefill upload for request {request.id} (user {request.user_id}, "
                    f"session {request.session_id}) on node {self.node_id} has no active legs"
                )
                self.scheduler.register(ur)
                self.upload_queue.append((ur, 0))

        return finished_requests

    def calculate_prefill_time(self, request: Request) -> float:
        result = run_static_inference(
            mode="prefill",
            built_session=self.session,
            isl=request.isl,
            osl=1,
            prefix=request.prefilled_tokens,
        )
        if result is None or "prefill_latency_ms" not in result:
            raise ValueError(
                f"Prefill latency not found in result for request with id: {request.id}, result: {result}, hardware: {self.hardware}, model: {self.model}"
            )
        time_ms = result["prefill_latency_ms"]
        log(
            LOG_INSTANCE,
            f"Calculated prefill time for request with id: {request.id} : {time_ms} ms",
        )
        return time_ms

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
