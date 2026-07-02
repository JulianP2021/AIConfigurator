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
        if dr.active_leg:
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

        # Drain transfers that the global scheduler has fully completed.  The
        # transfer objects are shared between the instance queue and the
        # scheduler; a finished transfer has no active leg left.
        while self.download_queue and self.download_queue[0][0].active_leg is None:
            download_request, _ = self.download_queue.pop(0)
            self.queue.append((download_request.request, 0))

        while self.upload_queue and self.upload_queue[0][0].active_leg is None:
            upload_request, _ = self.upload_queue.pop(0)
            finished_requests.append(upload_request.request)

        # Also handle the case where the head upload finished in the same step
        # that the prefill itself finished.  In that situation the upload
        # object is still at the head of upload_queue and has no active leg.
        while self.upload_queue and self.upload_queue[0][0].active_leg is None:
            upload_request, _ = self.upload_queue.pop(0)
            finished_requests.append(upload_request.request)

        # Active download.  Count elapsed transfer time for the request stats.
        if self.download_queue:
            download_request, _ = self.download_queue[0]
            download_request.request.kv_download_time_ms += time_ms

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
                if ur.active_leg:
                    self.scheduler.register(ur)
                self.upload_queue.append((ur, 0))

        # Active upload.  Count elapsed transfer time for the request stats.
        if self.upload_queue:
            upload_request, _ = self.upload_queue[0]
            upload_request.request.kv_upload_time_ms += time_ms

        return finished_requests

    def calculate_prefill_time(self, request: Request) -> float:
        result = run_static_inference(
            mode="prefill",
            built_session=self.session,
            isl=request.isl,
            osl=1,
            prefix=request.prefilled_tokens,
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
