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


class DecodeInstance:
    node_id: int
    hardware: GPUHardwareSpec
    queue: list[tuple[Request, float]]
    download_queue: list[tuple[DownloadRequest, float]]
    upload_queue: list[tuple[UploadRequest, float]]
    max_batch_size: int
    model: Model
    session: Any
    cache: Cache | None
    scheduler: BandwidthScheduler | None

    def __init__(
        self, node_id: int, hardware: GPUHardwareSpec, max_batch_size: int, model: Model
    ):
        self.node_id = node_id
        self.hardware = hardware
        self.queue = []
        self.download_queue = []
        self.upload_queue = []
        self.max_batch_size = max_batch_size
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
        dt = self.cache.download_kv(self.node_id, request)
        log(
            LOG_INSTANCE,
            f"Adding request {request.id} to decode instance {self.node_id}, "
            f"with {dt.remaining_bytes} bytes to download across {len(dt.legs)} legs",
        )
        if dt.active_leg:
            self.scheduler.register(dt)
            self.download_queue.append((dt, -1))
        else:
            self.queue.append((request, -1))

    def time_to_next_completion(self) -> float:
        """Return the remaining time until the active decode batch finishes one token.

        Transfer completion times are handled globally by the
        ``BandwidthScheduler``; this method only reports compute events.
        """
        batch = self.queue[: self.max_batch_size]
        if batch:
            return self.calculate_decode_time(batch)
        return float("inf")

    def process_queue(self, time_ms: float) -> list[Request]:
        assert self.cache is not None, "Cache must be set before processing queue"
        assert self.scheduler is not None, (
            "Scheduler must be set before processing queue"
        )

        log(
            LOG_INSTANCE,
            f"Processing decode queue for node {self.node_id} with time_ms: {time_ms}",
        )

        assert time_ms >= 0, "Time to process queue should be non-negative"
        kv_cache = 0
        for req, _ in self.queue:
            kv_cache += self.model.kv_size_per_token * req.isl
        assert kv_cache <= self.hardware.gpu_mem, (
            "KV cache exceeds GPU memory, too many requests in decode queue"
        )

        finished_requests: list[Request] = []

        # Active download. Download and upload may overlap on the same instance.
        if self.download_queue:
            download_request, _ = self.download_queue[0]
            download_request.request.kv_download_time_ms += time_ms
            leg = download_request.active_leg
            if leg:
                bytes_done = leg.bandwidth_bytes_per_ms * time_ms
                leg.remaining_bytes -= bytes_done
                if leg.remaining_bytes <= 0:
                    self.scheduler.unregister(download_request)
                    has_more = download_request.advance_leg()
                    if has_more:
                        self.scheduler.register(download_request)
                    else:
                        self.queue.append((download_request.request, 0))
                        self.download_queue.pop(0)

        # Active decode batch
        batch = self.queue[: self.max_batch_size]
        finished_in_batch: list[Request] = []
        if batch:
            decode_time_per_token = self.calculate_decode_time(batch)
            for request, _ in batch:
                request.decode_time_ms += time_ms
                if request.remaining_decode_time_ms == -1:
                    request.remaining_decode_time_ms = decode_time_per_token
                request.remaining_decode_time_ms -= time_ms
                if request.remaining_decode_time_ms <= 0:
                    request.remaining_decode_time_ms = 0
                    request.decoded_tokens += 1
                    if request.decoded_tokens >= request.osl:
                        log(
                            LOG_INSTANCE,
                            f"Finishing request decode with id: {request.id} after "
                            f"{request.decode_time_ms / 1000} seconds + process queue",
                        )
                        finished_in_batch.append(request)
                        ur = self.cache.upload_kv(self.node_id, request)
                        if ur.active_leg:
                            self.scheduler.register(ur)
                        self.upload_queue.append((ur, 0))

        # Remove finished requests from the decode queue by identity so that
        # finishing a request behind the head does not evict unrelated requests.
        if finished_in_batch:
            finished_set = {id(r) for r in finished_in_batch}
            self.queue = [(r, t) for r, t in self.queue if id(r) not in finished_set]

        # Active upload
        if self.upload_queue:
            upload_request, _ = self.upload_queue[0]
            upload_request.request.kv_upload_time_ms += time_ms
            leg = upload_request.active_leg
            if leg:
                bytes_done = leg.bandwidth_bytes_per_ms * time_ms
                leg.remaining_bytes -= bytes_done
                if leg.remaining_bytes <= 0:
                    self.scheduler.unregister(upload_request)
                    has_more = upload_request.advance_leg()
                    if has_more:
                        self.scheduler.register(upload_request)
                    else:
                        finished_requests.append(upload_request.request)
                        self.upload_queue.pop(0)

        return finished_requests

    def calculate_decode_time(self, batch: list[tuple[Request, float]]) -> float:
        avg_isl = int(
            sum(req.prefilled_tokens + req.decoded_tokens for req, _ in batch)
            / len(batch)
        )
        result = run_static_inference(
            mode="decode",
            built_session=self.session,
            isl=avg_isl,
            osl=2,
            prefix=avg_isl,
            batch_size=len(batch),
            stride=10,
        )
        time_ms = result["decode_latency_ms"]

        log(
            LOG_INSTANCE,
            f"Calculated decode time for batch"
            f"{[req.prefilled_tokens + req.decoded_tokens for req, _ in batch]} "
            f"of size {len(batch)}: {time_ms} ms",
        )
        return time_ms

    def log(self):
        log(
            LOG_INSTANCE,
            f"Decode instance state: {len(self.queue)} requests in queue, "
            f"{len(self.download_queue)} requests in download queue",
        )
        for request, _ in self.queue:
            log(
                LOG_INSTANCE,
                f"Request id: {request.id}, decoded tokens: {request.decoded_tokens}, "
                f"remaining tokens decode: {request.osl - request.decoded_tokens}, "
                f"decode time ms: {request.decode_time_ms}",
            )
