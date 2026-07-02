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

    # Decode runs in frozen batches of exactly one token. These fields track
    # the instance-level progress for the current batch.
    current_batch: list[Request] | None
    remaining_batch_time_ms: float | None
    current_batch_decode_time_ms: float | None

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
        self.current_batch = None
        self.remaining_batch_time_ms = None
        self.current_batch_decode_time_ms = None

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

    def _ensure_batch(self) -> None:
        """Freeze a new batch from the head of the queue when none is active."""
        if not self.current_batch:
            self.current_batch = [req for req, _ in self.queue[: self.max_batch_size]]
            self.remaining_batch_time_ms = None
            self.current_batch_decode_time_ms = None

    def time_to_next_completion(self) -> float:
        """Return a lower-bound time until one request in the batch finishes.

        This is intentionally a lowball estimate: remaining time for the current
        in-flight token plus the current per-token decode latency multiplied by
        the smallest number of remaining output tokens (minus the current one)
        for any request in the batch.

        Transfer completion times are handled globally by the
        ``BandwidthScheduler``; this method only reports compute events.
        """
        self._ensure_batch()
        if not self.current_batch:
            return float("inf")

        if self.current_batch_decode_time_ms is None:
            self.current_batch_decode_time_ms = self.calculate_decode_time([
                (r, 0) for r in self.current_batch
            ])

        if self.remaining_batch_time_ms is None:
            self.remaining_batch_time_ms = self.current_batch_decode_time_ms

        remaining_tokens = min(r.osl - r.decoded_tokens for r in self.current_batch)
        return self.remaining_batch_time_ms + self.current_batch_decode_time_ms * (
            remaining_tokens - 1
        )

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

        # Drain transfers that the global scheduler has fully completed.  The
        # transfer objects are shared between the instance queue and the
        # scheduler; a finished transfer has no active leg left.
        while self.download_queue and self.download_queue[0][0].active_leg is None:
            download_request, _ = self.download_queue.pop(0)
            self.queue.append((download_request.request, 0))

        while self.upload_queue and self.upload_queue[0][0].active_leg is None:
            upload_request, _ = self.upload_queue.pop(0)
            finished_requests.append(upload_request.request)

        # Active download. Count elapsed transfer time for the request stats.
        if self.download_queue:
            download_request, _ = self.download_queue[0]
            download_request.request.kv_download_time_ms += time_ms

        # Active decode batch
        self._ensure_batch()
        finished_in_batch: list[Request] = []
        if self.current_batch:
            if self.current_batch_decode_time_ms is None:
                self.current_batch_decode_time_ms = self.calculate_decode_time([
                    (r, 0) for r in self.current_batch
                ])

            if self.remaining_batch_time_ms is None:
                self.remaining_batch_time_ms = self.current_batch_decode_time_ms

            self.remaining_batch_time_ms -= time_ms

            # Count how many full tokens completed in this step. Recalculate
            # the per-token decode time each time a token completes, because
            # the average ISL in the batch grows by one.
            tokens_done = 0
            while self.remaining_batch_time_ms <= 0 and self.current_batch:
                tokens_done += 1
                # Update the batch's average ISL before recalculating the
                # next token's latency.
                for req in self.current_batch:
                    req.decoded_tokens += 1
                # Recompute per-token time for the now-longer sequences.
                next_decode_time = self.calculate_decode_time([
                    (r, 0) for r in self.current_batch
                ])
                self.remaining_batch_time_ms += next_decode_time
                self.current_batch_decode_time_ms = next_decode_time

            # Apply the elapsed wall-clock time to every request's total
            # decode time. Requests that completed a token already had their
            # decoded_tokens incremented above.
            for request in self.current_batch:
                request.decode_time_ms += time_ms
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

            # Remove finished requests from the queue and the frozen batch.
            if finished_in_batch:
                finished_set = {id(r) for r in finished_in_batch}
                self.queue = [
                    (r, t) for r, t in self.queue if id(r) not in finished_set
                ]
                self.current_batch = [
                    r for r in self.current_batch if id(r) not in finished_set
                ]

            # If we completed one or more tokens, or the batch is empty,
            # unfreeze so a new batch can be formed for the next token step.
            if tokens_done > 0 or not self.current_batch:
                self.current_batch = None
                self.remaining_batch_time_ms = None
                self.current_batch_decode_time_ms = None

        # Active upload. Count elapsed transfer time for the request stats.
        if self.upload_queue:
            upload_request, _ = self.upload_queue[0]
            upload_request.request.kv_upload_time_ms += time_ms

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
