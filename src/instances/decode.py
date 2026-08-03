from typing import Any

from src.cache.cache import Cache

# from src.aiconfigurator_lib.estimator import (
#     build_session,
#     get_meta,
#     run_static_inference,
# )
from src.eroors.errors import DecodeError, KVStoreTooSmallError
from src.hardware.hardware import GPUHardwareSpec
from src.logger import LOG_INSTANCE, log, should_log
from src.model.model import Model
from src.request.request import DownloadRequest, Request, UploadRequest
from src.scheduler.bandwidth_scheduler import BandwidthScheduler
from src.utils.utils import calculate_flops, calculate_memory


# Number of decode tokens a frozen batch commits to generate before the batch
# is recomputed.  Using a fixed commitment prevents background transfer events
# from breaking the stride apart and perturbing decode timing.
BATCH_TOKEN_COMMITMENT = 32


class DecodeInstance:
    node_id: int
    hardware: GPUHardwareSpec
    queue: list[tuple[Request, float]]
    download_queue: list[tuple[DownloadRequest, float]]
    background_download_queue: list[DownloadRequest]
    upload_queue: list[tuple[UploadRequest, float]]
    background_upload_queue: list[UploadRequest]
    max_batch_size: int
    model: Model
    session: Any
    cache: Cache | None
    scheduler: BandwidthScheduler | None

    # Decode runs in frozen batches that commit to a fixed number of tokens.
    # These fields track the instance-level progress for the current batch.
    current_batch: list[Request] | None
    remaining_batch_time_ms: float | None
    current_batch_decode_time_ms: float | None
    current_batch_tokens_remaining: int | None

    def __init__(
        self, node_id: int, hardware: GPUHardwareSpec, max_batch_size: int, model: Model
    ):
        self.node_id = node_id
        self.hardware = hardware
        self.queue = []
        self.download_queue = []
        self.background_download_queue = []
        self.upload_queue = []
        self.background_upload_queue = []
        self.max_batch_size = max_batch_size
        self.cache = None
        self.scheduler = None
        self.model = model
        self.current_batch = None
        self.remaining_batch_time_ms = None
        self.current_batch_decode_time_ms = None
        self.current_batch_tokens_remaining = None
        self._kv_cache_bytes: int = 0

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
        request.decode_download_start_ms = now
        dt = self.cache.download_kv(self.node_id, request)
        if should_log(LOG_INSTANCE):
            log(
                LOG_INSTANCE,
                f"Adding request {request.id} to decode instance {self.node_id}, "
                f"with {dt.remaining_bytes} bytes to download across {len(dt.tracks)} tracks",
            )
        if dt.active_legs:
            self.scheduler.register(dt)
            self.download_queue.append((dt, -1))
        else:
            request.decode_download_end_ms = now
            request.decode_queue_start_ms = now
            if not self.queue and not self.current_batch:
                request.decode_start_ms = now
            self.queue.append((request, -1))
            self._kv_cache_bytes += self.model.kv_size_per_token * request.cache_length
            # New arrivals cannot join a frozen batch mid-commitment; they will
            # be picked up when the current batch finishes its committed tokens.

    def _reset_batch_state(self) -> None:
        """Unfreeze the current batch and clear all batch progress state."""
        self.current_batch = None
        self.remaining_batch_time_ms = None
        self.current_batch_decode_time_ms = None
        self.current_batch_tokens_remaining = None

    def _ensure_batch(self) -> None:
        """Freeze a new batch from the head of the queue when none is active."""
        if self.current_batch is not None:
            return

        batch = [req for req, _ in self.queue[: self.max_batch_size]]
        if not batch:
            return

        now = self._global_time_ms()
        for req in batch:
            if req.decode_start_ms is None:
                req.decode_start_ms = now
            if req.decode_queue_start_ms is None:
                req.decode_queue_start_ms = now

        self.current_batch = batch
        self.current_batch_decode_time_ms = self.calculate_decode_time([
            (r, 0) for r in batch
        ])
        self.remaining_batch_time_ms = self.current_batch_decode_time_ms

        # Commit to a fixed number of tokens before recomputing the batch.
        # If any request can finish sooner, cap the commitment so the batch
        # is recomputed as soon as that request leaves.
        min_remaining_tokens = min(r.osl - r.decoded_tokens for r in batch)
        self.current_batch_tokens_remaining = min(
            BATCH_TOKEN_COMMITMENT, max(1, min_remaining_tokens)
        )

    def time_to_next_completion(self) -> float:
        """Return time until the current frozen batch completes its commitment.

        A batch commits to generating a fixed number of tokens before it is
        recomputed.  This method reports the full time until that commitment is
        fulfilled, allowing the event loop to stride multiple tokens at once.
        Background transfer events that occur before the commitment completes
        do not change the batch.

        If the head download in the queue has no active legs (e.g. a zero-byte
        download for a prefix that is already local), return 0 so the event
        loop drains it immediately.
        """
        if self.download_queue and self.download_queue[0][0].is_download_done():
            return 0.0
        self._ensure_batch()
        if not self.current_batch:
            return float("inf")

        if self.current_batch_decode_time_ms is None:
            self.current_batch_decode_time_ms = self.calculate_decode_time([
                (r, 0) for r in self.current_batch
            ])

        if self.remaining_batch_time_ms is None:
            self.remaining_batch_time_ms = self.current_batch_decode_time_ms

        assert self.current_batch_tokens_remaining is not None
        assert self.current_batch_tokens_remaining > 0

        # Time for the in-flight token plus the remaining committed tokens.
        return self.remaining_batch_time_ms + self.current_batch_decode_time_ms * (
            self.current_batch_tokens_remaining - 1
        )

    def process_queue(self, time_ms: float) -> list[Request]:
        assert self.cache is not None, "Cache must be set before processing queue"
        assert self.scheduler is not None, (
            "Scheduler must be set before processing queue"
        )

        now = self._global_time_ms()
        log(
            LOG_INSTANCE,
            f"[t={now:.3f} ms] Processing decode queue for node {self.node_id} "
            f"with time_ms: {time_ms}",
        )

        assert time_ms >= 0, "Time to process queue should be non-negative"
        if self._kv_cache_bytes > self.hardware.gpu_mem:
            log(
                LOG_INSTANCE,
                f"KV cache exceeds GPU memory for node {self.node_id}: "
                f"{self._kv_cache_bytes} bytes used by {len(self.queue)} requests with {sum(r.cache_length for r, _ in self.queue) / len(self.queue) if self.queue else 0} avg tokens , {self.hardware.gpu_mem} bytes available",
            )

            head = self.queue[0][0]
            sample_size = min(5, len(self.queue))
            sample_info = ", ".join(
                f"r{req.id}: isl={req.isl}, osl={req.osl}, "
                f"cache_length={req.cache_length}"
                for req, _ in self.queue[:sample_size]
            )
            raise DecodeError(
                f"KV cache exceeds GPU memory for node {self.node_id}: "
                f"{self._kv_cache_bytes} bytes used, {self.hardware.gpu_mem} bytes available. "
                f"Head request r{head.id}: isl={head.isl}, osl={head.osl}, "
                f"cache_length={head.cache_length}. "
                f"Sample requests (first {sample_size}): {sample_info}"
            )

        finished_requests: list[Request] = []

        # Drain completed decode downloads.  Only the data tracks (not the
        # background eviction tracks) gate decode start.  Completed downloads
        # are appended to the queue but do NOT interrupt a frozen batch; they
        # will join the next batch once the current commitment is fulfilled.
        while self.download_queue and self.download_queue[0][0].is_download_done():
            download_request, _ = self.download_queue.pop(0)
            request = download_request.request
            request.decode_download_end_ms = now
            request.decode_download_active_ms = (
                download_request.download_active_duration_ms()
            )
            request.decode_queue_start_ms = now
            self.queue.append((request, 0))
            self._kv_cache_bytes += self.model.kv_size_per_token * request.cache_length
            # Keep the DownloadRequest around so the full background eviction
            # duration can be captured once every track is exhausted.
            self.background_download_queue.append(download_request)

            if request.prefilled_tokens < request.isl:
                raise KVStoreTooSmallError(
                    "KV download did not return all prefilles tokens"
                )

        # Drain completed background downloads and record their eviction duration.
        still_running: list[DownloadRequest] = []
        for download_request in self.background_download_queue:
            if download_request.is_complete():
                request = download_request.request
                request.decode_download_background_active_ms = (
                    download_request.download_background_active_duration_ms()
                )
            else:
                still_running.append(download_request)
        self.background_download_queue = still_running

        # Drain completed decode uploads.  The actual upload is the last track;
        # once it finishes the request is done, while any eviction tracks keep
        # running in the background.
        while self.upload_queue and self.upload_queue[0][0].is_upload_done():
            upload_request, _ = self.upload_queue.pop(0)
            request = upload_request.request
            request.decode_upload_end_ms = now
            request.decode_upload_active_ms = upload_request.upload_active_duration_ms()
            finished_requests.append(request)
            # Remove the request's KV from the decode instance working set.
            self._kv_cache_bytes = max(
                0,
                int(self._kv_cache_bytes)
                - int(self.model.kv_size_per_token) * int(request.cache_length),
            )
            # Keep the UploadRequest around so the full background eviction
            # duration can be captured once every track is exhausted.
            self.background_upload_queue.append(upload_request)

        # Drain completed background uploads and record their eviction duration.
        still_running: list[UploadRequest] = []
        for upload_request in self.background_upload_queue:
            if upload_request.is_complete():
                request = upload_request.request
                request.decode_upload_background_active_ms = (
                    upload_request.background_active_duration_ms()
                )
            else:
                still_running.append(upload_request)
        self.background_upload_queue = still_running

        # Active decode batch.  A frozen batch commits to a fixed number of
        # tokens; it is only unfrozen when that commitment is exhausted or
        # the batch becomes empty.
        self._ensure_batch()
        if self.current_batch:
            assert self.current_batch_tokens_remaining is not None
            assert self.current_batch_decode_time_ms is not None
            assert self.remaining_batch_time_ms is not None

            self.remaining_batch_time_ms -= time_ms

            # Consume tokens until the commitment is fulfilled or there is no
            # more time budget.  Per-token decode time is recomputed each time
            # a token completes because the average ISL grows by one.
            while (
                self.remaining_batch_time_ms <= 0
                and self.current_batch
                and self.current_batch_tokens_remaining > 0
            ):
                old_decode_time = self.current_batch_decode_time_ms
                next_decode_time = self.calculate_decode_time([
                    (r, 0) for r in self.current_batch
                ])
                self.current_batch_decode_time_ms = next_decode_time

                # A request may already have reached its OSL; in that case
                # there are no remaining tokens to decode and we must stop
                # before applying a zero-width stride that would loop forever.
                remaining_tokens = min(
                    r.osl - r.decoded_tokens for r in self.current_batch
                )
                if remaining_tokens <= 0:
                    break

                # Stride multiple tokens when the decode time is stable, but
                # never beyond the batch's token commitment.
                if old_decode_time and abs(next_decode_time - old_decode_time) < 1e-9:
                    stride = int(-self.remaining_batch_time_ms / next_decode_time) + 1
                    stride = min(
                        self.current_batch_tokens_remaining,
                        remaining_tokens,
                        max(1, stride),
                    )
                    self.remaining_batch_time_ms += next_decode_time * stride
                else:
                    stride = 1
                    self.remaining_batch_time_ms += old_decode_time or next_decode_time

                for req in self.current_batch:
                    req.decoded_tokens += stride
                self.current_batch_tokens_remaining -= stride

            # Mark requests that finished decoding in this step.
            finished_in_batch: list[Request] = []
            for request in self.current_batch:
                if request.decoded_tokens >= request.osl:
                    if should_log(LOG_INSTANCE):
                        log(
                            LOG_INSTANCE,
                            f"[t={now:.3f} ms] Finishing request decode with id: "
                            f"{request.id}",
                        )
                    finished_in_batch.append(request)
                    request.decode_end_ms = now
                    ur = self.cache.upload_kv(self.node_id, request)
                    assert ur.active_legs, (
                        f"Decode upload for request {request.id} (user {request.user_id}, "
                        f"session {request.session_id}) on node {self.node_id} has no active legs"
                    )
                    request.decode_upload_start_ms = now
                    self.scheduler.register(ur)
                    self.upload_queue.append((ur, 0))

            # Remove finished requests from the queue and the frozen batch.
            if finished_in_batch:
                finished_set = {id(r) for r in finished_in_batch}
                removed_cache_bytes = 0
                new_queue: list[tuple[Request, float]] = []
                for r, t in self.queue:
                    if id(r) in finished_set:
                        removed_cache_bytes += (
                            self.model.kv_size_per_token * r.cache_length
                        )
                    else:
                        new_queue.append((r, t))
                self.queue = new_queue
                self.current_batch = [
                    r for r in self.current_batch if id(r) not in finished_set
                ]
                self._kv_cache_bytes -= removed_cache_bytes

            # Unfreeze only when the commitment is exhausted or no requests are
            # left in the batch.
            if self.current_batch_tokens_remaining <= 0 or not self.current_batch:
                self._reset_batch_state()

        return finished_requests

    def calculate_decode_time(self, batch: list[tuple[Request, float]]) -> int:
        flops = calculate_flops(self.model, batch, "decode")
        memory = calculate_memory(self.model, batch, "decode")

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
                f"Calculated decode time for batch"
                f"{[req.prefilled_tokens + req.decoded_tokens for req, _ in batch]} "
                f"of size {len(batch)}: {time_ms} ms",
            )
        return time_ms

    # def calculate_decode_time(self, batch: list[tuple[Request, float]]) -> float:
    #     avg_isl = int(
    #         sum(req.prefilled_tokens + req.decoded_tokens for req, _ in batch)
    #         / len(batch)
    #     )

    #     time_ms = 0

    #     # result = run_static_inference(
    #     #     mode="decode",
    #     #     built_session=self.session,
    #     #     isl=avg_isl,
    #     #     osl=2,
    #     #     prefix=avg_isl,
    #     #     batch_size=len(batch),
    #     #     stride=10,
    #     # )
    #     # if result is None or "decode_latency_ms" not in result:
    #     #     raise ValueError(
    #     #         f"Decode latency not found in result for batch "
    #     #         f"{[req.prefilled_tokens + req.decoded_tokens for req, _ in batch]} "
    #     #         f"of size {len(batch)}, result: {result}, hardware: {self.hardware}, model: {self.model}"
    #     #     )
    #     # time_ms = result["decode_latency_ms"]

    #     log(
    #         LOG_INSTANCE,
    #         f"Calculated decode time for batch"
    #         f"{[req.prefilled_tokens + req.decoded_tokens for req, _ in batch]} "
    #         f"of size {len(batch)}: {time_ms} ms",
    #     )
    #     return time_ms

    def log(self):
        if should_log(LOG_INSTANCE):
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
