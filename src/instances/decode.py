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


decode_id_counter: int = 0


class DecodeInstance:
    instance_id: int
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
    refresh_batch: bool = True

    # Decode runs in frozen batches of exactly one token. These fields track
    # the instance-level progress for the current batch.
    current_batch: list[Request] | None
    remaining_batch_time_ms: float | None
    current_batch_decode_time_ms: float | None

    batch_step: int = 16
    _kv_cache_bytes: int = 0

    # Maintained total of queued decode tokens across ``queue`` and
    # ``download_queue`` (full isl + osl per request).  Updated incrementally on
    # add/remove so the router does not have to scan the queues for every
    # routing decision.  ``None`` means uninitialized (e.g. test doubles built
    # via __new__), in which case the router falls back to scanning.
    active_decode_tokens: float | None = None

    requests_served: int = 0

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

        self.active_decode_tokens = 0.0

        global decode_id_counter
        self.instance_id = decode_id_counter
        decode_id_counter += 1

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

    def _ensure_batch(self, refresh: bool, time_ms: float = 0) -> None:
        """Freeze a new batch from the head of the queue when none is active."""
        if refresh:
            self.refresh_batch = False
        if not self.current_batch or refresh:
            self.current_batch = [req for req, _ in self.queue[: self.max_batch_size]]
            self.current_batch_decode_time_ms = 0.0
            self.remaining_batch_time_ms = 0.0
            self._calculate_batch_time()
            now = self._global_time_ms()
            for req in self.current_batch:
                if req.decode_start_ms is None:
                    req.decode_start_ms = now - time_ms
                if req.decode_queue_start_ms is None:
                    req.decode_queue_start_ms = now - time_ms

    def _calculate_batch_time(self):
        if not self.current_batch:
            self.current_batch_decode_time_ms = 0.0
            return
        assert (
            self.remaining_batch_time_ms == 0.0 or self.remaining_batch_time_ms is None
        ), "calculate batch time called with frozen batch"
        self.current_batch_decode_time_ms = 0.0
        min_remaining = min(r.osl - r.decoded_tokens for r in self.current_batch)
        tokens_done = 0
        while True:
            tokens = min(16, min_remaining - tokens_done)
            if tokens <= 0:
                break
            decode_time = self.calculate_decode_time(
                batch=[(r, 0) for r in self.current_batch], token_offset=tokens_done
            )
            self.current_batch_decode_time_ms += decode_time * tokens
            tokens_done += tokens

    def _tokens_in_time(self, time_ms: float):
        if not self.current_batch:
            return
        tokens_done = 0
        min_remaining = min(r.osl - r.decoded_tokens for r in self.current_batch)
        while time_ms > 0:
            if tokens_done >= min_remaining:
                break
            decode_time = self.calculate_decode_time(
                batch=[(r, 0) for r in self.current_batch],
                token_offset=tokens_done,
            )
            tokens = min(16, min_remaining - tokens_done)
            if tokens <= 0:
                break
            token_chunk_time = decode_time * tokens
            assert self.current_batch_decode_time_ms
            if time_ms >= token_chunk_time:
                tokens_done += tokens
                time_ms -= token_chunk_time
                self.current_batch_decode_time_ms -= token_chunk_time
            else:
                self.remaining_batch_time_ms = token_chunk_time - time_ms
                self.batch_step = tokens
                self.current_batch_decode_time_ms -= token_chunk_time
                break

        for request in self.current_batch:
            request.decoded_tokens += min(tokens_done, min_remaining)
        return

    def _advance_batch(self, tokens: int | None, time_ms: float | None):
        assert tokens or time_ms is not None, (
            f"called advance batch with {tokens, time_ms}"
        )
        if not self.current_batch:
            return

        if self.remaining_batch_time_ms:
            assert time_ms is not None, (
                f"called advance batch with {tokens} tokens even though remaining batch time is non zero ({self.remaining_batch_time_ms})"
            )
            if time_ms >= self.remaining_batch_time_ms:
                time_ms -= self.remaining_batch_time_ms
                self.remaining_batch_time_ms = 0.0
                self._advance_batch(tokens=self.batch_step, time_ms=None)
                if time_ms > 0:
                    self._advance_batch(tokens=None, time_ms=time_ms)
            else:
                self.remaining_batch_time_ms -= time_ms
            return
        if tokens:
            for request in self.current_batch:
                request.decoded_tokens += tokens
            return

        if time_ms and time_ms > 0:
            if self.refresh_batch:
                self._ensure_batch(True)
            self._tokens_in_time(time_ms)

    def add_request(self, request: Request):
        self.requests_served += 1
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
                f"[t={now:.3f} ms] [Decode {self.instance_id}] Adding request {request.id} to decode instance {self.node_id}, "
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
            self._kv_cache_bytes += self.model.kv_size_tokens(request.isl + request.osl)
            self.refresh_batch = True
        if self.active_decode_tokens is not None:
            self.active_decode_tokens += float(request.isl) + float(request.osl)

    def time_to_next_completion(self) -> float:
        """Return a lower-bound time until one request in the batch finishes.

        This is intentionally a lowball estimate: remaining time for the current
        in-flight token plus the current per-token decode latency multiplied by
        the smallest number of remaining output tokens (minus the current one)
        for any request in the batch.

        Transfer completion times are handled globally by the
        ``BandwidthScheduler``; this method only reports compute events.

        If the head download in the queue has no active legs (e.g. a zero-byte
        download for a prefix that is already local), return 0 so the event
        loop drains it immediately.
        """
        if self.download_queue and self.download_queue[0][0].is_download_done():
            return 0.0
        if (
            self.remaining_batch_time_ms == 0.0 or self.remaining_batch_time_ms is None
        ) and (self.current_batch_decode_time_ms == 0.0 or self.refresh_batch):
            self._ensure_batch(True)
        if (
            self.current_batch_decode_time_ms is None
            or self.remaining_batch_time_ms is None
        ):
            return float("inf")
        if self.current_batch_decode_time_ms == 0.0:
            return self.remaining_batch_time_ms or float("inf")
        if self.remaining_batch_time_ms == 0.0:
            return self.current_batch_decode_time_ms or float("inf")
        return self.current_batch_decode_time_ms + self.remaining_batch_time_ms

    def process_queue(self, time_ms: float) -> list[Request]:
        assert self.cache is not None, "Cache must be set before processing queue"
        assert self.scheduler is not None, (
            "Scheduler must be set before processing queue"
        )

        # Fast path: nothing to do for an idle instance.  Skip the per-call
        # bookkeeping (clock read, log guards, empty drain loops) so the event
        # loop pays almost nothing for instances with no pending work.  When
        # every queue is empty ``_kv_cache_bytes`` is 0, so the GPU-memory check
        # below could never fire.
        if (
            not (
                self.queue
                or self.download_queue
                or self.background_download_queue
                or self.upload_queue
                or self.background_upload_queue
            )
            and not self.current_batch
        ):
            return []

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

        # Drain completed decode uploads.  The actual upload is the last track;
        # once it finishes the request is done, while any eviction tracks keep
        # running in the background.
        while self.upload_queue and self.upload_queue[0][0].is_upload_done():
            upload_request, _ = self.upload_queue.pop(0)
            request = upload_request.request
            request.decode_upload_end_ms = now
            request.decode_upload_active_ms = upload_request.upload_active_duration_ms()
            finished_requests.append(request)
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

        self._ensure_batch(False, time_ms=time_ms)

        self._advance_batch(tokens=None, time_ms=time_ms)

        # Drain completed decode downloads.  Only the data tracks (not the
        # background eviction tracks) gate decode start; evictions are kept in
        # the scheduler and finish asynchronously.
        while self.download_queue and self.download_queue[0][0].is_download_done():
            download_request, _ = self.download_queue.pop(0)
            request = download_request.request
            request.decode_download_end_ms = now
            request.decode_download_active_ms = (
                download_request.download_active_duration_ms()
            )
            request.decode_queue_start_ms = now
            self.queue.append((request, 0))
            self.current_batch = None
            self._kv_cache_bytes += self.model.kv_size_tokens(request.cache_length)
            # Keep the DownloadRequest around so the full background eviction
            # duration can be captured once every track is exhausted.
            self.background_download_queue.append(download_request)
            self.refresh_batch = True
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

        finished_in_batch: list[Request] = []
        # Mark requests that finished decoding in this step.
        for request in self.current_batch or []:
            assert request.decoded_tokens <= request.osl, "Too many decoded tokens"
            if request.decoded_tokens >= request.osl:
                assert request.decoded_tokens == request.osl
                if should_log(LOG_INSTANCE):
                    log(
                        LOG_INSTANCE,
                        f"[t={now:.3f} ms] [Decode {self.instance_id}] Finishing request decode with id: "
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

        if should_log(LOG_INSTANCE):
            log(
                LOG_INSTANCE,
                f"[t={now:.3f} ms] [Decode {self.instance_id}] Removing finished requests {finished_in_batch}",
            )
        # Remove finished requests from the queue and the frozen batch.
        if finished_in_batch:
            finished_set = {id(r) for r in finished_in_batch}
            if self.active_decode_tokens is not None:
                for r in finished_in_batch:
                    self.active_decode_tokens -= float(r.isl) + float(r.osl)
            removed_cache_bytes = 0
            new_queue: list[tuple[Request, float]] = []
            for r, t in self.queue:
                if id(r) in finished_set:
                    if should_log(LOG_INSTANCE):
                        log(
                            LOG_INSTANCE,
                            f"[t={now:.3f} ms] [Decode {self.instance_id}] Removing finished request {(r.user_id, r.session_id)}",
                        )
                    removed_cache_bytes += self.model.kv_size_tokens(r.cache_length)
                else:
                    new_queue.append((r, t))
            self.queue = new_queue
            self._ensure_batch(True)
            self._kv_cache_bytes -= removed_cache_bytes
        return finished_requests

    def calculate_decode_time(
        self, batch: list[tuple[Request, float]], token_offset: int
    ) -> int:
        flops = calculate_flops(self.model, batch, "decode", token_offset)
        memory = calculate_memory(self.model, batch, "decode", token_offset)

        time_ms: int = int(
            max(
                float(flops) / self.hardware.flops,
                float(memory) / self.hardware.gpu_bw,
            )
            * 1000
        )
        # if should_log(LOG_INSTANCE):
        #     log(
        #         LOG_INSTANCE,
        #         f"[Decode {self.instance_id}] Calculated decode time for batch"
        #         f"{[req.prefilled_tokens + req.decoded_tokens + token_offset for req, _ in batch]} "
        #         f"of size {len(batch)}: {time_ms} ms: BATCH: {[r.id for r, _ in batch]}",
        #     )
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
