from typing import Any

from src.aiconfigurator_lib.estimator import (
    build_session,
    get_meta,
    run_static_inference,
)
from src.cache.cache import Cache
from src.hardware.hardware import GPUHardwareSpec
from src.logger import debug_print
from src.model.model import Model
from src.request.request import DownloadRequest, Request, UploadRequest


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

    def __init__(
        self, node_id: int, hardware: GPUHardwareSpec, max_batch_size: int, model: Model
    ):
        self.node_id = node_id
        self.hardware = hardware
        self.queue = []
        self.download_queue = []
        self.upload_queue = []
        self.max_batch_size = max_batch_size
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

    def add_request(self, request: Request):
        assert self.cache is not None, "Cache must be set before adding requests"
        dt = self.cache.download_kv(self.node_id, request)
        debug_print(
            f"Adding request {request.id} to decode instance {self.node_id}, with download_time {dt}"
        )
        self.download_queue.append((DownloadRequest(request, dt), -1))

    def _calculate_process_time(
        self, time_ms: float, is_downloading: bool, is_uploading: bool
    ) -> tuple[list[tuple[Request, float]], float, float, float, float]:
        """Calculate the time to process the next event in the queue. Returns a tuple of (batch, next_download_time, next_decode_time, next_upload_time, next_event_time)."""
        if self.download_queue:
            download_time = self.download_queue[0][0].remaining_download_time_ms
        else:
            download_time = float("inf")

        batch = self.queue[: self.max_batch_size]
        if len(batch) == 0:
            decode_time = float("inf")
        else:
            decode_time = self.calculate_decode_time(batch)

        if self.upload_queue:
            upload_time = self.upload_queue[0][0].remaining_upload_time_ms
        else:
            upload_time = float("inf")

        if is_downloading:
            next_event_time = min(download_time, decode_time)
        elif is_uploading:
            next_event_time = min(upload_time, decode_time)
        else:
            next_event_time = decode_time

        next_event_time = (
            max(next_event_time, decode_time)
            if decode_time != float("inf")
            else next_event_time
        )

        debug_print(
            f"Time to next completion for decode instance {
                self.node_id
            }: download_time={download_time}, decode_time={decode_time}, upload_time={
                upload_time
            }, min_time={next_event_time}, 'download_queue': {
                len(self.download_queue)
            }, 'decode_queue': {len(self.queue)}, 'upload_queue': {
                len(self.upload_queue)
            }"
        )

        return (
            batch,
            download_time,
            decode_time,
            upload_time,
            min(next_event_time, time_ms),
        )

    def time_to_next_completion(self) -> float:
        return self._calculate_process_time(
            float("inf"), len(self.download_queue) > 0, len(self.upload_queue) > 0
        )[4]

    def process_queue(self, time_ms: float) -> list[Request]:
        assert self.cache is not None, "Cache must be set before processing queue"

        debug_print(
            f"Processing decode queue for node {self.node_id} with time_ms: {time_ms}"
        )

        assert time_ms >= 0, "Time to process queue should be non-negative"
        kv_cache = 0
        for req, _ in self.queue:
            kv_cache += self.model.kv_size_per_token * req.isl
        assert kv_cache <= self.hardware.gpu_mem, (
            "KV cache exceeds GPU memory, too many requests in decode queue"
        )

        total_time_ms = time_ms

        self.upload_queue = [
            (upload_request, total_time_ms) for upload_request, _ in self.upload_queue
        ]

        self.queue = [
            (prefill_request, total_time_ms) for prefill_request, _ in self.queue
        ]

        self.download_queue = [
            (download_request, total_time_ms)
            for download_request, _ in self.download_queue
        ]

        finished_requests: list[Request] = []

        while time_ms > 0 and (self.download_queue or self.queue or self.upload_queue):
            is_downloading = len(self.download_queue) > 0
            is_uploading = len(self.upload_queue) > 0 and not is_downloading

            for _, start in self.download_queue:
                if start == -1:
                    start = time_ms

            for _, start in self.upload_queue:
                if start == -1:
                    start = time_ms

            batch, _, _, _, next_event_time = self._calculate_process_time(
                time_ms, is_downloading, is_uploading
            )

            if next_event_time == float("inf"):
                break

            process_time = min(next_event_time, time_ms)

            time_ms -= process_time

            if is_downloading and self.download_queue:
                download_request, download_start = self.download_queue[0]
                if download_start != -1:
                    download_request.request.kv_download_time_ms += process_time
                    download_request.remaining_download_time_ms -= process_time
                    if download_request.remaining_download_time_ms <= 0:
                        self.queue.append((download_request.request, time_ms))
                        self.download_queue.pop(0)

            if len(batch) > 0:
                for request, _ in batch:
                    request.decode_time_ms += process_time
                    request.decoded_tokens += 1
                    if request.decoded_tokens >= request.osl:
                        request.decode_time_ms += total_time_ms - time_ms
                        debug_print(
                            f"Finishing request decode with id: {request.id} after {request.decode_time_ms / 1000} seconds + process queue"
                        )
                        self.upload_queue.append((
                            UploadRequest(
                                request, self.cache.upload_kv(self.node_id, request)
                            ),
                            time_ms,
                        ))
                        self.queue.pop(0)

            if is_uploading and self.upload_queue:
                upload_request, upload_start = self.upload_queue[0]
                upload_request.request.kv_upload_time_ms += process_time
                upload_request.remaining_upload_time_ms -= process_time
                if upload_request.remaining_upload_time_ms <= 0:
                    finished_requests.append(upload_request.request)
                    self.upload_queue.pop(0)

        for download_request, download_start in self.download_queue:
            download_request.request.kv_download_time_ms += download_start

        for upload_request, upload_start in self.upload_queue:
            upload_request.request.kv_upload_time_ms += upload_start

        for req, decode_start in self.queue:
            req.decode_time_ms += decode_start

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

        debug_print(
            f"Calculated decode time for batch{[req.prefilled_tokens + req.decoded_tokens for req, _ in batch]} of size {len(batch)}: {time_ms} ms"
        )
        return time_ms

    def log(self):
        debug_print(
            f"Decode instance state: {len(self.queue)} requests in queue, {len(self.download_queue)} requests in download queue"
        )
        for request, _ in self.queue:
            debug_print(
                f"Request id: {request.id}, decoded tokens: {request.decoded_tokens}, remaining tokens decode: {request.osl - request.decoded_tokens}, decode time ms: {request.decode_time_ms}"
            )
