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


class PrefillInstance:
    node_id: int
    hardware: GPUHardwareSpec
    queue: list[tuple[Request, float]]
    upload_queue: list[tuple[UploadRequest, float]]
    download_queue: list[tuple[DownloadRequest, float]]
    cache: Cache | None

    model: Model
    session: Any

    def __init__(self, node_id: int, hardware: GPUHardwareSpec, model: Model):
        self.node_id = node_id
        self.hardware = hardware
        self.queue = []
        self.upload_queue = []
        self.download_queue = []
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
        dr = DownloadRequest(request, self.cache.download_kv(self.node_id, request))
        self.download_queue.append((dr, -1))

    def time_to_next_completion(self) -> float:
        return self._calculate_process_time(
            float("inf"), len(self.download_queue) > 0, len(self.upload_queue) > 0
        )

    def _calculate_process_time(
        self, time_ms: float, is_downloading: bool, is_uploading: bool
    ) -> float:
        if self.download_queue:
            download_time = self.download_queue[0][0].remaining_download_time_ms
        else:
            download_time = float("inf")

        if len(self.queue) == 0:
            prefill_time = float("inf")
        else:
            prefill_time = self.calculate_prefill_time(self.queue[0][0])

        if self.upload_queue:
            upload_time = self.upload_queue[0][0].remaining_upload_time_ms
        else:
            upload_time = float("inf")

        if is_downloading:
            next_event_time = min(download_time, prefill_time)
        elif is_uploading:
            next_event_time = min(upload_time, prefill_time)
        else:
            next_event_time = prefill_time

        next_event_time = (
            max(next_event_time, prefill_time)
            if prefill_time != float("inf")
            else next_event_time
        )

        debug_print(
            f"Time to next completion for prefill instance {
                self.node_id
            }: download_time={download_time}, prefill_time={prefill_time}, upload_time={
                upload_time
            }, min_time={next_event_time}, 'download_queue': {
                len(self.download_queue)
            }, 'prefill_queue': {len(self.queue)}, 'upload_queue': {
                len(self.upload_queue)
            }"
        )

        return min(next_event_time, time_ms)

    def process_queue(self, time_ms: float) -> list[Request]:
        assert len(self.queue) < 100, "Too many requests in prefill queue"
        assert self.cache is not None, "Cache must be set before processing queue"

        debug_print(
            f"Processing prefill queue for node {self.node_id} with time_ms: {time_ms}"
        )

        assert time_ms >= 0, "Time to process queue should be non-negative"
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

            process_time = min(self.time_to_next_completion(), time_ms)
            request = self.queue[0][0] if self.queue else None

            time_ms -= process_time

            if is_downloading and self.download_queue:
                download_request, download_start = self.download_queue[0]
                if download_start != -1:
                    download_request.request.kv_download_time_ms += process_time
                    download_request.remaining_download_time_ms -= process_time
                    if download_request.remaining_download_time_ms <= 0:
                        self.queue.append((download_request.request, time_ms))
                        self.download_queue.pop(0)
                        is_downloading = False

            if request:
                if request.remaining_prefill_time_ms != -1:
                    request.remaining_prefill_time_ms -= process_time
                else:
                    request.remaining_prefill_time_ms = (
                        self.calculate_prefill_time(request) - process_time
                    )

                if request.remaining_prefill_time_ms <= 0:
                    request.remaining_prefill_time_ms = 0
                    request.prefill_time_ms += total_time_ms - time_ms
                    request.prefilled_tokens += request.remaining_tokens_prefill
                    request.decoded_tokens = 1
                    debug_print(
                        f"Finished prefill for request with id: {request.id} after {request.prefill_time_ms / 1000} seconds + process queue"
                    )
                    self.queue.pop(0)
                    self.upload_queue.append((
                        UploadRequest(
                            request, self.cache.upload_kv(self.node_id, request)
                        ),
                        time_ms,
                    ))
                    is_downloading = False
                    is_uploading = True

            if is_uploading and self.upload_queue:
                upload_request, upload_start = self.upload_queue[0]
                upload_request.request.kv_upload_time_ms += process_time
                upload_request.remaining_upload_time_ms -= process_time
                if upload_request.remaining_upload_time_ms <= 0:
                    finished_requests.append(upload_request.request)
                    self.upload_queue.pop(0)
                    is_uploading = False

        for download_request, download_start in self.download_queue:
            download_request.request.kv_download_time_ms += download_start

        for upload_request, upload_start in self.upload_queue:
            upload_request.request.kv_upload_time_ms += upload_start

        for req, prefill_start in self.queue:
            req.prefill_time_ms += prefill_start

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
        debug_print(
            f"Calculated prefill time for request with id: {request.id} : {time_ms} ms"
        )
        return time_ms

    def log(self):
        debug_print(
            f"Prefill instance state: {len(self.queue)} requests in queue, {len(self.upload_queue)} requests in upload queue"
        )
        for request, _ in self.queue:
            debug_print(
                f"Request id: {request.id}, prefilled tokens: {request.prefilled_tokens}, remaining tokens prefill: {request.remaining_tokens_prefill}, prefill time ms: {request.prefill_time_ms}, remaining prefill time ms: {request.remaining_prefill_time_ms}"
            )
