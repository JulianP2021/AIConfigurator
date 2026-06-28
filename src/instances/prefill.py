from typing import Any

from src.aiconfigurator_lib.estimator import (
    build_session,
    get_meta,
    run_static_inference,
)
from src.hardware.hardware import GPUHardwareSpec
from src.logger import debug_print
from src.model.model import Model
from src.request.request import Request


class UploadRequest:
    request: Request
    remaining_upload_time_ms: float

    def __init__(self, request: Request, remaining_upload_time_ms: float):
        self.request = request
        self.remaining_upload_time_ms = remaining_upload_time_ms


class PrefillInstance:
    hardware: GPUHardwareSpec
    queue: list[Request]
    upload_queue: list[tuple[UploadRequest, float]]
    download_queue: list[tuple[UploadRequest, float]]

    model: Model
    session: Any

    def __init__(self, hardware: GPUHardwareSpec, model: Model):
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

    def add_request(self, request: Request):
        self.queue.append(request)

    def upload_kv_time_ms(self, _request: Request) -> float:
        # kv_size = self.model.kv_size_per_token * request.isl
        # time_ms: float = float((float(kv_size) / self.hardware.spec.network_bw) * 1000)
        # debug_print((
        #     f"Calculated KV upload time for request with id: {request.id} : {time_ms} ms"
        # )

        # return time_ms + 1
        return 1.0

    def time_to_next_completion(self) -> float:
        if not self.queue:
            if not self.upload_queue:
                return -1
            return self.upload_queue[0][0].remaining_upload_time_ms
        request = self.queue[0]
        debug_print(
            f"Calculating time to next completion for request with id: {request.id}: remaining_prefill_time_ms: {request.remaining_prefill_time_ms}"
        )
        return max(
            request.remaining_prefill_time_ms
            if request.remaining_prefill_time_ms > 0
            else self.calculate_prefill_time(request),
            1,
        )

    def process_queue(self, time_ms: float) -> list[Request]:
        assert len(self.queue) < 100, "Too many requests in prefill queue"

        def get_request_requiring_prefill(requests: list[Request]) -> Request | None:
            for req in requests:
                if req.remaining_tokens_prefill != 0:
                    return req
            return None

        assert time_ms >= 0, "Time to process queue should be non-negative"
        total_time_ms = time_ms
        self.upload_queue = [
            (upload_request, total_time_ms)
            for upload_request, _ in self.upload_queue
            if upload_request.remaining_upload_time_ms > 0
        ]

        while self.queue and time_ms > 0:
            request = get_request_requiring_prefill(self.queue)
            if not request:
                debug_print(
                    f"No requests require prefill, breaking out of loop {[req.remaining_tokens_prefill for req in self.queue]}",
                )
                break
            prefill_time = self.calculate_prefill_time(request)
            if request.remaining_prefill_time_ms != -1:
                prefill_time = request.remaining_prefill_time_ms
            time_ms -= prefill_time
            if time_ms >= 0:
                request.remaining_prefill_time_ms = 0
                request.prefill_time_ms += total_time_ms - time_ms
                request.prefilled_tokens += request.remaining_tokens_prefill
                request.decoded_tokens = 1
                debug_print(
                    f"Finished prefill for request with id: {request.id} after {request.prefill_time_ms / 1000} seconds + process queue"
                )
                self.queue.remove(request)
                self.upload_queue.append((
                    UploadRequest(request, self.upload_kv_time_ms(request)),
                    time_ms,
                ))
            else:
                # not finished in this time step, update remaining prefill time
                request.remaining_prefill_time_ms = -time_ms

        finished_requests: list[Request] = []
        time_ms = total_time_ms
        while self.upload_queue and time_ms > 0:
            upload_request, upload_start = self.upload_queue[0]

            if time_ms > upload_start:
                break
            upload_time = min(
                upload_start, upload_request.remaining_upload_time_ms, time_ms
            )
            total_time = upload_start - time_ms + upload_time

            upload_request.request.kv_upload_time_ms += total_time
            upload_request.remaining_upload_time_ms -= upload_time
            time_ms -= upload_time
            assert upload_request.remaining_upload_time_ms >= 0, (
                "Remaining upload time should not be negative"
            )
            if upload_request.remaining_upload_time_ms == 0:
                upload_request.request.kv_uploaded = True
                finished_requests.append(upload_request.request)
                self.upload_queue.pop(0)

        for upload_request, upload_start in self.upload_queue:
            upload_request.request.kv_upload_time_ms += upload_start

        for req in self.queue:
            req.prefill_time_ms += total_time_ms
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

    def finish_queue(self) -> tuple[float, list[Request]]:
        total_ms = 0
        for request in self.queue:
            req_time_ms = self.calculate_prefill_time(request)
            request.remaining_prefill_time_ms = 0
            request.prefilled_tokens += request.remaining_tokens_prefill
            request.prefill_time_ms += total_ms + req_time_ms
            total_ms += req_time_ms
            debug_print(
                f"Finishing request prefill with id: {request.id} after {request.prefill_time_ms / 1000} seconds + finish queue"
            )
        return total_ms, self.queue

    def log(self):
        debug_print(
            f"Prefill instance state: {len(self.queue)} requests in queue, {len(self.upload_queue)} requests in upload queue"
        )
        for request in self.queue:
            debug_print(
                f"Request id: {request.id}, prefilled tokens: {request.prefilled_tokens}, remaining tokens prefill: {request.remaining_tokens_prefill}, prefill time ms: {request.prefill_time_ms}, remaining prefill time ms: {request.remaining_prefill_time_ms}"
            )
