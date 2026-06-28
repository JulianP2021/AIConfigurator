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


class DecodeInstance:
    hardware: GPUHardwareSpec
    queue: list[Request]
    download_queue: list[Request]
    max_batch_size: int
    model: Model
    session: Any

    def __init__(self, hardware: GPUHardwareSpec, max_batch_size: int, model: Model):
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

    def add_request(self, request: Request):
        self.queue.append(request)

    def download_kv_time_ms(self, _request: Request) -> int:
        # kv_size = self.model.kv_size_per_token * request.isl
        # time_ms: int = int((float(kv_size) / self.hardware.spec.network_bw) * 1000)

        return 100  # example value, replace with actual calculation

    def process_queue(self, time_ms: float) -> list[Request]:
        processed_requests: list[Request] = []
        total_time_ms = time_ms

        kv_cache = 0
        for req in self.queue:
            kv_cache += self.model.kv_size_per_token * req.isl
        assert kv_cache <= self.hardware.gpu_mem, (
            "KV cache exceeds GPU memory, too many requests in decode queue"
        )

        while time_ms > 0.000001 and self.queue:
            batch = self.queue[: self.max_batch_size]
            decode_time = self.calculate_decode_time(batch)
            if decode_time > time_ms:
                # Not enough time to complete a single decode step
                break
            time_ms -= decode_time
            for request in batch:
                request.decoded_tokens += 1
            for request in batch:
                if request.decoded_tokens >= request.osl:
                    request.decode_time_ms += total_time_ms - time_ms
                    self.queue.remove(request)
                    processed_requests.append(request)
        for req in self.queue:
            req.decode_time_ms += total_time_ms - time_ms
        return processed_requests

    def calculate_decode_time(self, batch: list[Request]) -> float:
        avg_isl = int(
            sum(req.prefilled_tokens + req.decoded_tokens for req in batch) / len(batch)
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

        print(
            f"Calculated decode time for batch{[req.prefilled_tokens + req.decoded_tokens for req in batch]} of size {len(batch)}: {time_ms} ms"
        )
        return time_ms

    def finish_queue(self) -> tuple[float, list[Request]]:
        finished_requests: list[Request] = []
        total_ms = 0
        while self.queue:
            batch = self.queue[: self.max_batch_size]
            decode_time = self.calculate_decode_time(batch)
            total_ms += decode_time
            for request in batch:
                request.decoded_tokens += 1
                if request.decoded_tokens >= request.osl:
                    request.decode_time_ms += total_ms
                    debug_print(
                        f"Finishing request decode with id: {request.id} after {total_ms / 1000} seconds + finish queue"
                    )
                    self.queue.remove(request)
                    finished_requests.append(request)
        return total_ms, finished_requests

    def log(self):
        debug_print(
            f"Decode instance state: {len(self.queue)} requests in queue, {len(self.download_queue)} requests in download queue"
        )
        for request in self.queue:
            debug_print(
                f"Request id: {request.id}, decoded tokens: {request.decoded_tokens}, remaining tokens decode: {request.osl - request.decoded_tokens}, decode time ms: {request.decode_time_ms}"
            )
