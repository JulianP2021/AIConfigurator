from src.hardware.hardware import Hardware
from src.model.model import Model
from src.request.request import Request
from src.utils.utils import calculate_flops, calculate_memory


class DecodeInstance:
    hardware: Hardware
    queue: list[Request]
    download_queue: list[Request]
    max_batch_size: int
    model: Model

    def __init__(self, hardware: Hardware, max_batch_size: int, model: Model):
        self.hardware = hardware
        self.queue = []
        self.download_queue = []
        self.max_batch_size = max_batch_size
        self.model = model

    def add_request(self, request: Request):
        self.queue.append(request)

    def download_kv_time_ms(self, request: Request) -> int:
        kv_size = self.model.kv_size_per_token * request.isl
        time_ms: int = int((float(kv_size) / self.hardware.spec.network_bw) * 1000)
        return time_ms

    def process_queue(self, time_ms: int) -> list[Request]:
        processed_requests: list[Request] = []
        total_time_ms = time_ms
        batch = self.queue[: self.max_batch_size]
        while batch and time_ms > 0:
            decode_time = self.calculate_decode_time(batch)
            time_ms -= decode_time
            if time_ms < decode_time:
                return processed_requests
            for request in batch:
                request.decoded_tokens += 1
            for request in batch:
                if request.decoded_tokens >= request.osl:
                    request.decode_time_ms += total_time_ms - time_ms
                    self.queue.remove(request)
                    processed_requests.append(request)
        for req in self.queue:
            req.decode_time_ms += total_time_ms
        return processed_requests

    def calculate_decode_time(self, batch: list[Request]) -> int:
        flops = calculate_flops(self.model, batch, "decode")
        memory = calculate_memory(self.model, batch, "decode")

        time_ms: int = int(
            (
                float(flops) / self.hardware.spec.flops
                + float(memory) / self.hardware.spec.ram_bw
            )
            * 1000
        )
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
                    print(
                        f"Finishing request decode with id: {request.id} after {total_ms / 1000} seconds + finish queue"
                    )
                    self.queue.remove(request)
                    finished_requests.append(request)
        return total_ms, finished_requests

    def log(self):
        print(
            f"Decode instance state: {len(self.queue)} requests in queue, {len(self.download_queue)} requests in download queue"
        )
        for request in self.queue:
            print(
                f"Request id: {request.id}, decoded tokens: {request.decoded_tokens}, remaining tokens decode: {request.osl - request.decoded_tokens}, decode time ms: {request.decode_time_ms}"
            )
