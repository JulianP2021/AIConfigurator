from src.utils.utils import calculate_flops, calculate_memory
from src.hardware.hardware import Hardware
from src.model.model import Model
from src.request.request import Request


class DecodeInstance:
    hardware: Hardware
    queue: list[Request]
    max_batch_size: int
    model: Model

    def __init__(self, hardware: Hardware, max_batch_size: int, model: Model):
        self.hardware = hardware
        self.queue = []
        self.max_batch_size = max_batch_size
        self.model = model

    def add_request(self, request: Request):
        self.queue.append(request)

    def process_queue(self, time_ms: int) -> list[Request]:
        processed_requests: list[Request] = []
        total_time_ms = time_ms
        batch = self.queue[: self.max_batch_size]
        while batch and time_ms > 0:
            decode_time = self.calculate_decode_time(batch)
            time_ms -= decode_time
            if time_ms >= decode_time:
                for request in batch:
                    request.decoded_tokens += 1
                else: 
                    return processed_requests
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
            (float(flops) / self.hardware.flops
            + float(memory) / self.hardware.memoryGB_BW) * 1000
        ) 
        print(f"Calculated decode time for batch{[req.prefilled_tokens + req.decoded_tokens for req in batch]} of size {len(batch)}: {time_ms} ms")
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
                    print(f"Finishing request with id: {request.id} after {total_ms / 1000} seconds + finish queue")
                    self.queue.remove(request)
                    finished_requests.append(request)
        return total_ms, finished_requests