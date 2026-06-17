from src.utils.utils import calculate_flops, calculate_memory
from src.hardware.hardware import Hardware
from src.model.model import Model
from src.request.request import Request


class PrefillInstance:
    hardware: Hardware
    queue: list[Request]
    model: Model

    def __init__(self, hardware: Hardware, model: Model):
        self.hardware = hardware
        self.queue = []
        self.model = model

    def add_request(self, request: Request):
        self.queue.append(request)

    def process_queue(self, time_ms: int) -> list[Request]:
        total_time_ms = time_ms
        processed_requests: list[Request] = []
        while self.queue and time_ms > 0:
            request = self.queue[0]
            prefill_time = self.calculate_prefill_time(request)
            if request.remaining_prefill_time_ms != -1:
                prefill_time = request.remaining_prefill_time_ms
            else:
                request.prefilled_tokens += request.remaining_tokens_prefill
            time_ms -= prefill_time
            if time_ms > 0:
                request.remaining_prefill_time_ms = 0
                request.prefill_time_ms += total_time_ms - time_ms
                processed_requests.append(self.queue.pop(0))
            else:
                request.remaining_prefill_time_ms = -time_ms

        for req in self.queue:
            req.prefill_time_ms += time_ms
        return processed_requests

    def calculate_prefill_time(self, request: Request) -> int:
        flops = calculate_flops(self.model, [request], "prefill")
        memory = calculate_memory(self.model, [request], "prefill")

        time_ms: int = int(
            float(flops) / self.hardware.flops * 1000
            + float(memory) / self.hardware.memoryGB_BW * 1000
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

        return total_ms, self.queue
