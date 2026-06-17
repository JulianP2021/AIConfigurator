from hardware.hardware import Hardware
from model.model import Model
from request.request import Request


class PrefillInstance:
    hardware: Hardware
    queue: list[Request]
    model: Model

    def __init__(self, hardware: Hardware, queue: list[Request], model: Model):
        self.hardware = hardware
        self.queue = queue
        self.model = model

    def add_request(self, request: Request):
        self.queue.append(request)

    def process_queue(self, time_ms: int):
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
                self.queue.pop(0)
            else:
                request.remaining_prefill_time_ms = -time_ms

    def calculate_prefill_time(self, request: Request) -> int:
        def calculate_prefill_flops(model: Model, request: Request) -> int:
            qk_proj = (
                request.remaining_tokens_prefill * 4 * model.config["hidden_size"] ** 2
            )
            kv_proj = (
                request.remaining_tokens_prefill
                * 4
                * model.config["hidden_size"]
                * model.config["num_key_value_heads"]
                * model.config.get(
                    "head_dim",
                    model.config["hidden_size"] // model.config["num_attention_heads"],
                )
            )
            attn = (
                request.remaining_tokens_prefill
                * 4
                * (request.remaining_tokens_prefill + request.prefilled_tokens)
                * model.config["hidden_size"]
            )
            ffn = (
                request.remaining_tokens_prefill
                * 6
                * model.config["intermediate_size"]
                * model.config["hidden_size"]
            )
            per_layer_flops = qk_proj + kv_proj + attn + ffn
            total_flops = (
                per_layer_flops * model.config["num_hidden_layers"]
                + 2 * model.config["hidden_size"] * model.config["vocab_size"]
            )
            return total_flops

        def calculate_prefill_memory(model: Model, request: Request) -> int:
            qk_proj = request.remaining_tokens_prefill * 4 * model.config["hidden_size"]
            kv_proj = (
                request.remaining_tokens_prefill
                * 4
                * model.config["num_key_value_heads"]
                * model.config.get(
                    "head_dim",
                    model.config["hidden_size"] // model.config["num_attention_heads"],
                )
            )
            kv_entries = (
                2
                * (request.remaining_tokens_prefill + request.prefilled_tokens)
                * model.config["hidden_size"]
            )
            matrices = (
                4 * model.config["hidden_size"] ** 2
                + 3 * model.config["intermediate_size"] * model.config["hidden_size"]
            )
            per_layer_memory = qk_proj + kv_proj + kv_entries + matrices
            total_memory = (
                per_layer_memory * model.config["num_hidden_layers"]
                + 2 * model.config["hidden_size"] * model.config["vocab_size"]
            )
            return total_memory

        flops = calculate_prefill_flops(self.model, request)
        memory = calculate_prefill_memory(self.model, request)

        time_ms: int = int(
            float(flops) / self.hardware.flops * 1000
            + float(memory) / self.hardware.memoryBW * 1000
        )
        return time_ms
