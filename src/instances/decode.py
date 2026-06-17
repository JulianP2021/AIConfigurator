from hardware.hardware import Hardware
from model.model import Model
from request.request import Request


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

    def process_queue(self, time_ms: int):
        batch = self.queue[: self.max_batch_size]
        while batch and time_ms > 0:
            decode_time = self.calculate_decode_time(batch)
            time_ms -= decode_time
            if time_ms >= decode_time:
                for request in batch:
                    request.decoded_tokens += 1
                else: 
                    return
            for request in batch:
                if request.decoded_tokens >= request.osl:
                    self.queue.remove(request)


    def calculate_decode_time(self, batch: list[Request]) -> int:
        def calculate_prefill_flops(model: Model, batch: list[Request]) -> int:
            total_flops = 0
            for request in batch:
                qk_proj = 1 * 4 * model.config["hidden_size"] ** 2
                kv_proj = (
                    1
                    * 4
                    * model.config["hidden_size"]
                    * model.config["num_key_value_heads"]
                    * model.config.get(
                        "head_dim",
                        model.config["hidden_size"]
                        // model.config["num_attention_heads"],
                    )
                )
                attn = (
                    1
                    * 4
                    * (1 + request.prefilled_tokens + request.decoded_tokens)
                    * model.config["hidden_size"]
                )
                ffn = (
                    1
                    * 6
                    * model.config["intermediate_size"]
                    * model.config["hidden_size"]
                )
                per_layer_flops = qk_proj + kv_proj + attn + ffn
                total_flops += (
                    per_layer_flops * model.config["num_hidden_layers"]
                    + 2 * model.config["hidden_size"] * model.config["vocab_size"]
                )
            return total_flops

        def calculate_prefill_memory(model: Model, batch: list[Request]) -> int:
            total_memory = 0
            matrices = (
                4 * model.config["hidden_size"] ** 2
                + 3 * model.config["intermediate_size"] * model.config["hidden_size"]
            )
            total_memory += matrices * model.config["num_hidden_layers"]

            for request in batch:
                qk_proj = 1 * 4 * model.config["hidden_size"]
                kv_proj = (
                    1
                    * 4
                    * model.config["num_key_value_heads"]
                    * model.config.get(
                        "head_dim",
                        model.config["hidden_size"]
                        // model.config["num_attention_heads"],
                    )
                )
                kv_entries = (
                    2
                    * (request.prefilled_tokens + request.decoded_tokens)
                    * model.config["hidden_size"]
                )
                per_layer_memory = qk_proj + kv_proj + kv_entries
                total_memory = (
                    per_layer_memory * model.config["num_hidden_layers"]
                    + 2 * model.config["hidden_size"] * model.config["vocab_size"]
                )
            return total_memory

        flops = calculate_prefill_flops(self.model, batch)
        memory = calculate_prefill_memory(self.model, batch)

        time_ms: int = int(
            float(flops) / self.hardware.flops * 1000
            + float(memory) / self.hardware.memoryBW * 1000
        )
        return time_ms
