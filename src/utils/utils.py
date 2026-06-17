from src.model.model import Model
from src.request.request import Request

def calculate_flops(model: Model, batch: list[Request], mode: str) -> int:
    total_flops = 0
    for request in batch:
        if mode == "prefill":
            tokens_to_process = request.remaining_tokens_prefill
        else:
            tokens_to_process = 1
        qk_proj = (
            tokens_to_process * 4 * model.config["hidden_size"] ** 2
        )
        kv_proj = (
            tokens_to_process
            * 4
            * model.config["hidden_size"]
            * model.config["num_key_value_heads"]
            * model.config.get(
                "head_dim",
                model.config["hidden_size"] // model.config["num_attention_heads"],
            )
        )
        attn = (
            tokens_to_process
            * 4
            * (tokens_to_process + request.prefilled_tokens + request.decoded_tokens)
            * model.config["hidden_size"]
        )
        ffn = (
            tokens_to_process
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

def calculate_memory(model: Model, batch: list[Request], mode: str) -> int:
    d_kv = model.config.get(
                        "head_dim",
                        model.config["hidden_size"]
                        // model.config["num_attention_heads"],
                    )

    total_memory = 0
    matrices = (
        2 * model.dtype_size * (model.config["hidden_size"] ** 2)
        + 3 * model.dtype_size * model.config["intermediate_size"] * model.config["hidden_size"]
    )
        
    total_memory += matrices * model.config["num_hidden_layers"] + 2 * model.dtype_size * model.config["hidden_size"] * model.config["vocab_size"]


    for request in batch:
        if mode == "prefill":
            tokens_to_process = request.remaining_tokens_prefill
        else:
            tokens_to_process = 1
        qk_proj = tokens_to_process * 2 * model.dtype_size * model.config["hidden_size"]
        kv_proj = (
            tokens_to_process
            * 2 * model.dtype_size
            * model.config["num_key_value_heads"]
            * d_kv
        )
        kv_entries = (
            2 * model.dtype_size
            * (request.prefilled_tokens + request.decoded_tokens)
            * model.config["hidden_size"]
        )
        layer_norm = 2 * model.dtype_size * model.config["hidden_size"]
        per_layer_memory = qk_proj + kv_proj + kv_entries + layer_norm
        total_memory += (
            per_layer_memory * model.config["num_hidden_layers"]
        )
    return total_memory