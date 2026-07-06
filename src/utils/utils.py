from functools import lru_cache

from src.model.model import Model
from src.request.request import Request


@lru_cache(maxsize=1280)
def _calculate_flops(model: Model, tokens_to_process: int, cache_len: int) -> int:
    c = model.cost_constants
    qk_proj = tokens_to_process * 4 * c["hidden_size"] ** 2
    kv_proj = (
        tokens_to_process * 4 * c["hidden_size"] * c["num_key_value_heads"] * c["d_kv"]
    )
    attn = tokens_to_process * 4 * (tokens_to_process + cache_len) * c["hidden_size"]
    ffn = tokens_to_process * 6 * c["intermediate_size"] * c["hidden_size"]
    per_layer_flops = qk_proj + kv_proj + attn + ffn
    return int(
        per_layer_flops * c["num_hidden_layers"]
        + 2 * c["hidden_size"] * c["vocab_size"]
    )


def calculate_flops(model: Model, batch: list[Request], mode: str) -> int:
    total_flops = 0
    for request in batch:
        tokens_to_process = request.remaining_tokens_prefill if mode == "prefill" else 1
        total_flops += _calculate_flops(model, tokens_to_process, request.cache_length)
    return int(total_flops)


@lru_cache(maxsize=1280)
def _mem_model(model: Model) -> int:
    c = model.cost_constants
    matrices = (
        2 * model.dtype_size * (c["hidden_size"] ** 2)
        + 3 * model.dtype_size * c["intermediate_size"] * c["hidden_size"]
    )

    return int(
        matrices * c["num_hidden_layers"]
        + 2 * model.dtype_size * c["hidden_size"] * c["vocab_size"]
    )


def _calculate_memory(model: Model, tokens_to_process: int, cache_len: int) -> int:
    c = model.cost_constants
    qk_proj = tokens_to_process * 2 * model.dtype_size * c["hidden_size"]
    kv_proj = (
        tokens_to_process
        * 2
        * model.dtype_size
        * model.config["num_key_value_heads"]
        * c["d_kv"]
    )
    kv_entries = 2 * model.dtype_size * (cache_len) * c["hidden_size"]
    layer_norm = 2 * model.dtype_size * c["hidden_size"]
    per_layer_memory = qk_proj + kv_proj + kv_entries + layer_norm
    return per_layer_memory * c["num_hidden_layers"]


def calculate_memory(model: Model, batch: list[Request], mode: str) -> int:
    total_memory = _mem_model(model)
    for request in batch:
        tokens_to_process = request.remaining_tokens_prefill if mode == "prefill" else 1
        total_memory += _calculate_memory(
            model, tokens_to_process, request.cache_length
        )
    return total_memory
