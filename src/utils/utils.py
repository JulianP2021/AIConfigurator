from functools import lru_cache

from src.model.model import Model
from src.request.request import Request


@lru_cache(maxsize=65536)
def _calculate_flops(model: Model, tokens_to_process: int, cache_len: int) -> int:
    c = model.cost_constants
    embedding = tokens_to_process
    output_proj = 2 * c["hidden_size"] * c["vocab_size"]
    flops = embedding + output_proj

    def full_attn_flops():
        qo_proj = (
            tokens_to_process
            * 4
            * c["hidden_size"]
            * c["num_attention_heads"]
            * c["head_dim"]
        )
        kv_proj = (
            tokens_to_process
            * 4
            * c["hidden_size"]
            * c["num_key_value_heads"]
            * c["head_dim"]
        )
        attn = (
            2
            * ((tokens_to_process + cache_len) ** 2 - cache_len**2)
            * c["num_attention_heads"]
            * c["head_dim"]
        )
        ffn = tokens_to_process * 6 * c["intermediate_size"] * c["hidden_size"]
        per_layer_flops = qo_proj + kv_proj + attn + ffn
        return per_layer_flops * c["full_attn_layers"]

    def linear_attn_flops():
        q_proj = 2 * tokens_to_process * c["ld_q"] * c["hidden_size"]
        k_proj = 2 * tokens_to_process * c["ld_k"] * c["hidden_size"]
        voz_proj = 6 * tokens_to_process * c["ld_v"] * c["hidden_size"]
        ab_proj = 4 * tokens_to_process * (c["ld_k"] / c["head_dim"])
        attn = 2 * tokens_to_process * c["ld_q"] * c["ld_v"]
        ffn = tokens_to_process * 6 * c["intermediate_size"] * c["hidden_size"]

        per_layer_flops = q_proj + k_proj + voz_proj + +ab_proj + attn + ffn
        return per_layer_flops * c["linear_attn_layers"]

    flops += full_attn_flops() + linear_attn_flops()
    return int(flops)


def calculate_flops(
    model: Model, batch: list[tuple[Request, float]], mode: str, token_offset: int = 0
) -> int:
    total_flops = 0
    for request, _ in batch:
        tokens_to_process = request.remaining_tokens_prefill if mode == "prefill" else 1
        total_flops += _calculate_flops(
            model, tokens_to_process, request.cache_length + token_offset
        )
    return int(total_flops)


@lru_cache(maxsize=1280)
def _mem_model(model: Model) -> int:
    c = model.config
    cc = model.cost_constants
    ffn: float = 3 * int(c.get("intermediate_size", 0)) * cc["hidden_size"]
    memory = 0

    def full_attn_memory():
        full_attn_projection_matrices: float = cc["hidden_size"] * (
            2 * cc["num_attention_heads"] * cc["head_dim"]
            + 2 * c["num_key_value_heads"] * c["head_dim"]
        )
        return (full_attn_projection_matrices + ffn) * cc["full_attn_layers"]

    def linear_attn_memory():
        linear_attn_projection_matrices: float = cc["hidden_size"] * (
            cc["ld_q"] + cc["ld_k"] + cc["ld_v"] * 3
        )
        return (linear_attn_projection_matrices + ffn) * cc["linear_attn_layers"]

    memory += full_attn_memory() + linear_attn_memory()
    memory += 2 * cc["hidden_size"] * cc["vocab_size"]  # embedding + output projection

    return int(memory * model.dtype_size("dtype"))


@lru_cache(maxsize=65536)
def _calculate_memory(model: Model, tokens_to_process: int, cache_len: int) -> int:
    c = model.cost_constants
    memory = 0

    def full_attn_memory():
        qo_proj = tokens_to_process * 2 * model.dtype_size("dtype") * c["hidden_size"]
        kv_proj = (
            tokens_to_process
            * 2
            * model.dtype_size("dtype")
            * model.config["num_key_value_heads"]
            * c["head_dim"]
        )
        kv_entries = (
            2
            * model.dtype_size("dtype")
            * (cache_len)
            * model.config["num_key_value_heads"]
            * c["head_dim"]
        )
        return (qo_proj + kv_proj + kv_entries) * c["full_attn_layers"]

    def linear_attn_memory():
        q_proj = tokens_to_process * c["ld_q"]
        k_proj = tokens_to_process * c["ld_k"]
        voz_proj = tokens_to_process * c["ld_v"] * 3
        conv = 4 * (c["ld_q"] + c["ld_k"] + c["ld_v"])
        state = c["ld_k"] * c["ld_v"]
        return (
            (q_proj + k_proj + voz_proj + conv + state)
            * c["linear_attn_layers"]
            * model.dtype_size("mamba_ssm_dtype")
        )

    memory += full_attn_memory() + linear_attn_memory()

    return memory


def calculate_memory(
    model: Model, batch: list[tuple[Request, float]], mode: str, token_offset: int = 0
) -> int:
    total_memory = _mem_model(model)
    for request, _ in batch:
        tokens_to_process = request.remaining_tokens_prefill if mode == "prefill" else 1
        total_memory += _calculate_memory(
            model, tokens_to_process, request.cache_length + token_offset
        )
    return total_memory


def parse_float_list(text: str) -> list[float]:
    """Parse a comma-separated string into a list of floats."""
    return [float(x.strip()) for x in text.split(",")]


def parse_int_list(text: str) -> list[int]:
    """Parse a comma-separated string into a list of integers."""
    return [int(float(x.strip())) for x in text.split(",")]


def add_result_metadata(
    row: dict[str, object],
    label: str,
    cfg: dict[str, object],
    color: str,
    users: int | None = None,
    extra_fields: dict[str, object] | None = None,
) -> None:
    """Layer webserver-specific metadata onto a SimulationResult.to_dict() row.

    This helper is shared by execute_user_sweep_config.py and the webserver so the
    results JSON schema stays consistent. Per-request stats are stripped from
    the exported row to keep JSON output compact.
    """
    row.pop("per_request_stats", None)
    row.update({
        "label": label,
        "prefill_hardware": cfg.get("prefill_hardware", ""),
        "decode_hardware": cfg.get("decode_hardware", ""),
        "colocated": cfg.get("colocated", "False"),
        "color": color,
        "has_error": False,
    })
    if users is not None:
        row["users"] = users
    if extra_fields:
        row.update(extra_fields)


def get_shape(lst, shape=()):
    """Returns the shape of nested lists similarly to numpy's shape.

    :param lst: the nested list
    :param shape: the shape up to the current recursion depth
    :return: the shape including the current depth
            (finally this will be the full depth)
    """
    if not isinstance(lst, list):
        # base case
        return shape

    shape += (len(lst),)

    # recurse
    if len(lst) == 0:
        return shape
    max_v = max(len(item) for item in lst)
    if max_v == 0:
        return shape
    return get_shape(next(filter(lambda x: len(x) == max_v, lst)), shape)
