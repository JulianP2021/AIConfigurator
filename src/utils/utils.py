from functools import lru_cache

from src.model.model import Model
from src.request.request import Request


@lru_cache(maxsize=1280)
def _calculate_flops(model: Model, tokens_to_process: int, cache_len: int) -> int:
    c = model.cost_constants
    embedding = tokens_to_process
    qo_proj = tokens_to_process * 4 * c["hidden_size"] ** 2
    kv_proj = (
        tokens_to_process * 4 * c["hidden_size"] * c["num_key_value_heads"] * c["d_kv"]
    )
    attn = 2 * ((tokens_to_process + cache_len) ** 2 - cache_len**2) * c["hidden_size"]
    ffn = tokens_to_process * 6 * c["intermediate_size"] * c["hidden_size"]
    per_layer_flops = qo_proj + kv_proj + attn + ffn
    output_proj = 2 * c["hidden_size"] * c["vocab_size"]
    return int(embedding + per_layer_flops * c["num_hidden_layers"] + output_proj)


def calculate_flops(model: Model, batch: list[tuple[Request, float]], mode: str) -> int:
    total_flops = 0
    for request, _ in batch:
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


@lru_cache(maxsize=1280)
def _calculate_memory(model: Model, tokens_to_process: int, cache_len: int) -> int:
    c = model.cost_constants
    embedding = model.dtype_size * c["hidden_size"] * c["vocab_size"]
    qo_proj = tokens_to_process * 2 * model.dtype_size * c["hidden_size"]
    kv_proj = (
        tokens_to_process
        * 2
        * model.dtype_size
        * model.config["num_key_value_heads"]
        * c["d_kv"]
    )
    kv_entries = 2 * model.dtype_size * (cache_len) * c["hidden_size"]
    layer_norm = 2 * model.dtype_size * c["hidden_size"]
    per_layer_memory = qo_proj + kv_proj + kv_entries + layer_norm
    output_proj = model.dtype_size * c["hidden_size"] * c["vocab_size"]
    return embedding + per_layer_memory * c["num_hidden_layers"] + output_proj


def calculate_memory(
    model: Model, batch: list[tuple[Request, float]], mode: str
) -> int:
    total_memory = _mem_model(model)
    for request, _ in batch:
        tokens_to_process = request.remaining_tokens_prefill if mode == "prefill" else 1
        total_memory += _calculate_memory(
            model, tokens_to_process, request.cache_length
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
