from functools import lru_cache

from src.model.model import Model
from src.request.request import Request


# Match NVIDIA AI Configurator SOL model:
#  * Q/K/V are fused into one GEMM.
#  * O projection is a separate GEMM.
#  * FFN1 (gate+up) and FFN2 are separate GEMMs.
#  * Attention FLOPs use num_heads * head_dim and are causal.
#  * LM-head / logits GEMM only computes one logit per sequence in prefill.
#  * Input embedding is a lookup table and contributes negligible FLOPs.


def _gemm_flops(m: int, n: int, k: int) -> int:
    """Fused multiply-add FLOPs for a single GEMM: 2*m*n*k."""
    return 2 * m * n * k


def _gemm_mem_bytes(m: int, n: int, k: int, dtype_size: float) -> int:
    """Memory traffic for one GEMM under the SOL roofline model.

    NVIDIA SOL uses ``memory * (m*n + m*k + n*k)`` for the operand/result
    footprint.  This is the same convention used in
    ``perf_database.py::query_gemm``.
    """
    return int(dtype_size * (m * n + m * k + n * k))


@lru_cache(maxsize=1280)
def _calculate_flops(model: Model, tokens_to_process: int, cache_len: int) -> int:
    c = model.cost_constants
    cfg = model.config
    # dtype_size = model.dtype_size

    h = c["hidden_size"]
    n_kv = int(cfg["num_key_value_heads"])
    head_dim = c["d_kv"]
    n_heads = int(cfg.get("num_attention_heads", h // head_dim))
    inter = c["intermediate_size"]
    vocab = c["vocab_size"]
    layers = c["num_hidden_layers"]

    m = tokens_to_process
    full_s = tokens_to_process + cache_len

    # Fused QKV projection: output width = Q + K + V
    qkv_out = n_heads * head_dim + 2 * n_kv * head_dim
    qkv_gemm = _gemm_flops(m, qkv_out, h)

    # O projection
    o_proj = _gemm_flops(m, h, n_heads * head_dim)

    # Causal attention over (full_s) positions, but only ``m`` new queries.
    # Formula mirrors NVIDIA ``query_context_attention`` SOL path:
    #   ops = 2 * b * (full_s^2 - prefix^2) * n_heads * head_dim
    attn = 2 * (full_s * full_s - cache_len * cache_len) * n_heads * head_dim

    # FFN: gate+up fused, then down
    ffn1 = _gemm_flops(m, 2 * inter, h)
    ffn2 = _gemm_flops(m, h, inter)

    per_layer_flops = qkv_gemm + o_proj + attn + ffn1 + ffn2

    # LM head / logits: only one logit vector per sequence in prefill.
    # For decode, ``m`` is already 1.  This matches NVIDIA's ``logits_gemm``
    # where ``x = batch_size`` rather than the full token count.
    logits = _gemm_flops(1, vocab, h)

    return int(per_layer_flops * layers + logits)


def calculate_flops(model: Model, batch: list[tuple[Request, float]], mode: str) -> int:
    total_flops = 0
    for request, _ in batch:
        tokens_to_process = request.remaining_tokens_prefill if mode == "prefill" else 1
        total_flops += _calculate_flops(model, tokens_to_process, request.cache_length)
    return int(total_flops)


@lru_cache(maxsize=1280)
def _calculate_memory(model: Model, tokens_to_process: int, cache_len: int) -> int:
    """Activation/KV memory traffic for one forward step.

    This follows the NVIDIA SOL convention of summing operand/result traffic
    for each GEMM and attention operation, then taking ``max(flops/flops_bw,
    mem/mem_bw)`` at the caller.
    """
    c = model.cost_constants
    cfg = model.config
    dtype_size = model.dtype_size

    h = int(c["hidden_size"])
    n_kv = int(cfg["num_key_value_heads"])
    head_dim = int(c["d_kv"])
    n_heads = int(cfg.get("num_attention_heads", h // head_dim))
    inter = int(c["intermediate_size"])
    vocab = int(c["vocab_size"])
    layers = int(c["num_hidden_layers"])

    m = tokens_to_process
    full_s = tokens_to_process + cache_len

    qkv_out = n_heads * head_dim + 2 * n_kv * head_dim

    # GEMM memory traffic
    embedding = h * vocab
    qkv_mem = _gemm_mem_bytes(m, qkv_out, h, dtype_size)
    o_proj_mem = _gemm_mem_bytes(m, h, n_heads * head_dim, dtype_size)
    ffn1_mem = _gemm_mem_bytes(m, 2 * inter, h, dtype_size)
    ffn2_mem = _gemm_mem_bytes(m, h, inter, dtype_size)
    logits_mem = _gemm_mem_bytes(1, vocab, h, dtype_size)

    # Attention memory traffic (NVIDIA SOL style):
    #   Q read + output write at full precision, plus K/V cache read.
    #   We use the full-precision dtype size for Q/output; KV cache is also
    #   full precision in our model (no quantization).
    kv_bytes_per_elem = dtype_size
    attn_mem = int(
        2 * dtype_size * n_heads * head_dim * m  # Q read + output write
        + kv_bytes_per_elem * 2 * n_kv * head_dim * full_s  # K/V cache read
    )

    # Two layer-norm / residual element-wise passes per layer.
    # Approximate as read+write of hidden activations twice.
    elementwise_mem = int(2 * 2 * dtype_size * h * m)

    per_layer_mem = (
        qkv_mem + o_proj_mem + attn_mem + ffn1_mem + ffn2_mem + elementwise_mem
    )

    return int(embedding + per_layer_mem * layers + logits_mem)


def calculate_memory(
    model: Model, batch: list[tuple[Request, float]], mode: str
) -> int:
    """Return activation + GEMM operand memory traffic for the batch.

    We intentionally do **not** add the global weight footprint here.  NVIDIA's
    SOL model accounts for weight reads inside each GEMM operation via the
    ``m*n + m*k + n*k`` operand/result footprint, so summing per-op traffic
    already captures weight bandwidth once per forward pass.
    """
    total_memory = 0
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
