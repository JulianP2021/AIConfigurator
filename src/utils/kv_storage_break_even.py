"""Break-even storage duration for cached KV tensors.

Given a model, an input sequence length, and a hardware preset, compute the
length of time for which storing the KV cache is cheaper than recomputing it
from scratch on every request.

The comparison is per-GPU: one KV cache belongs to one GPU instance, so the
recompute cost uses the GPU's prefill time and the per-GPU hourly price.
"""

from src.hardware.hardware import Hardware
from src.hardware.scraper import get_pricing
from src.model.model import Model
from src.utils.utils import _calculate_flops, _calculate_memory


_S3_STORAGE_COST_USD_PER_GB_MONTH = 0.022
_HOURS_PER_MONTH = 30 * 24


def kv_storage_break_even_seconds(
    isl: int,
    model: Model,
    hardware: Hardware,
    ram_price_usd_per_gb_hour: float | None = None,
    ssd_price_usd_per_gb_hour: float | None = None,
    s3_storage_cost_usd_per_gb_month: float = _S3_STORAGE_COST_USD_PER_GB_MONTH,
) -> dict[str, float]:
    """Return break-even storage durations for RAM, SSD, and S3.

    For each storage tier, the returned value is the duration (in seconds) at
    which the cumulative storage cost equals the one-time cost of recomputing
    the full ``isl``-token prompt from scratch on the target GPU.

    Parameters
    ----------
    isl:
        Input sequence length (tokens). The full prompt is assumed uncached.
    model:
        Model object providing ``kv_size_per_token`` and FLOPs/memory helpers.
    hardware:
        Hardware preset. Must provide ``spec.gpu_hardware`` and a non-zero
        ``spec.dph_base`` (node hourly price). Per-GPU price is derived as
        ``dph_base / num_gpus``.
    ram_price_usd_per_gb_hour:
        Optional RAM storage price. If ``None``, loaded from the AWS pricing
        metadata; if still unavailable, RAM break-even is ``inf``.
    ssd_price_usd_per_gb_hour:
        Optional SSD storage price. If ``None``, loaded from the AWS pricing
        metadata; if still unavailable, SSD break-even is ``inf``.
    s3_storage_cost_usd_per_gb_month:
        S3 storage cost in USD per GB per month. Defaults to $0.022.

    Returns:
    -------
    Dictionary with keys ``ram``, ``ssd``, ``s3``. Values are seconds; shorter
    durations mean the tier pays for itself quickly. ``float("inf")`` means the
    tier is effectively free or the recompute cost is zero.
    """
    if isl <= 0:
        return {"ram": float("inf"), "ssd": float("inf"), "s3": float("inf")}

    gpu = hardware.spec.gpu_hardware
    flops = _calculate_flops(model, isl, cache_len=0)
    memory = _calculate_memory(model, isl, cache_len=0)
    recompute_time_s = max(flops / gpu.flops, memory / gpu.gpu_bw)

    per_gpu_hourly_price = hardware.spec.dph_base / max(1, hardware.spec.num_gpus)
    recompute_cost_usd = per_gpu_hourly_price * (recompute_time_s / 3600.0)

    kv_size_bytes = model.kv_size_tokens(isl)
    kv_size_gb = kv_size_bytes / (1024**3)

    print("KV size: ", kv_size_bytes, " in GB: ", kv_size_gb)

    pricing = get_pricing()
    if ram_price_usd_per_gb_hour is None:
        ram_price_usd_per_gb_hour = pricing.get("cpu_ram_usd_per_gb_hour", 0.0)
    if ssd_price_usd_per_gb_hour is None:
        ssd_price_usd_per_gb_hour = pricing.get("ssd_usd_per_gb_hour", 0.0)

    s3_price_per_gb_hour = s3_storage_cost_usd_per_gb_month / _HOURS_PER_MONTH

    def break_even(cost_per_gb_hour: float) -> float:
        if cost_per_gb_hour <= 0.0 or kv_size_gb <= 0.0 or recompute_cost_usd <= 0.0:
            return float("inf")
        return recompute_cost_usd / (kv_size_gb * cost_per_gb_hour) * 3600.0

    return {
        "ram": break_even(ram_price_usd_per_gb_hour),
        "ssd": break_even(ssd_price_usd_per_gb_hour),
        "s3": break_even(s3_price_per_gb_hour),
    }
