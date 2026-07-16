"""KV-cache storage cost helpers.

Utilities for computing the hourly cost of storing a given number of KV tokens
in RAM, SSD, or S3.  Costs are derived from the AWS pricing metadata by default
but can be overridden by the caller.
"""

from pathlib import Path

from src.model.model import Model


_S3_STORAGE_COST_USD_PER_GB_MONTH = 0.022
_HOURS_PER_MONTH = 30 * 24


def _default_aws_pricing() -> dict[str, float]:
    """Load unit prices from the AWS hardware JSON file, if present."""
    pricing_path = Path(__file__).parent.parent / "hardware" / "aws_hardware.json"
    if not pricing_path.exists():
        return {}
    import json

    data = json.loads(pricing_path.read_text(encoding="utf-8"))
    return data.get("_pricing", {})


def kv_storage_cost_per_hour_usd(
    tokens: int,
    model: Model,
    ram_price_usd_per_gb_hour: float | None = None,
    ssd_price_usd_per_gb_hour: float | None = None,
    s3_storage_cost_usd_per_gb_month: float = _S3_STORAGE_COST_USD_PER_GB_MONTH,
    kv_size_gb_per_token: float | None = None,
) -> dict[str, float]:
    """Return the hourly cost in USD to store ``tokens`` KV tokens per tier.

    Parameters
    ----------
    tokens:
        Number of KV tokens to store.
    model:
        Model object providing ``kv_size_per_token``. Used only when
        ``kv_size_gb_per_token`` is not provided.
    ram_price_usd_per_gb_hour:
        Optional RAM storage price. If ``None``, loaded from the AWS pricing
        metadata; if still unavailable, RAM cost is ``inf``.
    ssd_price_usd_per_gb_hour:
        Optional SSD storage price. If ``None``, loaded from the AWS pricing
        metadata; if still unavailable, SSD cost is ``inf``.
    s3_storage_cost_usd_per_gb_month:
        S3 storage cost in USD per GB per month. Defaults to $0.022.
    kv_size_gb_per_token:
        Optional KV size in GB per token. If provided, overrides the model's
        ``kv_size_per_token``.

    Returns:
    -------
    Dictionary with keys ``ram``, ``ssd``, ``s3`` giving the USD/hour cost for
    storing the KV cache in that tier.
    """
    if tokens <= 0:
        return {"ram": 0.0, "ssd": 0.0, "s3": 0.0}

    if kv_size_gb_per_token is None:
        kv_size_bytes = model.kv_size_per_token * tokens
        kv_size_gb = kv_size_bytes / (1024**3)
    else:
        kv_size_gb = kv_size_gb_per_token * tokens

    pricing = _default_aws_pricing()
    if ram_price_usd_per_gb_hour is None:
        ram_price_usd_per_gb_hour = pricing.get("cpu_ram_usd_per_gb_hour", 0.0)
    if ssd_price_usd_per_gb_hour is None:
        ssd_price_usd_per_gb_hour = pricing.get("ssd_usd_per_gb_hour", 0.0)

    s3_price_per_gb_hour = s3_storage_cost_usd_per_gb_month / _HOURS_PER_MONTH

    def cost(price_per_gb_hour: float) -> float:
        if price_per_gb_hour <= 0.0 or kv_size_gb <= 0.0:
            return float("inf") if price_per_gb_hour <= 0.0 else 0.0
        return kv_size_gb * price_per_gb_hour

    return {
        "ram": cost(ram_price_usd_per_gb_hour),
        "ssd": cost(ssd_price_usd_per_gb_hour),
        "s3": cost(s3_price_per_gb_hour),
    }
