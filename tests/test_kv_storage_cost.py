"""Tests for KV cache hourly storage cost."""

from src.model.model import Model
from src.utils.kv_storage_cost import kv_storage_cost_per_hour_usd


def test_cost_increases_with_tokens():
    model = Model("Qwen/Qwen3-8B")
    cheap = kv_storage_cost_per_hour_usd(100_000, model)
    expensive = kv_storage_cost_per_hour_usd(1_000_000, model)
    for tier in ("ram", "ssd", "s3"):
        assert expensive[tier] > cheap[tier]


def test_zero_tokens_is_zero():
    model = Model("Qwen/Qwen3-8B")
    result = kv_storage_cost_per_hour_usd(0, model)
    assert result == {"ram": 0.0, "ssd": 0.0, "s3": 0.0}


def test_custom_prices():
    model = Model("Qwen/Qwen3-8B")
    cheap = kv_storage_cost_per_hour_usd(
        1_000_000, model, ssd_price_usd_per_gb_hour=0.0001
    )
    expensive = kv_storage_cost_per_hour_usd(
        1_000_000, model, ssd_price_usd_per_gb_hour=0.01
    )
    assert cheap["ssd"] < expensive["ssd"]


def test_kv_size_override():
    model = Model("Qwen/Qwen3-8B")
    default = kv_storage_cost_per_hour_usd(1_000_000, model)
    override = kv_storage_cost_per_hour_usd(1_000_000, model, kv_size_gb_per_token=1e-6)
    # 1e-6 GB/token is much smaller than the real model KV size.
    assert override["ssd"] < default["ssd"]
    assert override["ram"] < default["ram"]
