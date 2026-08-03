"""Tests for KV cache break-even storage duration."""

from src.hardware.hardware import GPUHardwareSpec, Hardware, HardwareSpec
from src.model.model import Model
from src.utils.kv_storage_break_even import kv_storage_break_even_seconds


def _fake_hardware(dph_base: float = 27.52, num_gpus: int = 4) -> Hardware:
    spec = HardwareSpec(
        gpu_hardware=GPUHardwareSpec(
            flops=1_000_000_000_000, gpu_mem=80_000_000_000, gpu_bw=2_000_000_000
        ),
        num_gpus=num_gpus,
        nvme_mem=1_000_000_000_000,
        nvme_bw=10_000_000_000,
        network_inet_up=1_000_000_000,
        network_inet_down=1_000_000_000,
        network_inter_node_up=10_000_000_000,
        network_inter_node_down=10_000_000_000,
        cpu_ram=1_000_000_000,
        dph_base=dph_base,
        pcie_bw=100_000_000_000,
    )
    return Hardware(name="fake", spec=spec)


def test_break_even_increases_with_isl():
    model = Model("Qwen/Qwen3-8B")
    hardware = _fake_hardware()
    short = kv_storage_break_even_seconds(1_000, model, hardware)
    long = kv_storage_break_even_seconds(10_000, model, hardware)
    # Longer prompts take more time to recompute per GB, so break-even grows.
    assert long["ssd"] > short["ssd"]
    assert long["s3"] > short["s3"]


def test_break_even_zero_isl_is_infinity():
    model = Model("Qwen/Qwen3-8B")
    hardware = _fake_hardware()
    result = kv_storage_break_even_seconds(0, model, hardware)
    assert result["ram"] == float("inf")
    assert result["ssd"] == float("inf")
    assert result["s3"] == float("inf")


def test_break_even_faster_gpu_is_shorter():
    model = Model("Qwen/Qwen3-8B")
    slow_hw = _fake_hardware(dph_base=100.0)
    fast_hw = _fake_hardware(dph_base=10.0)
    slow = kv_storage_break_even_seconds(10_000, model, slow_hw)
    fast = kv_storage_break_even_seconds(10_000, model, fast_hw)
    # Cheaper/faster-per-dollar GPU makes recompute cheaper, so storage wins for
    # a shorter time.
    assert fast["ssd"] < slow["ssd"]


def test_custom_prices_override_defaults():
    model = Model("Qwen/Qwen3-8B")
    hardware = _fake_hardware()
    cheap = kv_storage_break_even_seconds(
        10_000, model, hardware, ssd_price_usd_per_gb_hour=0.0001
    )
    expensive = kv_storage_break_even_seconds(
        10_000, model, hardware, ssd_price_usd_per_gb_hour=0.01
    )
    assert cheap["ssd"] > expensive["ssd"]
