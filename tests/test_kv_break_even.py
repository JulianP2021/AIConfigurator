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
        cpu_cores=64,
        cpu_cores_effective=64.0,
        cpu_ghz=2.5,
        cpu_name="test",
        cpu_ram=1_000_000_000_000,
        disk_name="test",
        dlperf=1.0,
        dlperf_per_dphtotal=1.0,
        dph_base=dph_base,
        geolocation="test",
        gpu_display_active=False,
        gpu_frac=1.0,
        gpu_lanes=16,
        gpu_max_power=0.0,
        gpu_max_temp=0.0,
        has_avx=1,
        host_id=0,
        inet_down_cost=0.0,
        inet_up_cost=0.0,
        mobo_name="test",
        os_version="test",
        pci_gen=4.0,
        pcie_bw=100_000_000_000,
        network_bw=100_000_000_000,
        reliability=1.0,
        reliability_mult=1.0,
        score=1.0,
        storage_cost=0.0,
        storage_total_cost=0.0,
        verification="test",
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
