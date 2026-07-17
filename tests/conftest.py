"""Shared test fixtures for the simulator test suite."""

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from src.cache.cache import Cache
from src.hardware.hardware import GPUHardwareSpec, Hardware, HardwareSpec, S3Spec
from src.hardware.scraper import _clear_combined_machine_db_cache
from src.model.model import Model
from src.request.request import Request


@pytest.fixture
def qwen_model() -> Model:
    """A real Model object with a known kv_size_per_token.

    This requires the transformers config files to be available.  If the model
    cannot be loaded, tests that depend on it are skipped.
    """
    try:
        return Model("Qwen/Qwen3-8B")
    except Exception as exc:  # pragma: no cover - depends on local cache
        pytest.skip(f"Could not load Qwen/Qwen3-8B model config: {exc}")


@pytest.fixture
def fake_model() -> Model:
    """A lightweight model stub with kv_size_per_token = 100 bytes."""
    model = MagicMock(spec=Model)
    model.kv_size_per_token = 100
    return model


def _make_spec(
    ram_mem: int,
    ram_bw: int,
    nvme_mem: int,
    nvme_bw: int,
    network_inet_up: int,
    network_inet_down: int,
    network_inter_node_up: int = 12_500_000_000,
    network_inter_node_down: int = 12_500_000_000,
) -> HardwareSpec:
    """Build a HardwareSpec with all required fields from the legacy Vast.ai schema."""
    gpu_spec = GPUHardwareSpec(flops=1, gpu_mem=1_000_000_000, gpu_bw=1_000_000_000)
    return HardwareSpec(
        gpu_hardware=gpu_spec,
        num_gpus=1,
        nvme_mem=nvme_mem,
        nvme_bw=nvme_bw,
        network_inet_up=network_inet_up,
        network_inet_down=network_inet_down,
        network_inter_node_up=network_inter_node_up,
        network_inter_node_down=network_inter_node_down,
        cpu_cores=1,
        cpu_cores_effective=1.0,
        cpu_ghz=1.0,
        cpu_name="test",
        cpu_ram=ram_mem,
        disk_name="test",
        dlperf=1.0,
        dlperf_per_dphtotal=1.0,
        dph_base=1.0,
        geolocation="test",
        gpu_display_active=False,
        gpu_frac=1.0,
        gpu_lanes=1,
        gpu_max_power=1.0,
        gpu_max_temp=1.0,
        has_avx=1,
        host_id=0,
        inet_down_cost=0.0,
        inet_up_cost=0.0,
        mobo_name="test",
        os_version="test",
        pci_gen=1.0,
        pcie_bw=ram_bw,
        network_bw=1.0,
        reliability=1.0,
        reliability_mult=1.0,
        score=1.0,
        storage_cost=0.0,
        storage_total_cost=0.0,
        verification="test",
    )


@pytest.fixture
def s3_enabled() -> S3Spec:
    """S3 spec with enabled 25 Gbps symmetric bandwidth for tests."""
    return S3Spec.from_gbps(enabled=True)


@pytest.fixture
def tiny_hardware() -> Hardware:
    """A hardware preset with small, deterministic memory/bandwidth values."""
    spec = _make_spec(
        ram_mem=10_000_000_000,  # 10 GB
        ram_bw=10_000_000_000,  # 10 GB/s
        nvme_mem=5_000_000_000,  # 5 GB
        nvme_bw=1_000_000_000,  # 1 GB/s
        network_inet_up=100_000_000,  # 100 MB/s
        network_inet_down=200_000_000,  # 200 MB/s
    )
    return Hardware(name="tiny", spec=spec)


@pytest.fixture
def small_hardware() -> Hardware:
    """A hardware preset that can barely hold a few 512-token items in RAM/SSD."""
    spec = _make_spec(
        ram_mem=300_000,  # fits ~3 items of 100 bytes/token * 512 tokens
        ram_bw=1_000_000_000,
        nvme_mem=200_000,  # fits ~2 items
        nvme_bw=100_000_000,
        network_inet_up=10_000_000,
        network_inet_down=20_000_000,
    )
    return Hardware(name="small", spec=spec)


@pytest.fixture
def s3_tiny_hardware() -> Hardware:
    """A hardware preset that fits exactly one 512-token item in RAM/SSD."""
    spec = _make_spec(
        ram_mem=65_000,
        ram_bw=1_000_000_000,
        nvme_mem=65_000,
        nvme_bw=100_000_000,
        network_inet_up=10_000_000,
        network_inet_down=20_000_000,
    )
    return Hardware(name="s3-tiny", spec=spec)


@pytest.fixture
def cache_with_fake_model(fake_model: Model, tiny_hardware: Hardware) -> Cache:
    """A cache backed by tiny hardware and a fake 100-byte-per-token model."""
    return Cache(
        layers={},
        node_hardware={0: tiny_hardware, 1: tiny_hardware},
        model=fake_model,
        ram_usage_fraction=0.8,
        ssd_usage_fraction=0.8,
    )


@pytest.fixture
def request_factory() -> Generator:
    """Yield a helper that creates Request objects with sequential ids."""
    counter = {"value": 0}

    def make(
        isl: int = 128, osl: int = 8, cached: int = 0, user_id: int = 0
    ) -> Request:
        req = Request(isl=isl, osl=osl, user_id=user_id)
        req.prefilled_tokens = cached
        # Resetting the global counter after each request is not safe, so we
        # just leave ids increasing across the test.
        counter["value"] += 1
        return req

    return make


@pytest.fixture
def reset_logger_mask() -> Generator:
    """Restore the original log mask after a test."""
    from src import logger

    original = logger.get_log_mask()
    try:
        yield
    finally:
        logger.set_log_mask(original)


@pytest.fixture(autouse=True)
def reset_machine_cache() -> Generator:
    """Clear the hardware DB cache before each test."""
    _clear_combined_machine_db_cache()
    yield
    _clear_combined_machine_db_cache()
