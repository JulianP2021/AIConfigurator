"""Shared test fixtures for the simulator test suite."""

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from src.cache.cache import Cache
from src.hardware.hardware import GPUHardwareSpec, Hardware, HardwareSpec
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


@pytest.fixture
def tiny_hardware() -> Hardware:
    """A hardware preset with small, deterministic memory/bandwidth values."""
    gpu_spec = GPUHardwareSpec(flops=1, gpu_mem=1_000_000_000, gpu_bw=1_000_000_000)
    spec = HardwareSpec(
        gpu_hardware=gpu_spec,
        num_gpus=1,
        ram_mem=10_000_000_000,  # 10 GB
        ram_bw=10_000_000_000,  # 10 GB/s
        nvme_mem=5_000_000_000,  # 5 GB
        nvme_bw=1_000_000_000,  # 1 GB/s
        network_inet_up=100_000_000,  # 100 MB/s
        network_inet_down=200_000_000,  # 200 MB/s
        price_usd_per_hour=1.0,
        price_inet_up=0.0,
        price_inet_down=0.0,
    )
    return Hardware(name="tiny", spec=spec)


@pytest.fixture
def small_hardware() -> Hardware:
    """A hardware preset that can barely hold a few 512-token items in RAM/SSD."""
    gpu_spec = GPUHardwareSpec(flops=1, gpu_mem=1_000_000_000, gpu_bw=1_000_000_000)
    spec = HardwareSpec(
        gpu_hardware=gpu_spec,
        num_gpus=1,
        ram_mem=300_000,  # fits ~3 items of 100 bytes/token * 512 tokens
        ram_bw=1_000_000_000,
        nvme_mem=200_000,  # fits ~2 items
        nvme_bw=100_000_000,
        network_inet_up=10_000_000,
        network_inet_down=20_000_000,
        price_usd_per_hour=1.0,
        price_inet_up=0.0,
        price_inet_down=0.0,
    )
    return Hardware(name="small", spec=spec)


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
        req = Request(isl=isl, osl=osl, cached=cached, user_id=user_id)
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
