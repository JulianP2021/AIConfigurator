"""Tests for the two-tier distributed cache."""

import pytest

from src.cache.cache import Cache
from src.hardware.hardware import Hardware
from src.model.model import Model


class TestCacheValidation:
    def test_validation_passes_with_sufficient_capacity(
        self, fake_model: Model, small_hardware: Hardware
    ):
        Cache(
            layers={},
            node_hardware={0: small_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
        )

    def test_validation_passes_with_zero_nvme(
        self, fake_model: Model, no_ssd_hardware: Hardware
    ):
        """A node with no NVMe storage is allowed because SSD tier simply does not exist."""
        cache = Cache(
            layers={},
            node_hardware={0: no_ssd_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
        )
        assert cache._ssd_layer(0) is None

    def test_validation_fails_when_ram_too_small(
        self, fake_model: Model, small_hardware: Hardware
    ):
        with pytest.raises(ValueError, match="RAM capacity"):
            Cache(
                layers={},
                node_hardware={0: small_hardware},
                model=fake_model,
                ram_usage_fraction=1e-9,
                ssd_usage_fraction=0.8,
            )

    def test_validation_fails_when_ssd_too_small(
        self, fake_model: Model, small_hardware: Hardware
    ):
        with pytest.raises(ValueError, match="SSD capacity"):
            Cache(
                layers={},
                node_hardware={0: small_hardware},
                model=fake_model,
                ram_usage_fraction=0.8,
                ssd_usage_fraction=1e-9,
            )
