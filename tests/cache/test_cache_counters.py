"""Tests for the two-tier distributed cache."""

import pytest

from src.cache.cache import Cache, CacheItem
from src.hardware.hardware import Hardware, S3Spec
from src.model.model import Model


@pytest.fixture
def small_cache(fake_model: Model, small_hardware: Hardware) -> Cache:
    cache = Cache(
        layers={},
        node_hardware={0: small_hardware, 1: small_hardware},
        model=fake_model,
        ram_usage_fraction=0.8,
        ssd_usage_fraction=0.8,
    )
    # Shrink RAM to hold exactly one 512-token item; leave SSD roomy so tests
    # can observe RAM->SSD evictions without immediate SSD-delete churn.
    cache.ram_capacity_bytes[0] = 51_300
    cache.ssd_capacity_bytes[0] = 500_000
    cache.ram_capacity_bytes[1] = 51_300
    cache.ssd_capacity_bytes[1] = 500_000
    return cache


class TestCacheByteCounters:
    def test_ram_usage_bytes_updated_on_insert(self, cache_with_fake_model: Cache):
        item = CacheItem((1, 0), 0, 100)
        cache_with_fake_model.insert_cache_item(item, 0)
        assert cache_with_fake_model.ram_usage_bytes[0] == 10_000

    def test_ram_and_ssd_counters_updated_on_eviction(self, small_cache: Cache):
        item1 = CacheItem((1, 0), 0, 512)
        item2 = CacheItem((2, 0), 0, 512)
        item_size = small_cache._item_size(item1)
        small_cache.insert_cache_item(item1, 0)
        small_cache.insert_cache_item(item2, 0)

        assert small_cache.ram_usage_bytes[0] == item_size
        assert small_cache.ssd_usage_bytes[0] == item_size

    def test_delete_updates_ram_counter(self, cache_with_fake_model: Cache):
        item = CacheItem((1, 0), 0, 100)
        cache_with_fake_model.insert_cache_item(item, 0)
        cache_with_fake_model.delete_item(item)
        assert cache_with_fake_model.ram_usage_bytes[0] == 0

    def test_s3_counter_updated_on_eviction_upload(
        self, fake_model: Model, s3_tiny_hardware: Hardware, s3_enabled: S3Spec
    ):
        cache = Cache(
            layers={},
            node_hardware={0: s3_tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )
        item1 = CacheItem((1, 40), 0, 512)
        item2 = CacheItem((2, 0), 0, 512)
        item3 = CacheItem((3, 0), 0, 512)
        cache.insert_cache_item(item1, 0)
        cache.insert_cache_item(item2, 0)
        cache.insert_cache_item(item3, 0)

        item_size = cache._item_size(item1)
        assert cache.s3_usage_bytes == item_size

    def test_usage_summary_matches_counters(self, small_cache: Cache):
        small_cache.insert_cache_item(CacheItem((1, 0), 0, 512), 0)
        small_cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)

        summary = small_cache.usage_summary()
        assert summary["ram_usage_bytes"] == small_cache.ram_usage_bytes[0]
        assert summary["ssd_usage_bytes"] == small_cache.ssd_usage_bytes[0]
        assert summary["s3_usage_bytes"] == small_cache.s3_usage_bytes

    def test_usage_summary_sums_across_nodes(
        self, fake_model: Model, tiny_hardware: Hardware
    ):
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware, 1: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
        )
        cache.insert_cache_item(CacheItem((1, 0), 0, 100), 0)
        cache.insert_cache_item(CacheItem((2, 0), 0, 100), 1)

        summary = cache.usage_summary()
        assert summary["ram_usage_bytes"] == 20_000
        assert summary["ram_capacity_bytes"] == sum(cache.ram_capacity_bytes.values())

    def test_counters_after_delete_ssd_item(self, cache_with_fake_model: Cache):
        original_capacity = cache_with_fake_model.ram_capacity_bytes[0]
        # Shrink RAM to exactly one 100-token item so the second insert evicts to SSD.
        cache_with_fake_model.ram_capacity_bytes[0] = 10_050
        cache_with_fake_model.insert_cache_item(CacheItem((1, 0), 0, 100), 0)
        cache_with_fake_model.insert_cache_item(CacheItem((2, 0), 0, 100), 0)
        cache_with_fake_model.ram_capacity_bytes[0] = original_capacity

        # item (1,0) is now in SSD; delete it.
        ssd_item = next(
            iter(cache_with_fake_model._ssd_layer(0).content[(1, 0)].values())
        )
        cache_with_fake_model.delete_item(ssd_item)

        assert cache_with_fake_model.ssd_usage_bytes[0] == 0

    def test_counters_after_delete_s3_item(
        self, fake_model: Model, s3_tiny_hardware: Hardware, s3_enabled: S3Spec
    ):
        cache = Cache(
            layers={},
            node_hardware={0: s3_tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )
        cache.insert_cache_item(CacheItem((1, 0), 0, 512), 0)
        cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)
        cache.insert_cache_item(CacheItem((3, 0), 0, 512), 0)

        s3_item = next(iter(cache._s3_layer().content[(1, 0)].values()))
        cache.delete_item(s3_item)
        assert cache.s3_usage_bytes == 0
