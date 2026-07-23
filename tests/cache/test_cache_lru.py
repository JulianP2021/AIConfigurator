"""Tests for the two-tier distributed cache."""

import pytest

from src.cache.cache import Cache, CacheItem
from src.hardware.hardware import Hardware
from src.model.model import Model


@pytest.fixture
def medium_cache(fake_model: Model, small_hardware: Hardware) -> Cache:
    cache = Cache(
        layers={},
        node_hardware={0: small_hardware, 1: small_hardware},
        model=fake_model,
        ram_usage_fraction=0.8,
        ssd_usage_fraction=0.8,
    )
    # RAM fits three 512-token items (51_200 bytes each), SSD is roomy.
    cache.ram_capacity_bytes[0] = 160_000
    cache.ssd_capacity_bytes[0] = 500_000
    cache.ram_capacity_bytes[1] = 160_000
    cache.ssd_capacity_bytes[1] = 500_000
    return cache


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


class TestCacheLRU:
    def test_touch_updates_order(self, cache_with_fake_model: Cache):
        item1 = CacheItem((1, 0), 0, 100)
        item2 = CacheItem((2, 0), 0, 100)
        cache_with_fake_model.insert_cache_item(item1, 0)
        cache_with_fake_model.insert_cache_item(item2, 0)

        first_tick = item1.last_access_tick
        cache_with_fake_model._touch(item1, cache_with_fake_model._ram_layer(0))
        assert item1.last_access_tick > first_tick
        assert item1.last_access_tick > item2.last_access_tick

    def test_eviction_picks_lru_item(self, medium_cache: Cache):
        item1 = CacheItem((1, 0), 0, 512)
        item2 = CacheItem((2, 0), 0, 512)
        item3 = CacheItem((3, 0), 0, 512)
        medium_cache.insert_cache_item(item1, 0)
        medium_cache.insert_cache_item(item2, 0)
        medium_cache.insert_cache_item(item3, 0)

        # Touch item1 so it is more recently used than item2 and item3.
        medium_cache._touch(item1, medium_cache._ram_layer(0))

        item4 = CacheItem((4, 0), 0, 512)
        medium_cache.insert_cache_item(item4, 0)

        # item2 should have been evicted because it was LRU; item1 stays in RAM
        # because it was touched after item2 and item3 were inserted.
        assert item2 in medium_cache._ssd_layer(0).content[(2, 0)].values()
        assert item1 in medium_cache._ram_layer(0).content[(1, 0)].values()

    def test_lru_eviction_from_ssd_prefers_least_recently_used(
        self, small_cache: Cache
    ):
        """_evict_ssd_lru removes the least-recently-used SSD item."""
        small_cache.ssd_capacity_bytes[0] = 102_600  # exactly two 512-token items
        small_cache.insert_cache_item(CacheItem((1, 0), 0, 512), 0)
        small_cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)
        small_cache.insert_cache_item(CacheItem((3, 0), 0, 512), 0)
        # item1 and item2 are in SSD; item3 in RAM. item1 is LRU.
        assert small_cache.ssd_usage_bytes[0] == 2 * small_cache._item_size(
            CacheItem((1, 0), 0, 512)
        )

        small_cache._evict_ssd_lru(0)

        assert (1, 0) not in small_cache._ssd_layer(0).content
        assert (2, 0) in small_cache._ssd_layer(0).content
