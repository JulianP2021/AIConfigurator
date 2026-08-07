"""Tests for the two-tier distributed cache."""

import pytest

from src.cache.cache import Cache, CacheItem
from src.hardware.hardware import Hardware, S3Spec
from src.model.model import Model
from src.request.request import Request


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


class TestCacheInsertion:
    def test_insert_adds_item_to_ram(self, cache_with_fake_model: Cache):
        item = CacheItem((1, 0), 0, 100)
        cache_with_fake_model.insert_cache_item(item, 0)

        assert item in cache_with_fake_model._ram_layer(0).content[(1, 0)].values()
        assert cache_with_fake_model.ram_usage_bytes[0] == 10_000
        assert cache_with_fake_model.ssd_usage_bytes[0] == 0

    def test_insert_evicts_to_ssd_when_ram_full(self, small_cache: Cache):
        item1 = CacheItem((1, 0), 0, 512)
        small_cache.insert_cache_item(item1, 0)

        # Force eviction: capacity is 240_000 bytes, each item is 51_200 bytes.
        item2 = CacheItem((2, 0), 0, 512)
        item3 = CacheItem((3, 0), 0, 512)
        item4 = CacheItem((4, 0), 0, 512)
        small_cache.insert_cache_item(item2, 0)
        small_cache.insert_cache_item(item3, 0)
        eviction_legs = small_cache.insert_cache_item(item4, 0)

        assert item4 in small_cache._ram_layer(0).content[(4, 0)].values()
        assert item1 in small_cache._ssd_layer(0).content[(1, 0)].values()
        assert len(eviction_legs) == 1
        assert eviction_legs[0].bottleneck == "SSD_LOCAL"

    def test_insert_deletes_ssd_lru_when_both_tiers_full(self, small_cache: Cache):
        # Shrink SSD so it can hold only one item. Insert three items; each new
        # item evicts the RAM LRU to SSD. When SSD is full, its LRU is deleted
        # first to make room for the next eviction.
        small_cache.ssd_capacity_bytes[0] = 51_300
        item1 = CacheItem((1, 0), 0, 512)
        item2 = CacheItem((2, 0), 0, 512)
        item3 = CacheItem((3, 0), 0, 512)
        small_cache.insert_cache_item(item1, 0)
        small_cache.insert_cache_item(item2, 0)
        small_cache.insert_cache_item(item3, 0)

        # item1 was evicted to SSD and then deleted to make room for item2.
        assert (1, 0) not in small_cache._ssd_layer(0).content
        # item3 should be in RAM; item2 should be the SSD LRU.
        assert item3 in small_cache._ram_layer(0).content[(3, 0)].values()
        assert item2 in small_cache._ssd_layer(0).content[(2, 0)].values()

    def test_insert_merges_with_overlapping_ram_item(
        self, cache_with_fake_model: Cache
    ):
        cache_with_fake_model.insert_cache_item(CacheItem((1, 0), 0, 100), 0)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 0), 50, 150), 0)

        ram = cache_with_fake_model._ram_layer(0)
        assert len(ram.content[(1, 0)]) == 1
        item = next(iter(ram.content[(1, 0)].values()))
        assert item.token_start == 0
        assert item.token_end == 150

    def test_insert_evicts_directly_to_s3_when_no_ssd(
        self, fake_model: Model, no_ssd_hardware: Hardware, s3_enabled: S3Spec
    ):
        cache = Cache(
            layers={},
            node_hardware={0: no_ssd_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )
        # Capacity is 8GB so a 512-token item (51_200 bytes) fits without issue,
        # but we force an eviction by shrinking RAM.
        cache.ram_capacity_bytes[0] = 51_300
        cache.insert_cache_item(CacheItem((1, 0), 0, 512), 0)
        eviction_legs = cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)

        s3_item = next(iter(cache._s3_layer().content[(1, 0)].values()), None)
        assert s3_item is not None
        assert s3_item.token_start == 0
        assert s3_item.token_end == 512
        assert any(leg.bottleneck == "S3_UPLOAD" for leg in eviction_legs)

    def test_insert_drops_when_no_ssd_and_s3_disabled(
        self, fake_model: Model, no_ssd_hardware: Hardware
    ):
        cache = Cache(
            layers={},
            node_hardware={0: no_ssd_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=S3Spec.from_gbps(enabled=False),
        )
        cache.ram_capacity_bytes[0] = 51_300
        cache.insert_cache_item(CacheItem((1, 0), 0, 512), 0)

        cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)

        assert len(cache.download_kv(0, Request(0, 1000, 1, 0)).tracks) == 0

    def test_insert_merges_into_ssd_during_eviction(self, small_cache: Cache):
        """When a new RAM item overlaps an existing SSD item, the SSD copy merges in."""
        small_cache.insert_cache_item(CacheItem((1, 0), 0, 256), 0)
        small_cache.insert_cache_item(
            CacheItem((2, 0), 0, 512), 0
        )  # evicts item1 to SSD

        # A new item covering both the old RAM region and the SSD victim merges
        # with the SSD copy and replaces the RAM entry.
        small_cache.insert_cache_item(CacheItem((1, 0), 0, 512), 0)

        ssd = small_cache._ssd_layer(0)
        ssd_items = list(ssd.content.get((2, 0), {}).values())
        ram = small_cache._ram_layer(0)
        ram_items = list(ram.content.get((1, 0), {}).values())
        assert len(ram_items) == 1
        assert len(ssd_items) == 1
        assert ram_items[0].token_start == 0
        assert ram_items[0].token_end == 512

    def test_insert_item_larger_than_ram_fails(
        self, fake_model: Model, small_hardware: Hardware
    ):
        """Inserting an item larger than total RAM capacity raises RuntimeError.

        The cache validates at construction that RAM can hold a minimal 512-token
        item, but a user could still try to insert an item larger than the total
        RAM capacity. In that case all existing items are evicted, then the loop
        fails to find a victim and raises.
        """
        cache = Cache(
            layers={},
            node_hardware={0: small_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
        )
        # RAM capacity is ~240KB (0.8 * 300KB). Create an item > RAM capacity.
        # Each token is 100 bytes, so 3000 tokens = 300KB > 240KB.
        large_item = CacheItem((1, 0), 0, 3000)

        with pytest.raises(
            RuntimeError,
            match=r"Item size \(300000 bytes\) exceeds node 0 RAM capacity \(240000 bytes\) even after full eviction",
        ):
            cache.insert_cache_item(large_item, 0)

        # The large item should NOT be in RAM (insertion failed)
        assert (1, 0) not in cache._ram_layer(0).content
