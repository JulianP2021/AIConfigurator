"""Tests for the two-tier distributed cache."""

import pytest

from src.cache.cache import Cache, CacheItem
from src.hardware.hardware import Hardware
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


class TestCacheInsertion:
    def test_insert_adds_item_to_ram(self, cache_with_fake_model: Cache):
        item = CacheItem(1, 0, 100)
        cache_with_fake_model.insert_cache_item(item, 0)

        assert item in cache_with_fake_model._ram_layer(0).content
        assert cache_with_fake_model.ram_usage_bytes[0] == 10_000
        assert cache_with_fake_model.ssd_usage_bytes[0] == 0

    def test_insert_evicts_to_ssd_when_ram_full(self, small_cache: Cache):
        item1 = CacheItem(1, 0, 512)
        small_cache.insert_cache_item(item1, 0)

        # Force eviction: capacity is 240_000 bytes, each item is 51_200 bytes.
        item2 = CacheItem(2, 0, 512)
        item3 = CacheItem(3, 0, 512)
        item4 = CacheItem(4, 0, 512)
        small_cache.insert_cache_item(item2, 0)
        small_cache.insert_cache_item(item3, 0)
        eviction_legs = small_cache.insert_cache_item(item4, 0)

        assert item4 in small_cache._ram_layer(0).content
        assert item1 in small_cache._ssd_layer(0).content
        assert len(eviction_legs) == 1
        assert eviction_legs[0].bottleneck == "SSD_LOCAL"

    def test_insert_deletes_ssd_lru_when_both_tiers_full(self, small_cache: Cache):
        # Shrink SSD so it can hold only one item. Insert three items; each new
        # item evicts the RAM LRU to SSD. When SSD is full, its LRU is deleted
        # first to make room for the next eviction.
        small_cache.ssd_capacity_bytes[0] = 51_300
        item1 = CacheItem(1, 0, 512)
        item2 = CacheItem(2, 0, 512)
        item3 = CacheItem(3, 0, 512)
        small_cache.insert_cache_item(item1, 0)
        small_cache.insert_cache_item(item2, 0)
        small_cache.insert_cache_item(item3, 0)

        # item1 was evicted to SSD and then deleted to make room for item2.
        assert item1 not in small_cache._ssd_layer(0).content
        # item3 should be in RAM; item2 should be the SSD LRU.
        assert item3 in small_cache._ram_layer(0).content
        assert item2 in small_cache._ssd_layer(0).content


class TestCacheDownload:
    def test_download_without_cache_returns_empty_request(
        self, cache_with_fake_model: Cache, request_factory
    ):
        req = request_factory()
        dr = cache_with_fake_model.download_kv(0, req)
        assert dr.legs == []
        assert dr.active_leg is None

    def test_download_from_ram_local(self, cache_with_fake_model: Cache):
        req = Request(128, 8, 0, 1)
        req.id = 10
        cache_with_fake_model.insert_cache_item(CacheItem(req.id, 0, 128), 0)

        dr = cache_with_fake_model.download_kv(0, req)
        assert [leg.bottleneck for leg in dr.legs] == ["RAM_LOCAL"]

    def test_download_from_ram_remote(self, cache_with_fake_model: Cache):
        req = Request(128, 8, 0, 1)
        req.id = 11
        cache_with_fake_model.insert_cache_item(CacheItem(req.id, 0, 128), 0)

        dr = cache_with_fake_model.download_kv(1, req)
        assert [leg.bottleneck for leg in dr.legs] == [
            "RAM_LOCAL",
            "NETWORK",
            "RAM_LOCAL",
        ]

    def test_download_from_ssd_remote(self, small_cache: Cache):
        req = Request(512, 8, 0, 1)
        req.id = 12
        small_cache.insert_cache_item(CacheItem(req.id, 0, 512), 0)
        # Evict request 12 to SSD by inserting another item.
        small_cache.insert_cache_item(CacheItem(13, 0, 512), 0)

        source_layer = small_cache.find_cache_layer(small_cache.find_cache(req.id)[-1])
        assert source_layer is not None
        assert source_layer.name == "SSD"

        dr = small_cache.download_kv(1, req)
        assert [leg.bottleneck for leg in dr.legs] == [
            "SSD_LOCAL",
            "RAM_LOCAL",
            "NETWORK",
            "RAM_LOCAL",
        ]


class TestCacheLRU:
    def test_touch_updates_order(self, cache_with_fake_model: Cache):
        item1 = CacheItem(1, 0, 100)
        item2 = CacheItem(2, 0, 100)
        cache_with_fake_model.insert_cache_item(item1, 0)
        cache_with_fake_model.insert_cache_item(item2, 0)

        first_tick = item1.last_access_tick
        cache_with_fake_model._touch(item1)
        assert item1.last_access_tick > first_tick
        assert item1.last_access_tick > item2.last_access_tick

    def test_eviction_picks_lru_item(self, medium_cache: Cache):
        item1 = CacheItem(1, 0, 512)
        item2 = CacheItem(2, 0, 512)
        item3 = CacheItem(3, 0, 512)
        medium_cache.insert_cache_item(item1, 0)
        medium_cache.insert_cache_item(item2, 0)
        medium_cache.insert_cache_item(item3, 0)

        # Touch item1 so it is more recently used than item2 and item3.
        medium_cache._touch(item1)

        item4 = CacheItem(4, 0, 512)
        medium_cache.insert_cache_item(item4, 0)

        # item2 should have been evicted because it was LRU; item1 stays in RAM
        # because it was touched after item2 and item3 were inserted.
        assert item2 in medium_cache._ssd_layer(0).content
        assert item1 in medium_cache._ram_layer(0).content


class TestCacheUpload:
    def test_upload_creates_ram_local_leg(self, cache_with_fake_model: Cache):
        req = Request(128, 8, 0, 1)
        req.id = 20
        req.prefilled_tokens = 128
        req.decoded_tokens = 0

        ur = cache_with_fake_model.upload_kv(0, req)
        assert [leg.bottleneck for leg in ur.legs] == ["RAM_LOCAL"]
        assert ur.request.id == req.id

    def test_upload_appends_to_existing_cache(self, cache_with_fake_model: Cache):
        req = Request(128, 8, 0, 1)
        req.id = 21
        cache_with_fake_model.insert_cache_item(CacheItem(req.id, 0, 64), 0)
        req.prefilled_tokens = 128
        req.decoded_tokens = 0

        ur = cache_with_fake_model.upload_kv(0, req)
        assert ur.legs[-1].bottleneck == "RAM_LOCAL"
