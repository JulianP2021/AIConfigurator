"""Tests for the two-tier distributed cache."""

import pytest

from src.cache.cache import Cache, CacheItem
from src.hardware.hardware import Hardware, S3Spec
from src.model.model import Model
from src.request.request import Request


def bottleneck_names(tracks: list[list]) -> list[str]:
    """Return bottleneck names flattened from a list of leg tracks."""
    return [leg.bottleneck for track in tracks for leg in track]


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
        item = CacheItem((1, 0), 0, 100)
        cache_with_fake_model.insert_cache_item(item, 0)

        assert item in cache_with_fake_model._ram_layer(0).content
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

        assert item4 in small_cache._ram_layer(0).content
        assert item1 in small_cache._ssd_layer(0).content
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
        assert dr.tracks == []
        assert dr.active_legs == []
        assert req.prefilled_tokens == 0

    def test_download_sets_prefilled_tokens_to_cached_prefix(
        self, cache_with_fake_model: Cache
    ):
        """download_kv must update request.prefilled_tokens to the cached prefix length."""
        req = Request(128, 8, user_id=1, session_id=10)
        assert req.prefilled_tokens == 0
        cache_with_fake_model.insert_cache_item(CacheItem((1, 10), 0, 128), 0)

        dr = cache_with_fake_model.download_kv(0, req)

        assert req.prefilled_tokens == 128
        assert dr.tracks == []

    def test_download_from_ram_local(self, cache_with_fake_model: Cache):
        req = Request(128, 8, user_id=1, session_id=10)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 10), 0, 128), 0)

        dr = cache_with_fake_model.download_kv(0, req)
        assert dr.tracks == []

    def test_download_from_ram_local_replaces_existing_item(
        self, cache_with_fake_model: Cache
    ):
        req = Request(128, 8, user_id=1, session_id=10)
        old_item = CacheItem((1, 10), 0, 128)
        cache_with_fake_model.insert_cache_item(old_item, 0)
        original_tick = old_item.last_access_tick

        dr = cache_with_fake_model.download_kv(0, req)
        assert dr.tracks == []

        ram_items = cache_with_fake_model.find_cache((1, 10))
        assert len(ram_items) == 1
        assert ram_items[0].last_access_tick > original_tick
        assert old_item not in cache_with_fake_model._ram_layer(0).content

    def test_download_from_ram_remote(self, cache_with_fake_model: Cache):
        req = Request(128, 8, user_id=1, session_id=11)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 11), 0, 128), 0)

        dr = cache_with_fake_model.download_kv(1, req)
        assert bottleneck_names(dr.tracks) == ["NETWORK", "RAM_LOCAL"]

    def test_download_from_ssd_remote(self, small_cache: Cache):
        req = Request(512, 8, user_id=1, session_id=12)
        small_cache.insert_cache_item(CacheItem((1, 12), 0, 512), 0)
        # Evict request 12 to SSD by inserting another item.
        small_cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)

        source_layer = small_cache.find_cache_layer(small_cache.find_cache((1, 12))[-1])
        assert source_layer is not None
        assert source_layer.name == "SSD"

        dr = small_cache.download_kv(1, req)
        assert bottleneck_names(dr.tracks) == [
            "SSD_LOCAL",
            "RAM_LOCAL",
            "NETWORK",
            "RAM_LOCAL",
        ]

    def test_download_merges_local_ssd_and_remote_ram(
        self, cache_with_fake_model: Cache
    ):
        req = Request(200, 8, user_id=1, session_id=30)
        # Node 0 SSD has 0-100.
        ssd_item = CacheItem((1, 30), 0, 100)
        ssd_layer = cache_with_fake_model._ssd_layer(0)
        ssd_layer.content.append(ssd_item)
        cache_with_fake_model.ssd_usage_bytes[0] += cache_with_fake_model._item_size(
            ssd_item
        )
        # Node 1 RAM has 100-200.
        cache_with_fake_model.insert_cache_item(CacheItem((1, 30), 100, 200), 1)

        dr = cache_with_fake_model.download_kv(0, req)
        assert [bottleneck_names([track]) for track in dr.tracks] == [
            ["SSD_LOCAL"],
            ["NETWORK", "RAM_LOCAL"],
        ]
        assert dr.tracks[0][0].remaining_bytes == cache_with_fake_model.kv_size(
            cache_with_fake_model.model, 100
        )
        assert dr.tracks[1][0].remaining_bytes == cache_with_fake_model.kv_size(
            cache_with_fake_model.model, 100
        )

        local_items = cache_with_fake_model.find_cache((1, 30), node_id=0)
        assert len(local_items) == 1
        assert local_items[0].token_start == 0
        assert local_items[0].token_end == 200
        # Remote source copy is retained.
        remote_items = cache_with_fake_model._ram_layer(1).content
        assert any(
            item.session_id == (1, 30)
            and item.token_start == 100
            and item.token_end == 200
            for item in remote_items
        )
        # Local SSD copy was promoted, so SSD is empty.
        assert cache_with_fake_model.ssd_usage_bytes[0] == 0

    def test_download_skips_local_ram_segment(self, cache_with_fake_model: Cache):
        req = Request(200, 8, user_id=1, session_id=31)
        # Node 0 RAM already has 0-100.
        cache_with_fake_model.insert_cache_item(CacheItem((1, 31), 0, 100), 0)
        # Node 1 RAM has 100-200.
        cache_with_fake_model.insert_cache_item(CacheItem((1, 31), 100, 200), 1)

        dr = cache_with_fake_model.download_kv(0, req)
        assert bottleneck_names(dr.tracks) == ["NETWORK", "RAM_LOCAL"]

        local_items = cache_with_fake_model.find_cache((1, 31), node_id=0)
        assert len(local_items) == 1
        assert local_items[0].token_start == 0
        assert local_items[0].token_end == 200

    def test_download_merges_local_ssd_and_remote_ssd(self, small_cache: Cache):
        req = Request(512, 8, user_id=1, session_id=33)
        # Node 0 SSD has 0-256 (insert then evict to SSD).
        small_cache.insert_cache_item(CacheItem((1, 33), 0, 256), 0)
        small_cache.insert_cache_item(CacheItem((99, 0), 0, 512), 0)
        # Node 1 SSD has 256-512.
        small_cache.insert_cache_item(CacheItem((1, 33), 256, 512), 1)
        small_cache.insert_cache_item(CacheItem((98, 0), 0, 512), 1)

        # Make room on node 0 RAM so the merged item inserts without eviction.
        item99 = next(
            item
            for item in small_cache._ram_layer(0).content
            if item.session_id == (99, 0)
        )
        small_cache.delete_item(item99)

        dr = small_cache.download_kv(0, req)
        assert [bottleneck_names([track]) for track in dr.tracks] == [
            ["SSD_LOCAL"],
            ["SSD_LOCAL", "RAM_LOCAL", "NETWORK", "RAM_LOCAL"],
        ]

        local_items = small_cache.find_cache((1, 33), node_id=0)
        assert len(local_items) == 1
        assert local_items[0].token_start == 0
        assert local_items[0].token_end == 512


class TestCacheS3:
    def test_ssd_eviction_uploads_to_s3(
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

        cache.insert_cache_item(CacheItem((1, 40), 0, 512), 0)
        # First insertion: req 40 evicted to SSD.
        cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)
        # Second insertion: SSD full, req 40 uploaded to S3 to make room.
        cache.insert_cache_item(CacheItem((3, 0), 0, 512), 0)

        s3_items = [
            item for item in cache._s3_layer().content if item.session_id == (1, 40)
        ]
        assert len(s3_items) == 1
        assert s3_items[0].token_start == 0
        assert s3_items[0].token_end == 512

    def test_download_falls_back_to_s3(
        self, cache_with_fake_model: Cache, s3_enabled: S3Spec
    ):
        cache = Cache(
            layers={},
            node_hardware=cache_with_fake_model.node_hardware,
            model=cache_with_fake_model.model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )

        req = Request(512, 8, user_id=1, session_id=42)
        # Place a copy only in S3.
        s3_layer = cache._s3_layer()
        s3_layer.content.append(CacheItem((1, 42), 0, 512))

        dr = cache.download_kv(0, req)
        assert [bottleneck_names([track]) for track in dr.tracks] == [["S3_DOWNLOAD"]]

        local_items = cache.find_cache((1, 42), node_id=0)
        assert len(local_items) == 1
        assert local_items[0].token_start == 0
        assert local_items[0].token_end == 512

    def test_s3_not_used_when_disabled(self, cache_with_fake_model: Cache):
        cache = Cache(
            layers={},
            node_hardware=cache_with_fake_model.node_hardware,
            model=cache_with_fake_model.model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=S3Spec.from_gbps(enabled=False),
        )

        req = Request(512, 8, 0, 1)
        req.id = 45
        # No local cache and no S3 layer.
        dr = cache.download_kv(0, req)
        assert dr.tracks == []


class TestCacheLRU:
    def test_touch_updates_order(self, cache_with_fake_model: Cache):
        item1 = CacheItem((1, 0), 0, 100)
        item2 = CacheItem((2, 0), 0, 100)
        cache_with_fake_model.insert_cache_item(item1, 0)
        cache_with_fake_model.insert_cache_item(item2, 0)

        first_tick = item1.last_access_tick
        cache_with_fake_model._touch(item1)
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
        medium_cache._touch(item1)

        item4 = CacheItem((4, 0), 0, 512)
        medium_cache.insert_cache_item(item4, 0)

        # item2 should have been evicted because it was LRU; item1 stays in RAM
        # because it was touched after item2 and item3 were inserted.
        assert item2 in medium_cache._ssd_layer(0).content
        assert item1 in medium_cache._ram_layer(0).content


class TestCacheUpload:
    def test_upload_creates_ram_local_track(self, cache_with_fake_model: Cache):
        req = Request(128, 8, 0, 1)
        req.id = 20
        req.prefilled_tokens = 128
        req.decoded_tokens = 0

        ur = cache_with_fake_model.upload_kv(0, req)
        assert bottleneck_names(ur.tracks) == ["RAM_LOCAL"]
        assert ur.request.id == req.id

    def test_upload_appends_to_existing_cache(self, cache_with_fake_model: Cache):
        req = Request(128, 8, 0, 1)
        req.id = 21
        cache_with_fake_model.insert_cache_item(CacheItem((1, 21), 0, 64), 0)
        req.prefilled_tokens = 128
        req.decoded_tokens = 0

        ur = cache_with_fake_model.upload_kv(0, req)
        assert bottleneck_names(ur.tracks) == ["RAM_LOCAL"]
