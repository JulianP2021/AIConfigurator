"""Tests for the two-tier distributed cache."""

import pytest

from src.cache.cache import S3_NODE_ID, Cache, CacheItem, CacheLayer
from src.hardware.hardware import Hardware, S3Spec
from src.model.model import Model
from src.request.request import Request
from src.scheduler.global_clock import GlobalClock


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


class TestCacheLayerInternals:
    """Unit tests for CacheLayer helper methods."""

    def test_add_item_sets_back_pointers(self):
        layer = CacheLayer(5, "RAM")
        item = CacheItem((1, 0), 0, 100)
        layer._add_item(item)

        assert item.layer is layer
        assert item.node_id == 5
        assert layer._get_item((1, 0), 0, 100) is item

    def test_remove_item_clears_back_pointers(self):
        layer = CacheLayer(5, "RAM")
        item = CacheItem((1, 0), 0, 100)
        layer._add_item(item)
        layer._remove_item(item)

        assert item.layer is None
        assert item.node_id == -1
        assert (1, 0) not in layer.content

    def test_remove_item_keeps_other_session_items(self):
        layer = CacheLayer(5, "RAM")
        item_a = CacheItem((1, 0), 0, 100)
        item_b = CacheItem((2, 0), 0, 100)
        layer._add_item(item_a)
        layer._add_item(item_b)
        layer._remove_item(item_a)

        assert (2, 0) in layer.content
        assert (1, 0) not in layer.content

    def test_touch_and_pop_lru_ordering(self):
        layer = CacheLayer(0, "RAM")
        item1 = CacheItem((1, 0), 0, 100)
        item2 = CacheItem((2, 0), 0, 100)
        layer._add_item(item1)
        layer._add_item(item2)
        layer.touch(item1, 1)
        layer.touch(item2, 2)

        assert layer.pop_lru() is item1
        assert layer.pop_lru() is item2
        assert layer.pop_lru() is None

    def test_pop_lru_skips_stale_entries_after_remove(self):
        layer = CacheLayer(0, "RAM")
        item = CacheItem((1, 0), 0, 100)
        layer._add_item(item)
        layer.touch(item, 1)
        layer.remove_from_lru(item)
        layer._remove_item(item)

        assert layer.pop_lru() is None

    def test_pop_lru_skips_stale_entries_after_re_touch(self):
        layer = CacheLayer(0, "RAM")
        item = CacheItem((1, 0), 0, 100)
        layer._add_item(item)
        layer.touch(item, 1)
        layer.touch(item, 2)

        # The first heap entry (tick=1) is stale; pop_lru should skip it.
        assert layer.pop_lru() is item
        assert layer.pop_lru() is None

    def test_touch_updates_item_ticks(self):
        layer = CacheLayer(0, "RAM")
        item = CacheItem((1, 0), 0, 100)
        layer._add_item(item)
        layer.touch(item, 42)

        assert item.last_access_tick == 42


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

        with pytest.raises(RuntimeError, match="No S3 legs"):
            cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)

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
        assert bottleneck_names(dr.tracks) == ["RAM_LOCAL"]

    def test_download_from_ram_local(self, cache_with_fake_model: Cache):
        req = Request(128, 8, user_id=1, session_id=10)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 10), 0, 128), 0)

        dr = cache_with_fake_model.download_kv(0, req)
        assert bottleneck_names(dr.tracks) == ["RAM_LOCAL"]

    def test_download_from_ram_local_replaces_existing_item(
        self, cache_with_fake_model: Cache
    ):
        req = Request(128, 8, user_id=1, session_id=10)
        old_item = CacheItem((1, 10), 0, 128)
        cache_with_fake_model.insert_cache_item(old_item, 0)
        original_tick = old_item.last_access_tick

        dr = cache_with_fake_model.download_kv(0, req)
        assert bottleneck_names(dr.tracks) == ["RAM_LOCAL"]

        ram_items = cache_with_fake_model.find_cache((1, 10))
        assert len(ram_items) == 1
        assert ram_items[0].last_access_tick > original_tick
        assert (
            old_item
            not in cache_with_fake_model._ram_layer(0).content[(1, 10)].values()
        )

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
        ssd_layer.content.setdefault((1, 30), {})[(0, 100)] = ssd_item
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
        remote_items = (
            cache_with_fake_model._ram_layer(1).content.get((1, 30), {}).values()
        )
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
        assert [bottleneck_names([track]) for track in dr.tracks] == [
            ["RAM_LOCAL"],
            ["NETWORK", "RAM_LOCAL"],
        ]

        local_items = cache_with_fake_model.find_cache((1, 31), node_id=0)
        assert len(local_items) == 1
        assert local_items[0].token_start == 0
        assert local_items[0].token_end == 200

    def test_download_uses_remote_item_with_shared_start(
        self, fake_model: Model, tiny_hardware: Hardware
    ):
        """Remote items with the same token_start must not collide in the covering index.

        Regression: the remote/S3 covering index used token_start as the
        SortedDict key, so two remote items starting at 0 (e.g. [0, 30001] and
        [0, 62000]) overwrote each other.  The download resolver then stopped
        at the shorter range and failed to fetch the rest of the prefix.

        Items on the same node for the same session are now merged, so we place
        the colliding-start items on different remote nodes to keep the
        regression meaningful.
        """
        cache = Cache(
            layers={},
            node_hardware={
                0: tiny_hardware,
                1: tiny_hardware,
                2: tiny_hardware,
            },
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
        )
        req = Request(94000, 8, user_id=1, session_id=32)
        # Node 0 has a short prefix [0, 100].
        cache.insert_cache_item(CacheItem((1, 32), 0, 100), 0)
        # Nodes 1 and 2 both start at 0 but extend to different ends.  They
        # cannot merge because they are on different nodes, so the resolver's
        # covering index must key on (start, end) to retain both.
        cache.insert_cache_item(CacheItem((1, 32), 0, 200), 1)
        cache.insert_cache_item(CacheItem((1, 32), 0, 300), 2)

        dr = cache.download_kv(0, req)
        # The longest covering item is fetched in a single track because it
        # spans the remaining [100, 300) range.  The local [0, 100) prefix
        # now emits a RAM_LOCAL leg as well.
        assert [bottleneck_names([track]) for track in dr.tracks] == [
            ["RAM_LOCAL"],
            ["NETWORK", "RAM_LOCAL"],
        ]

        local_items = cache.find_cache((1, 32), node_id=0)
        assert len(local_items) == 1
        assert local_items[0].token_start == 0
        assert local_items[0].token_end == 300

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
            for items in small_cache._ram_layer(0).content.values()
            for item in items.values()
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

    def test_download_from_local_ssd(self, cache_with_fake_model: Cache):
        req = Request(128, 8, user_id=1, session_id=80)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 80), 0, 128), 0)
        # Evict to SSD by inserting a different session and shrinking capacity.
        original_capacity = cache_with_fake_model.ram_capacity_bytes[0]
        cache_with_fake_model.ram_capacity_bytes[0] = (
            12_900  # exactly one 128-token item
        )
        cache_with_fake_model.insert_cache_item(CacheItem((2, 0), 0, 128), 0)
        cache_with_fake_model.ram_capacity_bytes[0] = original_capacity

        source = cache_with_fake_model.find_cache_layer(
            cache_with_fake_model.find_cache((1, 80), node_id=0)[-1]
        )
        assert source is not None
        assert source.name == "SSD"

        dr = cache_with_fake_model.download_kv(0, req)
        assert [bottleneck_names([track]) for track in dr.tracks] == [["SSD_LOCAL"]]
        assert req.prefilled_tokens == 128

    def test_download_required_end_less_than_cached_prefix(
        self, cache_with_fake_model: Cache
    ):
        """download_kv merges local cache even when it extends beyond request.isl."""
        req = Request(50, 8, user_id=1, session_id=81)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 81), 0, 200), 0)

        dr = cache_with_fake_model.download_kv(0, req)
        # The local RAM item [0, 200) is connected to the required [0, 50) prefix,
        # so the merged RAM entry extends to 200 even though only 50 were asked.
        assert req.prefilled_tokens == 200
        assert bottleneck_names(dr.tracks) == ["RAM_LOCAL"]

    def test_download_merges_existing_overlapping_local_ram(
        self, cache_with_fake_model: Cache
    ):
        """An existing local RAM item that extends beyond effective_end merges in."""
        req = Request(100, 8, user_id=1, session_id=82)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 82), 0, 100), 0)
        # A remote copy that only covers [0, 100) makes the global prefix end at 100.
        cache_with_fake_model.insert_cache_item(CacheItem((1, 82), 0, 100), 1)

        # After the download the local RAM already holds [0, 100), so no transfer
        # is needed for that portion.
        dr = cache_with_fake_model.download_kv(0, req)
        assert req.prefilled_tokens == 100
        assert [bottleneck_names([track]) for track in dr.tracks] == [["RAM_LOCAL"]]

    def test_download_merges_and_extends_with_overlapping_local_ram(
        self, cache_with_fake_model: Cache
    ):
        """A local RAM item that reaches beyond the downloaded prefix extends it."""
        req = Request(100, 8, user_id=1, session_id=82)
        # Local RAM has the first half; remote RAM has the second half.
        cache_with_fake_model.insert_cache_item(CacheItem((1, 82), 0, 100), 0)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 82), 100, 200), 1)

        dr = cache_with_fake_model.download_kv(0, req)
        # The local [0, 100) item is adjacent to the downloaded [100, 200) remote
        # segment, so the merged RAM entry should extend to 200.
        assert req.prefilled_tokens == 200
        assert [bottleneck_names([track]) for track in dr.tracks] == [
            ["RAM_LOCAL"],
            ["NETWORK", "RAM_LOCAL"],
        ]
        local = cache_with_fake_model.find_cache((1, 82), node_id=0)
        assert len(local) == 1
        assert local[0].token_start == 0
        assert local[0].token_end == 200

    def test_download_empty_when_global_prefix_is_zero(
        self, cache_with_fake_model: Cache
    ):
        req = Request(100, 8, user_id=1, session_id=83)
        dr = cache_with_fake_model.download_kv(0, req)
        assert dr.tracks == []
        assert req.prefilled_tokens == 0

    def test_download_prefers_local_ram_over_remote(self, cache_with_fake_model: Cache):
        req = Request(100, 8, user_id=1, session_id=84)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 84), 0, 100), 0)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 84), 0, 100), 1)

        dr = cache_with_fake_model.download_kv(0, req)
        assert [bottleneck_names([track]) for track in dr.tracks] == [["RAM_LOCAL"]]

    def test_download_prefers_local_ssd_over_remote(self, cache_with_fake_model: Cache):
        req = Request(100, 8, user_id=1, session_id=85)
        # Local SSD copy.
        ssd_layer = cache_with_fake_model._ssd_layer(0)
        ssd_item = CacheItem((1, 85), 0, 100)
        ssd_layer._add_item(ssd_item)
        ssd_layer.touch(ssd_item, 1)
        cache_with_fake_model.ssd_usage_bytes[0] += cache_with_fake_model._item_size(
            ssd_item
        )
        # Remote RAM copy.
        cache_with_fake_model.insert_cache_item(CacheItem((1, 85), 0, 100), 1)

        dr = cache_with_fake_model.download_kv(0, req)
        assert [bottleneck_names([track]) for track in dr.tracks] == [["SSD_LOCAL"]]

    def test_download_merges_local_ram_and_local_ssd(
        self, cache_with_fake_model: Cache
    ):
        req = Request(200, 8, user_id=1, session_id=86)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 86), 0, 100), 0)
        ssd_layer = cache_with_fake_model._ssd_layer(0)
        ssd_item = CacheItem((1, 86), 100, 200)
        ssd_layer._add_item(ssd_item)
        ssd_layer.touch(ssd_item, 1)
        cache_with_fake_model.ssd_usage_bytes[0] += cache_with_fake_model._item_size(
            ssd_item
        )

        dr = cache_with_fake_model.download_kv(0, req)
        assert [bottleneck_names([track]) for track in dr.tracks] == [
            ["RAM_LOCAL"],
            ["SSD_LOCAL"],
        ]
        assert req.prefilled_tokens == 200
        local = cache_with_fake_model.find_cache((1, 86), node_id=0)
        assert len(local) == 1
        assert local[0].token_start == 0
        assert local[0].token_end == 200

    def test_download_prefers_remote_ram_over_s3(
        self, fake_model: Model, tiny_hardware: Hardware, s3_enabled: S3Spec
    ):
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware, 1: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )
        req = Request(100, 8, user_id=1, session_id=87)
        cache.insert_cache_item(CacheItem((1, 87), 0, 100), 1)
        s3_layer = cache._s3_layer()
        s3_item = CacheItem((1, 87), 0, 100)
        s3_layer._add_item(s3_item)
        cache.s3_usage_bytes += cache._item_size(s3_item)

        dr = cache.download_kv(0, req)
        assert [bottleneck_names([track]) for track in dr.tracks] == [
            ["NETWORK", "RAM_LOCAL"]
        ]

    def test_download_s3_only_source(
        self, fake_model: Model, tiny_hardware: Hardware, s3_enabled: S3Spec
    ):
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )
        req = Request(100, 8, user_id=1, session_id=88)
        s3_layer = cache._s3_layer()
        s3_item = CacheItem((1, 88), 0, 100)
        s3_layer._add_item(s3_item)
        cache.s3_usage_bytes += cache._item_size(s3_item)

        dr = cache.download_kv(0, req)
        assert [bottleneck_names([track]) for track in dr.tracks] == [["S3_DOWNLOAD"]]
        assert cache.s3_download_requests == 1

    def test_download_with_gap_returns_partial_prefix(
        self, fake_model: Model, tiny_hardware: Hardware
    ):
        """A gap in the global prefix caps effective_end before the gap."""
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware, 1: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
        )
        req = Request(200, 8, user_id=1, session_id=89)
        cache.insert_cache_item(CacheItem((1, 89), 0, 100), 0)
        cache.insert_cache_item(CacheItem((1, 89), 150, 200), 1)

        dr = cache.download_kv(0, req)
        assert req.prefilled_tokens == 100
        assert bottleneck_names(dr.tracks) == ["RAM_LOCAL"]

    def test_download_local_ssd_promoted_and_deleted_from_ssd(
        self, cache_with_fake_model: Cache
    ):
        req = Request(100, 8, user_id=1, session_id=90)
        ssd_layer = cache_with_fake_model._ssd_layer(0)
        ssd_item = CacheItem((1, 90), 0, 100)
        ssd_layer._add_item(ssd_item)
        ssd_layer.touch(ssd_item, 1)
        cache_with_fake_model.ssd_usage_bytes[0] += cache_with_fake_model._item_size(
            ssd_item
        )

        dr = cache_with_fake_model.download_kv(0, req)
        assert [bottleneck_names([track]) for track in dr.tracks] == [["SSD_LOCAL"]]
        assert cache_with_fake_model.ssd_usage_bytes[0] == 0
        assert (1, 90) not in ssd_layer.content


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
            item
            for items in cache._s3_layer().content.values()
            for item in items.values()
            if item.session_id == (1, 40)
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
        s3_layer.content.setdefault((1, 42), {})[(0, 512)] = CacheItem((1, 42), 0, 512)

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

    def test_s3_disabled_no_upload_when_ram_evicted(
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
        with pytest.raises(RuntimeError, match="No S3 legs"):
            cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)

    def test_s3_upload_skipped_when_covered(
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
        # Seed S3 with [0, 512).
        s3_layer = cache._s3_layer()
        s3_item = CacheItem((1, 0), 0, 512)
        s3_layer._add_item(s3_item)
        cache.s3_usage_bytes = cache._item_size(s3_item)
        cache.s3_peak_usage_bytes = cache.s3_usage_bytes

        # Put the same range in RAM and evict it; upload_to_s3 should skip.
        cache.insert_cache_item(CacheItem((1, 0), 0, 512), 0)
        cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)

        assert cache.s3_upload_requests == 0

    def test_s3_disabled_download_no_s3_layer(self, cache_with_fake_model: Cache):
        cache = Cache(
            layers={},
            node_hardware=cache_with_fake_model.node_hardware,
            model=cache_with_fake_model.model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=S3Spec.from_gbps(enabled=False),
        )
        req = Request(100, 8, user_id=1, session_id=91)
        dr = cache.download_kv(0, req)
        assert dr.tracks == []

    def test_s3_upload_records_cost_and_requests(
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
        cache.insert_cache_item(CacheItem((1, 0), 0, 8192), 0)
        cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)
        cache.insert_cache_item(CacheItem((3, 0), 0, 512), 0)

        assert cache.s3_upload_requests > 0
        assert cache.cost_usd > 0.0

    def test_s3_download_records_cost_and_requests(
        self, fake_model: Model, tiny_hardware: Hardware, s3_enabled: S3Spec
    ):
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )
        s3_layer = cache._s3_layer()
        s3_item = CacheItem((1, 0), 0, 1_000_000)
        s3_layer._add_item(s3_item)
        cache.s3_usage_bytes += cache._item_size(s3_item)

        req = Request(1_000_000, 8, user_id=1, session_id=0)
        cache.download_kv(0, req)

        assert cache.s3_download_requests == 1
        # Cost per byte is small for 100 bytes/token; use a large enough item.
        assert cache.cost_usd > 0.0

    def test_s3_evicts_stale_items_by_age(
        self, fake_model: Model, s3_tiny_hardware: Hardware
    ):
        """S3 items older than eviction_time_ms are removed after new uploads."""
        clock = GlobalClock()
        eviction_window_ms = 1000.0
        s3_spec = S3Spec.from_gbps(
            enabled=True,
            eviction_time_ms=eviction_window_ms,
        )
        cache = Cache(
            layers={},
            node_hardware={0: s3_tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_spec,
            clock=clock,
        )

        # Force an S3 upload by filling RAM, then SSD, then evicting to S3.
        old_item = CacheItem((1, 40), 0, 512)
        cache.insert_cache_item(old_item, 0)
        cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)
        cache.insert_cache_item(CacheItem((3, 0), 0, 512), 0)
        # old_item is now in S3, recorded at t=0.
        assert cache.s3_usage_bytes > 0
        assert cache._s3_layer().content[(1, 40)]
        old_size = cache.s3_usage_bytes

        # Advance the clock past the eviction window.
        clock.advance(eviction_window_ms + 1.0)

        # Trigger another SSD->S3 upload; this should run stale-object eviction.
        cache.insert_cache_item(CacheItem((4, 0), 0, 512), 0)

        # The old item was uploaded at t=0 and is now stale; the newly uploaded
        # item was uploaded after the clock advance and is still fresh.
        assert (1, 40) not in cache._s3_layer().content
        assert cache.s3_usage_bytes == old_size

    def test_s3_upload_records_wall_clock_access_time(
        self, fake_model: Model, s3_tiny_hardware: Hardware
    ):
        """Uploading to S3 must stamp last_access_ms with the current clock."""
        clock = GlobalClock()
        s3_spec = S3Spec.from_gbps(enabled=True, eviction_time_ms=1000.0)
        cache = Cache(
            layers={},
            node_hardware={0: s3_tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_spec,
            clock=clock,
        )

        cache.insert_cache_item(CacheItem((1, 40), 0, 512), 0)
        cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)
        cache.insert_cache_item(CacheItem((3, 0), 0, 512), 0)

        s3_item = next(iter(cache._s3_layer().content[(1, 40)].values()))
        # The upload happened at t=0, so the item should be fresh then.
        assert s3_item.last_access_ms == clock.time_ms

        clock.advance(500.0)
        cache.insert_cache_item(CacheItem((4, 0), 0, 512), 0)
        # The original item was uploaded at t=0; at t=500 it is still within
        # the 1000 ms window, so it must survive.
        assert (1, 40) in cache._s3_layer().content

    def test_s3_download_refreshes_access_time(
        self, fake_model: Model, s3_tiny_hardware: Hardware
    ):
        """Downloading an S3 item refreshes its age so it is not evicted."""
        clock = GlobalClock()
        eviction_window_ms = 1000.0
        s3_spec = S3Spec.from_gbps(enabled=True, eviction_time_ms=eviction_window_ms)
        cache = Cache(
            layers={},
            node_hardware={0: s3_tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_spec,
            clock=clock,
        )

        # Put an item in S3.
        cache.insert_cache_item(CacheItem((1, 40), 0, 512), 0)
        cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)
        cache.insert_cache_item(CacheItem((3, 0), 0, 512), 0)

        # Advance past the original upload time, but download right before the
        # eviction cutoff to refresh the item's age.
        clock.advance(eviction_window_ms - 1.0)
        req = Request(512, 8, user_id=1, session_id=40)
        cache.download_kv(0, req)

        s3_item = next(iter(cache._s3_layer().content[(1, 40)].values()))
        assert s3_item.last_access_ms == clock.time_ms

        # Advance further; the refreshed item should still be alive.
        clock.advance(eviction_window_ms * 0.9)
        cache.insert_cache_item(CacheItem((4, 0), 0, 512), 0)
        assert (1, 40) in cache._s3_layer().content

    def test_s3_peak_usage_tracks_maximum(
        self, fake_model: Model, s3_tiny_hardware: Hardware
    ):
        """s3_peak_usage_bytes records the highest S3 usage ever observed."""
        s3_spec = S3Spec.from_gbps(enabled=True)
        cache = Cache(
            layers={},
            node_hardware={0: s3_tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_spec,
        )

        # Fill RAM, then SSD, then push the oldest item to S3.
        cache.insert_cache_item(CacheItem((1, 40), 0, 512), 0)
        cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)
        cache.insert_cache_item(CacheItem((3, 0), 0, 512), 0)

        item_size = cache._item_size(CacheItem((1, 40), 0, 512))
        assert cache.s3_usage_bytes == item_size
        assert cache.s3_peak_usage_bytes == item_size
        assert cache.usage_summary()["s3_peak_usage_bytes"] == item_size

        # Each further insert evicts another SSD item to S3; peak tracks usage.
        cache.insert_cache_item(CacheItem((4, 0), 0, 512), 0)
        assert cache.s3_usage_bytes == 2 * item_size
        assert cache.s3_peak_usage_bytes == 2 * item_size

        cache.insert_cache_item(CacheItem((5, 0), 0, 512), 0)
        assert cache.s3_usage_bytes == 3 * item_size
        assert cache.s3_peak_usage_bytes == 3 * item_size

    def test_s3_spec_has_storage_cost_constant(self):
        """S3Spec exposes the hard-coded storage cost per GB per month."""
        s3_spec = S3Spec.from_gbps(enabled=True)
        assert s3_spec.S3_STORAGE_COST_GB_PER_MONTH == 0.022

    def test_s3_eviction_keeps_recently_accessed_items(
        self, fake_model: Model, s3_tiny_hardware: Hardware
    ):
        """S3 items touched within the eviction window are not removed."""
        clock = GlobalClock()
        eviction_window_ms = 1000.0
        s3_spec = S3Spec.from_gbps(
            enabled=True,
            eviction_time_ms=eviction_window_ms,
        )
        cache = Cache(
            layers={},
            node_hardware={0: s3_tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_spec,
            clock=clock,
        )

        old_item = CacheItem((1, 40), 0, 512)
        cache.insert_cache_item(old_item, 0)
        cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)
        cache.insert_cache_item(CacheItem((3, 0), 0, 512), 0)

        # Advance well into the eviction window, then re-touch the S3 copy
        # directly. This resets its age.
        clock.advance(eviction_window_ms * 1.5)
        s3_item = next(iter(cache._s3_layer().content[(1, 40)].values()))
        cache._touch(s3_item, cache._s3_layer())

        # Advance past the original upload time, but less than the eviction
        # window since the recent touch. The item should survive.
        clock.advance(eviction_window_ms * 0.6)
        cache.insert_cache_item(CacheItem((4, 0), 0, 512), 0)

        assert (1, 40) in cache._s3_layer().content
        assert cache.s3_usage_bytes > 0

    def test_s3_no_eviction_when_window_is_zero(
        self, fake_model: Model, s3_tiny_hardware: Hardware
    ):
        clock = GlobalClock()
        s3_spec = S3Spec.from_gbps(enabled=True, eviction_time_ms=0.0)
        cache = Cache(
            layers={},
            node_hardware={0: s3_tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_spec,
            clock=clock,
        )
        cache.insert_cache_item(CacheItem((1, 40), 0, 512), 0)
        cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)
        cache.insert_cache_item(CacheItem((3, 0), 0, 512), 0)

        clock.advance(10_000.0)
        cache.insert_cache_item(CacheItem((4, 0), 0, 512), 0)

        assert (1, 40) in cache._s3_layer().content

    def test_s3_enabled_layer_is_created(
        self, fake_model: Model, tiny_hardware: Hardware, s3_enabled: S3Spec
    ):
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )
        assert S3_NODE_ID in cache.layers
        assert any(layer.name == "S3" for layer in cache.layers[S3_NODE_ID])

    def test_s3_disabled_layer_is_not_created(
        self, fake_model: Model, tiny_hardware: Hardware
    ):
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=S3Spec.from_gbps(enabled=False),
        )
        assert S3_NODE_ID not in cache.layers


class TestCacheMerge:
    def test_ram_inserts_merge_adjacent_session_items(
        self, cache_with_fake_model: Cache
    ):
        """Two adjacent RAM items for the same session are merged into one."""
        cache_with_fake_model.insert_cache_item(CacheItem((1, 50), 0, 100), 0)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 50), 100, 200), 0)

        items = list(cache_with_fake_model._ram_layer(0).content[(1, 50)].values())
        assert len(items) == 1
        assert items[0].token_start == 0
        assert items[0].token_end == 200

    def test_ram_inserts_merge_overlapping_session_items(
        self, cache_with_fake_model: Cache
    ):
        """Two overlapping RAM items for the same session are merged into one."""
        cache_with_fake_model.insert_cache_item(CacheItem((1, 51), 0, 150), 0)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 51), 100, 200), 0)

        items = list(cache_with_fake_model._ram_layer(0).content[(1, 51)].values())
        assert len(items) == 1
        assert items[0].token_start == 0
        assert items[0].token_end == 200

    def test_ram_insert_does_not_merge_disjoint_session_items(
        self, cache_with_fake_model: Cache
    ):
        """Disjoint RAM items for the same session remain separate."""
        cache_with_fake_model.insert_cache_item(CacheItem((1, 52), 0, 100), 0)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 52), 200, 300), 0)

        items = list(cache_with_fake_model._ram_layer(0).content[(1, 52)].values())
        assert len(items) == 2

    def test_ram_insert_merges_different_sessions_separately(
        self, cache_with_fake_model: Cache
    ):
        """Items for different sessions are merged independently."""
        cache_with_fake_model.insert_cache_item(CacheItem((1, 60), 0, 100), 0)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 60), 100, 200), 0)
        cache_with_fake_model.insert_cache_item(CacheItem((2, 60), 0, 100), 0)
        cache_with_fake_model.insert_cache_item(CacheItem((2, 60), 100, 200), 0)

        assert len(cache_with_fake_model._ram_layer(0).content[(1, 60)]) == 1
        assert len(cache_with_fake_model._ram_layer(0).content[(2, 60)]) == 1

    def test_s3_eviction_merges_with_existing_s3_item(
        self, fake_model: Model, s3_tiny_hardware: Hardware, s3_enabled: S3Spec
    ):
        """SSD victims uploaded to S3 merge with existing S3 items for the session."""
        cache = Cache(
            layers={},
            node_hardware={0: s3_tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )
        # Fill RAM/SSD and push [0, 512) to S3.
        cache.insert_cache_item(CacheItem((1, 70), 0, 512), 0)
        cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)
        cache.insert_cache_item(CacheItem((3, 0), 0, 512), 0)

        s3_items = list(cache._s3_layer().content[(1, 70)].values())
        assert len(s3_items) == 1
        assert s3_items[0].token_start == 0
        assert s3_items[0].token_end == 512

        # Now put an overlapping [256, 768) in RAM and evict it to S3.
        cache.insert_cache_item(CacheItem((1, 70), 256, 768), 0)
        cache.insert_cache_item(CacheItem((4, 0), 0, 512), 0)
        cache.insert_cache_item(CacheItem((5, 0), 0, 512), 0)

        s3_items = list(cache._s3_layer().content[(1, 70)].values())
        assert len(s3_items) == 1
        assert s3_items[0].token_start == 0
        assert s3_items[0].token_end == 768

    def test_s3_eviction_skips_upload_when_existing_s3_item_covers_victim(
        self, fake_model: Model, s3_tiny_hardware: Hardware, s3_enabled: S3Spec
    ):
        """No upload is counted when the victim range is already covered by S3."""
        cache = Cache(
            layers={},
            node_hardware={0: s3_tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )
        # Seed S3 with [0, 512) for session (1, 71).
        s3_item = CacheItem((1, 71), 0, 512)
        s3_layer = cache._s3_layer()
        s3_layer._add_item(s3_item)
        cache.s3_usage_bytes = cache._item_size(s3_item)
        cache.s3_peak_usage_bytes = cache.s3_usage_bytes

        # Put an SSD victim [0, 256) for the same session; S3 already covers it.
        ssd_item = CacheItem((1, 71), 0, 256)
        ssd_layer = cache._ssd_layer(0)
        ssd_layer._add_item(ssd_item)
        ssd_layer.touch(ssd_item, 1)
        cache.ssd_usage_bytes[0] = cache._item_size(ssd_item)

        uploads_before = cache.s3_upload_requests
        s3_leg = cache._evict_ssd_lru(0)

        assert s3_leg is None
        assert cache.s3_upload_requests == uploads_before
        assert (1, 71) in s3_layer.content
        assert (1, 71) not in ssd_layer.content

    def test_merge_with_layer_items_absorbs_connected_component(
        self, cache_with_fake_model: Cache
    ):
        """_merge_with_layer_items should expand to absorb touching/overlapping items."""
        layer = cache_with_fake_model._ram_layer(0)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 0), 0, 100), 0)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 0), 200, 300), 0)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 0), 400, 500), 0)

        new_item = CacheItem((1, 0), 90, 210)
        cache_with_fake_model._merge_with_layer_items(layer, new_item)

        # [0,100] and [200,300] are connected via [90,210]; [400,500] is disjoint.
        assert new_item.token_start == 0
        assert new_item.token_end == 300
        # The merged items are removed and the new item is not yet inserted.
        assert len(layer.content[(1, 0)]) == 1
        remaining_item = next(iter(layer.content[(1, 0)].values()))
        assert remaining_item.token_start == 400
        assert remaining_item.token_end == 500

    def test_merge_with_layer_items_no_merge_when_disjoint(
        self, cache_with_fake_model: Cache
    ):
        layer = cache_with_fake_model._ram_layer(0)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 0), 0, 100), 0)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 0), 200, 300), 0)

        new_item = CacheItem((1, 0), 400, 500)
        result = cache_with_fake_model._merge_with_layer_items(layer, new_item)

        assert result is new_item
        assert new_item.token_start == 400
        assert new_item.token_end == 500


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

    def test_upload_zero_bytes_raises(self, fake_model: Model, tiny_hardware: Hardware):
        # Shrink RAM to exactly hold one 64-token item so the second insert
        # evicts the prior copy and creates a zero-increment update.
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
        )
        cache.ram_capacity_bytes[0] = 6_400
        cache.ssd_capacity_bytes[0] = 6_400

        user_id, session_id = 1, 22
        req = Request(64, 8, user_id, session_id)
        cache.insert_cache_item(CacheItem((user_id, session_id), 0, 64), 0)
        req.prefilled_tokens = 64
        req.decoded_tokens = 0

        with pytest.raises(ValueError, match="Zero-byte upload"):
            cache.upload_kv(0, req)

    def test_upload_includes_eviction_track(self, small_cache: Cache):
        req = Request(1024, 8, 0, 1)
        req.id = 23
        req.prefilled_tokens = 1024
        req.decoded_tokens = 0

        # Force eviction by inserting another item first. The eviction legs are
        # created when RAM overflows, so the upload tracks include them in
        # addition to the actual upload leg.
        small_cache.insert_cache_item(CacheItem((9, 9), 0, 512), 0)

        ur = small_cache.upload_kv(0, req)
        assert len(ur.tracks) == 2
        assert ur.is_upload_done() is False

    def test_upload_to_node_without_prior_cache(self, cache_with_fake_model: Cache):
        req = Request(128, 8, user_id=1, session_id=100)
        req.prefilled_tokens = 128
        req.decoded_tokens = 0

        ur = cache_with_fake_model.upload_kv(1, req)
        assert bottleneck_names(ur.tracks) == ["RAM_LOCAL"]

    def test_upload_incremental_bytes(self, cache_with_fake_model: Cache):
        req = Request(128, 8, user_id=1, session_id=101)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 101), 0, 64), 0)
        req.prefilled_tokens = 128
        req.decoded_tokens = 0

        ur = cache_with_fake_model.upload_kv(0, req)
        assert ur.tracks[-1][0].remaining_bytes == cache_with_fake_model.kv_size(
            cache_with_fake_model.model, 64
        )

    def test_upload_merges_with_existing_and_updates_range(
        self, cache_with_fake_model: Cache
    ):
        req = Request(128, 8, user_id=1, session_id=102)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 102), 0, 64), 0)
        req.prefilled_tokens = 128
        req.decoded_tokens = 0

        ur = cache_with_fake_model.upload_kv(0, req)
        assert bottleneck_names(ur.tracks) == ["RAM_LOCAL"]

        ram = cache_with_fake_model._ram_layer(0)
        item = next(iter(ram.content[(1, 102)].values()))
        assert item.token_start == 0
        assert item.token_end == 128

    def test_upload_request_id_preserved(self, cache_with_fake_model: Cache):
        req = Request(64, 8, user_id=1, session_id=103)
        req.prefilled_tokens = 64
        req.decoded_tokens = 0
        original_id = req.id

        ur = cache_with_fake_model.upload_kv(0, req)
        assert ur.request.id == original_id


class TestFindCacheAndPrefix:
    def test_find_cache_all_nodes_returns_contiguous_prefix(
        self, cache_with_fake_model: Cache
    ):
        cache_with_fake_model.insert_cache_item(CacheItem((1, 0), 0, 100), 0)
        cache_with_fake_model.insert_cache_item(CacheItem((1, 0), 100, 200), 1)

        prefix = cache_with_fake_model.find_cache((1, 0))
        assert len(prefix) == 2
        assert prefix[-1].token_end == 200

    def test_find_cache_on_node_merges_ram_and_ssd(self, cache_with_fake_model: Cache):
        cache_with_fake_model.insert_cache_item(CacheItem((1, 0), 0, 100), 0)
        ssd_layer = cache_with_fake_model._ssd_layer(0)
        ssd_item = CacheItem((1, 0), 100, 200)
        ssd_layer._add_item(ssd_item)
        ssd_layer.touch(ssd_item, 1)
        cache_with_fake_model.ssd_usage_bytes[0] += cache_with_fake_model._item_size(
            ssd_item
        )

        prefix = cache_with_fake_model.find_cache((1, 0), node_id=0)
        assert len(prefix) == 2
        assert prefix[-1].token_end == 200

    def test_find_cache_on_node_with_overlapping_ram_ssd_prefers_ram(
        self, cache_with_fake_model: Cache
    ):
        """When RAM and SSD overlap, the merged prefix should be contiguous."""
        cache_with_fake_model.insert_cache_item(CacheItem((1, 0), 0, 150), 0)
        ssd_layer = cache_with_fake_model._ssd_layer(0)
        ssd_item = CacheItem((1, 0), 100, 200)
        ssd_layer._add_item(ssd_item)
        ssd_layer.touch(ssd_item, 1)
        cache_with_fake_model.ssd_usage_bytes[0] += cache_with_fake_model._item_size(
            ssd_item
        )

        prefix = cache_with_fake_model.find_cache((1, 0), node_id=0)
        assert prefix[-1].token_end == 200

    def test_cached_prefix_on_node_zero_when_empty(self, cache_with_fake_model: Cache):
        assert cache_with_fake_model.cached_prefix_on_node((1, 0), 0) == 0

    def test_cached_prefix_on_node_returns_prefix_end(
        self, cache_with_fake_model: Cache
    ):
        cache_with_fake_model.insert_cache_item(CacheItem((1, 0), 0, 123), 0)
        assert cache_with_fake_model.cached_prefix_on_node((1, 0), 0) == 123

    def test_find_cache_layer_back_pointer(self, cache_with_fake_model: Cache):
        item = CacheItem((1, 0), 0, 100)
        cache_with_fake_model.insert_cache_item(item, 0)
        assert cache_with_fake_model.find_cache_layer(item).name == "RAM"

    def test_find_cache_layer_fallback_scan(self, cache_with_fake_model: Cache):
        item = CacheItem((1, 0), 0, 100)
        layer = cache_with_fake_model._ram_layer(0)
        layer._add_item(item)
        item.layer = None

        found = cache_with_fake_model.find_cache_layer(item)
        assert found is layer

    def test_find_cache_layer_returns_none_for_orphan_item(
        self, cache_with_fake_model: Cache
    ):
        item = CacheItem((1, 0), 0, 100)
        assert cache_with_fake_model.find_cache_layer(item) is None


class TestFindDownloadSegments:
    """Tests for Cache._find_download_segments gap resolution."""

    def _make_cache(self, fake_model: Model, tiny_hardware: Hardware) -> Cache:
        return Cache(
            layers={},
            node_hardware={
                0: tiny_hardware,
                1: tiny_hardware,
                2: tiny_hardware,
                7: tiny_hardware,
                11: tiny_hardware,
                12: tiny_hardware,
                13: tiny_hardware,
            },
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
        )

    def _add_item(
        self,
        cache: Cache,
        session: tuple[int, int],
        node_id: int,
        layer_name: str,
        start: int,
        end: int,
    ) -> CacheItem:
        item = CacheItem(session, start, end)
        layer = cache.get_layer(node_id, layer_name)
        layer._add_item(item)
        layer.touch(item, cache._access_tick)
        if layer_name == "RAM":
            cache.ram_usage_bytes[node_id] += cache._item_size(item)
        elif layer_name == "SSD":
            cache.ssd_usage_bytes[node_id] += cache._item_size(item)
        return item

    def test_prefers_longest_remote_covering_item(
        self, fake_model: Model, tiny_hardware: Hardware
    ):
        """Reproducer: at a given position, pick the remote item that extends furthest."""
        cache = self._make_cache(fake_model, tiny_hardware)
        session = (23, 0)
        # Local to dest=2
        self._add_item(cache, session, 2, "SSD", 0, 160_000)
        # Remote items
        self._add_item(cache, session, 1, "RAM", 160_000, 190_001)
        self._add_item(cache, session, 12, "SSD", 0, 192_000)
        self._add_item(cache, session, 12, "RAM", 192_000, 222_001)

        effective_end, segments = cache._find_download_segments(session, 2, 222_000)

        # The destination already holds a prefix starting at 0, so the download
        # extends to the end of the globally contiguous cached prefix.
        assert effective_end == 222_001
        assert segments == [
            (0, 160_000, 2, "SSD"),
            (160_000, 192_000, 12, "SSD"),
            (192_000, 222_001, 12, "RAM"),
        ]

    def test_multiple_candidates_same_start(
        self, fake_model: Model, tiny_hardware: Hardware
    ):
        """If several items start at or before pos, the longest end wins."""
        cache = self._make_cache(fake_model, tiny_hardware)
        session = (1, 0)
        # Local to dest=11
        self._add_item(cache, session, 11, "RAM", 0, 64_000)
        # Remote items
        self._add_item(cache, session, 7, "RAM", 0, 94_001)
        self._add_item(cache, session, 12, "SSD", 0, 30_000)
        self._add_item(cache, session, 13, "SSD", 0, 30_001)
        self._add_item(cache, session, 12, "RAM", 30_000, 32_000)

        effective_end, segments = cache._find_download_segments(session, 11, 94_000)

        # The destination already holds a prefix starting at 0, so the download
        # extends to the end of the globally contiguous cached prefix.
        assert effective_end == 94_001
        assert segments == [
            (0, 64_000, 11, "RAM"),
            (64_000, 94_001, 7, "RAM"),
        ]

    def test_local_prefix_limits_when_remote_does_not_extend(
        self, fake_model: Model, tiny_hardware: Hardware
    ):
        cache = self._make_cache(fake_model, tiny_hardware)
        session = (1, 0)
        self._add_item(cache, session, 0, "RAM", 0, 100)
        self._add_item(cache, session, 1, "RAM", 0, 50)

        effective_end, segments = cache._find_download_segments(session, 0, 200)

        assert effective_end == 100
        assert segments == [(0, 100, 0, "RAM")]

    def test_s3_fills_gap_after_remote(
        self, fake_model: Model, tiny_hardware: Hardware, s3_enabled: S3Spec
    ):
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware, 1: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )
        session = (1, 0)
        self._add_item(cache, session, 0, "SSD", 0, 100)
        self._add_item(cache, session, 1, "RAM", 100, 150)
        self._add_item(cache, session, -1, "S3", 150, 200)

        effective_end, segments = cache._find_download_segments(session, 0, 200)

        assert effective_end == 200
        assert segments == [
            (0, 100, 0, "SSD"),
            (100, 150, 1, "RAM"),
            (150, 200, -1, "S3"),
        ]

    def test_no_remote_covering_item_returns_local_only(
        self, fake_model: Model, tiny_hardware: Hardware
    ):
        cache = self._make_cache(fake_model, tiny_hardware)
        session = (1, 0)
        self._add_item(cache, session, 0, "RAM", 0, 100)
        self._add_item(cache, session, 1, "RAM", 200, 300)

        effective_end, segments = cache._find_download_segments(session, 0, 300)

        assert effective_end == 100
        assert segments == [(0, 100, 0, "RAM")]

    def test_remote_ram_used_when_no_local_prefix(
        self, fake_model: Model, tiny_hardware: Hardware, s3_enabled: S3Spec
    ):
        """Regression: destination with no local prefix must use remote RAM.

        A prior bug returned only an S3 segment [0, 192000) even though remote
        node 3 RAM contained [0, 192000) and remote node 2 RAM contained the
        tail [192000, 222001), which together cover the full required prefix.
        """
        cache = Cache(
            layers={},
            node_hardware={
                0: tiny_hardware,
                2: tiny_hardware,
                3: tiny_hardware,
            },
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )
        session = (202, 0)
        # Destination node 0 has no items for this session.
        # Remote RAM on node 3 covers the head.
        layer = cache.get_layer(3, "RAM")
        item = CacheItem(session, 0, 192_000)
        layer._add_item(item)
        layer.touch(item, 1)
        cache.ram_usage_bytes[3] += cache._item_size(item)
        # Remote RAM on node 2 covers the tail.
        layer = cache.get_layer(2, "RAM")
        item = CacheItem(session, 192_000, 222_001)
        layer._add_item(item)
        layer.touch(item, 1)
        cache.ram_usage_bytes[2] += cache._item_size(item)
        # S3 also holds [0, 192000); it must not shadow the remote RAM.
        layer = cache.get_layer(-1, "S3")
        item = CacheItem(session, 0, 192_000)
        layer._add_item(item)
        layer.touch(item, 1)
        cache.s3_usage_bytes += cache._item_size(item)

        effective_end, segments = cache._find_download_segments(session, 0, 222_000)

        assert effective_end == 222_000
        assert segments == [
            (0, 192_000, 3, "RAM"),
            (192_000, 222_000, 2, "RAM"),
        ]

    def test_remote_ram_extends_beyond_s3_and_local_ssd(
        self, fake_model: Model, tiny_hardware: Hardware, s3_enabled: S3Spec
    ):
        """Regression: remote RAM must win over S3 when it covers the gap further.

        A prior bug chose S3 [0, 192000] over remote node RAM [0, 192000] and
        stopped the prefix at 192000 instead of continuing to 222000 via another
        remote RAM item.
        """
        cache = Cache(
            layers={},
            node_hardware={
                4: tiny_hardware,
                5: tiny_hardware,
                7: tiny_hardware,
            },
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )
        session = (2602, 0)
        # Local SSD on destination node 5
        layer = cache.get_layer(5, "SSD")
        item = CacheItem(session, 0, 190_001)
        layer._add_item(item)
        layer.touch(item, 1)
        cache.ssd_usage_bytes[5] += cache._item_size(item)
        # Remote RAM on node 7 covers the next gap
        layer = cache.get_layer(7, "RAM")
        item = CacheItem(session, 0, 192_000)
        layer._add_item(item)
        layer.touch(item, 1)
        cache.ram_usage_bytes[7] += cache._item_size(item)
        # Remote RAM on node 4 covers the tail
        layer = cache.get_layer(4, "RAM")
        item = CacheItem(session, 192_000, 222_001)
        layer._add_item(item)
        layer.touch(item, 1)
        cache.ram_usage_bytes[4] += cache._item_size(item)
        # S3 also has a copy of [0, 192000]; remote RAM should still be preferred
        layer = cache.get_layer(-1, "S3")
        item = CacheItem(session, 0, 192_000)
        layer._add_item(item)
        layer.touch(item, 1)
        cache.s3_usage_bytes += cache._item_size(item)

        effective_end, segments = cache._find_download_segments(session, 5, 222_000)

        assert effective_end == 222_000
        assert segments == [
            (0, 190_001, 5, "SSD"),
            (190_001, 192_000, 7, "RAM"),
            (192_000, 222_000, 4, "RAM"),
        ]

    def test_find_download_segments_local_ram_only(
        self, fake_model: Model, tiny_hardware: Hardware
    ):
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware, 1: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
        )
        session = (1, 0)
        layer = cache.get_layer(0, "RAM")
        item = CacheItem(session, 0, 100)
        layer._add_item(item)
        layer.touch(item, 1)
        cache.ram_usage_bytes[0] += cache._item_size(item)

        effective_end, segments = cache._find_download_segments(session, 0, 100)
        assert effective_end == 100
        assert segments == [(0, 100, 0, "RAM")]

    def test_find_download_segments_s3_only(
        self, fake_model: Model, tiny_hardware: Hardware, s3_enabled: S3Spec
    ):
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )
        session = (1, 0)
        layer = cache.get_layer(S3_NODE_ID, "S3")
        item = CacheItem(session, 0, 100)
        layer._add_item(item)
        layer.touch(item, 1)
        cache.s3_usage_bytes += cache._item_size(item)

        effective_end, segments = cache._find_download_segments(session, 0, 100)
        assert effective_end == 100
        assert segments == [(0, 100, S3_NODE_ID, "S3")]

    def test_find_download_segments_required_end_capped_by_global_prefix(
        self, fake_model: Model, tiny_hardware: Hardware
    ):
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware, 1: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
        )
        session = (1, 0)
        layer = cache.get_layer(0, "RAM")
        item = CacheItem(session, 0, 100)
        layer._add_item(item)
        layer.touch(item, 1)
        cache.ram_usage_bytes[0] += cache._item_size(item)

        effective_end, segments = cache._find_download_segments(session, 0, 500)
        assert effective_end == 100
        assert segments == [(0, 100, 0, "RAM")]

    def test_find_download_segments_remote_item_with_exact_same_range(
        self, fake_model: Model, tiny_hardware: Hardware
    ):
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware, 1: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
        )
        session = (1, 0)
        self._add_item(cache, session, 0, "RAM", 0, 100)
        self._add_item(cache, session, 1, "RAM", 0, 100)

        effective_end, segments = cache._find_download_segments(session, 0, 100)
        assert effective_end == 100
        # Local RAM wins over identical remote copy.
        assert segments == [(0, 100, 0, "RAM")]

    def test_find_download_segments_stops_at_gap(
        self, fake_model: Model, tiny_hardware: Hardware
    ):
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware, 1: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
        )
        session = (1, 0)
        self._add_item(cache, session, 0, "RAM", 0, 100)
        self._add_item(cache, session, 1, "RAM", 150, 200)

        effective_end, segments = cache._find_download_segments(session, 0, 200)
        assert effective_end == 100
        assert segments == [(0, 100, 0, "RAM")]

    def test_find_download_segments_all_layers_merged(
        self, fake_model: Model, tiny_hardware: Hardware, s3_enabled: S3Spec
    ):
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware, 1: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )
        session = (1, 0)
        # Local SSD [0, 50)
        self._add_item(cache, session, 0, "SSD", 0, 50)
        # Remote RAM [50, 100)
        self._add_item(cache, session, 1, "RAM", 50, 100)
        # S3 [100, 150)
        self._add_item(cache, session, S3_NODE_ID, "S3", 100, 150)

        effective_end, segments = cache._find_download_segments(session, 0, 150)
        assert effective_end == 150
        assert segments == [
            (0, 50, 0, "SSD"),
            (50, 100, 1, "RAM"),
            (100, 150, S3_NODE_ID, "S3"),
        ]
