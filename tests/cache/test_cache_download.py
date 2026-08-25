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
            "SSD_LOCAL",
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
            ["SSD_LOCAL"],
            ["SSD_LOCAL", "NETWORK", "RAM_LOCAL"],
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
        ssd_item = CacheItem((1, 85), 0, 80)
        ssd_layer._add_item(ssd_item)
        ssd_layer.touch(ssd_item, 1)
        cache_with_fake_model.ssd_usage_bytes[0] += cache_with_fake_model._item_size(
            ssd_item
        )
        # Remote RAM copy.
        cache_with_fake_model.insert_cache_item(CacheItem((1, 85), 0, 100), 1)

        dr = cache_with_fake_model.download_kv(0, req)

        assert bottleneck_names([dr.tracks[dr.eviction_track_count]]) == ["SSD_LOCAL"]
        assert bottleneck_names([dr.tracks[1]]) == ["NETWORK", "RAM_LOCAL"]

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

    def test_download_full_node_ssd_prefix_ram_suffix(
        self, fake_model: Model, tiny_hardware: Hardware
    ):
        """Two filler RAM items are both evicted to SSD by a full-RAM download.

        One instance on one node where both RAM and SSD tiers are full.  The
        session's prefix [0, 140000) lives on SSD and the suffix [140000, 250000)
        lives on RAM; the rest of RAM is occupied by two smaller filler sessions.
        A download must merge the two session items into one contiguous RAM item,
        which evicts both filler RAM items to SSD.
        """
        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
        )
        # RAM fits the merged 250000-token item (25MB) and is full with the
        # suffix (11MB) plus two 70000-token fillers (7MB each).  SSD is full
        # with the prefix.
        cache.ram_capacity_bytes[0] = 25_000_000
        cache.ssd_capacity_bytes[0] = 14_000_000

        cache.insert_cache_item(CacheItem((1, 1), 140000, 250000), 0)
        cache.insert_cache_item(CacheItem((99, 0), 0, 70000), 0)
        cache.insert_cache_item(CacheItem((98, 0), 0, 70000), 0)

        ssd_layer = cache._ssd_layer(0)
        ssd_item = CacheItem((1, 1), 0, 140000)
        ssd_layer._add_item(ssd_item)
        ssd_layer.touch(ssd_item, 1)
        cache.ssd_usage_bytes[0] += cache._item_size(ssd_item)

        req = Request(250000, 8, user_id=1, session_id=1)
        dr = cache.download_kv(0, req)

        assert len(dr.tracks) == 3
        assert dr.eviction_track_count == 1

        # Both filler items must be evicted from RAM to SSD.
        ram_layer = cache._ram_layer(0)
        ssd_layer = cache._ssd_layer(0)
        assert (99, 0) not in ram_layer.content
        assert (98, 0) not in ram_layer.content
        assert (99, 0) in ssd_layer.content
        assert (98, 0) in ssd_layer.content
        assert cache.ssd_usage_bytes[0] == 14_000_000
        # The session itself is one contiguous RAM item.
        session_items = cache.find_cache((1, 1), node_id=0)
        assert len(session_items) == 1
        assert session_items[0].token_start == 0
        assert session_items[0].token_end == 250000
        # The SSD prefix read must count as an SSD download; the RAM suffix as a
        # RAM download.
        assert cache.ssd_download_requests == 1
        assert cache.ram_download_requests == 1
