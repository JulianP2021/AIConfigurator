"""Tests for the two-tier distributed cache."""

from src.cache.cache import S3_NODE_ID, Cache, CacheItem
from src.hardware.hardware import Hardware, S3Spec
from src.model.model import Model
from src.request.request import Request
from src.scheduler.global_clock import GlobalClock


def bottleneck_names(tracks: list[list]) -> list[str]:
    """Return bottleneck names flattened from a list of leg tracks."""
    return [leg.bottleneck for track in tracks for leg in track]


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
        cache.insert_cache_item(CacheItem((2, 0), 0, 512), 0)

        dr = cache.download_kv(0, Request(0, 1000, 1, 0))
        assert len(dr.tracks) == 0
        assert dr.request.prefilled_tokens == 0

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
        # s3_tiny_hardware RAM = 65000 * 0.8 = 52000 bytes (fits one 512-token item = 51200 bytes)
        cache.insert_cache_item(CacheItem((1, 0), 0, 512), 0)
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
