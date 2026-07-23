"""Tests for the two-tier distributed cache."""

from src.cache.cache import Cache, CacheItem
from src.hardware.hardware import Hardware, S3Spec
from src.model.model import Model


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
