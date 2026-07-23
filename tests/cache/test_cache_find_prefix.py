"""Tests for the two-tier distributed cache."""

from src.cache.cache import Cache, CacheItem


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
