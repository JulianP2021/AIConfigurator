"""Tests for the two-tier distributed cache."""

from src.cache.cache import CacheItem, CacheLayer


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
        assert item.node_id == -2
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
