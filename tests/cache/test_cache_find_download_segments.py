"""Tests for the two-tier distributed cache."""

from src.cache.cache import S3_NODE_ID, Cache, CacheItem
from src.hardware.hardware import Hardware, S3Spec
from src.model.model import Model


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

        effective_end, segments, _ = cache._find_download_segments(session, 2, 222_000)

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

        effective_end, segments, _ = cache._find_download_segments(session, 11, 94_000)

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

        effective_end, segments, _ = cache._find_download_segments(session, 0, 200)

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

        effective_end, segments, _ = cache._find_download_segments(session, 0, 200)

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

        effective_end, segments, _ = cache._find_download_segments(session, 0, 300)

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

        effective_end, segments, _ = cache._find_download_segments(session, 0, 222_000)

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

        effective_end, segments, _ = cache._find_download_segments(session, 5, 222_000)

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

        effective_end, segments, _ = cache._find_download_segments(session, 0, 100)
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

        effective_end, segments, _ = cache._find_download_segments(session, 0, 100)
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

        effective_end, segments, _ = cache._find_download_segments(session, 0, 500)
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

        effective_end, segments, _ = cache._find_download_segments(session, 0, 100)
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

        effective_end, segments, _ = cache._find_download_segments(session, 0, 200)
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

        effective_end, segments, _ = cache._find_download_segments(session, 0, 150)
        assert effective_end == 150
        assert segments == [
            (0, 50, 0, "SSD"),
            (50, 100, 1, "RAM"),
            (100, 150, S3_NODE_ID, "S3"),
        ]
