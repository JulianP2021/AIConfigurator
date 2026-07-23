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

    def test_upload_prior_cache_layer_names_survive_merge(
        self,
        fake_model: Model,
        tiny_hardware: Hardware,
        s3_enabled: S3Spec,
        caplog,
    ):
        """Layer names in the upload log must not become '?' after merging.

        Regression: upload_kv captured prior_cache before inserting, but logged
        layer names *after* insert_cache_item had deleted the prior CacheItem
        objects via _merge_with_layer_items. The deleted items had layer=None,
        so the log printed '?' even though the data existed in a layer.
        """
        from src import logger

        cache = Cache(
            layers={},
            node_hardware={0: tiny_hardware, 1: tiny_hardware},
            model=fake_model,
            ram_usage_fraction=0.8,
            ssd_usage_fraction=0.8,
            s3_spec=s3_enabled,
        )

        # Local RAM prefix [0, 100) plus S3 prefix [100, 200).
        cache.insert_cache_item(CacheItem((1, 5), 0, 100), 0)
        s3_item = CacheItem((1, 5), 100, 200)
        cache._s3_layer()._add_item(s3_item)
        cache.s3_usage_bytes += cache._item_size(s3_item)

        req = Request(200, 8, 1, 5)
        req.prefilled_tokens = 200
        req.decoded_tokens = 0

        old_mask = logger.get_log_mask()
        logger.set_log_mask(logger.LOG_CACHE)
        try:
            cache.upload_kv(0, req)
        finally:
            logger.set_log_mask(old_mask)

        # The prior_cache_ranges entry must show the real layer, not '?'.
        log_text = caplog.text
        assert "prior_cache_ranges: [(0, 100, 'RAM')]" in log_text
