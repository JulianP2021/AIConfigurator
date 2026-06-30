"""Tests for the bitmask logger."""

import logging

import pytest

from src import logger
from src.logger import (
    LOG_ALL,
    LOG_CACHE,
    LOG_INSTANCE,
    LOG_NONE,
    LOG_ROUTER,
    LOG_SIMULATION,
    get_log_mask,
    is_debug,
    log,
    set_debug,
    set_log_mask,
)


class TestLogMask:
    def test_default_mask_is_all(self):
        assert get_log_mask() == LOG_ALL

    def test_set_log_mask_changes_mask(self):
        set_log_mask(LOG_CACHE)
        assert get_log_mask() == LOG_CACHE
        set_log_mask(LOG_ALL)
        assert get_log_mask() == LOG_ALL

    def test_log_uses_bitmask(self, reset_logger_mask):
        recorded: list[str] = []

        def fake_log(level: int, msg: str) -> None:
            recorded.append(msg)

        set_log_mask(LOG_CACHE)
        original = logger.logger.log
        logger.logger.log = fake_log  # type: ignore[assignment]
        try:
            log(LOG_CACHE, "cache message")
            log(LOG_ROUTER, "router message")
            log(LOG_INSTANCE, "instance message")
            log(LOG_SIMULATION, "simulation message")
        finally:
            logger.logger.log = original

        assert any("cache message" in m for m in recorded)
        assert not any("router message" in m for m in recorded)
        assert not any("instance message" in m for m in recorded)
        assert not any("simulation message" in m for m in recorded)

    def test_log_respects_level(self, reset_logger_mask):
        recorded: list[str] = []

        def fake_log(level: int, msg: str) -> None:
            recorded.append((level, msg))

        set_log_mask(LOG_ALL)
        original = logger.logger.log
        logger.logger.log = fake_log  # type: ignore[assignment]
        try:
            log(LOG_CACHE, "debug message", level=logging.DEBUG)
            log(LOG_CACHE, "info message", level=logging.INFO)
        finally:
            logger.logger.log = original

        assert any(level == logging.DEBUG for level, _ in recorded)
        assert any(level == logging.INFO for level, _ in recorded)

    def test_set_debug_enables_all(self, reset_logger_mask):
        set_debug(True)
        assert get_log_mask() == LOG_ALL
        assert is_debug()

    def test_set_debug_false_disables_all(self, reset_logger_mask):
        set_debug(False)
        assert get_log_mask() == LOG_NONE
        assert not is_debug()

    def test_log_with_zero_mask_drops_everything(self, reset_logger_mask):
        recorded: list[str] = []

        def fake_log(level: int, msg: str) -> None:
            recorded.append(msg)

        set_log_mask(LOG_NONE)
        original = logger.logger.log
        logger.logger.log = fake_log  # type: ignore[assignment]
        try:
            log(LOG_CACHE, "dropped")
        finally:
            logger.logger.log = original

        assert recorded == []


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        (0, False),
        (LOG_CACHE, False),
        (LOG_INSTANCE, False),
        (LOG_ROUTER, False),
        (LOG_SIMULATION, False),
        (LOG_ALL, True),
    ],
)
def test_is_debug_various_masks(mask: int, expected: bool, reset_logger_mask):
    set_log_mask(mask)
    logger.set_min_level(logging.DEBUG)
    assert is_debug() == expected
