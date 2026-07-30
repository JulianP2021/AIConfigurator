"""Tests for the shared CLI argument parser helpers."""

import argparse

from types import SimpleNamespace

from src.logger import LOG_ALL, LOG_CONFIG_EXECUTOR, LOG_ROUTER, get_log_mask, is_debug
from src.utils.parser import _add_logging_args, apply_logging_args


def _make_env(log_mask: int = 0, debug: bool = False) -> SimpleNamespace:
    return SimpleNamespace(log_mask=log_mask, debug=debug)


def test_add_logging_args_defaults():
    parser = argparse.ArgumentParser()
    _add_logging_args(parser, _make_env(log_mask=4, debug=False))
    args = parser.parse_args([])
    assert args.log_mask == 4
    assert args.debug is False


def test_log_mask_parses_hex():
    parser = argparse.ArgumentParser()
    _add_logging_args(parser, _make_env())
    args = parser.parse_args(["--log-mask", "0x3f"])
    assert args.log_mask == 63


def test_debug_overrides_log_mask(reset_logger_mask):
    parser = argparse.ArgumentParser()
    _add_logging_args(parser, _make_env(log_mask=0, debug=False))
    args = parser.parse_args(["--log-mask", str(LOG_ROUTER), "--debug"])
    apply_logging_args(args)
    assert get_log_mask() == LOG_ALL
    assert is_debug()


def test_log_mask_without_debug(reset_logger_mask):
    parser = argparse.ArgumentParser()
    _add_logging_args(parser, _make_env(log_mask=0, debug=False))
    args = parser.parse_args(["--log-mask", str(LOG_CONFIG_EXECUTOR)])
    apply_logging_args(args)
    assert get_log_mask() == LOG_CONFIG_EXECUTOR
    assert not is_debug()


def test_no_flags_uses_env_defaults(reset_logger_mask):
    parser = argparse.ArgumentParser()
    _add_logging_args(parser, _make_env(log_mask=LOG_ROUTER, debug=False))
    args = parser.parse_args([])
    apply_logging_args(args)
    assert get_log_mask() == LOG_ROUTER
    assert not is_debug()
