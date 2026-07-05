"""Tests for src.hardware.scraper helpers.

These tests use a synthetic machine database so they remain stable when the
live Vast.ai cache (``_machine_db.json``) is refreshed.
"""

from unittest.mock import patch

import pytest

from src.hardware.scraper import resolve_machine_name


@pytest.fixture
def fake_machine_db():
    """A small, deterministic machine database."""
    return {
        "ExactMatchGPU x1 #deadbeef": {
            "name": "ExactMatchGPU x1 #deadbeef",
            "gpu_name": "ExactMatchGPU",
            "num_gpus": 1,
        },
        "UniqueGPU x4 #cafebabe": {
            "name": "UniqueGPU x4 #cafebabe",
            "gpu_name": "UniqueGPU",
            "num_gpus": 4,
        },
        "SharedName A x1 #00000001": {
            "name": "SharedName A x1 #00000001",
            "gpu_name": "SharedName",
            "num_gpus": 1,
        },
        "SharedName B x2 #00000002": {
            "name": "SharedName B x2 #00000002",
            "gpu_name": "SharedName",
            "num_gpus": 2,
        },
    }


def test_resolve_exact_name(fake_machine_db):
    exact = "ExactMatchGPU x1 #deadbeef"
    with patch("src.hardware.scraper.load_machine_db", return_value=fake_machine_db):
        assert resolve_machine_name(exact) == exact


def test_resolve_by_single_gpu_name_substring(fake_machine_db):
    with patch("src.hardware.scraper.load_machine_db", return_value=fake_machine_db):
        resolved = resolve_machine_name("UniqueGPU")
        assert resolved == "UniqueGPU x4 #cafebabe"


def test_resolve_no_match_raises(fake_machine_db):
    with (
        patch("src.hardware.scraper.load_machine_db", return_value=fake_machine_db),
        pytest.raises(ValueError, match="No machine matching"),
    ):
        resolve_machine_name("DefinitelyNotAGPU")


def test_resolve_ambiguous_gpu_name_raises(fake_machine_db):
    # "SharedName" matches two instance sizes, so it must be rejected.
    with (
        patch("src.hardware.scraper.load_machine_db", return_value=fake_machine_db),
        pytest.raises(ValueError, match="Multiple machines match"),
    ):
        resolve_machine_name("SharedName")


def test_resolve_is_case_insensitive(fake_machine_db):
    with patch("src.hardware.scraper.load_machine_db", return_value=fake_machine_db):
        assert resolve_machine_name("uniquegpu") == "UniqueGPU x4 #cafebabe"


def test_resolve_exact_key_takes_precedence_over_substring(fake_machine_db):
    """An exact key match should win even if another entry shares gpu_name."""
    # Create a DB where the key itself is also a substring of another gpu_name.
    ambiguous_db = {
        **fake_machine_db,
        "SharedName B x2 #00000002": {
            "name": "SharedName B x2 #00000002",
            "gpu_name": "ExactMatchGPU x1 #deadbeef is a substring of me",
            "num_gpus": 2,
        },
    }
    with patch("src.hardware.scraper.load_machine_db", return_value=ambiguous_db):
        assert (
            resolve_machine_name("ExactMatchGPU x1 #deadbeef")
            == "ExactMatchGPU x1 #deadbeef"
        )
