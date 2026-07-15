"""Tests for src.hardware.scraper helpers.

These tests use a synthetic machine database so they remain stable when the
legacy Vast.ai cache (``src/hardware/legacy/_machine_db.json``) is refreshed.
"""

from unittest.mock import patch

import pytest

from src.hardware.scraper import (
    fetch_machine_hardware,
    resolve_machine_name,
)


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


def test_custom_hardware_takes_precedence_over_machine_db(fake_machine_db):
    """Custom hardware entries are resolved before scraped machine entries."""
    custom = {
        "Custom A100 x2": {
            "name": "Custom A100 x2",
            "gpu_name": "A100_SXM4",
            "num_gpus": 2,
            "nvme_mem": 1_000_000_000_000,
            "nvme_bw": 10_000_000_000,
            "network_inet_up": 1_000_000_000,
            "network_inet_down": 1_000_000_000,
            "pcie_bw": 32_000_000_000,
            "cpu_ram": 256_000_000_000,
        }
    }
    with (
        patch("src.hardware.scraper.load_machine_db", return_value=fake_machine_db),
        patch(
            "src.hardware.scraper.load_gpu_db",
            return_value={
                "A100_SXM4": {"flops": 312e12, "gpu_mem": 80e9, "gpu_bw": 2e12},
            },
        ),
        patch("src.hardware.scraper.load_aws_hardware_db", return_value=({}, custom)),
    ):
        resolved = resolve_machine_name("Custom A100 x2")
        assert resolved == "Custom A100 x2"
        hw = fetch_machine_hardware("Custom A100 x2")
        assert hw.spec.num_gpus == 2
        assert hw.spec.nvme_mem == 1_000_000_000_000
        assert hw.spec.cpu_ram == 256_000_000_000


def test_custom_hardware_missing_required_field_raises(tmp_path, fake_machine_db):
    """Custom entries must supply the fields needed for simulation."""
    custom = {"BadGPU": {"gpu_name": "A100_SXM4", "num_gpus": 1}}
    with (
        patch("src.hardware.scraper.load_machine_db", return_value=fake_machine_db),
        patch(
            "src.hardware.scraper.load_gpu_db",
            return_value={
                "A100_SXM4": {"flops": 312e12, "gpu_mem": 80e9, "gpu_bw": 2e12},
            },
        ),
        patch("src.hardware.scraper.load_aws_hardware_db", return_value=({}, custom)),
        pytest.raises(ValueError, match="missing required field"),
    ):
        fetch_machine_hardware("BadGPU")
