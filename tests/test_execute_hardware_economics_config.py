"""Tests for the hardware-economics sweep runner."""

from __future__ import annotations
import json

from types import SimpleNamespace

import pytest

from configs.execute_hardware_economics_config import (
    _build_run_config,
    _expand_run_specs,
    _write_results_dir,
    validate_colocated_configs,
)


class TestValidateColocatedConfigs:
    def test_rejects_non_colocated_configs(self) -> None:
        with pytest.raises(ValueError, match="colocated"):
            validate_colocated_configs([
                {
                    "label": "bad",
                    "gpu": "H200",
                    "prefill_hardware": "H200 x8 #a",
                    "decode_hardware": "H200 x8 #a",
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                    "colocated": False,
                }
            ])


class TestTTFTExpansion:
    def test_cartesian_product_and_fixed_sla(self, monkeypatch) -> None:
        fake_env = SimpleNamespace(
            model="Qwen/Qwen3-8B",
            isl=128,
            osl=8,
            sessions_per_user=1,
            users=4,
            max_session_turns=1,
            think_time_ms=0.0,
            user_delay_fraction=0.0,
            user_delay_min_ms=0.0,
            user_delay_max_ms=0.0,
            startup_arrival_mean_ms=0.0,
            random_seed=42,
            sla_ttft_ms=30000.0,
            sla_tpot_ms=100.0,
        )
        monkeypatch.setattr("configs.execute_hardware_economics_config.load_env", lambda: fake_env)

        config = {
            "model": "Qwen/Qwen3-8B",
            "isl": 128,
            "osl": 8,
            "sessions_per_user": 1,
            "users": 4,
            "max_session_turns": 1,
            "think_time_ms": 0.0,
            "configs": [
                {
                    "label": "Focused H200 SSD BW 50.0GBps x8 r2200 s32625 p270 nvl483 sbw50.0 inet0.0/0.0",
                    "prefill_hardware": "H200 x8 #a",
                    "decode_hardware": "H200 x8 #a",
                    "gpu": "H200",
                    "prefill_nodes": 1,
                    "decode_nodes": 1,
                    "prefill_gpus_per_node": 4,
                    "decode_gpus_per_node": 4,
                    "batch_size": 16,
                    "colocated": True,
                }
            ],
        }

        common, runs = _expand_run_specs(config, [25.0, 50.0], [0.0, 10.0], 0.25)

        assert common["sla"]["ttft_ms"] == 25.0
        assert common["sla"]["tpot_ms"] == 100.0
        assert len(runs) == 4
        assert {run["ttft_ms"] for run in runs} == {25.0, 50.0}
        assert {run["user_delay_ms"] for run in runs} == {0.0, 10.0}
        for run in runs:
            assert run["common"]["user_delay_fraction"] == 0.25
            assert run["common"]["user_delay_min_ms"] == run["user_delay_ms"]
            assert run["common"]["user_delay_max_ms"] == run["user_delay_ms"]
            assert run["common"]["sla"]["tpot_ms"] == 100.0
            assert "TTFT=" in run["cfg"]["label"]
            assert "delay=" in run["cfg"]["label"]

    def test_build_run_config_encodes_label(self) -> None:
        cfg = {
            "label": "Focused H200 SSD BW 50.0GBps x8 r2200 s32625 p270 nvl483 sbw50.0 inet0.0/0.0",
            "gpu": "H200",
            "prefill_hardware": "H200 x8 #a",
            "decode_hardware": "H200 x8 #a",
            "prefill_nodes": 1,
            "decode_nodes": 1,
            "colocated": True,
        }

        run_cfg = _build_run_config(cfg, 100.0, 50.0)

        assert (
            run_cfg["label"]
            == "Focused H200 SSD BW 50.0GBps x8 r2200 s32625 p270 nvl483 sbw50.0 inet0.0/0.0 | TTFT=100ms | delay=50ms"
        )
        assert run_cfg["benchmark_ttft_ms"] == 100.0
        assert run_cfg["benchmark_user_delay_ms"] == 50.0
        assert run_cfg["benchmark_mode"] == "hardware_economics"

    def test_build_run_config_keeps_focus_metadata(self) -> None:
        cfg = {
            "label": "Focused H200 SSD BW 50.0GBps x8 r2200 s32625 p270 nvl483 sbw50.0 inet0.0/0.0",
            "gpu": "H200",
            "prefill_hardware": "H200 x8 #a",
            "decode_hardware": "H200 x8 #a",
            "prefill_nodes": 2,
            "decode_nodes": 2,
            "colocated": True,
            "focus": "nodes",
            "focus_value": 2,
        }

        run_cfg = _build_run_config(cfg, 100.0, 50.0)

        assert run_cfg["focus"] == "SSD BW"
        assert run_cfg["focus_value"] == "50.0"

    def test_write_results_dir_groups_by_focus(self, tmp_path) -> None:
        payload = {
            "source_config": "config.json",
            "ttft_values": [100.0],
            "user_delay_values": [50.0],
            "user_delay_fraction": 0.25,
        }
        results = [
            {"label": "a", "focus": "nodes", "focus_value": 2, "has_error": False},
            {"label": "b", "focus": "nodes", "focus_value": 4, "has_error": False},
        ]

        written = _write_results_dir(
            tmp_path,
            payload,
            results,
            [100.0],
            [50.0],
            0.25,
        )

        assert sorted(path.name for path in written) == [
            "results_nodes_2.json",
            "results_nodes_4.json",
        ]
        data = json.loads((tmp_path / "results_nodes_2.json").read_text())
        assert data["config"]["focus"] == "nodes"
        assert data["config"]["focus_value"] == "2"
