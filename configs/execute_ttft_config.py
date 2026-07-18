#!/usr/bin/env python3
"""Run colocated TTFT-vs-cost sweeps over user-delay values.

The input config file follows the same schema as execute_config.py:

    {
        "model": "Qwen/Qwen3-8B",
        "isl": 128,
        "osl": 8,
        "sessions_per_user": 1,
        "users": 4,
        "max_session_turns": 1,
        "configs": [
            {
                "label": "Colocated Config",
                "prefill_hardware": "H200 x8 #8a0e41af",
                "decode_hardware": "H200 x8 #8a0e41af",
                "prefill_nodes": 1,
                "decode_nodes": 1,
                "prefill_gpus_per_node": 4,
                "decode_gpus_per_node": 4,
                "batch_size": 10,
                "colocated": true
            }
        ]
    }

Every config must be colocated.  The runner expands the cartesian product of
TTFT values and user-delay values, executes each combination, and writes a
single JSON payload that is compatible with the web import page.
"""

from __future__ import annotations
import argparse
import concurrent.futures
import json
import re
import sys

from collections import defaultdict
from pathlib import Path
from typing import Any


# Ensure project root is on sys.path when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.utils.config_utils import get_focus
from src.hardware.hardware import S3Spec
from src.logger import LOG_CONFIG_EXECUTOR, log, set_log_mask, should_log
from src.result import SimulationResult
from src.router.router import RouterCostConfig
from src.utils.config_runner import (
    build_common_config,
    clone_config,
    load_config,
    run_single_config,
    validate_colocated_configs,
)
from src.utils.env_reader import load_env
from src.utils.utils import add_result_metadata, parse_float_list


def _parse_float_values(raw: str) -> list[float]:
    values = parse_float_list(raw)
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def _build_run_label(base_label: str, ttft_ms: float, user_delay_ms: float) -> str:
    return f"{base_label} | TTFT={ttft_ms:g}ms | delay={user_delay_ms:g}ms"


def _slugify(value: object) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return slug or "default"


def _run_focus(cfg: dict[str, Any]) -> tuple[str, Any]:
    return get_focus(cfg["label"], cfg["gpu"])


def _build_run_config(
    cfg: dict[str, Any],
    ttft_ms: float,
    user_delay_ms: float,
) -> dict[str, Any]:
    run_cfg = clone_config(cfg)
    run_cfg["label"] = _build_run_label(str(cfg["label"]), ttft_ms, user_delay_ms)
    run_cfg["benchmark_ttft_ms"] = ttft_ms
    run_cfg["benchmark_user_delay_ms"] = user_delay_ms
    run_cfg["benchmark_mode"] = "ttft_cost_by_delay"
    focus, focus_value = _run_focus(cfg)
    run_cfg["focus"] = focus
    run_cfg["focus_value"] = focus_value
    run_cfg["config_type"] = str(cfg.get("config_type", focus))
    return run_cfg


def _expand_run_specs(
    config: dict[str, Any],
    ttft_values: list[float],
    user_delay_values: list[float],
    user_delay_fraction: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = load_env()
    common = build_common_config(
        config,
        env,
        sla_override={"ttft_ms": float(ttft_values[0]), "tpot_ms": float("inf")},
        user_delay_fraction_override=user_delay_fraction,
        user_delay_min_ms_override=user_delay_values[0],
        user_delay_max_ms_override=user_delay_values[0],
    )

    base_configs = list(config.get("configs", []))
    validate_colocated_configs(base_configs)
    if not base_configs:
        raise ValueError("Config file does not contain any colocated configs")

    expanded_runs: list[dict[str, Any]] = []
    for ttft_ms in ttft_values:
        for user_delay_ms in user_delay_values:
            for cfg in base_configs:
                run_cfg = _build_run_config(cfg, ttft_ms, user_delay_ms)
                run_common = dict(common)
                run_common["sla"] = {"ttft_ms": ttft_ms, "tpot_ms": float("inf")}
                run_common["user_delay_fraction"] = user_delay_fraction
                run_common["user_delay_min_ms"] = user_delay_ms
                run_common["user_delay_max_ms"] = user_delay_ms
                expanded_runs.append({
                    "common": run_common,
                    "cfg": run_cfg,
                    "ttft_ms": ttft_ms,
                    "user_delay_ms": user_delay_ms,
                })

    return common, expanded_runs


def _run_sweep(
    config: dict[str, Any],
    ttft_values: list[float],
    user_delay_values: list[float],
    user_delay_fraction: float,
    timeout_s: float,
) -> list[dict[str, Any]]:
    env = load_env()
    _common, expanded_runs = _expand_run_specs(
        config,
        ttft_values,
        user_delay_values,
        user_delay_fraction,
    )

    s3_spec = S3Spec.from_gbps(
        enabled=bool(config.get("s3_enabled", env.s3_enabled)),
        up_gbps=float(config.get("s3_up_bw_gbps", env.s3_up_bw_gbps)),
        down_gbps=float(config.get("s3_down_bw_gbps", env.s3_down_bw_gbps)),
        eviction_time_ms=float(
            config.get("s3_eviction_time_ms", env.s3_eviction_time_ms)
        ),
    )
    router_cost_config = RouterCostConfig(
        prefill_load_scale=float(
            config.get("router_prefill_load_scale", env.router_prefill_load_scale)
        ),
        device_credit=float(
            config.get("router_device_credit", env.router_device_credit)
        ),
        remote_ram_credit=float(
            config.get("router_remote_ram_credit", env.router_remote_ram_credit)
        ),
        remote_ssd_credit=float(
            config.get("router_remote_ssd_credit", env.router_remote_ssd_credit)
        ),
        s3_credit=float(config.get("router_s3_credit", env.router_s3_credit)),
        busy_threshold_tokens=float(
            config.get("router_busy_threshold_tokens", env.router_busy_threshold_tokens)
        ),
    )

    results: list[dict[str, Any]] = []
    """RAM|NVLink|SSD|SSD BW|INET BW"""
    palette = {
        "RAM": "#58a6ff",
        "NVLink": "#3fb950",
        "SSD": "#f85149",
        "SSD BW": "#d29922",
        "INET BW": "#a371f7",
        # "#79c0ff",
        # "#56d364",
        # "#f0883e",
        # "#db61a2",
        # "#39c5cf",
    }

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                run_single_config,
                run_spec["common"],
                run_spec["cfg"],
                float(config.get("ram_usage_fraction", env.ram_usage_fraction)),
                float(config.get("ssd_usage_fraction", env.ssd_usage_fraction)),
                s3_spec,
                router_cost_config,
            ): (i, run_spec)
            for i, run_spec in enumerate(expanded_runs)
        }

        successful: dict[int, tuple[dict[str, Any], SimulationResult]] = {}
        failed: dict[int, Exception] = {}
        pending = dict(futures)
        import time

        end_time = time.monotonic() + timeout_s
        while pending:
            wait_s = max(0.0, min(end_time - time.monotonic(), 1.0))
            done, _ = concurrent.futures.wait(
                pending,
                timeout=wait_s,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done and time.monotonic() >= end_time:
                for future, (_, run_spec) in pending.items():
                    future.cancel()
                    failed[len(successful) + len(failed)] = RuntimeError(
                        f"timed out after {timeout_s}s"
                    )
                    print(
                        f"Config '{run_spec['cfg']['label']}' timed out after {timeout_s}s",
                        file=sys.stderr,
                    )
                break
            for future in done:
                i, run_spec = pending.pop(future)
                try:
                    successful[i] = (run_spec, future.result())
                except Exception as exc:
                    failed[i] = exc
                    if should_log(LOG_CONFIG_EXECUTOR):
                        log(
                            LOG_CONFIG_EXECUTOR,
                            f"Config '{run_spec['cfg']['label']}' failed: {exc}",
                        )

    for run_spec, result in successful.values():
        row = result.to_dict()
        focus, focus_value = _run_focus(run_spec["cfg"])
        color = palette[focus]
        add_result_metadata(
            row,
            str(run_spec["cfg"]["label"]),
            run_spec["cfg"],
            color,
            extra_fields={
                "benchmark_mode": "ttft_cost_by_delay",
                "ttft_sla_ms": run_spec["ttft_ms"],
                "tpot_sla_ms": float("inf"),
                "user_delay_ms": run_spec["user_delay_ms"],
                "user_delay_fraction": user_delay_fraction,
                "sweep_ttft_ms": run_spec["ttft_ms"],
                "focus": focus,
                "focus_value": focus_value,
            },
        )
        results.append(row)

    return results


def _write_results_dir(
    results_dir: Path,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    ttft_values: list[float],
    user_delay_values: list[float],
    user_delay_fraction: float,
) -> list[Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        focus = str(row.get("focus") or row.get("config_type") or "default")
        focus_value = row.get("focus_value")
        grouped[(focus, str(focus_value))].append(row)

    written: list[Path] = []
    for (focus, focus_value), rows in sorted(grouped.items()):
        payload = {
            "benchmark": "ttft_cost_by_delay",
            "config": {
                "source_config": str(config.get("source_config", "")),
                "ttft_values": ttft_values,
                "user_delay_values": user_delay_values,
                "user_delay_fraction": user_delay_fraction,
                "focus": focus,
                "focus_value": focus_value,
            },
            "results": rows,
        }
        file_name = f"results_{_slugify(focus)}_{_slugify(focus_value)}.json"
        out_path = results_dir / file_name
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        written.append(out_path)

    return written


def main() -> None:
    env = load_env()
    parser = argparse.ArgumentParser(description="Run TTFT-vs-cost colocated sweeps")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to the base config JSON"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory where per-focus results_*.json files will be written",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional legacy single-file output path",
    )
    parser.add_argument(
        "--ttft-values",
        type=_parse_float_values,
        required=True,
        help="Comma-separated TTFT SLA values in s, e.g. '10,20,50,100'",
    )
    parser.add_argument(
        "--user-delay-values",
        type=_parse_float_values,
        required=True,
        help="Comma-separated user-delay values in minutes, e.g. '20,50,100'",
    )
    parser.add_argument(
        "--user-delay-fraction",
        type=float,
        default=env.user_delay_fraction,
        help=f"Fixed user-delay fraction to use for all runs (default: {env.user_delay_fraction})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=240.0,
        help="Per-config timeout in seconds (default: 240.0)",
    )
    args = parser.parse_args()
    args.ttft_values = [f * 1000 for f in args.ttft_values]
    args.user_delay_values = [f * 1000 * 60 for f in args.user_delay_values]

    if args.results_dir is None and args.output is None:
        parser.error("Provide --results-dir")

    config = load_config(args.config)
    set_log_mask(LOG_CONFIG_EXECUTOR)

    results = _run_sweep(
        config,
        args.ttft_values,
        args.user_delay_values,
        args.user_delay_fraction,
        args.timeout,
    )

    payload = {
        "benchmark": "ttft_cost_by_delay",
        "config": {
            "source_config": str(args.config),
            "ttft_values": args.ttft_values,
            "user_delay_values": args.user_delay_values,
            "user_delay_fraction": args.user_delay_fraction,
        },
        "results": results,
    }

    if args.results_dir is not None:
        written = _write_results_dir(
            args.results_dir,
            payload["config"],
            results,
            args.ttft_values,
            args.user_delay_values,
            args.user_delay_fraction,
        )
        summary = {
            "benchmark": payload["benchmark"],
            "results_dir": str(args.results_dir),
            "files": [str(path) for path in written],
            "result_count": len(results),
        }
        print(summary)


if __name__ == "__main__":
    main()
