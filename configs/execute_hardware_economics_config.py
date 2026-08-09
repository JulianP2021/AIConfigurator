#!/usr/bin/env python3
"""Run hardware-economics sweeps over TTFT and user-delay values.

The input config file follows the same schema as execute_user_sweep_config.py:

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
                "config_type": "colocated"
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
import math
import re
import sys
import time

from collections import defaultdict
from pathlib import Path
from typing import Any


# Ensure project root is on sys.path when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.utils.config_utils import get_focus
from src.hardware.hardware import S3Spec
from src.logger import LOG_CONFIG_EXECUTOR, log, should_log
from src.result import SimulationResult
from src.router.router import RouterCostConfig
from src.utils.config_runner import (
    build_common_config,
    clone_config,
    load_config,
    run_single_config,
    validate_colocated_configs,
)
from src.utils.env_reader import EnvConfig, load_env
from src.utils.parser import _add_logging_args, apply_logging_args
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


def _resolve_tpot_ms(config: dict[str, Any], env: EnvConfig) -> float:
    """Return a finite TPOT SLA to use for the benchmark sweep.

    The benchmark focuses on TTFT, so TPOT is taken from the config file or
    environment defaults and must be a finite positive number.
    """
    raw = config.get("sla", {}).get("tpot_ms")
    if raw is None:
        raw = env.sla_tpot_ms
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"tpot_ms must be a finite positive number for scheduled arrivals, got {value}"
        )

    print(value)
    return value


def _build_run_config(
    cfg: dict[str, Any],
    ttft_ms: float,
    user_delay_ms: float,
) -> dict[str, Any]:
    run_cfg = clone_config(cfg)
    run_cfg["label"] = _build_run_label(str(cfg["label"]), ttft_ms, user_delay_ms)
    run_cfg["benchmark_ttft_ms"] = ttft_ms
    run_cfg["benchmark_user_delay_ms"] = user_delay_ms
    run_cfg["benchmark_mode"] = "hardware_economics"
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
    router_cost_config: RouterCostConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = load_env()
    tpot_ms = _resolve_tpot_ms(config, env)
    common = build_common_config(
        config,
        env,
        sla_override={"ttft_ms": float(ttft_values[0]), "tpot_ms": tpot_ms},
        user_delay_fraction_override=user_delay_fraction,
        user_delay_min_ms_override=user_delay_values[0],
        user_delay_max_ms_override=user_delay_values[0],
        router_cost_config=router_cost_config,
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
                run_common["sla"] = {"ttft_ms": ttft_ms, "tpot_ms": tpot_ms}
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


def _run_single_config_for_users(
    run_spec: dict[str, Any],
    users: int,
    config: dict[str, Any],
    env: EnvConfig,
    s3_spec: S3Spec,
    router_cost_config: RouterCostConfig,
) -> SimulationResult | Exception:
    """Run a single config with a specific user count, returning either the
    result or the exception that terminated the run.
    """
    common = dict(run_spec["common"])
    common["users"] = users
    # Prefer an explicit router config carried in the run spec (e.g. from
    # tuning or from the config file), otherwise use the worker's fallback.
    effective_router_config = router_cost_config
    common_router_cfg = run_spec.get("common", {}).get("router_cost_config")
    if isinstance(common_router_cfg, RouterCostConfig):
        effective_router_config = common_router_cfg
    ram_usage_fraction = float(config.get("ram_usage_fraction", env.ram_usage_fraction))
    ssd_usage_fraction = float(config.get("ssd_usage_fraction", env.ssd_usage_fraction))
    try:
        result = run_single_config(
            common,
            run_spec["cfg"],
            ram_usage_fraction,
            ssd_usage_fraction,
            s3_spec,
            effective_router_config,
        )
        if should_log(LOG_CONFIG_EXECUTOR):
            log(
                LOG_CONFIG_EXECUTOR,
                f"{run_spec['cfg']['label']} users={users} OK "
                f"max_ttft={result.max_ttft:.1f} max_tpot={result.max_tpot:.2f} "
                f"router=(aws={effective_router_config.active_work_scale:.6f}, "
                f"dc={effective_router_config.device_credit:.4f})",
            )
        return result
    except Exception as exc:
        if should_log(LOG_CONFIG_EXECUTOR):
            log(
                LOG_CONFIG_EXECUTOR,
                f"{run_spec['cfg']['label']} users={users} FAIL {type(exc).__name__}: {str(exc)[:80]}",
            )
        return exc


def _find_max_users(
    run_spec: dict[str, Any],
    config: dict[str, Any],
    env: EnvConfig,
    s3_spec: S3Spec,
    router_cost_config: RouterCostConfig,
    timeout_s: float,
) -> tuple[int, SimulationResult | None]:
    """Exponential + binary search for the largest user count that succeeds.

    A "success" means ``run_single_config`` returns a ``SimulationResult``
    without raising an exception. The search starts at ``users=1`` and doubles
    until the first failure (or an internal ceiling), then binary-searches
    between the last known success and the first known failure.

    Returns ``(max_users, result_at_max_users)``; ``result_at_max_users`` is
    ``None`` if even ``users=1`` fails.
    """

    def _is_valid(value: SimulationResult | Exception) -> bool:
        return isinstance(value, SimulationResult)

    start_time = time.monotonic()

    def _remaining_timeout() -> float:
        return max(0.0, timeout_s - (time.monotonic() - start_time))

    lo_result = _run_single_config_for_users(
        run_spec, 1, config, env, s3_spec, router_cost_config
    )
    if not _is_valid(lo_result):
        if should_log(LOG_CONFIG_EXECUTOR):
            log(
                LOG_CONFIG_EXECUTOR,
                f"Config '{run_spec['cfg']['label']}' failed even with users=1: {lo_result}",
            )
        return 0, None

    # Exponential search: find an upper bound where the config fails.
    lo = 1
    hi = 2
    hi_result: SimulationResult | Exception | None = None
    while True:
        remaining = _remaining_timeout()
        if remaining <= 0.0:
            if should_log(LOG_CONFIG_EXECUTOR):
                log(
                    LOG_CONFIG_EXECUTOR,
                    f"Config '{run_spec['cfg']['label']}' timed out during exponential search",
                )
            return lo, lo_result  # type: ignore[return-value]

        hi_result = _run_single_config_for_users(
            run_spec, hi, config, env, s3_spec, router_cost_config
        )
        if not _is_valid(hi_result):
            break
        lo = hi
        lo_result = hi_result
        hi *= 2
        # Hard ceiling so we do not run forever on extremely large counts.
        if hi > 1_000_000_000:
            if should_log(LOG_CONFIG_EXECUTOR):
                log(
                    LOG_CONFIG_EXECUTOR,
                    f"Config '{run_spec['cfg']['label']}' succeeded up to hard ceiling {hi}",
                )
            return hi, hi_result  # type: ignore[return-value]

    # Binary search between lo (known valid) and hi (known invalid).
    while hi - lo > 1:
        remaining = _remaining_timeout()
        if remaining <= 0.0:
            if should_log(LOG_CONFIG_EXECUTOR):
                log(
                    LOG_CONFIG_EXECUTOR,
                    f"Config '{run_spec['cfg']['label']}' timed out during binary search; returning lo={lo}",
                )
            break
        mid = lo + (hi - lo) // 2
        mid_result = _run_single_config_for_users(
            run_spec, mid, config, env, s3_spec, router_cost_config
        )
        if _is_valid(mid_result):
            lo = mid
            lo_result = mid_result
        else:
            hi = mid

    return lo, lo_result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Worker entry point for the process pool.
# ---------------------------------------------------------------------------


def _run_spec_worker(
    run_spec: dict[str, Any],
    config: dict[str, Any],
    env: EnvConfig,
    s3_spec: S3Spec,
    router_cost_config: RouterCostConfig,
    timeout_s: float,
    seed_override: int | None = None,
) -> tuple[dict[str, Any], SimulationResult | Exception | None]:
    """Run one expanded spec, searching for the max user count.

    When ``seed_override`` is provided, it replaces the random seed in the
    common payload so each per-seed run is deterministic and independent.
    """
    import copy

    # Work on a private copy so per-seed overrides do not leak back.
    run_spec = copy.deepcopy(run_spec)
    common = run_spec.setdefault("common", {})
    if seed_override is not None:
        common["random_seed"] = seed_override

    # Preserve any router config that arrived as part of the common payload.
    effective_router_config = router_cost_config
    if isinstance(common.get("router_cost_config"), RouterCostConfig):
        effective_router_config = common["router_cost_config"]

    max_users, result = _find_max_users(
        run_spec,
        config,
        env,
        s3_spec,
        effective_router_config,
        timeout_s,
    )
    meta = {"max_users": max_users, "run_spec": run_spec}
    if seed_override is not None:
        meta["seed"] = seed_override
    return meta, result


def _run_sweep(
    config: dict[str, Any],
    ttft_values: list[float],
    user_delay_values: list[float],
    user_delay_fraction: float,
    timeout_s: float,
    num_seeds: int = 1,
) -> list[dict[str, Any]]:
    env = load_env()
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
        active_work_scale=float(
            config.get("router_active_work_scale", env.router_active_work_scale)
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
    )

    _common, expanded_runs = _expand_run_specs(
        config,
        ttft_values,
        user_delay_values,
        user_delay_fraction,
        router_cost_config=router_cost_config,
    )

    base_seed = int(config.get("random_seed", env.random_seed or 0))
    seeds = [base_seed + i for i in range(max(1, num_seeds))]
    print(
        f"Finding max users for {len(expanded_runs)} specs x {len(seeds)} seed(s) = "
        f"{len(expanded_runs) * len(seeds)} jobs"
    )

    results: list[dict[str, Any]] = []
    """RAM|NVLink|SSD|SSD BW|INET BW"""
    palette = {
        "RAM": "#58a6ff",
        "NVLink": "#3fb950",
        "SSD": "#f85149",
        "SSD BW": "#d29922",
        "INET BW": "#a371f7",
        "INTER NODE BW": "#0f00b3",
        None: "#000000",
        # "#56d364",
        # "#f0883e",
        # "#db61a2",
        # "#39c5cf",
    }

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                _run_spec_worker,
                run_spec,
                config,
                env,
                s3_spec,
                router_cost_config,
                timeout_s,
                seed,
            ): (run_spec, seed)
            for run_spec in expanded_runs
            for seed in seeds
        }
        successful: dict[tuple[int, int], tuple[dict[str, Any], SimulationResult]] = {}
        failed: dict[tuple[int, int], Exception] = {}
        pending = dict(futures)

        end_time = time.monotonic() + timeout_s * len(expanded_runs) * len(seeds)
        while pending:
            wait_s = max(0.0, min(end_time - time.monotonic(), 1.0))
            done, _ = concurrent.futures.wait(
                pending,
                timeout=wait_s,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done and time.monotonic() >= end_time:
                for future, (run_spec, seed) in pending.items():
                    future.cancel()
                    failed[(len(successful) + len(failed), seed)] = RuntimeError(
                        f"timed out after {timeout_s}s"
                    )
                    print(
                        f"Config '{run_spec['cfg']['label']}' seed={seed} timed out after {timeout_s}s",
                        file=sys.stderr,
                    )
                break
            for future in done:
                run_spec, seed = pending.pop(future)
                try:
                    worker_meta, result = future.result()
                    max_users = int(worker_meta.get("max_users", 0))
                    if max_users <= 0 or not isinstance(result, SimulationResult):
                        failed[(id(run_spec), seed)] = RuntimeError(
                            "failed during max-users search"
                        )
                        print(
                            f"Config '{run_spec['cfg']['label']}' seed={seed} failed max-users search: {result}"
                        )
                        continue
                    row = result.to_dict()
                    focus, focus_value = _run_focus(run_spec["cfg"])
                    color = palette[focus]
                    total_cost = row.get("total_cost_usd_per_hour", 0.0)
                    price_per_user = (
                        total_cost / max_users if max_users > 0 else float("inf")
                    )
                    add_result_metadata(
                        row,
                        str(run_spec["cfg"]["label"]),
                        run_spec["cfg"],
                        color,
                        users=max_users,
                        extra_fields={
                            "benchmark_mode": "hardware_economics",
                            "ttft_sla_ms": run_spec["ttft_ms"],
                            "tpot_sla_ms": run_spec["common"]["sla"]["tpot_ms"],
                            "user_delay_ms": run_spec["user_delay_ms"],
                            "user_delay_fraction": user_delay_fraction,
                            "focus": focus,
                            "focus_value": focus_value,
                            "max_users": max_users,
                            "price_per_user": round(price_per_user, 6),
                            "seed": seed,
                        },
                    )
                    results.append(row)
                except Exception as exc:
                    failed[(id(run_spec), seed)] = exc
                    print(
                        f"Config '{run_spec['cfg']['label']}' seed={seed} failed: {exc}"
                    )

    return results


def _write_results_dir(
    results_dir: Path,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    ttft_values: list[float],
    user_delay_values: list[float],
    user_delay_fraction: float,
    num_seeds: int = 1,
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
            "benchmark": "hardware_economics",
            "config": {
                "source_config": str(config.get("source_config", "")),
                "ttft_values": ttft_values,
                "user_delay_values": user_delay_values,
                "user_delay_fraction": user_delay_fraction,
                "num_seeds": num_seeds,
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
    parser = argparse.ArgumentParser(description="Run hardware-economics sweeps")
    _add_logging_args(parser, env)
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
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of distinct random seeds to run per (config, ttft, delay) combination. "
        "The first seed is the config's random_seed; each additional seed is offset by +i "
        "so runs remain deterministic. When >1, every per-seed max_users result is "
        "written to the output JSON for variance/tail analysis (default: 1).",
    )
    args = parser.parse_args()
    args.ttft_values = [f * 1000 for f in args.ttft_values]
    args.user_delay_values = [f * 1000 * 60 for f in args.user_delay_values]

    for ttft_ms in args.ttft_values:
        if not math.isfinite(ttft_ms) or ttft_ms <= 0:
            parser.error(
                f"--ttft-values must be finite positive seconds, got {ttft_ms / 1000:g}s"
            )

    if args.results_dir is None:
        parser.error("Provide --results-dir")

    config = load_config(args.config)
    apply_logging_args(args)

    results = _run_sweep(
        config,
        args.ttft_values,
        args.user_delay_values,
        args.user_delay_fraction,
        args.timeout,
        num_seeds=args.seeds,
    )

    payload = {
        "benchmark": "hardware_economics",
        "config": {
            "source_config": str(args.config),
            "ttft_values": args.ttft_values,
            "user_delay_values": args.user_delay_values,
            "user_delay_fraction": args.user_delay_fraction,
        },
        "results": results,
    }

    written = _write_results_dir(
        args.results_dir,
        payload["config"],
        results,
        args.ttft_values,
        args.user_delay_values,
        args.user_delay_fraction,
        num_seeds=args.seeds,
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
