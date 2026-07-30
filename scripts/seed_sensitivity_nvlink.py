#!/usr/bin/env python3
"""Controlled seed-sensitivity analysis for NVLink variants.

Runs `find_max_users` for the NVLink-focused colocated configs in the
hardware-economics config file, repeating each variant over a set of random
seeds.  Reports per-seed max users plus mean/std/min/max so we can see whether
the bandwidth-aware router's max-users ranking is robust or just noise.
"""

from __future__ import annotations
import argparse
import concurrent.futures
import json
import math
import sys
import time

from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy

from configs.execute_hardware_economics_config import (
    _expand_run_specs,
    _run_spec_worker,
)
from src.hardware.hardware import S3Spec
from src.router.router import RouterCostConfig
from src.utils.config_runner import load_config
from src.utils.env_reader import EnvConfig, load_env


def _nvlink_only_specs(
    config: dict[str, Any], ttft_ms: float, _tpot_ms: float
) -> list[dict[str, Any]]:
    """Return only the NVLink-focused run specs from the full expansion."""
    env = load_env()
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
    user_delay_ms = 1.0  # seconds converted to ms below is handled by caller
    _, expanded = _expand_run_specs(
        config,
        [ttft_ms],
        [user_delay_ms],
        user_delay_fraction=0.0,
        router_cost_config=router_cost_config,
    )
    return [spec for spec in expanded if "nvlink" in spec["cfg"]["label"].lower()]


def _run_spec_with_seed(
    spec: dict[str, Any],
    config: dict[str, Any],
    env: EnvConfig,
    s3_spec: S3Spec,
    router_cost_config: RouterCostConfig,
    seed: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Run a single (spec, seed) pair through the max-users worker."""
    # Make a deep copy of the run spec so the original is not mutated, and patch
    # the random seed into the common payload.  RouterCostConfig is not JSON
    # serializable, so use copy.deepcopy instead of json round-tripping.

    run_spec = copy.deepcopy(spec)
    run_spec["common"]["random_seed"] = seed
    # Ensure the users value from the search is preserved (it is set by
    # _run_single_config_for_users at runtime, so no override is needed here).

    worker_meta, result = _run_spec_worker(
        run_spec,
        config,
        env,
        s3_spec,
        router_cost_config,
        timeout_s,
        tune_router=False,
        tune_grid=None,
        tune_max_workers=1,
        tune_timeout_s=30.0,
        tune_refine=False,
        tune_budget_fractions=None,
        estimate_timeout_s=30.0,
    )
    label = run_spec["cfg"]["label"]
    focus_value = run_spec["cfg"].get("focus_value", "unknown")
    max_users = worker_meta.get("max_users", 0)
    row = result.to_dict() if hasattr(result, "to_dict") else {}
    return {
        "label": label,
        "focus_value": focus_value,
        "seed": seed,
        "max_users": max_users,
        "max_ttft": row.get("max_ttft"),
        "max_tpot": row.get("max_tpot"),
        "tokens_per_second": row.get("tokens_per_second"),
        "error": None if max_users > 0 else str(result),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed-sensitivity for NVLink variants")
    parser.add_argument(
        "--config", type=Path, required=True, help="Base hardware-economics config JSON"
    )
    parser.add_argument(
        "--ttft-s", type=float, default=100.0, help="TTFT SLA in seconds"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066],
        help="Seeds to test",
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0, help="Per-config timeout in seconds"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="JSON file to write detailed results"
    )
    parser.add_argument(
        "--max-workers", type=int, default=8, help="Parallel worker count"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    env = load_env()
    tpot_ms = float(config.get("sla", {}).get("tpot_ms", env.sla_tpot_ms))
    ttft_ms = args.ttft_s * 1000.0

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

    specs = _nvlink_only_specs(config, ttft_ms, tpot_ms)
    if not specs:
        print("No NVLink configs found; aborting.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Running {len(specs)} NVLink variants x {len(args.seeds)} seeds = {len(specs) * len(args.seeds)} jobs"
    )

    tasks = [(spec, seed) for spec in specs for seed in args.seeds]

    detailed: list[dict[str, Any]] = []
    start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {
            executor.submit(
                _run_spec_with_seed,
                spec,
                config,
                env,
                s3_spec,
                router_cost_config,
                seed,
                args.timeout,
            ): (spec, seed)
            for spec, seed in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            spec, seed = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "label": spec["cfg"]["label"],
                    "focus_value": spec["cfg"].get("focus_value", "unknown"),
                    "seed": seed,
                    "max_users": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            detailed.append(row)
            print(f"  {row['label']} seed={row['seed']} -> users={row['max_users']}")

    elapsed = time.monotonic() - start

    # Aggregate by focus_value.
    by_variant: dict[str, list[int]] = {}
    for row in detailed:
        by_variant.setdefault(row["focus_value"], []).append(row["max_users"])

    print("\n=== Seed-sensitivity summary ===")
    print(f"TTFT = {args.ttft_s}s, seeds = {args.seeds}")
    print(f"Total wall time: {elapsed:.1f}s")
    print(f"{'NVLink':>8} {'mean':>6} {'std':>6} {'min':>5} {'max':>5} {'n':>4}")
    summary: dict[str, dict[str, Any]] = {}
    for variant, users in sorted(by_variant.items(), key=lambda kv: float(kv[0])):
        vals = [u for u in users if u > 0]
        n = len(vals)
        mean = sum(vals) / n if n else 0.0
        var = sum((u - mean) ** 2 for u in vals) / n if n else 0.0
        std = math.sqrt(var)
        summary[variant] = {
            "mean": mean,
            "std": std,
            "min": min(vals) if vals else 0,
            "max": max(vals) if vals else 0,
            "n": n,
        }
        print(
            f"{variant:>8} {mean:>6.1f} {std:>6.1f} {summary[variant]['min']:>5} {summary[variant]['max']:>5} {n:>4}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "config": str(args.config),
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "seeds": args.seeds,
                    "elapsed_s": elapsed,
                    "detailed": detailed,
                    "summary": summary,
                },
                fh,
                indent=2,
            )
        print(f"\nWrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
