#!/usr/bin/env python3
"""Run a side-by-side comparison between:
1. Your discrete-event simulator (configurator)
2. NVIDIA AI Configurator (aiconfigurator) estimate API.

Usage:
    python scripts/compare_with_nvidia.py --isl 1000 --osl 100 --requests 10 --model Qwen/Qwen3-8B
    python scripts/compare_with_nvidia.py --isl 1000 --osl 100 --unique-users
    python scripts/compare_with_nvidia.py --debug --save results.json
"""

import argparse
import json
import logging
import sys

from pathlib import Path

from aiconfigurator.cli.api import EstimateResult, cli_estimate


# Ensure project root is on sys.path when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hardware.hardware import Hardware
from src.logger import set_debug
from src.node.node import Node
from src.request.request import RequestScenario, TokenDistribution
from src.result import SimulationResult
from src.simulations.simulation_distributed import (
    DistributedScenario,
    simulate_run_distributed,
)


# Suppress noisy NVIDIA loggers by default
for noisy in ("aiconfigurator", "transformers", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SYSTEM = "h100_sxm"
BACKEND = "vllm"


def run_your_simulator(
    *,
    model: str,
    isl: int,
    osl: int,
    total_requests: int,
    req_rate: float,
    unique_users: bool,
    batch_size: int,
    prefill_workers: int,
    decode_workers: int,
    cache_pct: float,
) -> SimulationResult:
    """Run the discrete-event simulator with the matched topology."""
    print("[1/2] Running your simulator ...")

    # If unique_users, force max_users > total_requests so every request is a new user
    if unique_users:
        min_users = total_requests + 1
        max_users = total_requests + 1
        print(
            f"  Unique users: {max_users} users (> {total_requests} requests) → no shared prefix."
        )
    else:
        min_users = 1
        max_users = 10

    return simulate_run_distributed(
        DistributedScenario(
            name="compare",
            nodes=[
                Node(
                    hardware=Hardware.from_name("H100SXM"),
                    model_name=model,
                    batch_size=batch_size,
                    prefill_instances=prefill_workers,
                    decode_instances=0,
                ),
                Node(
                    hardware=Hardware.from_name("H100SXM"),
                    model_name=model,
                    batch_size=batch_size,
                    prefill_instances=0,
                    decode_instances=decode_workers,
                ),
            ],
            requests=RequestScenario(
                total_requests=total_requests,
                min_users=min_users,
                max_users=max_users,
                req_s=req_rate,
                token_distribution=TokenDistribution(
                    min_input_tokens=isl,
                    max_input_tokens=isl,
                    min_output_tokens=osl,
                    max_output_tokens=osl,
                    cache_percentage=cache_pct,
                ),
            ),
        ),
    )


def run_nvidia_disagg(
    *,
    model: str,
    isl: int,
    osl: int,
    batch_size: int,
    prefill_workers: int,
    decode_workers: int,
) -> EstimateResult:
    """Run NVIDIA AI Configurator disagg estimate with the same topology."""
    print(f"[2/2] Running NVIDIA AI Configurator ({BACKEND}, disagg) ...")
    return cli_estimate(
        model_path=model,
        system_name=SYSTEM,
        mode="disagg",
        backend_name=BACKEND,
        database_mode="SILICON",
        isl=isl,
        osl=osl,
        # prefill
        prefill_tp_size=1,
        prefill_pp_size=1,
        prefill_batch_size=1,
        prefill_num_workers=prefill_workers,
        # decode
        decode_tp_size=1,
        decode_pp_size=1,
        decode_batch_size=batch_size,
        decode_num_workers=decode_workers,
    )


def run_nvidia_agg(
    *,
    model: str,
    isl: int,
    osl: int,
    batch_size: int,
    total_gpus: int,
) -> EstimateResult:
    """Run NVIDIA AI Configurator agg estimate for comparison."""
    print(f"[2/2] Running NVIDIA AI Configurator ({BACKEND}, agg) ...")
    return cli_estimate(
        model_path=model,
        system_name=SYSTEM,
        mode="agg",
        backend_name=BACKEND,
        database_mode="SILICON",
        isl=isl,
        osl=osl,
        batch_size=batch_size,
        tp_size=total_gpus,
        pp_size=1,
    )


def nvidia_estimate_to_dict(est: EstimateResult) -> dict:
    """Normalize an EstimateResult into a plain dict."""
    return {
        "mode": est.mode,
        "model_path": est.model_path,
        "system_name": est.system_name,
        "backend_name": est.backend_name,
        "backend_version": est.backend_version,
        "isl": est.isl,
        "osl": est.osl,
        "batch_size": est.batch_size,
        "tp_size": est.tp_size,
        "pp_size": est.pp_size,
        "ttft_ms": round(est.ttft, 3),
        "tpot_ms": round(est.tpot, 3),
        "request_latency_ms": round(est.request_latency, 3),
        "tokens_per_second": round(est.tokens_per_second, 2),
        "tokens_per_second_per_gpu": round(est.tokens_per_second_per_gpu, 2),
        "tokens_per_second_per_user": round(est.tokens_per_second_per_user, 2),
        "seq_per_second": round(est.seq_per_second, 3),
        "concurrency": round(est.concurrency, 2),
        "power_w": round(est.power_w, 2),
        "memory_gb": round(est.memory, 2),
        "raw": est.raw,
    }


def print_comparison_table(
    sim_result: SimulationResult, nvidia_results: list[EstimateResult]
) -> None:
    """Pretty-print a side-by-side comparison table."""
    # Collect rows
    rows = [
        ("Your Simulator", sim_result.to_dict()),
    ]
    for est in nvidia_results:
        label = f"NVIDIA AIC ({est.mode})"
        rows.append((label, nvidia_estimate_to_dict(est)))

    # Keys to compare
    keys = [
        ("ttft_ms", "TTFT (ms)"),
        ("tpot_ms", "TPOT (ms)"),
        ("request_latency_ms", "Request Latency (ms)"),
        ("max_request_latency_ms", "max Latency (ms)"),
        ("tokens_per_second", "tokens/s"),
        ("tokens_per_second_per_gpu", "tokens/s/gpu"),
        ("tokens_per_second_per_user", "tokens/s/user"),
        ("request_rate", "req/s"),
        ("memory_gb", "Memory (GB)"),
    ]

    # Print header
    col_w = 22
    val_w = 20
    header = f"{'Metric':>{col_w}}"
    for label, _ in rows:
        header += f" | {label:^{val_w}}"
    sep = "-" * (col_w + 3 + (val_w + 3) * len(rows))
    print("\n" + "=" * len(sep))
    print("  Side-by-Side Comparison")
    print("=" * len(sep))
    print(header)
    print(sep)

    for key, display in keys:
        line = f"{display:>{col_w}}"
        for _, d in rows:
            v = d.get(key, "N/A")
            s = f"{v:,.2f}" if isinstance(v, float) else str(v)
            line += f" | {s:^{val_w}}"
        print(line)

    print(sep + "\n")

    # Print topology details
    print("Topology Details:")
    print(
        f"  Your Simulator:   {sim_result.num_prefill_workers} prefill worker(s) x {sim_result.prefill_gpus_per_worker} GPU + "
        f"{sim_result.num_decode_workers} decode worker(s) x {sim_result.decode_gpus_per_worker} GPU, batch={sim_result.batch_size}"
    )
    for est in nvidia_results:
        if est.mode == "disagg":
            raw = est.raw
            print(
                f"  NVIDIA ({est.mode}):   "
                f"(p){raw.get('(p)workers', '?')} worker(s) x {raw.get('(p)tp', '?')} GPU + "
                f"(d){raw.get('(d)workers', '?')} worker(s) x {raw.get('(d)tp', '?')} GPU, "
                f"(d)bs={raw.get('(d)bs', '?')}"
            )
        else:
            print(
                f"  NVIDIA ({est.mode}):     {est.tp_size} GPU(s) TP, batch={est.batch_size}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Compare your simulator with NVIDIA AI Configurator"
    )
    # Model & workload
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=1000,
        help="Input sequence length (fixed, default: 1000)",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=100,
        help="Output sequence length (fixed, default: 100)",
    )
    parser.add_argument(
        "--requests", type=int, default=10, help="Total requests (default: 10)"
    )
    parser.add_argument(
        "--req-rate",
        type=float,
        default=2.0,
        help="Request arrival rate in req/s (default: 2.0)",
    )
    parser.add_argument(
        "--unique-users",
        action="store_true",
        help="Force max_users > requests so every request is a unique user (no shared prefix)",
    )
    parser.add_argument(
        "--cache-pct",
        type=float,
        default=0.0,
        help="Cache percentage for prefix caching (default: 0.0)",
    )
    # Topology
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Decode batch size (default: 10)"
    )
    parser.add_argument(
        "--prefill-workers", type=int, default=1, help="Prefill workers (default: 1)"
    )
    parser.add_argument(
        "--decode-workers", type=int, default=1, help="Decode workers (default: 1)"
    )
    # Comparison mode
    parser.add_argument(
        "--mode",
        choices=["disagg", "agg", "both"],
        default="disagg",
        help="Which NVIDIA mode to compare against (default: disagg)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Path to save JSON results"
    )
    args = parser.parse_args()

    if args.debug:
        set_debug(True)
        logging.getLogger("aiconfigurator").setLevel(logging.DEBUG)
        print("Debug logging enabled.")

    total_gpus = args.prefill_workers + args.decode_workers

    # Run your simulator
    sim_result = run_your_simulator(
        model=args.model,
        isl=args.isl,
        osl=args.osl,
        total_requests=args.requests,
        req_rate=args.req_rate,
        unique_users=args.unique_users,
        batch_size=args.batch_size,
        prefill_workers=args.prefill_workers,
        decode_workers=args.decode_workers,
        cache_pct=args.cache_pct,
    )

    # Run NVIDIA estimates
    nvidia_results: list[EstimateResult] = []
    if args.mode in ("disagg", "both"):
        try:
            nvidia_results.append(
                run_nvidia_disagg(
                    model=args.model,
                    isl=args.isl,
                    osl=args.osl,
                    batch_size=args.batch_size,
                    prefill_workers=args.prefill_workers,
                    decode_workers=args.decode_workers,
                )
            )
        except Exception as exc:
            print(f"WARNING: NVIDIA disagg estimate failed: {exc}")
    if args.mode in ("agg", "both"):
        try:
            nvidia_results.append(
                run_nvidia_agg(
                    model=args.model,
                    isl=args.isl,
                    osl=args.osl,
                    batch_size=args.batch_size,
                    total_gpus=total_gpus,
                )
            )
        except Exception as exc:
            print(f"WARNING: NVIDIA agg estimate failed: {exc}")

    if not nvidia_results:
        print("ERROR: No NVIDIA estimates succeeded. Nothing to compare.")
        sys.exit(1)

    # Print comparison
    print_comparison_table(sim_result, nvidia_results)

    # Optional JSON save
    if args.save:
        payload = {
            "your_simulator": sim_result.to_dict(),
            "nvidia": [nvidia_estimate_to_dict(est) for est in nvidia_results],
            "settings": {
                "model": args.model,
                "system": SYSTEM,
                "backend": BACKEND,
                "isl": args.isl,
                "osl": args.osl,
                "total_requests": args.requests,
                "req_rate": args.req_rate,
                "unique_users": args.unique_users,
                "batch_size": args.batch_size,
                "prefill_workers": args.prefill_workers,
                "decode_workers": args.decode_workers,
                "cache_pct": args.cache_pct,
            },
        }
        with Path(args.save).open("w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved comparison to {args.save}")


if __name__ == "__main__":
    main()
