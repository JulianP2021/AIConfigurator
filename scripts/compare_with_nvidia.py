#!/usr/bin/env python3
"""Run a side-by-side comparison between:
1. Your discrete-event simulator (configurator)
2. NVIDIA AI Configurator (aiconfigurator) estimate API.

Usage:
    python scripts/compare_with_nvidia.py --isl 1000 --osl 100 --sessions-per-user 1 --users 10 --model Qwen/Qwen3-8B
    python scripts/compare_with_nvidia.py --isl 1000 --osl 100 --unique-users
    python scripts/compare_with_nvidia.py --debug --save results.json
"""

import argparse
import json
import logging
import sys

from pathlib import Path
from typing import Any

from aiconfigurator.cli.api import EstimateResult, cli_estimate
from aiconfigurator.sdk.perf_database import set_systems_paths


# Ensure project root is on sys.path when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hardware.scraper import fetch_machine_hardware
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
SYSTEM = "h200_sxm"
BACKEND = "vllm"
# Default hardware from our dynamically-loaded presets (single GPU)
DEFAULT_HARDWARE = "H200 x1 #b731cab8"


def run_your_simulator(
    *,
    model: str,
    isl: int,
    osl: int,
    sessions_per_user: int,
    users: int,
    batch_size: int,
    prefill_workers: int,
    decode_workers: int,
    hardware: str,
) -> SimulationResult:
    """Run the discrete-event simulator with the matched topology."""
    print("[1/2] Running your simulator ...")

    print(users, sessions_per_user)

    return simulate_run_distributed(
        DistributedScenario(
            name="compare",
            nodes=[
                Node(
                    hardware=fetch_machine_hardware(hardware),
                    model_name=model,
                    batch_size=batch_size,
                    prefill_instances=prefill_workers,
                    decode_instances=0,
                ),
                Node(
                    hardware=fetch_machine_hardware(hardware),
                    model_name=model,
                    batch_size=batch_size,
                    prefill_instances=0,
                    decode_instances=decode_workers,
                ),
            ],
            requests=RequestScenario(
                sessions_per_user=sessions_per_user,
                users=users,
                max_session_turns=1,
                think_time_ms=0,
                token_distribution=TokenDistribution(
                    min_input_tokens=isl,
                    max_input_tokens=isl,
                    min_output_tokens=osl,
                    max_output_tokens=osl,
                ),
            ),
        ),
        sla={"ttft_ms": 30000.0, "tpot_ms": 100.0},
    )


def run_nvidia_disagg(
    *,
    model: str,
    isl: int,
    osl: int,
    batch_size: int,
    prefill_workers: int,
    decode_workers: int,
    database_mode: str,
) -> EstimateResult:
    """Run NVIDIA AI Configurator disagg estimate with the same topology."""
    print(
        f"[2/2] Running NVIDIA AI Configurator ({BACKEND}, disagg, {database_mode}) ..."
    )
    set_systems_paths("default")

    return cli_estimate(
        model_path=model,
        system_name=SYSTEM,
        mode="disagg",
        backend_name=BACKEND,
        database_mode=database_mode,
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
    # Attach database_mode so it can be read later in the comparison table


# Available database modes from NVIDIA AI Configurator
DATABASE_MODES = ["SOL"]


def nvidia_estimate_to_dict(est: EstimateResult, database_mode: str) -> dict[str, Any]:
    """Normalize an EstimateResult into a plain dict."""
    return {
        "mode": est.mode,
        "database_mode": database_mode,
        "model_path": est.model_path,
        "system_name": est.system_name,
        "backend_name": est.backend_name,
        "backend_version": est.backend_version,
        "isl": est.isl,
        "osl": est.osl,
        "batch_size": est.batch_size,
        "tp_size": est.tp_size,
        "pp_size": est.pp_size,
        "ttft": round(est.ttft, 3),
        "tpot": round(est.tpot, 3),
        "request_latency": round(est.request_latency, 3),
        "tokens_per_second": round(est.tokens_per_second, 2),
        "tokens_per_second_per_gpu": round(est.tokens_per_second_per_gpu, 2),
        "tokens_per_second_per_user": round(est.tokens_per_second_per_user, 2),
        "seq_per_second": round(est.seq_per_second, 3),
        "concurrency": round(est.concurrency, 2),
        "power_w": round(est.power_w, 2),
        "memory_gb": round(est.memory, 2),
        "raw": est.raw,
    }


def _label(est: EstimateResult, db_mode: str) -> str:
    return f"NVIDIA AIC ({est.mode}, {db_mode})"


def print_comparison_table(
    sim_result: SimulationResult, nvidia_results: list[tuple[str, EstimateResult]]
) -> None:
    """Pretty-print a side-by-side comparison table."""
    # Collect rows
    rows = [
        ("Your Simulator", sim_result.to_dict()),
    ]
    for db_mode, est in nvidia_results:
        rows.append((_label(est, db_mode), nvidia_estimate_to_dict(est, db_mode)))

    # Keys to compare
    keys = [
        ("ttft", "TTFT (ms)"),
        ("tpot", "TPOT (ms)"),
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
        for name, d in rows:
            v = d.get(key, "N/A")
            if name == "Your Simulator" and key == "ttft":
                v = d.get("avg_prefill_time_ms")
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
    for db_mode, est in nvidia_results:
        if est.mode == "disagg":
            raw: dict[str, Any] = est.raw
            print(
                f"  NVIDIA ({est.mode}, {db_mode}):   "
                f"(p){raw.get('(p)workers', '?')} worker(s) x {raw.get('(p)tp', '?')} GPU + "
                f"(d){raw.get('(d)workers', '?')} worker(s) x {raw.get('(d)tp', '?')} GPU, "
                f"(d)bs={raw.get('(d)bs', '?')}"
            )
        else:
            print(
                f"  NVIDIA ({est.mode}, {db_mode}):     {est.tp_size} GPU(s) TP, batch={est.batch_size}"
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
        default=2,
        help="Output sequence length (fixed, default: 2)",
    )
    parser.add_argument(
        "--sessions-per-user",
        type=int,
        default=1,
        help="Sessions per user (default: 1). Total requests = users * sessions_per_user.",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=10,
        help="Fixed pool of users taking turns (default: 10)",
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
    parser.add_argument(
        "--hardware",
        type=str,
        default=DEFAULT_HARDWARE,
        help=f"Hardware preset for your simulator (default: {DEFAULT_HARDWARE})",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--database-modes",
        nargs="+",
        choices=DATABASE_MODES,
        default=DATABASE_MODES,
        help=f"NVIDIA database mode(s) to compare ({', '.join(DATABASE_MODES)}). Default: all",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Path to save JSON results"
    )
    args = parser.parse_args()

    if args.debug:
        set_debug(True)
        logging.getLogger("aiconfigurator").setLevel(logging.DEBUG)
        print("Debug logging enabled.")

    # Run your simulator
    sim_result = run_your_simulator(
        model=args.model,
        isl=args.isl,
        osl=args.osl,
        sessions_per_user=args.sessions_per_user,
        users=args.users,
        batch_size=args.batch_size,
        prefill_workers=args.prefill_workers,
        decode_workers=args.decode_workers,
        hardware=args.hardware,
    )

    # Run NVIDIA estimates across all requested database modes
    nvidia_results: list[tuple[str, EstimateResult]] = []
    for db_mode in args.database_modes:
        try:
            nvidia_results.append((
                db_mode,
                run_nvidia_disagg(
                    model=args.model,
                    isl=args.isl,
                    osl=args.osl,
                    batch_size=args.batch_size,
                    prefill_workers=args.prefill_workers,
                    decode_workers=args.decode_workers,
                    database_mode=db_mode,
                ),
            ))
        except Exception as exc:
            print(f"WARNING: NVIDIA disagg estimate failed ({db_mode}): {exc}")

    if not nvidia_results:
        print("ERROR: No NVIDIA estimates succeeded. Nothing to compare.")
        sys.exit(1)

    # Print comparison
    print_comparison_table(sim_result, nvidia_results)

    # Optional JSON save
    if args.save:
        payload = {
            "your_simulator": sim_result.to_dict(),
            "nvidia": [
                nvidia_estimate_to_dict(est, db_mode) for db_mode, est in nvidia_results
            ],
            "settings": {
                "model": args.model,
                "system": SYSTEM,
                "backend": BACKEND,
                "isl": args.isl,
                "osl": args.osl,
                "sessions_per_user": args.sessions_per_user,
                "users": args.users,
                "batch_size": args.batch_size,
                "prefill_workers": args.prefill_workers,
                "decode_workers": args.decode_workers,
            },
        }
        with Path(args.save).open("w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved comparison to {args.save}")


if __name__ == "__main__":
    main()
