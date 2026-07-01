#!/usr/bin/env python3
"""Run the distributed simulator with configurable CLI parameters.

Defaults are read from ``.env`` at the project root. CLI arguments override
``.env`` values, and shell environment variables override ``.env`` values.

Usage examples:
    # Default scenario (from .env)
    python main.py

    # Custom model and lengths
    python main.py --model Qwen/Qwen3-8B --isl 1000 --osl 100 --requests 10 --req-rate 2

    # Unique users (no shared prefix / no repeat users)
    python main.py --model Qwen/Qwen3-8B --isl 1000 --osl 100 --requests 10 --unique-users

    # Debug mode
    python main.py --debug --model Qwen/Qwen3-8B --isl 1000 --osl 100
"""

import argparse

from src.hardware.hardware import Hardware
from src.hardware.scraper import fetch_machine_hardware
from src.logger import set_debug, set_log_mask
from src.node.node import Node
from src.request.request import RequestScenario, TokenDistribution
from src.simulations.simulation_distributed import (
    DistributedScenario,
    simulate_run_distributed,
)
from src.utils.env_reader import EnvConfig, load_env


def build_parser(env: EnvConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed LLM inference simulator")
    parser.add_argument(
        "--model",
        type=str,
        default=env.model,
        help=f"HuggingFace model name (default: {env.model})",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=env.isl,
        help=f"Input sequence length (fixed, default: {env.isl})",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=env.osl,
        help=f"Output sequence length (fixed, default: {env.osl})",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=env.requests,
        help=f"Total requests to simulate (default: {env.requests})",
    )
    parser.add_argument(
        "--req-rate",
        type=float,
        default=env.req_rate,
        help=f"Request arrival rate in req/s (default: {env.req_rate})",
    )
    parser.add_argument(
        "--unique-users",
        action="store_true",
        default=env.unique_users,
        help="Set max_users > total_requests so every request gets a unique user (no shared prefix)",
    )
    parser.add_argument(
        "--min-users",
        type=int,
        default=env.min_users,
        help=f"Minimum number of users (default: {env.min_users})",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=env.max_users,
        help=f"Maximum number of users (default: {env.max_users})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=env.batch_size,
        help=f"Decode batch size (default: {env.batch_size})",
    )
    parser.add_argument(
        "--prefill-workers",
        type=int,
        default=env.prefill_workers,
        help=f"Number of prefill workers (default: {env.prefill_workers})",
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=env.decode_workers,
        help=f"Number of decode workers (default: {env.decode_workers})",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=env.gpus_per_node,
        help=f"GPUs per node (default: {env.gpus_per_node})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=env.debug,
        help="Enable verbose debug logging (sets LOG_MASK to all components)",
    )
    parser.add_argument(
        "--log-mask",
        type=lambda s: int(s, 0),
        default=env.log_mask,
        help=(
            "Component logging bitmask: bit 0 (1)=cache, bit 1 (2)=instances, "
            "bit 2 (4)=router, bit 3 (8)=simulation. 0=none, 15=all "
            f"(default: {env.log_mask})"
        ),
    )
    parser.add_argument(
        "--cache-pct",
        type=float,
        default=env.cache_pct,
        help=f"Cache percentage for prefix caching (default: {env.cache_pct})",
    )
    parser.add_argument(
        "--ram-usage-fraction",
        type=float,
        default=env.ram_usage_fraction,
        help=f"Fraction of node RAM usable for the KV cache layer (default: {env.ram_usage_fraction})",
    )
    parser.add_argument(
        "--ssd-usage-fraction",
        type=float,
        default=env.ssd_usage_fraction,
        help=f"Fraction of node SSD usable for the KV cache layer (default: {env.ssd_usage_fraction})",
    )
    return parser


def main():
    env = load_env()
    parser = build_parser(env)
    args = parser.parse_args()

    if args.debug:
        set_debug(True)
        print("Debug logging enabled (LOG_MASK=all).")
    else:
        set_log_mask(args.log_mask)
        if args.log_mask:
            print(f"Logging enabled with LOG_MASK={args.log_mask}.")

    # If --unique-users, force min_users and max_users > total_requests so every
    # request is a new user (no shared prefix / no repeat users).
    if args.unique_users:
        min_users = args.requests + 1
        max_users = args.requests + 1
        print(
            f"--unique-users set: forcing {min_users} users (> {args.requests} requests) → no shared prefix."
        )
    else:
        max_users = args.max_users
        min_users = args.min_users

    # ISL/OSL are fixed (min=max) as requested
    token_dist = TokenDistribution(
        min_input_tokens=args.isl,
        max_input_tokens=args.isl,
        min_output_tokens=args.osl,
        max_output_tokens=args.osl,
        cache_percentage=args.cache_pct,
    )

    hardware = fetch_machine_hardware("H200 x1 #440acd08")
    assert type(hardware) is Hardware, (
        f"Expected Hardware instance, got {type(hardware)}"
    )
    assert hardware is not None, "Failed to fetch hardware for H200 x1 #440acd08"

    scenario = DistributedScenario(
        name="cli_run",
        nodes=[
            Node(
                hardware=hardware,
                model_name=args.model,
                batch_size=args.batch_size,
                prefill_instances=args.prefill_workers,
                decode_instances=0,
            ),
            Node(
                hardware=hardware,
                model_name=args.model,
                batch_size=args.batch_size,
                prefill_instances=0,
                decode_instances=args.decode_workers,
            ),
        ],
        requests=RequestScenario(
            token_distribution=token_dist,
            total_requests=args.requests,
            min_users=min_users,
            max_users=max_users,
            req_s=args.req_rate,
        ),
    )

    result = simulate_run_distributed(
        scenario,
        ram_usage_fraction=args.ram_usage_fraction,
        ssd_usage_fraction=args.ssd_usage_fraction,
    )

    # Print compact JSON for piping
    import json

    print("\n--- JSON ---")
    print(json.dumps(result.to_dict()))


if __name__ == "__main__":
    main()
