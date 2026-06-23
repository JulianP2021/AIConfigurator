#!/usr/bin/env python3
"""
Run the distributed simulator with configurable CLI parameters.

Usage examples:
    # Default scenario
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
from src.logger import set_debug
from src.node.node import Node
from src.request.request import RequestScenario, TokenDistribution
from src.simulations.simulation_distributed import (
    DistributedScenario,
    simulate_run_distributed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed LLM inference simulator")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name (default: Qwen/Qwen3-8B)",
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
        "--requests",
        type=int,
        default=10,
        help="Total requests to simulate (default: 10)",
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
        help="Set max_users > total_requests so every request gets a unique user (no shared prefix)",
    )
    parser.add_argument(
        "--min-users", type=int, default=1, help="Minimum number of users (default: 1)"
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=10,
        help="Maximum number of users (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Decode batch size (default: 10)"
    )
    parser.add_argument(
        "--prefill-workers",
        type=int,
        default=1,
        help="Number of prefill workers (default: 1)",
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=1,
        help="Number of decode workers (default: 1)",
    )
    parser.add_argument(
        "--gpus-per-node", type=int, default=1, help="GPUs per node (default: 1)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--cache-pct",
        type=float,
        default=0.0,
        help="Cache percentage for prefix caching (default: 0.0, i.e. no cache hit)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.debug:
        set_debug(True)
        print("Debug logging enabled.")

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

    scenario = DistributedScenario(
        name="cli_run",
        nodes=[
            Node(
                hardware=Hardware.from_name("DGX SPARK"),
                model_name=args.model,
                batch_size=args.batch_size,
                prefill_instances=args.prefill_workers,
                decode_instances=0,
            ),
            Node(
                hardware=Hardware.from_name("DGX SPARK"),
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

    result = simulate_run_distributed(scenario)

    # Print compact JSON for piping
    import json

    print("\n--- JSON ---")
    print(json.dumps(result.to_dict()))


if __name__ == "__main__":
    main()
