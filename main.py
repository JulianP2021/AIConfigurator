#!/usr/bin/env python3
"""Run the distributed simulator with configurable CLI parameters.

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

from src.hardware.hardware import Hardware, S3Spec
from src.hardware.scraper import fetch_machine_hardware
from src.logger import set_debug, set_log_mask
from src.node.node import Node
from src.request.request import RequestScenario, TokenDistribution
from src.simulations.simulation_distributed import (
    DistributedScenario,
    simulate_run_distributed,
)
from src.utils.env_reader import load_env
from src.utils.parser import get_main_parser


def main():
    env = load_env()
    parser = get_main_parser(env)
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
    )

    hardware = fetch_machine_hardware("H200 x1 #8a0e41af")
    assert type(hardware) is Hardware, (
        f"Expected Hardware instance, got {type(hardware)}"
    )
    assert hardware is not None, "Failed to fetch hardware for H200 x1 #8a0e41af"

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
            max_session_turns=args.max_session_turns,
            req_s=args.req_rate,
        ),
    )

    s3_spec = S3Spec.from_gbps(
        enabled=args.s3_enabled,
        up_gbps=args.s3_up_bw_gbps,
        down_gbps=args.s3_down_bw_gbps,
    )

    result = simulate_run_distributed(
        scenario,
        ram_usage_fraction=args.ram_usage_fraction,
        ssd_usage_fraction=args.ssd_usage_fraction,
        s3_spec=s3_spec,
    )

    # Print compact JSON for piping
    import json

    print("\n--- JSON ---")
    print(json.dumps(result.to_dict()))


if __name__ == "__main__":
    main()
