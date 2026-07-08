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
from src.router.router import RouterCostConfig
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

    hardware = fetch_machine_hardware(args.machine_hardware)
    assert type(hardware) is Hardware, (
        f"Expected Hardware instance, got {type(hardware)}"
    )
    assert hardware is not None, f"Failed to fetch hardware for {args.machine_hardware}"

    total_gpus = hardware.spec.num_gpus
    prefill_split_explicit = args.prefill_gpus_per_node >= 0
    prefill_gpus_per_node = (
        args.prefill_gpus_per_node if prefill_split_explicit else args.prefill_workers
    )

    nodes: list[Node] = []
    if args.colocated:
        if prefill_gpus_per_node >= total_gpus:
            raise ValueError(
                f"--prefill-gpus-per-node ({prefill_gpus_per_node}) must be "
                f"less than total GPUs per node ({total_gpus}) in colocated mode."
            )
        decode_gpus_per_node = total_gpus - prefill_gpus_per_node
        for _ in range(args.num_prefill_nodes):
            nodes.append(
                Node(
                    hardware=hardware,
                    model_name=args.model,
                    batch_size=args.batch_size,
                    prefill_instances=prefill_gpus_per_node,
                    decode_instances=decode_gpus_per_node,
                )
            )
        print(
            f"Colocated mode: {args.num_prefill_nodes} node(s), each with "
            f"{prefill_gpus_per_node} prefill + {decode_gpus_per_node} decode GPU(s)."
        )
    else:
        # Non-colocated nodes dedicate all GPUs to their single role, unless
        # the user explicitly set --prefill-gpus-per-node.
        prefill_instances_per_node = (
            prefill_gpus_per_node if prefill_split_explicit else total_gpus
        )
        decode_instances_per_node = total_gpus
        for _ in range(args.num_prefill_nodes):
            nodes.append(
                Node(
                    hardware=hardware,
                    model_name=args.model,
                    batch_size=args.batch_size,
                    prefill_instances=prefill_instances_per_node,
                    decode_instances=0,
                )
            )
        for _ in range(args.num_decode_nodes):
            nodes.append(
                Node(
                    hardware=hardware,
                    model_name=args.model,
                    batch_size=args.batch_size,
                    prefill_instances=0,
                    decode_instances=decode_instances_per_node,
                )
            )
        print(
            f"Non-colocated mode: {args.num_prefill_nodes} prefill-only node(s) "
            f"({prefill_instances_per_node} GPU(s) each), "
            f"{args.num_decode_nodes} decode-only node(s) "
            f"({decode_instances_per_node} GPU(s) each)."
        )

    scenario = DistributedScenario(
        name="cli_run",
        nodes=nodes,
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

    router_cost_config = RouterCostConfig(
        prefill_load_scale=args.router_prefill_load_scale,
        device_credit=args.router_device_credit,
        remote_ram_credit=args.router_remote_ram_credit,
        ssd_credit=args.router_ssd_credit,
        s3_credit=args.router_s3_credit,
        busy_threshold_tokens=args.router_busy_threshold_tokens,
    )

    result = simulate_run_distributed(
        scenario,
        ram_usage_fraction=args.ram_usage_fraction,
        ssd_usage_fraction=args.ssd_usage_fraction,
        s3_spec=s3_spec,
        router_cost_config=router_cost_config,
    )

    # Print compact JSON for piping
    import json

    print("\n--- JSON ---")
    print(json.dumps(result.to_dict()))


if __name__ == "__main__":
    main()
