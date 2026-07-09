#!/usr/bin/env python3
"""Run the distributed simulator with configurable CLI parameters.

Usage examples:
    # Default scenario (from .env)
    python main.py

    # Custom model and lengths
    python main.py --model Qwen/Qwen3-8B --isl 1000 --osl 100 --sessions-per-user 1 --users 10

    # Unique users (no shared prefix / no repeat users)
    python main.py --model Qwen/Qwen3-8B --isl 1000 --osl 100 --sessions-per-user 1 --users 20

    # Debug mode
    python main.py --debug --model Qwen/Qwen3-8B --isl 1000 --osl 100

    # Scenario scale
    # total_requests = users * sessions_per_user * max_session_turns
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

    # If users >= total_requests every request gets its own user, which means no
    # shared prefix / no repeat users.  We keep the user's chosen value otherwise.
    total_requests = args.users * args.sessions_per_user * args.max_session_turns
    if args.users >= total_requests:
        print(
            f"--users ({args.users}) >= total_requests ({total_requests}): each request gets a unique user → no shared prefix."
        )

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
        args.prefill_gpus_per_node if prefill_split_explicit else total_gpus // 2
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
        prefill_instances_per_node = total_gpus
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
            sessions_per_user=args.sessions_per_user,
            users=args.users,
            max_session_turns=args.max_session_turns,
            think_time_ms=args.think_time_ms,
        ),
    )

    s3_spec = S3Spec.from_gbps(
        enabled=args.s3_enabled,
        up_gbps=args.s3_up_bw_gbps,
        down_gbps=args.s3_down_bw_gbps,
        eviction_time_ms=args.s3_eviction_time_ms,
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
        sla=args.sla,
    )

    # CLI-only detail: break out S3 storage vs transfer costs.
    print("\n--- Cost breakdown ---")
    print(
        f"  S3 transfer cost/hour: ${result.s3_cost_usd_per_hour - result.s3_storage_cost_usd_per_hour:.4f}"
    )
    print(f"  S3 storage cost/hour:  ${result.s3_storage_cost_usd_per_hour:.4f}")
    print(f"  S3 total cost/hour:    ${result.s3_cost_usd_per_hour:.4f}")

    # Print compact JSON for piping
    import json

    print("\n--- JSON ---")
    print(json.dumps(result.to_dict()))


if __name__ == "__main__":
    main()
