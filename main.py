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

import os

from src.hardware.hardware import GPUHardwareSpec, Hardware, S3Spec
from src.hardware.mixed_gpu import fetch_mixed_gpu_hardware
from src.hardware.scraper import (
    fetch_machine_hardware,
    load_combined_machine_db,
    resolve_machine_name,
)
from src.logger import set_debug, set_log_mask
from src.node.node import Node
from src.request.request import RequestScenario, TokenDistribution
from src.router.router import RouterCostConfig
from src.simulations.simulation_distributed import (
    DistributedScenario,
    simulate_run_distributed,
)
from src.utils.env_reader import load_env
from src.utils.output_filter import compact_json
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

    # Inter-node bandwidth can be overridden from CLI/env; set it globally so
    # fetch_machine_hardware() applies it to the constructed HardwareSpec.
    os.environ["INTER_NODE_NETWORK_UP_GBPS"] = str(args.inter_node_network_up_gbps)
    os.environ["INTER_NODE_NETWORK_DOWN_GBPS"] = str(args.inter_node_network_down_gbps)

    mixed_mode = args.mixed
    if mixed_mode and not args.mixed_gpu_donor:
        # Mixed mode is a colocated topology with different GPU types for
        # prefill and decode.  A donor machine must be supplied.
        raise ValueError("--mixed requires --mixed-gpu-donor to be set.")

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
    if args.colocated or mixed_mode:
        if prefill_gpus_per_node >= total_gpus:
            raise ValueError(
                f"--prefill-gpus-per-node ({prefill_gpus_per_node}) must be "
                f"less than total GPUs per node ({total_gpus}) in colocated/mixed mode."
            )
        decode_gpus_per_node = total_gpus - prefill_gpus_per_node
        decode_gpu_spec: GPUHardwareSpec | None = None

        if mixed_mode:
            mixed_gpu_count = (
                args.mixed_gpu_count
                if args.mixed_gpu_count >= 0
                else decode_gpus_per_node
            )
            donor_hw_name = resolve_machine_name(args.mixed_gpu_donor)
            hardware = fetch_mixed_gpu_hardware(
                args.machine_hardware,
                prefill_gpus_per_node,
                donor_hw_name,
                mixed_gpu_count,
            )
            donor_gpu_name = load_combined_machine_db()[donor_hw_name]["gpu_name"]
            from src.hardware.scraper import lookup as lookup_gpu

            donor_gpu_config = lookup_gpu(donor_gpu_name)
            decode_gpu_spec = GPUHardwareSpec(
                flops=donor_gpu_config["flops"],
                gpu_mem=donor_gpu_config["gpu_mem"],
                gpu_bw=donor_gpu_config["gpu_bw"],
            )
            print(
                f"Mixed-GPU mode: {args.num_prefill_nodes} node(s), each with "
                f"{prefill_gpus_per_node} prefill GPU(s) from {args.machine_hardware} + "
                f"{mixed_gpu_count} decode GPU(s) from {donor_hw_name}."
            )
        else:
            print(
                f"Colocated mode: {args.num_prefill_nodes} node(s), each with "
                f"{prefill_gpus_per_node} prefill + {decode_gpus_per_node} decode GPU(s)."
            )

        for _ in range(args.num_prefill_nodes):
            nodes.append(
                Node(
                    hardware=hardware,
                    model_name=args.model,
                    batch_size=args.batch_size,
                    prefill_instances=prefill_gpus_per_node,
                    decode_instances=decode_gpus_per_node,
                    decode_gpu_hardware=decode_gpu_spec,
                )
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
        active_work_scale=args.router_active_work_scale,
        device_credit=args.router_device_credit,
        remote_ram_credit=args.router_remote_ram_credit,
        remote_ssd_credit=args.router_remote_ssd_credit,
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
        user_delay_fraction=args.user_delay_fraction,
        user_delay_min_ms=args.user_delay_min_ms,
        user_delay_max_ms=args.user_delay_max_ms,
        random_seed=args.random_seed,
    )

    # CLI-only: average prompt size accounting for cumulative ISL growth
    # across session turns.  The first turn has ISL tokens; each subsequent
    # turn starts from the previous turn's (ISL + OSL) and adds another ISL.
    turns = args.max_session_turns
    avg_input_tokens = args.isl * (turns + 1) / 2.0 + args.osl * (turns - 1) / 2.0
    avg_total_tokens = avg_input_tokens + args.osl
    print("\n--- Request shape ---")
    print(f"  Sessions per user:    {args.sessions_per_user}")
    print(f"  Max session turns:    {turns}")
    print(f"  Avg input tokens:     {avg_input_tokens:,.0f}")
    print(f"  Avg output tokens:    {args.osl}")
    print(f"  Avg tokens / prompt:  {avg_total_tokens:,.0f}")

    # CLI-only detail: break out S3 storage vs transfer costs.
    print("\n--- Cost breakdown ---")
    print(
        f"  S3 transfer cost/hour: ${result.s3_cost_usd_per_hour - result.s3_storage_cost_usd_per_hour:.4f}"
    )
    print(f"  S3 storage cost/hour:  ${result.s3_storage_cost_usd_per_hour:.4f}")
    print(f"  S3 total cost/hour:    ${result.s3_cost_usd_per_hour:.4f}")

    # Diagnostic counters are only logged here in the CLI, never in the
    # webserver or execute_config.py output.
    print("\n--- KV cache read counters ---")
    print(f"  RAM download requests:  {result.ram_download_requests}")
    print(f"  SSD download requests:  {result.ssd_download_requests}")
    print(f"  S3 upload requests:     {result.s3_upload_requests}")
    print(f"  S3 download requests:   {result.s3_download_requests}")

    # Print compact JSON for piping (full result is still exported via files / webserver)
    print("\n--- JSON ---")
    print(compact_json(result.to_dict()))


if __name__ == "__main__":
    main()
