from pathlib import Path
from typing import Any

from src.hardware.scraper import load_gpu_db, load_machine_db, parse_gpu_count
from src.utils.env_reader import load_env
from src.utils.parser import get_create_config_parser


machine_db = load_machine_db()
gpu_db = load_gpu_db()

if __name__ == "__main__":
    env = load_env()
    parser = get_create_config_parser(env)
    args = parser.parse_args()

    config = {}
    config["model"] = args.model
    config["isl"] = args.isl
    config["osl"] = args.osl
    config["sessions_per_user"] = args.sessions_per_user
    config["users"] = args.users
    config["think_time_ms"] = args.think_time_ms
    config["max_session_turns"] = args.max_session_turns
    config["ram_usage_fraction"] = args.ram_usage_fraction
    config["ssd_usage_fraction"] = args.ssd_usage_fraction
    config["s3_enabled"] = args.s3_enabled
    config["s3_up_bw_gbps"] = args.s3_up_bw_gbps
    config["s3_down_bw_gbps"] = args.s3_down_bw_gbps
    config["s3_eviction_time_ms"] = args.s3_eviction_time_ms
    config["inter_node_network_up_gbps"] = args.inter_node_network_up_gbps
    config["inter_node_network_down_gbps"] = args.inter_node_network_down_gbps
    config["sla"] = args.sla
    config["user_delay_fraction"] = args.user_delay_fraction
    config["user_delay_min_ms"] = args.user_delay_min_ms
    config["user_delay_max_ms"] = args.user_delay_max_ms
    config["random_seed"] = args.random_seed
    config["batch_size"] = args.batch_size
    config["num_prefill_nodes"] = args.num_prefill_nodes
    config["num_decode_nodes"] = args.num_decode_nodes
    config["colocated"] = args.colocated
    config["prefill_gpus_per_node"] = args.prefill_gpus_per_node

    possible_machines: list[tuple[str, dict[str, Any]]] = []

    for machine_name, machine in machine_db.items():
        if machine["gpu_name"] in gpu_db:
            for already_considered_machine, _ in possible_machines:
                this_key = machine_name.split("x")
                old_key = already_considered_machine.split("x")
                if this_key[0] == old_key[0] and this_key[1][:1] == old_key[1][:1]:
                    print(
                        f"Skipping {machine_name} because {already_considered_machine} is already considered."
                    )
                    break
            else:
                possible_machines.append((machine_name, machine))

    sorted_possible_machines = sorted(possible_machines, key=lambda x: x[0])

    colocation_configs: list[dict[str, str]] = []
    mixed_configs: list[dict[str, str]] = []
    separate_configs: list[dict[str, str]] = []

    colocated_nodes_values = [1, 2, 4, 8, 12, 16]
    max_num_nodes = max(colocated_nodes_values)
    prefill_node_values = [1, 2, 4, 8, 12, 14]
    decode_node_values = [1, 2, 4, 8]
    batch_size_values = [64, 128]

    # For mixed-GPU colocated configs, use these donor GPU types.
    mixed_gpu_donor_pool = sorted({
        name
        for name, _ in sorted_possible_machines
        if "B200" in name or "H200" in name or "H100" in name or "RTX 4090" in name
    })

    for machine_name, prefill_machine in sorted_possible_machines:
        print(f"Machine: {machine_name}")
        for nodes in colocated_nodes_values:
            for batch_size in batch_size_values:
                if prefill_machine["num_gpus"] < 2:
                    continue
                prefill_gpus_per_node_values = []
                if prefill_machine["num_gpus"] == 2:
                    prefill_gpus_per_node_values = [1]
                if prefill_machine["num_gpus"] == 4:
                    prefill_gpus_per_node_values = [2, 3]
                if prefill_machine["num_gpus"] == 6:
                    prefill_gpus_per_node_values = [3, 4]
                if prefill_machine["num_gpus"] == 8:
                    prefill_gpus_per_node_values = [4, 6]

                for prefill_gpus_per_node in prefill_gpus_per_node_values:
                    decode_gpus_per_node = (
                        int(prefill_machine["num_gpus"]) - prefill_gpus_per_node
                    )
                    print(
                        f"label: Colocated: {machine_name} - {nodes} - {prefill_gpus_per_node}- batch {batch_size}"
                    )

                    colocation_configs.append(
                        {
                            "colocated": "true",
                            "prefill_hardware": machine_name,
                            "decode_hardware": machine_name,  # only so it passes, not used
                            "prefill_gpus_per_node": str(prefill_gpus_per_node),
                            "decode_gpus_per_node": str(decode_gpus_per_node),
                            "prefill_nodes": str(nodes),
                            "decode_nodes": str(nodes),  # only so it passes, not used
                            "batch_size": str(batch_size),
                            "label": f"Colocated: {machine_name} - {nodes} - {prefill_gpus_per_node}- batch {batch_size}",
                        },
                    )

    # Generate mixed-GPU configs as a separate category.
    for machine_name, prefill_machine in sorted_possible_machines:
        if prefill_machine["num_gpus"] < 2:
            continue
        prefill_gpus_per_node_values = []
        if prefill_machine["num_gpus"] == 2:
            prefill_gpus_per_node_values = [1]
        if prefill_machine["num_gpus"] == 4:
            prefill_gpus_per_node_values = [2, 3]
        if prefill_machine["num_gpus"] == 6:
            prefill_gpus_per_node_values = [3, 4]
        if prefill_machine["num_gpus"] == 8:
            prefill_gpus_per_node_values = [4, 6]

        for nodes in colocated_nodes_values:
            for batch_size in batch_size_values:
                for prefill_gpus_per_node in prefill_gpus_per_node_values:
                    decode_gpus_per_node = (
                        int(prefill_machine["num_gpus"]) - prefill_gpus_per_node
                    )
                    for donor_name in mixed_gpu_donor_pool:
                        if donor_name == machine_name:
                            continue
                        donor_total_gpus = parse_gpu_count(donor_name)
                        if donor_total_gpus < decode_gpus_per_node:
                            continue
                        print(
                            f"label: Mixed: {machine_name} + {decode_gpus_per_node}x {donor_name} - {nodes} - {prefill_gpus_per_node}- batch {batch_size}"
                        )
                        mixed_configs.append(
                            {
                                "mixed": "true",
                                "colocated": "false",
                                "prefill_hardware": machine_name,
                                "decode_hardware": donor_name,
                                "prefill_gpus_per_node": str(prefill_gpus_per_node),
                                "decode_gpus_per_node": str(decode_gpus_per_node),
                                "prefill_nodes": str(nodes),
                                "decode_nodes": str(nodes),
                                "batch_size": str(batch_size),
                                "mixed_gpu_donor": donor_name,
                                "label": f"Mixed: {machine_name} + {decode_gpus_per_node}x {donor_name} - {nodes} - {prefill_gpus_per_node}- batch {batch_size}",
                            },
                        )

    for prefill_machine_name, _ in sorted_possible_machines:
        for decode_machine_name, _ in sorted_possible_machines:
            for prefill_nodes in prefill_node_values:
                for decode_nodes in decode_node_values:
                    if prefill_nodes + decode_nodes > max_num_nodes:
                        continue
                    for batch_size in batch_size_values:
                        separate_configs.append(
                            {
                                "colocated": "false",
                                "prefill_hardware": prefill_machine_name,
                                "decode_hardware": decode_machine_name,
                                "prefill_nodes": str(prefill_nodes),
                                "decode_nodes": str(decode_nodes),
                                "batch_size": str(batch_size),
                                "label": f"separate: {prefill_machine_name} - {decode_machine_name} - {prefill_nodes} - {decode_nodes} - batch {batch_size}",
                            },
                        )

    print(
        f"Generated {len(colocation_configs)} colocation configs, {len(mixed_configs)} mixed configs, and {len(separate_configs)} separate configs."
    )

    config["configs"] = colocation_configs + mixed_configs + separate_configs

    with Path(args.config_name).open("w", encoding="utf-8") as f:
        import json

        json.dump(config, f, indent=4)
