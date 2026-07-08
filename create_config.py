from pathlib import Path
from typing import Any

from src.hardware.scraper import load_gpu_db, load_machine_db
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
    config["requests"] = args.requests
    config["users"] = args.users
    config["think_time_ms"] = args.think_time_ms
    config["max_session_turns"] = args.max_session_turns
    config["ram_usage_fraction"] = args.ram_usage_fraction
    config["ssd_usage_fraction"] = args.ssd_usage_fraction
    config["s3_enabled"] = args.s3_enabled
    config["s3_up_bw_gbps"] = args.s3_up_bw_gbps
    config["s3_down_bw_gbps"] = args.s3_down_bw_gbps
    config["sla"] = args.sla
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
    for machine_name, _ in sorted_possible_machines:
        print(f"Machine: {machine_name}")

    colocation_configs: list[dict[str, str]] = []
    single_node_configs: list[dict[str, str]] = []

    for prefill_machine_name, prefill_machine in sorted_possible_machines:
        for prefill_nodes in [1, 2, 4, 8]:
            for batch_size in [64, 128]:
                for prefill_gpus_per_node in [6, 4]:
                    decode_gpus_per_node = (
                        int(prefill_machine["num_gpus"]) - prefill_gpus_per_node
                    )
                    if decode_gpus_per_node < 1:
                        continue
                    colocation_configs.append(
                        {
                            "colocated": "true",
                            "prefill_hardware": prefill_machine_name,
                            "decode_hardware": prefill_machine_name,  # only so it passes, not used
                            "prefill_gpus_per_node": str(prefill_gpus_per_node),
                            "decode_gpus_per_node": str(decode_gpus_per_node),
                            "prefill_nodes": str(prefill_nodes),
                            "decode_nodes": str(prefill_nodes),
                            "batch_size": str(batch_size),
                            "label": f"Colocated: {prefill_machine_name} - {prefill_nodes} - {prefill_gpus_per_node}- batch {batch_size}",
                        },
                    )

        for decode_machine_name, _ in sorted_possible_machines:
            for prefill_nodes in [1, 2, 4, 8]:
                for decode_nodes in [1, 2, 4, 8]:
                    for batch_size in [64, 128]:
                        single_node_configs.append(
                            {
                                "colocated": "false",
                                "prefill_hardware": prefill_machine_name,
                                "decode_hardware": decode_machine_name,
                                "prefill_nodes": str(prefill_nodes),
                                "decode_nodes": str(decode_nodes),
                                "batch_size": str(batch_size),
                                "label": f"Single Node: {prefill_machine_name} - {decode_machine_name} - {prefill_nodes} - {decode_nodes} - batch {batch_size}",
                            },
                        )

    print(
        f"Generated {len(colocation_configs)} colocation configs and {len(single_node_configs)} single node configs."
    )

    config["configs"] = colocation_configs + single_node_configs

    with Path(args.config_name).open("w", encoding="utf-8") as f:
        import json

        json.dump(config, f, indent=4)
