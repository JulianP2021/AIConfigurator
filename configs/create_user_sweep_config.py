import sys

from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.utils.config_utils import build_base_config
from src.hardware.scraper import (
    load_aws_hardware_db,
    load_gpu_db,
    load_machine_db,
    parse_gpu_count,
)
from src.utils.env_reader import load_env
from src.utils.parser import get_create_config_parser


if __name__ == "__main__":
    env = load_env()
    parser = get_create_config_parser(env)
    args = parser.parse_args()

    if args.custom_hardware is not None:
        _, machine_db = load_aws_hardware_db(args.custom_hardware)
    elif args.legacy:
        machine_db = load_machine_db()
    else:
        _, machine_db = load_aws_hardware_db()

    gpu_db = load_gpu_db()

    print(args.sla)

    config = build_base_config(args, "USER")

    # High-end training GPUs to keep when --high-end-only is set.
    HIGH_END_GPUS = {
        "A100_40GB",
        "A100_80GB",
        "H100 NVL",
        "H200",
        "H200 NVL",
        "B200",
        "B300",
    }

    possible_machines: list[tuple[str, dict[str, Any]]] = []

    for machine_name, machine in machine_db.items():
        print(machine_name, " | ", machine["gpu_name"])
        if machine["gpu_name"] not in gpu_db:
            continue
        if args.high_end_only and machine["gpu_name"] not in HIGH_END_GPUS:
            print(
                f"Skipping {machine_name} because {machine['gpu_name']} is not a high-end GPU."
            )
            continue
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

    config_types = {t.strip().lower() for t in args.config_types.split(",")}
    allowed_types = {"colocated", "mixed", "separate"}
    invalid = config_types - allowed_types
    if invalid:
        print(
            f"Invalid config type(s): {sorted(invalid)}. "
            f"Allowed: {sorted(allowed_types)}",
            file=sys.stderr,
        )
        sys.exit(1)

    colocation_configs: list[dict[str, str]] = []
    mixed_configs: list[dict[str, str]] = []
    separate_configs: list[dict[str, str]] = []

    colocated_nodes_values = [1, 2, 4, 8, 12, 16]
    max_num_nodes = max(colocated_nodes_values)
    prefill_node_values = [1, 2, 4, 8, 12, 14]
    decode_node_values = [1, 2, 4, 8]
    batch_size_values = [64]

    mixed_gpu_donor_pool = sorted({name for name, _ in sorted_possible_machines})

    if "colocated" in config_types:
        for machine_name, prefill_machine in sorted_possible_machines:
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
                        colocation_configs.append(
                            {
                                "config_type": "colocated",
                                "prefill_hardware": machine_name,
                                "decode_hardware": machine_name,  # only so it passes, not used
                                "prefill_gpus_per_node": str(prefill_gpus_per_node),
                                "decode_gpus_per_node": str(decode_gpus_per_node),
                                "prefill_nodes": str(nodes),
                                "decode_nodes": str(
                                    nodes
                                ),  # only so it passes, not used
                                "batch_size": str(batch_size),
                                "label": f"Colocated: {machine_name} - {nodes} - {prefill_gpus_per_node} - batch {batch_size}",
                            },
                        )

    # Generate mixed-GPU configs as a separate category.
    if "mixed" in config_types:
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
                            mixed_configs.append(
                                {
                                    "config_type": "mixed",
                                    "prefill_hardware": machine_name,
                                    "decode_hardware": donor_name,
                                    "prefill_gpus_per_node": str(prefill_gpus_per_node),
                                    "decode_gpus_per_node": str(decode_gpus_per_node),
                                    "prefill_nodes": str(nodes),
                                    "decode_nodes": str(nodes),
                                    "batch_size": str(batch_size),
                                    "mixed_gpu_donor": donor_name,
                                    "label": f"Mixed: {machine_name} + {decode_gpus_per_node}x {donor_name} - {nodes} - batch {batch_size}",
                                },
                            )

    if "separate" in config_types:
        for prefill_machine_name, _ in sorted_possible_machines:
            for decode_machine_name, _ in sorted_possible_machines:
                for prefill_nodes in prefill_node_values:
                    for decode_nodes in decode_node_values:
                        if prefill_nodes + decode_nodes > max_num_nodes:
                            continue
                        for batch_size in batch_size_values:
                            separate_configs.append(
                                {
                                    "config_type": "separate",
                                    "prefill_hardware": prefill_machine_name,
                                    "decode_hardware": decode_machine_name,
                                    "prefill_nodes": str(prefill_nodes),
                                    "decode_nodes": str(decode_nodes),
                                    "batch_size": str(batch_size),
                                    "label": f"Separate: {prefill_machine_name} - {decode_machine_name} - {prefill_nodes} - {decode_nodes} - batch {batch_size}",
                                },
                            )

    print(
        f"Generated {len(colocation_configs)} colocation configs, {len(mixed_configs)} mixed configs, and {len(separate_configs)} separate configs for {len(possible_machines)} possible machines (types: {','.join(sorted(config_types))})."
    )

    config["configs"] = colocation_configs + mixed_configs + separate_configs

    with Path(args.config_name).open("w", encoding="utf-8") as f:
        import json

        json.dump(config, f, indent=4)
