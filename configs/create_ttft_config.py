#!/usr/bin/env python3
"""Generate fixed-topology configs for TTFT benchmark sweeps.

This script keeps the topology dimensions fixed and selects exactly one config
family via ``--config-type``:

- ``colocated``: prefill and decode share the same nodes.
- ``mixed``: colocated nodes with different prefill/decode GPU types.
- ``separate``: distinct prefill-only and decode-only nodes.

The generated config JSON can then be fed into ``execute_ttft_config.py``.
"""

from __future__ import annotations
import argparse
import json
import sys

from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.utils.config_utils import build_base_config, get_focus
from src.hardware.scraper import (
    load_aws_hardware_db,
    load_gpu_db,
    parse_gpu_count,
)
from src.utils.env_reader import load_env
from src.utils.parser import _base_parser


_HI_END_GPU_NAMES = {"H100", "H200", "B200", "B300", "A100", "INF1", "INF2"}


def _parse_config_type(value: str) -> str:
    config_type = value.strip().lower()
    allowed = {"colocated", "mixed", "separate"}
    if config_type not in allowed:
        raise argparse.ArgumentTypeError(
            f"Invalid config type '{value}'. Allowed: {sorted(allowed)}"
        )
    return config_type

def _machine_candidates(
    machine_db: dict[str, dict[str, Any]],
    gpu_db: dict[str, dict[str, Any]],
    high_end_only: bool,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for machine_name, machine in machine_db.items():
        if machine["gpu_name"] not in gpu_db:
            continue
        if high_end_only and all(
            token not in machine["gpu_name"] for token in _HI_END_GPU_NAMES
        ):
            continue
        candidates.append((machine_name, machine["gpu_name"]))
    return sorted(candidates)


def _generate_configs(
    args: argparse.Namespace,
    machine_db: dict[str, dict[str, Any]],
    gpu_db: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    config_type = args.config_type
    base = build_base_config(args, config_type)
    candidates = _machine_candidates(machine_db, gpu_db, False)

    if not candidates:
        raise ValueError("No candidate machines matched the selected filters")

    configs: list[dict[str, Any]] = []

    if config_type == "colocated":
        if args.prefill_nodes != args.decode_nodes:
            raise ValueError("Colocated configs require prefill_nodes == decode_nodes")
        for machine_name, gpu_name in candidates:
            total_gpus = int(parse_gpu_count(machine_name))
            decode_gpus_per_node = total_gpus - args.prefill_gpus_per_node

            assert decode_gpus_per_node > 0
            assert decode_gpus_per_node < total_gpus

            focus, focus_value = get_focus(machine_name, gpu_name)
            configs.append({
                "label": (f"{config_type.title()}: {machine_name}"),
                "gpu": gpu_name,
                "prefill_hardware": machine_name,
                "decode_hardware": machine_name,
                "prefill_nodes": args.prefill_nodes,
                "decode_nodes": args.decode_nodes,
                "prefill_gpus_per_node": args.prefill_gpus_per_node,
                "decode_gpus_per_node": decode_gpus_per_node,
                "batch_size": args.batch_size,
                "colocated": True,
                "config_type": config_type,
                "focus": focus,
                "focus_value": focus_value,
            })
    # elif config_type == "mixed":
    # if not args.mixed_gpu_donor:
    #     raise ValueError("--mixed-gpu-donor is required for mixed configs")
    # donor_name = args.mixed_gpu_donor
    # donor_total = int(parse_gpu_count(donor_name))
    # for machine_name, gpu_name in candidates:
    #     total_gpus = int(parse_gpu_count(machine_name))
    #     if args.prefill_gpus_per_node + args.decode_gpus_per_node != total_gpus:
    #         continue
    #     if args.decode_gpus_per_node > donor_total:
    #         continue
    #     focus, focus_value = _build_focus_metadata(
    #         machine_name, gpu_name
    #     )
    #     configs.append(
    #         {
    #             "label": (
    #                 f"{config_type.title()}: {machine_name} + {donor_name} - {args.prefill_nodes} nodes - batch {args.batch_size}"
    #             ),
    #             "prefill_hardware": machine_name,
    #             "decode_hardware": donor_name,
    #             "prefill_nodes": args.prefill_nodes,
    #             "decode_nodes": args.decode_nodes,
    #             "prefill_gpus_per_node": args.prefill_gpus_per_node,
    #             "decode_gpus_per_node": args.decode_gpus_per_node,
    #             "mixed": True,
    #             "mixed_gpu_donor": donor_name,
    #             "batch_size": args.batch_size,
    #             "colocated": True,
    #             "config_type": config_type,
    #             "focus": focus,
    #             "focus_value": focus_value,
    #         }
    #     )
    else:
        # for prefill_name,  in candidates:
        #     for decode_name in candidates:
        #         focus, focus_value = _build_focus_metadata(
        #             machine_name, gpu_name
        #         )
        #         configs.append(
        #             {
        #                 "label": (
        #                     f"{config_type.title()}: {prefill_name} -> {decode_name} - {args.prefill_nodes}/{args.decode_nodes} nodes - batch {args.batch_size}"
        #                 ),
        #                 "prefill_hardware": prefill_name,
        #                 "decode_hardware": decode_name,
        #                 "prefill_nodes": args.prefill_nodes,
        #                 "decode_nodes": args.decode_nodes,
        #                 "prefill_gpus_per_node": args.prefill_gpus_per_node,
        #                 "decode_gpus_per_node": args.decode_gpus_per_node,
        #                 "batch_size": args.batch_size,
        #                 "colocated": False,
        #                 "config_type": config_type,
        #                 "focus": focus,
        #                 "focus_value": focus_value,
        #             }
        #         )
        raise RuntimeError("Separated/Mixed not supported yet")

    base["configs"] = configs
    return base


def main() -> None:
    env = load_env()
    parser = _base_parser(env)
    parser.add_argument("--config-name", type=Path, default=Path("config.json"))
    parser.add_argument("--config-type", type=_parse_config_type, required=True)
    parser.add_argument("--prefill-nodes", type=int, default=env.num_prefill_nodes)
    parser.add_argument("--decode-nodes", type=int, default=env.num_decode_nodes)
    parser.add_argument(
        "--prefill-gpus-per-node", type=int, default=env.prefill_gpus_per_node
    )
    parser.add_argument("--batch-size", type=int, default=env.batch_size)
    parser.add_argument("--mixed-gpu-donor", type=str, default=None)
    parser.add_argument("--custom-hardware", type=str)
    args = parser.parse_args()

    if args.custom_hardware is not None:
        _, machine_db = load_aws_hardware_db(args.custom_hardware)
    else:
        _, machine_db = load_aws_hardware_db()

    gpu_db = load_gpu_db()

    if args.config_type in {"mixed"} and args.prefill_gpus_per_node <= 0:
        raise ValueError("--prefill-gpus-per-node must be set for mixed configs")
    if args.config_type == "separate" and args.prefill_gpus_per_node <= 0:
        raise ValueError("--prefill-gpus-per-node must be set for separate configs")
    payload = _generate_configs(args, machine_db, gpu_db)
    if not payload["configs"]:
        raise ValueError(f"No configs generated for config type '{args.config_type}'")

    with args.config_name.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=4)

    print(
        f"Generated {len(payload['configs'])} {args.config_type} config(s) into {args.config_name}"
    )


if __name__ == "__main__":
    main()
