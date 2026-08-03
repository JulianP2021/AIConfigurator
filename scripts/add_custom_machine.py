#!/usr/bin/env python3
"""Add custom machine presets to the local hardware database with derived pricing.

The script looks up the GPU name in the local GPU database and derives hourly
prices using per-family component costs from ``src/hardware/custom_hardware.json``
(falling back to ``src/hardware/aws_hardware.json`` for pricing metadata when the
custom file does not yet exist). If a component price is missing for a GPU
family, the global fallback price from the same file is used.

Every numeric option accepts a comma-separated list of values. The script
computes the Cartesian product of all lists and emits one machine preset per
combination. By default it runs in dry-run mode; use ``--write`` to persist the
entries.

Usage example::

    .venv/bin/python scripts/add_custom_machine.py "My H200" H200 \
        --num-gpus 1,2,4 \
        --pcie-bw-gbps 128,256 \
        --nvlink-bw-gbps 0,900 \
        --ram-mem-gb 256,512 \
        --ssd-mem-gb 2048,4096 \
        --ssd-bw-gbps 12.8,25 \
        --inet-bw-gbps 25 \
        --write

This writes to ``src/hardware/custom_hardware.json`` by default. Use
``--custom-hardware`` to write to a different file.
"""

import argparse
import itertools
import json
import sys

from pathlib import Path
from typing import Any


# Allow importing ``src`` when the script is executed from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hardware.scraper import (
    _machine_config_with_defaults,
    _resolve_inter_node_bw,
    load_aws_hardware_db,
    load_gpu_db,
)
from src.utils.utils import parse_float_list, parse_int_list


_GB = 1000**3

# GPUs known to expose NVLink/C2C-class die-to-die bandwidth. Used only for
# warnings when the user supplies --nvlink-bw-gbps for a GPU outside this set.
_NVLINK_CAPABLE_GPUS = {
    "TESLA_V100",
    "A100_40GB",
    "A100_80GB",
    "H100 NVL",
    "H200",
    "H200 NVL",
    "B200",
    "B300",
}


def _gb_to_bytes(gb: float) -> int:
    return int(gb * _GB)


def _derive_nvme_bw_from_capacity(ssd_mem_gb: float) -> int:
    """Return NVMe bandwidth in bytes/s from installed SSD capacity.

    Custom/focused machines use a simple rule of 8 GB/s for every 4 TB of SSD
    storage. This reflects the simulator's pricing model where SSD bandwidth is
    bundled with SSD capacity rather than priced separately.
    """
    return int((ssd_mem_gb / 4096.0) * 8.0 * _GB)


def _build_machine_config(settings: dict[str, Any]) -> dict:
    """Convert one combination of settings into a raw machine config dict."""
    inet_up_gbps = settings.get("inet_up_gbps", settings["inet_bw_gbps"])
    inet_down_gbps = settings.get("inet_down_gbps", settings["inet_bw_gbps"])

    config: dict[str, Any] = {
        "name": settings["machine_name"],
        "gpu_name": settings["gpu_name"],
        "num_gpus": settings["num_gpus"],
        "pcie_bw": _gb_to_bytes(settings["pcie_bw_gbps"]),
        "nvlink_bw": _gb_to_bytes(settings["nvlink_bw_gbps"]),
        "cpu_ram": _gb_to_bytes(settings["ram_mem_gb"]),
        "nvme_mem": _gb_to_bytes(settings["ssd_mem_gb"]),
        "nvme_bw": _derive_nvme_bw_from_capacity(settings["ssd_mem_gb"]),
        "network_inet_up": _gb_to_bytes(inet_up_gbps),
        "network_inet_down": _gb_to_bytes(inet_down_gbps),
        "network_inter_node_up": _resolve_inter_node_bw(
            str(settings["inter_node_up_gbps"])
        ),
        "network_inter_node_down": _resolve_inter_node_bw(
            str(settings["inter_node_down_gbps"])
        ),
    }

    return config


def _variant_name(base: str, settings: dict[str, Any]) -> str:
    """Build a deterministic, unique machine name for one combination."""
    slug = (
        f"x{settings['num_gpus']} "
        f"r{settings['ram_mem_gb']:.0f} "
        f"s{settings['ssd_mem_gb']:.0f} "
        f"p{settings['pcie_bw_gbps']:.0f}"
    )
    if settings["nvlink_bw_gbps"]:
        slug += f" nvl{settings['nvlink_bw_gbps']:.0f}"

    ssd_bw_gbps = _derive_nvme_bw_from_capacity(settings["ssd_mem_gb"]) / _GB
    slug += f" sbw{ssd_bw_gbps:.1f}"

    inter_node_up = settings.get("inter_node_up_gbps")
    inter_node_down = settings.get("inter_node_down_gbps")

    slug += f" in{inter_node_up:.1f}/{inter_node_down:.1f}"

    inet_up = settings.get("inet_up_gbps", settings["inet_bw_gbps"])
    inet_down = settings.get("inet_down_gbps", settings["inet_bw_gbps"])
    slug += f" inet{inet_up:.1f}/{inet_down:.1f}"
    return f"{base} {slug}"


def _get_pricing(custom_path: Path) -> dict:
    """Return pricing metadata, falling back to the default file if needed."""
    target_pricing, _ = load_aws_hardware_db(custom_path)
    if target_pricing:
        return target_pricing
    default_pricing, _ = load_aws_hardware_db()
    if not default_pricing:
        raise RuntimeError(
            "Could not load pricing metadata from "
            "src/hardware/aws_hardware.json. Run scripts/derive_family_pricing.py first."
        )
    return default_pricing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add custom machine presets with derived hourly pricing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Bandwidth values are interpreted as Gbps; memory values as GB. "
        "Each option may be a comma-separated list; the script emits the Cartesian product.",
    )
    parser.add_argument("machine_name", help="Base name for the new machine presets.")
    parser.add_argument(
        "gpu_name",
        help="GPU name(s) as they appear in _gpu_db.json. May be comma-separated.",
    )
    parser.add_argument(
        "--num-gpus",
        type=str,
        default="1",
        help="Number of GPUs per node. Comma-separated list accepted (default: 1).",
    )
    parser.add_argument(
        "--pcie-bw-gbps",
        type=str,
        required=True,
        help="PCIe bandwidth in Gbps. Comma-separated list accepted.",
    )
    parser.add_argument(
        "--nvlink-bw-gbps",
        type=str,
        default="0",
        help="NVLink bandwidth in Gbps. Comma-separated list accepted (default: 0).",
    )
    parser.add_argument(
        "--ram-mem-gb",
        type=str,
        required=True,
        help="CPU RAM size in GB. Comma-separated list accepted.",
    )
    parser.add_argument(
        "--ssd-mem-gb",
        type=str,
        required=True,
        help="Local SSD size in GB. Comma-separated list accepted.",
    )
    parser.add_argument(
        "--inet-bw-gbps",
        type=str,
        default="0",
        help="Symmetric internet up/down bandwidth in Gbps. Comma-separated list accepted (default: 0).",
    )
    parser.add_argument(
        "--inet-up-gbps",
        type=str,
        default=None,
        help="Internet upload bandwidth in Gbps (comma-separated; overrides --inet-bw-gbps).",
    )
    parser.add_argument(
        "--inet-down-gbps",
        type=str,
        default=None,
        help="Internet download bandwidth in Gbps (comma-separated; overrides --inet-bw-gbps).",
    )
    parser.add_argument(
        "--inter-node-up-gbps",
        type=str,
        default="100",
        help="Datacenter NIC upload bandwidth in Gbps. Comma-separated list accepted (default: 100).",
    )
    parser.add_argument(
        "--inter-node-down-gbps",
        type=str,
        default="100",
        help="Datacenter NIC download bandwidth in Gbps. Comma-separated list accepted (default: 100).",
    )
    parser.add_argument(
        "--custom-hardware",
        type=Path,
        default=Path("src/hardware/custom_hardware.json"),
        help="Path to the custom hardware JSON file (default: src/hardware/custom_hardware.json).",
    )
    parser.add_argument(
        "--gpu-compute-fraction",
        type=float,
        default=None,
        help="GPU compute fraction to validate against the pricing table (default: read from pricing.json).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write the entries to the custom hardware file. Without this flag the script runs in dry-run mode.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="When writing, replace all existing machines in the custom hardware file with the generated entries.",
    )
    args = parser.parse_args()

    gpu_names = [g.strip() for g in args.gpu_name.split(",")]

    # Validate GPU names in the local database.
    gpu_db = load_gpu_db()
    missing = [g for g in gpu_names if g not in gpu_db]
    if missing:
        print(
            f"GPU(s) not found in local database: {missing}. "
            f"Available: {sorted(gpu_db)}",
            file=sys.stderr,
        )
        return 1

    nvlink_bws = parse_float_list(args.nvlink_bw_gbps)
    for gpu, bw in itertools.product(gpu_names, nvlink_bws):
        if bw > 0 and gpu not in _NVLINK_CAPABLE_GPUS:
            print(
                f"Warning: {gpu!r} is not in the known NVLink/C2C-capable GPU set. "
                f"The supplied --nvlink-bw-gbps will still be stored and priced as "
                f"a custom interconnect.",
                file=sys.stderr,
            )

    pricing = _get_pricing(args.custom_hardware)
    family_pricing = pricing.get("gpu_family_pricing", {})
    missing_families = [g for g in gpu_names if g not in family_pricing]
    if missing_families:
        print(
            f"Error: no per-family pricing for {missing_families}. "
            "Add pricing entries under _pricing.gpu_family_pricing first.",
            file=sys.stderr,
        )
        return 1

    # Build dimension lists.
    dimensions: dict[str, list[Any]] = {
        "gpu_name": gpu_names,
        "num_gpus": parse_int_list(args.num_gpus),
        "pcie_bw_gbps": parse_float_list(args.pcie_bw_gbps),
        "nvlink_bw_gbps": parse_float_list(args.nvlink_bw_gbps),
        "ram_mem_gb": parse_float_list(args.ram_mem_gb),
        "ssd_mem_gb": parse_float_list(args.ssd_mem_gb),
        "inet_bw_gbps": parse_float_list(args.inet_bw_gbps),
        "inter_node_up_gbps": parse_float_list(args.inter_node_up_gbps),
        "inter_node_down_gbps": parse_float_list(args.inter_node_down_gbps),
    }

    if args.inet_up_gbps is not None:
        dimensions["inet_up_gbps"] = parse_float_list(args.inet_up_gbps)
    if args.inet_down_gbps is not None:
        dimensions["inet_down_gbps"] = parse_float_list(args.inet_down_gbps)

    gpu_compute_fraction = args.gpu_compute_fraction
    if gpu_compute_fraction is None:
        gpu_compute_fraction = float(pricing.get("gpu_compute_fraction", 0.6))

    keys = list(dimensions.keys())
    combinations = list(itertools.product(*(dimensions[k] for k in keys)))

    entries: dict[str, dict[str, Any]] = {}
    printed: list[dict[str, Any]] = []

    for combo in combinations:
        settings = dict(zip(keys, combo, strict=False))
        settings["machine_name"] = _variant_name(args.machine_name, settings)

        raw_config = _build_machine_config(settings)
        final_config = _machine_config_with_defaults(
            raw_config, pricing, compute_price_fraction=gpu_compute_fraction
        )
        entries[settings["machine_name"]] = final_config

        printed.append({
            "name": settings["machine_name"],
            "gpu": settings["gpu_name"],
            "num_gpus": settings["num_gpus"],
            "ram_gb": settings["ram_mem_gb"],
            "ssd_gb": settings["ssd_mem_gb"],
            "ssd_bw_gbps": _derive_nvme_bw_from_capacity(settings["ssd_mem_gb"]) / _GB,
            "pcie_gbps": settings["pcie_bw_gbps"],
            "nvlink_gbps": settings["nvlink_bw_gbps"],
            "inet_up_gbps": settings.get("inet_up_gbps", settings["inet_bw_gbps"]),
            "inet_down_gbps": settings.get("inet_down_gbps", settings["inet_bw_gbps"]),
            "inter_node_up_gbps": settings["inter_node_up_gbps"],
            "inter_node_down_gbps": settings["inter_node_down_gbps"],
            "price": final_config.get("dph_base", 0.0),
        })

    # Print summary.
    print(
        f"Generated {len(printed)} machine preset(s) from base name {args.machine_name!r}:\n"
    )
    for entry in printed:
        print(f"  {entry['name']}")
        print(f"    GPU:    {entry['gpu']} x{entry['num_gpus']}")
        print(f"    RAM:    {entry['ram_gb']:.1f} GB")
        print(f"    SSD:    {entry['ssd_gb']:.1f} GB @ {entry['ssd_bw_gbps']:.1f} Gbps")
        print(f"    PCIe:   {entry['pcie_gbps']:.1f} Gbps")
        print(f"    NVLink: {entry['nvlink_gbps']:.1f} Gbps")
        print(
            f"    Internet up/down: {entry['inet_up_gbps']:.1f} / {entry['inet_down_gbps']:.1f} Gbps"
        )
        print(
            f"    Inter-node up/down: {entry['inter_node_up_gbps']:.1f} / {entry['inter_node_down_gbps']:.1f} Gbps"
        )
        print(f"    Derived hourly price: ${entry['price']:.4f}/h")
        print()

    if not args.write:
        print("Dry run; entries not written. Pass --write to persist them.")
        if args.clean:
            print(
                "Note: --clean only affects write mode; dry-run output always shows only the generated entries."
            )
        print(json.dumps(entries, indent=2))
        return 0

    if args.clean:
        output_data = {
            "_pricing": pricing,
            "machines": entries,
        }
        print(
            f"Replacing all existing machines in {args.custom_hardware} with {len(entries)} generated entries."
        )
    else:
        _, existing_machines = load_aws_hardware_db(args.custom_hardware)
        output_data = {
            "_pricing": pricing,
            "machines": {**existing_machines, **entries},
        }

    args.custom_hardware.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {args.custom_hardware}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
