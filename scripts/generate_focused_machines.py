#!/usr/bin/env python3
"""Generate a focused set of custom machine presets for sensitivity analysis.

Given a fixed GPU family and a baseline node configuration, this script emits
one machine preset per focused dimension (e.g. RAM, NVLink bandwidth, SSD
memory, SSD bandwidth). The GPU count stays fixed across all machines; only the
focused dimension changes. All non-focused dimensions stay at the baseline
value, so each machine isolates a single hardware variable.

Example::

    .venv/bin/python scripts/generate_focused_machines.py H200 \
        --num-gpus 4 \
        --pcie-bw-gbps 128 \
        --nvlink-bw-gbps 800 \
        --ram-mem-gb 512 \
        --ssd-mem-gb 2048 \
        --inet-bw-gbps 25 \
        --focus ram,nvlink,ssd_mem,ssd_bw \
        --write --clean

This writes to ``src/hardware/custom_hardware.json`` by default. Use
``--custom-hardware`` to write to a different file.
"""

import argparse
import json
import sys

from pathlib import Path
from typing import Any


# Allow importing ``src`` when the script is executed from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.add_custom_machine import (
    _NVLINK_CAPABLE_GPUS,
    _build_machine_config,
    _get_pricing,
    _machine_config_with_defaults,
    _variant_name,
)
from src.hardware.scraper import load_aws_hardware_db
from src.utils.utils import parse_float_list


_DEFAULT_FOCUS_VALUES: dict[str, list[float]] = {
    "ssd_mem": [16000.0, 32000.0],
    "inter_node_bw": [200.0, 1000.0],
    "inet_bw": [0.0, 400.0],
}


def _generate_focused_entries(
    gpu_name: str,
    base_name: str,
    baseline: dict[str, Any],
    focus_dimensions: list[str],
    focus_values: dict[str, list[float]],
    pricing_path: Path,
    *,
    gpu_compute_fraction: float = 0.6,
) -> dict[str, dict[str, Any]]:
    """Build machine entries, one sweep per focused dimension."""
    pricing = _get_pricing(pricing_path)
    keys = [
        "cpu_ram_usd_per_gb_hour",
        "ssd_usd_per_gb_hour",
        "pcie_bw_usd_per_gb_s_hour",
        "nvlink_bw_usd_per_gb_s_hour",
        "inter_node_up_usd_per_gbps_hour",
        "inter_node_down_usd_per_gbps_hour",
        "inet_up_usd_per_gbps_hour",
        "inet_down_usd_per_gbps_hour",
    ]
    for key in keys:
        assert pricing.get(key, 0) > 0, (
            f"{key} not found in pricing, pricing: {pricing}"
        )
    if gpu_name not in _NVLINK_CAPABLE_GPUS and baseline["nvlink_bw_gbps"] > 0:
        print(
            f"Warning: {gpu_name!r} is not in the known NVLink/C2C-capable GPU set. "
            "The supplied NVLink bandwidth will still be stored and priced as a custom interconnect.",
            file=sys.stderr,
        )

    # Validate that the requested compute split matches the pricing table.
    table_fraction = pricing.get("gpu_compute_fraction", 0.6)
    if abs(gpu_compute_fraction - table_fraction) > 1e-6:
        raise ValueError(
            f"Requested GPU compute fraction {gpu_compute_fraction} does not match "
            f"the pricing table's split {table_fraction}. "
            f"Regenerate pricing with the desired split, e.g. "
            f".venv/bin/python scripts/derive_family_pricing.py --gpu-compute-fraction {gpu_compute_fraction}"
        )

    entries: dict[str, dict[str, Any]] = {}

    def _add_entry(settings: dict[str, Any]) -> None:
        settings["machine_name"] = _variant_name(
            settings.get("base_name", base_name), settings
        )
        raw_config = _build_machine_config(settings)
        final_config = _machine_config_with_defaults(
            raw_config, pricing, compute_price_fraction=gpu_compute_fraction
        )
        entries[settings["machine_name"]] = final_config

    # Always include the baseline machine first.
    baseline_scalar = {
        **baseline,
        "gpu_name": gpu_name,
    }
    _add_entry({**baseline_scalar})

    for dim in focus_dimensions:
        values = focus_values.get(dim, [])
        for value in values:
            settings: dict[str, Any] = {**baseline_scalar}
            if dim == "ram":
                settings["ram_mem_gb"] = value
                name_suffix = f" RAM {value:.0f}GB"
            elif dim == "nvlink":
                settings["nvlink_bw_gbps"] = value
                name_suffix = f" NVLink {value:.0f}GBps"
            elif dim == "ssd_mem":
                settings["ssd_mem_gb"] = value
                name_suffix = f" SSD {value:.0f}GB"
            elif dim == "ssd_bw":
                settings["ssd_bw_gbps"] = value
                name_suffix = f" SSD BW {value:.1f}GBps"
            elif dim == "ssd_mem":
                # Re-derive bandwidth for the new capacity unless the user is
                # explicitly sweeping SSD bandwidth at the same time.
                if "ssd_bw" not in focus_dimensions:
                    settings["ssd_bw_gbps"] = (value / 4096.0) * 8.0
            elif dim == "inter_node_bw":
                settings["inter_node_up_gbps"] = value
                settings["inter_node_down_gbps"] = value
                name_suffix = f" INTER NODE BW {value:.1f}Gbps"
            elif dim == "inet_bw":
                settings["inet_bw_gbps"] = value
                name_suffix = f" INET BW {value:.1f}Gbps"
            else:
                continue

            # Avoid duplicating the baseline entry.
            if all(
                settings[k] == baseline_scalar[k]
                for k in (
                    "num_gpus",
                    "ram_mem_gb",
                    "nvlink_bw_gbps",
                    "ssd_mem_gb",
                    "ssd_bw_gbps",
                    "inter_node_up_gbps",
                    "inter_node_down_gbps",
                    "inet_bw_gbps",
                )
            ):
                continue

            # Use a per-focus base name so labels are readable.
            focus_base = f"{base_name}{name_suffix}"
            settings["base_name"] = focus_base
            _add_entry(settings)

    return entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate focused custom machine presets for sensitivity analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Bandwidth values are interpreted as Gbps; memory values as GB.",
    )
    parser.add_argument(
        "gpu_name", help="GPU family name as it appears in _gpu_db.json."
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default=None,
        help="Base name prefix for generated machines (default: Focused <gpu_name>).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs per node; fixed across all generated machines (default: 4).",
    )
    parser.add_argument(
        "--pcie-bw-gbps",
        type=float,
        default=128.0,
        help="Baseline PCIe bandwidth in GBps (default: 128).",
    )
    parser.add_argument(
        "--nvlink-bw-gbps",
        type=float,
        default=800.0,
        help="Baseline NVLink bandwidth in GBps (default: 800).",
    )
    parser.add_argument(
        "--ram-mem-gb",
        type=float,
        default=512.0,
        help="Baseline CPU RAM size in GB (default: 512).",
    )
    parser.add_argument(
        "--ssd-mem-gb",
        type=float,
        default=2048.0,
        help="Baseline local SSD size in GB (default: 2048).",
    )
    parser.add_argument(
        "--ssd-bw-gbps",
        type=float,
        default=None,
        help="Baseline SSD (NVMe) bandwidth in GBps. Defaults to 8 GB/s per 4 TB of --ssd-mem-gb.",
    )
    parser.add_argument(
        "--inter-node-bw-gbps",
        type=float,
        default=400.0,
        help="Baseline INTER NODE bandwidth in Gbps (default: 400.0).",
    )
    parser.add_argument(
        "--inet-bw-gbps",
        type=float,
        default=200.0,
        help="Symmetric internet up/down bandwidth in Gbps (default: 200).",
    )
    parser.add_argument(
        "--focus",
        type=str,
        default="ram,nvlink,ssd_mem,ssd_bw,inter_node_bw,inet_bw",
        help="Comma-separated list of dimensions to sweep (default: ram,nvlink,ssd_mem,ssd_bw,inter_node_bw,inet_bw).",
    )
    parser.add_argument(
        "--focus-values-ram",
        type=str,
        default=None,
        help="Override RAM focus sweep in GB (comma-separated).",
    )
    parser.add_argument(
        "--focus-values-nvlink",
        type=str,
        default=None,
        help="Override NVLink focus sweep in GBps (comma-separated).",
    )
    parser.add_argument(
        "--focus-values-ssd-mem",
        type=str,
        default=None,
        help="Override SSD memory focus sweep in GB (comma-separated).",
    )
    parser.add_argument(
        "--focus-values-inter-node-bw",
        type=str,
        default=None,
        help="Override INTER NODE bandwidth focus sweep in GBps (comma-separated).",
    )
    parser.add_argument(
        "--focus-values-inet-bw",
        type=str,
        default=None,
        help="Override INET bandwidth focus sweep in Gbps (comma-separated).",
    )
    parser.add_argument(
        "--custom-hardware",
        type=Path,
        default=Path("src/hardware/data/custom_hardware.json"),
        help="Path to the custom hardware JSON file (default: src/hardware/data/custom_hardware.json).",
    )
    parser.add_argument(
        "--hardware-prices",
        type=Path,
        default=Path("src/hardware/data/pricing.json"),
        help="Path to the custom hardware JSON file (default: src/hardware/data/pricing.json).",
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

    base_name = args.base_name or f"Focused {args.gpu_name}"

    focus_dimensions = [d.strip().lower() for d in args.focus.split(",")]
    allowed = {"ram", "nvlink", "ssd_mem", "ssd_bw", "inter_node_bw", "inet_bw"}
    invalid = set(focus_dimensions) - allowed
    if invalid:
        print(
            f"Invalid focus dimension(s): {sorted(invalid)}. Allowed: {sorted(allowed)}",
            file=sys.stderr,
        )
        return 1

    focus_values: dict[str, list[float]] = {}
    if args.focus_values_ram is not None:
        focus_values["ram"] = parse_float_list(args.focus_values_ram)
    if args.focus_values_nvlink is not None:
        focus_values["nvlink"] = parse_float_list(args.focus_values_nvlink)
    if args.focus_values_ssd_mem is not None:
        focus_values["ssd_mem"] = parse_float_list(args.focus_values_ssd_mem)
    if args.focus_values_inter_node_bw is not None:
        focus_values["inter_node_bw"] = parse_float_list(
            args.focus_values_inter_node_bw
        )
    if args.focus_values_inet_bw is not None:
        focus_values["inet_bw"] = parse_float_list(args.focus_values_inet_bw)

    for dim in ("ram", "nvlink", "ssd_mem", "inter_node_bw", "inet_bw"):
        if dim in focus_dimensions and dim not in focus_values:
            if dim == "nvlink":
                pcie = args.pcie_bw_gbps
                focus_values[dim] = [pcie * 2]
            elif dim == "ram":
                ram = args.ram_mem_gb
                focus_values[dim] = [ram / 2, ram]
            else:
                focus_values[dim] = _DEFAULT_FOCUS_VALUES[dim]

    ssd_bw_gbps = args.ssd_bw_gbps
    if ssd_bw_gbps is None:
        ssd_bw_gbps = (args.ssd_mem_gb / 4096.0) * 8.0

    baseline = {
        "num_gpus": args.num_gpus,
        "pcie_bw_gbps": args.pcie_bw_gbps,
        "nvlink_bw_gbps": args.nvlink_bw_gbps,
        "ram_mem_gb": args.ram_mem_gb,
        "ssd_mem_gb": args.ssd_mem_gb,
        "ssd_bw_gbps": ssd_bw_gbps,
        "inet_bw_gbps": args.inet_bw_gbps,
        "inet_up_gbps": args.inet_bw_gbps,
        "inet_down_gbps": args.inet_bw_gbps,
        "inter_node_up_gbps": args.inter_node_bw_gbps,
        "inter_node_down_gbps": args.inter_node_bw_gbps,
    }

    pricing = _get_pricing(args.hardware_prices)
    gpu_compute_fraction = args.gpu_compute_fraction
    if gpu_compute_fraction is None:
        gpu_compute_fraction = float(pricing.get("gpu_compute_fraction", 0.6))

    entries = _generate_focused_entries(
        args.gpu_name,
        base_name,
        baseline,
        focus_dimensions,
        focus_values,
        args.hardware_prices,
        gpu_compute_fraction=gpu_compute_fraction,
    )

    if not entries:
        print("No machine presets generated.", file=sys.stderr)
        return 1

    print(f"Generated {len(entries)} focused machine preset(s):\n")
    for name, cfg in sorted(entries.items()):
        print(f"  {name} -> ${cfg.get('dph_base', 0.0):.4f}/h")

    if not args.write:
        print("\nDry run; entries not written. Pass --write to persist them.")
        return 0

    if args.clean:
        output_data = {"machines": entries}
        print(
            f"\nReplacing all existing machines in {args.custom_hardware} with {len(entries)} generated entries."
        )
    else:
        _, existing_machines = load_aws_hardware_db(args.custom_hardware)
        output_data = {
            "machines": {**existing_machines, **entries},
        }

    args.custom_hardware.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {args.custom_hardware}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
