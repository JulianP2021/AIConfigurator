#!/usr/bin/env python3
"""Derive per-GPU-family component prices from AWS instance configs.

For each GPU family the script assigns GPU compute to a fixed fraction of the
AWS on-demand price (default 60%) after subtracting known fixed component costs
(RAM, SSD, inter-node bandwidth, NVLink bandwidth).  The remaining components
(PCIe bandwidth and internet bandwidth) are priced at fixed unit rates rather
than being fitted.

The output is a ``_pricing.gpu_family_pricing`` table used by custom and focused
machine presets to scale component by component under the same coefficients.

Usage:

    .venv/bin/python scripts/derive_family_pricing.py

This updates ``src/hardware/data/pricing.json`` in place.
"""

import argparse
import json
import sys

from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hardware.scraper import load_aws_hardware_db


_GB = 1024**3
_G = 1e9

# Fixed component prices considered ground truth.
_RAM_PRICE = 0.005
_SSD_PRICE = 0.000125
_INTER_NODE_UP_PRICE = 0.006
_INTER_NODE_DOWN_PRICE = 0.006
_NVLINK_BW_PRICE = 0.003
_PCIE_BW_PRICE = 0.004
_INET_UP_PRICE = 0.2
_INET_DOWN_PRICE = 0.2


def _fixed_cost(cfg: dict) -> float:
    """Return the total fixed-component cost for one machine config."""
    ram = (cfg["cpu_ram"] / _GB) * _RAM_PRICE
    ssd = (cfg["nvme_mem"] / _GB) * _SSD_PRICE
    inter_node = (
        cfg.get("network_inter_node_up", 100 * _G) / _G * _INTER_NODE_UP_PRICE
        + cfg.get("network_inter_node_down", 100 * _G) / _G * _INTER_NODE_DOWN_PRICE
    )
    nvlink = (cfg.get("nvlink_bw", 0) / _GB) * cfg["num_gpus"] * _NVLINK_BW_PRICE
    pcie = (cfg["pcie_bw"] / _GB) * _PCIE_BW_PRICE
    inet = (cfg["network_inet_up"] / _G) * _INET_UP_PRICE
    inet += (cfg["network_inet_down"] / _G) * _INET_DOWN_PRICE
    return ram + ssd + inter_node + nvlink + pcie + inet


def _derive_family(
    configs: list[dict], compute_fraction: float = 0.6
) -> dict[str, float]:
    """Return the per-family component price table.

    GPU compute is priced as ``compute_fraction`` of the average per-GPU AWS
    price.  All other components use the fixed unit prices above and are
    charged on top of that compute cost.
    """
    compute_anchor = compute_fraction * (
        sum(c["dph_base"] / c["num_gpus"] for c in configs) / len(configs)
    )

    return {
        "compute_usd_per_gpu_hour": compute_anchor,
        "gpu_compute_fraction": compute_fraction,
    }


def _predict(cfg: dict, pricing: dict[str, float]) -> float:
    """Recompute an hourly price from a per-family pricing table."""
    price = cfg["num_gpus"] * pricing["compute_usd_per_gpu_hour"]
    price += _fixed_cost(cfg)
    return price


def _derive_all_family_pricing(
    machines: dict[str, dict],
    compute_fraction: float = 0.6,
) -> dict[str, dict[str, float]]:
    """Group AWS instances by GPU family and build component tables."""
    families: dict[str, list[dict]] = {}
    for name, cfg in machines.items():
        if not name.startswith("AWS"):
            continue
        family = cfg.get("gpu_name", "")
        if not family:
            continue
        families.setdefault(family, []).append(cfg)

    family_pricing: dict[str, dict[str, float]] = {}
    for family, configs in sorted(families.items()):
        pricing = _derive_family(configs, compute_fraction)
        family_pricing[family] = pricing

        max_err = 0.0
        for cfg in configs:
            predicted = _predict(cfg, pricing)
            if abs((predicted - cfg["dph_base"]) / cfg["dph_base"]) > abs(max_err):
                assert cfg["dph_base"]
                max_err = (predicted - cfg["dph_base"]) / cfg["dph_base"]

        print(
            f"{family}: configs={len(configs)}, "
            f"max_rel_err={max_err * 100:.4f}%, "
            f"compute=${pricing['compute_usd_per_gpu_hour']:.4f}/gpu/h"
        )

    return family_pricing


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive per-GPU-family component prices from AWS preset data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/hardware/data/aws_hardware.json"),
        help="Path to aws_hardware.json (default: src/hardware/data/aws_hardware.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/hardware/data/pricing.json"),
        help="Output path (default: src/hardware/data/pricing.json).",
    )
    parser.add_argument(
        "--gpu-compute-fraction",
        type=float,
        default=0.6,
        help="Fraction of per-GPU AWS price attributed to GPU compute (default: 0.6).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    _pricing, machines = load_aws_hardware_db(args.input)
    if not machines:
        raise RuntimeError(f"No AWS machines loaded from {args.input}")

    family_pricing = _derive_all_family_pricing(
        machines, compute_fraction=args.gpu_compute_fraction
    )

    output = {
        "_pricing": {
            "cpu_ram_usd_per_gb_hour": _RAM_PRICE,
            "ssd_usd_per_gb_hour": _SSD_PRICE,
            "inter_node_up_usd_per_gbps_hour": _INTER_NODE_UP_PRICE,
            "inter_node_down_usd_per_gbps_hour": _INTER_NODE_DOWN_PRICE,
            "pcie_bw_usd_per_gb_s_hour": _PCIE_BW_PRICE,
            "nvlink_bw_usd_per_gb_s_hour": _NVLINK_BW_PRICE,
            "inet_up_usd_per_gbps_hour": _INET_UP_PRICE,
            "inet_down_usd_per_gbps_hour": _INET_DOWN_PRICE,
            "gpu_family_pricing": family_pricing,
            "gpu_compute_fraction": args.gpu_compute_fraction,
        }
    }

    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote per-family pricing to {args.output}")


if __name__ == "__main__":
    main()
