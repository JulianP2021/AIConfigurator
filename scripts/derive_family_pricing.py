#!/usr/bin/env python3
"""Derive per-GPU-family component prices from AWS instance configs.

For each GPU family the script builds a linear system

    price_i = sum_j (quantity_ij * unit_price_j)

over all AWS instances of that family, then solves for the non-negative
component unit prices using ``scipy.optimize.nnls``.  CPU RAM and SSD storage
prices are anchored to fixed values (0.0089 USD/GB/h and 0.00037 USD/GB/h)
and removed from the regression so the remaining components are bandwidth and
per-GPU compute.

The output is a ``_pricing.gpu_family_pricing`` table that reproduces real
AWS on-demand prices for every instance in the input database.  Custom or
``focused'' machines with the same ``gpu_name`` then scale component by
component under the same family coefficients.

Usage:

    .venv/bin/python scripts/derive_family_pricing.py

This updates ``src/hardware/data/pricing.json`` in place.
"""

import argparse
import json
import sys

from pathlib import Path

import numpy as np

from scipy.optimize import nnls


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hardware.scraper import load_aws_hardware_db


_GB = 1024**3
_G = 1e9

# Fixed storage prices considered ground truth.
_RAM_PRICE = 0.0089
_SSD_PRICE = 0.00037

# Component price keys, in the same order as the design matrix columns.
_COMPONENT_KEYS = [
    "ssd_bw_usd_per_gb_s_hour",
    "nvlink_bw_usd_per_gb_s_hour",
    "pcie_bw_usd_per_gb_s_hour",
    "inter_node_up_usd_per_gbps_hour",
    "inter_node_down_usd_per_gbps_hour",
    "inet_up_usd_per_gbps_hour",
    "inet_down_usd_per_gbps_hour",
    "compute_usd_per_gpu_hour",
]

# Component quantities for one AWS config row.
_QUANTITY_EXTRACTORS = [
    lambda cfg: cfg["nvme_bw"] / _GB,
    lambda cfg: (cfg.get("nvlink_bw", 0) / _GB) * cfg["num_gpus"],
    lambda cfg: cfg["pcie_bw"] / _GB,
    lambda cfg: cfg.get("network_inter_node_up", 100 * _G) / _G,
    lambda cfg: cfg.get("network_inter_node_down", 100 * _G) / _G,
    lambda cfg: cfg["network_inet_up"] / _G,
    lambda cfg: cfg["network_inet_down"] / _G,
    lambda cfg: cfg["num_gpus"],
]


def _derive_family(
    configs: list[dict], compute_fraction: float = 0.6
) -> dict[str, float]:
    """Return the per-family component price table using non-negative LS.

    The per-GPU compute price is anchored to ``compute_fraction`` of the
    average per-GPU AWS price for the family (after subtracting RAM and SSD
    storage costs).  The remaining ``1 - compute_fraction`` is attributed to
    bandwidth components, which are derived by solving a non-negative
    least-squares problem against the residual prices.  This keeps compute
    prices physically meaningful while still reproducing AWS on-demand prices.

    ``compute_fraction`` is also stored in the table metadata so that focused
    and mixed-GPU pricing can validate they are using the same split.
    """
    adjusted_per_gpu = [
        (
            c["dph_base"]
            - (c["cpu_ram"] / _GB) * _RAM_PRICE
            - (c["nvme_mem"] / _GB) * _SSD_PRICE
        )
        / c["num_gpus"]
        for c in configs
    ]
    compute_anchor = compute_fraction * (sum(adjusted_per_gpu) / len(adjusted_per_gpu))

    a: list[list[float]] = []
    y: list[float] = []

    for cfg in configs:
        # All columns except the last (compute) one.
        row = [extract(cfg) for extract in _QUANTITY_EXTRACTORS[:-1]]
        residual = (
            cfg["dph_base"]
            - (cfg["cpu_ram"] / _GB) * _RAM_PRICE
            - (cfg["nvme_mem"] / _GB) * _SSD_PRICE
            - cfg["num_gpus"] * compute_anchor
        )
        a.append(row)
        y.append(residual)

    a_arr = np.array(a, dtype=float)
    y_arr = np.array(y, dtype=float)
    coeff, _ = nnls(a_arr, y_arr)

    result: dict[str, float] = {
        key: float(value)
        for key, value in zip(_COMPONENT_KEYS[:-1], coeff, strict=True)
    }
    result["compute_usd_per_gpu_hour"] = compute_anchor
    result["cpu_ram_usd_per_gb_hour"] = _RAM_PRICE
    result["ssd_usd_per_gb_hour"] = _SSD_PRICE
    result["gpu_compute_fraction"] = compute_fraction
    return result


def _predict(cfg: dict, pricing: dict[str, float]) -> float:
    """Recompute an hourly price from a per-family pricing table."""
    price = cfg["num_gpus"] * pricing["compute_usd_per_gpu_hour"]
    price += (cfg["cpu_ram"] / _GB) * pricing["cpu_ram_usd_per_gb_hour"]
    price += (cfg["nvme_mem"] / _GB) * pricing["ssd_usd_per_gb_hour"]
    price += (cfg["nvme_bw"] / _GB) * pricing["ssd_bw_usd_per_gb_s_hour"]
    price += (cfg.get("nvlink_bw", 0) / _GB * cfg["num_gpus"]) * pricing[
        "nvlink_bw_usd_per_gb_s_hour"
    ]
    price += (cfg["pcie_bw"] / _GB) * pricing["pcie_bw_usd_per_gb_s_hour"]
    price += (cfg.get("network_inter_node_up", 100 * _G) / _G) * pricing[
        "inter_node_up_usd_per_gbps_hour"
    ]
    price += (cfg.get("network_inter_node_down", 100 * _G) / _G) * pricing[
        "inter_node_down_usd_per_gbps_hour"
    ]
    price += (cfg["network_inet_up"] / _G) * pricing["inet_up_usd_per_gbps_hour"]
    price += (cfg["network_inet_down"] / _G) * pricing["inet_down_usd_per_gbps_hour"]
    return price


def _derive_all_family_pricing(
    machines: dict[str, dict],
    compute_fraction: float = 0.6,
) -> dict[str, dict[str, float]]:
    """Group AWS instances by GPU family and solve for component prices."""
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

        max_abs_err = 0.0
        for cfg in configs:
            predicted = _predict(cfg, pricing)
            err = abs(predicted - cfg["dph_base"])
            max_abs_err = max(max_abs_err, err)

        print(
            f"{family}: configs={len(configs)}, "
            f"max_abs_err=${max_abs_err:.4f}/h, "
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

    pricing, machines = load_aws_hardware_db(args.input)
    if not machines:
        raise RuntimeError(f"No AWS machines loaded from {args.input}")

    family_pricing = _derive_all_family_pricing(
        machines, compute_fraction=args.gpu_compute_fraction
    )

    output = {
        "_pricing": {
            "cpu_ram_usd_per_gb_hour": _RAM_PRICE,
            "ssd_usd_per_gb_hour": _SSD_PRICE,
            "inter_node_up_usd_per_gbps_hour": pricing.get(
                "inter_node_up_usd_per_gbps_hour", 0.0059
            ),
            "inter_node_down_usd_per_gbps_hour": pricing.get(
                "inter_node_down_usd_per_gbps_hour", 0.0059
            ),
            "ssd_bw_usd_per_gb_s_hour": pricing.get("ssd_bw_usd_per_gb_s_hour", 0.001),
            "pcie_bw_usd_per_gb_s_hour": pricing.get(
                "pcie_bw_usd_per_gb_s_hour", 0.001
            ),
            "nvlink_bw_usd_per_gb_s_hour": pricing.get(
                "nvlink_bw_usd_per_gb_s_hour", 0.002
            ),
            "inet_up_usd_per_gbps_hour": pricing.get("inet_up_usd_per_gbps_hour", 0.2),
            "inet_down_usd_per_gbps_hour": pricing.get(
                "inet_down_usd_per_gbps_hour", 0.2
            ),
            "gpu_family_pricing": family_pricing,
            "gpu_compute_fraction": args.gpu_compute_fraction,
        }
    }

    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote per-family pricing to {args.output}")


if __name__ == "__main__":
    main()
