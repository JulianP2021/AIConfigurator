#!/usr/bin/env python3
"""Derive per-GPU-family component prices from AWS instance configs.

The script loads the AWS preset database and, for each GPU family, tries to
interpolate component prices from pairs of instance configs where only one
resource dimension differs.  If a component cannot be derived for a family, the
global price from ``_pricing`` is used as a fallback.

Usage:

    .venv/bin/python scripts/derive_family_pricing.py

This updates ``src/hardware/aws_hardware.json`` in place.
"""

import argparse
import json
import math
import sys

from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hardware.scraper import load_gpu_db


_GB = 1024**3
_HOUR_S = 3600.0


_COMPONENT_KEYS = [
    "compute_usd_per_gpu_hour",
    "cpu_ram_usd_per_gb_hour",
    "ssd_usd_per_gb_hour",
    "ssd_bw_usd_per_gb_s_hour",
    "inet_up_usd_per_gbps_hour",
    "inet_down_usd_per_gbps_hour",
    "inter_node_up_usd_per_gbps_hour",
    "inter_node_down_usd_per_gbps_hour",
    "pcie_bw_usd_per_gb_s_hour",
    "nvlink_bw_usd_per_gb_s_hour",
]


def _bytes_to_gb_h(val: float) -> float:
    return float(val) / _GB * _HOUR_S


def _bytes_to_gbps(val: float) -> float:
    return float(val) * 8.0 / 1e9


def _extract_feature_vector(
    cfg: dict,
    gpu_mem_gb_per_gpu: float,
) -> tuple[dict[str, float], float]:
    """Return a normalized feature dict and the on-demand price for a config."""
    num_gpus = int(cfg.get("num_gpus", 1))
    features: dict[str, float] = {
        "compute": float(num_gpus),
        "cpu_ram_gb": float(cfg.get("cpu_ram", 0)) / _GB,
        "ssd_gb": float(cfg.get("nvme_mem", 0)) / _GB,
        "ssd_bw_gb_s": float(cfg.get("nvme_bw", 0)) / _GB,
        "inet_up_gbps": _bytes_to_gbps(float(cfg.get("network_inet_up", 0))),
        "inet_down_gbps": _bytes_to_gbps(float(cfg.get("network_inet_down", 0))),
        "inter_up_gbps": _bytes_to_gbps(float(cfg.get("network_inter_node_up", 0))),
        "inter_down_gbps": _bytes_to_gbps(float(cfg.get("network_inter_node_down", 0))),
        "pcie_gb_s": float(cfg.get("pcie_bw", 0)) / _GB,
        "nvlink_gb_s": float(cfg.get("nvlink_bw", 0)) / _GB,
        "hbm_gb": gpu_mem_gb_per_gpu * num_gpus,
    }
    return features, float(cfg.get("dph_base", 0.0))


def _derive_compute_per_gpu(rows: list[tuple[dict[str, float], float]]) -> float | None:
    """Derive compute price per GPU from proportional configs.

    Looks for two configs where every non-compute feature scales exactly with
    the number of GPUs (i.e. per-GPU ratios are identical).  The price
    difference divided by the GPU difference then gives the compute price.
    """
    best: float | None = None
    for i in range(len(rows)):
        fi, pi = rows[i]
        for j in range(i + 1, len(rows)):
            fj, pj = rows[j]
            gpu_diff = fj["compute"] - fi["compute"]
            if gpu_diff == 0:
                continue
            proportional = True
            for key in fi:
                if key == "compute":
                    continue
                if fj[key] * fi["compute"] != fi[key] * fj["compute"]:
                    proportional = False
                    break
            if not proportional:
                continue
            value = (pj - pi) / gpu_diff
            best = value if best is None else (best + value) / 2.0
    return best


def _derive_component_price(
    rows: list[tuple[dict[str, float], float]],
    feature_key: str,
    ignore_keys: tuple[str, ...] = ("compute",),
) -> float | None:
    """Derive a unit price for ``feature_key`` from configs that differ only there.

    Looks for pairs of configs where all other features are identical (or both
    zero) except for ``feature_key``.  Averages the resulting price estimates.
    """
    estimates: list[float] = []
    for i in range(len(rows)):
        fi, pi = rows[i]
        for j in range(i + 1, len(rows)):
            fj, pj = rows[j]
            diff = fj[feature_key] - fi[feature_key]
            if diff == 0:
                continue
            other_equal = True
            for key in fi:
                if key == feature_key or key in ignore_keys:
                    continue
                if not math.isclose(fj[key], fi[key], rel_tol=1e-9, abs_tol=1e-12):
                    other_equal = False
                    break
            if not other_equal:
                continue
            estimates.append((pj - pi) / diff)
    if not estimates:
        return None
    # Use median to be robust against a single outlier pair.
    estimates.sort()
    return estimates[len(estimates) // 2]


def _derive_family(
    _family: str,
    configs: list[dict],
    gpu_mem_gb_per_gpu: float,
    _global_prices: dict[str, float],
) -> dict[str, float]:
    """Return the derived per-family component price table."""
    rows = [_extract_feature_vector(c, gpu_mem_gb_per_gpu) for c in configs]

    result: dict[str, float] = {}

    # Map feature keys to component price keys.
    feature_to_component = {
        "cpu_ram_gb": "cpu_ram_usd_per_gb_hour",
        "ssd_gb": "ssd_usd_per_gb_hour",
        "ssd_bw_gb_s": "ssd_bw_usd_per_gb_s_hour",
        "inet_up_gbps": "inet_up_usd_per_gbps_hour",
        "inet_down_gbps": "inet_down_usd_per_gbps_hour",
        "inter_up_gbps": "inter_node_up_usd_per_gbps_hour",
        "inter_down_gbps": "inter_node_down_usd_per_gbps_hour",
        "pcie_gb_s": "pcie_bw_usd_per_gb_s_hour",
        "nvlink_gb_s": "nvlink_bw_usd_per_gb_s_hour",
    }

    # Step 1: derive non-GPU component prices from configs that differ only in
    # one component (same GPU count, same other resources).  Components that
    # cannot be interpolated for this family are left at zero; their bundled
    # cost is folded into the per-GPU compute price below.
    known_costs = np.zeros(len(rows))
    for feature_key, component_key in feature_to_component.items():
        derived = _derive_component_price(rows, feature_key)
        if derived is not None and derived > 0:
            result[component_key] = derived
            known_costs += np.array([f[feature_key] for f, _ in rows]) * derived
        else:
            result[component_key] = 0.0

    # Step 2: from the residual (price minus known non-GPU costs), derive the
    # compute price per GPU using configs that scale proportionally.  This
    # absorbs all undifferentiated bundled costs (RAM, SSD, NIC, etc.) into the
    # GPU compute price.
    residuals = np.array([p for _, p in rows]) - known_costs
    compute_price = _derive_compute_per_gpu([
        (f, r) for f, r in zip([f for f, _ in rows], residuals, strict=False)
    ])
    if compute_price is None:
        # Fall back to average residual per GPU.
        total_gpus = sum(f["compute"] for f, _ in rows)
        compute_price = sum(residuals) / max(1, total_gpus)
    result["compute_usd_per_gpu_hour"] = max(0.0, float(compute_price))

    # Ensure all expected component keys exist and prices are non-negative.
    for key in _COMPONENT_KEYS:
        result.setdefault(key, 0.0)
    for key, value in result.items():
        result[key] = max(0.0, value)

    return result


def _derive_all_family_pricing(aws_data: dict) -> dict[str, dict[str, float]]:
    machines = aws_data.get("machines", {})
    global_prices = aws_data.get("_pricing", {})
    gpu_db = load_gpu_db()

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
        gpu_mem_gb = gpu_db.get(family, {}).get("gpu_mem", 0) / _GB
        pricing = _derive_family(family, configs, gpu_mem_gb, global_prices)
        family_pricing[family] = pricing

        # Validate against actual prices.
        mape_sum = 0.0
        count = 0
        for cfg in configs:
            f, actual = _extract_feature_vector(cfg, gpu_mem_gb)
            predicted = (
                f["compute"] * pricing["compute_usd_per_gpu_hour"]
                + f["cpu_ram_gb"] * pricing["cpu_ram_usd_per_gb_hour"]
                + f["ssd_gb"] * pricing["ssd_usd_per_gb_hour"]
                + f["ssd_bw_gb_s"] * pricing["ssd_bw_usd_per_gb_s_hour"]
                + f["inet_up_gbps"] * pricing["inet_up_usd_per_gbps_hour"]
                + f["inet_down_gbps"] * pricing["inet_down_usd_per_gbps_hour"]
                + f["inter_up_gbps"] * pricing["inter_node_up_usd_per_gbps_hour"]
                + f["inter_down_gbps"] * pricing["inter_node_down_usd_per_gbps_hour"]
                + f["pcie_gb_s"] * pricing["pcie_bw_usd_per_gb_s_hour"]
                + f["nvlink_gb_s"] * pricing["nvlink_bw_usd_per_gb_s_hour"]
            )
            if actual > 0:
                mape_sum += abs(predicted - actual) / actual
                count += 1
        mape = (mape_sum / count * 100.0) if count else 0.0
        print(
            f"{family}: configs={len(configs)}, mape={mape:.2f}%, "
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
        default=Path("src/hardware/aws_hardware.json"),
        help="Path to aws_hardware.json (default: src/hardware/aws_hardware.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to the input path (in-place update).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    aws_data = json.loads(args.input.read_text(encoding="utf-8"))
    family_pricing = _derive_all_family_pricing(aws_data)

    aws_data["_pricing"]["gpu_family_pricing"] = family_pricing
    # Global fallbacks for bandwidth components (USD per GB/s per GPU per hour).
    aws_data["_pricing"].setdefault("ssd_bw_usd_per_gb_s_hour", 1.1088)
    aws_data["_pricing"].setdefault("pcie_bw_usd_per_gb_s_hour", 0.0)
    aws_data["_pricing"].setdefault("nvlink_bw_usd_per_gb_s_hour", 0.001)

    output_path = args.output or args.input
    output_path.write_text(json.dumps(aws_data, indent=2), encoding="utf-8")
    print(f"\nWrote per-family pricing to {output_path}")


if __name__ == "__main__":
    main()
