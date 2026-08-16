#!/usr/bin/env python3
"""Compute arithmetic intensity and prefill compute/memory balance for every
GPU in the GPU database.

Arithmetic intensity is reported as FLOPs / byte of GPU memory (min / max /
mean across the database).  For prefill, the time model from
``src.instances.prefill`` is used::

    time_ms = max(flops / gpu_flops, memory / gpu_bw) * 1000

with ``flops`` / ``memory`` computed by the formulas in ``src.utils.utils``
(``calculate_flops`` / ``calculate_memory``).  A GPU is *compute bound* when
the FLOPs term dominates.

Usage:

    python scripts/gpu_arithmetic_intensity.py [--isl 100000] [--model Qwen/Qwen3-8B]

Output is a JSON object with arithmetic-intensity stats plus a per-GPU prefill
breakdown and summary counts of compute vs memory bound GPUs.
"""

import argparse
import json
import sys

from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hardware.scraper import load_gpu_db
from src.model.model import Model
from src.utils.utils import _calculate_flops, _calculate_memory, _mem_model


def prefill_flops_mem(model: Model, isl: int) -> tuple[int, int]:
    """FLOPs / bytes to prefresh an ISL-token request with an empty cache."""
    flops = _calculate_flops(model, isl, 0)
    mem = _mem_model(model) + _calculate_memory(model, isl, 0)
    return flops, mem


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU arithmetic intensity and prefill compute/memory balance."
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=100000,
        help="Prefill input sequence length in tokens.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name (default: Qwen/Qwen3-8B).",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print a LaTeX table of arithmetic intensities instead of JSON.",
    )
    return parser.parse_args()


def _fmt_ai(value: float) -> str:
    """Format arithmetic intensity for a LaTeX table (FLOP/byte)."""
    return f"{value:,.0f}"


def main() -> None:
    args = _parse_args()
    gpu_db = load_gpu_db()
    model = Model(args.model)
    flops, mem = prefill_flops_mem(model, args.isl)

    intensities: dict[str, float] = {}
    for name, spec in gpu_db.items():
        mem_bytes = spec["gpu_bw"]
        if mem_bytes <= 0:
            raise ValueError(f"{name}: gpu_mem must be > 0, got {mem_bytes}")
        intensities[name] = spec["flops"] / mem_bytes

    min_name = min(intensities, key=intensities.get)
    max_name = max(intensities, key=intensities.get)
    mean = sum(intensities.values()) / len(intensities)

    if args.latex:
        print(_render_latex(intensities, min_name, max_name, mean))
        return

    prefill_hit: dict[str, float] = {}
    bound: dict[str, str] = {}
    for name, spec in gpu_db.items():
        gpu_flops = spec["flops"]
        gpu_bw = spec["gpu_bw"]
        compute_ms = flops / gpu_flops * 1000
        memory_ms = mem / gpu_bw * 1000
        prefill_hit[name] = max(compute_ms, memory_ms)
        bound[name] = "compute" if compute_ms >= memory_ms else "memory"

    compute_count = sum(1 for b in bound.values() if b == "compute")
    min_hit_name = min(prefill_hit, key=prefill_hit.get)
    max_hit_name = max(prefill_hit, key=prefill_hit.get)

    output = {
        "unit": "FLOPs per byte",
        "count": len(intensities),
        "arithmetic_intensity": {
            "min": {"name": min_name, "value": intensities[min_name]},
            "max": {"name": max_name, "value": intensities[max_name]},
            "mean": mean,
        },
        "prefill": {
            "model": args.model,
            "isl": args.isl,
            "unit_hit_ms": "milliseconds",
            "hit_min": {"name": min_hit_name, "value": prefill_hit[min_hit_name]},
            "hit_max": {"name": max_hit_name, "value": prefill_hit[max_hit_name]},
            "hit_mean": sum(prefill_hit.values()) / len(prefill_hit),
            "compute_bound": compute_count,
            "memory_bound": len(prefill_hit) - compute_count,
            "gpus": {
                name: {
                    "arithmetic_intensity": intensities[name],
                    "bound": bound[name],
                    "prefill_hit_ms": prefill_hit[name],
                }
                for name in gpu_db
            },
        },
    }

    print(json.dumps(output, indent=2))


def _render_latex(
    intensities: dict[str, float],
    min_name: str,
    max_name: str,
    mean: float,
) -> str:
    """Render the arithmetic intensities as a LaTeX table sorted descending."""
    names = sorted(intensities, key=lambda n: intensities[n], reverse=True)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{l r}",
        r"\toprule",
        r"GPU & Arithmetic Intensity (FLOP/byte) \\",
        r"\midrule",
    ]
    for name in names:
        lines.append(f"{name} & {_fmt_ai(intensities[name])} \\\\")
    lines += [
        r"\midrule",
        f"Mean & {_fmt_ai(mean)} \\\\",
        f"Min ({min_name.replace('_', r'\\_')}) & {_fmt_ai(intensities[min_name])} \\\\",
        f"Max ({max_name.replace('_', r'\\_')}) & {_fmt_ai(intensities[max_name])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
