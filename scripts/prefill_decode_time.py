#!/usr/bin/env python3
"""Compute prefill time and single-token decode time given a GPU and a request.

Uses the same analytical model as the simulator (``src.instances.prefill`` /
``src.instances.decode``)::

    time_ms = max(flops / gpu_flops, memory / gpu_bw) * 1000

with ``flops`` / ``memory`` computed by the formulas in ``src.utils.utils``
(``_calculate_flops`` / ``_calculate_memory``).  A GPU is *compute bound* when
the FLOPs term dominates, *memory bound* otherwise.

Usage:

    python scripts/prefill_decode_time.py \
        --flops 9.9e14 --mem-bw 4.8e12 --isl 100000 --cached 50000

Output is a small JSON object with the prefill and per-token decode times.
"""

import argparse
import json
import sys

from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.model import Model
from src.utils.utils import _calculate_flops, _calculate_memory, _mem_model


def prefill_flops_mem(model: Model, isl: int, cached: int) -> tuple[int, int]:
    """FLOPs / bytes to prefill a request with ``cached`` tokens already cached."""
    tokens = max(isl - cached, 0)
    flops = _calculate_flops(model, tokens, cached)
    memory = _mem_model(model) + _calculate_memory(model, tokens, cached)
    return flops, memory


def decode_flops_mem(model: Model, isl: int) -> tuple[int, int]:
    """FLOPs / bytes to decode one token after the full prompt is cached."""
    flops = _calculate_flops(model, 1, isl)
    memory = _mem_model(model) + _calculate_memory(model, 1, isl)
    return flops, memory


def time_ms(
    flops: int, memory: int, gpu_flops: float, gpu_bw: float
) -> tuple[float, str]:
    """Simulator time model: max of compute and memory terms, in ms."""
    compute_ms = flops / gpu_flops * 1000
    memory_ms = memory / gpu_bw * 1000
    bound = "compute" if compute_ms >= memory_ms else "memory"
    return max(compute_ms, memory_ms), bound


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefill and per-token decode time from GPU flops / memory bw."
    )
    parser.add_argument(
        "--flops",
        type=float,
        required=True,
        help="GPU FLOPs (float ops/s), e.g. 9.9e14 for an H200.",
    )
    parser.add_argument(
        "--mem-bw",
        type=float,
        required=True,
        help="GPU memory bandwidth in bytes/s, e.g. 4.8e12 for an H200.",
    )
    parser.add_argument(
        "--isl",
        type=int,
        required=True,
        help="Input sequence length in tokens.",
    )
    parser.add_argument(
        "--cached",
        type=int,
        default=0,
        help="Tokens already in the KV cache (cached prefix).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name (default: Qwen/Qwen3-8B).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cached > args.isl:
        print(
            f"error: --cached ({args.cached}) cannot exceed --isl ({args.isl})",
            file=sys.stderr,
        )
        sys.exit(1)

    model = Model(args.model)
    p_flops, p_mem = prefill_flops_mem(model, args.isl, args.cached)
    d_flops, d_mem = decode_flops_mem(model, args.isl)

    prefill_ms, prefill_bound = time_ms(p_flops, p_mem, args.flops, args.mem_bw)
    decode_ms, decode_bound = time_ms(d_flops, d_mem, args.flops, args.mem_bw)

    print(
        json.dumps(
            {
                "model": args.model,
                "isl": args.isl,
                "cached": args.cached,
                "gpu_flops": args.flops,
                "gpu_mem_bw_bytes_s": args.mem_bw,
                "prefill": {
                    "time_ms": prefill_ms,
                    "flops": p_flops,
                    "memory_bytes": p_mem,
                    "bound": prefill_bound,
                },
                "decode_one_token": {
                    "time_ms": decode_ms,
                    "flops": d_flops,
                    "memory_bytes": d_mem,
                    "bound": decode_bound,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
