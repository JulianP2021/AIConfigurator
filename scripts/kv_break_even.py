#!/usr/bin/env python3
"""CLI for computing KV-cache break-even storage duration.

Usage examples:

    python scripts/kv_break_even.py --isl 30000 --model Qwen/Qwen3-8B \
        --machine-hardware "AWS p5.24xlarge (H100 NVL x4)"

    python scripts/kv_break_even.py --isl 10000 \
        --machine-hardware "AWS p5en.48xlarge (H200 x8)" \
        --ram-price 0.01 --ssd-price 0.0005

Output is a JSON object with break-even seconds for RAM, SSD, and S3, plus
recompute time and cost.
"""

import argparse
import json
import sys

from pathlib import Path


# Add the project root to the import path so this script can be invoked from
# anywhere without an editable install.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hardware.scraper import fetch_machine_hardware
from src.model.model import Model
from src.utils.kv_storage_break_even import kv_storage_break_even_seconds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute KV-cache break-even storage duration for a given ISL, model, and hardware preset."
    )
    parser.add_argument(
        "--isl",
        type=int,
        required=True,
        help="Input sequence length (tokens).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name (default: Qwen/Qwen3-8B).",
    )
    parser.add_argument(
        "--machine-hardware",
        type=str,
        required=True,
        help="Hardware preset name, e.g. 'AWS p5.24xlarge (H100 NVL x4)'.",
    )
    parser.add_argument(
        "--ram-price",
        type=float,
        default=None,
        help="RAM storage price in USD per GB per hour (optional).",
    )
    parser.add_argument(
        "--ssd-price",
        type=float,
        default=None,
        help="SSD storage price in USD per GB per hour (optional).",
    )
    parser.add_argument(
        "--s3-month-price",
        type=float,
        default=None,
        help="S3 storage price in USD per GB per month (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model = Model(args.model)
    hardware = fetch_machine_hardware(args.machine_hardware)

    kwargs: dict = {}
    if args.ram_price is not None:
        kwargs["ram_price_usd_per_gb_hour"] = args.ram_price
    if args.ssd_price is not None:
        kwargs["ssd_price_usd_per_gb_hour"] = args.ssd_price
    if args.s3_month_price is not None:
        kwargs["s3_storage_cost_usd_per_gb_month"] = args.s3_month_price

    result = kv_storage_break_even_seconds(
        isl=args.isl,
        model=model,
        hardware=hardware,
        **kwargs,
    )

    output = {
        "unit": "seconds",
        "break_even": result,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
