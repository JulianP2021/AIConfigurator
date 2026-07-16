#!/usr/bin/env python3
"""CLI for computing KV-cache hourly storage cost.

Usage examples:

    python scripts/kv_storage_cost.py --tokens 1000000 --model Qwen/Qwen3-8B

    python scripts/kv_storage_cost.py --tokens 1000000 \
        --ssd-price 0.0005 --ram-price 0.01

Output is a JSON object with the USD/hour cost to store the KV cache in RAM,
SSD, and S3.
"""

import argparse
import json
import sys

from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.model import Model
from src.utils.kv_storage_cost import kv_storage_cost_per_hour_usd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute KV-cache hourly storage cost for a given token count and model."
    )
    parser.add_argument(
        "--tokens",
        type=int,
        required=True,
        help="Number of KV tokens to store (e.g. 1000000).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name (default: Qwen/Qwen3-8B).",
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
    parser.add_argument(
        "--kv-size-gb-per-token",
        type=float,
        default=None,
        help="Override the model's KV size in GB per token (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model = Model(args.model)

    kwargs: dict = {}
    if args.ram_price is not None:
        kwargs["ram_price_usd_per_gb_hour"] = args.ram_price
    if args.ssd_price is not None:
        kwargs["ssd_price_usd_per_gb_hour"] = args.ssd_price
    if args.s3_month_price is not None:
        kwargs["s3_storage_cost_usd_per_gb_month"] = args.s3_month_price
    if args.kv_size_gb_per_token is not None:
        kwargs["kv_size_gb_per_token"] = args.kv_size_gb_per_token

    result = kv_storage_cost_per_hour_usd(
        tokens=args.tokens,
        model=model,
        **kwargs,
    )

    output = {
        "unit": "usd_per_hour",
        "storage_cost": result,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
