#!/usr/bin/env python3
"""Refresh/add GPUs to the local spec database backed by Vast.ai data.

Examples::

    python scripts/refresh_gpu_db.py B300
    python scripts/refresh_gpu_db.py B300 B200 H100
    python scripts/refresh_gpu_db.py --list
"""

import argparse
import sys

from pathlib import Path


# Allow importing ``src`` when the script is executed from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hardware.legacy.vast_scraper import refresh_file
from src.hardware.scraper import load_gpu_db


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh the cached GPU spec database from Vast.ai.",
    )
    parser.add_argument(
        "gpus",
        nargs="*",
        help="GPU identifiers to refresh (e.g. B300 B200).",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="Show the GPUs currently in the database and exit.",
    )
    args = parser.parse_args()

    if args.list:
        try:
            db = load_gpu_db()
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1

        if not db:
            print("Database is empty.")
            return 0

        print(f"{'GPU':<15} {'FLOPS':<18} {'VRAM':<12} {'BW':<14} {'$/hr'}")
        print("-" * 65)
        for name, spec in sorted(db.items()):
            flops = spec["flops"]
            mem = spec["gpu_mem"]
            bw = spec["gpu_bw"]
            price = spec["price_usd_per_hour"]
            print(
                f"{name:<15} "
                f"{flops:.2e}  "
                f"{mem / 1e9:.1f} GB    "
                f"{bw / 1e12:.2f} TB/s  "
                f"${price:.4f}"
            )
        return 0

    if not args.gpus:
        print("Refetching all GPUs")
        refresh_file()

    print(f"Refreshing {len(args.gpus)} GPU(s): {', '.join(args.gpus)}")
    refresh_file(args.gpus)
    print("Done.  Data written to src/hardware/legacy/_gpu_db.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
