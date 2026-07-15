#!/usr/bin/env python3
"""Refresh the local machine / node spec database backed by Vast.ai data.

Examples::

    python scripts/refresh_machines.py
    python scripts/refresh_machines.py --list
"""

import argparse
import sys

from pathlib import Path


# Allow importing ``src`` when the script is executed from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hardware.legacy.vast_scraper import refresh_machines_file
from src.hardware.scraper import load_machine_db


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh the cached machine spec database from Vast.ai.",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="Show the machines currently in the database and exit.",
    )
    args = parser.parse_args()

    if args.list:
        try:
            db = load_machine_db()
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1

        if not db:
            print("Database is empty.")
            return 0

        print(
            f"{'Machine':<45} "
            f"{'GPUs':<5} "
            f"{'RAM':<10} "
            f"{'NVME':<10} "
            f"{'Net Up':<8} "
            f"{'Net Down':<8} "
            f"{'$/hr'}"
        )
        print("-" * 100)
        for name, spec in sorted(db.items()):
            gpus = spec["num_gpus"]
            ram = spec["cpu_ram"]
            nvme = spec["nvme_mem"]
            net_up = spec["network_inet_up"]
            net_down = spec["network_inet_down"]
            price = spec["dph_base"]
            print(
                f"{name:<45} "
                f"{gpus:<5} "
                f"{ram:<10.0f} "
                f"{nvme:<10.0f} "
                f"{net_up:<8.0f} "
                f"{net_down:<8.0f} "
                f"${price:.4f}"
            )
        return 0

    print("Refreshing machines from Vast.ai...")
    refresh_machines_file()
    db = load_machine_db()
    print(f"Done. {len(db)} machine(s) written to src/hardware/legacy/_machine_db.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
