#!/usr/bin/env python3
"""Recalculate hardware-economics result prices using current hardware pricing.

For every ``results_*.json`` file in ``--results-dir`` the prefill hardware
price is looked up in the local hardware database by the row's
``prefill_hardware`` label and multiplied by the number of prefill nodes
(``num_prefill_workers``, 5 for the colocated 5-node sweep). The result is
written back into ``compute_price_usd_per_hour``.

``total_cost_usd_per_hour`` is recomputed as compute + S3 transfer + S3
storage. Storage is counted exactly once: ``s3_cost_usd_per_hour`` already
includes ``s3_storage_cost_usd_per_hour``, so the S3 transfer cost is taken as
their difference. ``price_per_user`` is recomputed as total / ``max_users``.

A full backup of ``--results-dir`` is created before any modification, and the
stale ``results_ram_1200.json`` file is removed because the 1200 GB RAM hardware
preset no longer exists in the pricing database.
"""

from __future__ import annotations
import argparse
import json
import shutil
import sys
import time

from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hardware.scraper import load_combined_machine_db


_STALE_RAM_1200_FILE = "results_ram_1200.json"


def _round(value: float, digits: int) -> float:
    return round(float(value), digits)


def _lookup_hardware_price(db: dict[str, dict[str, Any]], label: str) -> float:
    try:
        price = float(db[label]["dph_base"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Hardware label {label!r} not found in hardware database"
        ) from exc
    if price <= 0.0:
        raise ValueError(f"Hardware label {label!r} has non-positive dph_base {price}")
    return price


def _reprice_row(
    db: dict[str, dict[str, Any]],
    row: dict[str, Any],
    *,
    compute_digits: int = 4,
    total_digits: int = 4,
    price_per_user_digits: int = 6,
) -> dict[str, Any]:
    prefill_label = row["prefill_hardware"]
    node_price = _lookup_hardware_price(db, prefill_label)
    num_nodes = int(row["num_prefill_workers"])

    compute_price = node_price * num_nodes
    s3_cost = float(row["s3_cost_usd_per_hour"])
    s3_storage = float(row["s3_storage_cost_usd_per_hour"])
    s3_transfer = s3_cost - s3_storage

    total_cost = compute_price + s3_transfer + s3_storage

    max_users = int(row["max_users"])
    price_per_user = total_cost / max_users if max_users > 0 else float("inf")

    row["compute_price_usd_per_hour"] = _round(compute_price, compute_digits)
    row["total_cost_usd_per_hour"] = _round(total_cost, total_digits)
    row["price_per_user"] = _round(price_per_user, price_per_user_digits)
    return row


def _backup_results_dir(results_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = results_dir.parent / f"{results_dir.name}-backup-{stamp}"
    shutil.copytree(results_dir, backup_dir)
    print(f"Backed up {results_dir} -> {backup_dir}")
    return backup_dir


def _process_results_file(
    db: dict[str, dict[str, Any]],
    path: Path,
) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    for row in rows:
        _reprice_row(db, row)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute compute/total cost and price-per-user in hardware "
            "economics results using current hardware prices."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("hardware-results"),
        help="Directory containing results_*.json files (default: hardware-results).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_dir():
        parser.error(f"Results directory does not exist: {results_dir}")

    _backup_results_dir(results_dir)

    stale = results_dir / _STALE_RAM_1200_FILE
    if stale.exists():
        stale.unlink()
        print(f"Removed stale results file: {stale}")

    db = load_combined_machine_db()

    updated = 0
    total_rows = 0
    for path in sorted(results_dir.glob("results_*.json")):
        rows = _process_results_file(db, path)
        updated += 1
        total_rows += rows
        print(f"Updated {path.name} ({rows} rows)")

    print(f"Done: {updated} file(s), {total_rows} row(s) repriced.")


if __name__ == "__main__":
    main()
