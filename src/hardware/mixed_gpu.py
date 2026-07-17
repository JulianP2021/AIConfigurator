"""Mixed-GPU node pricing and hardware construction.

When a colocated node uses one GPU type for prefill and another for decode, the
hourly price is derived from the base machine's full price: we subtract the
compute cost of the GPUs removed from the base machine and add the compute cost
of the donor GPUs.

GPU-only compute prices are read from
``src/hardware/aws_hardware.json`` under ``_pricing.gpu_compute_prices_usd_per_hour``.
If a GPU type is missing from that table, we fall back to a fixed fraction of the
machine's full hourly price.
"""

import copy

from typing import Any

from src.hardware.hardware import Hardware
from src.hardware.scraper import (
    fetch_machine_hardware,
    load_aws_hardware_db,
    load_combined_machine_db,
)


def _gpu_compute_price(machine_name: str, compute_price_fraction: float = 0.6) -> float:
    """Return the compute-only hourly price for one GPU of ``machine_name``.

    Looks up the per-family compute price from
    ``aws_hardware.json::_pricing.gpu_family_pricing``.  If the machine's GPU
    type is not listed or has no positive compute price, fall back to
    ``compute_price_fraction`` of the machine's full hourly price divided by GPU
    count.
    """
    db = load_combined_machine_db()
    config = db[machine_name]
    gpu_name = config.get("gpu_name", "")

    pricing, _ = load_aws_hardware_db()

    family_pricing = pricing.get("gpu_family_pricing", {})
    family = family_pricing.get(gpu_name, {})
    compute_price = family.get("compute_usd_per_gpu_hour", 0.0)
    if compute_price > 0.0:
        return float(compute_price)

    total_gpus = int(config.get("num_gpus", 1))
    compute_only_price = float(config.get("dph_base", 0.0)) * compute_price_fraction
    return compute_only_price / total_gpus if total_gpus > 0 else 0.0


def adjust_price_for_gpu_mix(
    base_machine_name: str,
    base_gpus_to_keep: int,
    donor_machine_name: str,
    donor_gpus_to_add: int,
    *,
    compute_price_fraction: float = 0.6,
) -> tuple[float, dict[str, Any]]:
    """Compute the hourly price for a mixed-GPU machine.

    The price is adjusted by removing the per-GPU compute cost of the GPUs
    taken out of the base machine and adding the per-GPU compute cost of the
    donor GPUs.  RAM and SSD are ignored in the swap; the base machine keeps its
    original memory configuration and price contribution.

    Parameters
    ----------
    base_machine_name:
        Machine that provides the chassis, CPU, NIC, RAM and SSD baseline.
    base_gpus_to_keep:
        Number of GPUs retained from ``base_machine``.
    donor_machine_name:
        Machine whose GPU type is being added.
    donor_gpus_to_add:
        Number of donor GPUs to add.
    compute_price_fraction:
        Fraction of ``dph_base`` attributed to compute when deriving the
        per-GPU compute price.

    Returns:
    -------
    ``(new_price_per_hour, breakdown_dict)``.
    """
    db = load_combined_machine_db()
    base_config = db[base_machine_name]

    base_total_gpus = int(base_config.get("num_gpus", 1))
    if base_gpus_to_keep < 0 or base_gpus_to_keep > base_total_gpus:
        raise ValueError(
            f"base_gpus_to_keep ({base_gpus_to_keep}) must be between 0 and "
            f"{base_total_gpus} for {base_machine_name!r}"
        )
    base_gpus_removed = base_total_gpus - base_gpus_to_keep

    base_gpu_price = _gpu_compute_price(base_machine_name, compute_price_fraction)
    donor_gpu_price = _gpu_compute_price(donor_machine_name, compute_price_fraction)

    base_full_price = float(base_config.get("dph_base", 0.0))
    new_price = (
        base_full_price
        - base_gpu_price * base_gpus_removed
        + donor_gpu_price * donor_gpus_to_add
    )

    breakdown = {
        "base_machine": base_machine_name,
        "donor_machine": donor_machine_name,
        "base_full_price": base_full_price,
        "base_gpu_price": base_gpu_price,
        "donor_gpu_price": donor_gpu_price,
        "base_gpus_to_keep": base_gpus_to_keep,
        "base_gpus_removed": base_gpus_removed,
        "donor_gpus_to_add": donor_gpus_to_add,
        "new_price_per_hour": new_price,
    }
    return new_price, breakdown


def fetch_mixed_gpu_hardware(
    base_machine_name: str,
    base_gpus_to_keep: int,
    donor_machine_name: str,
    donor_gpus_to_add: int,
    *,
    compute_price_fraction: float = 0.6,
) -> Hardware:
    """Build a :class:`Hardware` instance for a node with mixed GPU types.

    The returned object keeps the base machine's chassis, CPU, RAM, SSD and
    network attributes, but its GPU count and hourly price reflect a swap where
    some base GPUs are replaced by donor GPUs.
    """
    new_price, breakdown = adjust_price_for_gpu_mix(
        base_machine_name,
        base_gpus_to_keep,
        donor_machine_name,
        donor_gpus_to_add,
        compute_price_fraction=compute_price_fraction,
    )

    db = load_combined_machine_db()
    base_config = copy.deepcopy(db[base_machine_name])
    donor_config = db[donor_machine_name]
    total_gpus = base_gpus_to_keep + donor_gpus_to_add

    # Mixed GPU nodes are limited by the slower GPU's local memory bandwidth.
    base_nvlink = float(base_config.get("nvlink_bw", 0.0))
    donor_nvlink = float(donor_config.get("nvlink_bw", 0.0))
    base_config["nvlink_bw"] = (
        min(base_nvlink, donor_nvlink) if base_nvlink and donor_nvlink else 0.0
    )

    base_config["num_gpus"] = total_gpus
    base_config["dph_base"] = new_price
    base_config["name"] = (
        f"{base_machine_name} + {donor_gpus_to_add}x {donor_machine_name}"
    )
    base_config["_mixed_gpu"] = breakdown

    mixed_key = base_config["name"]
    return fetch_machine_hardware(mixed_key, machine_config_override=base_config)
