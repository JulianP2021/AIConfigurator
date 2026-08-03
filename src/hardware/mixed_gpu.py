"""Mixed-GPU node pricing and hardware construction.

When a colocated node uses one GPU type for prefill and another for decode, the
hourly price is derived from the base machine's full price.  We subtract the
swappable portion of each GPU slot removed from the base machine and add the
swappable portion of each donor GPU slot.  The swappable portion is controlled
by ``compute_price_fraction``; the remainder stays with the base machine as a
fixed chassis residual.
"""

import copy

from typing import Any

from src.hardware.hardware import Hardware
from src.hardware.scraper import (
    fetch_machine_hardware,
    load_aws_hardware_db,
    load_combined_machine_db,
)


GB = 1024.0**3


def _gpu_slot_price(machine_name: str, compute_price_fraction: float = 0.6) -> float:
    """Return the swappable portion of one GPU slot for ``machine_name``.

    A GPU slot is the GPU plus its proportional share of the reference AWS
    machine's RAM, SSD, NVLink, PCIe, and network resources, priced with the
    per-family component table.  ``compute_price_fraction`` controls how much
    of that slot cost is actually moved during a mixed-GPU swap; the remaining
    ``1 - compute_price_fraction`` stays with the base machine as a fixed
    chassis residual.

    The per-family table itself is calibrated for a particular split (stored
    in ``gpu_compute_fraction``).  If ``compute_price_fraction`` differs from
    that table split, this function raises an error and asks the user to
    regenerate the pricing table with the desired split.
    """
    db = load_combined_machine_db()
    config = db[machine_name]
    gpu_name = config.get("gpu_name", "")

    pricing, _ = load_aws_hardware_db()
    family_pricing = pricing.get("gpu_family_pricing", {})
    family = family_pricing.get(gpu_name, {})

    table_fraction = float(
        pricing.get("gpu_compute_fraction") or family.get("gpu_compute_fraction") or 0.6
    )
    if abs(compute_price_fraction - table_fraction) > 1e-6:
        raise ValueError(
            f"Requested GPU compute fraction {compute_price_fraction} does not match "
            f"the pricing table's split {table_fraction} for family {gpu_name!r}. "
            f"Regenerate src/hardware/data/pricing.json with the desired split, e.g. "
            f".venv/bin/python scripts/derive_family_pricing.py --gpu-compute-fraction {compute_price_fraction}"
        )

    aws_instances = [
        cfg
        for cfg in db.values()
        if cfg.get("gpu_name") == gpu_name
        and str(cfg.get("name", "")).startswith("AWS")
    ]
    if not aws_instances:
        # Last resort: derive from the machine's own dph_base.
        total_gpus = int(config.get("num_gpus", 1))
        slot_price = (
            float(config.get("dph_base", 0.0)) / total_gpus if total_gpus > 0 else 0.0
        )
        return slot_price * compute_price_fraction

    ref = max(aws_instances, key=lambda cfg: int(cfg.get("num_gpus", 1)))
    n = int(ref["num_gpus"])

    def fprice(key: str) -> float:
        return float(family.get(key, 0.0))

    slot_price = fprice("compute_usd_per_gpu_hour")
    slot_price += fprice("cpu_ram_usd_per_gb_hour") * (ref["cpu_ram"] / GB) / n
    slot_price += fprice("ssd_usd_per_gb_hour") * (ref["nvme_mem"] / GB) / n
    slot_price += fprice("pcie_bw_usd_per_gb_s_hour") * (ref["pcie_bw"] / GB) / n
    slot_price += (
        fprice("nvlink_bw_usd_per_gb_s_hour") * (ref.get("nvlink_bw", 0.0) / GB) / n
    )
    slot_price += (
        (
            fprice("inter_node_up_usd_per_gbps_hour")
            + fprice("inter_node_down_usd_per_gbps_hour")
        )
        / 2.0
        * (ref.get("network_inter_node_up", 100e9) / 1e9)
        / n
    )
    slot_price += (
        (fprice("inet_up_usd_per_gbps_hour") + fprice("inet_down_usd_per_gbps_hour"))
        / 2.0
        * (ref["network_inet_up"] / 1e9 + ref["network_inet_down"] / 1e9)
        / n
    )
    return slot_price * compute_price_fraction


def adjust_price_for_gpu_mix(
    base_machine_name: str,
    base_gpus_to_keep: int,
    donor_machine_name: str,
    donor_gpus_to_add: int,
    *,
    compute_price_fraction: float = 0.6,
) -> tuple[float, dict[str, Any]]:
    """Compute the hourly price for a mixed-GPU machine.

    The price is adjusted by removing the all-in per-GPU cost of the GPUs
    taken out of the base machine and adding the all-in per-GPU cost of the
    donor GPUs.  RAM, SSD, chassis and NIC bandwidth stay with the base machine
    and are not adjusted.

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

    base_slot_price = _gpu_slot_price(base_machine_name, compute_price_fraction)
    donor_slot_price = _gpu_slot_price(donor_machine_name, compute_price_fraction)

    base_full_price = float(base_config.get("dph_base", 0.0))
    new_price = (
        base_full_price
        - base_slot_price * base_gpus_removed
        + donor_slot_price * donor_gpus_to_add
    )

    breakdown = {
        "base_machine": base_machine_name,
        "donor_machine": donor_machine_name,
        "base_full_price": base_full_price,
        "base_slot_price": base_slot_price,
        "donor_slot_price": donor_slot_price,
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
