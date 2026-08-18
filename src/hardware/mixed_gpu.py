"""Mixed-GPU node pricing and hardware construction.

When a colocated node uses one GPU type for prefill and another for decode, the
hourly price is derived as follows:
- 40% of the more expensive machine's price is fixed (chassis, CPU, RAM, SSD, NIC)
- 60% of each machine's price is attributed to its GPUs
- The GPU cost scales with the actual number of GPUs of each type in the mix
"""

import copy

from typing import Any

from src.hardware.hardware import Hardware
from src.hardware.scraper import (
    fetch_machine_hardware,
    load_combined_machine_db,
)


GB = 1024.0**3


def adjust_price_for_gpu_mix(
    base_machine_name: str,
    base_gpus_to_keep: int,
    donor_machine_name: str,
    donor_gpus_to_add: int,
    *,
    compute_price_fraction: float = 0.6,
) -> tuple[float, dict[str, Any]]:
    """Compute the hourly price for a mixed-GPU machine.

    Pricing model:
    - Fixed cost = 40% of the more expensive machine's full price
    - GPU cost = sum of (gpu_count * (60% of machine_price / machine_gpu_count)) for each type

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
        Fraction of ``dph_base`` attributed to compute (default 0.6).

    Returns:
    -------
    ``(new_price_per_hour, breakdown_dict)``.
    """
    db = load_combined_machine_db()
    base_config = db[base_machine_name]
    donor_config = db[donor_machine_name]

    base_total_gpus = int(base_config.get("num_gpus", 1))
    donor_total_gpus = int(donor_config.get("num_gpus", 1))

    if base_gpus_to_keep < 0 or base_gpus_to_keep > base_total_gpus:
        raise ValueError(
            f"base_gpus_to_keep ({base_gpus_to_keep}) must be between 0 and "
            f"{base_total_gpus} for {base_machine_name!r}"
        )

    base_full_price = float(base_config.get("dph_base", 0.0))
    donor_full_price = float(donor_config.get("dph_base", 0.0))

    # Fixed cost = 40% of the more expensive machine
    more_expensive_price = max(base_full_price, donor_full_price)
    fixed_cost = more_expensive_price * (1.0 - compute_price_fraction)

    # GPU cost: each GPU type priced at its own machine's rate
    base_gpu_per_unit = (
        (base_full_price * compute_price_fraction) / base_total_gpus
        if base_total_gpus > 0
        else 0.0
    )
    donor_gpu_per_unit = (
        (donor_full_price * compute_price_fraction) / donor_total_gpus
        if donor_total_gpus > 0
        else 0.0
    )

    base_gpu_cost = base_gpus_to_keep * base_gpu_per_unit
    donor_gpu_cost = donor_gpus_to_add * donor_gpu_per_unit

    new_price = fixed_cost + base_gpu_cost + donor_gpu_cost

    breakdown = {
        "base_machine": base_machine_name,
        "donor_machine": donor_machine_name,
        "base_full_price": base_full_price,
        "donor_full_price": donor_full_price,
        "more_expensive_price": more_expensive_price,
        "compute_price_fraction": compute_price_fraction,
        "fixed_cost": fixed_cost,
        "base_gpu_per_unit": base_gpu_per_unit,
        "donor_gpu_per_unit": donor_gpu_per_unit,
        "base_gpus_to_keep": base_gpus_to_keep,
        "donor_gpus_to_add": donor_gpus_to_add,
        "base_gpu_cost": base_gpu_cost,
        "donor_gpu_cost": donor_gpu_cost,
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

    The returned object uses the max of the two component values for all
    hardware specs (bandwidth, memory, etc.) and the new computed price.
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

    # Use max of the two component values for all hardware specs
    for key in [
        "nvlink_bw",
        "pcie_bw",
        "nvme_bw",
        "network_inter_node_up",
        "network_inter_node_down",
        "network_inet_up",
        "network_inet_down",
        "cpu_ram",
        "nvme_mem",
    ]:
        base_val = float(base_config.get(key, 0.0))
        donor_val = float(donor_config.get(key, 0.0))
        if base_val and donor_val:
            base_config[key] = max(base_val, donor_val)
        elif donor_val:
            base_config[key] = donor_val

    base_config["num_gpus"] = total_gpus
    base_config["dph_base"] = new_price
    base_config["name"] = (
        f"{base_machine_name} + {donor_gpus_to_add}x {donor_machine_name}"
    )
    base_config["_mixed_gpu"] = breakdown

    mixed_key = base_config["name"]
    return fetch_machine_hardware(mixed_key, machine_config_override=base_config)
