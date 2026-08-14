#!/usr/bin/env python3
"""Refresh AWS hardware presets from the Vantage Instances API.

This script queries the public Vantage Instances API (https://instances.vantage.sh)
via the ``instances-api-client`` Python library and (re)builds
``src/hardware/data/aws_hardware.json``.  It is the single source of truth for how
simulator hardware fields are derived from upstream API data.

Run with:

    .venv/bin/python scripts/refresh_aws_hardware.py

Use ``--dry-run`` to print the computed data without writing the file.
"""

import argparse
import json
import re
import sys

from pathlib import Path
from typing import Any

from instances_api_client import APIV1Client


ROOT = Path(__file__).resolve().parent.parent
AWS_HARDWARE_PATH = ROOT / "src" / "hardware" / "data" / "aws_hardware.json"
GPU_DB_PATH = ROOT / "src" / "hardware" / "legacy" / "_gpu_db.json"

# Per-GPU PCIe encoded throughput in GB/s (decimal).  These are theoretical x16
# link throughputs after 128b/130b encoding: Gen3 = 126 Gb/s / 8 = 15.75 GB/s,
# Gen4 = 252 Gb/s / 8 = 31.5 GB/s, Gen5 = 504 Gb/s / 8 = 63.0 GB/s.
_PCIE_BW_PER_GPU_GB_S: dict[str, float] = {
    "TESLA_V100": 15.75,
    "TESLA_T4": 15.75,
    "INF1": 15.75,
    "A100_40GB": 31.5,
    "A100_80GB": 31.5,
    "A10G": 31.5,
    "L4": 31.5,
    "L40S": 31.5,
    "INF2": 31.5,
    "H100 NVL": 63.0,
    "H200": 63.0,
    "H200 NVL": 63.0,
    "B200": 63.0,
    "B300": 63.0,
    "GB202": 63.0,
    "RTX_PRO_4500": 63.0,
}

# Per-GPU NVLink/C2C bandwidth in GB/s (decimal).  Values are one half of the
# commonly quoted aggregate bi-directional bandwidth for each GPU generation.
_NVLINK_BW_PER_GPU_GB_S: dict[str, float] = {
    "TESLA_V100": 75.0,
    "A100_40GB": 300.0,
    "A100_80GB": 300.0,
    "H100 NVL": 450.0,
    "H200": 450.0,
    "H200 NVL": 450.0,
    "B200": 900.0,
    "B300": 900.0,
}

# Map display GPU names used in machine keys to internal GPU database names.
_GPU_NAME_MAP: dict[str, str] = {
    "Tesla V100": "TESLA_V100",
    "A100 40GB": "A100_40GB",
    "A100 80GB": "A100_80GB",
    "H100 NVL": "H100 NVL",
    "H200": "H200",
    "H200 NVL": "H200 NVL",
    "B200": "B200",
    "B300": "B300",
    "GB202": "GB202",
    "RTX PRO 4500": "RTX_PRO_4500",
    "TESLA_T4": "TESLA_T4",
    "A10G": "A10G",
    "L4": "L4",
    "L40S": "L40S",
    "INF1": "INF1",
    "INF2": "INF2",
}

# Map API gpu_model strings to internal GPU database names.
_GPU_MODEL_MAP: dict[str, str] = {
    "NVIDIA Tesla V100": "TESLA_V100",
    "NVIDIA T4 Tensor Core": "TESLA_T4",
    "NVIDIA A100": "A100_40GB",
    "NVIDIA A10G": "A10G",
    "NVIDIA L4": "L4",
    "NVIDIA L40S": "L40S",
    "NVIDIA H100": "H100 NVL",
    "NVIDIA H200": "H200",
    "NVIDIA B200": "B200",
    "NVIDIA B300": "B300",
    "NVIDIA GB202": "GB202",
    "NVIDIA RTX PRO 4500": "RTX_PRO_4500",
    "NVIDIA RTX PRO 4500 Blackwell": "RTX_PRO_4500",
    "AWS Inferentia1": "INF1",
    "AWS Inferentia2": "INF2",
    "AWS Inferentia": "INF1",
    "Inferentia": "INF1",
    "Inferentia2": "INF2",
}

_GB = 1e9
_GIB = 1024**3
_IOPS_PAGE_BYTES = 4096


def _parse_network_performance_gbps(value: str | None) -> float:
    """Return the network bandwidth in Gbps from a Vantage ``network_performance`` string.

    Handles formats like ``"25 Gigabit"``, ``"Up to 25 Gigabit"``,
    ``"4x 100 Gigabit"`` (aggregate 400) and ``"3200 Gigabit"``.
    Returns ``0.0`` when the value cannot be parsed.
    """
    if not value:
        return 0.0
    text = value.lower()
    # Match either "Nx M Gigabit" (aggregate product) or "N Gigabit".
    multi = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*gigabit", text)
    if multi:
        return float(multi.group(1)) * float(multi.group(2))
    single = re.search(r"(\d+(?:\.\d+)?)\s*gigabit", text)
    if single:
        return float(single.group(1))
    return 0.0


def _load_gpu_db() -> dict[str, dict[str, Any]]:
    return json.loads(GPU_DB_PATH.read_text(encoding="utf-8"))


def _map_gpu_model(api_gpu_model: str | None, gpu_db: dict[str, dict[str, Any]]) -> str:
    """Return the internal GPU name for an API gpu_model string."""
    if not api_gpu_model:
        raise ValueError("API response is missing gpu_model")
    mapped = _GPU_MODEL_MAP.get(api_gpu_model)
    if mapped:
        return mapped
    # Fall back to substring matching against known GPU DB names.
    lower = api_gpu_model.lower()
    for db_name in gpu_db:
        if db_name.lower() in lower or lower in db_name.lower():
            return db_name
    raise ValueError(f"Unknown gpu_model from API: {api_gpu_model!r}")


def _extract_instance_type_and_gpu(machine_name: str) -> tuple[str, str]:
    """Extract the EC2 instance type and expected GPU name from the machine key.

    The key looks like ``AWS p5.48xlarge (H100 NVL x8)``.  The GPU name inside
    the parentheses is preserved so that instances such as ``p4de.24xlarge``
    (80 GB A100) are not conflated with ``p4d.24xlarge`` (40 GB A100).
    """
    if not machine_name.startswith("AWS "):
        raise ValueError(f"Cannot parse machine key {machine_name!r}")
    body = machine_name[len("AWS ") :]
    itype = body.split()[0]
    # GPU name is the text before " xN" inside parentheses, e.g. "H100 NVL".
    paren_open = body.find("(")
    paren_close = body.find(")")
    if paren_open == -1 or paren_close == -1:
        raise ValueError(f"Cannot parse GPU name from {machine_name!r}")
    gpu_part = body[paren_open + 1 : paren_close].strip()
    gpu_name = gpu_part.rsplit(" x", 1)[0].strip()
    return itype, gpu_name


def _compute_nvme_bw(read_iops: int | None, write_iops: int | None) -> int:
    """Return NVMe bandwidth in bytes/s from IOPS.

    Uses ``min(read_iops, write_iops) * 4 KiB``.  If either IOPS value is
    missing/0, the result is 0 and we do not interpolate.
    """
    r = read_iops or 0
    w = write_iops or 0
    if r == 0 or w == 0:
        return 0
    return int(min(r, w) * _IOPS_PAGE_BYTES)


def _compute_machine_config(
    api_inst: Any,
    expected_gpu_name: str,
    gpu_db: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a raw machine config dict from one Vantage API EC2 instance."""
    itype = api_inst.instance_type
    if expected_gpu_name in gpu_db:
        gpu_name = expected_gpu_name
    elif expected_gpu_name in _GPU_NAME_MAP:
        gpu_name = _GPU_NAME_MAP[expected_gpu_name]
    else:
        gpu_name = _map_gpu_model(api_inst.gpu_model, gpu_db)
    num_gpus = max(1, int(api_inst.gpu or 1))

    # CPU RAM: API reports memory in GiB, store as bytes (binary).
    cpu_ram_bytes = int(api_inst.memory * _GIB)

    # NVMe: only count storage when it is reported and NVMe-backed.
    storage = api_inst.storage or {}
    if storage.get("nvme_ssd") and storage.get("size"):
        nvme_mem_bytes = int(storage["size"] * _GIB)
        nvme_bw_bytes = _compute_nvme_bw(
            storage.get("storage_read_iops"),
            storage.get("storage_write_iops"),
        )
    else:
        nvme_mem_bytes = 0
        nvme_bw_bytes = 0

    # PCIe bandwidth is per-GPU, aggregate linearly.
    pcie_bw_per_gpu = _PCIE_BW_PER_GPU_GB_S.get(gpu_name, 0.0)
    pcie_bw_bytes = int(pcie_bw_per_gpu * num_gpus * _GB)

    # NVLink bandwidth is per-GPU.  We store per-GPU in the JSON; the simulator
    # scales by num_gpus internally.
    nvlink_bw_per_gpu = _NVLINK_BW_PER_GPU_GB_S.get(gpu_name, 0.0)
    nvlink_bw_bytes = int(nvlink_bw_per_gpu * _GB)

    # On-demand Linux price in us-east-1 is the reference hourly price.
    pricing = api_inst.pricing
    region_pricing = pricing.get("us-east-1", pricing.get("us-east-2", {}))
    linux = region_pricing.get("linux", region_pricing.get("ubuntu", {}))
    dph_base = float(linux.get("ondemand", 0.0))

    # Internet up/down bandwidth.  EC2 ``network_performance`` describes the
    # instance's maximum aggregate network bandwidth, symmetric in both
    # directions, so inet up and down share the same value.
    inet_gbps = _parse_network_performance_gbps(api_inst.network_performance)
    network_inet_up = int(inet_gbps * _GB / 8)
    network_inet_down = int(inet_gbps * _GB / 8)

    # Inter-node (datacenter NIC) bandwidth for node-to-node KV transfers.
    # AWS does not publish a separate inter-node figure, so it inherits the
    # instance's network bandwidth (the same symmetric ENA/EFA aggregate),
    # doubled to reflect the typically higher datacenter-internal throughput.
    network_inter_node_up = network_inet_up * 2
    network_inter_node_down = network_inet_down * 2

    config: dict[str, Any] = {
        "name": f"AWS {itype} ({gpu_name} x{num_gpus})",
        "gpu_name": gpu_name,
        "num_gpus": num_gpus,
        "nvme_mem": nvme_mem_bytes,
        "nvme_bw": nvme_bw_bytes,
        "network_inet_up": network_inet_up,
        "network_inet_down": network_inet_down,
        "network_inter_node_up": network_inter_node_up,
        "network_inter_node_down": network_inter_node_down,
        "pcie_bw": pcie_bw_bytes,
        "cpu_ram": cpu_ram_bytes,
        "dph_base": dph_base,
    }
    if nvlink_bw_bytes > 0:
        config["nvlink_bw"] = nvlink_bw_bytes

    return config


def _backfill_inter_node(cfg: dict[str, Any]) -> dict[str, Any]:
    """Ensure an entry carries inter-node bandwidth fields (2x its inet)."""
    cfg = {**cfg}
    cfg["network_inter_node_up"] = cfg.get("network_inet_up", 0) * 2
    cfg["network_inter_node_down"] = cfg.get("network_inet_down", 0) * 2
    return cfg


def _build_aws_hardware(
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a new aws_hardware document from the Vantage API.

    Existing entries are preserved if the API lookup fails, but their fields are
    recomputed when a fresh API response is available.
    """
    client = APIV1Client("")
    gpu_db = _load_gpu_db()

    old_machines: dict[str, dict[str, Any]] = {}
    if existing and isinstance(existing.get("machines"), dict):
        old_machines = existing["machines"]

    machines: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    for name, old_cfg in old_machines.items():
        try:
            itype, expected_gpu = _extract_instance_type_and_gpu(name)
        except ValueError as exc:
            print(f"Skipping {name}: {exc}", file=sys.stderr)
            continue

        try:
            api_inst = client.get_global_instance("ec2", itype)
        except Exception as exc:
            print(
                f"API lookup failed for {name} ({itype}): {exc}; keeping old entry",
                file=sys.stderr,
            )
            machines[name] = _backfill_inter_node(old_cfg)
            missing.append(itype)
            continue

        try:
            new_cfg = _compute_machine_config(api_inst, expected_gpu, gpu_db)
        except Exception as exc:
            print(
                f"Config computation failed for {name} ({itype}): {exc}; keeping old entry",
                file=sys.stderr,
            )
            machines[name] = _backfill_inter_node(old_cfg)
            continue

        # Preserve the original key so downstream consumers stay stable.
        machines[name] = new_cfg

    if missing:
        print(
            f"Warning: {len(missing)} instance(s) could not be refreshed: {missing}",
            file=sys.stderr,
        )

    # Pricing metadata is intentionally left empty here; it is generated by
    # ``scripts/derive_family_pricing.py`` from the refreshed machine data.
    return {"machines": machines}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh AWS hardware presets from the Vantage Instances API."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated JSON instead of writing it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=AWS_HARDWARE_PATH,
        help=f"Output path (default: {AWS_HARDWARE_PATH})",
    )
    args = parser.parse_args()

    existing: dict[str, Any] | None = None
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))

    data = _build_aws_hardware(existing)

    if args.dry_run:
        print(json.dumps(data, indent=2))
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {len(data['machines'])} machine preset(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
