"""Vast.ai web scraping helpers (legacy / optional).

This module contains the network-dependent parts of the hardware scraper.
They are kept separate so the core simulator can run entirely from the
local JSON caches (``_gpu_db.json`` and ``_machine_db.json``) without any
external HTTP requests.

``refresh_file()`` and ``refresh_machines_file()`` can still be called to
re-populate those caches from Vast.ai when desired.
"""

import json
import pathlib
import re

from typing import Any

import requests

from bs4 import BeautifulSoup


_PRICING_PAGE = "https://vast.ai/pricing/gpu"
_MARKET_API = "https://vast.ai/api/v0/bundles"
_DB_PATH = pathlib.Path(__file__).parent / "_gpu_db.json"
_MACHINE_DB_PATH = pathlib.Path(__file__).parent / "_machine_db.json"


def _to_url_slug(gpu_name: str) -> str:
    """Convert a raw GPU name into the slug used by vast.ai pricing URLs.

    Examples:
        >>> _to_url_slug("Tesla V100")
        'TESLA-V100'
        >>> _to_url_slug("RTX PRO 6000 WS")
        'RTX-PRO-6000-WS'
    """
    slug = gpu_name.upper().replace(" ", "-").replace("_", "-")
    slug = re.sub(r"[^A-Z0-9-]", "", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _parse_number_with_unit(value: str) -> tuple[float, str]:
    """Parse a numeric value with a unit suffix.

    Examples:
        >>> _parse_number_with_unit("288 GB")
        (288.0, 'GB')
        >>> _parse_number_with_unit("8.00 TB/s")
        (8.0, 'TB/s')
    """
    match = re.match(r"([\d,.]+)\s*(\S+)", value.strip())
    if not match:
        msg = f"Cannot parse value: {value!r}"
        raise ValueError(msg)
    number = float(match.group(1).replace(",", ""))
    unit = match.group(2)
    return number, unit


def _to_bytes(value: str) -> int:
    """Convert a memory-size string (e.g. "288 GB") to bytes."""
    number, unit = _parse_number_with_unit(value)
    multipliers = {
        "GB": 10**9,
        "TB": 10**12,
        "MB": 10**6,
        "KB": 10**3,
    }
    if unit not in multipliers:
        msg = f"Unexpected memory unit: {unit!r}"
        raise ValueError(msg)
    return int(number * multipliers[unit])


def _to_bytes_per_second(value: str) -> int:
    """Convert a bandwidth string (e.g. "8.00 TB/s") to bytes / s."""
    number, unit = _parse_number_with_unit(value)
    base = unit.removesuffix("/s")
    multipliers = {
        "TB": 10**12,
        "GB": 10**9,
        "MB": 10**6,
    }
    if base not in multipliers:
        msg = f"Unexpected bandwidth unit: {unit!r}"
        raise ValueError(msg)
    return int(number * multipliers[base])


def _to_flops(value: str) -> int:
    """Convert a FLOPS string (e.g. "75.0 TFLOPS") to raw FLOPS count."""
    number, unit = _parse_number_with_unit(value)
    multipliers = {
        "TFLOPS": 10**12,
        "GFLOPS": 10**9,
        "PFLOPS": 10**15,
    }
    if unit not in multipliers:
        msg = f"Unexpected FLOPS unit: {unit!r}"
        raise ValueError(msg)
    return int(number * multipliers[unit])


def _fetch_specs(gpu_name: str) -> dict[str, Any]:
    """Pull static hardware specs from the GPU detail page."""
    url = f"{_PRICING_PAGE}/{_to_url_slug(gpu_name)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    script_tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if not script_tag or not getattr(script_tag, "string", None):
        msg = f"__NEXT_DATA__ script tag not found on {url}"
        raise ValueError(msg)

    data = json.loads(str(script_tag.string))
    gpu_details = data.get("props", {}).get("pageProps", {}).get("gpuDetails")
    if not gpu_details:
        msg = f"No gpuDetails found for {gpu_name!r}"
        raise ValueError(msg)

    fp16 = gpu_details.get("FP16 (half)")
    mem_size = gpu_details.get("Memory Size")
    bandwidth = gpu_details.get("Bandwidth")

    missing = [
        k
        for k, v in {
            "FP16 (half)": fp16,
            "Memory Size": mem_size,
            "Bandwidth": bandwidth,
        }.items()
        if not v
    ]
    if missing:
        msg = f"Missing required fields for {gpu_name!r}: {missing, gpu_details}"
        raise ValueError(msg)

    return {
        "name": gpu_name,
        "flops": _to_flops(fp16),
        "gpu_mem": _to_bytes(mem_size),
        "gpu_bw": _to_bytes_per_second(bandwidth),
        # The GPU detail page does not expose power/temperature details.
        "gpu_max_power": 0.0,
        "gpu_max_temp": 0.0,
    }


def _fetch_live_price(gpu_name: str) -> float:
    """Fetch the average marketplace price for one *full* GPU (USD / hour).

    Queries ``/api/v0/bundles`` and averages ``gpuCostPerHour`` normalised
    by the effective GPU count (``num_gpus * gpu_frac``).  This gives the
    price of renting enough fractional slots to equal one whole physical GPU
    with 100%% of its FLOPS, VRAM and bandwidth.

    Prefer rentable listings; fall back to all matching listings if none
    are rentable.  Returns ``0.0`` when no listings exist.
    """
    resp = requests.get(_MARKET_API, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    offers = data.get("offers", [])

    all_matching = [
        o for o in offers if gpu_name.upper() in o.get("gpu_name", "").upper()
    ]
    if not all_matching:
        return 0.0

    rentable = [o for o in all_matching if o.get("rentable")]
    pool = rentable if rentable else all_matching

    prices: list[float] = []
    for o in pool:
        gpu_cost = o.get("search", {}).get("gpuCostPerHour")
        if gpu_cost is None:
            gpu_cost = o.get("dph_base")
        if gpu_cost is None:
            continue

        num_gpus = o.get("num_gpus", 1)
        if num_gpus <= 0:
            continue

        prices.append(gpu_cost / num_gpus)

    return sum(prices) / len(prices) if prices else 0.0


def refresh_file(gpu_names: list[str] | None = None) -> None:
    """Re-scrape Vast.ai and overwrite ``_gpu_db.json``.

    This makes one GET to ``/pricing/gpu/<name>`` per GPU and one GET to
    ``/api/v0/bundles`` shared across all GPUs.

    Args:
        gpu_names: GPU identifiers to collect, e.g. ``["B300", "B200"]``.
    """
    db: dict[str, dict[str, Any]] = {}
    prior_content = _DB_PATH.read_text(
        encoding="utf-8"
    )  # Ensure file exists before writing.
    if prior_content:
        db = json.loads(prior_content)

    if not gpu_names:
        gpu_names = [
            "B200",
            "B300",
            "H200",
            "H200_NVL",
            "H100_NVL",
            "H100_SXM",
            "H100_PCIE",
            "RTX_PRO_6000_S",
            "RTX_PRO_6000_WS",
            "RTX_5090",
            "RTX_4090",
            "RTX_PRO_4000",
            "RTX_5080",
            "RTX_5070_TI",
            "RTX_5070",
            "RTX_5060_TI",
            "RTX_5060",
            "L40S",
            "L4",
            "RTX_4080",
            "RTX_4080S",
            "RTX_4070S_TI",
            "RTX_4060_TI",
            "RTX_4070",
            "RTX_4070_TI",
            "RTX_4070S",
            "RTX_4060",
            "A100_SXM4",
            "A100_PCIE",
            "RTX_A6000",
            "A40",
            "A10",
            "RTX_A5000",
            "RTX_3090_TI",
            "RTX_3090",
            "RTX_3080_TI",
            "RTX_3080",
            "RTX_A4000",
            "RTX_3070",
            "RTX_3060_LAPTOP",
            "RTX_3060_TI",
            "RTX_3060",
            "RTX_A2000",
            "Q_RTX_8000",
            "Q_RTX_6000",
            "RTX_2060S",
            "RTX_2080_TI",
            "TESLA_V100",
            "GTX_1080_TI",
            "QUADRO_P4000",
            "TITAN_XP",
            "GTX_1070_TI",
            "GTX_1080",
            "RTX_PRO_5000",
            "RTX_6000_Ada",
            "RTX_5880_Ada",
            "GTX_1660_S",
        ]

    for name in gpu_names:
        try:
            specs = _fetch_specs(name)
        except Exception:
            print("GPU not found/ no fp16 flops", name)
            continue
        else:
            print("Fetched ", name)
        specs["price_usd_per_hour"] = _fetch_live_price(name)
        name = name.replace("_", " ")
        db[name] = specs

    _DB_PATH.write_text(json.dumps(db, indent=2), encoding="utf-8")


def _make_machine_name(offer: dict[str, Any]) -> str:
    gpu_name = offer.get("gpu_name", "Unknown")
    num_gpus = offer.get("num_gpus", 0)
    # Create a short hash of the offer so every unique listing is kept.
    offer_hash = hash(json.dumps(offer, sort_keys=True, default=str)) & 0xFFFF_FFFF
    return f"{gpu_name} x{num_gpus} #{offer_hash:08x}"


def _extract_machine(offer: dict[str, Any]) -> dict[str, Any]:
    """Extract all numeric and descriptive fields from a Vast.ai offer."""
    num_gpus = offer.get("num_gpus", 1)
    if num_gpus <= 0:
        num_gpus = 1

    return {
        "name": _make_machine_name(offer),
        "gpu_name": offer.get("gpu_name", ""),
        "num_gpus": num_gpus,
        # "flops": offer.get("total_flops", 0),
        # "gpu_mem": offer.get("gpu_ram", 0) * 1024**2,
        # "gpu_bw": offer.get("gpu_mem_bw", 0) * 1024**3,
        # "gpu_max_power": offer.get("gpu_max_power", 0.0),
        # "gpu_max_temp": offer.get("gpu_max_temp", 0.0),
        "cpu_cores": offer.get("cpu_cores", 0),
        "cpu_cores_effective": offer.get("cpu_cores_effective", 0.0),
        "cpu_ghz": offer.get("cpu_ghz", 0.0),
        "cpu_name": offer.get("cpu_name", ""),
        "cpu_ram": offer.get("cpu_ram", 0) * 1024**2,
        "disk_name": offer.get("disk_name", ""),
        "dlperf": offer.get("dlperf", 0.0),
        "dlperf_per_dphtotal": offer.get("dlperf_per_dphtotal", 0.0),
        "dph_base": offer.get("dph_total", 0.0),
        "geolocation": offer.get("geolocation", ""),
        "gpu_display_active": offer.get("gpu_display_active", False),
        "gpu_frac": offer.get("gpu_frac", 1.0),
        "gpu_lanes": offer.get("gpu_lanes", 0),
        "has_avx": offer.get("has_avx", 0),
        "host_id": offer.get("host_id", 0),
        "inet_down_cost": offer.get("inet_down_cost", 0.0),
        "inet_up_cost": offer.get("inet_up_cost", 0.0),
        "mobo_name": offer.get("mobo_name", ""),
        "os_version": offer.get("os_version", ""),
        "pci_gen": offer.get("pci_gen", 0.0),
        "pcie_bw": offer.get("pcie_bw", 0.0) * 1024**3,
        "nvme_mem": offer.get("disk_space", 0.0) * 1024**3,
        "nvme_bw": offer.get("disk_bw", 0.0) * 1024**2,
        "network_bw": offer.get("bw_nvlink", 0.0) * 1024**3,
        "network_inet_up": offer.get("inet_up", 0.0) * 1024**2 / 8,
        "network_inet_down": offer.get("inet_down", 0.0) * 1024**2 / 8,
        "reliability": offer.get("reliability", 0.0),
        "reliability_mult": offer.get("reliability_mult", 0.0),
        "score": offer.get("score", 0.0),
        "storage_cost": offer.get("storage_cost", 0.0) / 1024**3,
        "storage_total_cost": offer.get("storage_total_cost", 0.0) / 1024**3,
        "verification": offer.get("verification", ""),
    }


def refresh_machines_file() -> None:
    """Fetch all live Vast.ai offers and write machine configs to ``_machine_db.json``."""
    resp = requests.get(_MARKET_API, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    offers = data.get("offers", [])

    db: dict[str, dict[str, Any]] = {}
    for offer in offers:
        machine = _extract_machine(offer)
        key = machine["name"]
        if key not in db:
            db[key] = machine

    _MACHINE_DB_PATH.write_text(json.dumps(db, indent=2), encoding="utf-8")
