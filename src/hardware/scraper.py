"""GPU spec lookup backed by a cached JSON file.

The module reads from ``_gpu_db.json`` by default, avoiding any network
calls at import or runtime.  A ``refresh_file()`` helper is provided to
re-populate the file from Vast.ai when needed.
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
    url = f"{_PRICING_PAGE}/{gpu_name}"
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

    data = json.loads(script_tag.string)
    gpu_details = data.get("props", {}).get("pageProps", {}).get("gpuDetails")
    if not gpu_details:
        msg = f"No gpuDetails found for {gpu_name!r}"
        raise ValueError(msg)

    fp32 = gpu_details.get("FP32 (float)")
    mem_size = gpu_details.get("Memory Size")
    bandwidth = gpu_details.get("Bandwidth")

    missing = [
        k
        for k, v in {
            "FP32 (float)": fp32,
            "Memory Size": mem_size,
            "Bandwidth": bandwidth,
        }.items()
        if not v
    ]
    if missing:
        msg = f"Missing required fields for {gpu_name!r}: {missing}"
        raise ValueError(msg)

    return {
        "name": gpu_name,
        "flops": _to_flops(fp32),
        "gpu_mem": _to_bytes(mem_size),
        "gpu_bw": _to_bytes_per_second(bandwidth),
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


def refresh_file(gpu_names: list[str]) -> None:
    """Re-scrape Vast.ai and overwrite ``_gpu_db.json``.

    This makes one GET to ``/pricing/gpu/<name>`` per GPU and one GET to
    ``/api/v0/bundles`` shared across all GPUs.

    Args:
        gpu_names: GPU identifiers to collect, e.g. ``["B300", "B200"]``.
    """
    db: dict[str, dict[str, Any]] = {}
    for name in gpu_names:
        specs = _fetch_specs(name)
        specs["price_usd_per_hour"] = _fetch_live_price(name)
        db[name] = specs

    _DB_PATH.write_text(json.dumps(db, indent=2), encoding="utf-8")


def _load_db() -> dict[str, dict[str, Any]]:
    if not _DB_PATH.exists():
        msg = f"GPU database not found at {_DB_PATH}. Run refresh_file([...]) first."
        raise RuntimeError(msg)
    return json.loads(_DB_PATH.read_text(encoding="utf-8"))


def lookup(gpu_name: str) -> dict[str, Any]:
    """Return cached specs for *gpu_name* from ``_gpu_db.json``."""
    db = _load_db()
    try:
        return db[gpu_name]
    except KeyError:
        msg = f"GPU {gpu_name!r} not in database. Run refresh_file(['{gpu_name}'])."
        raise KeyError(msg) from None


def fetch_hardware(gpu_name: str) -> "Hardware":  # noqa: F821, UP037
    """Build a :class:`~hardware.hardware.Hardware` instance from the file cache.

    Fields that are not provided by vast.ai (RAM, NVMe, network) are
    defaulted to ``0`` so the caller can fill them in afterwards.
    """
    from .hardware import Hardware

    scraped = lookup(gpu_name)
    return Hardware(
        name=scraped["name"],
        flops=scraped["flops"],
        gpu_mem=scraped["gpu_mem"],
        gpu_bw=scraped["gpu_bw"],
        ram_mem=0,
        ram_bw=0,
        nvme_mem=0,
        nvme_bw=0,
        network_bw=0,
        price_usd_per_hour=scraped["price_usd_per_hour"],
    )
