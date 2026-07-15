"""GPU spec lookup backed by a cached JSON file.

The module reads from ``_gpu_db.json`` by default, avoiding any network
calls at import or runtime.  A ``refresh_file()`` helper is provided to
re-populate the file from Vast.ai when needed.
"""

import json
import os
import pathlib
import re

from typing import Any

import requests

from bs4 import BeautifulSoup

from src.hardware.hardware import Hardware


_PRICING_PAGE = "https://vast.ai/pricing/gpu"
_MARKET_API = "https://vast.ai/api/v0/bundles"

_DB_PATH = pathlib.Path(__file__).parent / "_gpu_db.json"


def _resolve_inter_node_bw(value: str | None, default_gbps: float = 100.0) -> int:
    """Resolve an optional Gbps override to bytes per second.

    Falls back to ``default_gbps`` when ``value`` is None or empty.
    """
    if value is None or value.strip() == "":
        return int(default_gbps * 1e9 / 8.0)
    return int(float(value) * 1e9 / 8.0)


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


def load_gpu_db() -> dict[str, dict[str, Any]]:
    if not _DB_PATH.exists():
        msg = f"GPU database not found at {_DB_PATH}. Run refresh_file([...]) first."
        raise RuntimeError(msg)
    return json.loads(_DB_PATH.read_text(encoding="utf-8"))


def lookup(gpu_name: str) -> dict[str, Any]:
    """Return cached specs for *gpu_name* from ``_gpu_db.json``."""
    db = load_gpu_db()
    try:
        return db[gpu_name]
    except KeyError:
        msg = f"GPU {gpu_name!r} not in database. Run refresh_file(['{gpu_name}'])."
        raise KeyError(msg) from None


# ---------------------------------------------------------------------------
# Machine / node scraper backed by ``_machine_db.json``
# ---------------------------------------------------------------------------

_MACHINE_DB_PATH = pathlib.Path(__file__).parent / "_machine_db.json"
_CUSTOM_HARDWARE_PATH = pathlib.Path(__file__).parent / "_custom_hardware.json"


def load_custom_hardware_db(
    path: pathlib.Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load user-supplied custom hardware definitions.

    The file is a JSON object with two top-level keys:

    * ``_pricing`` (optional): global unit prices used to derive hourly cost
      for custom machines.

      * ``cpu_ram_usd_per_gb_hour`` USD per GB of CPU RAM per hour.
      * ``ssd_usd_per_gb_hour`` USD per GB of SSD storage per hour.
      * ``inet_up_usd_per_gb_hour`` USD per GB/hour of internet upload
        bandwidth.
      * ``inet_down_usd_per_gb_hour`` USD per GB/hour of internet download
        bandwidth.
      * ``inter_node_up_usd_per_gb_hour`` USD per GB/hour of datacenter NIC
        upload bandwidth.
      * ``inter_node_down_usd_per_gb_hour`` USD per GB/hour of datacenter NIC
        download bandwidth.
      * ``pcie_usd_per_gb_hour`` USD per GB/hour of PCIe bandwidth.

    * ``machines``: mapping of hardware names to config dicts.  Each config
      must provide ``gpu_name`` so the GPU spec can be resolved from the cached
      GPU database, plus a ``gpu_price_usd_per_hour`` for that specific machine.
      Any remaining field is filled with safe defaults.

    Any field absent in a machine config is filled with sensible defaults
    before being passed to :func:`fetch_machine_hardware`.

    Parameters
    ----------
    path:
        Path to the custom hardware JSON file.  If ``None``, the default
        ``src/hardware/_custom_hardware.json`` is used.

    Returns:
    -------
    Tuple ``(pricing, machines)`` where ``pricing`` is the pricing dict (or
    empty) and ``machines`` is a name -> config dict.  When the file does not
    exist, ``({}, {})`` is returned.
    """
    target = pathlib.Path(path) if path is not None else _CUSTOM_HARDWARE_PATH
    if not target.exists():
        return {}, {}
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Custom hardware file {target} must contain a JSON object")

    pricing = data.get("_pricing", {})
    if not isinstance(pricing, dict):
        raise ValueError(
            f"Custom hardware file {target}: '_pricing' must be a JSON object"
        )

    raw_machines = data.get("machines", {})
    if not isinstance(raw_machines, dict):
        raise ValueError(
            f"Custom hardware file {target}: 'machines' must be a JSON object mapping names to configs"
        )

    # Normalise each entry: ensure a name key exists.
    normalized: dict[str, dict[str, Any]] = {}
    for name, config in raw_machines.items():
        if not isinstance(config, dict):
            raise ValueError(
                f"Custom hardware entry {name!r} in {target} must be a JSON object"
            )
        if "name" not in config:
            config = {**config, "name": name}
        normalized[name] = config
    return pricing, normalized


def _default_custom_hardware_path() -> pathlib.Path:
    """Return the default location for the custom hardware file."""
    return _CUSTOM_HARDWARE_PATH


# Per-byte / per-second constants used for unit-price cost derivation.
_GB = 1024**3
_HOUR_S = 3600.0


def _derive_custom_price(
    config: dict[str, Any],
    pricing: dict[str, float],
) -> float:
    """Derive an hourly price for a custom machine from unit prices.

    The price is the sum of:

    * ``gpu_price_usd_per_hour`` (machine-specific, required in the config)
    * CPU RAM: ``cpu_ram`` bytes * ``cpu_ram_usd_per_gb_hour`` / GB
    * SSD: ``nvme_mem`` bytes * ``ssd_usd_per_gb_hour`` / GB
    * Internet upload/download: respective bandwidths (bytes/s) converted to
      GB/hour and multiplied by the corresponding unit prices.
    * Inter-node upload/download: same conversion.
    * PCIe: ``pcie_bw`` (bytes/s) converted to GB/hour.

    Parameters
    ----------
    config:
        A machine config dict (already normalised/merged with defaults).
    pricing:
        The ``_pricing`` dict from the custom hardware file.

    Returns:
    -------
    Hourly price in USD.
    """
    gpu_price = float(config.get("gpu_price_usd_per_hour", 0.0))

    def unit(key: str) -> float:
        return float(pricing.get(key, 0.0))

    def bytes_to_gb_h(val: float) -> float:
        return float(val) / _GB * _HOUR_S

    cpu_ram_gb = float(config.get("cpu_ram", 0)) / _GB
    ssd_gb = float(config.get("nvme_mem", 0)) / _GB

    price = gpu_price
    price += cpu_ram_gb * unit("cpu_ram_usd_per_gb_hour")
    price += ssd_gb * unit("ssd_usd_per_gb_hour")
    price += bytes_to_gb_h(config.get("network_inet_up", 0)) * unit(
        "inet_up_usd_per_gb_hour"
    )
    price += bytes_to_gb_h(config.get("network_inet_down", 0)) * unit(
        "inet_down_usd_per_gb_hour"
    )
    price += bytes_to_gb_h(config.get("network_inter_node_up", 0)) * unit(
        "inter_node_up_usd_per_gb_hour"
    )
    price += bytes_to_gb_h(config.get("network_inter_node_down", 0)) * unit(
        "inter_node_down_usd_per_gb_hour"
    )
    price += bytes_to_gb_h(config.get("pcie_bw", 0)) * unit("pcie_usd_per_gb_hour")

    return price


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


def load_machine_db() -> dict[str, dict[str, Any]]:
    if not _MACHINE_DB_PATH.exists():
        msg = (
            f"Machine database not found at {_MACHINE_DB_PATH}. "
            "Run refresh_machines_file() first."
        )
        raise RuntimeError(msg)
    return json.loads(_MACHINE_DB_PATH.read_text(encoding="utf-8"))


def parse_gpu_count(machine_name: str) -> int:
    """Extract the GPU count from a machine key like ``RTX 5090 x2 #...``.

    Returns the integer after ``x`` in the key. Falls back to 1 when the
    pattern is not present.
    """
    match = re.search(r"\bx(\d+)\b", machine_name)
    if match:
        return int(match.group(1))
    return 1


def load_combined_machine_db(
    custom_path: pathlib.Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return machine database with custom hardware entries merged in.

    Custom hardware (from :func:`load_custom_hardware_db`) takes precedence
    over entries in ``_machine_db.json``.  This lets users define local
    hardware presets without modifying the scraped database.
    """
    db = load_machine_db()
    _, custom = load_custom_hardware_db(custom_path)
    if custom:
        db = {**db, **custom}
    return db


def resolve_machine_name(
    machine_name: str, custom_path: pathlib.Path | str | None = None
) -> str:
    """Resolve ``machine_name`` to an exact machine key from the local cache.

    * If ``machine_name`` is already an exact key, return it unchanged.
    * Otherwise search entries whose ``gpu_name`` contains ``machine_name`` as a
      case-insensitive substring.  A single match is returned; zero or multiple
      matches raise ``ValueError`` with helpful context.

    Custom hardware entries are consulted first so user-defined presets take
    precedence over the scraped database.
    """
    db = load_combined_machine_db(custom_path)
    if machine_name in db:
        return machine_name

    query = machine_name.lower()
    matches = [
        key for key, config in db.items() if query in config.get("gpu_name", "").lower()
    ]
    if not matches:
        available = sorted({config.get("gpu_name", key) for key, config in db.items()})
        raise ValueError(
            f"No machine matching {machine_name!r} found in local database. "
            f"Available GPU names: {available}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple machines match {machine_name!r}: {sorted(matches)}. "
            "Please pass an exact instance name."
        )
    return matches[0]


def lookup_machine(
    machine_name: str, custom_path: pathlib.Path | str | None = None
) -> dict[str, Any]:
    """Return cached machine config, checking custom hardware first."""
    db = load_combined_machine_db(custom_path)
    try:
        return db[machine_name]
    except KeyError:
        msg = f"Machine {machine_name!r} not in database. Run refresh_machines_file()."
        raise KeyError(msg) from None


def _machine_config_with_defaults(
    config: dict[str, Any],
    pricing: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fill missing machine-config fields with safe defaults.

    Custom hardware entries only need to specify the values that matter for
    simulation; everything else is zeroed or defaulted so ``HardwareSpec``
    construction still succeeds.

    For custom entries (those without a scraped ``dph_base``), if a pricing
    dict is supplied the hourly price is derived from unit prices and the
    machine's resources and written into ``dph_base``.
    """
    defaults: dict[str, Any] = {
        "cpu_cores": 0,
        "cpu_cores_effective": 0.0,
        "cpu_ghz": 0.0,
        "cpu_name": "",
        "disk_name": "",
        "dlperf": 0.0,
        "dlperf_per_dphtotal": 0.0,
        "dph_base": 0.0,
        "geolocation": "",
        "gpu_display_active": False,
        "gpu_frac": 1.0,
        "gpu_lanes": 0,
        "gpu_max_power": 0.0,
        "gpu_max_temp": 0.0,
        "has_avx": 0,
        "host_id": 0,
        "inet_down_cost": 0.0,
        "inet_up_cost": 0.0,
        "mobo_name": "",
        "os_version": "",
        "pci_gen": 0.0,
        "network_bw": 0.0,
        "reliability": 0.0,
        "reliability_mult": 0.0,
        "score": 0.0,
        "storage_cost": 0.0,
        "storage_total_cost": 0.0,
        "verification": "",
    }
    merged = {**defaults, **config}
    # Fields that must exist and cannot be defaulted.
    for required in (
        "name",
        "gpu_name",
        "num_gpus",
        "nvme_mem",
        "nvme_bw",
        "network_inet_up",
        "network_inet_down",
        "pcie_bw",
        "cpu_ram",
    ):
        if required not in merged or merged[required] is None:
            raise ValueError(
                f"Custom hardware {merged.get('name', config)!r} is missing required field {required!r}"
            )

    # Derive price for custom entries when pricing metadata is provided.
    if pricing and not merged.get("dph_base"):
        merged["dph_base"] = _derive_custom_price(merged, pricing)

    return merged


def fetch_machine_hardware(
    machine_name: str,
    machine_config_override: dict[str, Any] | None = None,
    custom_path: pathlib.Path | str | None = None,
) -> Hardware:
    """Build a :class:`~hardware.hardware.Hardware` instance from the machine cache.

    Parameters
    ----------
    machine_name:
        Exact key in ``_machine_db.json`` or a key in the custom hardware file.
    machine_config_override:
        Optional machine config dict to use instead of looking up
        ``machine_name`` in the database.  This allows callers such as
        ``mixed_gpu`` to inject a modified config while still using the
        normal GPU lookup and bandwidth-resolution logic.
    custom_path:
        Optional path to a custom hardware JSON file.  When provided, custom
        entries are looked up before the scraped machine database.
    """
    from .hardware import GPUHardwareSpec, Hardware, HardwareSpec

    pricing, _ = load_custom_hardware_db(custom_path)
    if machine_config_override is not None:
        scraped = _machine_config_with_defaults(machine_config_override, pricing)
    else:
        scraped = _machine_config_with_defaults(
            lookup_machine(machine_name, custom_path), pricing
        )
    gpu = lookup(scraped["gpu_name"])
    if not gpu:
        gpu = _fetch_specs(scraped["gpu_name"])

    gpu_spec = GPUHardwareSpec(
        flops=gpu["flops"],
        gpu_mem=gpu["gpu_mem"],
        gpu_bw=gpu["gpu_bw"],
    )

    # Network bandwidths.  Internet up/down come from the machine DB; inter-node
    # (datacenter NIC) bandwidths default to 100 Gb/s and can be overridden via
    # environment variables / CLI / webserver.
    # Internet up/down use the scraped machine DB values.
    network_inet_up = int(scraped.get("network_inet_up", 0.0))
    network_inet_down = int(scraped.get("network_inet_down", 0.0))
    network_inter_node_up = _resolve_inter_node_bw(
        os.environ.get("INTER_NODE_NETWORK_UP_GBPS"),
        default_gbps=100.0,
    )
    network_inter_node_down = _resolve_inter_node_bw(
        os.environ.get("INTER_NODE_NETWORK_DOWN_GBPS"),
        default_gbps=100.0,
    )

    spec = HardwareSpec(
        gpu_hardware=gpu_spec,
        num_gpus=scraped["num_gpus"],
        nvme_mem=scraped["nvme_mem"],
        nvme_bw=scraped["nvme_bw"],
        network_inet_up=network_inet_up,
        network_inet_down=network_inet_down,
        network_inter_node_up=network_inter_node_up,
        network_inter_node_down=network_inter_node_down,
        cpu_cores=scraped.get("cpu_cores", 0),
        cpu_cores_effective=scraped.get("cpu_cores_effective", 0.0),
        cpu_ghz=scraped.get("cpu_ghz", 0.0),
        cpu_name=scraped.get("cpu_name", ""),
        cpu_ram=scraped.get("cpu_ram", 0),
        disk_name=scraped.get("disk_name", ""),
        dlperf=scraped.get("dlperf", 0.0),
        dlperf_per_dphtotal=scraped.get("dlperf_per_dphtotal", 0.0),
        dph_base=scraped.get("dph_base", 0.0),
        geolocation=scraped.get("geolocation", ""),
        gpu_display_active=scraped.get("gpu_display_active", False),
        gpu_frac=scraped.get("gpu_frac", 1.0),
        gpu_lanes=scraped.get("gpu_lanes", 0),
        gpu_max_power=scraped.get("gpu_max_power", 0.0),
        gpu_max_temp=scraped.get("gpu_max_temp", 0.0),
        has_avx=scraped.get("has_avx", 0),
        host_id=scraped.get("host_id", 0),
        inet_down_cost=scraped.get("inet_down_cost", 0.0),
        inet_up_cost=scraped.get("inet_up_cost", 0.0),
        mobo_name=scraped.get("mobo_name", ""),
        os_version=scraped.get("os_version", ""),
        pci_gen=scraped.get("pci_gen", 0.0),
        pcie_bw=scraped.get("pcie_bw", 0.0),
        network_bw=scraped.get("network_bw", 0.0),
        reliability=scraped.get("reliability", 0.0),
        reliability_mult=scraped.get("reliability_mult", 0.0),
        score=scraped.get("score", 0.0),
        storage_cost=scraped.get("storage_cost", 0.0),
        storage_total_cost=scraped.get("storage_total_cost", 0.0),
        verification=scraped.get("verification", ""),
    )
    return Hardware(name=scraped["name"], spec=spec)
