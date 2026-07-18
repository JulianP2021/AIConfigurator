"""GPU / machine spec lookup backed by local JSON files.

The module reads from ``_gpu_db.json`` and ``_machine_db.json`` by default,
avoiding any network calls at import or runtime.  The network-dependent
Vast.ai scrapers live in ``src.hardware.legacy.vast_scraper`` and are only
used when explicitly refreshing the cached JSON files.
"""

import json
import os
import pathlib
import re

from typing import Any

from src.hardware.hardware import Hardware
from src.hardware.legacy.vast_scraper import _fetch_specs


_GPU_DB_PATH = pathlib.Path(__file__).parent / "legacy" / "_gpu_db.json"


def _resolve_inter_node_bw(value: str | None, default_gbps: float = 100.0) -> int:
    """Resolve an optional Gbps override to bytes per second.

    Falls back to ``default_gbps`` when ``value`` is None or empty.
    """
    if value is None or value.strip() == "":
        return int(default_gbps * 1e9 / 8.0)
    return int(float(value) * 1e9 / 8.0)


def load_gpu_db() -> dict[str, dict[str, Any]]:
    if not _GPU_DB_PATH.exists():
        msg = f"GPU database not found at {_GPU_DB_PATH}. Run legacy.vast_scraper.refresh_file([...]) first."
        raise RuntimeError(msg)
    return json.loads(_GPU_DB_PATH.read_text(encoding="utf-8"))


def lookup(gpu_name: str) -> dict[str, Any]:
    """Return cached specs for *gpu_name* from ``_gpu_db.json``."""
    db = load_gpu_db()
    try:
        return db[gpu_name]
    except KeyError:
        msg = f"GPU {gpu_name!r} not in database. Run legacy.vast_scraper.refresh_file(['{gpu_name}'])."
        raise KeyError(msg) from None


# ---------------------------------------------------------------------------
# Machine / node scraper backed by ``_machine_db.json``
# ---------------------------------------------------------------------------

_MACHINE_DB_PATH = pathlib.Path(__file__).parent / "legacy" / "_machine_db.json"
_AWS_HARDWARE_PATH = pathlib.Path(__file__).parent / "aws_hardware.json"


def load_aws_hardware_db(
    path: pathlib.Path | str | None = None,
) -> tuple[dict[Any, Any], dict[str, dict[str, Any]]]:
    """Load user-supplied custom hardware definitions.

    The file is a JSON object with two top-level keys:

    * ``_pricing`` (optional): global unit prices used to derive hourly cost
      for custom machines.

      * ``cpu_ram_usd_per_gb_hour`` USD per GB of CPU RAM per hour.
      * ``ssd_usd_per_gb_hour`` USD per GB of SSD storage per hour.
      * ``ssd_bw_usd_per_gb_s_hour`` USD per GB/s per hour of SSD (NVMe)
        bandwidth.
      * ``inet_up_usd_per_gbps_hour`` USD per Gbps per hour of internet upload
        bandwidth.
      * ``inet_down_usd_per_gbps_hour`` USD per Gbps per hour of internet download
        bandwidth.
      * ``inter_node_up_usd_per_gbps_hour`` USD per Gbps per hour of datacenter
        NIC upload bandwidth.
      * ``inter_node_down_usd_per_gbps_hour`` USD per Gbps per hour of datacenter
        NIC download bandwidth.
      * ``pcie_bw_usd_per_gb_s_hour`` USD per GB/s per hour of PCIe bandwidth.
      * ``nvlink_bw_usd_per_gb_s_hour`` USD per GB/s per hour of NVLink/C2C
        bandwidth.

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
    target = pathlib.Path(path) if path is not None else _AWS_HARDWARE_PATH
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
    return _AWS_HARDWARE_PATH


# Per-byte / per-second constants used for unit-price cost derivation.
_GB = 1024**3
_HOUR_S = 3600.0


def _derive_custom_price(
    config: dict[str, Any],
    pricing: dict[str, Any],
) -> float:
    """Derive an hourly price for a custom machine from unit prices.

    If the config specifies a ``gpu_name`` that exists in the per-family pricing
    table, the per-family component prices are used (with global prices as
    fallback for missing components).  Otherwise the legacy path is used:

    * ``gpu_price_usd_per_hour`` (machine-specific)
    * global CPU RAM, SSD, bandwidth, internet, inter-node and PCIe prices.

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

    def unit(key: str) -> float:
        return float(pricing.get(key, 0.0))

    def bytes_to_gb_h(val: float) -> float:
        return float(val) / _GB * _HOUR_S

    def bytes_to_gbps(val: float) -> float:
        return float(val) * 8.0 / 1e9

    gpu_name = config.get("gpu_name", "")
    family_pricing = pricing.get("gpu_family_pricing", {})
    family = family_pricing.get(gpu_name, {}) if gpu_name else {}

    def family_or_global(key: str) -> float:
        """Return family price if positive, otherwise the global fallback.

        Per-family component tables set values to 0.0 when a cost is bundled
        into the per-GPU compute price. For custom machines we still want to
        apply a global unit price when one is configured.
        """
        val = float(family.get(key, 0.0))
        return val if val > 0 else float(pricing.get(key, 0.0))

    num_gpus = int(config.get("num_gpus", 1))
    cpu_ram_gb = float(config.get("cpu_ram", 0)) / _GB
    ssd_gb = float(config.get("nvme_mem", 0)) / _GB

    # Per-family GPU compute price, otherwise legacy machine-specific GPU price.
    if "compute_usd_per_gpu_hour" in family:
        price = num_gpus * family["compute_usd_per_gpu_hour"]
    else:
        price = float(config.get("gpu_price_usd_per_hour", 0.0))

    price += cpu_ram_gb * family_or_global("cpu_ram_usd_per_gb_hour")
    price += ssd_gb * family_or_global("ssd_usd_per_gb_hour")
    price += (float(config.get("nvme_bw", 0)) / _GB) * family_or_global(
        "ssd_bw_usd_per_gb_s_hour"
    )
    price += bytes_to_gbps(config.get("network_inet_up", 0)) * family_or_global(
        "inet_up_usd_per_gbps_hour"
    )
    price += bytes_to_gbps(config.get("network_inet_down", 0)) * family_or_global(
        "inet_down_usd_per_gbps_hour"
    )
    # Inter-node bandwidth is defined by environment/CLI defaults, not the
    # machine config, but its unit price is still part of _pricing.  When the
    # machine config does not provide explicit inter-node bandwidths, fall
    # back to the same env default used at runtime.
    inter_node_up_gbps = bytes_to_gbps(
        config.get("network_inter_node_up", _resolve_inter_node_bw(None))
    )
    inter_node_down_gbps = bytes_to_gbps(
        config.get("network_inter_node_down", _resolve_inter_node_bw(None))
    )
    price += inter_node_up_gbps * family_or_global("inter_node_up_usd_per_gbps_hour")
    price += inter_node_down_gbps * family_or_global(
        "inter_node_down_usd_per_gbps_hour"
    )
    price += (float(config.get("pcie_bw", 0)) / _GB) * family_or_global(
        "pcie_bw_usd_per_gb_s_hour"
    )

    # NVLink/C2C bandwidth is priced per GPU per GB/s of bandwidth per hour.
    # The config stores per-GPU bandwidth in bytes/sec.
    nvlink_bw_gb_s = float(config.get("nvlink_bw", 0)) / _GB
    if nvlink_bw_gb_s > 0:
        price += (
            nvlink_bw_gb_s * num_gpus * family_or_global("nvlink_bw_usd_per_gb_s_hour")
        )

    return price


def load_machine_db() -> dict[str, dict[str, Any]]:
    if not _MACHINE_DB_PATH.exists():
        msg = (
            f"Machine database not found at {_MACHINE_DB_PATH}. "
            "Run legacy.vast_scraper.refresh_machines_file() first."
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


_combined_machine_db_cache: dict[str | None, dict[str, dict[str, Any]]] = {}


def _clear_combined_machine_db_cache() -> None:
    """Clear the combined machine database cache.

    Tests that mock the underlying JSON loaders must call this before
    exercising the lookup functions.
    """
    _combined_machine_db_cache.clear()


def load_combined_machine_db(
    custom_path: pathlib.Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return machine database with custom hardware entries merged in.

    Custom hardware (from :func:`load_aws_hardware_db`) takes precedence
    over entries in ``_machine_db.json``.  This lets users define local
    hardware presets without modifying the scraped database.

    When ``custom_path`` is not provided, the default ``aws_hardware.json``
    and ``custom_hardware.json`` are both merged in, with later files taking
    precedence.

    The result is cached per ``custom_path`` because the JSON files are
    read repeatedly for large config matrices.
    """
    cache_key = str(custom_path) if custom_path is not None else "__default__"
    if cache_key in _combined_machine_db_cache:
        return _combined_machine_db_cache[cache_key]

    db = load_machine_db()

    # Default AWS-style custom hardware (e.g. scraped AWS presets).
    _, aws_custom = load_aws_hardware_db()
    if aws_custom:
        db = {**db, **aws_custom}

    if custom_path is not None:
        _, user_custom = load_aws_hardware_db(custom_path)
        if user_custom:
            db = {**db, **user_custom}
    else:
        # User-specific custom presets live in custom_hardware.json.
        default_custom_path = pathlib.Path(__file__).parent / "custom_hardware.json"
        _, default_custom = load_aws_hardware_db(default_custom_path)
        if default_custom:
            db = {**db, **default_custom}

    _combined_machine_db_cache[cache_key] = db
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
        msg = f"Machine {machine_name!r} not in database. Run legacy.vast_scraper.refresh_machines_file()."
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

    pricing, _ = load_aws_hardware_db(custom_path)
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
        nvlink_bw=scraped.get("nvlink_bw", 0.0),
        network_bw=scraped.get("network_bw", 0.0),
        reliability=scraped.get("reliability", 0.0),
        reliability_mult=scraped.get("reliability_mult", 0.0),
        score=scraped.get("score", 0.0),
        storage_cost=scraped.get("storage_cost", 0.0),
        storage_total_cost=scraped.get("storage_total_cost", 0.0),
        verification=scraped.get("verification", ""),
    )
    return Hardware(name=scraped["name"], spec=spec)
