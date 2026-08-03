"""GPU / machine spec lookup backed by local JSON files.

The module reads from ``_gpu_db.json`` and ``_machine_db.json`` by default,
avoiding any network calls at import or runtime.  The network-dependent
Vast.ai scrapers live in ``src.hardware.legacy.vast_scraper`` and are only
used when explicitly refreshing the cached JSON files.
"""

import json
import os
import re

from pathlib import Path
from typing import Any

from src.hardware.hardware import Hardware
from src.hardware.legacy.vast_scraper import _fetch_specs


_GPU_DB_PATH = Path(__file__).parent / "legacy" / "_gpu_db.json"


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

_MACHINE_DB_PATH = Path(__file__).parent / "legacy" / "_machine_db.json"
_AWS_HARDWARE_PATH = Path(__file__).parent / "data/" / "aws_hardware.json"
_PRICING_PATH = Path(__file__).parent / "data" / "pricing.json"


def load_aws_hardware_db(
    aws_path: Path | str | None = None,
    pricing_path: Path | str | None = None,
) -> tuple[dict[Any, Any], dict[str, dict[str, Any]]]:
    """Load user-supplied custom hardware definitions.

    The file is a JSON object with two top-level keys:

    * ``_pricing`` (optional): global unit prices used to derive hourly cost
      for custom machines.

      * ``cpu_ram_usd_per_gb_hour`` USD per GB of CPU RAM per hour.
      * ``ssd_usd_per_gb_hour`` USD per GB of SSD storage per hour.
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
      GPU database and so its hourly price can be derived from the matching
      per-family pricing table.  Any remaining field is filled with safe
      defaults.

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
    target = Path(aws_path) if aws_path is not None else _AWS_HARDWARE_PATH
    if not target.exists():
        return {}, {}
    aws_data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(aws_data, dict):
        raise ValueError(f"Custom hardware file {target} must contain a JSON object")

    raw_machines = aws_data.get("machines", {})
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
    target = Path(pricing_path) if pricing_path is not None else _PRICING_PATH
    if not target.exists():
        return {}, {}
    pricing_data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(pricing_data, dict):
        raise ValueError(f"Custom hardware file {target} must contain a JSON object")

    pricing = pricing_data.get("_pricing", {})
    if not isinstance(pricing, dict):
        raise ValueError(
            f"Custom hardware file {target}: '_pricing' must be a JSON object"
        )

    return pricing, normalized


# Per-byte / per-second constants used for unit-price cost derivation.
_GB = 1024**3
_HOUR_S = 3600.0


def _all_in_per_gpu_price(
    gpu_name: str,
    pricing: dict[str, Any],
    db: dict[str, dict[str, Any]] | None = None,
) -> float:
    """Return the all-in per-GPU price for a GPU family using the current table.

    Uses the largest AWS instance of the family as the reference.  This is the
    price a reference node would have if every component (GPU, RAM, SSD,
    bandwidth) were priced at the per-family unit prices stored in the table.
    """
    family_pricing = pricing.get("gpu_family_pricing", {})
    family = family_pricing.get(gpu_name, {})
    if db is None:
        db = load_combined_machine_db()

    aws_instances = [
        cfg
        for cfg in db.values()
        if cfg.get("gpu_name") == gpu_name
        and str(cfg.get("name", "")).startswith("AWS")
    ]
    if not aws_instances:
        # No AWS reference; fall back to the table's compute price as the
        # dominant component.
        return float(family.get("compute_usd_per_gpu_hour", 0.0)) / 0.6

    ref = max(aws_instances, key=lambda cfg: int(cfg.get("num_gpus", 1)))
    return _derive_custom_price_raw(ref, pricing) / int(ref["num_gpus"])


def _compute_fraction_from_table(gpu_name: str, pricing: dict[str, Any]) -> float:
    """Return the compute fraction encoded in the pricing table.

    First checks the top-level ``gpu_compute_fraction`` metadata, then the
    per-family ``gpu_compute_fraction`` field, and finally falls back to
    inferring it from the largest AWS instance of the family.
    """
    top_level = pricing.get("gpu_compute_fraction")
    if top_level is not None:
        return float(top_level)

    family_pricing = pricing.get("gpu_family_pricing", {})
    family = family_pricing.get(gpu_name, {})
    family_fraction = family.get("gpu_compute_fraction")
    if family_fraction is not None:
        return float(family_fraction)

    db = load_combined_machine_db()
    aws_instances = [
        cfg
        for cfg in db.values()
        if cfg.get("gpu_name") == gpu_name
        and str(cfg.get("name", "")).startswith("AWS")
    ]
    if not aws_instances:
        return 0.6

    ref = max(aws_instances, key=lambda cfg: int(cfg.get("num_gpus", 1)))
    all_in = _derive_custom_price_raw(ref, pricing)
    compute_cost = int(ref["num_gpus"]) * float(
        family.get("compute_usd_per_gpu_hour", 0.0)
    )
    ram_cost = (ref["cpu_ram"] / _GB) * float(
        family.get("cpu_ram_usd_per_gb_hour", 0.0089)
    )
    ssd_cost = (ref["nvme_mem"] / _GB) * float(
        family.get("ssd_usd_per_gb_hour", 0.00037)
    )
    adjusted = all_in - ram_cost - ssd_cost
    if adjusted <= 0:
        return 0.6
    return compute_cost / adjusted


def _derive_custom_price_raw(
    config: dict[str, Any],
    pricing: dict[str, Any],
) -> float:
    """Derive a price using the stored table values without runtime re-split.

    This is the baseline calculation at the table's native 60/40 split and is
    used internally to compute reference all-in prices.
    """

    def bits_to_gbps(val: float) -> float:
        return float(val) / 1e9

    gpu_name = config.get("gpu_name", "")
    if not gpu_name:
        raise ValueError(
            f"Machine config {config.get('name', '<unknown>')!r} must specify gpu_name"
        )

    family_pricing = pricing.get("gpu_family_pricing", {})
    family = family_pricing.get(gpu_name)
    if family is None:
        raise ValueError(
            f"No per-family pricing table for gpu_name {gpu_name!r}. "
            f"Add an entry under _pricing.gpu_family_pricing."
        )

    def family_price(key: str) -> float:
        if key not in family:
            return float(pricing[key])
        if float(family[key]) == 0:
            return float(pricing[key])
        return float(family[key])

    num_gpus = int(config.get("num_gpus", 1))
    compute_price = family_price("compute_usd_per_gpu_hour")

    cpu_ram_gb = float(config.get("cpu_ram", 0)) / _GB
    ssd_gb = float(config.get("nvme_mem", 0)) / _GB

    price = num_gpus * compute_price
    price += cpu_ram_gb * family_price("cpu_ram_usd_per_gb_hour")
    price += ssd_gb * family_price("ssd_usd_per_gb_hour")
    price += bits_to_gbps(config.get("network_inet_up", 0)) * family_price(
        "inet_up_usd_per_gbps_hour"
    )
    price += bits_to_gbps(config.get("network_inet_down", 0)) * family_price(
        "inet_down_usd_per_gbps_hour"
    )
    inter_node_up_gbps = bits_to_gbps(
        config.get("network_inter_node_up", _resolve_inter_node_bw(None))
    )
    inter_node_down_gbps = bits_to_gbps(
        config.get("network_inter_node_down", _resolve_inter_node_bw(None))
    )
    price += inter_node_up_gbps * family_price("inter_node_up_usd_per_gbps_hour")
    price += inter_node_down_gbps * family_price("inter_node_down_usd_per_gbps_hour")

    nvlink_bw_gb_s = float(config.get("nvlink_bw", 0)) / _GB
    if nvlink_bw_gb_s > 0:
        price += nvlink_bw_gb_s * num_gpus * family_price("nvlink_bw_usd_per_gb_s_hour")
    price += (float(config.get("pcie_bw", 0)) / _GB) * family_price(
        "pcie_bw_usd_per_gb_s_hour"
    )
    return price


def _derive_custom_price(
    config: dict[str, Any],
    pricing: dict[str, Any],
    *,
    compute_price_fraction: float | None = None,
) -> float:
    """Derive an hourly price for a custom machine from per-family unit prices.

    The config must specify a ``gpu_name`` that exists in the per-family pricing
    table.  Each GPU family has its own component prices (CPU RAM, SSD, PCIe,
    NVLink, SSD bandwidth, internet, inter-node) plus a per-GPU compute price.
    This lets prices scale with every hardware component exactly as observed in
    AWS on-demand pricing for that family.

    The pricing table itself encodes a fixed compute/bandwidth split.  If a
    caller requests a ``compute_price_fraction`` that differs from the split
    stored in the table, this function raises an error and asks the user to
    regenerate the pricing table with the desired split.

    Parameters
    ----------
    config:
        A machine config dict (already normalised/merged with defaults).
    pricing:
        The ``_pricing`` dict from the custom hardware file.
    compute_price_fraction:
        Optional fraction to validate against the table.  If provided and it
        does not match the table's stored split, ``ValueError`` is raised.

    Returns:
    -------
    Hourly price in USD.
    """
    gpu_name = config.get("gpu_name", "")
    if not gpu_name:
        raise ValueError(
            f"Machine config {config.get('name', '<unknown>')!r} must specify gpu_name"
        )

    family_pricing = pricing.get("gpu_family_pricing", {})
    family = family_pricing.get(gpu_name)
    if family is None:
        raise ValueError(
            f"No per-family pricing table for gpu_name {gpu_name!r}. "
            f"Add an entry under _pricing.gpu_family_pricing."
        )

    num_gpus = int(config.get("num_gpus", 1))

    # If this family has no identifiable per-GPU compute price (the AWS data
    # only contains single-GPU instances for it), prevent users from deriving
    # prices for multi-GPU configs because the scaling is unvalidated.
    if float(family.get("compute_usd_per_gpu_hour", 0.0)) == 0.0 and num_gpus > 1:
        raise ValueError(
            f"gpu_name {gpu_name!r} has no multi-GPU pricing data; "
            f"cannot derive price for {num_gpus} GPUs"
        )

    if compute_price_fraction is not None:
        table_fraction = _compute_fraction_from_table(gpu_name, pricing)
        if abs(compute_price_fraction - table_fraction) > 1e-6:
            raise ValueError(
                f"Requested GPU compute fraction {compute_price_fraction} does not match "
                f"the pricing table's split {table_fraction:.4f} for family {gpu_name!r}. "
                f"Regenerate src/hardware/data/pricing.json with the desired split, e.g. "
                f".venv/bin/python scripts/derive_family_pricing.py --gpu-compute-fraction {compute_price_fraction}"
            )

    return _derive_custom_price_raw(config, pricing)


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
    custom_path: Path | str | None = None,
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
        default_custom_path = Path(__file__).parent / "data" / "custom_hardware.json"
        _, default_custom = load_aws_hardware_db(default_custom_path)
        if default_custom:
            db = {**db, **default_custom}

    _combined_machine_db_cache[cache_key] = db
    return db


def resolve_machine_name(
    machine_name: str, custom_path: Path | str | None = None
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
    machine_name: str, custom_path: Path | str | None = None
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
    *,
    compute_price_fraction: float | None = None,
) -> dict[str, Any]:
    """Fill missing machine-config fields with safe defaults.

    Custom hardware entries only need to specify the values that matter for
    simulation; everything else is zeroed or defaulted so ``HardwareSpec``
    construction still succeeds.

    For custom entries (those without a scraped ``dph_base``), if a pricing
    dict is supplied the hourly price is derived from unit prices and the
    machine's resources and written into ``dph_base``.
    """
    merged = {**config}
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

    # Inter-node bandwidth defaults to 100 Gb/s when not supplied; it is
    # normally overridden via environment variables / CLI / webserver.
    for inter_key in ("network_inter_node_up", "network_inter_node_down"):
        if inter_key not in merged or merged[inter_key] is None:
            merged[inter_key] = _resolve_inter_node_bw(None, default_gbps=100.0)

    # Derive price for custom entries when pricing metadata is provided.
    if pricing and not merged.get("dph_base"):
        merged["dph_base"] = _derive_custom_price(
            merged, pricing, compute_price_fraction=compute_price_fraction
        )

    return merged


def fetch_machine_hardware(
    machine_name: str,
    machine_config_override: dict[str, Any] | None = None,
    custom_path: Path | str | None = None,
    *,
    compute_price_fraction: float | None = None,
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
        scraped = _machine_config_with_defaults(
            machine_config_override,
            pricing,
            compute_price_fraction=compute_price_fraction,
        )
    else:
        scraped = _machine_config_with_defaults(
            lookup_machine(machine_name, custom_path),
            pricing,
            compute_price_fraction=compute_price_fraction,
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
        cpu_ram=scraped.get("cpu_ram", 0),
        dph_base=scraped.get("dph_base", 0.0),
        pcie_bw=scraped.get("pcie_bw", 0.0),
        nvlink_bw=scraped.get("nvlink_bw", 0.0),
    )
    return Hardware(name=scraped["name"], spec=spec)


def get_pricing(path: Path | None = None):
    """Load unit prices from the AWS hardware JSON file, if present."""
    if not path:
        path = _PRICING_PATH
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("_pricing", {})
