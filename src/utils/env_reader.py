"""Lightweight environment configuration reader.

Reads a ``.env`` file at the project root and exposes typed defaults for the
simulator. Values can be overridden by shell environment variables; CLI
arguments can then override those values.
"""

import os

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnvConfig:
    # Model and request shape
    model: str = "Qwen/Qwen3-8B"
    isl: int = 1000
    osl: int = 100

    # Scenario scale
    requests: int = 10
    req_rate: float = 2.0
    unique_users: bool = False
    min_users: int = 1
    max_users: int = 10

    # Topology
    batch_size: int = 10
    prefill_workers: int = 1
    decode_workers: int = 1
    gpus_per_node: int = 1

    # Cache
    cache_pct: float = 0.0
    ram_usage_fraction: float = 0.8
    ssd_usage_fraction: float = 0.8

    # Logging bitmask:
    #   bit 0 (1)  = cache
    #   bit 1 (2)  = instances
    #   bit 2 (4)  = router
    #   bit 3 (8)  = simulation
    #   0 = nothing, 15 = everything
    log_mask: int = 15
    debug: bool = False


_DEFAULTS = {
    "MODEL": "Qwen/Qwen3-8B",
    "ISL": "1000",
    "OSL": "100",
    "REQUESTS": "10",
    "REQ_RATE": "2.0",
    "UNIQUE_USERS": "false",
    "MIN_USERS": "1",
    "MAX_USERS": "10",
    "BATCH_SIZE": "10",
    "PREFILL_WORKERS": "1",
    "DECODE_WORKERS": "1",
    "GPUS_PER_NODE": "1",
    "CACHE_PCT": "0.0",
    "RAM_USAGE_FRACTION": "0.8",
    "SSD_USAGE_FRACTION": "0.8",
    "LOG_MASK": "15",
    "DEBUG": "false",
}


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "on"}


def _typed(key: str, value: str) -> str | int | float | bool:
    if key in {"MODEL"}:
        return value
    if key in {
        "ISL",
        "OSL",
        "REQUESTS",
        "MIN_USERS",
        "MAX_USERS",
        "BATCH_SIZE",
        "PREFILL_WORKERS",
        "DECODE_WORKERS",
        "GPUS_PER_NODE",
    }:
        return int(value)
    if key in {"REQ_RATE", "CACHE_PCT", "RAM_USAGE_FRACTION", "SSD_USAGE_FRACTION"}:
        return float(value)
    if key == "LOG_MASK":
        return int(value, 0)
    if key in {"UNIQUE_USERS", "DEBUG"}:
        return _parse_bool(value)
    return value


def _env_path(project_root: Path | None = None) -> Path:
    if project_root is None:
        # src/utils/env_reader.py -> project root is two parents up
        project_root = Path(__file__).resolve().parents[2]
    return project_root / ".env"


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def load_env(project_root: Path | None = None) -> EnvConfig:
    """Load configuration from ``.env`` and shell environment variables.

    Precedence (highest first):
        1. Shell environment variables
        2. ``.env`` file at the project root
        3. Hard-coded defaults
    """
    path = _env_path(project_root)
    file_values = _read_env_file(path)

    merged: dict[str, str] = {}
    for key, default in _DEFAULTS.items():
        merged[key] = os.environ.get(key, file_values.get(key, default))

    print(f"Loaded environment configuration from {path}:")
    for key, value in merged.items():
        print(f"  {key} = {value}")

    return EnvConfig(
        model=str(merged["MODEL"]),
        isl=int(merged["ISL"]),
        osl=int(merged["OSL"]),
        requests=int(merged["REQUESTS"]),
        req_rate=float(merged["REQ_RATE"]),
        unique_users=_parse_bool(merged["UNIQUE_USERS"]),
        min_users=int(merged["MIN_USERS"]),
        max_users=int(merged["MAX_USERS"]),
        batch_size=int(merged["BATCH_SIZE"]),
        prefill_workers=int(merged["PREFILL_WORKERS"]),
        decode_workers=int(merged["DECODE_WORKERS"]),
        gpus_per_node=int(merged["GPUS_PER_NODE"]),
        cache_pct=float(merged["CACHE_PCT"]),
        ram_usage_fraction=float(merged["RAM_USAGE_FRACTION"]),
        ssd_usage_fraction=float(merged["SSD_USAGE_FRACTION"]),
        log_mask=int(merged["LOG_MASK"], 0),
        debug=_parse_bool(merged["DEBUG"]),
    )
