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
    max_session_turns: int = 5

    # Per-request latency SLAs (default inf = disabled)
    sla_ttft_ms: float = float("inf")
    sla_tpot_ms: float = float("inf")

    # Topology
    batch_size: int = 10
    prefill_workers: int = 1
    decode_workers: int = 1
    gpus_per_node: int = 1

    # Cache
    ram_usage_fraction: float = 0.8
    ssd_usage_fraction: float = 0.8

    # Shared S3/object-store cache
    s3_enabled: bool = True
    s3_up_bw_gbps: float = 25.0
    s3_down_bw_gbps: float = 25.0

    # Dynamo-style KV cache routing cost parameters (tokens).
    # Credits reduce the effective prefill load when the worker already holds
    # the corresponding KV tier. Higher credit -> stronger preference for locality.
    router_prefill_load_scale: float = 1.0
    router_device_credit: float = 1.0
    router_remote_ram_credit: float = 0.0
    router_ssd_credit: float = 0.3
    router_s3_credit: float = 0.1
    router_busy_threshold_tokens: float = 1_000_000.0

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
    "MAX_SESSION_TURNS": "5",
    "SLA_TTFT_MS": "inf",
    "SLA_TPOT_MS": "inf",
    "BATCH_SIZE": "10",
    "PREFILL_WORKERS": "1",
    "DECODE_WORKERS": "1",
    "GPUS_PER_NODE": "1",
    "RAM_USAGE_FRACTION": "0.8",
    "SSD_USAGE_FRACTION": "0.8",
    "S3_ENABLED": "true",
    "S3_UP_BW_GBPS": "25.0",
    "S3_DOWN_BW_GBPS": "25.0",
    "ROUTER_PREFILL_LOAD_SCALE": "1.0",
    "ROUTER_DEVICE_CREDIT": "1.0",
    "ROUTER_REMOTE_RAM_CREDIT": "0.0",
    "ROUTER_SSD_CREDIT": "0.3",
    "ROUTER_S3_CREDIT": "0.1",
    "ROUTER_BUSY_THRESHOLD_TOKENS": "1000000.0",
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
    if key in {
        "REQ_RATE",
        "RAM_USAGE_FRACTION",
        "SSD_USAGE_FRACTION",
        "S3_UP_BW_GBPS",
        "S3_DOWN_BW_GBPS",
        "SLA_TTFT_MS",
        "SLA_TPOT_MS",
        "ROUTER_PREFILL_LOAD_SCALE",
        "ROUTER_DEVICE_CREDIT",
        "ROUTER_REMOTE_RAM_CREDIT",
        "ROUTER_SSD_CREDIT",
        "ROUTER_S3_CREDIT",
        "ROUTER_BUSY_THRESHOLD_TOKENS",
    }:
        return float(value)
    if key == "LOG_MASK":
        return int(value, 0)
    if key in {"UNIQUE_USERS", "DEBUG", "S3_ENABLED"}:
        return _parse_bool(value)
    if key == "MAX_SESSION_TURNS":
        return int(value)
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
        max_session_turns=int(merged["MAX_SESSION_TURNS"]),
        sla_ttft_ms=float(merged["SLA_TTFT_MS"]),
        sla_tpot_ms=float(merged["SLA_TPOT_MS"]),
        batch_size=int(merged["BATCH_SIZE"]),
        prefill_workers=int(merged["PREFILL_WORKERS"]),
        decode_workers=int(merged["DECODE_WORKERS"]),
        gpus_per_node=int(merged["GPUS_PER_NODE"]),
        ram_usage_fraction=float(merged["RAM_USAGE_FRACTION"]),
        ssd_usage_fraction=float(merged["SSD_USAGE_FRACTION"]),
        s3_enabled=_parse_bool(merged["S3_ENABLED"]),
        s3_up_bw_gbps=float(merged["S3_UP_BW_GBPS"]),
        s3_down_bw_gbps=float(merged["S3_DOWN_BW_GBPS"]),
        router_prefill_load_scale=float(merged["ROUTER_PREFILL_LOAD_SCALE"]),
        router_device_credit=float(merged["ROUTER_DEVICE_CREDIT"]),
        router_remote_ram_credit=float(merged["ROUTER_REMOTE_RAM_CREDIT"]),
        router_ssd_credit=float(merged["ROUTER_SSD_CREDIT"]),
        router_s3_credit=float(merged["ROUTER_S3_CREDIT"]),
        router_busy_threshold_tokens=float(merged["ROUTER_BUSY_THRESHOLD_TOKENS"]),
        log_mask=int(merged["LOG_MASK"], 0),
        debug=_parse_bool(merged["DEBUG"]),
    )
