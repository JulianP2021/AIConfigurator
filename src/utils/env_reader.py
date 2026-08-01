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
    sessions_per_user: int = 1
    users: int = 10
    max_session_turns: int = 5
    think_time_ms: float = 0.0

    # Per-user random delay (added on top of think_time_ms)
    user_delay_fraction: float = 0.0
    user_delay_min_ms: float = 0.0
    user_delay_max_ms: float = 0.0

    # Mean startup arrival offset per user (exponential distribution).
    # 0 means all users start at t=0.
    startup_arrival_mean_ms: float = 0.0

    # Random seed for reproducible request timing (including user delays)
    random_seed: int | None = None

    # Per-request latency SLAs. Must be finite positive numbers because the
    # request generator builds a deterministic arrival schedule from them.
    sla_ttft_ms: float = 30000.0
    sla_tpot_ms: float = 100.0

    # Topology
    batch_size: int = 10
    num_prefill_nodes: int = 1
    num_decode_nodes: int = 1
    colocated: bool = False
    prefill_gpus_per_node: int = -1

    # Mixed-GPU topology (colocated node with different prefill/decode GPU types)
    mixed: bool = False
    mixed_gpu_donor: str = ""
    mixed_gpu_count: int = -1
    gpu_compute_fraction: float = 0.6

    # Cache
    ram_usage_fraction: float = 0.8
    ssd_usage_fraction: float = 0.8

    # Shared S3/object-store cache
    s3_enabled: bool = True
    s3_up_bw_gbps: float = 25.0
    s3_down_bw_gbps: float = 25.0
    s3_eviction_time_ms: float = 0.0

    # Inter-node (datacenter NIC) bandwidth used for node-to-node KV transfers.
    # Defaults to 100 Gbps symmetric.
    inter_node_network_up_gbps: float = 100.0
    inter_node_network_down_gbps: float = 100.0

    # Dynamo-style KV cache routing cost parameters (tokens).
    # Credits reduce the effective prefill load when the worker already holds
    # the corresponding KV tier. Higher credit -> stronger preference for locality.
    router_prefill_load_scale: float = 1.0
    router_active_work_scale: float = 0.001
    router_device_credit: float = 0.8
    router_remote_ram_credit: float = 0.0
    router_remote_ssd_credit: float = 0.3
    router_s3_credit: float = 0.1

    # When true, use the bandwidth-aware completion-time router (no tunable
    # parameters).  When false, use the Dynamo-style cost model.
    bandwidth_aware_routing: bool = True

    log_mask: int = 0
    debug: bool = False

    # Hardware preset
    machine_hardware: str = "AWS p5en.48xlarge (H200 x8)"


_DEFAULTS = {
    "MODEL": "Qwen/Qwen3-8B",
    "ISL": "1000",
    "OSL": "100",
    "SESSIONS_PER_USER": "1",
    "USERS": "10",
    "MAX_SESSION_TURNS": "5",
    "THINK_TIME_MS": "0.0",
    "USER_DELAY_FRACTION": "0.0",
    "USER_DELAY_MIN_MS": "0.0",
    "USER_DELAY_MAX_MS": "0.0",
    "STARTUP_ARRIVAL_MEAN_MS": "0.0",
    "RANDOM_SEED": "",
    "SLA_TTFT_MS": "30000.0",
    "SLA_TPOT_MS": "100.0",
    "BATCH_SIZE": "10",
    "NUM_PREFILL_NODES": "1",
    "NUM_DECODE_NODES": "1",
    "COLOCATED": "false",
    "PREFILL_GPUS_PER_NODE": "-1",
    "MIXED": "false",
    "MIXED_GPU_DONOR": "",
    "MIXED_GPU_COUNT": "-1",
    "GPU_COMPUTE_FRACTION": "0.6",
    "RAM_USAGE_FRACTION": "0.8",
    "SSD_USAGE_FRACTION": "0.8",
    "S3_ENABLED": "true",
    "S3_UP_BW_GBPS": "25.0",
    "S3_DOWN_BW_GBPS": "25.0",
    "S3_EVICTION_TIME_MS": "0.0",
    "INTER_NODE_NETWORK_UP_GBPS": "100.0",
    "INTER_NODE_NETWORK_DOWN_GBPS": "100.0",
    "ROUTER_PREFILL_LOAD_SCALE": "1.0",
    "ROUTER_ACTIVE_WORK_SCALE": "0.001",
    "ROUTER_DEVICE_CREDIT": "1.0",
    "ROUTER_REMOTE_RAM_CREDIT": "0.0",
    "ROUTER_REMOTE_SSD_CREDIT": "0.3",
    "ROUTER_S3_CREDIT": "0.1",
    "BANDWIDTH_AWARE_ROUTING": "true",
    "LOG_MASK": "15",
    "DEBUG": "false",
    "MACHINE_HARDWARE": "AWS p5en.48xlarge (H200 x8)",
}


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "on"}


def _typed(key: str, value: str) -> str | int | float | bool:
    if key in {"MODEL"}:
        return value
    if key in {
        "ISL",
        "OSL",
        "SESSIONS_PER_USER",
        "USERS",
        "BATCH_SIZE",
        "NUM_PREFILL_NODES",
        "NUM_DECODE_NODES",
        "PREFILL_GPUS_PER_NODE",
        "MIXED_GPU_COUNT",
    }:
        return int(value)
    if key in {
        "RAM_USAGE_FRACTION",
        "SSD_USAGE_FRACTION",
        "S3_UP_BW_GBPS",
        "S3_DOWN_BW_GBPS",
        "INTER_NODE_NETWORK_UP_GBPS",
        "INTER_NODE_NETWORK_DOWN_GBPS",
        "SLA_TTFT_MS",
        "SLA_TPOT_MS",
        "THINK_TIME_MS",
        "USER_DELAY_FRACTION",
        "USER_DELAY_MIN_MS",
        "USER_DELAY_MAX_MS",
        "STARTUP_ARRIVAL_MEAN_MS",
        "GPU_COMPUTE_FRACTION",
        "ROUTER_PREFILL_LOAD_SCALE",
        "ROUTER_ACTIVE_WORK_SCALE",
        "ROUTER_DEVICE_CREDIT",
        "ROUTER_REMOTE_RAM_CREDIT",
        "ROUTER_REMOTE_SSD_CREDIT",
        "ROUTER_S3_CREDIT",
    }:
        return float(value)
    if key in {
        "LOG_MASK",
        "RANDOM_SEED",
    }:
        return int(value, 0)
    if key in {"DEBUG", "S3_ENABLED", "COLOCATED", "MIXED", "BANDWIDTH_AWARE_ROUTING"}:
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

    print(f"Environment defaults from {path} (CLI flags can override):")
    for key, value in merged.items():
        print(f"  {key} = {value}")

    return EnvConfig(
        model=str(merged["MODEL"]),
        isl=int(merged["ISL"]),
        osl=int(merged["OSL"]),
        sessions_per_user=int(merged["SESSIONS_PER_USER"]),
        users=int(merged["USERS"]),
        max_session_turns=int(merged["MAX_SESSION_TURNS"]),
        think_time_ms=float(merged["THINK_TIME_MS"]),
        user_delay_fraction=float(merged["USER_DELAY_FRACTION"]),
        user_delay_min_ms=float(merged["USER_DELAY_MIN_MS"]),
        user_delay_max_ms=float(merged["USER_DELAY_MAX_MS"]),
        startup_arrival_mean_ms=float(merged["STARTUP_ARRIVAL_MEAN_MS"]),
        random_seed=int(merged["RANDOM_SEED"], 0) if merged["RANDOM_SEED"] else None,
        sla_ttft_ms=float(merged["SLA_TTFT_MS"]),
        sla_tpot_ms=float(merged["SLA_TPOT_MS"]),
        batch_size=int(merged["BATCH_SIZE"]),
        num_prefill_nodes=int(merged["NUM_PREFILL_NODES"]),
        num_decode_nodes=int(merged["NUM_DECODE_NODES"]),
        colocated=_parse_bool(merged["COLOCATED"]),
        prefill_gpus_per_node=int(merged["PREFILL_GPUS_PER_NODE"]),
        mixed=_parse_bool(merged["MIXED"]),
        mixed_gpu_donor=str(merged["MIXED_GPU_DONOR"]),
        mixed_gpu_count=int(merged["MIXED_GPU_COUNT"]),
        gpu_compute_fraction=float(merged["GPU_COMPUTE_FRACTION"]),
        ram_usage_fraction=float(merged["RAM_USAGE_FRACTION"]),
        ssd_usage_fraction=float(merged["SSD_USAGE_FRACTION"]),
        s3_enabled=_parse_bool(merged["S3_ENABLED"]),
        s3_up_bw_gbps=float(merged["S3_UP_BW_GBPS"]),
        s3_down_bw_gbps=float(merged["S3_DOWN_BW_GBPS"]),
        s3_eviction_time_ms=float(merged["S3_EVICTION_TIME_MS"]),
        inter_node_network_up_gbps=float(merged["INTER_NODE_NETWORK_UP_GBPS"]),
        inter_node_network_down_gbps=float(merged["INTER_NODE_NETWORK_DOWN_GBPS"]),
        router_prefill_load_scale=float(merged["ROUTER_PREFILL_LOAD_SCALE"]),
        router_active_work_scale=float(merged["ROUTER_ACTIVE_WORK_SCALE"]),
        router_device_credit=float(merged["ROUTER_DEVICE_CREDIT"]),
        router_remote_ram_credit=float(merged["ROUTER_REMOTE_RAM_CREDIT"]),
        router_remote_ssd_credit=float(merged["ROUTER_REMOTE_SSD_CREDIT"]),
        router_s3_credit=float(merged["ROUTER_S3_CREDIT"]),
        log_mask=int(merged["LOG_MASK"], 0),
        debug=_parse_bool(merged["DEBUG"]),
        machine_hardware=str(merged["MACHINE_HARDWARE"]),
    )
