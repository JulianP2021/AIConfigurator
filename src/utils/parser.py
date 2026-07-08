import argparse
import json

from src.utils.env_reader import EnvConfig


def _parse_sla(value: str) -> dict[str, float]:
    """Parse a JSON SLA dict from a CLI string.

    Accepts either a JSON object (e.g. '{"ttft_ms":100,"tpot_ms":50}') or the
    literal ``inf`` / ``none`` / ``null`` to mean no SLA.  JSON ``inf`` values
    are supported both as quoted strings and (in Python 3.9+) as bare ``Infinity``.
    """
    value = value.strip()
    if value.lower() in {"inf", "infinity", "none", "null"}:
        return {"ttft_ms": float("inf"), "tpot_ms": float("inf")}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("SLA must be a JSON object")
    for key in ("ttft_ms", "tpot_ms"):
        if key in parsed:
            v = parsed[key]
            if isinstance(v, str) and v.lower() in {"inf", "infinity", "+inf"}:
                parsed[key] = float("inf")
            elif isinstance(v, str) and v.lower() in {"-inf", "-infinity"}:
                parsed[key] = float("-inf")
            else:
                parsed[key] = float(v)
    return parsed


def _base_parser(env: EnvConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed LLM inference simulator")
    parser.add_argument(
        "--model",
        type=str,
        default=env.model,
        help=f"HuggingFace model name (default: {env.model})",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=env.isl,
        help=f"Input sequence length (fixed, default: {env.isl})",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=env.osl,
        help=f"Output sequence length (fixed, default: {env.osl})",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=env.requests,
        help=f"Total requests to simulate (default: {env.requests})",
    )
    parser.add_argument(
        "--req-rate",
        type=float,
        default=env.req_rate,
        help=f"Request arrival rate in req/s (default: {env.req_rate})",
    )
    parser.add_argument(
        "--unique-users",
        action="store_true",
        default=env.unique_users,
        help="Set max_users > total_requests so every request gets a unique user (no shared prefix)",
    )
    parser.add_argument(
        "--min-users",
        type=int,
        default=env.min_users,
        help=f"Minimum number of users (default: {env.min_users})",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=env.max_users,
        help=f"Maximum number of users (default: {env.max_users})",
    )
    parser.add_argument(
        "--max-session-turns",
        type=int,
        default=env.max_session_turns,
        help=f"Max requests per user session before starting a new session (default: {env.max_session_turns})",
    )
    parser.add_argument(
        "--sla",
        type=_parse_sla,
        default={"ttft_ms": env.sla_ttft_ms, "tpot_ms": env.sla_tpot_ms},
        help=(
            "Per-request latency SLA as a JSON object with 'ttft_ms' and "
            "'tpot_ms' keys, e.g. '{\"ttft_ms\":100,\"tpot_ms\":50}' "
            f"(default: ttft_ms={env.sla_ttft_ms}, tpot_ms={env.sla_tpot_ms})"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=env.debug,
        help="Enable verbose debug logging (sets LOG_MASK to all components)",
    )

    parser.add_argument(
        "--ram-usage-fraction",
        type=float,
        default=env.ram_usage_fraction,
        help=f"Fraction of node RAM usable for the KV cache layer (default: {env.ram_usage_fraction})",
    )
    parser.add_argument(
        "--ssd-usage-fraction",
        type=float,
        default=env.ssd_usage_fraction,
        help=f"Fraction of node SSD usable for the KV cache layer (default: {env.ssd_usage_fraction})",
    )
    parser.add_argument(
        "--s3-enabled",
        action="store_true",
        default=env.s3_enabled,
        help="Enable the shared S3/object-store cache fallback (default: %(default)s)",
    )
    parser.add_argument(
        "--s3-up-bw-gbps",
        type=float,
        default=env.s3_up_bw_gbps,
        help=f"S3 upload bandwidth in Gbps (default: {env.s3_up_bw_gbps})",
    )
    parser.add_argument(
        "--s3-down-bw-gbps",
        type=float,
        default=env.s3_down_bw_gbps,
        help=f"S3 download bandwidth in Gbps (default: {env.s3_down_bw_gbps})",
    )
    parser.add_argument(
        "--router-prefill-load-scale",
        type=float,
        default=env.router_prefill_load_scale,
        help=f"Weight of prefill load in routing cost (default: {env.router_prefill_load_scale})",
    )
    parser.add_argument(
        "--router-device-credit",
        type=float,
        default=env.router_device_credit,
        help=f"Credit for device-local KV hits (default: {env.router_device_credit})",
    )
    parser.add_argument(
        "--router-remote-ram-credit",
        type=float,
        default=env.router_remote_ram_credit,
        help=f"Credit for remote RAM KV hits (default: {env.router_remote_ram_credit})",
    )
    parser.add_argument(
        "--router-ssd-credit",
        type=float,
        default=env.router_ssd_credit,
        help=f"Credit for SSD KV hits (default: {env.router_ssd_credit})",
    )
    parser.add_argument(
        "--router-s3-credit",
        type=float,
        default=env.router_s3_credit,
        help=f"Credit for S3 KV hits (default: {env.router_s3_credit})",
    )
    parser.add_argument(
        "--router-busy-threshold-tokens",
        type=float,
        default=env.router_busy_threshold_tokens,
        help=(
            "Workers with active load above this token count are skipped "
            f"(default: {env.router_busy_threshold_tokens})"
        ),
    )
    return parser


def get_main_parser(env: EnvConfig) -> argparse.ArgumentParser:
    """CLI parser for main.py (adds simulator-specific flags on top of base)."""
    parser = _base_parser(env)
    parser.add_argument(
        "--log-mask",
        type=lambda s: int(s, 0),
        default=env.log_mask,
        help=(
            "Component logging bitmask: bit 0 (1)=cache, bit 1 (2)=instances, "
            "bit 2 (4)=router,  bit 3 (8)=simulation, bit 4 (16)=bandwidth. 0=none, 31=all "
            f"(default: {env.log_mask})"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=env.batch_size,
        help=f"Decode batch size (default: {env.batch_size})",
    )
    parser.add_argument(
        "--prefill-workers",
        type=int,
        default=env.prefill_workers,
        help=f"Number of prefill workers (default: {env.prefill_workers})",
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=env.decode_workers,
        help=f"Number of decode workers (default: {env.decode_workers})",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=env.gpus_per_node,
        help=f"GPUs per node (default: {env.gpus_per_node})",
    )
    parser.add_argument(
        "--machine-hardware",
        type=str,
        default="H200 x8 #692c33bd",
        help=(
            "Hardware preset key from the machine database. "
            "Quote values containing spaces/hash, e.g. "
            '"H200 x8 #692c33bd"'
        ),
    )
    parser.add_argument(
        "--colocated",
        action="store_true",
        default=False,
        help=(
            "Run colocated nodes that host both prefill and decode instances. "
            "The GPU split is controlled by --prefill-gpus-per-node; the rest "
            "are decode instances (default: disabled; separate prefill/decode nodes)"
        ),
    )
    parser.add_argument(
        "--num-prefill-nodes",
        type=int,
        default=1,
        help=(
            "Number of distinct prefill-only nodes when --colocated is not set "
            "(default: 1)"
        ),
    )
    parser.add_argument(
        "--num-decode-nodes",
        type=int,
        default=1,
        help=(
            "Number of distinct decode-only nodes when --colocated is not set "
            "(default: 1)"
        ),
    )
    parser.add_argument(
        "--prefill-gpus-per-node",
        type=int,
        default=-1,
        help=(
            "GPUs on each node to use as prefill instances. "
            "In --colocated mode the remaining GPUs become decode instances. "
            "When omitted, colocated mode falls back to --prefill-workers and "
            "non-colocated mode uses all GPUs per prefill-only node."
        ),
    )
    return parser


def get_create_config_parser(env: EnvConfig) -> argparse.ArgumentParser:
    parser = _base_parser(env)
    parser.add_argument(
        "--config-name",
        type=str,
        default="config.json",
        help="Name of the output configuration file (default: config.json)",
    )
    return parser


# Alias kept for backwards compatibility with any direct imports.
def get_simulation_parser(env: EnvConfig) -> argparse.ArgumentParser:
    return get_main_parser(env)
