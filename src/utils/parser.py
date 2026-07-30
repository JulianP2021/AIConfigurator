import argparse
import json
import math

from src.utils.env_reader import EnvConfig


def _parse_sla(value: str) -> dict[str, float]:
    """Parse a JSON SLA dict from a CLI string.

    Accepts a JSON object (e.g. '{"ttft_ms":100,"tpot_ms":50}').  Both
    ``ttft_ms`` and ``tpot_ms`` must be finite positive numbers because the
    request generator builds a deterministic arrival schedule from them.
    """
    value = value.strip()
    if value.lower() in {"inf", "infinity", "none", "null"}:
        raise argparse.ArgumentTypeError(
            "SLA values must be finite positive numbers; 'inf'/'none'/'null' are not allowed"
        )
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("SLA must be a JSON object")
    for key in ("ttft_ms", "tpot_ms"):
        if key not in parsed:
            raise argparse.ArgumentTypeError(f"SLA must include '{key}'")
        v = parsed[key]
        if isinstance(v, str) and v.lower() in {
            "inf",
            "infinity",
            "+inf",
            "-inf",
            "-infinity",
        }:
            raise argparse.ArgumentTypeError(
                f"{key} must be a finite positive number, got {v!r}"
            )
        parsed[key] = float(v)
        if not math.isfinite(parsed[key]) or parsed[key] <= 0:
            raise argparse.ArgumentTypeError(
                f"{key} must be a finite positive number, got {parsed[key]}"
            )
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
        "--sessions-per-user",
        type=int,
        default=env.sessions_per_user,
        help=f"Sessions per user (default: {env.sessions_per_user}). "
        "Total requests = users * sessions_per_user * max_session_turns.",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=env.users,
        help=f"Fixed pool of users that take turns sending requests (default: {env.users})",
    )
    parser.add_argument(
        "--max-session-turns",
        type=int,
        default=env.max_session_turns,
        help=f"Max requests per user session before starting a new session (default: {env.max_session_turns})",
    )
    parser.add_argument(
        "--think-time-ms",
        type=float,
        default=env.think_time_ms,
        help=f"Think time between a user's consecutive requests in ms (default: {env.think_time_ms})",
    )
    parser.add_argument(
        "--user-delay-fraction",
        type=float,
        default=env.user_delay_fraction,
        help=(
            "Fraction of requests/users that receive an extra random delay "
            f"after finishing (default: {env.user_delay_fraction})"
        ),
    )
    parser.add_argument(
        "--user-delay-min-ms",
        type=float,
        default=env.user_delay_min_ms,
        help=f"Minimum extra user delay in ms (default: {env.user_delay_min_ms})",
    )
    parser.add_argument(
        "--user-delay-max-ms",
        type=float,
        default=env.user_delay_max_ms,
        help=f"Maximum extra user delay in ms (default: {env.user_delay_max_ms})",
    )
    parser.add_argument(
        "--startup-arrival-mean-ms",
        type=float,
        default=env.startup_arrival_mean_ms,
        help=(
            "Mean startup arrival offset per user, drawn from an exponential "
            "distribution. 0 means all users start at t=0 "
            f"(default: {env.startup_arrival_mean_ms})"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=env.random_seed,
        help=(
            "Seed for the request generator's random number generator; "
            "makes user delays and startup offsets reproducible "
            f"(default: {env.random_seed})"
        ),
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
        "--s3-eviction-time-ms",
        type=float,
        default=env.s3_eviction_time_ms,
        help=(
            "Evict S3 objects that have not been accessed in this many ms. "
            f"0 disables eviction (default: {env.s3_eviction_time_ms})"
        ),
    )
    parser.add_argument(
        "--inter-node-network-up-gbps",
        type=float,
        default=env.inter_node_network_up_gbps,
        help=(
            "Inter-node (datacenter NIC) upload bandwidth in Gbps for "
            f"node-to-node KV transfers (default: {env.inter_node_network_up_gbps})"
        ),
    )
    parser.add_argument(
        "--inter-node-network-down-gbps",
        type=float,
        default=env.inter_node_network_down_gbps,
        help=(
            "Inter-node (datacenter NIC) download bandwidth in Gbps for "
            f"node-to-node KV transfers (default: {env.inter_node_network_down_gbps})"
        ),
    )
    parser.add_argument(
        "--router-prefill-load-scale",
        type=float,
        default=env.router_prefill_load_scale,
        help=f"Weight of prefill load in routing cost (default: {env.router_prefill_load_scale})",
    )
    parser.add_argument(
        "--router-active-work-scale",
        type=float,
        default=env.router_active_work_scale,
        help=f"Weight of active work in routing cost (default: {env.router_active_work_scale})",
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
        "--router-remote-ssd-credit",
        type=float,
        default=env.router_remote_ssd_credit,
        help=f"Credit for SSD KV hits (default: {env.router_remote_ssd_credit})",
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
            "bit 2 (4)=router, bit 3 (8)=simulation, bit 4 (16)=bandwidth, "
            "bit 5 (32)=config executor. 0=none, 63=all "
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
        "--num-prefill-nodes",
        type=int,
        default=env.num_prefill_nodes,
        help=f"Number of distinct prefill nodes (default: {env.num_prefill_nodes})",
    )
    parser.add_argument(
        "--num-decode-nodes",
        type=int,
        default=env.num_decode_nodes,
        help=f"Number of distinct decode nodes (default: {env.num_decode_nodes})",
    )
    parser.add_argument(
        "--colocated",
        action="store_true",
        default=env.colocated,
        help=(
            "Run colocated nodes that host both prefill and decode instances "
            f"(default: {env.colocated})"
        ),
    )
    parser.add_argument(
        "--prefill-gpus-per-node",
        type=int,
        default=env.prefill_gpus_per_node,
        help=(
            "GPUs on each node to use as prefill instances. "
            "In --colocated mode the remaining GPUs become decode instances. "
            f"When omitted, colocated mode falls back to --num-prefill-nodes and "
            "non-colocated mode uses all GPUs per prefill-only node "
            f"(default: {env.prefill_gpus_per_node})"
        ),
    )
    parser.add_argument(
        "--machine-hardware",
        type=str,
        default=env.machine_hardware,
        help=(
            f"Hardware preset key from the machine database (default: {env.machine_hardware}). "
            "Quote values containing spaces/hash, e.g. "
            '"H200 x8 #692c33bd"'
        ),
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        default=env.mixed,
        help=(
            "Build a mixed-GPU colocated node: the base machine provides "
            "prefill GPUs and the donor machine provides decode GPUs "
            f"(default: {env.mixed})"
        ),
    )
    parser.add_argument(
        "--mixed-gpu-donor",
        type=str,
        default=env.mixed_gpu_donor,
        help=(
            "Machine name to use as the decode GPU donor in --mixed mode. "
            "If omitted and --mixed is set, falls back to the decode hardware "
            f"(default: {env.mixed_gpu_donor!r})"
        ),
    )
    parser.add_argument(
        "--mixed-gpu-count",
        type=int,
        default=env.mixed_gpu_count,
        help=(
            "Number of donor GPUs to install in each mixed node. "
            "-1 means use the decode-side GPU count "
            f"(default: {env.mixed_gpu_count})"
        ),
    )
    parser.add_argument(
        "--gpu-compute-fraction",
        type=float,
        default=env.gpu_compute_fraction,
        help=(
            "Fraction of a GPU slot's all-in cost attributed to GPU compute. "
            "The remaining fraction is attributed to RAM, SSD, and bandwidth. "
            "Used both for mixed-GPU swaps and for pricing custom/focused machines. "
            f"(default: {env.gpu_compute_fraction})"
        ),
    )
    return parser


def get_create_config_parser(env: EnvConfig) -> argparse.ArgumentParser:
    """CLI parser for create_config.py; mirrors main.py topology flags."""
    parser = get_main_parser(env)
    parser.add_argument(
        "--config-name",
        type=str,
        default="config.json",
        help="Name of the output configuration file (default: config.json)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        default=False,
        help=(
            "Use the legacy Vast.ai scraped machine database instead of the "
            "AWS hardware presets when generating configs."
        ),
    )
    parser.add_argument(
        "--custom-hardware",
        type=str,
        default=None,
        help=(
            "Path to a custom hardware JSON file to use as the machine "
            "database when generating configs. Takes precedence over --legacy."
        ),
    )
    parser.add_argument(
        "--high-end-only",
        action="store_true",
        default=False,
        help=(
            "Only include high-end training GPUs (H100, H200, A100, B200, "
            "B300) when generating configs."
        ),
    )
    parser.add_argument(
        "--config-types",
        type=str,
        default="colocated,mixed,separate",
        help=(
            "Comma-separated list of config categories to generate. "
            "Choices: colocated, mixed, separate (default: all)."
        ),
    )
    return parser
