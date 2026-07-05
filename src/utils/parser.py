import argparse

from src.utils.env_reader import EnvConfig


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


def get_main_parser(env: EnvConfig) -> argparse.ArgumentParser:
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
    return parser
