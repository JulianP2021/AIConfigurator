import argparse
import re

from typing import Any


def get_focus(machine_name: str, gpu_name: str) -> tuple[str, Any]:
    pattern = re.compile(
        rf"^(?:Colocated:\s+)?Focused {re.escape(gpu_name)}(?: "
        r"(?P<focus>RAM|NVLink|SSD|SSD BW|INET BW) "
        r"(?P<value>\d+(?:\.\d+)?)"
        r"(?P<unit>GB|GBps|Gbps)"
        r")?\s+x\d+\b"
    )
    m = pattern.match(machine_name)

    assert m, f"Machine name: {machine_name}, gpu name: {gpu_name}"

    m = m.groupdict()
    return m["focus"], m["value"]


def build_base_config(args: argparse.Namespace, config_type: str) -> dict[str, Any]:
    return {
        "model": args.model,
        "isl": args.isl,
        "osl": args.osl,
        "sessions_per_user": args.sessions_per_user,
        "users": args.users,
        "think_time_ms": args.think_time_ms,
        "max_session_turns": args.max_session_turns,
        "ram_usage_fraction": args.ram_usage_fraction,
        "ssd_usage_fraction": args.ssd_usage_fraction,
        "router_prefill_load_scale": args.router_prefill_load_scale,
        "router_device_credit": args.router_device_credit,
        "router_remote_ram_credit": args.router_remote_ram_credit,
        "router_remote_ssd_credit": args.router_remote_ssd_credit,
        "router_s3_credit": args.router_s3_credit,
        "router_busy_threshold_tokens": args.router_busy_threshold_tokens,
        "s3_enabled": args.s3_enabled,
        "s3_up_bw_gbps": args.s3_up_bw_gbps,
        "s3_down_bw_gbps": args.s3_down_bw_gbps,
        "s3_eviction_time_ms": args.s3_eviction_time_ms,
        "inter_node_network_up_gbps": args.inter_node_network_up_gbps,
        "inter_node_network_down_gbps": args.inter_node_network_down_gbps,
        "sla": {
            "ttft_ms": f"{args.sla['ttft_ms']}",
            "tpot_ms": f"{args.sla['ttft_ms']}",
        },
        "user_delay_fraction": args.user_delay_fraction,
        "user_delay_min_ms": args.user_delay_min_ms,
        "user_delay_max_ms": args.user_delay_max_ms,
        "random_seed": args.random_seed,
        "config_type": config_type,
    }
