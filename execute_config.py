#!/usr/bin/env python3
"""Multi-config simulator comparison runner.

Reads a JSON config matching the webserver form schema, resolves hardware names
against the local machine database, runs each configuration in an isolated
process, and prints a comparison table.

Usage:
    python compare.py --config config.json

The input JSON schema mirrors the webserver export/import state:

    {
        "model": "Qwen/Qwen3-8B",
        "isl": 128,
        "osl": 8,
        "requests": 4,
        "req_rate": 10.0,
        "max_session_turns": 2,
        "ram_usage_fraction": 0.8,
        "ssd_usage_fraction": 0.8,
        "s3_enabled": true,
        "s3_up_bw_gbps": 25.0,
        "s3_down_bw_gbps": 25.0,
        "configs": [
            {
                "label": "Config 1",
                "prefill_hardware": "H200 x1 #8a0e41af",
                "decode_hardware": "H200 x1 #8a0e41af",
                "prefill_nodes": 1,
                "decode_nodes": 1,
                "batch_size": 10,
                "colocated": false
            }
        ]
    }

Colocated example (prefill + decode share each node):

    {
        "configs": [
            {
                "label": "Colocated",
                "prefill_hardware": "H200 x1 #8a0e41af",
                "decode_hardware": "H200 x1 #8a0e41af",
                "prefill_nodes": 1,
                "decode_nodes": 1,
                "prefill_gpus_per_node": 1,
                "decode_gpus_per_node": 0,
                "batch_size": 10,
                "colocated": true
            }
        ]
    }
"""

import argparse
import concurrent.futures
import json
import sys

from pathlib import Path
from typing import Any


# Ensure project root is on sys.path when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.hardware.hardware import S3Spec
from src.hardware.scraper import resolve_machine_name
from src.logger import set_log_mask
from src.node.node import Node
from src.request.request import RequestScenario, TokenDistribution
from src.result import SimulationResult
from src.simulations.simulation_distributed import (
    DistributedScenario,
    simulate_run_distributed,
)
from src.utils.env_reader import load_env


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def build_scenario(common: dict[str, Any], cfg: dict[str, Any]) -> DistributedScenario:
    """Build a DistributedScenario from one config entry.

    Each config can describe either separate prefill/decode nodes or colocated
    prefill+decode nodes.  When a node has both instance types the KV cache is
    reused locally, eliminating the network KV download.
    """
    required = {
        "label",
        "prefill_hardware",
        "decode_hardware",
        "prefill_nodes",
        "decode_nodes",
        "batch_size",
    }
    missing = sorted(required - cfg.keys())
    if missing:
        raise ValueError(
            f"Config '{cfg.get('label', '<unknown>')}' missing required fields: {missing}"
        )

    prefill_hw_name = resolve_machine_name(cfg["prefill_hardware"])
    decode_hw_name = resolve_machine_name(cfg["decode_hardware"])

    from src.hardware.scraper import fetch_machine_hardware

    prefill_hw = fetch_machine_hardware(prefill_hw_name)
    decode_hw = fetch_machine_hardware(decode_hw_name)

    batch_size = int(cfg["batch_size"])
    prefill_nodes = int(cfg["prefill_nodes"])
    decode_nodes = int(cfg["decode_nodes"])
    colocated = bool(cfg.get("colocated", False))

    # Infer total GPUs per node from the machine key (e.g. "RTX 5090 x2 #...").
    from src.hardware.scraper import parse_gpu_count

    prefill_total_gpus = parse_gpu_count(prefill_hw_name)
    decode_total_gpus = parse_gpu_count(decode_hw_name)
    prefill_gpus = (
        int(cfg.get("prefill_gpus_per_node", prefill_total_gpus)) or prefill_total_gpus
    )
    decode_gpus = (
        int(cfg.get("decode_gpus_per_node", decode_total_gpus)) or decode_total_gpus
    )

    nodes: list[Node] = []
    if colocated:
        if prefill_nodes != decode_nodes:
            raise ValueError(
                f"Config '{cfg.get('label')}' is colocated but prefill_nodes ({prefill_nodes}) "
                f"!= decode_nodes ({decode_nodes}). In colocated mode both values represent the number of shared nodes."
            )
        if prefill_hw_name != decode_hw_name:
            raise ValueError(
                f"Config '{cfg.get('label')}' is colocated but prefill_hardware ({prefill_hw_name}) "
                f"!= decode_hardware ({decode_hw_name}). A colocated node must use one GPU type."
            )
        if prefill_gpus + decode_gpus != prefill_total_gpus:
            raise ValueError(
                f"Config '{cfg.get('label')}' GPU split {prefill_gpus}+{decode_gpus} does not equal "
                f"total GPUs per node ({prefill_total_gpus})."
            )
        for _ in range(prefill_nodes):
            nodes.append(
                Node(
                    hardware=prefill_hw,
                    model_name=common["model"],
                    batch_size=batch_size,
                    prefill_instances=prefill_gpus,
                    decode_instances=decode_gpus,
                )
            )
    else:
        for _ in range(prefill_nodes):
            nodes.append(
                Node(
                    hardware=prefill_hw,
                    model_name=common["model"],
                    batch_size=batch_size,
                    prefill_instances=prefill_gpus,
                    decode_instances=0,
                )
            )
        for _ in range(decode_nodes):
            nodes.append(
                Node(
                    hardware=decode_hw,
                    model_name=common["model"],
                    batch_size=batch_size,
                    prefill_instances=0,
                    decode_instances=decode_gpus,
                )
            )

    return DistributedScenario(
        name=cfg["label"],
        nodes=nodes,
        requests=RequestScenario(
            token_distribution=TokenDistribution(
                min_input_tokens=int(common["isl"]),
                max_input_tokens=int(common["isl"]),
                min_output_tokens=int(common["osl"]),
                max_output_tokens=int(common["osl"]),
            ),
            total_requests=int(common["requests"]),
            min_users=1,
            max_users=10,
            max_session_turns=int(common.get("max_session_turns", 5)),
            req_s=float(common["req_rate"]),
        ),
    )


def _run_single_config(
    common: dict[str, Any],
    cfg: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
) -> SimulationResult:
    """Top-level worker function suitable for process-pool pickling."""
    scenario = build_scenario(common, cfg)
    return simulate_run_distributed(
        scenario,
        ram_usage_fraction=ram_usage_fraction,
        ssd_usage_fraction=ssd_usage_fraction,
        s3_spec=s3_spec,
    )


def print_table(results: list[tuple[str, SimulationResult]]) -> None:
    """Print a simple comparison table to stdout."""
    header = (
        f"{'Label':<20} {'TTFT':>10} {'TPOT':>10} {'Latency':>10} "
        f"{'Tokens/s':>12} {'$/h':>10}"
    )
    print(header)
    print("-" * len(header))
    for label, result in results:
        print(
            f"{label:<20} "
            f"{result.ttft:>10.2f} "
            f"{result.tpot:>10.2f} "
            f"{result.request_latency:>10.2f} "
            f"{result.tokens_per_second:>12.2f} "
            f"{result.price_usd_per_hour:>10.4f}"
        )


def build_results_data(
    results: list[tuple[str, SimulationResult]],
    configs: list[dict[str, Any]],
    colors: list[str],
) -> list[dict[str, Any]]:
    """Convert simulation results into the webserver results JSON schema."""
    results_data: list[dict[str, Any]] = []
    for i, (label, result) in enumerate(results):
        cfg = configs[i]
        results_data.append({
            "label": label,
            "prefill_hardware": cfg.get("prefill_hardware", ""),
            "decode_hardware": cfg.get("decode_hardware", ""),
            "prefill_nodes": cfg.get("prefill_nodes", 0),
            "decode_nodes": cfg.get("decode_nodes", 0),
            "prefill_gpus_per_node": cfg.get("prefill_gpus_per_node", 0),
            "decode_gpus_per_node": cfg.get("decode_gpus_per_node", 0),
            "batch_size": cfg.get("batch_size", 0),
            "colocated": cfg.get("colocated", False),
            "ttft": result.ttft,
            "kv_upload_time": result.kv_upload_time,
            "kv_download_time": result.kv_download_time,
            "max_ttft": result.max_ttft,
            "tpot": result.tpot,
            "max_tpot": result.max_tpot,
            "request_latency": result.request_latency,
            "max_request_latency": result.max_request_latency,
            "tokens_per_second": result.tokens_per_second,
            "tokens_per_second_per_gpu": result.tokens_per_second_per_gpu,
            "request_rate": result.seq_per_second,
            "price_usd_per_hour": result.price_usd_per_hour,
            "color": colors[i % len(colors)],
            "has_error": False,
            "prefill_time": result.avg_prefill_time_ms,
            "prefill_wait": result.avg_prefill_wait_ms,
            "prefill_download_active": result.avg_prefill_download_active_ms,
            "prefill_download_wait": result.avg_prefill_download_wait_ms,
            "prefill_upload_active": result.avg_prefill_upload_active_ms,
            "prefill_upload_wait": result.avg_prefill_upload_wait_ms,
            "decode_download_active": result.avg_decode_download_active_ms,
            "decode_download_wait": result.avg_decode_download_wait_ms,
            "decode_time": result.avg_decode_time_ms,
            "decode_wait": result.avg_decode_wait_ms,
            "decode_upload_active": result.avg_decode_upload_active_ms,
            "decode_upload_wait": result.avg_decode_upload_wait_ms,
            "clean_ttft": result.avg_clean_ttft_ms,
            "clean_latency": result.avg_clean_latency_ms,
        })
    return results_data


def main() -> None:
    env = load_env()
    set_log_mask(0)

    parser = argparse.ArgumentParser(
        description="Run multi-config distributed simulator comparisons"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the JSON config file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write webserver-compatible results JSON (default: <config_dir>/results.json)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    common = {
        "model": config.get("model", env.model),
        "isl": config.get("isl", env.isl),
        "osl": config.get("osl", env.osl),
        "requests": config.get("requests", env.requests),
        "req_rate": config.get("req_rate", env.req_rate),
        "max_session_turns": config.get("max_session_turns", env.max_session_turns),
    }
    ram_usage_fraction = float(config.get("ram_usage_fraction", env.ram_usage_fraction))
    ssd_usage_fraction = float(config.get("ssd_usage_fraction", env.ssd_usage_fraction))
    s3_spec = S3Spec.from_gbps(
        enabled=bool(config.get("s3_enabled", env.s3_enabled)),
        up_gbps=float(config.get("s3_up_bw_gbps", env.s3_up_bw_gbps)),
        down_gbps=float(config.get("s3_down_bw_gbps", env.s3_down_bw_gbps)),
    )

    configs = config.get("configs", [])
    if not configs:
        print("No configs found in JSON.")
        sys.exit(1)

    # Validate hardware names up front for clearer errors.
    for cfg in configs:
        resolve_machine_name(cfg["prefill_hardware"])
        resolve_machine_name(cfg["decode_hardware"])

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                _run_single_config,
                common,
                cfg,
                ram_usage_fraction,
                ssd_usage_fraction,
                s3_spec,
            ): cfg["label"]
            for cfg in configs
        }
        results: list[tuple[str, SimulationResult]] = []
        for future in concurrent.futures.as_completed(futures):
            label = futures[future]
            try:
                result = future.result()
                results.append((label, result))
            except Exception as exc:
                print(f"Config '{label}' failed: {exc}", file=sys.stderr)

    results.sort(key=lambda x: x[0])
    print_table(results)

    # Re-order configs to match sorted results so the JSON rows line up.
    sorted_configs = sorted(configs, key=lambda c: c["label"])

    colors = [
        "#58a6ff",
        "#3fb950",
        "#f85149",
        "#d29922",
        "#a371f7",
        "#79c0ff",
        "#56d364",
        "#f0883e",
    ]
    results_data = build_results_data(results, sorted_configs, colors)

    output_path = args.output or args.config.with_name("results.json")
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump({"results": results_data}, fh, indent=2)
    print(f"\nWrote webserver-compatible results to {output_path}")


if __name__ == "__main__":
    main()
