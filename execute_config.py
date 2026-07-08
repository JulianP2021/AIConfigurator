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

from collections.abc import Callable
from itertools import zip_longest
from pathlib import Path
from typing import Any, TypeVar


# Ensure project root is on sys.path when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.eroors.errors import (
    DecodeError,
    DecodeLatencyError,
    PrefillError,
    PrefillLatencyError,
)
from src.hardware.hardware import S3Spec
from src.hardware.scraper import resolve_machine_name
from src.logger import LOG_CONFIG_EXECUTOR, log, set_log_mask
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
        "colocated",
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
    colocated = bool(
        cfg.get("colocated", "False").lower()
        in ["true", "1", "t", "y", "yes", "yeah", "yup", "certainly", "uh-huh"]
    )

    # Infer total GPUs per node from the machine key (e.g. "RTX 5090 x2 #...").
    from src.hardware.scraper import parse_gpu_count

    prefill_total_gpus = parse_gpu_count(prefill_hw_name)
    decode_total_gpus = parse_gpu_count(decode_hw_name)
    prefill_gpus = int(cfg.get("prefill_gpus_per_node", prefill_total_gpus))
    decode_gpus = int(cfg.get("decode_gpus_per_node", decode_total_gpus))

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
                f"!= decode_hardware ({decode_hw_name}). A colocated node must use one GPU type.; {cfg.get('colocated')}"
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
            users=int(common.get("users", 10)),
            max_session_turns=int(common.get("max_session_turns", 5)),
            think_time_ms=float(common.get("think_time_ms", 0.0)),
        ),
    )


T = TypeVar("T")


def eytzinger_layout[T](
    arr: list[T],
    key: Callable[[T], Any],
) -> list[T]:
    arr = sorted(arr, key=key)

    result: list[T] = [arr[0]] * len(arr)
    i = 0

    def build(k: int) -> None:
        nonlocal i
        if k >= len(result):
            return
        build(2 * k + 1)
        result[k] = arr[i]
        i += 1
        build(2 * k + 2)

    build(0)
    return result


def _run_single_config(
    common: dict[str, Any],
    cfg: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
) -> SimulationResult:
    """Top-level worker function suitable for process-pool pickling."""
    scenario = build_scenario(common, cfg)
    sla = common.get("sla")
    return simulate_run_distributed(
        scenario,
        ram_usage_fraction=ram_usage_fraction,
        ssd_usage_fraction=ssd_usage_fraction,
        s3_spec=s3_spec,
        should_print=False,
        sla=sla,
    )


def print_table(results: list[tuple[str, str, SimulationResult]]) -> None:
    """Print a simple comparison table to stdout."""
    header = (
        f"{'Label':<50} {'avg TTFT':>10} {'TTFT':>10} {'TPOT':>10} {'KV Download':>10} {'KV Upload':>10} {'Latency':>10} "
        f"{'Tokens/s':>12} {'$/h':>10}"
    )
    print(header)
    print("-" * len(header))
    for _, label, result in results:
        print(
            f"{label:<50} "
            f"{result.avg_prefill_time_ms:>10.2f} "
            f"{result.ttft:>10.2f} "
            f"{result.tpot:>10.2f} "
            f"{result.kv_download_time:>10.2f} "
            f"{result.kv_upload_time:>10.2f} "
            f"{result.request_latency:>10.2f} "
            f"{result.tokens_per_second:>12.2f} "
            f"{result.price_usd_per_hour:>10.4f}"
        )


def build_results_data(
    results: list[tuple[str, str, SimulationResult]],
    configs: list[dict[str, Any]],
    colors: list[str],
) -> list[dict[str, Any]]:
    """Convert simulation results into the webserver results JSON schema."""
    results_data: list[dict[str, Any]] = []
    for i, (_, label, result) in enumerate(results):
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


def _group_colocated_configs(
    configs: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group colocated configs by prefill hardware and same number of prefill GPUs.

    All configs in a group share the same node type so they can be compared
    safely in the same execution batch.
    """
    groups: list[list[dict[str, Any]]] = []
    for config in configs:
        for batch in groups:
            if all(
                config["prefill_hardware"] == c["prefill_hardware"]
                and config["prefill_gpus_per_node"] == c["prefill_gpus_per_node"]
                for c in batch
            ):
                batch.append(config)
                break
        else:
            groups.append([config])
    return groups


def _group_single_node_configs(
    configs: list[dict[str, Any]],
) -> list[list[list[dict[str, Any]]]]:
    """Group single-node (non-colocated) configs for batched execution.

    First level groups by ``(prefill_hardware, prefill_nodes)``.
    Within each first-level group, configs are further grouped by
    ``decode_hardware`` so that every inner batch shares both the prefill
    side and the decode GPU type.
    """
    groups: list[list[list[dict[str, Any]]]] = []
    for config in configs:
        placed = False
        for prefill_batch in groups:
            if all(
                c[0]["prefill_hardware"] == config["prefill_hardware"]
                and c[0]["prefill_nodes"] == config["prefill_nodes"]
                for c in prefill_batch
            ):
                for decode_batch in prefill_batch:
                    if all(
                        c["decode_hardware"] == config["decode_hardware"]
                        for c in decode_batch
                    ):
                        decode_batch.append(config)
                        placed = True
                        break
                if not placed:
                    prefill_batch.append([config])
                    placed = True
                break
        if not placed:
            groups.append([[config]])
    return groups


def create_colocated_batches(
    splitted_configs: list[list[dict[str, Any]]],
) -> list[list[tuple[str, dict[str, Any]]]]:
    config_batches: list[list[tuple[str, dict[str, Any]]]] = []

    for group in zip_longest(*splitted_configs):
        _batch: list[tuple[str, dict[str, Any]]] = []
        for x in group:
            if x is not None:
                _batch.append(("valid", x))
            else:
                _batch.append(("invalid", {}))
        config_batches.append(_batch)

    log(
        LOG_CONFIG_EXECUTOR,
        f"Running {len(splitted_configs)} configs in {len(config_batches)} batches...",
    )
    return config_batches


def create_single_node_batches(
    splitted_configs: list[list[list[dict[str, Any]]]],
) -> list[list[list[tuple[str, dict[str, Any]]]]]:
    config_batches: list[list[list[tuple[str, dict[str, Any]]]]] = []

    for prefill_batch in splitted_configs:
        large_batch: list[list[tuple[str, dict[str, Any]]]] = []
        for group in zip_longest(*prefill_batch):
            _batch: list[tuple[str, dict[str, Any]]] = []
            for x in group:
                if x is not None:
                    _batch.append(("valid", x))
                else:
                    _batch.append(("invalid", {}))
            large_batch.append(_batch)
        config_batches.append(large_batch)

    log(
        LOG_CONFIG_EXECUTOR,
        f"Running {len(splitted_configs)} configs in {len(config_batches)} batches...",
    )
    return config_batches


def _run_colocated_configs(
    config_batches: list[list[tuple[str, dict[str, Any]]]],
    common: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
) -> list[tuple[str, str, SimulationResult]]:
    results: list[tuple[str, str, SimulationResult]] = []
    while config_batches:
        batch = config_batches.pop(0)

        log(LOG_CONFIG_EXECUTOR, f"\nRunning batch of {len(batch)} configs:")
        for status, cfg in batch:
            if status == "valid":
                log(
                    LOG_CONFIG_EXECUTOR,
                    f"  {cfg['label']} (prefill: {cfg['prefill_hardware']} x{cfg['prefill_nodes']}, "
                    f"decode: {cfg['decode_hardware']} x{cfg['decode_nodes']})",
                )
        successful: list[tuple[int, dict[str, Any]]] = []
        failed: list[tuple[int, dict[str, Any], Exception]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    _run_single_config,
                    common,
                    cfg,
                    ram_usage_fraction,
                    ssd_usage_fraction,
                    s3_spec,
                ): (i, cfg)
                for (i, (status, cfg)) in enumerate(batch)
                if status == "valid"
            }

            for future in concurrent.futures.as_completed(futures):
                (i, config) = futures[future]
                try:
                    result = future.result()
                    results.append((
                        config["prefill_hardware"],
                        config["label"],
                        result,
                    ))
                    successful.append((i, config))
                    print(f"Config '{config['label']}' succeeded", file=sys.stdout)

                except Exception as exc:
                    failed.append((i, config, exc))
                    print(f"Config '{config['label']}' failed: {exc}", file=sys.stderr)

            for next_batch in config_batches:
                if len(next_batch) == 0:
                    config_batches.remove(next_batch)
                    continue
                for i, failed_config, exc in failed:
                    assert len(next_batch) > i, (
                        f"Next batch does not have index {i} for successful config {failed_config['label']}, {next_batch}"
                    )
                    (status, cfg) = next_batch[i]
                    if status != "valid":
                        continue
                    assert (
                        cfg["prefill_hardware"] == failed_config["prefill_hardware"]
                    ), (
                        f"Config mismatch for failed config {failed_config['label']}, {cfg['prefill_hardware']} != {failed_config['prefill_hardware']}"
                    )
                    assert cfg["decode_hardware"] == failed_config["decode_hardware"], (
                        f"Config mismatch for failed config {failed_config['label']}, {cfg['decode_hardware']} != {failed_config['decode_hardware']}"
                    )
                    if isinstance(exc, (PrefillError, PrefillLatencyError)) and (
                        cfg["prefill_nodes"] < failed_config["prefill_nodes"]
                    ):
                        next_batch[i] = ("invalid", cfg)
                    if isinstance(exc, DecodeError) and (
                        cfg["prefill_nodes"] <= failed_config["prefill_nodes"]
                        and cfg["batch_size"] < failed_config["batch_size"]
                    ):
                        next_batch[i] = ("invalid", cfg)
                    if isinstance(exc, DecodeLatencyError) and (
                        cfg["prefill_nodes"] <= failed_config["prefill_nodes"]
                        and cfg["batch_size"] > failed_config["batch_size"]
                    ):
                        next_batch[i] = ("invalid", cfg)

                for i, successful_config in successful:
                    assert len(next_batch) > i, (
                        f"Next batch does not have index {i} for successful config {successful_config['label']}, {next_batch}"
                    )
                    (status, cfg) = next_batch[i]
                    if status != "valid":
                        continue
                    assert (
                        cfg["prefill_hardware"] == successful_config["prefill_hardware"]
                    ), (
                        f"Config mismatch for successful config {successful_config['label']}, {cfg['prefill_hardware']} != {successful_config['prefill_hardware']}"
                    )
                    assert (
                        cfg["decode_hardware"] == successful_config["decode_hardware"]
                    ), (
                        f"Config mismatch for successful config {successful_config['label']}, {cfg['decode_hardware']} != {successful_config['decode_hardware']}"
                    )
                    if cfg["prefill_nodes"] > successful_config["prefill_nodes"] or (
                        cfg["prefill_nodes"] == successful_config["prefill_nodes"]
                        and cfg["batch_size"] > successful_config["batch_size"]
                    ):
                        next_batch[i] = ("invalid", cfg)
    return results


def _run_single_node_configs(
    single_node_batches: list[list[list[tuple[str, dict[str, Any]]]]],
    common: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
) -> list[tuple[str, str, SimulationResult]]:
    results: list[tuple[str, str, SimulationResult]] = []

    while single_node_batches:
        config_batches = single_node_batches.pop(0)
        prefill_hw = config_batches[0][0][1]["prefill_hardware"]
        print(
            f"Running single-node configs for prefill_hardware: {prefill_hw}",
            file=sys.stdout,
        )
        prefill_nodes = config_batches[0][0][1]["prefill_nodes"]
        try:
            while config_batches:
                batch = config_batches.pop(0)

                log(LOG_CONFIG_EXECUTOR, f"\nRunning batch of {len(batch)} configs:")
                for status, cfg in batch:
                    if status == "valid":
                        log(
                            LOG_CONFIG_EXECUTOR,
                            f"  {cfg['label']} (prefill: {cfg['prefill_hardware']} x{cfg['prefill_nodes']}, "
                            f"decode: {cfg['decode_hardware']} x{cfg['decode_nodes']})",
                        )
                successful: list[tuple[int, dict[str, Any]]] = []
                failed: list[tuple[int, dict[str, Any], Exception]] = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                    futures = {
                        executor.submit(
                            _run_single_config,
                            common,
                            cfg,
                            ram_usage_fraction,
                            ssd_usage_fraction,
                            s3_spec,
                        ): (i, cfg)
                        for (i, (status, cfg)) in enumerate(batch)
                        if status == "valid"
                    }

                    for future in concurrent.futures.as_completed(futures):
                        (i, config) = futures[future]
                        try:
                            result = future.result()
                            results.append((
                                config["prefill_hardware"]
                                + " + "
                                + config["decode_hardware"],
                                config["label"],
                                result,
                            ))
                            successful.append((i, config))
                            print(
                                f"Config '{config['label']}' succeeded", file=sys.stdout
                            )

                        except Exception as exc:
                            if isinstance(exc, (PrefillError, PrefillLatencyError)):
                                raise exc
                            failed.append((i, config, exc))
                            print(
                                f"Config '{config['label']}' failed: {exc}",
                                file=sys.stderr,
                            )
                    invalidated = 0
                    for next_batch in config_batches:
                        if len(next_batch) == 0:
                            config_batches.remove(next_batch)
                            continue
                        for i, failed_config, exc in failed:
                            assert len(next_batch) > i, (
                                f"Next batch does not have index {i} for successful config {failed_config['label']}, {next_batch}"
                            )
                            (status, cfg) = next_batch[i]
                            if status != "valid":
                                continue
                            assert (
                                cfg["prefill_hardware"]
                                == failed_config["prefill_hardware"]
                            ), (
                                f"Config mismatch for failed config {failed_config['label']}, {cfg['prefill_hardware']} != {failed_config['prefill_hardware']}"
                            )
                            assert (
                                cfg["decode_hardware"]
                                == failed_config["decode_hardware"]
                            ), (
                                f"Config mismatch for failed config {failed_config['label']}, {cfg['decode_hardware']} != {failed_config['decode_hardware']}"
                            )
                            if isinstance(exc, DecodeError) and (
                                cfg["decode_nodes"] <= failed_config["decode_nodes"]
                                and cfg["batch_size"] < failed_config["batch_size"]
                            ):
                                next_batch[i] = ("invalid", cfg)
                                invalidated += 1
                                # print(f"Invalidated config {cfg['label']} in the next batch due to failed config {failed_config['label']}.", file=sys.stdout)
                            if isinstance(exc, DecodeLatencyError) and (
                                cfg["decode_nodes"] <= failed_config["decode_nodes"]
                                and cfg["batch_size"] > failed_config["batch_size"]
                            ):
                                next_batch[i] = ("invalid", cfg)
                                invalidated += 1
                                # print(f"Invalidated config {cfg['label']} in the next batch due to failed config {failed_config['label']}.", file=sys.stdout)
                        for i, successful_config in successful:
                            assert len(next_batch) > i, (
                                f"Next batch does not have index {i} for successful config {successful_config['label']}, {next_batch}"
                            )
                            (status, cfg) = next_batch[i]
                            if status != "valid":
                                continue
                            assert (
                                cfg["prefill_hardware"]
                                == successful_config["prefill_hardware"]
                            ), (
                                f"Config mismatch for successful config {successful_config['label']}, {cfg['prefill_hardware']} != {successful_config['prefill_hardware']}"
                            )
                            assert (
                                cfg["decode_hardware"]
                                == successful_config["decode_hardware"]
                            ), (
                                f"Config mismatch for successful config {successful_config['label']}, {cfg['decode_hardware']} != {successful_config['decode_hardware']}"
                            )
                            if cfg["decode_nodes"] > successful_config[
                                "decode_nodes"
                            ] or (
                                cfg["decode_nodes"] == successful_config["decode_nodes"]
                                and cfg["batch_size"] > successful_config["batch_size"]
                            ):
                                invalidated += 1
                                next_batch[i] = ("invalid", cfg)
                                # print(f"Invalidated config {cfg['label']} in the next batch due to successful config {successful_config['label']}.", file=sys.stdout)
                        # print(f"Invalidated {invalidated} configs in the next batch., {len(config_batches)}, {len(config_batches[0])}", file=sys.stdout)
                    print(
                        f"Invalidated {invalidated} configs in the config batches.",
                        file=sys.stdout,
                    )
        except RuntimeError as e:
            if isinstance(e, (PrefillError, PrefillLatencyError)):
                single_node_batches = [
                    prefill_batch
                    for prefill_batch in single_node_batches
                    if len(prefill_batch) > 0
                    and not (
                        prefill_batch[0][0][1]["prefill_hardware"] == prefill_hw
                        and prefill_batch[0][0][1]["prefill_nodes"] <= prefill_nodes
                    )
                ]
            else:
                raise e
        else:
            # remove every prefill config with more prefill nodes
            single_node_batches = [
                prefill_batch
                for prefill_batch in single_node_batches
                if len(prefill_batch) > 0
                and not (
                    prefill_batch[0][0][1]["prefill_hardware"] == prefill_hw
                    and prefill_batch[0][0][1]["prefill_nodes"] >= prefill_nodes
                )
            ]

    return results


def main() -> None:
    env = load_env()
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
    set_log_mask(LOG_CONFIG_EXECUTOR)

    common = {
        "model": config.get("model", env.model),
        "isl": config.get("isl", env.isl),
        "osl": config.get("osl", env.osl),
        "requests": config.get("requests", env.requests),
        "users": config.get("users", env.users),
        "think_time_ms": config.get("think_time_ms", env.think_time_ms),
        "max_session_turns": config.get("max_session_turns", env.max_session_turns),
        "sla": config.get(
            "sla",
            {
                "ttft_ms": config.get("sla_ttft_ms", env.sla_ttft_ms),
                "tpot_ms": config.get("sla_tpot_ms", env.sla_tpot_ms),
            },
        ),
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
        raise ValueError("No configs found in the input JSON file.")

    # Validate hardware names up front for clearer errors.
    for cfg in configs:
        resolve_machine_name(cfg["prefill_hardware"])
        resolve_machine_name(cfg["decode_hardware"])

    colocated_configs = [
        cfg for cfg in configs if cfg.get("colocated", False) == "true"
    ]
    single_node_configs = [
        cfg for cfg in configs if cfg.get("colocated", False) != "true"
    ]

    colocated_config_splitted = _group_colocated_configs(colocated_configs)
    single_node_config_splitted = _group_single_node_configs(single_node_configs)

    for i, batch in enumerate(colocated_config_splitted):
        log(
            LOG_CONFIG_EXECUTOR,
            f"B{i}: {[cfg['label'] for cfg in batch]}, {[cfg['prefill_nodes'] for cfg in batch]}",
        )
        colocated_config_splitted[i] = eytzinger_layout(
            batch,
            key=lambda c: int(c["prefill_nodes"]) * 1000 + int(c["batch_size"]),
        )
        log(
            LOG_CONFIG_EXECUTOR,
            f"A{i}: {[cfg['label'] for cfg in colocated_config_splitted[i]]}, {[cfg['prefill_nodes'] for cfg in colocated_config_splitted[i]]}",
        )

    for i, prefill_batch in enumerate(single_node_config_splitted):
        for j, decode_batch in enumerate(prefill_batch):
            log(
                LOG_CONFIG_EXECUTOR,
                f"B{i}.{j}: {[cfg['label'] for cfg in decode_batch]}, {[cfg['decode_nodes'] for cfg in decode_batch]}",
            )
            single_node_config_splitted[i][j] = eytzinger_layout(
                decode_batch,
                key=lambda c: int(c["decode_nodes"]) * 100 + int(c["batch_size"]),
            )
            log(
                LOG_CONFIG_EXECUTOR,
                f"A{i}.{j}: {[cfg['label'] for cfg in single_node_config_splitted[i][j]]}, {[cfg['decode_nodes'] for cfg in single_node_config_splitted[i][j]]}",
            )

    colocated_config_batches = create_colocated_batches(colocated_config_splitted)
    single_node_config_batches = create_single_node_batches(single_node_config_splitted)

    from src.utils.utils import get_shape

    print(
        "Shapes: ",
        get_shape(colocated_config_splitted),
        get_shape(single_node_config_splitted),
    )

    results: list[tuple[str, str, SimulationResult]] = []

    results.extend(
        _run_colocated_configs(
            colocated_config_batches,
            common,
            ram_usage_fraction,
            ssd_usage_fraction,
            s3_spec,
        )
    )
    results.sort(key=lambda x: x[0])
    single_node_results = _run_single_node_configs(
        single_node_config_batches,
        common,
        ram_usage_fraction,
        ssd_usage_fraction,
        s3_spec,
    )
    single_node_results.sort(key=lambda x: x[0])
    results.extend(single_node_results)

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
