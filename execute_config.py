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
        "sessions_per_user": 1,
        "users": 4,
        "max_session_turns": 1,
        "ram_usage_fraction": 0.8,
        "ssd_usage_fraction": 0.8,
        "s3_enabled": true,
        "s3_up_bw_gbps": 25.0,
        "s3_down_bw_gbps": 25.0,
        "s3_eviction_time_ms": 0.0,
        "inter_node_network_up_gbps": 100.0,
        "inter_node_network_down_gbps": 100.0,
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
import copy
import json
import os
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
from src.hardware.hardware import GPUHardwareSpec, S3Spec
from src.hardware.mixed_gpu import fetch_mixed_gpu_hardware
from src.hardware.scraper import fetch_machine_hardware, resolve_machine_name
from src.logger import LOG_CONFIG_EXECUTOR, log, set_log_mask
from src.node.node import Node
from src.request.request import RequestScenario, TokenDistribution
from src.result import SimulationResult
from src.simulations.simulation_distributed import (
    DistributedScenario,
    simulate_run_distributed,
)
from src.utils.env_reader import load_env
from src.utils.output_filter import compact_json
from src.utils.utils import add_result_metadata


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def parse_users_arg(raw: str | None) -> list[int] | None:
    """Parse a --users string into a sorted list of unique positive integers.

    Accepts a comma-separated list of explicit user counts, e.g. ``1,10,100``.
    Each value is validated to be a positive integer.
    """
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"--users values must be positive integers, got: {part}")
        values.append(value)
    if not values:
        raise ValueError(f"--users list is empty: {raw}")
    return sorted(set(values))


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

    mixed_gpu_donor = cfg.get("mixed_gpu_donor")
    mixed_gpu_count = cfg.get("mixed_gpu_count")
    mixed_gpu_count = int(mixed_gpu_count) if mixed_gpu_count is not None else None

    is_mixed = cfg.get("mixed", "").lower() in ["true", "1", "t", "y", "yes"]

    nodes: list[Node] = []
    if colocated or is_mixed:
        if prefill_nodes != decode_nodes:
            raise ValueError(
                f"Config '{cfg.get('label')}' is colocated/mixed but prefill_nodes ({prefill_nodes}) "
                f"!= decode_nodes ({decode_nodes}). In colocated/mixed mode both values represent the number of shared nodes."
            )
        if prefill_gpus + decode_gpus != prefill_total_gpus:
            raise ValueError(
                f"Config '{cfg.get('label')}' GPU split {prefill_gpus}+{decode_gpus} does not equal "
                f"total GPUs per node ({prefill_total_gpus})."
            )

        if is_mixed or mixed_gpu_donor:
            if mixed_gpu_count is None:
                mixed_gpu_count = decode_gpus
            donor_hw_name = resolve_machine_name(mixed_gpu_donor or decode_hw_name)
            node_hw = fetch_mixed_gpu_hardware(
                prefill_hw_name,
                prefill_gpus,
                donor_hw_name,
                mixed_gpu_count,
            )
            from src.hardware.scraper import lookup as lookup_gpu
            from src.hardware.scraper import lookup_machine

            donor_machine_config = lookup_machine(donor_hw_name)
            donor_gpu_name = donor_machine_config["gpu_name"]
            donor_gpu_config = lookup_gpu(donor_gpu_name)
            donor_gpu_spec = GPUHardwareSpec(
                flops=donor_gpu_config["flops"],
                gpu_mem=donor_gpu_config["gpu_mem"],
                gpu_bw=donor_gpu_config["gpu_bw"],
            )
        else:
            if prefill_hw_name != decode_hw_name:
                raise ValueError(
                    f"Config '{cfg.get('label')}' is colocated but prefill_hardware ({prefill_hw_name}) "
                    f"!= decode_hardware ({decode_hw_name}). A colocated node must use one GPU type unless mixed_gpu_donor is set.; {cfg.get('colocated')}"
                )
            node_hw = fetch_machine_hardware(prefill_hw_name)
            donor_gpu_spec = None

        for _ in range(prefill_nodes):
            nodes.append(
                Node(
                    hardware=node_hw,
                    model_name=common["model"],
                    batch_size=batch_size,
                    prefill_instances=prefill_gpus,
                    decode_instances=decode_gpus,
                    decode_gpu_hardware=donor_gpu_spec,
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
            sessions_per_user=int(common["sessions_per_user"]),
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


def extreme_first_eytzinger_layout[T](
    arr: list[T],
    key: Callable[[T], Any],
) -> list[T]:
    """Return arr ordered as [largest, smallest, ...rest in Eytzinger order].

    This probes the two extremes first so the smart runner can invalidate the
    full range quickly: a failure at the largest key can eliminate everything
    smaller, a failure at the smallest key can eliminate everything larger, and
    a success/failure in between is handled by the normal Eytzinger traversal of
    the remaining configs.
    """
    if len(arr) <= 2:
        return sorted(arr, key=key, reverse=True)
    sorted_arr = sorted(arr, key=key)
    largest = sorted_arr[-1]
    smallest = sorted_arr[0]
    middle = sorted_arr[1:-1]
    return [largest, smallest, *eytzinger_layout(middle, key)]


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


def _collect_future_results(
    futures: dict[concurrent.futures.Future, tuple[int, dict[str, Any]]],
    timeout_s: float,
) -> tuple[
    list[tuple[int, dict[str, Any], SimulationResult]],
    list[tuple[int, dict[str, Any], Exception]],
]:
    """Wait up to ``timeout_s`` for all ``futures`` and collect results.

    Returns a pair ``(successful, failed)``.  Any future still pending when its
    per-config deadline expires is cancelled and reported as a timeout.
    """
    import time

    deadline = time.monotonic() + timeout_s
    remaining = dict(futures)
    successful: list[tuple[int, dict[str, Any], SimulationResult]] = []
    failed: list[tuple[int, dict[str, Any], Exception]] = []

    while remaining:
        now = time.monotonic()
        wait_s = max(0.0, min(deadline - now, 1.0))
        done, _ = concurrent.futures.wait(
            remaining, timeout=wait_s, return_when=concurrent.futures.FIRST_COMPLETED
        )

        # If nothing finished and we are past the deadline, treat the rest as
        # timed out so the runner cannot hang on a single slow config.
        if not done and time.monotonic() >= deadline:
            for future, (i, config) in remaining.items():
                future.cancel()
                failed.append((
                    i,
                    config,
                    RuntimeError(f"timed out after {timeout_s}s"),
                ))
                print(
                    f"Config '{config['label']}' timed out after {timeout_s}s",
                    file=sys.stderr,
                )
            remaining.clear()
            break

        for future in done:
            i, config = remaining.pop(future)
            try:
                result = future.result()
                successful.append((i, config, result))
                print(f"Config '{config['label']}' succeeded", file=sys.stdout)
            except Exception as exc:
                failed.append((i, config, exc))
                print(f"Config '{config['label']}' failed: {exc}", file=sys.stderr)

    return successful, failed


def print_table(results: list[tuple[str, str, SimulationResult]]) -> None:
    """Print compact JSON rows for each configuration to stdout.

    The full result rows (including capacity, per-request stats, and phase
    breakdowns) are still written unchanged to the JSON output file.
    """
    for _, label, result in results:
        row = result.to_dict()
        row["label"] = label
        print(compact_json(row))


def build_results_data(
    results: list[tuple[str, str, SimulationResult]],
    configs: list[dict[str, Any]],
    colors: list[str],
    user_counts: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Convert simulation results into the webserver results JSON schema."""
    results_data: list[dict[str, Any]] = []
    for i, (_, label, result) in enumerate(results):
        cfg = configs[i % len(configs)]
        users = user_counts[i // len(configs)] if user_counts else None
        row = result.to_dict()
        add_result_metadata(row, label, cfg, colors[i % len(colors)], users=users)
        results_data.append(row)
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


def _group_separate_configs(
    configs: list[dict[str, Any]],
) -> list[list[list[dict[str, Any]]]]:
    """Group separate (non-colocated) configs for batched execution.

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


def _group_mixed_configs(
    configs: list[dict[str, Any]],
) -> list[list[list[dict[str, Any]]]]:
    """Group mixed-GPU configs for batched execution.

    Mixed configs are colocated nodes where the prefill GPUs come from one
    machine and the decode GPUs come from another.  First level groups by
    ``(prefill_hardware, prefill_gpus_per_node, prefill_nodes)``.  Within each
    first-level group, configs are grouped by ``decode_hardware`` (the donor
    GPU type) and ``mixed_gpu_count``.
    """
    groups: list[list[list[dict[str, Any]]]] = []
    for config in configs:
        placed = False
        for prefill_batch in groups:
            first = prefill_batch[0][0]
            if (
                first["prefill_hardware"] == config["prefill_hardware"]
                and first["prefill_nodes"] == config["prefill_nodes"]
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


def create_separate_batches(
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


def create_mixed_batches(
    splitted_configs: list[list[list[dict[str, Any]]]],
) -> list[list[list[tuple[str, dict[str, Any]]]]]:
    """Create batches for mixed-GPU configs.

    Same structure as ``create_separate_batches``.
    """
    return create_separate_batches(splitted_configs)


def _run_colocated_configs(
    config_batches: list[list[tuple[str, dict[str, Any]]]],
    common: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
    timeout_s: float = 240.0,
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

            collected, failed_in_batch = _collect_future_results(futures, timeout_s)
            for i, config, result in collected:
                results.append((
                    config["prefill_hardware"],
                    config["label"],
                    result,
                ))
                successful.append((i, config))
            failed.extend(failed_in_batch)

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
                        int(cfg["prefill_nodes"]) < int(failed_config["prefill_nodes"])
                    ):
                        next_batch[i] = ("invalid", cfg)
                    if isinstance(exc, DecodeError) and (
                        int(cfg["prefill_nodes"]) <= int(failed_config["prefill_nodes"])
                        and int(cfg["batch_size"]) < int(failed_config["batch_size"])
                    ):
                        next_batch[i] = ("invalid", cfg)
                    if isinstance(exc, DecodeLatencyError) and (
                        int(cfg["prefill_nodes"]) <= int(failed_config["prefill_nodes"])
                        and int(cfg["batch_size"]) > int(failed_config["batch_size"])
                    ):
                        next_batch[i] = ("invalid", cfg)
                        log(
                            LOG_CONFIG_EXECUTOR,
                            f"Invalidated config {cfg['label']} in the next batch due to failed config {failed_config['label']}.",
                        )

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
                    if int(cfg["prefill_nodes"]) > int(
                        successful_config["prefill_nodes"]
                    ) or (
                        int(cfg["prefill_nodes"])
                        == int(successful_config["prefill_nodes"])
                        and int(cfg["batch_size"])
                        > int(successful_config["batch_size"])
                    ):
                        next_batch[i] = ("invalid", cfg)
                        log(
                            LOG_CONFIG_EXECUTOR,
                            f"Invalidated config {cfg['label']} in the next batch due to successful config {successful_config['label']}.",
                        )
    return results


def _run_separate_configs(
    separate_batches: list[list[list[tuple[str, dict[str, Any]]]]],
    common: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
    timeout_s: float = 240.0,
) -> list[tuple[str, str, SimulationResult]]:
    results: list[tuple[str, str, SimulationResult]] = []

    while separate_batches:
        config_batches = separate_batches.pop(0)
        prefill_hw = config_batches[0][0][1]["prefill_hardware"]
        prefill_nodes = int(config_batches[0][0][1]["prefill_nodes"])

        print(
            f"Running separate configs for prefill_hardware: {prefill_hw}, {prefill_nodes}",
            file=sys.stdout,
        )
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

                    collected, failed_in_batch = _collect_future_results(
                        futures, timeout_s
                    )
                    for i, config, result in collected:
                        results.append((
                            config["prefill_hardware"]
                            + " + "
                            + config["decode_hardware"],
                            config["label"],
                            result,
                        ))
                        successful.append((i, config))
                    failed.extend(failed_in_batch)
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
                                int(cfg["decode_nodes"])
                                <= int(failed_config["decode_nodes"])
                                and int(cfg["batch_size"])
                                < int(failed_config["batch_size"])
                            ):
                                next_batch[i] = ("invalid", cfg)
                                invalidated += 1
                                log(
                                    LOG_CONFIG_EXECUTOR,
                                    f"Invalidated config {cfg['label']} in the next batch due to failed config {failed_config['label']}.",
                                )
                            if isinstance(exc, DecodeLatencyError) and (
                                int(cfg["decode_nodes"])
                                <= int(failed_config["decode_nodes"])
                                and int(cfg["batch_size"])
                                > int(failed_config["batch_size"])
                            ):
                                next_batch[i] = ("invalid", cfg)
                                invalidated += 1
                                log(
                                    LOG_CONFIG_EXECUTOR,
                                    f"Invalidated config {cfg['label']} in the next batch due to failed config {failed_config['label']}.",
                                )
                            if isinstance(exc, (DecodeError, DecodeLatencyError)):
                                continue
                            raise exc
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
                            if int(cfg["decode_nodes"]) > int(
                                successful_config["decode_nodes"]
                            ) or (
                                int(cfg["decode_nodes"])
                                == int(successful_config["decode_nodes"])
                                and int(cfg["batch_size"])
                                > int(successful_config["batch_size"])
                            ):
                                invalidated += 1
                                next_batch[i] = ("invalid", cfg)
                                log(
                                    LOG_CONFIG_EXECUTOR,
                                    f"Invalidated config {cfg['label']} in the next batch due to successful config {successful_config['label']}.",
                                )
                        log(
                            LOG_CONFIG_EXECUTOR,
                            f"Invalidated {invalidated} configs in the next batch., {len(config_batches)}, {len(config_batches[0])}",
                        )
                    log(
                        LOG_CONFIG_EXECUTOR,
                        f"Invalidated {invalidated} configs in the config batches.",
                    )
        except Exception as e:
            if isinstance(e, (PrefillError, PrefillLatencyError)):
                separate_batches = [
                    prefill_batch
                    for prefill_batch in separate_batches
                    if len(prefill_batch) > 0
                    and not (
                        prefill_batch[0][0][1]["prefill_hardware"] == prefill_hw
                        and int(prefill_batch[0][0][1]["prefill_nodes"])
                        <= prefill_nodes
                    )
                ]
                log(
                    LOG_CONFIG_EXECUTOR,
                    f"Prefill error occurred for prefill_hardware: {prefill_hw}, {prefill_nodes} nodes. Skipping all configs with this prefill hardware and fewer or equal prefill nodes.",
                )
            else:
                raise e
        else:
            # remove every prefill config with more prefill nodes
            separate_batches = [
                prefill_batch
                for prefill_batch in separate_batches
                if len(prefill_batch) > 0
                and not (
                    prefill_batch[0][0][1]["prefill_hardware"] == prefill_hw
                    and int(prefill_batch[0][0][1]["prefill_nodes"]) >= prefill_nodes
                )
            ]

    return results


def run_all_colocated_configs(
    config_batches: list[list[tuple[str, dict[str, Any]]]],
    common: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
    timeout_s: float = 240.0,
) -> list[tuple[str, str, SimulationResult]]:
    results: list[tuple[str, str, SimulationResult]] = []
    for batch in config_batches:
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

            collected, _ = _collect_future_results(futures, timeout_s)
            for _, config, result in collected:
                results.append((
                    config["prefill_hardware"],
                    config["label"],
                    result,
                ))
    return results


def run_all_separate_configs(
    separate_batches: list[list[list[tuple[str, dict[str, Any]]]]],
    common: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
    timeout_s: float = 240.0,
) -> list[tuple[str, str, SimulationResult]]:
    results: list[tuple[str, str, SimulationResult]] = []
    while separate_batches:
        config_batches = separate_batches.pop(0)
        prefill_hw = config_batches[0][0][1]["prefill_hardware"]
        prefill_nodes = int(config_batches[0][0][1]["prefill_nodes"])

        print(
            f"Running separate configs for prefill_hardware: {prefill_hw}, {prefill_nodes}",
            file=sys.stdout,
        )
        while config_batches:
            batch = config_batches.pop(0)
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

                collected, _ = _collect_future_results(futures, timeout_s)
                for _, config, result in collected:
                    results.append((
                        config["prefill_hardware"],
                        config["label"],
                        result,
                    ))
    return results


def _run_mixed_configs(
    mixed_batches: list[list[list[tuple[str, dict[str, Any]]]]],
    common: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
    timeout_s: float = 240.0,
) -> list[tuple[str, str, SimulationResult]]:
    """Run mixed-GPU configs with the same smart invalidation as separate configs.

    Mixed configs are colocated nodes with prefill GPUs from one machine and
    decode (donor) GPUs from another.  Invalidation logic mirrors
    ``_run_separate_configs`` but uses ``mixed_gpu_count`` instead of
    ``decode_nodes``.
    """
    results: list[tuple[str, str, SimulationResult]] = []

    while mixed_batches:
        config_batches = mixed_batches.pop(0)
        prefill_hw = config_batches[0][0][1]["prefill_hardware"]
        prefill_nodes = int(config_batches[0][0][1]["prefill_nodes"])

        print(
            f"Running mixed-GPU configs for prefill_hardware: {prefill_hw}, {prefill_nodes}",
            file=sys.stdout,
        )
        try:
            while config_batches:
                batch = config_batches.pop(0)

                log(
                    LOG_CONFIG_EXECUTOR,
                    f"\nRunning batch of {len(batch)} mixed configs:",
                )
                for status, cfg in batch:
                    if status == "valid":
                        log(
                            LOG_CONFIG_EXECUTOR,
                            f"  {cfg['label']} x{cfg['prefill_nodes']}(prefill: {cfg['prefill_hardware']} {cfg['prefill_gpus_per_node']}, "
                            f"decode: {cfg['decode_hardware']} x{cfg['decode_gpus_per_node']})",
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

                    collected, failed_in_batch = _collect_future_results(
                        futures, timeout_s
                    )
                    for i, config, result in collected:
                        results.append((
                            config["prefill_hardware"]
                            + " + "
                            + config["decode_hardware"],
                            config["label"],
                            result,
                        ))
                        successful.append((i, config))
                    failed.extend(failed_in_batch)
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
                                int(cfg["decode_gpus_per_node"])
                                <= int(failed_config["decode_gpus_per_node"])
                                and int(cfg["batch_size"])
                                < int(failed_config["batch_size"])
                            ):
                                next_batch[i] = ("invalid", cfg)
                                invalidated += 1
                                log(
                                    LOG_CONFIG_EXECUTOR,
                                    f"Invalidated config {cfg['label']} in the next batch due to failed config {failed_config['label']}.",
                                )
                            if isinstance(exc, DecodeLatencyError) and (
                                int(cfg["decode_gpus_per_node"])
                                <= int(failed_config["decode_gpus_per_node"])
                                and int(cfg["batch_size"])
                                > int(failed_config["batch_size"])
                            ):
                                next_batch[i] = ("invalid", cfg)
                                invalidated += 1
                                log(
                                    LOG_CONFIG_EXECUTOR,
                                    f"Invalidated config {cfg['label']} in the next batch due to failed config {failed_config['label']}.",
                                )
                            if isinstance(exc, (DecodeError, DecodeLatencyError)):
                                continue
                            raise exc
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
                            if int(cfg["decode_gpus_per_node"]) > int(
                                successful_config["decode_gpus_per_node"]
                            ) or (
                                int(cfg["decode_gpus_per_node"])
                                == int(successful_config["decode_gpus_per_node"])
                                and int(cfg["batch_size"])
                                > int(successful_config["batch_size"])
                            ):
                                invalidated += 1
                                next_batch[i] = ("invalid", cfg)
                                log(
                                    LOG_CONFIG_EXECUTOR,
                                    f"Invalidated config {cfg['label']} in the next batch due to successful config {successful_config['label']}.",
                                )
                        log(
                            LOG_CONFIG_EXECUTOR,
                            f"Invalidated {invalidated} configs in the next batch., {len(config_batches)}, {len(config_batches[0])}",
                        )
                    log(
                        LOG_CONFIG_EXECUTOR,
                        f"Invalidated {invalidated} configs in the config batches.",
                    )
        except Exception as e:
            if isinstance(e, (PrefillError, PrefillLatencyError)):
                mixed_batches = [
                    prefill_batch
                    for prefill_batch in mixed_batches
                    if len(prefill_batch) > 0
                    and not (
                        prefill_batch[0][0][1]["prefill_hardware"] == prefill_hw
                        and int(prefill_batch[0][0][1]["prefill_nodes"])
                        <= prefill_nodes
                    )
                ]
                log(
                    LOG_CONFIG_EXECUTOR,
                    f"Prefill error occurred for mixed prefill_hardware: {prefill_hw}, {prefill_nodes} nodes. Skipping all mixed configs with this prefill hardware and fewer or equal prefill nodes.",
                )
            else:
                raise e
        else:
            # remove every mixed config with more prefill nodes
            mixed_batches = [
                prefill_batch
                for prefill_batch in mixed_batches
                if len(prefill_batch) > 0
                and not (
                    prefill_batch[0][0][1]["prefill_hardware"] == prefill_hw
                    and int(prefill_batch[0][0][1]["prefill_nodes"]) >= prefill_nodes
                )
            ]

    return results


def run_all_mixed_configs(
    mixed_batches: list[list[list[tuple[str, dict[str, Any]]]]],
    common: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
    timeout_s: float = 240.0,
) -> list[tuple[str, str, SimulationResult]]:
    """Run all mixed-GPU configs without smart invalidation."""
    results: list[tuple[str, str, SimulationResult]] = []
    while mixed_batches:
        config_batches = mixed_batches.pop(0)
        prefill_hw = config_batches[0][0][1]["prefill_hardware"]
        prefill_nodes = int(config_batches[0][0][1]["prefill_nodes"])

        print(
            f"Running all mixed-GPU configs for prefill_hardware: {prefill_hw}, {prefill_nodes}",
            file=sys.stdout,
        )
        while config_batches:
            batch = config_batches.pop(0)
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

                collected, _ = _collect_future_results(futures, timeout_s)
                for _, config, result in collected:
                    results.append((
                        config["prefill_hardware"] + " + " + config["decode_hardware"],
                        config["label"],
                        result,
                    ))
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
        "--results-dir",
        type=Path,
        required=True,
        help="Directory to write one results JSON file per user count",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-config timeout in seconds (default: 240.0, override config file 'timeout_s')",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        default=False,
        help="Run every valid config without pruning based on prior successes/failures",
    )
    parser.add_argument(
        "--users",
        type=str,
        default=None,
        help="Comma-separated user counts to run, e.g. '1,10,100'",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_log_mask(LOG_CONFIG_EXECUTOR)

    common = {
        "model": config.get("model", env.model),
        "isl": config.get("isl", env.isl),
        "osl": config.get("osl", env.osl),
        "sessions_per_user": config.get("sessions_per_user", env.sessions_per_user),
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
        eviction_time_ms=float(
            config.get("s3_eviction_time_ms", env.s3_eviction_time_ms)
        ),
    )

    # Apply inter-node bandwidth override globally so fetch_machine_hardware()
    # uses it when constructing HardwareSpec instances.
    os.environ["INTER_NODE_NETWORK_UP_GBPS"] = str(
        config.get("inter_node_network_up_gbps", env.inter_node_network_up_gbps)
    )
    os.environ["INTER_NODE_NETWORK_DOWN_GBPS"] = str(
        config.get("inter_node_network_down_gbps", env.inter_node_network_down_gbps)
    )

    configs = config.get("configs", [])
    if not configs:
        raise ValueError("No configs found in the input JSON file.")

    # Validate hardware names up front for clearer errors.
    for cfg in configs:
        resolve_machine_name(cfg["prefill_hardware"])
        resolve_machine_name(cfg["decode_hardware"])

    colocated_configs = [
        cfg
        for cfg in configs
        if cfg.get("colocated", False) == "true" and cfg.get("mixed", "") != "true"
    ]
    mixed_configs = [cfg for cfg in configs if cfg.get("mixed", "") == "true"]
    separate_configs = [
        cfg
        for cfg in configs
        if cfg.get("colocated", False) != "true" and cfg.get("mixed", "") != "true"
    ]

    colocated_config_splitted = _group_colocated_configs(colocated_configs)
    mixed_config_splitted = _group_mixed_configs(mixed_configs)
    separate_config_splitted = _group_separate_configs(separate_configs)

    for i, batch in enumerate(colocated_config_splitted):
        log(
            LOG_CONFIG_EXECUTOR,
            f"B{i}: {[cfg['label'] for cfg in batch]}, {[cfg['prefill_nodes'] for cfg in batch]}",
        )
        colocated_config_splitted[i] = extreme_first_eytzinger_layout(
            batch,
            key=lambda c: (int(c["prefill_nodes"]), int(c["batch_size"])),
        )
        log(
            LOG_CONFIG_EXECUTOR,
            f"A{i}: {[cfg['label'] for cfg in colocated_config_splitted[i]]}, {[cfg['prefill_nodes'] for cfg in colocated_config_splitted[i]]}",
        )

    # Order the outer prefill groups for mixed and separate configs using the
    # same extreme-first Eytzinger strategy, then order the inner decode batches.
    mixed_config_splitted = extreme_first_eytzinger_layout(
        mixed_config_splitted,
        key=lambda pb: (
            pb[0][0]["prefill_hardware"],
            int(pb[0][0]["prefill_nodes"]),
            int(pb[0][0]["prefill_gpus_per_node"]),
        ),
    )
    for i, prefill_batch in enumerate(mixed_config_splitted):
        for j, decode_batch in enumerate(prefill_batch):
            log(
                LOG_CONFIG_EXECUTOR,
                f"B{i}.{j}: {[cfg['label'] for cfg in decode_batch]}, {[cfg['decode_gpus_per_node'] for cfg in decode_batch]}",
            )
            mixed_config_splitted[i][j] = extreme_first_eytzinger_layout(
                decode_batch,
                key=lambda c: (int(c["decode_gpus_per_node"]), int(c["batch_size"])),
            )
            log(
                LOG_CONFIG_EXECUTOR,
                f"A{i}.{j}: {[cfg['label'] for cfg in mixed_config_splitted[i][j]]}, {[cfg['decode_gpus_per_node'] for cfg in mixed_config_splitted[i][j]]}",
            )

    separate_config_splitted = extreme_first_eytzinger_layout(
        separate_config_splitted,
        key=lambda pb: (
            pb[0][0]["prefill_hardware"],
            int(pb[0][0]["prefill_nodes"]),
        ),
    )
    for i, prefill_batch in enumerate(separate_config_splitted):
        for j, decode_batch in enumerate(prefill_batch):
            log(
                LOG_CONFIG_EXECUTOR,
                f"B{i}.{j}: {[cfg['label'] for cfg in decode_batch]}, {[cfg['decode_nodes'] for cfg in decode_batch]}",
            )
            separate_config_splitted[i][j] = extreme_first_eytzinger_layout(
                decode_batch,
                key=lambda c: (int(c["decode_nodes"]), int(c["batch_size"])),
            )
            log(
                LOG_CONFIG_EXECUTOR,
                f"A{i}.{j}: {[cfg['label'] for cfg in separate_config_splitted[i][j]]}, {[cfg['decode_nodes'] for cfg in separate_config_splitted[i][j]]}",
            )

    colocated_config_batches = create_colocated_batches(colocated_config_splitted)
    mixed_config_batches = create_mixed_batches(mixed_config_splitted)
    separate_config_batches = create_separate_batches(separate_config_splitted)

    from src.utils.utils import get_shape

    print(
        "Shapes: ",
        get_shape(colocated_config_splitted),
        get_shape(mixed_config_splitted),
        get_shape(separate_config_splitted),
    )

    timeout_s = (
        args.timeout
        if args.timeout is not None
        else float(config.get("timeout_s", 240.0))
    )

    def _run_all_configs(
        run_all: bool,
        users: int,
    ) -> list[tuple[str, str, SimulationResult]]:
        """Run colocated, mixed, and separate configs for a single user count.

        The grouped config structures are deep-copied for each user count
        because the smart runner mutates batch status tuples in place.
        """
        local_common = dict(common)
        local_common["users"] = users

        local_colocated = copy.deepcopy(colocated_config_batches)
        local_mixed = copy.deepcopy(mixed_config_batches)
        local_separate = copy.deepcopy(separate_config_batches)

        run_results: list[tuple[str, str, SimulationResult]] = []
        if run_all:
            run_results.extend(
                run_all_colocated_configs(
                    local_colocated,
                    local_common,
                    ram_usage_fraction,
                    ssd_usage_fraction,
                    s3_spec,
                    timeout_s=timeout_s,
                )
            )
            run_results.sort(key=lambda x: x[0])
            mixed_results = run_all_mixed_configs(
                local_mixed,
                local_common,
                ram_usage_fraction,
                ssd_usage_fraction,
                s3_spec,
                timeout_s=timeout_s,
            )
            mixed_results.sort(key=lambda x: x[0])
            run_results.extend(mixed_results)
            separate_results = run_all_separate_configs(
                local_separate,
                local_common,
                ram_usage_fraction,
                ssd_usage_fraction,
                s3_spec,
                timeout_s=timeout_s,
            )
        else:
            run_results.extend(
                _run_colocated_configs(
                    local_colocated,
                    local_common,
                    ram_usage_fraction,
                    ssd_usage_fraction,
                    s3_spec,
                    timeout_s=timeout_s,
                )
            )
            run_results.sort(key=lambda x: x[0])
            mixed_results = _run_mixed_configs(
                local_mixed,
                local_common,
                ram_usage_fraction,
                ssd_usage_fraction,
                s3_spec,
                timeout_s=timeout_s,
            )
            mixed_results.sort(key=lambda x: x[0])
            run_results.extend(mixed_results)
            separate_results = _run_separate_configs(
                local_separate,
                local_common,
                ram_usage_fraction,
                ssd_usage_fraction,
                s3_spec,
                timeout_s=timeout_s,
            )
        separate_results.sort(key=lambda x: x[0])
        run_results.extend(separate_results)
        return run_results

    user_counts = parse_users_arg(args.users)
    if user_counts is None:
        user_counts = [int(common["users"])]

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

    all_results: list[tuple[str, str, SimulationResult]] = []
    for users in user_counts:
        print(f"\n### Running with users={users} ###", file=sys.stdout)
        run_results = _run_all_configs(args.run_all, users)
        # Tag each result row with the user count in its label and metadata.
        tagged_results: list[tuple[str, str, SimulationResult]] = []
        for key, label, result in run_results:
            tagged_label = f"{label} (users={users})"
            tagged_results.append((key, tagged_label, result))
            all_results.append((key, tagged_label, result))
        print_table(run_results)

        results_dir = args.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        results_data = build_results_data(
            tagged_results, sorted_configs, colors, [users]
        )
        output_path = results_dir / f"results_users_{users}.json"
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump({"results": results_data}, fh, indent=2)
        print(f"\nWrote results for users={users} to {output_path}")


if __name__ == "__main__":
    main()
