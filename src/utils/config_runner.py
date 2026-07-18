"""Shared helpers for multi-config simulator runners."""

from __future__ import annotations
import concurrent.futures
import copy
import json
import sys

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from src.hardware.hardware import GPUHardwareSpec, S3Spec
from src.hardware.mixed_gpu import fetch_mixed_gpu_hardware
from src.hardware.scraper import fetch_machine_hardware, resolve_machine_name
from src.node.node import Node
from src.request.request import RequestScenario, TokenDistribution
from src.result import SimulationResult
from src.router.router import RouterCostConfig
from src.simulations.simulation_distributed import (
    DistributedScenario,
    simulate_run_distributed,
)
from src.utils.env_reader import EnvConfig


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "t", "y", "yes"}


def build_common_config(
    config: dict[str, Any],
    env: EnvConfig,
    *,
    sla_override: dict[str, float] | None = None,
    user_delay_fraction_override: float | None = None,
    user_delay_min_ms_override: float | None = None,
    user_delay_max_ms_override: float | None = None,
) -> dict[str, Any]:
    raw_sla = config.get("sla")
    if sla_override is not None:
        sla = {
            "ttft_ms": float(sla_override["ttft_ms"]),
            "tpot_ms": float(sla_override["tpot_ms"]),
        }
    elif raw_sla:
        sla = {
            "ttft_ms": float(raw_sla["ttft_ms"]),
            "tpot_ms": float(raw_sla["tpot_ms"]),
        }
    else:
        sla = {"ttft_ms": float(env.sla_ttft_ms), "tpot_ms": float(env.sla_tpot_ms)}

    return {
        "model": config.get("model", env.model),
        "isl": config.get("isl", env.isl),
        "osl": config.get("osl", env.osl),
        "sessions_per_user": config.get("sessions_per_user", env.sessions_per_user),
        "users": config.get("users", env.users),
        "think_time_ms": config.get("think_time_ms", env.think_time_ms),
        "max_session_turns": config.get("max_session_turns", env.max_session_turns),
        "user_delay_fraction": (
            user_delay_fraction_override
            if user_delay_fraction_override is not None
            else config.get("user_delay_fraction", env.user_delay_fraction)
        ),
        "user_delay_min_ms": (
            user_delay_min_ms_override
            if user_delay_min_ms_override is not None
            else config.get("user_delay_min_ms", env.user_delay_min_ms)
        ),
        "user_delay_max_ms": (
            user_delay_max_ms_override
            if user_delay_max_ms_override is not None
            else config.get("user_delay_max_ms", env.user_delay_max_ms)
        ),
        "random_seed": config.get("random_seed", env.random_seed),
        "sla": sla,
    }


def build_scenario(
    common: dict[str, Any], cfg: dict[str, Any], custom_path: str | None = None
) -> DistributedScenario:
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

    prefill_hw_name = resolve_machine_name(cfg["prefill_hardware"], custom_path)
    decode_hw_name = resolve_machine_name(cfg["decode_hardware"], custom_path)

    prefill_hw = fetch_machine_hardware(prefill_hw_name, custom_path=custom_path)
    decode_hw = fetch_machine_hardware(decode_hw_name, custom_path=custom_path)

    batch_size = int(cfg["batch_size"])
    prefill_nodes = int(cfg["prefill_nodes"])
    decode_nodes = int(cfg["decode_nodes"])
    colocated = _parse_bool(cfg.get("colocated", False))

    from src.hardware.scraper import parse_gpu_count

    prefill_total_gpus = parse_gpu_count(prefill_hw_name)
    decode_total_gpus = parse_gpu_count(decode_hw_name)
    prefill_gpus = int(cfg.get("prefill_gpus_per_node", prefill_total_gpus))
    decode_gpus = int(cfg.get("decode_gpus_per_node", decode_total_gpus))

    mixed_gpu_donor = cfg.get("mixed_gpu_donor")
    mixed_gpu_count = cfg.get("mixed_gpu_count")
    mixed_gpu_count = int(mixed_gpu_count) if mixed_gpu_count is not None else None

    is_mixed = _parse_bool(cfg.get("mixed", False))

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


def run_single_config(
    common: dict[str, Any],
    cfg: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
    router_cost_config: RouterCostConfig,
) -> SimulationResult:
    scenario = build_scenario(common, cfg)
    return simulate_run_distributed(
        scenario,
        ram_usage_fraction=ram_usage_fraction,
        ssd_usage_fraction=ssd_usage_fraction,
        s3_spec=s3_spec,
        router_cost_config=router_cost_config,
        should_print=False,
        sla=common.get("sla"),
        user_delay_fraction=float(common.get("user_delay_fraction", 0.0)),
        user_delay_min_ms=float(common.get("user_delay_min_ms", 0.0)),
        user_delay_max_ms=float(common.get("user_delay_max_ms", 0.0)),
        random_seed=int(str(common["random_seed"]), 0)
        if common.get("random_seed") is not None
        else None,
    )


def collect_future_results(
    futures: dict[concurrent.futures.Future, tuple[int, dict[str, Any]]],
    timeout_s: float,
) -> tuple[
    list[tuple[int, dict[str, Any], SimulationResult]],
    list[tuple[int, dict[str, Any], Exception]],
]:
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


def validate_colocated_configs(configs: list[dict[str, Any]]) -> None:
    invalid: list[str] = []
    for cfg in configs:
        colocated = _parse_bool(cfg.get("colocated", False))
        same_hardware = cfg.get("prefill_hardware") == cfg.get("decode_hardware")
        same_nodes = str(cfg.get("prefill_nodes")) == str(cfg.get("decode_nodes"))
        if not (colocated and same_hardware and same_nodes):
            invalid.append(str(cfg.get("label", "<unknown>")))
    if invalid:
        raise ValueError(
            "All configs must be colocated for this runner; invalid configs: "
            + ", ".join(invalid)
        )


def clone_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(cfg)


def _run_colocated_batches(
    config_batches: list[list[tuple[str, dict[str, Any]]]],
    common: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
    router_cost_config: RouterCostConfig,
    timeout_s: float = 240.0,
) -> list[tuple[str, str, SimulationResult]]:
    results: list[tuple[str, str, SimulationResult]] = []
    while config_batches:
        batch = config_batches.pop(0)
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    run_single_config,
                    common,
                    cfg,
                    ram_usage_fraction,
                    ssd_usage_fraction,
                    s3_spec,
                    router_cost_config,
                ): (i, cfg)
                for (i, (status, cfg)) in enumerate(batch)
                if status == "valid"
            }

            collected, failed = collect_future_results(futures, timeout_s)
            if failed:
                # Colocated-only benchmarks should not silently continue on errors.
                first = failed[0][2]
                raise first
            for _, config, result in collected:
                results.append((config["prefill_hardware"], config["label"], result))
    return results


def run_flat_configs(
    configs: list[dict[str, Any]],
    common: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
    router_cost_config: RouterCostConfig,
    timeout_s: float = 240.0,
) -> list[tuple[str, str, SimulationResult]]:
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                run_single_config,
                common,
                cfg,
                ram_usage_fraction,
                ssd_usage_fraction,
                s3_spec,
                router_cost_config,
            ): (i, cfg)
            for i, cfg in enumerate(configs)
        }
        collected, failed = collect_future_results(futures, timeout_s)
        if failed:
            first = failed[0][2]
            raise first

    results: list[tuple[str, str, SimulationResult]] = []
    for _, config, result in collected:
        results.append((config["prefill_hardware"], config["label"], result))
    return results


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
    if len(arr) <= 2:
        return sorted(arr, key=key, reverse=True)
    sorted_arr = sorted(arr, key=key)
    largest = sorted_arr[-1]
    smallest = sorted_arr[0]
    middle = sorted_arr[1:-1]
    return [largest, smallest, *eytzinger_layout(middle, key)]
