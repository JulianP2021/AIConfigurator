"""Router parameter tuning helper for multi-config simulator runners.

Provides a reusable grid search that finds the best ``RouterCostConfig`` for a
single scenario, using the same cost function and success semantics as the rest
of the simulator. The tuner is intentionally minimal: it evaluates a small
parameter grid in parallel and picks the configuration that (1) meets the SLAs,
(2) maximizes throughput, and (3) has the most latency headroom.

Hardware cost is **not** part of the score: every candidate uses the same
machines, so ``total_cost_usd_per_hour`` is identical. The only variables are
the router's own knobs (``active_work_scale``, ``device_credit``,
``busy_threshold_tokens``), which affect placement and therefore latency/throughput.
"""

from __future__ import annotations
import concurrent.futures

from dataclasses import dataclass
from typing import Any

from src.hardware.hardware import S3Spec
from src.result import SimulationResult
from src.router.router import RouterCostConfig
from src.utils.config_runner import run_single_config


@dataclass(frozen=True)
class TunableRouterParams:
    """One point in the router tuning grid."""

    active_work_scale: float
    device_credit: float
    busy_threshold_tokens: float

    def to_config(self, base: RouterCostConfig) -> RouterCostConfig:
        return RouterCostConfig(
            prefill_load_scale=base.prefill_load_scale,
            active_work_scale=self.active_work_scale,
            device_credit=self.device_credit,
            remote_ram_credit=base.remote_ram_credit,
            remote_ssd_credit=base.remote_ssd_credit,
            s3_credit=base.s3_credit,
            busy_threshold_tokens=self.busy_threshold_tokens,
        )


_DEFAULT_GRID: list[TunableRouterParams] = [
    TunableRouterParams(0.001, 0.5, 1_000_000.0),
    TunableRouterParams(0.001, 0.8, 1_000_000.0),
    TunableRouterParams(0.001, 1.0, 1_000_000.0),
    TunableRouterParams(0.01, 0.5, 1_000_000.0),
    TunableRouterParams(0.01, 0.8, 1_000_000.0),
    TunableRouterParams(0.01, 1.0, 1_000_000.0),
    TunableRouterParams(0.001, 0.8, 2_000_000.0),
    TunableRouterParams(0.01, 0.8, 2_000_000.0),
]


def _result_score(
    result: SimulationResult | Exception | None,
) -> tuple[bool, float, float, float]:
    """Return a sortable score for a tuning result.

    Higher is better. A successful run that meets SLAs always outranks a
    failure. Among successes, prefer higher request rate and lower latency
    (TTFT + request_latency). Cost is intentionally ignored: every candidate
    uses the same hardware, so ``total_cost_usd_per_hour`` is identical.
    """
    if not isinstance(result, SimulationResult):
        return (False, 0.0, 0.0, 0.0)
    rate = float(getattr(result, "request_rate", 0.0))
    # Latency headroom: lower combined wait-inclusive latency is better.
    latency_penalty = result.ttft + result.request_latency
    max_latency_penalty = result.max_ttft + result.max_request_latency
    return (True, rate, -latency_penalty, -max_latency_penalty)


def tune_router_for_config(
    common: dict[str, Any],
    cfg: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
    base_router_config: RouterCostConfig,
    grid: list[TunableRouterParams] | None = None,
    max_workers: int = 4,
    timeout_s: float = 120.0,
) -> RouterCostConfig:
    """Run a small grid search and return the best ``RouterCostConfig``.

    The grid is evaluated in parallel using a process pool. Each candidate gets
    the same ``common``/``cfg`` scenario. The winner is the candidate that
    produces a ``SimulationResult`` (no exception) with the highest request
    rate; ties are broken by lower wait-inclusive latency. If every candidate
    fails, the original ``base_router_config`` is returned so the caller can
    still attempt a full run.

    Args:
        common: Shared scenario parameters as produced by ``build_common_config``.
        cfg: One config entry from the input JSON.
        ram_usage_fraction: Fraction of node RAM usable for the KV cache.
        ssd_usage_fraction: Fraction of node SSD usable for the KV cache.
        s3_spec: Shared S3/object-store specification.
        base_router_config: Default router cost config; scalar values are kept,
            only the tuned knobs (active_work_scale, device_credit,
            busy_threshold_tokens) are varied.
        grid: Optional custom parameter grid. If ``None``, a sensible default
            grid is used.
        max_workers: Number of parallel workers for the grid search.
        timeout_s: Per-candidate timeout in seconds.

    Returns:
        The best ``RouterCostConfig`` found, or ``base_router_config`` if all
        candidates failed.
    """
    candidates = grid if grid is not None else _DEFAULT_GRID
    if not candidates:
        return base_router_config

    def _run(
        params: TunableRouterParams,
    ) -> tuple[TunableRouterParams, SimulationResult | Exception | None]:
        router_cfg = params.to_config(base_router_config)
        try:
            result = run_single_config(
                common,
                cfg,
                ram_usage_fraction,
                ssd_usage_fraction,
                s3_spec,
                router_cfg,
            )
            return (params, result)
        except Exception as exc:
            return (params, exc)

    best_params = candidates[0]
    best_result: SimulationResult | Exception | None = None
    best_score = _result_score(best_result)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run, params): params for params in candidates}
        deadline = concurrent.futures.wait(
            futures,
            timeout=timeout_s * len(candidates),
            return_when=concurrent.futures.ALL_COMPLETED,
        )[0]

        for future in deadline:
            params = futures[future]
            try:
                _, result = future.result(timeout=1.0)
            except Exception:
                result = None
            score = _result_score(result)
            if score > best_score:
                best_score = score
                best_params = params
                best_result = result

    if isinstance(best_result, SimulationResult) and hasattr(
        best_result, "router_active_work_scale"
    ):
        best_result.router_active_work_scale = best_params.active_work_scale
        best_result.router_device_credit = best_params.device_credit
        best_result.router_busy_threshold_tokens = best_params.busy_threshold_tokens

    return best_params.to_config(base_router_config)
