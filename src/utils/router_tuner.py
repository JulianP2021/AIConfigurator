"""Router parameter tuning helper for multi-config simulator runners.

Provides a dynamic, multi-fidelity search that finds the best ``RouterCostConfig``
for a single scenario. The caller supplies an explicit list of user budgets;
candidates are evaluated at each budget in order, failures are dropped, and
survivors are promoted. A final local-refinement step samples around the best
region at the highest successful budget.

Hardware cost is **not** part of the score: every candidate uses the same
machines, so ``total_cost_usd_per_hour`` is identical. The only variables are
the router's own knobs (``active_work_scale`` and ``device_credit``),
which affect placement and therefore latency/throughput.
"""

from __future__ import annotations
import concurrent.futures
import math

from dataclasses import dataclass
from typing import Any, Callable

from src.hardware.hardware import S3Spec
from src.logger import LOG_CONFIG_EXECUTOR, log, should_log
from src.result import SimulationResult
from src.router.router import RouterCostConfig
from src.utils.config_runner import run_single_config


@dataclass(frozen=True)
class TunableRouterParams:
    """One point in the router tuning grid."""

    active_work_scale: float
    device_credit: float

    def to_config(self, base: RouterCostConfig) -> RouterCostConfig:
        return RouterCostConfig(
            prefill_load_scale=base.prefill_load_scale,
            active_work_scale=self.active_work_scale,
            device_credit=self.device_credit,
            remote_ram_credit=base.remote_ram_credit,
            remote_ssd_credit=base.remote_ssd_credit,
            s3_credit=base.s3_credit,
        )


# Coarse grid that covers the empirically successful regime for colocated
# multi-node configs: device_credit around or above 1.0 and active_work_scale
# in the 1e-4..1e-2 range.
_DEFAULT_GRID: list[TunableRouterParams] = [
    TunableRouterParams(0.0001, 1.0),
    TunableRouterParams(0.001, 1.0),
    TunableRouterParams(0.01, 1.0),
    TunableRouterParams(0.0001, 0.8),
    TunableRouterParams(0.001, 0.8),
    TunableRouterParams(0.01, 0.8),
    TunableRouterParams(0.0001, 1.2),
    TunableRouterParams(0.001, 1.2),
]

# Default number of refinement candidates generated around the best coarse point.
_REFINE_COUNT = 4


def _result_score(
    result: SimulationResult | Exception | None,
    budget: int,
) -> tuple[bool, int, float, float, float]:
    """Return a sortable score for a tuning result at a given user budget.

    Higher is better. A successful run that meets SLAs always outranks a
    failure. Among successes, prefer the highest user budget, then higher
    request rate, then lower latency (TTFT + request_latency). For failures,
    the budget at which the run failed is still informative: a config that
    survived until 100 users before failing outranks one that failed at 2.

    Cost is intentionally ignored: every candidate uses the same hardware, so
    ``total_cost_usd_per_hour`` is identical.
    """
    if not isinstance(result, SimulationResult):
        return (False, budget, 0.0, 0.0, 0.0)
    # Throughput: completed sequences per second; higher is better.
    rate = result.seq_per_second
    # Latency headroom: lower combined wait-inclusive latency is better.
    latency_penalty = result.ttft + result.request_latency
    max_latency_penalty = result.max_ttft + result.max_request_latency
    return (True, budget, rate, -latency_penalty, -max_latency_penalty)


def _refinement_grid(
    best: TunableRouterParams,
    count: int = _REFINE_COUNT,
) -> list[TunableRouterParams]:
    """Generate local candidates around ``best``.

    The generated points perturb ``active_work_scale`` and ``device_credit``
    by small factors (±sqrt(2) in log space) so that we explore the neighborhood
    without leaving the valid positive region.
    """
    factors = [2 ** -0.5, 1.0, 2 ** 0.5]
    candidates: list[TunableRouterParams] = []
    for aws_factor in factors:
        for dc_factor in factors:
            # Skip the exact center; it was already evaluated in the coarse grid.
            if aws_factor == 1.0 and dc_factor == 1.0:
                continue
            candidates.append(
                TunableRouterParams(
                    active_work_scale=best.active_work_scale * aws_factor,
                    device_credit=best.device_credit * dc_factor,
                )
            )
    # Return a deterministic, bounded subset.
    return candidates[:count]


def _evaluate_candidates(
    candidates: list[TunableRouterParams],
    common: dict[str, Any],
    cfg: dict[str, Any],
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_spec: S3Spec,
    base_router_config: RouterCostConfig,
    users: int,
    timeout_s: float,
    max_workers: int,
    runner: Callable[..., SimulationResult],
) -> list[tuple[TunableRouterParams, SimulationResult | Exception | None, int]]:
    """Run every candidate at ``users`` and return (params, result, users) tuples."""
    if not candidates:
        return []

    def _run(
        params: TunableRouterParams,
    ) -> tuple[TunableRouterParams, SimulationResult | Exception | None, int]:
        router_cfg = params.to_config(base_router_config)
        run_common = dict(common)
        run_common["users"] = users
        try:
            result = runner(
                run_common,
                cfg,
                ram_usage_fraction,
                ssd_usage_fraction,
                s3_spec,
                router_cfg,
            )
            return (params, result, users)
        except Exception as exc:
            return (params, exc, users)

    results: list[tuple[TunableRouterParams, SimulationResult | Exception | None, int]] = []

    # Run sequentially when only one worker is requested. This also supports
    # injected runners (e.g. in tests) that cannot be pickled into a subprocess.
    if max_workers == 1:
        for params in candidates:
            results.append(_run(params))
        return results

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
                result = future.result(timeout=1.0)
            except Exception:
                result = (params, None, users)
            results.append(result)

    return results


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
    *,
    budgets: list[int] | None = None,
    top_fraction: float = 0.5,
    refine: bool = True,
    _runner: Callable[..., SimulationResult] = run_single_config,
) -> RouterCostConfig:
    """Multi-fidelity search for the best ``RouterCostConfig``.

    The caller supplies an ordered list of user budgets. Candidates are evaluated
    at each budget in ascending order; failures are dropped and survivors are
    promoted to the next budget. Finally, the best coarse point is refined by
    sampling its neighborhood at the highest successful budget.

    Args:
        common: Shared scenario parameters as produced by ``build_common_config``.
            The ``users`` field is overridden internally for each budget.
        cfg: One config entry from the input JSON.
        ram_usage_fraction: Fraction of node RAM usable for the KV cache.
        ssd_usage_fraction: Fraction of node SSD usable for the KV cache.
        s3_spec: Shared S3/object-store specification.
        base_router_config: Default router cost config; scalar values are kept,
            only the tuned knobs (active_work_scale, device_credit) are varied.
        grid: Optional custom parameter grid. If ``None``, a sensible default
            grid is used.
        max_workers: Number of parallel workers for each evaluation round.
        timeout_s: Per-candidate timeout in seconds for each round.
        budgets: Ordered list of user budgets to evaluate. If ``None`` or empty,
            defaults to ``[2]``.
        top_fraction: Fraction of successful survivors to keep after each
            intermediate round. Set to 1.0 to keep all survivors.
        refine: If True, generate a local refinement grid around the best coarse
            point and evaluate it at the highest successful budget.
        _runner: Injectable simulation runner, used for testing. Defaults to
            ``run_single_config``.

    Returns:
        The best ``RouterCostConfig`` found, or ``base_router_config`` if every
        candidate failed at every budget.
    """
    candidates = list(grid if grid is not None else _DEFAULT_GRID)
    if not candidates:
        return base_router_config

    if not budgets:
        budgets = [2]
    budgets = sorted({max(1, int(b)) for b in budgets})

    survivors = list(candidates)
    evaluated: list[
        tuple[TunableRouterParams, SimulationResult | Exception | None, int]
    ] = []
    highest_successful_budget = 0

    for round_idx, users in enumerate(budgets):
        is_final_round = round_idx == len(budgets) - 1
        results = _evaluate_candidates(
            survivors,
            common,
            cfg,
            ram_usage_fraction,
            ssd_usage_fraction,
            s3_spec,
            base_router_config,
            users,
            timeout_s,
            max_workers,
            _runner,
        )
        evaluated.extend(results)

        successes = [
            (params, result, budget)
            for params, result, budget in results
            if isinstance(result, SimulationResult)
        ]
        if not successes:
            break

        highest_successful_budget = max(highest_successful_budget, users)

        if is_final_round:
            survivors = [params for params, _result, _budget in successes]
        else:
            # Sort by score and keep the top fraction for the next round.
            successes.sort(
                key=lambda item: _result_score(item[1], item[2]), reverse=True
            )
            keep = max(1, int(math.ceil(len(successes) * top_fraction)))
            survivors = [params for params, _result, _budget in successes[:keep]]

    # Optional local refinement around the best coarse point.
    if refine and evaluated and highest_successful_budget > 0:
        best_coarse = max(
            evaluated, key=lambda item: _result_score(item[1], item[2])
        )[0]
        refined = _refinement_grid(best_coarse)
        if refined:
            refined_results = _evaluate_candidates(
                refined,
                common,
                cfg,
                ram_usage_fraction,
                ssd_usage_fraction,
                s3_spec,
                base_router_config,
                highest_successful_budget,
                timeout_s,
                max_workers,
                _runner,
            )
            evaluated.extend(refined_results)

    # Pick the best configuration across all evaluated points.
    best_params = candidates[0]
    best_score = _result_score(None, 0)
    best_successful_budget = 0
    for params, result, budget in evaluated:
        score = _result_score(result, budget)
        if score > best_score:
            best_score = score
            best_params = params
            best_successful_budget = budget if isinstance(result, SimulationResult) else 0

    if should_log(LOG_CONFIG_EXECUTOR):
        log(
            LOG_CONFIG_EXECUTOR,
            f"Tuned router: {best_params} (best successful budget={best_successful_budget})",
        )

    if best_score[0] is False:
        return base_router_config

    return best_params.to_config(base_router_config)
