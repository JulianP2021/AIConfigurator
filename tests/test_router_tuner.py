"""Tests for the dynamic router tuner."""

import pytest

from src.hardware.hardware import S3Spec
from src.result import SimulationResult
from src.router.router import RouterCostConfig
from src.utils.router_tuner import (
    TunableRouterParams,
    _evaluate_candidates,
    _refinement_grid,
    _result_score,
    tune_router_for_config,
)


def _make_result(
    seq_per_second: float = 1.0, ttft: float = 100.0, latency: float = 200.0
) -> SimulationResult:
    """Return a minimal SimulationResult for scoring."""
    return SimulationResult(
        scenario_name="test",
        total_gpus=1,
        num_prefill_workers=1,
        num_decode_workers=1,
        prefill_gpus_per_worker=1,
        decode_gpus_per_worker=1,
        batch_size=1,
        ttft=ttft,
        tpot=10.0,
        kv_upload_time=0.0,
        kv_download_time=0.0,
        request_latency=latency,
        max_request_latency=latency,
        max_ttft=ttft,
        max_tpot=10.0,
        tokens_per_second=100.0,
        tokens_per_second_per_gpu=100.0,
        tokens_per_second_per_user=10.0,
        seq_per_second=seq_per_second,
        concurrency=1.0,
        memory_gb=1.0,
        compute_price_usd_per_hour=1.0,
    )


def test_result_score_failure_is_worst():
    success = _make_result()
    failure = RuntimeError("boom")
    assert _result_score(success, 2) > _result_score(failure, 2)
    assert _result_score(None, 2) == _result_score(failure, 2)


def test_result_score_prefers_higher_rate():
    fast = _make_result(seq_per_second=2.0)
    slow = _make_result(seq_per_second=1.0)
    assert _result_score(fast, 2) > _result_score(slow, 2)


def test_result_score_tiebreaks_on_latency():
    low_latency = _make_result(seq_per_second=1.0, ttft=50.0, latency=100.0)
    high_latency = _make_result(seq_per_second=1.0, ttft=150.0, latency=300.0)
    assert _result_score(low_latency, 2) > _result_score(high_latency, 2)


def test_result_score_prefers_higher_budget():
    low_budget = _make_result(seq_per_second=1.0)
    high_budget = _make_result(seq_per_second=0.5)
    assert _result_score(high_budget, 16) > _result_score(low_budget, 2)


def test_refinement_grid_excludes_center():
    best = TunableRouterParams(0.001, 1.0)
    grid = _refinement_grid(best, count=100)
    assert TunableRouterParams(0.001, 1.0) not in grid
    # All generated points are positive perturbations of the center.
    assert all(p.active_work_scale > 0 and p.device_credit > 0 for p in grid)


def test_refinement_grid_bounded_count():
    best = TunableRouterParams(0.001, 1.0)
    grid = _refinement_grid(best, count=4)
    assert len(grid) == 4


def test_tune_router_returns_base_when_grid_empty():
    base = RouterCostConfig()
    result = tune_router_for_config(
        {"users": 10},
        {},
        0.1,
        0.1,
        S3Spec.from_gbps(False, 0.0, 0.0),
        base,
        grid=[],
    )
    assert result is base


def test_tune_router_drops_failures_and_promotes_survivors():
    """A config that fails at low users is never evaluated at high users."""
    calls: list[tuple[int, TunableRouterParams]] = []

    def fake_runner(common, _cfg, _ram, _ssd, _s3, router_cfg):
        users = int(common["users"])
        # Find which params were used by comparing active_work_scale.
        for p in [TunableRouterParams(0.0001, 1.0), TunableRouterParams(0.1, 0.5)]:
            if abs(router_cfg.active_work_scale - p.active_work_scale) < 1e-9:
                calls.append((users, p))
                if p.active_work_scale == 0.1:
                    raise RuntimeError("bad config")
                return _make_result()
        raise ValueError("unexpected params")

    grid = [
        TunableRouterParams(0.0001, 1.0),  # good
        TunableRouterParams(0.1, 0.5),  # bad
    ]
    base = RouterCostConfig()
    chosen = tune_router_for_config(
        {"users": 8},
        {},
        0.1,
        0.1,
        S3Spec.from_gbps(False, 0.0, 0.0),
        base,
        grid=grid,
        budgets=[1, 2, 4, 8],
        top_fraction=0.5,
        refine=False,
        max_workers=1,
        _runner=fake_runner,
    )

    assert chosen.active_work_scale == 0.0001
    assert chosen.device_credit == 1.0
    # Bad config should only have been evaluated at the first budget.
    bad_calls = [u for u, p in calls if p.active_work_scale == 0.1]
    assert bad_calls == [1]


def test_tune_router_refines_around_best():
    """Refinement runs additional candidates at the full target load."""
    calls: list[int] = []

    def fake_runner(common, _cfg, _ram, _ssd, _s3, router_cfg):
        users = int(common["users"])
        calls.append(users)
        return _make_result(seq_per_second=router_cfg.device_credit)

    grid = [TunableRouterParams(0.001, 1.0)]
    base = RouterCostConfig()
    chosen = tune_router_for_config(
        {"users": 8},
        {},
        0.1,
        0.1,
        S3Spec.from_gbps(False, 0.0, 0.0),
        base,
        grid=grid,
        budgets=[2, 4, 8],
        refine=True,
        max_workers=1,
        _runner=fake_runner,
    )

    # The best refinement point has device_credit = 1.0 * sqrt(2) ~ 1.414,
    # which yields a higher seq_per_second in the fake runner.
    assert chosen.device_credit == pytest.approx(1.0 * (2**0.5))
    # Refinement candidates run at the highest successful budget (8).
    assert 8 in calls


def test_tune_router_falls_back_when_all_fail():
    base = RouterCostConfig(active_work_scale=0.123, device_credit=0.456)

    def fake_runner(_common, _cfg, _ram, _ssd, _s3, _router_cfg):
        raise RuntimeError("always fails")

    chosen = tune_router_for_config(
        {"users": 4},
        {},
        0.1,
        0.1,
        S3Spec.from_gbps(False, 0.0, 0.0),
        base,
        grid=[TunableRouterParams(0.001, 1.0)],
        budgets=[4],
        refine=False,
        max_workers=1,
        _runner=fake_runner,
    )

    assert chosen is base
    assert chosen.active_work_scale == 0.123
    assert chosen.device_credit == 0.456


def test_evaluate_candidates_empty():
    assert (
        _evaluate_candidates(
            [],
            {"users": 1},
            {},
            0.1,
            0.1,
            S3Spec.from_gbps(False, 0.0, 0.0),
            RouterCostConfig(),
            1,
            10.0,
            1,
            lambda **_kw: _make_result(),
        )
        == []
    )


def test_tune_router_uses_budgets_parameter():
    """The tuner respects the explicit budgets list."""
    calls: list[int] = []

    def fake_runner(common, _cfg, _ram, _ssd, _s3, _router_cfg):
        calls.append(int(common["users"]))
        return _make_result()

    base = RouterCostConfig()
    tune_router_for_config(
        {"users": 100},
        {},
        0.1,
        0.1,
        S3Spec.from_gbps(False, 0.0, 0.0),
        base,
        grid=[TunableRouterParams(0.001, 1.0)],
        budgets=[2, 8, 16],
        refine=False,
        max_workers=1,
        _runner=fake_runner,
    )

    assert sorted(set(calls)) == [2, 8, 16]


def test_tune_router_budgets_are_deduped_and_sorted():
    """Duplicate or out-of-order budgets are normalized."""
    calls: list[int] = []

    def fake_runner(common, _cfg, _ram, _ssd, _s3, _router_cfg):
        calls.append(int(common["users"]))
        return _make_result()

    base = RouterCostConfig()
    tune_router_for_config(
        {"users": 100},
        {},
        0.1,
        0.1,
        S3Spec.from_gbps(False, 0.0, 0.0),
        base,
        grid=[TunableRouterParams(0.001, 1.0)],
        budgets=[8, 2, 8, 2],
        refine=False,
        max_workers=1,
        _runner=fake_runner,
    )

    assert calls == [2, 8]
