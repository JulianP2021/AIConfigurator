"""Tests for per-user random delay and reproducible request generation."""

import pytest

from src.request.request import (
    RequestGenerator,
    RequestScenario,
    TokenDistribution,
    set_request_rng,
)


def _make_generator(
    users: int = 4,
    delay_fraction: float = 0.0,
    delay_min_ms: float = 0.0,
    delay_max_ms: float = 0.0,
) -> RequestGenerator:
    return RequestGenerator(
        users=users,
        max_session_turns=1,
        think_time_ms=10.0,
        sessions_per_user=1,
        delay_fraction=delay_fraction,
        delay_min_ms=delay_min_ms,
        delay_max_ms=delay_max_ms,
    )


class TestUserDelay:
    def test_no_delay_when_fraction_zero(self):
        set_request_rng(1)
        gen = _make_generator()
        scenario = RequestScenario(
            token_distribution=TokenDistribution(8, 8, 1, 1),
            sessions_per_user=1,
            users=4,
            max_session_turns=1,
            think_time_ms=10.0,
        )
        # Force every user to be past its startup offset.
        req = gen.generate_request(scenario, 1_000.0)
        assert req is not None
        gen.finish_request(req, 5.0)
        assert gen._next_available_ms[req.user_id] == pytest.approx(15.0)

    def test_fixed_delay_added_on_top_of_think_time(self):
        set_request_rng(2)
        gen = _make_generator(delay_fraction=1.0, delay_min_ms=50.0, delay_max_ms=50.0)
        scenario = RequestScenario(
            token_distribution=TokenDistribution(8, 8, 1, 1),
            sessions_per_user=1,
            users=4,
            max_session_turns=1,
            think_time_ms=10.0,
        )
        req = gen.generate_request(scenario, 1_000.0)
        assert req is not None
        gen.finish_request(req, 5.0)
        assert gen._next_available_ms[req.user_id] == pytest.approx(65.0)

    def test_random_delay_within_range(self):
        set_request_rng(3)
        gen = _make_generator(delay_fraction=1.0, delay_min_ms=20.0, delay_max_ms=80.0)
        scenario = RequestScenario(
            token_distribution=TokenDistribution(8, 8, 1, 1),
            sessions_per_user=1,
            users=4,
            max_session_turns=1,
            think_time_ms=10.0,
        )
        req = gen.generate_request(scenario, 1_000.0)
        assert req is not None
        gen.finish_request(req, 5.0)
        ready = gen._next_available_ms[req.user_id]
        # 5 + 10 = 15 base, plus delay in [20, 80]
        assert 35.0 <= ready <= 95.0

    def test_fraction_partially_applies(self):
        set_request_rng(4)
        gen = _make_generator(
            delay_fraction=1.0, delay_min_ms=100.0, delay_max_ms=100.0
        )
        scenario = RequestScenario(
            token_distribution=TokenDistribution(8, 8, 1, 1),
            sessions_per_user=1,
            users=4,
            max_session_turns=1,
            think_time_ms=10.0,
        )
        # Finish a request for every user; all should be delayed by exactly 100ms.
        seen_users: set[int] = set()
        while len(seen_users) < 4:
            req = gen.generate_request(scenario, 1_000.0)
            assert req is not None
            seen_users.add(req.user_id)
            gen.finish_request(req, 0.0)
        for ready in gen._next_available_ms.values():
            assert ready == pytest.approx(110.0)


class TestRequestRNGSeed:
    def test_seed_makes_startup_offsets_reproducible(self):
        set_request_rng(42)
        gen_a = _make_generator()
        offsets_a = [gen_a._next_available_ms[uid] for uid in range(4)]

        set_request_rng(42)
        gen_b = _make_generator()
        offsets_b = [gen_b._next_available_ms[uid] for uid in range(4)]

        assert offsets_a == offsets_b

    def test_different_seed_gives_different_offsets(self):
        set_request_rng(42)
        gen_a = _make_generator()
        offsets_a = [gen_a._next_available_ms[uid] for uid in range(4)]

        set_request_rng(7)
        gen_b = _make_generator()
        offsets_b = [gen_b._next_available_ms[uid] for uid in range(4)]

        assert offsets_a != offsets_b

    def test_seed_makes_delays_reproducible(self):
        set_request_rng(123)
        gen_a = _make_generator(
            delay_fraction=1.0, delay_min_ms=20.0, delay_max_ms=80.0
        )
        scenario = RequestScenario(
            token_distribution=TokenDistribution(8, 8, 1, 1),
            sessions_per_user=1,
            users=4,
            max_session_turns=1,
            think_time_ms=10.0,
        )
        req_a = gen_a.generate_request(scenario, 1_000.0)
        assert req_a is not None
        gen_a.finish_request(req_a, 5.0)
        ready_a = gen_a._next_available_ms[req_a.user_id]

        set_request_rng(123)
        gen_b = _make_generator(
            delay_fraction=1.0, delay_min_ms=20.0, delay_max_ms=80.0
        )
        req_b = gen_b.generate_request(scenario, 1_000.0)
        assert req_b is not None
        gen_b.finish_request(req_b, 5.0)
        ready_b = gen_b._next_available_ms[req_b.user_id]

        assert ready_a == pytest.approx(ready_b)
