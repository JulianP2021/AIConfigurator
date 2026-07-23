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
    ttft_sla_ms: float = 100.0,
    tpot_sla_ms: float = 10.0,
    sessions_per_user: int = 2,
    think_time_ms: float = 10.0,
    startup_arrival_mean_ms: float = 0.0,
) -> RequestGenerator:
    return RequestGenerator(
        users=users,
        max_session_turns=1,
        think_time_ms=think_time_ms,
        sessions_per_user=sessions_per_user,
        delay_fraction=delay_fraction,
        delay_min_ms=delay_min_ms,
        delay_max_ms=delay_max_ms,
        ttft_sla_ms=ttft_sla_ms,
        tpot_sla_ms=tpot_sla_ms,
        startup_arrival_mean_ms=startup_arrival_mean_ms,
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
        # next_ready = finish + think (SLA-based service time no longer added)
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
        # finish 5 + think 10 + fixed delay 50
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
        # base = finish 5 + think 10 = 15, plus delay in [20, 80]
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
            # finish 0 + think 10 + fixed delay 100
            assert ready == pytest.approx(110.0)

    def test_think_time_blocks_second_request(self):
        """A user cannot generate its next request before think_time_ms elapses."""
        set_request_rng(5)
        gen = _make_generator(users=1, sessions_per_user=2, think_time_ms=50.0)
        scenario = RequestScenario(
            token_distribution=TokenDistribution(8, 8, 1, 1),
            sessions_per_user=2,
            users=1,
            max_session_turns=1,
            think_time_ms=50.0,
        )
        # Wait past the random startup offset.
        now_ms = 4_000.0
        req1 = gen.generate_request(scenario, now_ms)
        assert req1 is not None
        assert req1.session_id == 1
        gen.finish_request(req1, now_ms)
        next_ready = gen._next_available_ms[req1.user_id]
        # Immediately try to generate again: should be blocked.
        req2 = gen.generate_request(scenario, now_ms + 10.0)
        assert req2 is None
        # After think_time_ms has passed: should generate session 2.
        req2 = gen.generate_request(scenario, next_ready)
        assert req2 is not None
        assert req2.session_id == 2


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
        gen_a = _make_generator(startup_arrival_mean_ms=10.0)
        offsets_a = [gen_a._next_available_ms[uid] for uid in range(4)]

        set_request_rng(7)
        gen_b = _make_generator(startup_arrival_mean_ms=10.0)
        offsets_b = [gen_b._next_available_ms[uid] for uid in range(4)]

        assert offsets_a != offsets_b

    def test_zero_mean_starts_all_at_zero(self):
        set_request_rng(123)
        gen = _make_generator(startup_arrival_mean_ms=0.0)
        for uid in range(4):
            assert gen._next_available_ms[uid] == 0.0

    def test_positive_mean_gives_non_negative_offsets(self):
        set_request_rng(123)
        gen = _make_generator(startup_arrival_mean_ms=50.0)
        for uid in range(4):
            assert gen._next_available_ms[uid] >= 0.0

    def test_startup_offsets_are_exponential(self):
        """With a positive mean, at least one offset should be non-zero."""
        set_request_rng(1)
        gen = _make_generator(startup_arrival_mean_ms=10.0)
        assert any(gen._next_available_ms[uid] > 0.0 for uid in range(100))

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


class TestSLADrivenSchedule:
    def test_finite_sla_required(self):
        with pytest.raises(
            ValueError, match="ttft_sla_ms must be a finite positive number, got inf"
        ):
            RequestGenerator(
                users=2,
                max_session_turns=1,
                think_time_ms=10.0,
                sessions_per_user=1,
                delay_fraction=0.0,
                delay_min_ms=0.0,
                delay_max_ms=0.0,
                ttft_sla_ms=float("inf"),
                tpot_sla_ms=10.0,
            )
        with pytest.raises(
            ValueError, match="tpot_sla_ms must be a finite positive number, got inf"
        ):
            RequestGenerator(
                users=2,
                max_session_turns=1,
                think_time_ms=10.0,
                sessions_per_user=1,
                delay_fraction=0.0,
                delay_min_ms=0.0,
                delay_max_ms=0.0,
                ttft_sla_ms=100.0,
                tpot_sla_ms=float("inf"),
            )

    def test_next_ready_is_finish_plus_think_time(self):
        set_request_rng(0)
        gen = RequestGenerator(
            users=1,
            max_session_turns=1,
            think_time_ms=10.0,
            sessions_per_user=2,
            delay_fraction=0.0,
            delay_min_ms=0.0,
            delay_max_ms=0.0,
            ttft_sla_ms=50.0,
            tpot_sla_ms=5.0,
        )
        scenario = RequestScenario(
            token_distribution=TokenDistribution(8, 8, 1, 3),
            sessions_per_user=2,
            users=1,
            max_session_turns=1,
            think_time_ms=10.0,
        )
        req = gen.generate_request(scenario, 1_000.0)
        assert req is not None
        gen.finish_request(req, 7.0)
        # SLA-based service time is no longer added; next ready = finish + think.
        assert gen._next_available_ms[req.user_id] == pytest.approx(7.0 + 10.0)

    def test_schedule_is_finish_driven(self):
        set_request_rng(0)
        gen = _make_generator(sessions_per_user=2)
        scenario = RequestScenario(
            token_distribution=TokenDistribution(8, 8, 1, 1),
            sessions_per_user=2,
            users=1,
            max_session_turns=1,
            think_time_ms=10.0,
        )
        req = gen.generate_request(scenario, 1_000.0)
        assert req is not None
        # When the request finishes late, the next ready time is now late + think.
        gen.finish_request(req, 10_000.0)
        assert gen._next_available_ms[req.user_id] == pytest.approx(10_000.0 + 10.0)

    def test_sessions_start_at_one(self):
        set_request_rng(0)
        gen = RequestGenerator(
            users=2,
            max_session_turns=1,
            think_time_ms=10.0,
            sessions_per_user=2,
            delay_fraction=0.0,
            delay_min_ms=0.0,
            delay_max_ms=0.0,
            ttft_sla_ms=50.0,
            tpot_sla_ms=5.0,
        )
        scenario = RequestScenario(
            token_distribution=TokenDistribution(8, 8, 1, 1),
            sessions_per_user=2,
            users=2,
            max_session_turns=1,
            think_time_ms=10.0,
        )
        seen: set[tuple[int, int]] = set()
        now_ms = 1_000.0
        while len(seen) < gen.total_requests:
            req = gen.generate_request(scenario, now_ms)
            if req is None:
                now_ms = gen.next_ready_time_ms(now_ms)
                continue
            seen.add((req.user_id, req.session_id))
            assert req.session_id >= 1, "session ids must start at 1"
            gen.finish_request(req, req.generated_ms)

    def test_sessions_per_user_is_hard_cap(self):
        set_request_rng(0)
        gen = RequestGenerator(
            users=3,
            max_session_turns=2,
            think_time_ms=10.0,
            sessions_per_user=2,
            delay_fraction=0.0,
            delay_min_ms=0.0,
            delay_max_ms=0.0,
            ttft_sla_ms=50.0,
            tpot_sla_ms=5.0,
        )
        scenario = RequestScenario(
            token_distribution=TokenDistribution(8, 8, 1, 1),
            sessions_per_user=2,
            users=3,
            max_session_turns=2,
            think_time_ms=10.0,
        )
        schedule: list[tuple[int, int]] = []
        now_ms = 1_000.0
        while len(schedule) < gen.total_requests:
            req = gen.generate_request(scenario, now_ms)
            if req is None:
                now_ms = gen.next_ready_time_ms(now_ms)
                continue
            schedule.append((req.user_id, req.session_id))
            gen.finish_request(req, req.generated_ms)

        per_user = {}
        for uid, sid in schedule:
            per_user[uid] = max(per_user.get(uid, 0), sid)
        assert all(count == 2 for count in per_user.values())
        assert len(schedule) == 3 * 2 * 2
