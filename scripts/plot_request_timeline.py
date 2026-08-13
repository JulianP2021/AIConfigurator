#!/usr/bin/env python3
r"""Plot a per-user request timeline from the SLA-driven request generator.

The script builds the exact schedule the simulator will see (startup offsets,
think time, optional user delays, and SLA-driven expected service time) and
draws one horizontal bar per request.  Each user gets its own row; colors
represent session ids; bar width represents the scheduled service window.

Usage:
    python scripts/plot_request_timeline.py --seed 42 --users 40 \\
        --sessions-per-user 20 --max-session-turns 7 \\
        --think-time-ms 1000 --isl 30000 --osl 2000 \\
        --ttft-ms 30000 --tpot-ms 100 \\
        --user-delay-fraction 0.1 \\
        --user-delay-min-ms 60000 --user-delay-max-ms 60000 \\
        --output timeline.png
"""

import argparse
import math
import sys

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.request.request import (
    RequestGenerator,
    RequestScenario,
    TokenDistribution,
    set_request_rng,
)
from src.utils.env_reader import load_env


def _parse_args(env) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-user request timeline from the SLA-driven generator."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=env.random_seed,
        help="Random seed for reproducible startup offsets and user delays "
        f"(default: {env.random_seed})",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=env.users,
        help=f"Number of users (default: {env.users})",
    )
    parser.add_argument(
        "--sessions-per-user",
        type=int,
        default=env.sessions_per_user,
        help=f"Sessions per user (default: {env.sessions_per_user})",
    )
    parser.add_argument(
        "--max-session-turns",
        type=int,
        default=env.max_session_turns,
        help=f"Max requests per session (default: {env.max_session_turns})",
    )
    parser.add_argument(
        "--think-time-ms",
        type=float,
        default=env.think_time_ms,
        help=f"Think time between requests in ms (default: {env.think_time_ms})",
    )
    parser.add_argument(
        "--startup-arrival-mean-ms",
        type=float,
        default=env.startup_arrival_mean_ms,
        help=f"Mean startup arrival offset per user in ms (exponential distribution, default: {env.startup_arrival_mean_ms})",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=env.isl,
        help=f"Input sequence length (default: {env.isl})",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=env.osl,
        help=f"Output sequence length (default: {env.osl})",
    )
    parser.add_argument(
        "--ttft-ms",
        type=float,
        default=env.sla_ttft_ms,
        help=f"TTFT SLA in ms; drives the schedule (default: {env.sla_ttft_ms})",
    )
    parser.add_argument(
        "--tpot-ms",
        type=float,
        default=env.sla_tpot_ms,
        help=f"TPOT SLA in ms; drives the schedule (default: {env.sla_tpot_ms})",
    )
    parser.add_argument(
        "--user-delay-fraction",
        type=float,
        default=env.user_delay_fraction,
        help=f"Fraction of requests that get an extra user delay (default: {env.user_delay_fraction})",
    )
    parser.add_argument(
        "--user-delay-min-ms",
        type=float,
        default=env.user_delay_min_ms,
        help=f"Minimum extra user delay in ms (default: {env.user_delay_min_ms})",
    )
    parser.add_argument(
        "--user-delay-max-ms",
        type=float,
        default=env.user_delay_max_ms,
        help=f"Maximum extra user delay in ms (default: {env.user_delay_max_ms})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("timeline.png"),
        help="Output PNG path (default: timeline.png)",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("ttft_ms", "tpot_ms"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number, got {value}")
    if args.users <= 0:
        raise ValueError(f"users must be positive, got {args.users}")
    if args.sessions_per_user <= 0:
        raise ValueError(
            f"sessions-per-user must be positive, got {args.sessions_per_user}"
        )
    if args.max_session_turns <= 0:
        raise ValueError(
            f"max-session-turns must be positive, got {args.max_session_turns}"
        )


def build_schedule(args: argparse.Namespace) -> list[dict]:
    """Generate the full request schedule and return per-request metadata."""
    set_request_rng(args.seed)

    generator = RequestGenerator(
        users=args.users,
        max_session_turns=args.max_session_turns,
        think_time_ms=args.think_time_ms,
        sessions_per_user=args.sessions_per_user,
        delay_fraction=args.user_delay_fraction,
        delay_min_ms=args.user_delay_min_ms,
        delay_max_ms=args.user_delay_max_ms,
        ttft_sla_ms=args.ttft_ms,
        tpot_sla_ms=args.tpot_ms,
        startup_arrival_mean_ms=args.startup_arrival_mean_ms,
    )

    scenario = RequestScenario(
        token_distribution=TokenDistribution(
            min_input_tokens=args.isl,
            max_input_tokens=args.isl,
            min_output_tokens=args.osl,
            max_output_tokens=args.osl,
        ),
        sessions_per_user=args.sessions_per_user,
        users=args.users,
        max_session_turns=args.max_session_turns,
        think_time_ms=args.think_time_ms,
    )

    schedule: list[dict] = []
    now_ms = 0.0
    total_requests = generator.total_requests

    while len(schedule) < total_requests:
        request = generator.generate_request(scenario, now_ms)
        if request is None:
            next_ready = generator.next_ready_time_ms(now_ms)
            if not math.isfinite(next_ready):
                raise RuntimeError(
                    "No more ready users but schedule is incomplete; "
                    "this usually means the generator state is inconsistent."
                )
            now_ms = next_ready
            continue

        expected_service_ms = args.ttft_ms + args.tpot_ms * request.osl
        schedule.append({
            "id": request.id,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "turn": generator._user_session_turns.get(request.user_id, 0),
            "generated_ms": request.generated_ms,
            "expected_service_ms": expected_service_ms,
            "isl": request.isl,
            "osl": request.osl,
        })
        generator.finish_request(request, request.generated_ms + expected_service_ms)

    return schedule


def _auto_time_unit(max_time_s: float) -> tuple[float, str]:
    """Return a divisor and label for the x-axis based on the total span."""
    if max_time_s < 120:
        return round(max_time_s, 3), "s"
    if max_time_s < 7200:
        return round(max_time_s / 60.0, 3), "min"
    return round(max_time_s / 3600, 3), "h"


def plot_schedule(schedule: list[dict], args: argparse.Namespace) -> None:
    """Draw and save the per-user timeline PNG."""
    if not schedule:
        raise ValueError("No requests to plot.")

    users = sorted({entry["user_id"] for entry in schedule})
    max_session_id = max(entry["session_id"] for entry in schedule)

    # Use a qualitative colormap; wrap around for many sessions.
    cmap = plt.get_cmap("tab20")
    colors = {
        session_id: cmap(session_id % cmap.N)
        for session_id in range(max_session_id + 1)
    }

    max_time_s = max(
        (entry["generated_ms"] + entry["expected_service_ms"]) / 1000.0
        for entry in schedule
    )
    divisor, unit = (
        max_time_s / _auto_time_unit(max_time_s)[0],
        _auto_time_unit(max_time_s)[1],
    )

    fig_height = max(6.0, len(users) * 0.35)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    user_index = {user_id: i for i, user_id in enumerate(users)}

    for entry in schedule:
        y = user_index[entry["user_id"]]
        start_s = entry["generated_ms"] / 1000.0 / divisor
        width_s = entry["expected_service_ms"] / 1000.0 / divisor
        color = colors[entry["session_id"]]
        ax.barh(
            y,
            width=width_s,
            left=start_s,
            height=0.6,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_yticks(range(len(users)))
    ax.set_yticklabels([f"user {uid}" for uid in users])
    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("User")

    ttft = _auto_time_unit(args.ttft_ms / 1000)
    tpot = _auto_time_unit(args.tpot_ms / 1000)
    user_delay = _auto_time_unit(args.user_delay_max_ms / 1000)
    think_time = _auto_time_unit(args.think_time_ms / 1000)
    startup_arrival = _auto_time_unit(args.startup_arrival_mean_ms / 1000)

    ax.set_title(
        f"Request schedule: {args.users} users, {args.sessions_per_user} sessions/user, "
        f"{args.max_session_turns} turns, seed={args.seed}, "
        f" ttft: {ttft[0]}{ttft[1]}, tpot: {tpot[0]}{tpot[1]}, user delay: {user_delay[0]}{user_delay[1]} * {args.user_delay_fraction}, thinking time: {think_time[0]}{think_time[1]}, startup arrival: {startup_arrival[0]}{startup_arrival[1]}"
    )

    # Legend for sessions, but keep it compact for many sessions.
    visible_sessions = sorted({entry["session_id"] for entry in schedule})
    handles = [
        mpatches.Patch(color=colors[sid], label=f"session {sid}")
        for sid in visible_sessions
    ]
    ncol = min(10, max(1, len(handles) // 20 + 1))
    ax.legend(
        handles=handles,
        loc="upper right",
        ncol=ncol,
        fontsize="small",
    )

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-0.5, top=len(users) - 0.5)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close(fig)


def main() -> None:
    env = load_env()
    args = _parse_args(env)
    _validate_args(args)
    schedule = build_schedule(args)
    plot_schedule(schedule, args)
    print(
        f"Wrote timeline for {len(schedule)} requests across {args.users} users "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
