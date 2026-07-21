"""FastAPI webserver for the Configurator Simulator."""

import asyncio
import base64
import concurrent.futures
import io
import json
import math
import os
import re
import sys
import uuid

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, redirect_stdout, suppress
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, PlainTextResponse, Response


# Ensure project root is on sys.path when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.hardware.hardware import S3Spec
from src.hardware.scraper import (
    fetch_machine_hardware,
    load_combined_machine_db,
    parse_gpu_count,
    resolve_machine_name,
)
from src.logger import set_log_mask
from src.node.node import Node
from src.request.request import RequestScenario, TokenDistribution
from src.result import SimulationResult
from src.router.router import RouterCostConfig
from src.simulations.simulation_distributed import (
    DistributedScenario,
    simulate_run_distributed,
)
from src.utils.env_reader import load_env
from src.utils.utils import add_result_metadata


matplotlib.use("Agg")

# Load .env configuration (including LOG_MASK) at startup.
_env = load_env()
set_log_mask(_env.log_mask)

# In-memory storage for plots (keyed by UUID)
_plot_store: dict[str, str] = {}  # id -> base64 PNG

# Process pool used to isolate CPU-bound simulations. Each simulation has
# module-level mutable state (request IDs, caches, schedulers); running them in
# separate processes keeps results deterministic and prevents cross-run state
# leakage. Concurrency is capped at 8 workers.
_executor: concurrent.futures.ProcessPoolExecutor | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Start the process pool at server startup and shut it down cleanly."""
    global _executor
    _executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)
    try:
        yield
    finally:
        if _executor is not None:
            _executor.shutdown(wait=True, cancel_futures=True)
            _executor = None


app = FastAPI(title="Configurator Simulator", lifespan=lifespan)


def _build_nodes(
    prefill_hardware_name: str,
    decode_hardware_name: str,
    prefill_nodes: int,
    decode_nodes: int,
    prefill_gpus_per_node: int,
    decode_gpus_per_node: int,
    batch_size: int,
    model: str,
    colocated: bool = False,
    inter_node_network_up_gbps: float = 100.0,
    inter_node_network_down_gbps: float = 100.0,
) -> list[Node]:
    # Apply per-request inter-node bandwidth overrides from the webserver form
    # before resolving/loading any machine hardware.
    os.environ["INTER_NODE_NETWORK_UP_GBPS"] = str(inter_node_network_up_gbps)
    os.environ["INTER_NODE_NETWORK_DOWN_GBPS"] = str(inter_node_network_down_gbps)

    prefill_hw_name = resolve_machine_name(prefill_hardware_name)
    decode_hw_name = resolve_machine_name(decode_hardware_name)
    prefill_total_gpus = parse_gpu_count(prefill_hw_name)
    decode_total_gpus = parse_gpu_count(decode_hw_name)
    # Use explicit counts if provided, otherwise infer from the machine key.
    if prefill_gpus_per_node == 0:
        prefill_gpus_per_node = prefill_total_gpus
    if decode_gpus_per_node == 0:
        decode_gpus_per_node = decode_total_gpus

    prefill_hw = fetch_machine_hardware(prefill_hw_name)
    decode_hw = fetch_machine_hardware(decode_hw_name)
    print(
        f"Building nodes for prefill hardware: {prefill_hw}, decode hardware: {decode_hw}, colocated={colocated}"
    )

    nodes: list[Node] = []
    if colocated:
        if prefill_nodes != decode_nodes:
            raise ValueError(
                f"Colocated config requires prefill_nodes ({prefill_nodes}) == decode_nodes ({decode_nodes})."
            )
        if prefill_hw_name != decode_hw_name:
            raise ValueError(
                "Colocated config requires identical prefill and decode hardware."
            )
        if prefill_gpus_per_node + decode_gpus_per_node != prefill_total_gpus:
            raise ValueError(
                f"GPU split {prefill_gpus_per_node}+{decode_gpus_per_node} does not equal "
                f"total GPUs per node ({prefill_total_gpus})."
            )
        for _ in range(prefill_nodes):
            nodes.append(
                Node(
                    hardware=prefill_hw,
                    model_name=model,
                    batch_size=batch_size,
                    prefill_instances=prefill_gpus_per_node,
                    decode_instances=decode_gpus_per_node,
                )
            )
    else:
        for _ in range(prefill_nodes):
            nodes.append(
                Node(
                    hardware=prefill_hw,
                    model_name=model,
                    batch_size=batch_size,
                    prefill_instances=prefill_gpus_per_node,
                    decode_instances=0,
                )
            )
        for _ in range(decode_nodes):
            nodes.append(
                Node(
                    hardware=decode_hw,
                    model_name=model,
                    batch_size=batch_size,
                    prefill_instances=0,
                    decode_instances=decode_gpus_per_node,
                )
            )
    return nodes


def _run_single_config(
    *,
    label: str,
    prefill_hardware: str,
    decode_hardware: str,
    prefill_nodes: int,
    decode_nodes: int,
    prefill_gpus_per_node: int,
    decode_gpus_per_node: int,
    batch_size: int,
    model: str,
    isl: int,
    osl: int,
    sessions_per_user: int,
    users: int,
    think_time_ms: float,
    max_session_turns: int,
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_enabled: bool,
    s3_up_bw_gbps: float,
    s3_down_bw_gbps: float,
    s3_eviction_time_ms: float,
    inter_node_network_up_gbps: float,
    inter_node_network_down_gbps: float,
    router_prefill_load_scale: float,
    router_device_credit: float,
    router_remote_ram_credit: float,
    router_remote_ssd_credit: float,
    router_s3_credit: float,
    router_busy_threshold_tokens: float,
    user_delay_fraction: float = 0.0,
    user_delay_min_ms: float = 0.0,
    user_delay_max_ms: float = 0.0,
    random_seed: int | None = None,
    colocated: bool = False,
    ttft_sla_ms: float = 30000.0,
    tpot_sla_ms: float = 100.0,
) -> SimulationResult | None:
    nodes = _build_nodes(
        prefill_hardware,
        decode_hardware,
        prefill_nodes,
        decode_nodes,
        prefill_gpus_per_node,
        decode_gpus_per_node,
        batch_size,
        model,
        colocated,
        inter_node_network_up_gbps=inter_node_network_up_gbps,
        inter_node_network_down_gbps=inter_node_network_down_gbps,
    )
    scenario = DistributedScenario(
        name=label,
        nodes=nodes,
        requests=RequestScenario(
            token_distribution=TokenDistribution(
                min_input_tokens=isl,
                max_input_tokens=isl,
                min_output_tokens=osl,
                max_output_tokens=osl,
            ),
            sessions_per_user=sessions_per_user,
            users=users,
            max_session_turns=max_session_turns,
            think_time_ms=think_time_ms,
        ),
    )
    print(f"Simulating scenario: {scenario}")

    for key, value in (("ttft_sla_ms", ttft_sla_ms), ("tpot_sla_ms", tpot_sla_ms)):
        if not math.isfinite(value) or value <= 0:
            print(f"Error: {key} must be a finite positive number, got {value}")
            return None

    s3_spec = S3Spec.from_gbps(
        enabled=s3_enabled,
        up_gbps=s3_up_bw_gbps,
        down_gbps=s3_down_bw_gbps,
        eviction_time_ms=s3_eviction_time_ms,
    )
    router_cost_config = RouterCostConfig(
        prefill_load_scale=router_prefill_load_scale,
        device_credit=router_device_credit,
        remote_ram_credit=router_remote_ram_credit,
        remote_ssd_credit=router_remote_ssd_credit,
        s3_credit=router_s3_credit,
        busy_threshold_tokens=router_busy_threshold_tokens,
        active_work_scale=0.001,
        # @TODO add as input
    )
    with io.StringIO() as buf, redirect_stdout(buf):
        try:
            print(f"Simulating scenario: {scenario}")

            return simulate_run_distributed(
                scenario,
                ram_usage_fraction=ram_usage_fraction,
                ssd_usage_fraction=ssd_usage_fraction,
                s3_spec=s3_spec,
                router_cost_config=router_cost_config,
                sla={"ttft_ms": ttft_sla_ms, "tpot_ms": tpot_sla_ms},
                user_delay_fraction=user_delay_fraction,
                user_delay_min_ms=user_delay_min_ms,
                user_delay_max_ms=user_delay_max_ms,
                random_seed=random_seed,
            )
        except Exception as exc:
            print(f"Error during simulation for config '{label}': {exc}")
            return None


# Top-level function passed to the process pool.  Pickling a bound method is
# unreliable across processes; a plain module-level function is safest.
def _run_single_config_in_process(kwargs: dict[str, object]) -> SimulationResult | None:
    return _run_single_config(**kwargs)


def _metric_card(value: str, label: str) -> str:
    return f"""<div class="metric"><div class="value">{value}</div><div class="label">{label}</div></div>"""


def _build_results_page(
    results: list[dict[str, float | int | str]],
    plot_urls: list[str] | None = None,
    error: str | None = None,
    show_debug_tables: bool = False,
    plot_title: str = "Cost-Latency Plots",
) -> str:
    """Build the results HTML page (for /simulate full page) or just the inner content."""
    inner = _results_inner_html(
        results, plot_urls, error, show_debug_tables, plot_title
    )
    return (
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Configurator Simulator - Results</title>
    <style>
        :root {
            --bg: #0d1117;
            --card: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --accent-hover: #79b8ff;
            --success: #3fb950;
            --danger: #f85149;
            --warning: #d29922;
            --purple: #a371f7;
            --orange: #f0883e;
            --cyan: #39c5cf;
            --pink: #db61a2;
        }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 2rem;
            line-height: 1.5;
        }
        h1 { text-align: center; margin-bottom: 0.25rem; font-weight: 600; }
        .subtitle { text-align: center; color: var(--text-secondary); margin-bottom: 2rem; font-size: 0.95rem; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            overflow-x: auto;
        }
        .card h2 {
            margin-top: 0;
            font-size: 1.1rem;
            font-weight: 600;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }
        .metric {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .metric .value { font-size: 1.3rem; font-weight: 600; color: var(--accent); }
        .metric .label { font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem; }
        .plot-img { width: 100%; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 1rem; }
        .btn {
            display: inline-block;
            padding: 0.65rem 1.5rem;
            background: var(--accent);
            color: #fff;
            border: none;
            border-radius: 6px;
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            text-decoration: none;
            transition: background 0.15s;
        }
        .btn:hover { background: var(--accent-hover); }
        .error {
            color: var(--danger);
            padding: 1rem;
            background: rgba(248, 81, 73, 0.1);
            border: 1px solid var(--danger);
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        th, td { padding: 0.5rem 0.75rem; border: 1px solid var(--border); text-align: left; }
        th { background: var(--bg); color: var(--text-secondary); font-weight: 600; }
        td { color: var(--text); }
        .legend-color {
            display: inline-block;
            width: 10px; height: 10px;
            border-radius: 2px;
            margin-right: 0.4rem;
        }
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 1rem;
        }
        .plot-card { padding: 0; }
        .debug-toggle {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        .debug-toggle input { width: auto; }
        .debug-section { display: none; }
        .debug-section.show { display: block; }
        .user-list { margin-bottom: 1rem; }
        .user-list h3 { margin: 0 0 0.5rem; font-size: 1rem; color: var(--text-secondary); }
        .user-list ul { margin: 0; padding-left: 1.25rem; color: var(--text); }
        .user-list li { margin-bottom: 0.25rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Configurator Simulator</h1>
        <p class="subtitle">Comparison Results</p>
"""
        + inner
        + """
    </div>
    <script>
        function toggleDebugTables() {
            const sections = document.querySelectorAll('.debug-section');
            const checked = document.getElementById('debugToggle').checked;
            sections.forEach(s => s.classList.toggle('show', checked));
        }
        toggleDebugTables();
    </script>
</body>
</html>"""
    )


def _results_inner_html(
    results: list[dict[str, float | int | str]],
    plot_urls: list[str] | None = None,
    error: str | None = None,
    _show_debug_tables: bool = False,
    plot_title: str = "Cost-Latency Plots",
) -> str:
    """Inner HTML for results (used when injecting via JS)."""
    html = ""
    if error:
        html += f'<div class="error">{error}</div>\n'

    if results:
        # ---- Best config metric card ----
        valid_results = [r for r in results if not r.get("has_error")]
        best = min(valid_results or results, key=lambda r: r["request_latency"])
        html += (
            '<div class="card"><h2>Best Config (by Latency)</h2><div class="metrics">\n'
        )
        html += _metric_card(best["label"], "Configuration")
        html += _metric_card(f"{best['ttft']:.2f}", "TTFT (ms)")
        html += _metric_card(f"{best['tpot']:.2f}", "TPOT (ms)")
        html += _metric_card(
            f"${best['total_cost_usd_per_hour']:.2f}", "Total cost / hour"
        )
        html += _metric_card(f"${best['s3_cost_usd_per_hour']:.2f}", "S3 cost / hour")
        html += _metric_card(f"{best['max_request_latency']:.2f}", "max Latency (ms)")
        html += "</div></div>\n"

        # ---- Users-based ordered legend (only when results have users) ----
        if any("users" in row for row in results):
            html += '<div class="card"><h2>Configurations by Users</h2>'
            html += '<div class="user-list"><h3>Mode colors</h3><ul>'
            for mode, color in MODE_COLORS.items():
                html += (
                    f'<li><span class="legend-color" style="background:{color}"></span>'
                    f"{mode.capitalize()}"
                    f"</li>"
                )
            html += "</ul></div>"
            by_users: dict[int, list[dict[str, Any]]] = {}
            for row in results:
                users = int(row.get("users", 0))
                if users > 0:
                    by_users.setdefault(users, []).append(row)
            for users in sorted(by_users):
                html += f'<div class="user-list"><h3>Users = {users}</h3><ul>'
                # Show the top-5 cheapest configs *per mode* for this user count.
                grouped_by_mode: dict[str, list[dict[str, Any]]] = {}
                for row in by_users[users]:
                    grouped_by_mode.setdefault(_mode_for_row(row), []).append(row)
                displayed = 0
                for mode in MODE_COLORS:
                    mode_rows = sorted(
                        grouped_by_mode.get(mode, []),
                        key=lambda r: r.get("total_cost_usd_per_hour", float("inf")),
                    )
                    for row in mode_rows[:5]:
                        color = row.get("color", MODE_COLORS.get(mode, "#58a6ff"))
                        html += (
                            f'<li><span class="legend-color" style="background:{color}"></span>'
                            f"[{mode.capitalize()}] {row['label']} — ${row['total_cost_usd_per_hour']:.2f}/h"
                            f"</li>"
                        )
                        displayed += 1
                html += f"</ul><p style='color:var(--text-secondary);font-size:0.8rem;margin:0.25rem 0 0;'>Showing top 5 per mode ({displayed} total)</p></div>"
            html += "</div>\n"

        # ---- Debug toggle ----
        html += (
            '<div class="card"><h2>Debug Details</h2>'
            '<label class="debug-toggle">'
            '<input type="checkbox" id="debugToggle" onchange="toggleDebugTables()">'
            "Show comparison and timing breakdown tables"
            "</label></div>\n"
        )

        # ---- Comparison table (hidden by default) ----
        html += '<div class="card debug-section"><h2>Configuration Comparison</h2><table><thead><tr>'
        html += "<th>Label</th><th>Prefill HW</th><th>Decode HW</th><th>Nodes (P/D)</th><th>Batch</th>"
        html += "<th>TTFT</th><th>max TTFT</th><th>TPOT</th><th>max TPOT</th>"
        html += (
            "<th>Latency</th><th>max Latency</th><th>KV Upload</th><th>KV Download</th>"
        )
        html += "<th>Compute $/h</th><th>S3 $/h</th><th>Total $/h</th>"
        html += "</tr></thead><tbody>"
        for row in results:
            if row.get("has_error"):
                html += (
                    f'<tr style="opacity:0.7;">'
                    f'<td><span class="legend-color" style="background:{row.get("color", "#58a6ff")}"></span>{row["label"]} <span style="color:var(--danger);font-size:0.75rem;">(failed)</span></td>'
                    f'<td colspan="15" style="text-align:center;color:var(--danger);">Simulation failed — see error banner above</td>'
                    f"</tr>"
                )
                continue
            html += (
                f"<tr>"
                f'<td><span class="legend-color" style="background:{row.get("color", "#58a6ff")}"></span>{row["label"]}</td>'
                f"<td>{row['prefill_hardware']}</td>"
                f"<td>{row['decode_hardware']}</td>"
                f"<td>{row['num_prefill_workers']} / {row['num_decode_workers']}</td>"
                f"<td>{row['batch_size']}</td>"
                f"<td>{row['ttft']:.2f}</td>"
                f"<td>{row['max_ttft']:.2f}</td>"
                f"<td>{row['tpot']:.2f}</td>"
                f"<td>{row['max_tpot']:.2f}</td>"
                f"<td>{row['request_latency']:.2f}</td>"
                f"<td>{row['max_request_latency']:.2f}</td>"
                f"<td>{row['kv_upload_time']:.2f}</td>"
                f"<td>{row['kv_download_time']:.2f}</td>"
                f"<td>${row['compute_price_usd_per_hour']:.2f}</td>"
                f"<td>${row['s3_cost_usd_per_hour']:.2f}</td>"
                f"<td>${row['total_cost_usd_per_hour']:.2f}</td>"
                f"</tr>"
            )
        html += "</tbody></table></div>\n"

        # ---- Timing breakdown table (hidden by default) ----
        html += '<div class="card debug-section"><h2>Timing Breakdown</h2><table><thead><tr>'
        html += "<th>Label</th><th>Prefill active</th><th>Prefill wait</th>"
        html += "<th>Prefill dl active</th><th>Prefill dl wait</th><th>Prefill up active</th><th>Prefill up wait</th>"
        html += "<th>Decode dl active</th><th>Decode dl wait</th><th>Decode active</th><th>Decode wait</th><th>Decode up active</th><th>Decode up wait</th>"
        html += "<th>Clean TTFT</th><th>Wait TTFT</th><th>Clean Latency</th><th>Wait Latency</th>"
        html += "</tr></thead><tbody>"
        for row in results:
            if row.get("has_error"):
                html += (
                    f'<tr style="opacity:0.7;">'
                    f'<td><span class="legend-color" style="background:{row.get("color", "#58a6ff")}"></span>{row["label"]} <span style="color:var(--danger);font-size:0.75rem;">(failed)</span></td>'
                    f'<td colspan="17" style="text-align:center;color:var(--danger);">Simulation failed</td>'
                    f"</tr>"
                )
                continue
            html += (
                f"<tr>"
                f'<td><span class="legend-color" style="background:{row.get("color", "#58a6ff")}"></span>{row["label"]}</td>'
                f"<td>{row['avg_prefill_time_ms']:.2f}</td>"
                f"<td>{row['avg_prefill_wait_ms']:.2f}</td>"
                f"<td>{row['avg_prefill_download_active_ms']:.2f}</td>"
                f"<td>{row['avg_prefill_download_wait_ms']:.2f}</td>"
                f"<td>{row['avg_prefill_upload_active_ms']:.2f}</td>"
                f"<td>{row['avg_prefill_upload_wait_ms']:.2f}</td>"
                f"<td>{row['avg_decode_download_active_ms']:.2f}</td>"
                f"<td>{row['avg_decode_download_wait_ms']:.2f}</td>"
                f"<td>{row['avg_decode_time_ms']:.2f}</td>"
                f"<td>{row['avg_decode_wait_ms']:.2f}</td>"
                f"<td>{row['avg_decode_upload_active_ms']:.2f}</td>"
                f"<td>{row['avg_decode_upload_wait_ms']:.2f}</td>"
                f"<td>{row['avg_clean_ttft_ms']:.2f}</td>"
                f"<td>{row['ttft']:.2f}</td>"
                f"<td>{row['avg_clean_latency_ms']:.2f}</td>"
                f"<td>{row['request_latency']:.2f}</td>"
                f"</tr>"
            )
        html += "</tbody></table></div>\n"

    if plot_urls:
        html += (
            f'<div class="card plot-card"><h2>{plot_title}</h2><div class="plot-grid">'
        )
        for url in plot_urls:
            html += (
                f'<div><img src="{url}" class="plot-img" alt="Cost-Latency Plot"></div>'
            )
        html += "</div></div>\n"

    return html


def _build_single_plot(
    results: list[dict[str, float | int | str]],
    x_key: str,
    x_label: str,
    title: str,
) -> str:
    """Generate a single scatter plot and return its plot ID."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in results:
        ax.scatter(
            row[x_key],
            row["total_cost_usd_per_hour"],
            s=120,
            color=row.get("color", "#58a6ff"),
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax.annotate(
            row["label"],
            (row[x_key], row["total_cost_usd_per_hour"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
            color=row.get("color", "#58a6ff"),
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Total cost ($/hour)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    pid = str(uuid.uuid4())
    _plot_store[pid] = base64.b64encode(buf.read()).decode("utf-8")
    return f"/plot/{pid}"


def _extract_user_delay_ms(row: dict[str, Any]) -> float | None:
    for key in ("user_delay_ms", "user_delay_min_ms", "user_delay_max_ms"):
        value = row.get(key)
        if value is not None:
            return float(value)
    label = str(row.get("label", ""))
    match = re.search(r"delay=([0-9.+-eE]+)ms", label)
    if match:
        return float(match.group(1))
    return None


def _build_ttft_cost_plots_by_delay(
    results: list[dict[str, float | int | str]],
) -> list[str]:
    valid_rows = [row for row in results if not row.get("has_error")]
    color_map: dict[tuple[str, str], str] = {}
    for row in valid_rows:
        key = row["focus"]
        color_map[key] = row["color"]

    by_delay: dict[float, list[dict[str, Any]]] = {}
    for row in valid_rows:
        delay_ms = _extract_user_delay_ms(row)
        if delay_ms is None:
            continue
        by_delay.setdefault(round(delay_ms, 6), []).append(row)

    plot_urls: list[str] = []
    for delay_ms in sorted(by_delay):
        rows = sorted(
            by_delay[delay_ms],
            key=lambda r: (
                r.get("ttft", float("inf")),
                r.get("total_cost_usd_per_hour", float("inf")),
            ),
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        for row in rows:
            ax.scatter(
                row["kv_download_time"],
                row["total_cost_usd_per_hour"],
                s=120,
                color=row["color"],
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
            ax.annotate(
                "D" + (row["focus_value"] if row["focus_value"] else ""),
                (row["kv_download_time"], row["total_cost_usd_per_hour"]),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=9,
                color=row["color"],
            )

            ax.scatter(
                row["kv_upload_time"],
                row["total_cost_usd_per_hour"],
                s=120,
                color=row["color"],
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
            ax.annotate(
                "U" + (row["focus_value"] if row["focus_value"] else ""),
                (row["kv_upload_time"], row["total_cost_usd_per_hour"]),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=9,
                color=row["color"],
            )

        ax.set_xlabel("UP/DOWNLOAD (ms)")
        ax.set_ylabel("Total cost ($/hour)")
        ax.set_title(f"UP/DOWNLOAD vs Cost (user delay {delay_ms / 1000 / 60:g} min)")
        # ax.set_ylim(bottom=0)
        # ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        pid = str(uuid.uuid4())
        _plot_store[pid] = base64.b64encode(buf.read()).decode("utf-8")
        plot_urls.append(f"/plot/{pid}")

    return plot_urls


def _load_results_from_dir(results_dir: Path) -> list[dict[str, Any]]:
    """Load every results_*.json file from a directory and tag rows with users."""
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("results_*.json")):
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        for row in data.get("results", []):
            if "users" not in row:
                # Infer users from filename like results_users_100.json
                stem = path.stem.replace("results_users_", "")
                with suppress(ValueError):
                    row["users"] = int(stem)
            rows.append(row)
    return rows


def _resolve_results_dir(results_dir: str) -> Path:
    """Resolve a results directory relative to the server or project root."""
    path = Path(results_dir)
    if path.is_dir():
        return path
    # If the server was started from a different working directory, fall back
    # to resolving the path relative to the project root.
    project_root = Path(__file__).resolve().parents[2]
    alt = project_root / path
    if alt.is_dir():
        return alt
    return path


def _build_users_cost_plot(
    rows: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Generate a users-vs-cost line/scatter plot.

    The plot now shows *all* valid rows, colored by mode (separate, colocated,
    mixed).  Labels are only shown for the 10 cheapest configurations per user
    count, but every point is rendered.  A separate line connects the cheapest
    configuration of each mode across user counts.

    Returns the plot URL and the full set of valid rows, with each row's color
    set to the mode color.
    """
    valid_rows = [row for row in rows if not row.get("has_error")]

    # Group all valid rows by user count for plotting and top-10 selection.
    by_users: dict[int, list[dict[str, Any]]] = {}
    for row in valid_rows:
        users = int(row.get("users", 0))
        if users > 0:
            by_users.setdefault(users, []).append(row)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Draw all points per user count, colored by mode.
    for users in sorted(by_users):
        for row in by_users[users]:
            color = _color_for_row(row)
            row["color"] = color
            ax.scatter(
                row["users"],
                row["total_cost_usd_per_hour"],
                s=90,
                color=color,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )

    # Draw best-per-mode lines: for each mode and each user count, pick the
    # cheapest valid row and connect them in ascending user-count order.
    best_per_mode: dict[str, dict[int, dict[str, Any]]] = {
        mode: {} for mode in MODE_COLORS
    }
    for row in valid_rows:
        users = int(row.get("users", 0))
        if users <= 0:
            continue
        mode = _mode_for_row(row)
        cost = row.get("total_cost_usd_per_hour", float("inf"))
        current_best = best_per_mode[mode].get(users)
        if current_best is None or cost < current_best.get(
            "total_cost_usd_per_hour", float("inf")
        ):
            best_per_mode[mode][users] = row

    for mode, best_by_users in best_per_mode.items():
        if not best_by_users:
            continue
        series = sorted(best_by_users.items(), key=lambda item: item[0])
        xs = [u for u, _ in series]
        ys = [r["total_cost_usd_per_hour"] for _, r in series]
        color = MODE_COLORS[mode]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.5,
            linestyle="--",
            marker="s",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.5,
            alpha=0.85,
            zorder=2,
            label=f"Best {mode}",
        )

    ax.set_xlabel("Users")
    ax.set_ylabel("Total cost ($/hour)")
    ax.set_title("Cost vs Users (all configs, top-10 labeled, best-per-mode line)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, which="both", linestyle="--")
    ax.legend(loc="upper left")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    pid = str(uuid.uuid4())
    _plot_store[pid] = base64.b64encode(buf.read()).decode("utf-8")
    return f"/plot/{pid}", valid_rows


async def _run_configs_parallel(
    config_kwargs: list[dict[str, object]],
) -> list[SimulationResult | BaseException | None]:
    """Run all configs concurrently in separate processes, capped at 8 workers.

    Each config gets its own process so module-level simulator state is fully
    isolated. Exceptions are returned so the caller can handle per-config
    failures gracefully.
    """
    assert _executor is not None, "Process pool is not initialized"
    loop = asyncio.get_running_loop()
    futures = [
        loop.run_in_executor(_executor, _run_single_config_in_process, kwargs)
        for kwargs in config_kwargs
    ]
    return await asyncio.gather(*futures, return_exceptions=True)


def _build_comparison_plots(results: list[dict[str, float | int | str]]) -> list[str]:
    """Generate base64-encoded PNGs for multi-config comparison.
    Returns list of plot IDs (8 plots: avg and max for TTFT, TPOT, Latency, KV Upload Time, KV Download Time).
    """
    return [
        # Average values
        _build_single_plot(results, "ttft", "TTFT (ms)", "Price vs TTFT"),
        _build_single_plot(results, "max_ttft", "Max TTFT (ms)", "Price vs Max TTFT"),
        _build_single_plot(results, "tpot", "TPOT (ms)", "Price vs TPOT"),
        _build_single_plot(results, "max_tpot", "Max TPOT (ms)", "Price vs Max TPOT"),
        _build_single_plot(
            results, "request_latency", "Latency (ms)", "Price vs Latency"
        ),
        _build_single_plot(
            results, "max_request_latency", "Max Latency (ms)", "Price vs Max Latency"
        ),
        _build_single_plot(
            results, "kv_upload_time", "KV Upload Time (ms)", "Price vs KV Upload Time"
        ),
        _build_single_plot(
            results,
            "kv_download_time",
            "KV Download Time (ms)",
            "Price vs KV Download Time",
        ),
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

COLORS = [
    "#58a6ff",
    "#3fb950",
    "#f85149",
    "#d29922",
    "#a371f7",
    "#79c0ff",
    "#56d364",
    "#f0883e",
    "#db61a2",
    "#39c5cf",
]


# Per-mode colors used for imported directory results.
MODE_COLORS = {
    "separate": "#58a6ff",  # blue
    "colocated": "#3fb950",  # green
    "mixed": "#f0883e",  # orange
}


def _mode_for_row(row: dict[str, Any]) -> str:
    """Return the deployment mode key for a result row.

    The mode is determined primarily from the row's label, because exported
    result JSON often has stringly-typed booleans or omits the ``mixed`` field
    entirely.  Labels generated by ``create_config.py`` start with
    ``Colocated:``, ``Mixed:`` or ``separate:``.

    As a secondary fallback we still check the explicit ``mixed`` and
    ``colocated`` fields (coercing strings to booleans) and the mixed-GPU
    hardware-name marker `` + ``.
    """
    label = str(row.get("label", "")).strip().lower()
    if label.startswith("colocated"):
        return "colocated"
    if label.startswith("mixed"):
        return "mixed"
    if label.startswith("separate"):
        return "separate"

    def _truthy(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"true", "1", "yes", "on", "t", "y"}

    if _truthy(row.get("mixed")):
        return "mixed"

    prefill_hw = str(row.get("prefill_hardware", ""))
    if " + " in prefill_hw:
        return "mixed"

    if _truthy(row.get("colocated")):
        return "colocated"

    return "separate"


def _color_for_row(row: dict[str, Any]) -> str:
    """Return the color for a result row based on its mode."""
    return MODE_COLORS.get(_mode_for_row(row), "#58a6ff")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    with Path(html_path).open(encoding="utf-8") as fh:
        return HTMLResponse(content=fh.read())


@app.post("/simulate", response_class=HTMLResponse)
async def simulate(
    model: str = Form(...),
    isl: int = Form(...),
    osl: int = Form(...),
    sessions_per_user: int = Form(...),
    users: int = Form(_env.users),
    think_time_ms: float = Form(_env.think_time_ms),
    max_session_turns: int = Form(_env.max_session_turns),
    cfg_prefill_hardware: list[str] = Form(...),
    cfg_decode_hardware: list[str] = Form(...),
    cfg_prefill_nodes: list[str] = Form(...),
    cfg_decode_nodes: list[str] = Form(...),
    cfg_prefill_gpus: list[str] = Form(...),
    cfg_decode_gpus: list[str] = Form(...),
    cfg_colocated: list[str] = Form(...),
    cfg_batch: list[str] = Form(...),
    cfg_label: list[str] = Form(...),
    ram_usage_fraction: float = Form(_env.ram_usage_fraction),
    ssd_usage_fraction: float = Form(_env.ssd_usage_fraction),
    s3_enabled: str = Form("true" if _env.s3_enabled else "false"),
    s3_up_bw_gbps: float = Form(_env.s3_up_bw_gbps),
    s3_down_bw_gbps: float = Form(_env.s3_down_bw_gbps),
    s3_eviction_time_ms: float = Form(_env.s3_eviction_time_ms),
    inter_node_network_up_gbps: float = Form(_env.inter_node_network_up_gbps),
    inter_node_network_down_gbps: float = Form(_env.inter_node_network_down_gbps),
    router_prefill_load_scale: float = Form(_env.router_prefill_load_scale),
    router_device_credit: float = Form(_env.router_device_credit),
    router_remote_ram_credit: float = Form(_env.router_remote_ram_credit),
    router_remote_ssd_credit: float = Form(_env.router_remote_ssd_credit),
    router_s3_credit: float = Form(_env.router_s3_credit),
    router_busy_threshold_tokens: float = Form(_env.router_busy_threshold_tokens),
    user_delay_fraction: float = Form(_env.user_delay_fraction),
    user_delay_min_ms: float = Form(_env.user_delay_min_ms),
    user_delay_max_ms: float = Form(_env.user_delay_max_ms),
    random_seed: int | None = Form(_env.random_seed),
    xhr: str = Form("0"),
):
    try:
        # Gather configs
        n = len(cfg_prefill_hardware)
        if not (
            len(cfg_decode_hardware)
            == len(cfg_prefill_nodes)
            == len(cfg_decode_nodes)
            == len(cfg_prefill_gpus)
            == len(cfg_decode_gpus)
            == len(cfg_batch)
            == len(cfg_label)
            == n
        ):
            raise ValueError("Configuration arrays must all have the same length.")

        # Unchecked checkboxes are not submitted, so normalize colocated to a
        # boolean per config using a set of submitted indices.
        colocated_set = {
            i
            for i, v in enumerate(cfg_colocated)
            if v.strip().lower() in {"true", "1", "yes", "on"}
        }

        s3_on = s3_enabled.strip().lower() in {"true", "1", "yes", "on"}
        common = {
            "model": model,
            "isl": isl,
            "osl": osl,
            "sessions_per_user": sessions_per_user,
            "users": users,
            "think_time_ms": think_time_ms,
            "max_session_turns": max_session_turns,
            "ram_usage_fraction": ram_usage_fraction,
            "ssd_usage_fraction": ssd_usage_fraction,
            "s3_enabled": s3_on,
            "s3_up_bw_gbps": s3_up_bw_gbps,
            "s3_down_bw_gbps": s3_down_bw_gbps,
            "s3_eviction_time_ms": s3_eviction_time_ms,
            "inter_node_network_up_gbps": inter_node_network_up_gbps,
            "inter_node_network_down_gbps": inter_node_network_down_gbps,
            "router_prefill_load_scale": router_prefill_load_scale,
            "router_device_credit": router_device_credit,
            "router_remote_ram_credit": router_remote_ram_credit,
            "router_remote_ssd_credit": router_remote_ssd_credit,
            "router_s3_credit": router_s3_credit,
            "router_busy_threshold_tokens": router_busy_threshold_tokens,
            "user_delay_fraction": user_delay_fraction,
            "user_delay_min_ms": user_delay_min_ms,
            "user_delay_max_ms": user_delay_max_ms,
            "random_seed": random_seed,
        }

        config_kwargs: list[dict[str, object]] = []
        for i in range(n):
            prefill_hw = cfg_prefill_hardware[i]
            decode_hw = cfg_decode_hardware[i]
            prefill_n = int(cfg_prefill_nodes[i])
            colocated = i in colocated_set
            # In colocated mode decode_nodes mirrors prefill_nodes.
            decode_n = int(cfg_decode_nodes[i]) if cfg_decode_nodes[i] else prefill_n
            prefill_gpus = (
                int(cfg_prefill_gpus[i])
                if cfg_prefill_gpus[i]
                else parse_gpu_count(prefill_hw)
            )
            decode_gpus = (
                int(cfg_decode_gpus[i])
                if cfg_decode_gpus[i]
                else parse_gpu_count(decode_hw)
            )
            batch = int(cfg_batch[i])
            label = cfg_label[i] or f"Config {i + 1}"

            if prefill_n == 0 and decode_n == 0:
                raise ValueError(
                    f"Config '{label}' must have at least one prefill or decode node."
                )

            kwargs: dict[str, object] = {
                "label": label,
                "prefill_hardware": prefill_hw,
                "decode_hardware": decode_hw,
                "prefill_nodes": prefill_n,
                "decode_nodes": decode_n,
                "prefill_gpus_per_node": prefill_gpus,
                "decode_gpus_per_node": decode_gpus,
                "batch_size": batch,
                "colocated": colocated,
                **common,
            }
            config_kwargs.append((label, kwargs, i))

        results_raw = await _run_configs_parallel([
            kwargs for _, kwargs, _ in config_kwargs
        ])

        results_data: list[dict[str, float | int | str]] = []
        skipped_labels: list[str] = []
        error_labels: list[str] = []
        for (label, kwargs, i), result in zip(config_kwargs, results_raw, strict=False):
            if isinstance(
                result, AssertionError
            ) and "Too many requests in prefill queue" in str(result):
                skipped_labels.append(label)
                print(f"Skipping config '{label}' due to prefill queue overflow.")
                continue
            if isinstance(result, BaseException):
                error_labels.append(label)
                print(f"Config '{label}' failed: {result}")
                continue

            assert isinstance(result, SimulationResult)
            row = result.to_dict()
            add_result_metadata(row, label, kwargs, COLORS[i % len(COLORS)])
            results_data.append(row)

        error_msg = None
        errors: list[str] = []
        if skipped_labels:
            errors.append(
                f"Skipped (too many requests in prefill queue): {', '.join(skipped_labels)}"
            )
        if error_labels:
            errors.append(f"Failed: {', '.join(error_labels)}")
        if errors:
            error_msg = "; ".join(errors)

        for label in error_labels:
            results_data.append({
                "label": label,
                "prefill_hardware": "",
                "decode_hardware": "",
                "prefill_nodes": 0,
                "decode_nodes": 0,
                "prefill_gpus_per_node": 0,
                "decode_gpus_per_node": 0,
                "batch_size": 0,
                "colocated": False,
                "ttft": 0.0,
                "kv_upload_time": 0.0,
                "kv_download_time": 0.0,
                "max_ttft": 0.0,
                "tpot": 0.0,
                "max_tpot": 0.0,
                "request_latency": float("inf"),
                "max_request_latency": float("inf"),
                "tokens_per_second": 0.0,
                "tokens_per_second_per_gpu": 0.0,
                "request_rate": 0.0,
                "compute_price_usd_per_hour": 0.0,
                "s3_cost_usd_per_hour": 0.0,
                "s3_storage_cost_usd_per_hour": 0.0,
                "total_cost_usd_per_hour": 0.0,
                "has_error": True,
            })

        plot_urls = _build_comparison_plots(results_data)
        if xhr == "1":
            return HTMLResponse(
                content=_results_inner_html(results_data, plot_urls, error_msg)
            )
        return HTMLResponse(
            content=_build_results_page(results_data, plot_urls, error_msg)
        )

    except Exception as exc:
        import traceback

        err = f"{exc}\n{traceback.format_exc()}"
        if xhr == "1":
            return HTMLResponse(content=_results_inner_html([], None, err))
        return HTMLResponse(content=_build_results_page([], None, err))


@app.get("/import", response_class=HTMLResponse)
async def import_page():
    """Serve the import page where users can paste JSON or select a directory."""
    html_path = Path(__file__).parent / "templates" / "import.html"
    with Path(html_path).open(encoding="utf-8") as fh:
        return HTMLResponse(content=fh.read())


@app.post("/import_results", response_class=HTMLResponse)
async def import_results(
    results_json: str = Form(""),
    results_dir: str = Form(""),
    plot_mode: str = Form("comparison"),
):
    """Render a results page from pasted JSON or from a results directory.

    Accepts either a raw JSON body (legacy / JS fetch) or form fields from the
    import page: ``results_json`` or ``results_dir``.
    """
    try:
        results: list[dict[str, float | int | str]] = []

        from_directory = False
        if results_dir:
            from_directory = True
            path = _resolve_results_dir(results_dir)
            if not path.is_dir():
                raise ValueError(f"Not a directory: {results_dir}")
            rows = _load_results_from_dir(path)
            if not rows:
                raise ValueError(
                    f"Directory '{results_dir}' has results_*.json files, but none contain valid result rows."
                )
            results.extend(rows)
        elif results_json:
            payload = json.loads(results_json)
            if isinstance(payload, list):
                results = payload
            elif isinstance(payload, dict):
                results = payload.get("results", [])
                if not isinstance(results, list):
                    raise ValueError("Payload 'results' must be a list")
            else:
                raise ValueError("Payload must be a list or {'results': [...]}")
        else:
            raise ValueError("Provide either results_json or results_dir")

        if not isinstance(results, list) or not results:
            raise ValueError("No result rows found")

        # Ensure every row has a plot color.
        for i, row in enumerate(results):
            if "color" not in row:
                row["color"] = COLORS[i % len(COLORS)]

        benchmark_mode = plot_mode.strip().lower()
        if benchmark_mode in {"ttft", "ttft_cost", "ttft_cost_by_delay"}:
            plot_urls = _build_ttft_cost_plots_by_delay(results)
            if not plot_urls:
                raise ValueError(
                    "Imported results do not contain user_delay_ms metadata needed for TTFT plots"
                )
        elif from_directory:
            plot_url, selected = _build_users_cost_plot(results)
            plot_urls = [plot_url]
            results = selected
        else:
            # For single JSON imports, color the points by mode as well so the
            # per-mode color legend stays consistent with directory imports.
            for row in results:
                row["color"] = _color_for_row(row)
            plot_urls = _build_comparison_plots(results)
        plot_title = (
            "TTFT vs Cost by User Delay"
            if benchmark_mode in {"ttft", "ttft_cost", "ttft_cost_by_delay"}
            else "Cost-Latency Plots"
        )
        return HTMLResponse(
            content=_build_results_page(results, plot_urls, None, plot_title=plot_title)
        )
    except Exception as exc:
        import traceback

        err = f"{exc}\n{traceback.format_exc()}"
        return HTMLResponse(content=_build_results_page([], None, err), status_code=400)


@app.get("/api/hardware")
async def hardware_options():
    """Return the list of available hardware preset names."""
    return {"hardware": list(load_combined_machine_db().keys())}


@app.get("/plot/{plot_id}")
async def get_plot(plot_id: str):
    b64 = _plot_store.get(plot_id)
    if not b64:
        return PlainTextResponse("Plot not found", status_code=404)
    return Response(content=base64.b64decode(b64), media_type="image/png")


@app.post("/plot_users_cost", response_class=HTMLResponse)
async def plot_users_cost(results_dir: str = Form(...)):
    """Render a plot of users vs total cost from a directory of results files.

    The directory must contain files named ``results_users_<N>.json``. For each
    user count, only the 10 cheapest configurations (by total cost/hour) are
    plotted.
    """
    try:
        path = _resolve_results_dir(results_dir)
        if not path.is_dir():
            return HTMLResponse(
                content=_build_results_page(
                    [], None, f"Not a directory: {results_dir}"
                ),
                status_code=400,
            )

        rows = _load_results_from_dir(path)
        if not rows:
            return HTMLResponse(
                content=_build_results_page(
                    [], None, f"No results_*.json files found in {results_dir}"
                ),
                status_code=404,
            )

        plot_url, selected = _build_users_cost_plot(rows)
        html = _build_results_page(selected, [plot_url], None)
        return HTMLResponse(content=html)
    except Exception as exc:
        import traceback

        err = f"{exc}\n{traceback.format_exc()}"
        return HTMLResponse(content=_build_results_page([], None, err), status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
