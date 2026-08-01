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

from collections import defaultdict
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
    config_type: str = "separate",
    inter_node_network_up_gbps: float = 100.0,
    inter_node_network_down_gbps: float = 100.0,
) -> list[Node]:
    # Apply per-request inter-node bandwidth overrides from the webserver form
    # before resolving/loading any machine hardware.
    os.environ["INTER_NODE_NETWORK_UP_GBPS"] = str(inter_node_network_up_gbps)
    os.environ["INTER_NODE_NETWORK_DOWN_GBPS"] = str(inter_node_network_down_gbps)

    if config_type not in {"separate", "colocated", "mixed"}:
        raise ValueError(
            f"Invalid config_type '{config_type}'. Use 'separate', 'colocated', or 'mixed'."
        )

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
        f"Building nodes for prefill hardware: {prefill_hw}, decode hardware: {decode_hw}, config_type={config_type}"
    )

    nodes: list[Node] = []
    is_colocated = config_type == "colocated"
    is_mixed = config_type == "mixed"
    if is_colocated or is_mixed:
        if prefill_nodes != decode_nodes:
            raise ValueError(
                f"{config_type.title()} config requires prefill_nodes ({prefill_nodes}) == decode_nodes ({decode_nodes})."
            )
        if prefill_gpus_per_node + decode_gpus_per_node != prefill_total_gpus:
            raise ValueError(
                f"GPU split {prefill_gpus_per_node}+{decode_gpus_per_node} does not equal "
                f"total GPUs per node ({prefill_total_gpus})."
            )

        if is_mixed:
            from src.hardware.hardware import GPUHardwareSpec
            from src.hardware.mixed_gpu import fetch_mixed_gpu_hardware
            from src.hardware.scraper import lookup as lookup_gpu
            from src.hardware.scraper import lookup_machine

            node_hw = fetch_mixed_gpu_hardware(
                prefill_hw_name,
                prefill_gpus_per_node,
                decode_hw_name,
                decode_gpus_per_node,
                compute_price_fraction=0.6,
            )
            donor_machine_config = lookup_machine(decode_hw_name)
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
                    "Colocated config requires identical prefill and decode hardware."
                )
            node_hw = prefill_hw
            donor_gpu_spec = None

        for _ in range(prefill_nodes):
            nodes.append(
                Node(
                    hardware=node_hw,
                    model_name=model,
                    batch_size=batch_size,
                    prefill_instances=prefill_gpus_per_node,
                    decode_instances=decode_gpus_per_node,
                    decode_gpu_hardware=donor_gpu_spec,
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
    user_delay_fraction: float = 0.0,
    user_delay_min_ms: float = 0.0,
    user_delay_max_ms: float = 0.0,
    startup_arrival_mean_ms: float = 0.0,
    random_seed: int | None = None,
    config_type: str = "separate",
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
        config_type,
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
                startup_arrival_mean_ms=startup_arrival_mean_ms,
                random_seed=random_seed,
                bandwidth_aware_routing=False,
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
    show_users_section: bool = True,
) -> str:
    """Build the results HTML page (for /simulate full page) or just the inner content."""
    inner = _results_inner_html(
        results, plot_urls, error, show_debug_tables, plot_title, show_users_section
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
    show_users_section: bool = True,
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
        if "users" in best:
            html += _metric_card(f"{best['users']:,}", "Max users")
        if "price_per_user" in best:
            html += _metric_card(f"${best['price_per_user']:.4f}", "Price / user / h")
        html += _metric_card(f"{best['max_request_latency']:.2f}", "max Latency (ms)")
        html += "</div></div>\n"

        # ---- Users-based ordered legend (only when results have users and section enabled) ----
        if show_users_section and any("users" in row for row in results):
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
                        users = row.get("users")
                        users_str = f" ({users:,} users)" if users is not None else ""
                        html += (
                            f'<li><span class="legend-color" style="background:{color}"></span>'
                            f"[{mode.capitalize()}] {row['label']}{users_str} — ${row['total_cost_usd_per_hour']:.2f}/h"
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
        html += "<th>Label</th><th>Prefill HW</th><th>Decode HW</th><th>Nodes (P/D)</th><th>Batch</th><th>Users</th>"
        html += "<th>TTFT</th><th>max TTFT</th><th>TPOT</th><th>max TPOT</th>"
        html += (
            "<th>Latency</th><th>max Latency</th><th>KV Upload</th><th>KV Download</th>"
        )
        html += "<th>Compute $/h</th><th>S3 $/h</th><th>Total $/h</th><th>Price / user / h</th>"
        html += "</tr></thead><tbody>"
        for row in results:
            if row.get("has_error"):
                html += (
                    f"<tr>"
                    f'<td><span class="legend-color" style="background:{row.get("color", "#58a6ff")}"></span>{row["label"]} <span style="color:var(--danger);font-size:0.75rem;">(failed)</span></td>'
                    f'<td colspan="17" style="text-align:center;color:var(--danger);">Simulation failed — see error banner above</td>'
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
                f"<td>{row.get('users', '—')}</td>"
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
                f"<td>${row.get('price_per_user', float('inf')):.4f}</td>"
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


def _build_results_page_hardware_economics(
    results: list[dict[str, float | int | str]],
    economics_plots: dict[str, list[str]],
) -> str:
    """Build a results page for hardware-economics imports with a TTFT selector."""
    # Shared head style from _build_results_page.
    head = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Configurator Simulator - Hardware Economics</title>
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
        .plot-img { width: 100%; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 1rem; }
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 1rem;
        }
        .plot-card { padding: 0; }
        select {
            background: var(--bg);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        .plot-group { display: none; }
        .plot-group.active { display: block; }
        .legend-color {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 0.4rem;
            vertical-align: middle;
        }
        table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        th, td { padding: 0.5rem 0.75rem; border: 1px solid var(--border); text-align: left; }
        th { background: var(--bg); color: var(--text-secondary); font-weight: 600; }
        td { color: var(--text); }
    </style>
</head>
<body>
    <div class="container">
        <h1>Configurator Simulator</h1>
        <p class="subtitle">Hardware Economics Results</p>
"""

    ttft_keys = sorted(economics_plots.keys())
    selector = '<div class="card"><h2>Select TTFT</h2><select id="ttftSelector" onchange="showTTFT(this.value)">'
    for key in ttft_keys:
        selector += f'<option value="{key}">{key}</option>'
    selector += "</select></div>\n"

    body = selector
    body += '<div id="plotContainer">\n'
    for key in ttft_keys:
        body += f'<div class="plot-group" id="group-{key}" data-ttft="{key}">\n'
        plot_urls = economics_plots.get(key, [])
        if plot_urls:
            body += '<div class="card plot-card"><h2>Max Users and Price per User vs Focus</h2><div class="plot-grid">\n'
            for url in plot_urls:
                body += f'<div><img src="{url}" class="plot-img" alt="Economics Plot"></div>\n'
            body += "</div></div>\n"
        body += "</div>\n"
    body += "</div>\n"

    # Comparison table (always shown, hidden by default would need a toggle; keep simple).
    body += '<div class="card"><h2>Result Rows</h2><table><thead><tr>'
    body += "<th></th><th>Label</th><th>Focus</th><th>TTFT SLA</th><th>User delay (min)</th><th>Max users</th><th>Price / user / h</th><th>Total $/h</th>"
    body += "</tr></thead><tbody>"
    for row in results:
        if row.get("has_error"):
            continue
        ttft = (
            row.get("ttft_sla_ms", row.get("benchmark_ttft_ms", row.get("ttft", 0)))
            / 1000
        )
        delay = _extract_user_delay_ms(row)
        delay_min = f"{delay / 1000 / 60:g}" if delay is not None else "—"
        focus = row.get("focus_value", row.get("focus", "—"))
        users = row.get("users", "—")
        price = row.get("price_per_user", float("inf"))
        total = row.get("total_cost_usd_per_hour", 0.0)
        color = row.get("color", "#58a6ff")
        seed_info = ""
        seed_results = row.get("seed_results")
        if isinstance(seed_results, list) and len(seed_results) > 1:
            users_list = [int(r.get("max_users", 0)) for r in seed_results]
            prices_list = [float(r.get("price_per_user", 0)) for r in seed_results]
            seed_info = (
                f"<br><span style='font-size:0.75rem;color:var(--text-secondary);'>"
                f"seeds: users {users_list}, prices {[f'${p:.2f}' for p in prices_list]}"
                f"</span>"
            )
        body += (
            f'<tr><td><span class="legend-color" style="background:{color}"></span></td>'
            f"<td>{row['label']}{seed_info}</td><td>{focus}</td><td>{ttft:g}s</td><td>{delay_min}</td>"
            f"<td>{users}</td><td>${price:.4f}</td><td>${total:.2f}</td></tr>"
        )
    body += "</tbody></table></div>\n"

    script = """
    <script>
        function showTTFT(ttft) {
            document.querySelectorAll('.plot-group').forEach(g => g.classList.remove('active'));
            const group = document.getElementById('group-' + ttft);
            if (group) group.classList.add('active');
            localStorage.setItem('hardware_economics_last_ttft', ttft);
        }
        const last = localStorage.getItem('hardware_economics_last_ttft');
        const selector = document.getElementById('ttftSelector');
        if (last && Array.from(selector.options).some(o => o.value === last)) {
            selector.value = last;
        }
        showTTFT(selector.value);
    </script>
"""

    tail = """    </div>
</body>
</html>"""
    return head + body + script + tail


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


def _extract_ttft_ms(row: dict[str, Any]) -> float | None:
    for key in ("ttft_sla_ms", "benchmark_ttft_ms", "ttft"):
        value = row.get(key)
        if value is not None:
            return float(value)
    label = str(row.get("label", ""))
    match = re.search(r"TTFT=([0-9.+-eE]+)ms", label)
    if match:
        return float(match.group(1))
    return None


def _aggregate_seed_results(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return mean/min/max/std of seed_results if multiple seeds are present."""
    seed_results = row.get("seed_results")
    if not isinstance(seed_results, list) or len(seed_results) < 2:
        return None
    users = [
        int(r.get("max_users", 0))
        for r in seed_results
        if int(r.get("max_users", 0)) > 0
    ]
    prices = [
        float(r.get("price_per_user", 0))
        for r in seed_results
        if r.get("price_per_user") is not None
    ]
    if not users:
        return None
    return {
        "users_mean": sum(users) / len(users),
        "users_min": min(users),
        "users_max": max(users),
        "users_std": math.sqrt(
            sum((u - sum(users) / len(users)) ** 2 for u in users) / len(users)
        ),
        "price_mean": sum(prices) / len(prices) if prices else 0.0,
        "price_min": min(prices) if prices else 0.0,
        "price_max": max(prices) if prices else 0.0,
        "n": len(users),
    }


def _build_ttft_economics_plots_by_delay(
    results: list[dict[str, float | int | str]],
) -> dict[str, list[str]]:
    """Generate dual-y-axis plots per (TTFT, user_delay, focus) bucket.

    For every focus value on the x-axis the plot shows:
      * a bar for the mean max_users across seeds, with vertical error bars
        spanning [min, max] when multiple seeds are present;
      * a diamond point on the right y-axis for the mean price_per_user;
      * a line connecting the price points to highlight the cost trend;
      * numeric labels for both the mean max users and the mean price.

    Each focus category (e.g., NVLink, SSD, RAM) gets its own plot so the
    x-axis stays readable and comparisons within a category are meaningful.
    Returns a mapping from TTFT key (e.g. "TTFT=10s") to a list of plot URLs,
    one URL per (user-delay, focus) combination.
    """
    valid_rows = [row for row in results if not row.get("has_error")]

    # Group by (ttft_key, delay_ms, focus, focus_value).  When multiple seeds
    # were requested, the runner emits one row per seed, but every row carries
    # the full ``seed_results`` aggregate.  We therefore deduplicate on
    # focus_value and keep only the first representative row per value.
    seen: dict[tuple[str, float, str, str], dict[str, Any]] = {}
    for row in valid_rows:
        ttft_ms = _extract_ttft_ms(row)
        delay_ms = _extract_user_delay_ms(row)
        focus = str(row.get("focus") or "default")
        focus_value = str(row.get("focus_value") or "")
        if ttft_ms is None or delay_ms is None:
            continue
        ttft_key = f"TTFT={ttft_ms / 1000:g}s"
        key = (ttft_key, round(delay_ms, 6), focus, focus_value)
        if key not in seen:
            seen[key] = row

    buckets: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for (ttft_key, delay_ms, focus, _focus_value), row in seen.items():
        buckets[(ttft_key, delay_ms, focus)].append(row)

    def _sort_key(row: dict[str, Any]) -> tuple[float, str, str]:
        focus_value = str(row.get("focus_value") or "")
        try:
            numeric = float(focus_value)
        except ValueError:
            numeric = float("inf")
        label = str(row.get("label", ""))
        return (numeric, focus_value, label)

    plots_by_ttft: dict[str, list[str]] = defaultdict(list)
    for (ttft_key, delay_ms, focus), rows in sorted(buckets.items()):
        rows = sorted(rows, key=_sort_key)

        focus_labels: list[str] = []
        users_means: list[float] = []
        users_mins: list[float] = []
        users_maxs: list[float] = []
        price_means: list[float] = []
        price_mins: list[float] = []
        price_maxs: list[float] = []
        seed_users: list[list[int]] = []
        has_multi_seed = False

        for row in rows:
            stats = _aggregate_seed_results(row)
            multi_seed = stats is not None
            if multi_seed:
                has_multi_seed = True
                assert stats is not None
                users_means.append(stats["users_mean"])
                users_mins.append(stats["users_min"])
                users_maxs.append(stats["users_max"])
                price_means.append(stats["price_mean"])
                price_mins.append(stats["price_min"])
                price_maxs.append(stats["price_max"])
                seed_users.append([int(r["max_users"]) for r in row["seed_results"]])
            else:
                users_means.append(float(row.get("users", 0)))
                users_mins.append(float(row.get("users", 0)))
                users_maxs.append(float(row.get("users", 0)))
                price_means.append(float(row.get("price_per_user", 0)))
                price_mins.append(float(row.get("price_per_user", 0)))
                price_maxs.append(float(row.get("price_per_user", 0)))
                seed_users.append([int(row.get("users", 0))])

            focus_value = str(row.get("focus_value") or "")
            focus_labels.append(focus_value if focus_value else focus)

        x = list(range(len(rows)))

        fig, ax_users = plt.subplots(figsize=(8, 5))
        ax_price = ax_users.twinx()

        users_color = "#58a6ff"
        price_color = "#f0883e"
        seed_color = "#1f2328"

        # Mean max users as errorbar points with [min, max] caps.
        users_lower = [
            max(0, mean - min_val)
            for mean, min_val in zip(users_means, users_mins, strict=False)
        ]
        users_upper = [
            max_val - mean
            for mean, max_val in zip(users_means, users_maxs, strict=False)
        ]
        ax_users.errorbar(
            x,
            users_means,
            yerr=[users_lower, users_upper],
            fmt="o",
            color=users_color,
            ecolor=users_color,
            elinewidth=2,
            capsize=5,
            capthick=1.5,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label="Max users (mean)",
            zorder=4,
        )

        # Individual seed dots for users.
        if has_multi_seed:
            for i, dot_users in enumerate(seed_users):
                for seed_idx, du in enumerate(dot_users):
                    offset = (seed_idx - len(dot_users) / 2) * 0.05
                    ax_users.scatter(
                        i + offset,
                        du,
                        s=25,
                        color=seed_color,
                        edgecolors="none",
                        alpha=0.6,
                        zorder=3,
                    )

        # Mean price per user as a simple diamond point (no error bars).
        ax_price.plot(
            x,
            price_means,
            color=price_color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
            zorder=2,
        )
        ax_price.scatter(
            x,
            price_means,
            s=120,
            color=price_color,
            marker="D",
            edgecolors="white",
            linewidths=0.5,
            label="Price / user / h (mean)",
            zorder=5,
        )

        # Numeric labels for mean values.
        for i, (u_mean, p_mean) in enumerate(
            zip(users_means, price_means, strict=False)
        ):
            ax_users.annotate(
                f"{u_mean:.1f}",
                (i, u_mean),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color=users_color,
                fontweight="bold",
                zorder=6,
            )
            ax_price.annotate(
                f"${p_mean:.2f}",
                (i, p_mean),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color=price_color,
                zorder=6,
            )

        ax_users.set_xlabel("Focus value")
        ax_users.set_ylabel("Max users (mean)", color=users_color)
        ax_price.set_ylabel("Price per user ($ / h)", color=price_color)
        ax_users.set_xticks(x)
        ax_users.set_xticklabels(focus_labels, rotation=45, ha="right")
        ax_users.tick_params(axis="y", labelcolor=users_color)
        ax_price.tick_params(axis="y", labelcolor=price_color)

        title = f"{focus} — Max Users and Price / User ({ttft_key}, delay {delay_ms / 1000 / 60:g} min)"
        if has_multi_seed:
            title += " (mean ± range over seeds)"
        ax_users.set_title(title)
        ax_users.grid(True, alpha=0.3, axis="y")

        # Combined legend.
        legend_handles = [
            plt.Line2D(
                [],
                [],
                color=users_color,
                marker="o",
                linestyle="None",
                markersize=8,
                markeredgecolor="white",
                label="Max users (mean)",
            ),
            plt.Line2D(
                [],
                [],
                color=price_color,
                marker="D",
                linestyle="--",
                markersize=8,
                markeredgecolor="white",
                label="Price / user / h (mean)",
            ),
        ]
        if has_multi_seed:
            legend_handles.append(
                plt.Line2D(
                    [],
                    [],
                    color=seed_color,
                    marker="o",
                    linestyle="None",
                    markersize=5,
                    label="Individual seed results",
                )
            )
        ax_users.legend(handles=legend_handles, loc="upper left")

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        pid = str(uuid.uuid4())
        _plot_store[pid] = base64.b64encode(buf.read()).decode("utf-8")
        plots_by_ttft[ttft_key].append(f"/plot/{pid}")

    return dict(plots_by_ttft)


def _load_results_from_dir(results_dir: Path) -> tuple[list[dict[str, Any]], str]:
    """Load every results_*.json file from a directory and detect the benchmark type.

    Returns the flat list of result rows plus the detected benchmark key:
    ``user_sweep`` when files are named ``results_users_<N>.json``,
    ``hardware_economics`` otherwise (e.g. ``results_<focus>_<value>.json``).
    """
    rows: list[dict[str, Any]] = []
    benchmark: str | None = None
    for path in sorted(results_dir.glob("results_*.json")):
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        file_benchmark = data.get("benchmark")
        if isinstance(file_benchmark, str):
            benchmark = file_benchmark
        elif path.stem.startswith("results_users_"):
            benchmark = "user_sweep"
        else:
            benchmark = "hardware_economics"
        for row in data.get("results", []):
            if "users" not in row and path.stem.startswith("results_users_"):
                # Infer users from filename like results_users_100.json
                stem = path.stem.replace("results_users_", "")
                with suppress(ValueError):
                    row["users"] = int(stem)
            rows.append(row)
    return rows, benchmark or "hardware_economics"


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
    entirely.  Labels generated by ``create_user_sweep_config.py`` start with
    ``Colocated:``, ``Mixed:`` or ``separate:``.

    As a secondary fallback we still check the explicit ``config_type`` field
    (normalizing booleans/strings) and the mixed-GPU hardware-name marker `` + ``.
    """
    label = str(row.get("label", "")).strip().lower()
    if label.startswith("colocated"):
        return "colocated"
    if label.startswith("mixed"):
        return "mixed"
    if label.startswith("separate"):
        return "separate"

    config_type = str(row.get("config_type", "separate")).strip().lower()
    if config_type in {"colocated", "true", "1", "yes", "on"}:
        return "colocated"
    if config_type == "mixed":
        return "mixed"

    prefill_hw = str(row.get("prefill_hardware", ""))
    if " + " in prefill_hw:
        return "mixed"

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
    cfg_config_type: list[str] = Form(...),
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
    user_delay_fraction: float = Form(_env.user_delay_fraction),
    user_delay_min_ms: float = Form(_env.user_delay_min_ms),
    user_delay_max_ms: float = Form(_env.user_delay_max_ms),
    startup_arrival_mean_ms: float = Form(_env.startup_arrival_mean_ms),
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
            == len(cfg_config_type)
            == len(cfg_batch)
            == len(cfg_label)
            == n
        ):
            raise ValueError("Configuration arrays must all have the same length.")

        def _normalize_config_type(raw: str) -> str:
            value = str(raw).strip().lower()
            if value in {"colocated", "true", "1", "yes", "on"}:
                return "colocated"
            if value == "mixed":
                return "mixed"
            return "separate"

        config_types = [_normalize_config_type(v) for v in cfg_config_type]

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
            "user_delay_fraction": user_delay_fraction,
            "user_delay_min_ms": user_delay_min_ms,
            "user_delay_max_ms": user_delay_max_ms,
            "startup_arrival_mean_ms": startup_arrival_mean_ms,
            "random_seed": random_seed,
        }

        config_kwargs: list[dict[str, object]] = []
        for i in range(n):
            prefill_hw = cfg_prefill_hardware[i]
            decode_hw = cfg_decode_hardware[i]
            prefill_n = int(cfg_prefill_nodes[i])
            config_type = config_types[i]
            # In colocated/mixed mode decode_nodes mirrors prefill_nodes.
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
                "config_type": config_type,
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
                "config_type": "separate",
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


@app.get("/help", response_class=HTMLResponse)
async def help_page():
    """Serve the webserver usage guide."""
    html_path = Path(__file__).parent / "templates" / "help.html"
    with Path(html_path).open(encoding="utf-8") as fh:
        return HTMLResponse(content=fh.read())


def _detect_benchmark(payload: Any) -> str:
    """Detect the benchmark type from an imported JSON payload."""
    if isinstance(payload, dict):
        benchmark = payload.get("benchmark")
        if isinstance(benchmark, str):
            return benchmark
        results = payload.get("results", [])
        if isinstance(results, list) and results:
            return _detect_row_benchmark(results[0])
    if isinstance(payload, list) and payload:
        return _detect_row_benchmark(payload[0])
    return "hardware_economics"


def _detect_row_benchmark(row: dict[str, Any]) -> str:
    """Heuristic: hardware_economics rows carry user_delay_ms; user_sweep rows carry users."""
    if row.get("user_delay_ms") is not None:
        return "hardware_economics"
    if row.get("users") is not None:
        return "user_sweep"
    label = str(row.get("label", "")).lower()
    if "delay=" in label or "ttft=" in label:
        return "hardware_economics"
    if "users=" in label:
        return "user_sweep"
    return "hardware_economics"


@app.post("/import_results", response_class=HTMLResponse)
async def import_results(
    results_json: str = Form(""),
    results_dir: str = Form(""),
    plot_mode: str = Form("auto"),
):
    """Render a results page from pasted JSON or from a results directory.

    The result type is auto-detected:
    - ``user_sweep`` → directory of ``results_users_<N>.json`` (cost vs users).
    - ``hardware_economics`` → TTFT/user-delay sweeps (cost and price/user by delay).
    """
    try:
        results: list[dict[str, float | int | str]] = []

        from_directory = False
        if results_dir:
            from_directory = True
            path = _resolve_results_dir(results_dir)
            if not path.is_dir():
                raise ValueError(f"Not a directory: {results_dir}")
            rows, dir_benchmark = _load_results_from_dir(path)
            if not rows:
                raise ValueError(
                    f"Directory '{results_dir}' has results_*.json files, but none contain valid result rows."
                )
            results.extend(rows)
            benchmark_mode = (
                plot_mode.strip().lower()
                if plot_mode.strip().lower() != "auto"
                else dir_benchmark
            )
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
            benchmark_mode = (
                plot_mode.strip().lower()
                if plot_mode.strip().lower() != "auto"
                else _detect_benchmark(payload)
            )
        else:
            raise ValueError("Provide either results_json or results_dir")

        if not isinstance(results, list) or not results:
            raise ValueError("No result rows found")

        # Ensure every row has a plot color.
        for i, row in enumerate(results):
            if "color" not in row:
                row["color"] = COLORS[i % len(COLORS)]

        if benchmark_mode not in {"user_sweep", "hardware_economics"}:
            benchmark_mode = "hardware_economics"

        if benchmark_mode == "hardware_economics":
            economics_plots = _build_ttft_economics_plots_by_delay(results)
            if not economics_plots:
                raise ValueError(
                    "Imported hardware-economics results do not contain ttft_sla_ms and user_delay_ms metadata."
                )
            plot_title = "Hardware Economics (max users and price per user by focus)"
            show_users_section = True
            return HTMLResponse(
                content=_build_results_page_hardware_economics(results, economics_plots)
            )
        if benchmark_mode == "user_sweep":
            if not from_directory:
                raise ValueError(
                    "user_sweep mode requires a results directory (results_users_*.json)."
                )
            plot_url, selected = _build_users_cost_plot(results)
            plot_urls = [plot_url]
            results = selected
            plot_title = "User Sweep (cost vs users)"
            show_users_section = True
        else:
            raise ValueError(
                f"Unknown plot mode '{benchmark_mode}'. Use 'auto', 'user_sweep', or 'hardware_economics'."
            )
        return HTMLResponse(
            content=_build_results_page(
                results,
                plot_urls,
                None,
                plot_title=plot_title,
                show_users_section=show_users_section,
            )
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
