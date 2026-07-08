"""FastAPI webserver for the Configurator Simulator."""

import asyncio
import base64
import concurrent.futures
import io
import sys
import uuid

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, redirect_stdout
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, PlainTextResponse, Response


# Ensure project root is on sys.path when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.hardware.hardware import S3Spec
from src.hardware.scraper import (
    fetch_machine_hardware,
    load_machine_db,
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
) -> list[Node]:
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
    total_requests: int,
    users: int,
    think_time_ms: float,
    max_session_turns: int,
    ram_usage_fraction: float,
    ssd_usage_fraction: float,
    s3_enabled: bool,
    s3_up_bw_gbps: float,
    s3_down_bw_gbps: float,
    router_prefill_load_scale: float,
    router_device_credit: float,
    router_remote_ram_credit: float,
    router_ssd_credit: float,
    router_s3_credit: float,
    router_busy_threshold_tokens: float,
    colocated: bool = False,
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
            total_requests=total_requests,
            users=users,
            max_session_turns=max_session_turns,
            think_time_ms=think_time_ms,
        ),
    )
    print(f"Simulating scenario: {scenario}")

    s3_spec = S3Spec.from_gbps(
        enabled=s3_enabled,
        up_gbps=s3_up_bw_gbps,
        down_gbps=s3_down_bw_gbps,
    )
    router_cost_config = RouterCostConfig(
        prefill_load_scale=router_prefill_load_scale,
        device_credit=router_device_credit,
        remote_ram_credit=router_remote_ram_credit,
        ssd_credit=router_ssd_credit,
        s3_credit=router_s3_credit,
        busy_threshold_tokens=router_busy_threshold_tokens,
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
                sla={"ttft_ms": float("inf"), "tpot_ms": float("inf")},
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
) -> str:
    """Build the results HTML page (for /simulate full page) or just the inner content."""
    inner = _results_inner_html(results, plot_urls, error)
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
</body>
</html>"""
    )


def _results_inner_html(
    results: list[dict[str, float | int | str]],
    plot_urls: list[str] | None = None,
    error: str | None = None,
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
        html += _metric_card(f"${best['price_usd_per_hour']:.2f}", "Price / hour")
        html += _metric_card(f"{best['max_request_latency']:.2f}", "max Latency (ms)")
        html += "</div></div>\n"

        # ---- Comparison table ----
        html += '<div class="card"><h2>Configuration Comparison</h2><table><thead><tr>'
        html += "<th>Label</th><th>Prefill HW</th><th>Decode HW</th><th>Nodes (P/D)</th><th>Batch</th>"
        html += "<th>TTFT</th><th>max TTFT</th><th>TPOT</th><th>max TPOT</th>"
        html += (
            "<th>Latency</th><th>max Latency</th><th>KV Upload</th><th>KV Download</th>"
        )
        html += "<th>Price/h</th>"
        html += "</tr></thead><tbody>"
        for row in results:
            if row.get("has_error"):
                html += (
                    f'<tr style="opacity:0.7;">'
                    f'<td><span class="legend-color" style="background:{row.get("color", "#58a6ff")}"></span>{row["label"]} <span style="color:var(--danger);font-size:0.75rem;">(failed)</span></td>'
                    f'<td colspan="13" style="text-align:center;color:var(--danger);">Simulation failed — see error banner above</td>'
                    f"</tr>"
                )
                continue
            html += (
                f"<tr>"
                f'<td><span class="legend-color" style="background:{row.get("color", "#58a6ff")}"></span>{row["label"]}</td>'
                f"<td>{row['prefill_hardware']}</td>"
                f"<td>{row['decode_hardware']}</td>"
                f"<td>{row['prefill_nodes']} / {row['decode_nodes']}</td>"
                f"<td>{row['batch_size']}</td>"
                f"<td>{row['ttft']:.2f}</td>"
                f"<td>{row['max_ttft']:.2f}</td>"
                f"<td>{row['tpot']:.2f}</td>"
                f"<td>{row['max_tpot']:.2f}</td>"
                f"<td>{row['request_latency']:.2f}</td>"
                f"<td>{row['max_request_latency']:.2f}</td>"
                f"<td>{row['kv_upload_time']:.2f}</td>"
                f"<td>{row['kv_download_time']:.2f}</td>"
                f"<td>${row['price_usd_per_hour']:.2f}</td>"
                f"</tr>"
            )
        html += "</tbody></table></div>\n"

        # ---- Timing breakdown table ----
        html += '<div class="card"><h2>Timing Breakdown</h2><table><thead><tr>'
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
                f"<td>{row['prefill_time']:.2f}</td>"
                f"<td>{row['prefill_wait']:.2f}</td>"
                f"<td>{row['prefill_download_active']:.2f}</td>"
                f"<td>{row['prefill_download_wait']:.2f}</td>"
                f"<td>{row['prefill_upload_active']:.2f}</td>"
                f"<td>{row['prefill_upload_wait']:.2f}</td>"
                f"<td>{row['decode_download_active']:.2f}</td>"
                f"<td>{row['decode_download_wait']:.2f}</td>"
                f"<td>{row['decode_time']:.2f}</td>"
                f"<td>{row['decode_wait']:.2f}</td>"
                f"<td>{row['decode_upload_active']:.2f}</td>"
                f"<td>{row['decode_upload_wait']:.2f}</td>"
                f"<td>{row['clean_ttft']:.2f}</td>"
                f"<td>{row['ttft']:.2f}</td>"
                f"<td>{row['clean_latency']:.2f}</td>"
                f"<td>{row['request_latency']:.2f}</td>"
                f"</tr>"
            )
        html += "</tbody></table></div>\n"

    if plot_urls:
        html += '<div class="card plot-card"><h2>Cost-Latency Plots</h2><div class="plot-grid">'
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
            row["price_usd_per_hour"],
            s=120,
            color=row.get("color", "#58a6ff"),
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax.annotate(
            row["label"],
            (row[x_key], row["price_usd_per_hour"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
            color=row.get("color", "#58a6ff"),
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Price ($/hour)")
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
]


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
    requests: int = Form(...),
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
    router_prefill_load_scale: float = Form(_env.router_prefill_load_scale),
    router_device_credit: float = Form(_env.router_device_credit),
    router_remote_ram_credit: float = Form(_env.router_remote_ram_credit),
    router_ssd_credit: float = Form(_env.router_ssd_credit),
    router_s3_credit: float = Form(_env.router_s3_credit),
    router_busy_threshold_tokens: float = Form(_env.router_busy_threshold_tokens),
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
            "total_requests": requests,
            "users": users,
            "think_time_ms": think_time_ms,
            "max_session_turns": max_session_turns,
            "ram_usage_fraction": ram_usage_fraction,
            "ssd_usage_fraction": ssd_usage_fraction,
            "s3_enabled": s3_on,
            "s3_up_bw_gbps": s3_up_bw_gbps,
            "s3_down_bw_gbps": s3_down_bw_gbps,
            "router_prefill_load_scale": router_prefill_load_scale,
            "router_device_credit": router_device_credit,
            "router_remote_ram_credit": router_remote_ram_credit,
            "router_ssd_credit": router_ssd_credit,
            "router_s3_credit": router_s3_credit,
            "router_busy_threshold_tokens": router_busy_threshold_tokens,
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
            results_data.append({
                "label": label,
                "prefill_hardware": kwargs["prefill_hardware"],
                "decode_hardware": kwargs["decode_hardware"],
                "prefill_nodes": kwargs["prefill_nodes"],
                "decode_nodes": kwargs["decode_nodes"],
                "prefill_gpus_per_node": kwargs.get("prefill_gpus_per_node", 0),
                "decode_gpus_per_node": kwargs.get("decode_gpus_per_node", 0),
                "batch_size": kwargs["batch_size"],
                "colocated": kwargs.get("colocated", False),
                "ttft": result.ttft,
                "kv_upload_time": result.kv_upload_time,
                "kv_download_time": result.kv_download_time,
                "max_ttft": result.max_ttft,
                "tpot": result.tpot,
                "max_tpot": result.max_tpot,
                "request_latency": result.request_latency,
                "max_request_latency": result.max_request_latency,
                "tokens_per_second": result.tokens_per_second,
                "tokens_per_second_per_gpu": result.tokens_per_second_per_gpu,
                "request_rate": result.seq_per_second,
                "price_usd_per_hour": result.price_usd_per_hour,
                "color": COLORS[i % len(COLORS)],
                "has_error": False,
                # Timing breakdown fields
                "prefill_time": result.avg_prefill_time_ms,
                "prefill_wait": result.avg_prefill_wait_ms,
                "prefill_download_active": result.avg_prefill_download_active_ms,
                "prefill_download_wait": result.avg_prefill_download_wait_ms,
                "prefill_upload_active": result.avg_prefill_upload_active_ms,
                "prefill_upload_wait": result.avg_prefill_upload_wait_ms,
                "decode_download_active": result.avg_decode_download_active_ms,
                "decode_download_wait": result.avg_decode_download_wait_ms,
                "decode_time": result.avg_decode_time_ms,
                "decode_wait": result.avg_decode_wait_ms,
                "decode_upload_active": result.avg_decode_upload_active_ms,
                "decode_upload_wait": result.avg_decode_upload_wait_ms,
                "clean_ttft": result.avg_clean_ttft_ms,
                "clean_latency": result.avg_clean_latency_ms,
            })

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
                "price_usd_per_hour": 0.0,
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


@app.post("/import_results", response_class=HTMLResponse)
async def import_results(payload: dict[str, list[dict[str, float | int | str]]]):
    """Render a results page from a previously exported results_data JSON.

    This lets users import the raw results list produced by compare.py or the
    webserver's /simulate handler and regenerate the comparison plots/table.
    """
    try:
        results = payload.get("results", payload)
        if not isinstance(results, list):
            raise ValueError(
                "Payload must be a list of result rows or {'results': [...]}"
            )

        # Ensure every row has a plot color.
        for i, row in enumerate(results):
            if "color" not in row:
                row["color"] = COLORS[i % len(COLORS)]

        plot_urls = _build_comparison_plots(results)
        return HTMLResponse(content=_build_results_page(results, plot_urls, None))
    except Exception as exc:
        import traceback

        err = f"{exc}\n{traceback.format_exc()}"
        return HTMLResponse(content=_build_results_page([], None, err))


@app.get("/api/hardware")
async def hardware_options():
    """Return the list of available hardware preset names."""
    return {"hardware": list(load_machine_db().keys())}


@app.get("/plot/{plot_id}")
async def get_plot(plot_id: str):
    b64 = _plot_store.get(plot_id)
    if not b64:
        return PlainTextResponse("Plot not found", status_code=404)
    return Response(content=base64.b64decode(b64), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
