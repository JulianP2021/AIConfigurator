"""FastAPI webserver for the Configurator Simulator."""

import base64
import io
import sys
import uuid

from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, PlainTextResponse, Response


# Ensure project root is on sys.path when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.hardware.hardware import Hardware
from src.node.node import Node
from src.request.request import RequestScenario, TokenDistribution
from src.result import SimulationResult
from src.simulations.simulation_distributed import (
    DistributedScenario,
    simulate_run_distributed,
)


matplotlib.use("Agg")

app = FastAPI(title="Configurator Simulator")

# In-memory storage for plots (keyed by UUID)
_plot_store: dict[str, str] = {}  # id -> base64 PNG


def _build_nodes(
    prefill_hardware_name: str,
    decode_hardware_name: str,
    prefill_nodes: int,
    decode_nodes: int,
    batch_size: int,
    model: str,
) -> list[Node]:
    prefill_hw = Hardware.from_name(prefill_hardware_name)
    decode_hw = Hardware.from_name(decode_hardware_name)
    prefill_gpus_per_node = prefill_hw.spec.num_gpus
    decode_gpus_per_node = decode_hw.spec.num_gpus
    nodes: list[Node] = []
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
    batch_size: int,
    model: str,
    isl: int,
    osl: int,
    total_requests: int,
    req_rate: float,
    cache_pct: float,
) -> SimulationResult:
    nodes = _build_nodes(
        prefill_hardware,
        decode_hardware,
        prefill_nodes,
        decode_nodes,
        batch_size,
        model,
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
                cache_percentage=cache_pct,
            ),
            total_requests=total_requests,
            min_users=1,
            max_users=10,
            req_s=req_rate,
        ),
    )
    with io.StringIO() as buf, redirect_stdout(buf):
        return simulate_run_distributed(scenario)


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
        best = min(results, key=lambda r: r["request_latency"])
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
        html += "<th>Latency</th><th>max Latency</th><th>tok/s</th><th>tok/s/GPU</th><th>req/s</th>"
        html += "<th>Price/h</th>"
        html += "</tr></thead><tbody>"
        for row in results:
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
                f"<td>{row['tokens_per_second']:.2f}</td>"
                f"<td>{row['tokens_per_second_per_gpu']:.2f}</td>"
                f"<td>{row['request_rate']:.3f}</td>"
                f"<td>${row['price_usd_per_hour']:.2f}</td>"
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


def _build_comparison_plots(results: list[dict[str, float | int | str]]) -> list[str]:
    """Generate base64-encoded PNGs for multi-config comparison.
    Returns list of plot IDs (6 plots: avg and max for TTFT, TPOT, Latency).
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
    req_rate: float = Form(...),
    cache_pct: float = Form(...),
    cfg_prefill_hardware: list[str] = Form(...),
    cfg_decode_hardware: list[str] = Form(...),
    cfg_prefill_nodes: list[str] = Form(...),
    cfg_decode_nodes: list[str] = Form(...),
    cfg_batch: list[str] = Form(...),
    cfg_label: list[str] = Form(...),
    xhr: str = Form("0"),
):
    try:
        # Gather configs
        n = len(cfg_prefill_hardware)
        if not (
            len(cfg_decode_hardware)
            == len(cfg_prefill_nodes)
            == len(cfg_decode_nodes)
            == len(cfg_batch)
            == len(cfg_label)
            == n
        ):
            raise ValueError("Configuration arrays must all have the same length.")

        results_data: list[dict[str, float | int | str]] = []
        skipped_labels: list[str] = []
        for i in range(n):
            prefill_hw = cfg_prefill_hardware[i]
            decode_hw = cfg_decode_hardware[i]
            prefill_n = int(cfg_prefill_nodes[i])
            decode_n = int(cfg_decode_nodes[i])
            batch = int(cfg_batch[i])
            label = cfg_label[i] or f"Config {i + 1}"

            if prefill_n == 0 and decode_n == 0:
                raise ValueError(
                    f"Config '{label}' must have at least one prefill or decode node."
                )

            try:
                result = _run_single_config(
                    label=label,
                    prefill_hardware=prefill_hw,
                    decode_hardware=decode_hw,
                    prefill_nodes=prefill_n,
                    decode_nodes=decode_n,
                    batch_size=batch,
                    model=model,
                    isl=isl,
                    osl=osl,
                    total_requests=requests,
                    req_rate=req_rate,
                    cache_pct=cache_pct,
                )
            except AssertionError as exc:
                if "Too many requests in prefill queue" in str(exc):
                    skipped_labels.append(label)
                    print(f"Skipping config '{label}' due to prefill queue overflow.")
                    continue
                raise

            results_data.append({
                "label": label,
                "prefill_hardware": prefill_hw,
                "decode_hardware": decode_hw,
                "prefill_nodes": prefill_n,
                "decode_nodes": decode_n,
                "batch_size": batch,
                "ttft": result.ttft,
                "max_ttft": result.max_ttft,
                "tpot": result.tpot,
                "max_tpot": result.max_tpot,
                "request_latency": result.request_latency,
                "max_request_latency": result.max_request_latency,
                "tokens_per_second": result.tokens_per_second,
                "tokens_per_second_per_gpu": result.tokens_per_second_per_gpu,
                "request_rate": result.request_rate,
                "price_usd_per_hour": result.price_usd_per_hour,
                "color": COLORS[i % len(COLORS)],
            })

        error_msg = None
        if skipped_labels:
            error_msg = f"Skipped configs (too many requests in prefill queue): {', '.join(skipped_labels)}"

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


@app.get("/api/hardware")
async def hardware_options():
    """Return the list of available hardware preset names."""
    return {"hardware": list(Hardware.PRESETS.keys())}


@app.get("/plot/{plot_id}")
async def get_plot(plot_id: str):
    b64 = _plot_store.get(plot_id)
    if not b64:
        return PlainTextResponse("Plot not found", status_code=404)
    return Response(content=base64.b64decode(b64), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
