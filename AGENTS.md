# Agent Guide for the Distributed LLM Inference Simulator

This file is the source of truth for project structure, conventions, and common pitfalls when working on the simulator.

## Project Overview

Discrete-event simulator for **disaggregated (prefill/decode) LLM inference** with distributed prefix KV caching.

Main entry points:
- `main.py` — command-line simulator.
- `src/webserver/server.py` — FastAPI server exposing the same simulator through `simulate_run_distributed`.

## Architecture

```
main.py / src/webserver/server.py
    │
    ▼
src/simulations/simulation_distributed.py
    │
    ├── Router ............. src/router/router.py
    ├── PrefillInstance .... src/instances/prefill.py
    ├── DecodeInstance ..... src/instances/decode.py
    ├── Cache .............. src/cache/cache.py
    └── BandwidthScheduler . src/scheduler/bandwidth_scheduler.py
```

Other key modules:
- `src/request/request.py` — `Request`, `RequestScenario`, `RequestGenerator`, `TransferLeg`, `DownloadRequest`, `UploadRequest`.
- `src/hardware/hardware.py` — `Hardware`, `HardwareSpec`, `GPUHardwareSpec` loaded from `_machine_db.json`.
- `src/hardware/mixed_gpu.py` — mixed-GPU node pricing and `fetch_mixed_gpu_hardware`.
- `src/model/model.py` — thin wrapper around a HF model name; provides `kv_size_per_token`.
- `src/logger.py` — bitmask-based logging.
- `src/utils/env_reader.py` — `.env` loader and `EnvConfig` defaults.
- `src/result.py` — `SimulationResult` dataclass used by both CLI and webserver.

## Build / Run

No build step is required. Use the local `.venv` for every Python command.

```bash
# CLI
.venv/bin/python main.py

# Web server
.venv/bin/python -m uvicorn src.webserver.server:app --reload

# Tests
.venv/bin/python -m pytest tests/

# Module import check
.venv/bin/python -m py_compile main.py src/**/*.py src/webserver/server.py tests/*.py
```

## Configuration

Configuration is read from `.env` at the project root and can be overridden by shell environment variables or CLI flags. The canonical list of parameters lives in:
- `.env`
- `src/utils/env_reader.py` (`EnvConfig` and `_DEFAULTS`)
- `main.py` argument parser
- `src/webserver/server.py` (for server-exposed parameters)

When adding a new CLI/env parameter, mirror it in **all four** places unless it is intentionally CLI-only.

### Notable parameters

```bash
MODEL=Qwen/Qwen3-8B
ISL=1000
OSL=100
SESSIONS_PER_USER=1
USERS=10
MAX_SESSION_TURNS=5
THINK_TIME_MS=0

SLA_TTFT_MS=inf          # per-request TTFT SLA (inf = disabled)
SLA_TPOT_MS=inf          # per-request TPOT SLA (inf = disabled)

BATCH_SIZE=10
NUM_PREFILL_NODES=1
NUM_DECODE_NODES=1
COLOCATED=false
PREFILL_GPUS_PER_NODE=-1 # -1 means use all GPUs from the hardware preset
MACHINE_HARDWARE=B200 x8 #15825275
MIXED=false              # mixed-GPU colocated node
MIXED_GPU_DONOR=         # donor machine for decode GPUs
MIXED_GPU_COUNT=-1       # -1 means use the decode-side GPU count

RAM_USAGE_FRACTION=0.8
SSD_USAGE_FRACTION=0.8

S3_ENABLED=true
S3_UP_BW_GBPS=25.0
S3_DOWN_BW_GBPS=25.0

INTER_NODE_NETWORK_UP_GBPS=100.0    # datacenter NIC for node-to-node KV transfers
INTER_NODE_NETWORK_DOWN_GBPS=100.0

ROUTER_PREFILL_LOAD_SCALE=1.0
ROUTER_DEVICE_CREDIT=1.0
ROUTER_REMOTE_RAM_CREDIT=0.5
ROUTER_SSD_CREDIT=0.3
ROUTER_S3_CREDIT=0.1
ROUTER_BUSY_THRESHOLD_TOKENS=1000000.0

LOG_MASK=0
DEBUG=false
```

### Logging bitmask (`LOG_MASK`)

`LOG_MASK` is an integer built by OR-ing component bits:

| Bit | Value | Component |
|-----|-------|-----------|
| 0   | 1     | Cache |
| 1   | 2     | Instances (prefill/decode) |
| 2   | 4     | Router |
| 3   | 8     | Simulation |
| 4   | 16    | Bandwidth |
| 5   | 32    | Config executor |

Examples: `0` = nothing, `1` = cache only, `15` = cache + instances + router + simulation, `63` = everything.

Use `src.logger.set_log_mask()` or `--log-mask` to change it at runtime. In new code, call `log(LOG_*, msg)`. Do not use the legacy `debug_print()`.

## Coding Conventions

- Python 3.12+ syntax is fine; type hints are encouraged.
- Keep changes minimal and focused on the requested goal.
- Do not change existing test logic when refactoring. If a change requires test fixture updates, keep the update as small as possible (e.g., add a single initialized attribute).
- When adding new CLI/env parameters, mirror them in `.env`, `src/utils/env_reader.py`, `main.py`, and `src/webserver/server.py`.
- Use `from src.logger import LOG_*, log` for new log points.
- Do not use the legacy `debug_print()` in new code.

## Cache Model

- Every node has two cache tiers: `RAM` and `SSD`.
- Capacities are derived from `HardwareSpec.ram_mem` and `nvme_mem`, multiplied by `ram_usage_fraction` / `ssd_usage_fraction`.
- New KV chunks are inserted into RAM first.
- When RAM is full, the least-recently-used (LRU) item is evicted to SSD.
- When SSD is full, its LRU item is deleted permanently.
- The cache raises `ValueError` at construction time if either tier cannot hold a 512-token KV item.
- Remote reads from SSD generate a sequential `SSD_LOCAL → RAM_LOCAL → NETWORK → RAM_LOCAL` leg chain.
- `Cache.usage_summary()` and `SimulationResult` report RAM/SSD/S3 byte usage using maintained counters, not on-demand summation.

## Bandwidth Model

- Bandwidth is scheduled globally by `BandwidthScheduler` using equal-share fairness.
- Bottlenecks:
  - `RAM_LOCAL` shares the node's `pcie_bw`.
  - `SSD_LOCAL` shares the node's `nvme_bw`.
  - `NETWORK` (node-to-node KV transfers) uses the minimum of the source node's `network_inter_node_up` share and the destination node's `network_inter_node_down` share.
  - `S3_UPLOAD` / `S3_DOWNLOAD` use the node's `network_inet_up` / `network_inet_down` internet link.
  - `S3_UPLOAD` / `S3_DOWNLOAD` use the configured S3 link bandwidth.
- `DownloadRequest` / `UploadRequest` contain parallel *tracks* of sequential `TransferLeg`s. Only the active leg on each track receives bandwidth.
- Instances register/unregister transfers with the scheduler as they start and finish.

## Request Generation

- The simulator uses a fixed pool of `users`. Each user can only have **one active request at a time** and only one active session at a time.
- After a user's request finishes, that user enters a think time (`THINK_TIME_MS`) before it can generate the next request. The simulator only generates a new request when a user is idle and its think time has elapsed.
- A user's session is rolled over to the next `session_id` only after the previous session has reached `max_session_turns` **and** the user is idle.
- Input sequence length within a session grows cumulatively: each new request starts from the maximum `isl + osl` seen so far in that `(user, session)`.
- `total_requests` is derived from `users * sessions_per_user * max_session_turns`; it is no longer a standalone input.
- If `users >= total_requests`, every request gets a unique user, which disables shared-prefix caching across requests.
- `RequestGenerator` tracks active/idle users, per-user session IDs, turn counts, and cached last total tokens. The simulator calls `start_request()` when a request is generated and `finish_request(request, now_ms)` when it completes.
- Do not change `request_id_counter` behavior unless explicitly asked; it is module-level global state used by `Request`.

## Decode Batch Model

- Decode runs in frozen batches of exactly one token.
- `DecodeInstance` tracks instance-level state (`current_batch`, `remaining_batch_time_ms`, `current_batch_decode_time_ms`) instead of per-request timers.
- The batch is formed from the head of the queue and frozen until one token is decoded for every active request.
- After the token, finished requests are removed and trigger KV upload; the batch is then reformed (adding any newly arrived requests) for the next token.
- Partial progress within a token is banked instance-side; if a smaller transfer/arrival event occurs, the decode timer is decremented but no token completes.
- When a token completes, the per-token decode time is recalculated because the average ISL in the batch has grown by one.
- `DecodeInstance` keeps a maintained `_kv_cache_bytes` counter that must be updated at every queue mutation (`add_request`, download drain, finished-request removal).

## Common Pitfalls

- Bandwidth fields in `HardwareSpec` are stored as **bytes/second**. The machine DB loader already converts from Mbit/s.
- `Model.kv_size_per_token` returns bytes per token; multiply by token count to get KV bytes.
- `HardwareSpec` uses `pcie_bw` for local RAM bandwidth, not `ram_bw`.
- The simulation event loop advances by the minimum of the next compute event, the next transfer event, and the next request arrival.
- `request_id_counter` is module-level global state in `src/request/request.py`.

## Testing

A pytest suite lives in `tests/`.

```bash
.venv/bin/python -m pytest tests/        # all tests
.venv/bin/python -m pytest tests/ -v     # verbose
.venv/bin/python main.py --sessions-per-user 1 --isl 128 --osl 8 --users 4 --max-session-turns 1  # smoke test
```

Covered areas:
- `tests/test_logger.py` — bitmask logging behavior.
- `tests/test_bandwidth_scheduler.py` — equal-share scheduling for RAM/SSD/NETWORK bottlenecks.
- `tests/test_cache.py` — two-tier cache, LRU eviction, capacity validation, and transfer-leg generation.
- `tests/test_request.py` — `TransferLeg`, `DownloadRequest`, `UploadRequest`, and `Request` basics.
- `tests/test_decode.py` — frozen one-token decode batches and instance-level partial progress.
- `tests/test_analytics.py` — phase-level timing analytics.
- `tests/test_sla.py` — per-request TTFT/TPOT SLA enforcement.
- `tests/test_router.py` — routing and cost scoring.
- `tests/test_execute_config.py` / `tests/test_scraper.py` — config-execution utilities.

## Batch runner (`execute_config.py`)

`execute_config.py --config config.json [--output results.json] [--timeout T]` runs a matrix of scenarios in parallel.

- Each valid config is simulated in a separate process (default `max_workers=8`).
- A **per-config timeout** (`--timeout` seconds, default `120.0`) is enforced with
  `concurrent.futures.wait`.  Configs that do not finish before their deadline are
  cancelled and reported as `timed out after {T}s`.
- Fast failures (e.g. `PrefillError`) still terminate immediately so dependent
  configs can be invalidated.
- The config file may also set `"timeout_s": 300.0`; the CLI flag overrides it.

## Contact / Ownership

This is a research simulator. When in doubt, keep the model simple, add focused logging, and avoid silent failures.
