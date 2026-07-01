# Agent Guide for the Distributed LLM Inference Simulator

This file contains the project-specific context and conventions needed by coding agents when working on this simulator.

## Project Overview

This repository implements a discrete-event simulator for disaggregated (prefill/decode) LLM inference with distributed prefix KV caching. The main entry points are:

- `main.py` — command-line simulator.
- `src/webserver/server.py` — FastAPI server exposing the same simulator.

## Architecture

```
main.py / webserver/server.py
    │
    ▼
src/simulations/simulation_distributed.py
    │
    ├── Router ........... src/router/router.py
    ├── PrefillInstance .. src/instances/prefill.py
    ├── DecodeInstance ... src/instances/decode.py
    ├── Cache ............ src/cache/cache.py
    └── BandwidthScheduler src/scheduler/bandwidth_scheduler.py
```

Other important pieces:

- `src/request/request.py` — `Request`, `RequestScenario`, `TransferLeg`, `DownloadRequest`, `UploadRequest`.
- `src/hardware/hardware.py` — `Hardware`, `HardwareSpec`, `GPUHardwareSpec` loaded from `_machine_db.json`.
- `src/model/model.py` — thin wrapper around the HF model name; provides `kv_size_per_token`.
- `src/logger.py` — bitmask-based logging.
- `src/utils/env_reader.py` — `.env` loader.
- `src/result.py` — `SimulationResult` dataclass used by both CLI and webserver.

## Build / Run

No build step is required. The project uses a local `.venv`.

```bash
# CLI
.venv/bin/python main.py

# Web server
.venv/bin/python -m uvicorn src.webserver.server:app --reload
```

Use `.venv/bin/python` for all Python commands.

## Configuration

Configuration is read from `.env` at the project root and can be overridden by shell environment variables or CLI flags.

Relevant `.env` entries:

```bash
MODEL=Qwen/Qwen3-8B
ISL=1000
OSL=100
REQUESTS=10
REQ_RATE=2.0
BATCH_SIZE=10
PREFILL_WORKERS=1
DECODE_WORKERS=1
CACHE_PCT=0.0
RAM_USAGE_FRACTION=0.8
SSD_USAGE_FRACTION=0.8
LOG_MASK=15
DEBUG=false
```

### Logging bitmask (`LOG_MASK`)

`LOG_MASK` is an integer built by OR-ing component bits:

| Bit | Value | Component |
|-----|-------|-----------|
| 0   | 1     | Cache     |
| 1   | 2     | Instances (prefill/decode) |
| 2   | 4     | Router    |
| 3   | 8     | Simulation |

Examples: `0` = nothing, `1` = cache only, `4` = router only, `15` = everything.

Use `src.logger.set_log_mask()` or `--log-mask` to change it at runtime.

## Coding Conventions

- Python 3.12+ syntax is fine; type hints are encouraged.
- Keep changes minimal and focused on the requested goal.
- Do not change existing test logic when refactoring.
- When adding new CLI/env parameters, mirror them in:
  - `.env`
  - `src/utils/env_reader.py` (`EnvConfig` and `_DEFAULTS`)
  - `main.py` argument parser
  - `src/webserver/server.py` if it exposes the parameter
- When adding new log points, use `from src.logger import LOG_*, log` and call `log(LOG_*, msg)`.
  - Do not use the legacy `debug_print()` in new code.

## Cache Model

- Every node has two cache tiers: `RAM` and `SSD`.
- Capacities are derived from `HardwareSpec.ram_mem` and `nvme_mem`, multiplied by `ram_usage_fraction` / `ssd_usage_fraction`.
- New KV chunks are inserted into RAM first.
- When RAM is full, the least-recently-used (LRU) item is evicted to SSD.
- When SSD is full, its LRU item is deleted permanently.
- The cache raises `ValueError` at construction time if either tier cannot hold a 512-token KV item.
- Remote reads from SSD generate a sequential `SSD_LOCAL → RAM_LOCAL → NETWORK → RAM_LOCAL` leg chain.

## Bandwidth Model

- Bandwidth is scheduled globally by `BandwidthScheduler` using equal-share fairness.
- Three independent bottlenecks:
  - `RAM_LOCAL` shares the node's `ram_bw`.
  - `SSD_LOCAL` shares the node's `nvme_bw`.
  - `NETWORK` uses the minimum of the source's `network_inet_up` share and the destination's `network_inet_down` share.
- `DownloadRequest` / `UploadRequest` contain sequential `TransferLeg`s. Only the active leg receives bandwidth.
- Instances register/unregister legs with the scheduler as they start and finish.

## Decode Batch Model

- Decode runs in frozen batches of exactly one token.
- `DecodeInstance` tracks instance-level state (`current_batch`, `remaining_batch_time_ms`, `current_batch_decode_time_ms`) instead of per-request timers.
- The batch is formed from the head of the queue and frozen until one token is decoded for every active request.
- After the token, finished requests are removed and trigger KV upload; the batch is then reformed (adding any newly arrived requests) for the next token.
- Partial progress within a token is banked instance-side; if a smaller transfer/arrival event occurs, the decode timer is decremented but no token completes.
- When a token completes, the per-token decode time is recalculated because the average ISL in the batch has grown by one.

## Common Pitfalls

- `HardwareSpec.ram_bw` and `nvme_bw` are stored in bytes/second.
- `HardwareSpec.network_inet_up` / `network_inet_down` are also bytes/second (the loader already converts from Mbit/s in the JSON database).
- `Model.kv_size_per_token` returns bytes per token; multiply by token count to get KV bytes.
- The simulation event loop advances by the minimum of the next compute event, the next transfer event, and the next request arrival.
- Do not change `request_id_counter` behavior unless explicitly asked; it is module-level global state used by `Request`.

## Testing

A pytest suite lives in `tests/`.

```bash
# Run all unit tests
.venv/bin/python -m pytest tests/

# Run with verbose output
.venv/bin/python -m pytest tests/ -v

# Quick simulator smoke test
.venv/bin/python main.py --requests 4 --isl 128 --osl 8 --req-rate 10 --cache-pct 0.5

# Module import check
.venv/bin/python -m py_compile main.py src/**/*.py src/webserver/server.py tests/*.py
```

The test suite covers:

- `tests/test_logger.py` — bitmask logging behavior.
- `tests/test_bandwidth_scheduler.py` — equal-share scheduling for RAM/SSD/NETWORK bottlenecks.
- `tests/test_cache.py` — two-tier cache, LRU eviction, capacity validation, and transfer-leg generation.
- `tests/test_request.py` — `TransferLeg`, `DownloadRequest`, `UploadRequest`, and `Request` basics.
- `tests/test_decode.py` — frozen one-token decode batches and instance-level partial progress.

## Contact / Ownership

This is a research simulator. When in doubt, keep the model simple, add focused logging, and avoid silent failures.
