# Distributed LLM Inference Simulator

This repository is a discrete-event simulator for **disaggregated (prefill/decode) LLM inference** with distributed prefix KV caching. It models how requests flow through prefill and decode nodes, how KV caches are placed across RAM, SSD, and a shared S3/object-store tier, and how a router decides where to send each request based on locality, load, and cost.

The simulator is primarily used to compare hardware topologies, batch sizes, cache sizes, and routing policies for long-context workloads where prefix reuse across user sessions matters.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Repository layout](#repository-layout)
3. [Scripts and entry points](#scripts-and-entry-points)
   - [`main.py`](#mainpy)
   - [`configs/create_user_sweep_config.py`](#configscreate_user_sweep_configpy)
   - [`configs/execute_user_sweep_config.py`](#configsexecute_user_sweep_configpy)
   - [`configs/create_hardware_economics_config.py`](#configscreate_hardware_economics_configpy)
   - [`configs/execute_hardware_economics_config.py`](#configsexecute_hardware_economics_configpy)
   - [`src/webserver/server.py`](#srcwebserverserverpy)
   - [`scripts/add_custom_machine.py`](#scriptsadd_custom_machinepy)
   - [`scripts/generate_focused_machines.py`](#scriptsgenerate_focused_machinespy)
   - [`scripts/derive_family_pricing.py`](#scriptsderive_family_pricingpy)
   - [`scripts/compare_with_nvidia.py`](#scriptscompare_with_nvidiapy)
   - [`scripts/kv_break_even.py`](#scriptskv_break_evenpy)
   - [`scripts/kv_storage_cost.py`](#scriptskv_storage_costpy)
4. [Configuration](#configuration)
   - [`.env` defaults](#env-defaults)
   - [Hardware presets](#hardware-presets)
   - [Example `config.json`](#example-configjson)
5. [Development](#development)
   - [Tests](#tests)
   - [Linting](#linting)
6. [Model and cache assumptions](#model-and-cache-assumptions)
7. [License](#license)

---

## Quick start

The project uses a local virtual environment under `.venv`.

```bash
# Run one scenario from the .env defaults
.venv/bin/python main.py

# Override a few parameters on the CLI
.venv/bin/python main.py \
  --model Qwen/Qwen3-8B \
  --isl 1000 --osl 100 \
  --sessions-per-user 1 --users 10 \
  --num-prefill-nodes 1 --num-decode-nodes 1

# Generate and run a swept config matrix
.venv/bin/python configs/create_user_sweep_config.py --config-name config.json
.venv/bin/python configs/execute_user_sweep_config.py --config config.json --output results.json

# Start the FastAPI web server
.venv/bin/python -m uvicorn src.webserver.server:app --reload

# Run the test suite
.venv/bin/python -m pytest tests/
```

---

## Repository layout

```
.
├── .env                          # Default simulator parameters
├── main.py                       # CLI entry point for single simulations
├── configs/create_user_sweep_config.py      # Generate swept config.json matrices
├── configs/execute_user_sweep_config.py     # Run config matrices in parallel
├── configs/create_hardware_economics_config.py  # Generate fixed-topology hardware-economics config JSON
├── configs/execute_hardware_economics_config.py # Run hardware-economics sweeps into grouped results files
├── config.json                   # Example generated config matrix
├── src/
│   ├── cache/cache.py            # Two-tier RAM/SSD/S3 KV cache
│   ├── hardware/                 # Machine / GPU spec databases
│   ├── instances/                # Prefill and decode instance models
│   ├── request/request.py        # Request generator and transfer legs
│   ├── router/router.py          # Locality-aware routing
│   ├── scheduler/                # Global bandwidth scheduler
│   ├── simulations/              # Distributed simulation engine
│   ├── utils/                    # Env parsing, CLI parsers, output filters
│   └── webserver/server.py       # FastAPI server + HTML UI
├── scripts/                      # Utility scripts for hardware DBs
└── tests/                        # pytest suite
```

---

## Scripts and entry points

### `main.py`

Run a single distributed simulation from the command line. It reads defaults from `.env` and accepts overrides for model, sequence lengths, topology, cache, routing, and SLA parameters.

Key CLI flags:

| Flag | Description |
|------|-------------|
| `--model` | HuggingFace model name (e.g. `Qwen/Qwen3-8B`) |
| `--isl` / `--osl` | Fixed input / output sequence length |
| `--users` / `--sessions-per-user` / `--max-session-turns` | User pool and session shape |
| `--think-time-ms` | Idle time between a user's requests |
| `--num-prefill-nodes` / `--num-decode-nodes` | Topology size |
| `--colocated` | Share nodes between prefill and decode |
| `--machine-hardware` | Hardware preset from the machine DB |
| `--batch-size` | Decode batch size |
| `--ram-usage-fraction` / `--ssd-usage-fraction` | Cache capacity fractions |
| `--s3-enabled` / `--s3-*` | Shared S3 tier settings |
| `--router-*` | Router cost credits |
| `--log-mask` | Component logging bitmask |
| `--debug` | Enable all logging |

Example:

```bash
.venv/bin/python main.py \
  --model Qwen/Qwen3-8B \
  --isl 30000 --osl 2000 \
  --sessions-per-user 20 --users 40 --max-session-turns 7 \
  --num-prefill-nodes 2 --num-decode-nodes 31 \
  --batch-size 100 \
  --machine-hardware "AWS p5en.48xlarge (H200 x8)" \
  --ram-usage-fraction 0.1 --ssd-usage-fraction 0.1 \
  --s3-enabled --s3-up-bw-gbps 25.0 --s3-down-bw-gbps 25.0
```

The output is a compact JSON `SimulationResult` plus human-readable breakdowns of request shape, cost, and S3 counters.

### `configs/create_user_sweep_config.py`

Generate a `config.json` that sweeps over hardware topologies. It mirrors the CLI flags of `main.py` and emits one entry per combination of machine, node count, GPU split, and batch size.

Supported topology categories (`--config-types`):

- `colocated` — prefill and decode share the same nodes.
- `mixed` — colocated nodes with different GPU types for prefill and decode.
- `separate` — distinct prefill-only and decode-only nodes.

Example:

```bash
.venv/bin/python configs/create_user_sweep_config.py \
  --model Qwen/Qwen3-8B \
  --isl 30000 --osl 2000 \
  --sessions-per-user 20 --users 40 \
  --config-types colocated,separate \
  --high-end-only \
  --config-name config.json
```

### `configs/execute_user_sweep_config.py`

Run a generated `config.json` in parallel. Each valid config is executed in a separate process with a configurable per-config timeout. Failed configs can invalidate related smaller/larger configs in the sweep so the matrix terminates early.

```bash
.venv/bin/python configs/execute_user_sweep_config.py \
  --config config.json \
  --output results.json \
  --timeout 120.0
```

The output JSON contains one `SimulationResult` per config with metadata such as label, color, and user count, suitable for plotting or importing back into the webserver.

### `configs/create_hardware_economics_config.py`

Generate a fixed-topology config for the hardware-economics sweep. It keeps the deployment mode constant and sweeps TTFT SLA and user-delay values.

### `configs/execute_hardware_economics_config.py`

Run a hardware-economics config in parallel, searching for the maximum users each fixed topology can serve while meeting SLAs. Results are grouped into `results_<focus>_<value>.json` files suitable for importing into the webserver.

### `src/webserver/server.py`

A FastAPI server exposing the same simulator through `simulate_run_distributed`. It provides:

- Form-based submission at `/`.
- JSON execution endpoint `/simulate`.
- Plot endpoints for result visualization.
- Import/export of config matrices and results.

Run it with:

```bash
.venv/bin/python -m uvicorn src.webserver.server:app --reload
```

### `scripts/add_custom_machine.py`

Add custom machine presets to the local hardware database with derived pricing. It computes the Cartesian product of comma-separated options and writes to `src/hardware/custom_hardware.json` (or `--custom-hardware`).

```bash
.venv/bin/python scripts/add_custom_machine.py "My H200" H200 \
  --num-gpus 1,2,4 \
  --pcie-bw-gbps 128,256 \
  --nvlink-bw-gbps 0,900 \
  --ram-mem-gb 256,512 \
  --ssd-mem-gb 2048,4096 \
  --ssd-bw-gbps 12.8,25 \
  --inet-bw-gbps 25 \
  --write
```

### `scripts/generate_focused_machines.py`

Generate a focused sensitivity-analysis matrix from a baseline GPU. It emits one machine per focused dimension (RAM, NVLink bandwidth, SSD memory, SSD bandwidth), keeping all other dimensions constant.

```bash
.venv/bin/python scripts/generate_focused_machines.py H200 \
  --num-gpus 4 \
  --pcie-bw-gbps 128 \
  --nvlink-bw-gbps 800 \
  --ram-mem-gb 512 \
  --ssd-mem-gb 2048 \
  --ssd-bw-gbps 12.8 \
  --inet-bw-gbps 25 \
  --focus ram,nvlink,ssd_mem,ssd_bw \
  --write --clean
```

### `scripts/derive_family_pricing.py`

Derive per-GPU-family component prices from AWS instance configs by interpolating across pairs of instances where only one resource dimension differs. Updates `src/hardware/aws_hardware.json` in place.

```bash
.venv/bin/python scripts/derive_family_pricing.py
```

### `scripts/compare_with_nvidia.py`

Run the same scenario through both this simulator and the NVIDIA AI Configurator estimate API, then print a side-by-side comparison.

```bash
.venv/bin/python scripts/compare_with_nvidia.py \
  --isl 1000 --osl 100 \
  --sessions-per-user 1 --users 10 \
  --model Qwen/Qwen3-8B
```

### `scripts/kv_break_even.py`

Quick standalone calculation of the KV-cache break-even point: when does prefix reuse justify the storage cost over re-computing the prefix every time? Useful for sanity checks before running a full simulation.

### `scripts/kv_storage_cost.py`

Standalone cost model for KV cache storage across RAM, SSD, and S3, using the pricing metadata from the hardware database.

---

## Configuration

### `.env` defaults

The file `.env` at the project root provides defaults for all simulator parameters. Any value can be overridden by a shell environment variable or a CLI flag. The canonical list lives in `src/utils/env_reader.py`.

Current defaults in this checkout:

```bash
# Model and request shape
MODEL=Qwen/Qwen3-8B
ISL=30000
OSL=2000

# Scenario scale
SESSIONS_PER_USER=20
USERS=40
MAX_SESSION_TURNS=7
THINK_TIME_MS=1000

# Per-user random delay
USER_DELAY_FRACTION=0.0
USER_DELAY_MIN_MS=360000000
USER_DELAY_MAX_MS=360000000

# Reproducibility
RANDOM_SEED=42

# Per-request latency SLAs. Must be finite positive numbers because the
# request generator builds a deterministic arrival schedule from them.
SLA_TTFT_MS=30000
SLA_TPOT_MS=100

# Topology
BATCH_SIZE=100
NUM_PREFILL_NODES=2
NUM_DECODE_NODES=31
COLOCATED=false
PREFILL_GPUS_PER_NODE=-1
MACHINE_HARDWARE=AWS p5en.48xlarge (H200 x8)

# Mixed-GPU topology
MIXED=false
MIXED_GPU_DONOR=
MIXED_GPU_COUNT=-1

# GPU compute/bandwidth split; must match the split in pricing.json
GPU_COMPUTE_FRACTION=0.6

# Cache
RAM_USAGE_FRACTION=0.1
SSD_USAGE_FRACTION=0.1

# Shared S3/object-store fallback
S3_ENABLED=true
S3_UP_BW_GBPS=25.0
S3_DOWN_BW_GBPS=25.0
S3_EVICTION_TIME_MS=3600000000

# Inter-node KV-transfer bandwidth
INTER_NODE_NETWORK_UP_GBPS=100.0
INTER_NODE_NETWORK_DOWN_GBPS=100.0

# Router cost parameters
ROUTER_PREFILL_LOAD_SCALE=1.0
ROUTER_DEVICE_CREDIT=0.8
ROUTER_REMOTE_RAM_CREDIT=0.5
ROUTER_REMOTE_SSD_CREDIT=0.3
ROUTER_S3_CREDIT=0.1

# Logging
LOG_MASK=63
DEBUG=true
```

### Hardware presets

Machine presets are loaded from a combined database (`src/hardware/aws_hardware.json`, `src/hardware/custom_hardware.json`, and the legacy `src/hardware/legacy/_machine_db.json`). They are grouped below by source.

#### AWS presets (62)

- `AWS g4dn.12xlarge (TESLA_T4 x4)`
- `AWS g4dn.16xlarge (TESLA_T4 x1)`
- `AWS g4dn.2xlarge (TESLA_T4 x1)`
- `AWS g4dn.4xlarge (TESLA_T4 x1)`
- `AWS g4dn.8xlarge (TESLA_T4 x1)`
- `AWS g4dn.metal (TESLA_T4 x8)`
- `AWS g4dn.xlarge (TESLA_T4 x1)`
- `AWS g5.12xlarge (A10G x4)`
- `AWS g5.16xlarge (A10G x1)`
- `AWS g5.24xlarge (A10G x4)`
- `AWS g5.2xlarge (A10G x1)`
- `AWS g5.48xlarge (A10G x8)`
- `AWS g5.4xlarge (A10G x1)`
- `AWS g5.8xlarge (A10G x1)`
- `AWS g5.xlarge (A10G x1)`
- `AWS g6.12xlarge (L4 x4)`
- `AWS g6.16xlarge (L4 x1)`
- `AWS g6.24xlarge (L4 x4)`
- `AWS g6.2xlarge (L4 x1)`
- `AWS g6.48xlarge (L4 x8)`
- `AWS g6.4xlarge (L4 x1)`
- `AWS g6.8xlarge (L4 x1)`
- `AWS g6.xlarge (L4 x1)`
- `AWS g6e.12xlarge (L40S x4)`
- `AWS g6e.16xlarge (L40S x1)`
- `AWS g6e.24xlarge (L40S x4)`
- `AWS g6e.2xlarge (L40S x1)`
- `AWS g6e.48xlarge (L40S x8)`
- `AWS g6e.4xlarge (L40S x1)`
- `AWS g6e.8xlarge (L40S x1)`
- `AWS g6e.xlarge (L40S x1)`
- `AWS g7.12xlarge (RTX_PRO_4500 x2)`
- `AWS g7.24xlarge (RTX_PRO_4500 x4)`
- `AWS g7.2xlarge (RTX_PRO_4500 x1)`
- `AWS g7.48xlarge (RTX_PRO_4500 x8)`
- `AWS g7.4xlarge (RTX_PRO_4500 x1)`
- `AWS g7.8xlarge (RTX_PRO_4500 x1)`
- `AWS g7e.24xlarge (GB202 x4)`
- `AWS g7e.2xlarge (GB202 x1)`
- `AWS g7e.48xlarge (GB202 x8)`
- `AWS g7e.4xlarge (GB202 x1)`
- `AWS g7e.8xlarge (GB202 x1)`
- `AWS inf1.24xlarge (INF1 x16)`
- `AWS inf1.2xlarge (INF1 x1)`
- `AWS inf1.6xlarge (INF1 x4)`
- `AWS inf1.xlarge (INF1 x1)`
- `AWS inf2.24xlarge (INF2 x6)`
- `AWS inf2.48xlarge (INF2 x12)`
- `AWS inf2.8xlarge (INF2 x2)`
- `AWS inf2.xlarge (INF2 x1)`
- `AWS p3.16xlarge (Tesla V100 x8)`
- `AWS p3.2xlarge (Tesla V100 x1)`
- `AWS p3.8xlarge (Tesla V100 x4)`
- `AWS p4d.24xlarge (A100 40GB x8)`
- `AWS p4de.24xlarge (A100 80GB x8)`
- `AWS p5.24xlarge (H100 NVL x4)`
- `AWS p5.48xlarge (H100 NVL x8)`
- `AWS p5.4xlarge (H100 NVL x1)`
- `AWS p5en.48xlarge (H200 NVL x8)`
- `AWS p5en.48xlarge (H200 x8)`
- `AWS p6-b200.48xlarge (B200 x8)`
- `AWS p6-b300.48xlarge (B300 x8)`

#### Custom focused presets (13)

- `Focused H200 NVLink 0Gbps x4 r512 s2048 p128 sbw12.8 inet25.0/25.0`
- `Focused H200 NVLink 1600Gbps x4 r512 s2048 p128 nvl1600 sbw12.8 inet25.0/25.0`
- `Focused H200 NVLink 400Gbps x4 r512 s2048 p128 nvl400 sbw12.8 inet25.0/25.0`
- `Focused H200 RAM 1024GB x4 r1024 s2048 p128 nvl800 sbw12.8 inet25.0/25.0`
- `Focused H200 RAM 2048GB x4 r2048 s2048 p128 nvl800 sbw12.8 inet25.0/25.0`
- `Focused H200 RAM 256GB x4 r256 s2048 p128 nvl800 sbw12.8 inet25.0/25.0`
- `Focused H200 SSD 1024GB x4 r512 s1024 p128 nvl800 sbw12.8 inet25.0/25.0`
- `Focused H200 SSD 4096GB x4 r512 s4096 p128 nvl800 sbw12.8 inet25.0/25.0`
- `Focused H200 SSD 8192GB x4 r512 s8192 p128 nvl800 sbw12.8 inet25.0/25.0`
- `Focused H200 SSD BW 25.0Gbps x4 r512 s2048 p128 nvl800 sbw25.0 inet25.0/25.0`
- `Focused H200 SSD BW 50.0Gbps x4 r512 s2048 p128 nvl800 sbw50.0 inet25.0/25.0`
- `Focused H200 SSD BW 6.4Gbps x4 r512 s2048 p128 nvl800 sbw6.4 inet25.0/25.0`
- `Focused H200 x4 r512 s2048 p128 nvl800 sbw12.8 inet25.0/25.0`

#### Vast.ai legacy presets (64)

- `B200 x1 #b1ee2f4f`
- `B200 x8 #15825275`
- `H100 NVL x1 #c59a6e27`
- `H100 SXM x2 #e449f99f`
- `H200 NVL x1 #51040666`
- `H200 NVL x2 #484c7942`
- `H200 NVL x8 #527a5cb7`
- `H200 x1 #b731cab8`
- `H200 x1 #cc151216`
- `H200 x2 #9a70630b`
- `H200 x4 #0a43645c`
- `RTX 3080 x1 #20ac03e0`
- `RTX 3090 x1 #e40649b0`
- `RTX 4070S Ti x1 #06f020ea`
- `RTX 4080S x1 #060e23f1`
- `RTX 4080S x1 #396b71df`
- `RTX 4090 x1 #7ce276d0`
- `RTX 4090 x1 #8b3af580`
- `RTX 4090 x1 #e14e3bd2`
- `RTX 4090 x2 #314fce06`
- `RTX 4090 x2 #8706becf`
- `RTX 4090 x2 #a580383c`
- `RTX 4090 x4 #6ad4ac65`
- `RTX 4090D x1 #a7678a67`
- `RTX 4090D x2 #2c84d969`
- `RTX 5060 Ti x1 #ffa2e256`
- `RTX 5070 Ti x1 #fc0b6893`
- `RTX 5070 Ti x2 #5039e780`
- `RTX 5070 x1 #39613ceb`
- `RTX 5080 x1 #6bd50f64`
- `RTX 5080 x1 #d10d0df1`
- `RTX 5080 x2 #187b0ad5`
- `RTX 5080 x2 #bd87c28e`
- `RTX 5090 x1 #02180525`
- `RTX 5090 x1 #02749557`
- `RTX 5090 x1 #221e2e40`
- `RTX 5090 x1 #cfe521b3`
- `RTX 5090 x1 #d24fce61`
- `RTX 5090 x1 #e172a2fe`
- `RTX 5090 x1 #f1380001`
- `RTX 5090 x2 #0fdde6eb`
- `RTX 5090 x2 #3fbe8016`
- `RTX 5090 x2 #75764919`
- `RTX 5090 x2 #97fddd29`
- `RTX 5090 x2 #f05e88d0`
- `RTX 5090 x4 #f4e7e6ec`
- `RTX PRO 4500 x1 #0d2aeea3`
- `RTX PRO 4500 x1 #3d545ce7`
- `RTX PRO 5000 x2 #5dd9d3a4`
- `RTX PRO 5000 x2 #c8cd89ea`
- `RTX PRO 6000 S x1 #461bc043`
- `RTX PRO 6000 S x1 #5d8ce858`
- `RTX PRO 6000 S x1 #606aae61`
- `RTX PRO 6000 S x2 #32f8c665`
- `RTX PRO 6000 S x2 #d1a2bde0`
- `RTX PRO 6000 WS x1 #2fda3f22`
- `RTX PRO 6000 WS x1 #3dd810d5`
- `RTX PRO 6000 WS x1 #b046a9d9`
- `RTX PRO 6000 WS x2 #0e41bd7c`
- `RTX PRO 6000 WS x2 #ae320bba`
- `Tesla V100 x1 #99ac69f3`
- `Tesla V100 x2 #4281293d`
- `Tesla V100 x4 #c6865551`
- `Tesla V100 x8 #63755d58`

### Example `config.json`

The following example matches the current `.env` defaults and sweeps over a few node counts and batch sizes on AWS H200 hardware. Save it as `config.json` and run with `configs/execute_user_sweep_config.py`.

```json
{
  "model": "Qwen/Qwen3-8B",
  "isl": 30000,
  "osl": 2000,
  "sessions_per_user": 20,
  "users": 40,
  "max_session_turns": 7,
  "think_time_ms": 1000,
  "ram_usage_fraction": 0.1,
  "ssd_usage_fraction": 0.1,
  "router_prefill_load_scale": 1.0,
  "router_device_credit": 0.8,
  "router_remote_ram_credit": 0.5,
  "router_remote_ssd_credit": 0.3,
  "router_s3_credit": 0.1,
  "s3_enabled": true,
  "s3_up_bw_gbps": 25.0,
  "s3_down_bw_gbps": 25.0,
  "s3_eviction_time_ms": 3600000000,
  "inter_node_network_up_gbps": 100.0,
  "inter_node_network_down_gbps": 100.0,
  "sla": {
    "ttft_ms": 30000,
    "tpot_ms": 100
  },
  "user_delay_fraction": 0.0,
  "user_delay_min_ms": 360000000.0,
  "user_delay_max_ms": 360000000.0,
  "random_seed": 42,
  "configs": [
    {
      "label": "Separate: AWS p5en.48xlarge (H200 x8) - 2 prefill / 31 decode - batch 100",
      "prefill_hardware": "AWS p5en.48xlarge (H200 x8)",
      "decode_hardware": "AWS p5en.48xlarge (H200 x8)",
      "prefill_nodes": 2,
      "decode_nodes": 31,
      "batch_size": 100,
      "colocated": false
    },
    {
      "label": "Separate: AWS p5en.48xlarge (H200 x8) - 2 prefill / 16 decode - batch 128",
      "prefill_hardware": "AWS p5en.48xlarge (H200 x8)",
      "decode_hardware": "AWS p5en.48xlarge (H200 x8)",
      "prefill_nodes": 2,
      "decode_nodes": 16,
      "batch_size": 128,
      "colocated": false
    },
    {
      "label": "Colocated: AWS p5en.48xlarge (H200 x8) - 8 nodes 4p+4d - batch 100",
      "prefill_hardware": "AWS p5en.48xlarge (H200 x8)",
      "decode_hardware": "AWS p5en.48xlarge (H200 x8)",
      "prefill_nodes": 8,
      "decode_nodes": 8,
      "prefill_gpus_per_node": 4,
      "decode_gpus_per_node": 4,
      "batch_size": 100,
      "colocated": true
    }
  ]
}
```

Run it:

```bash
.venv/bin/python configs/execute_user_sweep_config.py --config config.json --output results.json --timeout 120.0
```

---

## Development

### Tests

```bash
.venv/bin/python -m pytest tests/        # all tests
.venv/bin/python -m pytest tests/ -v     # verbose
.venv/bin/python main.py --sessions-per-user 1 --isl 128 --osl 8 --users 4 --max-session-turns 1
```

### Linting

The project uses `ruff`. Configuration is in `ruff.toml`.

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format .
```

---

## Model and cache assumptions

- **Model**: `Model` wraps a HuggingFace model name and provides `kv_size_per_token` in bytes. The FLOPs and memory model are analytical.
- **Request shape**: ISL and OSL are fixed in a given run. Total requests = `users * sessions_per_user * max_session_turns`. Within a session, ISL grows cumulatively from the previous turn's `isl + osl`.
- **Cache tiers**: Each node has RAM and SSD. The shared S3 tier is optional. Items are merged per `(user_id, session_id)` to keep one contiguous entry per tier. Eviction follows LRU within each tier.
- **Bandwidth**: The global `BandwidthScheduler` applies equal-share fairness across `RAM_LOCAL`, `SSD_LOCAL`, `NETWORK`, `S3_UPLOAD`, and `S3_DOWNLOAD` legs.
- **Router**: Picks the cheapest prefill/decode worker using a cost model that credits local, remote RAM, remote SSD, and S3 KV hits, plus load. Ties are broken deterministically using the configured random seed.

---

## License

This is a research simulator. See the repository license file for details.
