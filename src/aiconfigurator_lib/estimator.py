"""Estimator wrapper around NVIDIA aiconfigurator.

Provides both:
1. High-level ``estimate_latency()`` — flat dict with corrected TTFT/TPOT/seq/s.
2. Low-level ``run_static_inference()`` — raw per-op latency + memory breakdown.

The low-level path lets you build your own simulator logic (KV transfer, queuing,
etc.) on top of NVIDIA's compute latency math.
"""

from __future__ import annotations
import logging
import os
import tempfile

from pathlib import Path
from typing import Any

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.config import ModelConfig
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import BaseModel, get_model
from aiconfigurator.sdk.perf_database import (
    PerfDatabase,
    get_database,
    set_systems_paths,
)


logger = logging.getLogger(__name__)


def _create_custom_system_dir(
    system_name: str,
    mem_bw: float,
    mem_capacity: float,
    bfloat16_tc_flops: float,
    sm_version: int = 90,
    num_gpus_per_node: int = 1,
    intra_node_bw: float = 450_000_000_000,
    inter_node_bw: float = 50_000_000_000,
    pcie_bw: float = 64_000_000_000,
    p2p_latency: float = 0.00001,
    power: float = 700,
    mem_bw_scaling: float = 0.8,
    mem_constant_latency: float = 0.000003,
) -> str:
    """Create a temporary systems directory with a YAML + empty perf files for SOL."""
    tmpdir = tempfile.mkdtemp(prefix="aic_systems_")
    systems_dir = Path(tmpdir, "systems")
    systems_dir.mkdir(parents=True, exist_ok=True)

    yaml_content = f"""data_dir: data/{system_name}
gpu:
  mem_bw: {int(mem_bw)}
  mem_bw_empirical_scaling_factor: {mem_bw_scaling:.6f}
  mem_empirical_constant_latency: {mem_constant_latency:.10f}
  mem_capacity: {int(mem_capacity)}
  bfloat16_tc_flops: {int(bfloat16_tc_flops)}
  int8_tc_flops: {int(bfloat16_tc_flops * 2)}
  fp8_tc_flops: {int(bfloat16_tc_flops * 2)}
  power: {int(power)}
  sm_version: {sm_version}
node:
  num_gpus_per_node: {num_gpus_per_node}
  inter_node_bw: {int(inter_node_bw)}
  intra_node_bw: {int(intra_node_bw)}
  pcie_bw: {int(pcie_bw)}
  p2p_latency: {p2p_latency}
misc:
  nccl_mem:
    1: 0
    2: 358612992
    4: 411041792
    8: 411041792
  other_mem: 3758096384
  nccl_version: '2.26.2'
"""
    with Path.open(Path(systems_dir, f"{system_name}.yaml"), "w") as f:
        f.write(yaml_content)

    data_path = Path(systems_dir, "data", system_name, "vllm", "sol")
    Path(data_path).mkdir(parents=True, exist_ok=True)
    for fname in [
        "gemm_perf.txt",
        "context_attention_perf.txt",
        "generation_attention_perf.txt",
        "moe_perf.txt",
        "custom_allreduce_perf.txt",
        "context_mla_perf.txt",
        "generation_mla_perf.txt",
    ]:
        Path.open(Path(data_path, fname), "a").close()

    return systems_dir.as_posix()


def create_database_for_hardware(
    systems_dir: str,
    system_name: str,
    *,
    mem_bw: float,
    mem_capacity: float,
    bfloat16_tc_flops: float,
    sm_version: int = 90,
    num_gpus_per_node: int = 1,
    intra_node_bw: float = 450_000_000_000,
    inter_node_bw: float = 50_000_000_000,
    pcie_bw: float = 64_000_000_000,
    p2p_latency: float = 0.00001,
    power: float = 700,
    mem_bw_scaling: float = 0.8,
    mem_constant_latency: float = 0.000003,
) -> str:
    """Create a **persistent** systems directory for a custom hardware spec.

    Unlike ``_create_custom_system_dir`` (used internally for one-shot
    estimates), this writes into *systems_dir* so you can reuse the same
    database across many calls without re-creating temp folders every time.

    After calling this, pass ``system_name`` and ``backend_version="sol"``
    to ``run_static_inference`` or ``estimate_latency``.

    Returns the absolute path to the systems directory (same as *systems_dir*).
    """
    Path(systems_dir).mkdir(parents=True, exist_ok=True)
    yaml_content = f"""data_dir: data/{system_name}
gpu:
  mem_bw: {int(mem_bw)}
  mem_bw_empirical_scaling_factor: {mem_bw_scaling:.6f}
  mem_empirical_constant_latency: {mem_constant_latency:.10f}
  mem_capacity: {int(mem_capacity)}
  bfloat16_tc_flops: {int(bfloat16_tc_flops)}
  int8_tc_flops: {int(bfloat16_tc_flops * 2)}
  fp8_tc_flops: {int(bfloat16_tc_flops * 2)}
  power: {int(power)}
  sm_version: {sm_version}
node:
  num_gpus_per_node: {num_gpus_per_node}
  inter_node_bw: {int(inter_node_bw)}
  intra_node_bw: {int(intra_node_bw)}
  pcie_bw: {int(pcie_bw)}
  p2p_latency: {p2p_latency}
misc:
  nccl_mem:
    1: 0
    2: 358612992
    4: 411041792
    8: 411041792
  other_mem: 3758096384
  nccl_version: '2.26.2'
"""
    yaml_path = Path(systems_dir, f"{system_name}.yaml")
    with Path(yaml_path).open("w") as f:
        f.write(yaml_content)

    data_path = Path(systems_dir, "data", system_name, "vllm", "sol")
    Path(data_path).mkdir(parents=True, exist_ok=True)
    for fname in [
        "gemm_perf.txt",
        "context_attention_perf.txt",
        "generation_attention_perf.txt",
        "moe_perf.txt",
        "custom_allreduce_perf.txt",
        "context_mla_perf.txt",
        "generation_mla_perf.txt",
    ]:
        Path.open(Path(data_path, fname), "a").close()

    return systems_dir


def _resolve_system_and_db(
    system_name: str,
    backend_version: str,
    # custom hardware overrides
    mem_bw: float | None = None,
    mem_capacity: float | None = None,
    bfloat16_tc_flops: float | None = None,
    sm_version: int = 90,
) -> tuple[str, str]:
    """Return (system_name, backend_version) ready for get_database().

    If *mem_bw* is passed, creates a temp custom system dir and returns the
    generated system name + ``"sol"`` version.
    """
    if mem_bw is not None:
        assert mem_capacity is not None
        assert bfloat16_tc_flops is not None
        system_name = f"custom_{os.urandom(4).hex()}"
        systems_dir = _create_custom_system_dir(
            system_name=system_name,
            mem_bw=mem_bw,
            mem_capacity=mem_capacity,
            bfloat16_tc_flops=bfloat16_tc_flops,
            sm_version=sm_version,
        )
        set_systems_paths(systems_dir)
        return system_name, "sol"
    return system_name, backend_version


def build_session(
    model_name: str,
    system_name: str,
    backend_name: str,
    backend_version: str | None,
    database_mode: str,
    tp_size: int = 1,
    pp_size: int = 1,
    attention_dp_size: int = 1,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    gemm_quant_mode: str | None = None,
    kvcache_quant_mode: str | None = None,
    fmha_quant_mode: str | None = None,
    moe_quant_mode: str | None = None,
    comm_quant_mode: str | None = None,
):
    """Create model, database, backend, and InferenceSession."""
    from aiconfigurator.sdk.backends.factory import get_backend
    from aiconfigurator.sdk.inference_session import InferenceSession
    from aiconfigurator.sdk.perf_database import get_latest_database_version

    model_cfg = ModelConfig(
        tp_size=tp_size,
        pp_size=pp_size,
        attention_dp_size=attention_dp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        gemm_quant_mode=common.GEMMQuantMode[gemm_quant_mode]
        if gemm_quant_mode
        else None,
        kvcache_quant_mode=common.KVCacheQuantMode[kvcache_quant_mode]
        if kvcache_quant_mode
        else None,
        fmha_quant_mode=common.FMHAQuantMode[fmha_quant_mode]
        if fmha_quant_mode
        else None,
        moe_quant_mode=common.MoEQuantMode[moe_quant_mode] if moe_quant_mode else None,
        comm_quant_mode=common.CommQuantMode[comm_quant_mode]
        if comm_quant_mode
        else None,
    )

    model = get_model(model_name, model_cfg, backend_name)

    resolved_version = backend_version
    if resolved_version is None or resolved_version.lower() == "latest":
        resolved_version = get_latest_database_version(system_name, backend_name)
    if resolved_version is None:
        raise RuntimeError(
            f"No database version found for system={system_name}, backend={backend_name}"
        )

    database = get_database(
        system_name,
        backend_name,
        resolved_version,
        allow_missing_data=database_mode.upper() != "SILICON",
        database_mode=database_mode,
    )
    if database is None:
        raise RuntimeError(
            f"Failed to load database for system={system_name}, backend={backend_name}, version={resolved_version}"
        )
    # Workaround: PerfDatabase.__init__ hardcodes _default_database_mode=SILICON.
    # Force it to the requested mode so SOL/HYBRID queries actually use that mode.
    if database_mode.upper() == "SOL":
        database._default_database_mode = common.DatabaseMode.SOL
    elif database_mode.upper() == "HYBRID":
        database._default_database_mode = common.DatabaseMode.HYBRID
    elif database_mode.upper() == "EMPIRICAL":
        database._default_database_mode = common.DatabaseMode.EMPIRICAL

    backend = get_backend(backend_name)
    session = InferenceSession(model, database, backend)
    return model, database, backend, session


def get_meta(
    backend_version: str,
    mem_bw: float | None = None,
    mem_capacity: float | None = None,
    bfloat16_tc_flops: float | None = None,
    sm_version: int = 90,
    **hw_kwargs: Any,
):
    return _resolve_system_and_db(
        system_name="custom" if mem_bw is not None else "default",
        backend_version=backend_version,
        mem_bw=mem_bw,
        mem_capacity=mem_capacity,
        bfloat16_tc_flops=bfloat16_tc_flops,
        sm_version=sm_version,
        **hw_kwargs,
    )


# ---------------------------------------------------------------------------
# Low-level API: raw per-op latency + memory
# ---------------------------------------------------------------------------


def run_static_inference(
    mode: str,
    built_session: tuple[BaseModel, PerfDatabase, BaseBackend, InferenceSession],
    *,
    isl: int = 1000,
    osl: int = 100,
    prefix: int = 0,
    batch_size: int = 1,
    stride: int = 1,
    latency_correction_scale: float = 1.0,
) -> dict[str, Any]:
    """Run raw static inference and return per-op latency + memory breakdown.

    This is the low-level path.  It runs prefill and decode **independently**
    (no disagg rate-matching, no KV transfer, no autoscale corrections).

    Args:
        mode: ``"prefill"`` or ``"decode"``.
        built_session: Tuple of (model, database, backend, session) from ``build_session()``.
        isl: Input sequence length.
        osl: Output sequence length.
        batch_size: Batch size for **both** prefill and decode.
        prefix: Prefix cache length.
        stride: Stride for static inference (default 1).
        latency_correction_scale: Multiplicative correction applied to raw
            latencies (default 1.0).

    Returns:
        Dict with keys:

        - ``prefill_latency_ms`` — raw sum of context op latencies (no corrections)
        - ``decode_latency_ms`` — raw sum of generation op latencies (no corrections)
        - ``ttft`` — alias for ``prefill_latency_ms``
        - ``tpot`` — ``decode_latency_ms / (osl - 1)`` (or 0 if osl ≤ 1)
        - ``prefill_ops`` — dict of ``{op_name: latency_ms}``
        - ``decode_ops`` — dict of ``{op_name: latency_ms}``
        - ``memory`` — memory dict from ``_get_memory_usage`` (weights, activations, kvcache)
        - ``model_family`` — e.g. ``"LLAMA"``, ``"DEEPSEEK"``
        - ``num_layers`` — number of transformer layers
    """
    from aiconfigurator.sdk.config import RuntimeConfig

    try:
        model, database, backend, session = built_session

        runtime = RuntimeConfig(isl=isl, osl=osl, batch_size=batch_size, prefix=prefix)

        # --- prefill ---
        if mode == "prefill":
            prefill_summary = session.run_static(
                runtime,
                mode="static_ctx",
                stride=stride,
                latency_correction_scale=latency_correction_scale,
            )
            prefill_ops = dict(prefill_summary.get_context_latency_dict())
            prefill_latency_ms = sum(prefill_ops.values())

            # --- memory ---
            memory = backend._get_memory_usage(
                model, database, batch_size, 1, isl, osl, prefix=prefix
            )

            return {
                "prefill_latency_ms": prefill_latency_ms,
                "prefill_ops": prefill_ops,
                "memory": memory,
                "model_family": model.model_family,
                "num_layers": model._num_layers,
            }

        # --- decode ---
        decode_summary = session.run_static(
            runtime,
            mode="static_gen",
            stride=stride,
            latency_correction_scale=latency_correction_scale,
        )
        decode_ops = dict(decode_summary.get_generation_latency_dict())
        decode_latency_ms = sum(decode_ops.values())

        # --- memory ---
        memory = backend._get_memory_usage(
            model, database, batch_size, 1, isl, osl, prefix=prefix
        )

        return {
            "decode_latency_ms": decode_latency_ms,
            "tpot": decode_latency_ms / (osl - 1) if osl > 1 else 0.0,
            "decode_ops": decode_ops,
            "memory": memory,
            "model_family": model.model_family,
            "num_layers": model._num_layers,
        }
    except Exception as e:
        print(f"Error in run_static_inference: {e}")
        set_systems_paths("default")
