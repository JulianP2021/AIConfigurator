"""Tests for per-request latency SLAs in simulate_run_distributed."""

import pickle

import pytest

from src.eroors.errors import DecodeLatencyError, PrefillLatencyError
from src.hardware.hardware import Hardware
from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.model.model import Model
from src.node.node import Node
from src.request.request import RequestScenario, TokenDistribution
from src.simulations.simulation_distributed import (
    DistributedScenario,
    simulate_run_distributed,
)


def _tiny_hardware() -> Hardware:
    """Hardware fast enough for tiny workloads but slow enough to miss tight SLAs."""
    from src.hardware.hardware import GPUHardwareSpec, HardwareSpec

    gpu_spec = GPUHardwareSpec(
        flops=1_000_000_000,  # 1 GFLOPS - slow enough to produce non-zero ms times
        gpu_mem=10_000_000_000,  # 10 GB
        gpu_bw=1_000_000_000,  # 1 GB/s
    )
    spec = HardwareSpec(
        gpu_hardware=gpu_spec,
        num_gpus=2,
        nvme_mem=1_000_000_000,
        nvme_bw=1_000_000_000,
        network_inet_up=10_000_000_000,
        network_inet_down=10_000_000_000,
        network_inter_node_up=10_000_000_000,
        network_inter_node_down=10_000_000_000,
        cpu_ram=25_000_000_000,
        dph_base=1.0,
        pcie_bw=1_000_000_000_000,
    )
    return Hardware(name="tiny", spec=spec)


def _fake_model() -> Model:
    from unittest.mock import MagicMock

    model = MagicMock(spec=Model)
    model.kv_size_per_token = 100
    model.name = "fake"
    model.dtype_size = 2
    model.config = {
        "num_key_value_heads": 4,
        "num_hidden_layers": 2,
        "head_dim": 64,
        "head_size": 64,
    }
    model.cost_constants = {
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "vocab_size": 1000,
        "d_kv": 64,
        "dtype_size": 2.0,
        "output_flops": 512_000,
        "matrices": 1_000_000,
        "embedding_memory": 2_000_000,
    }
    return model


def _separate_scenario(
    isl: int, osl: int, sessions_per_user: int
) -> DistributedScenario:
    hardware = _tiny_hardware()
    model = _fake_model()
    node = Node(
        hardware=hardware,
        model_name="Qwen/Qwen3-8B",
        batch_size=1,
        prefill_instances=1,
        decode_instances=1,
    )
    # Patch the node instances to use our fake model directly; avoids HF lookups.
    node.prefill_instances = [
        PrefillInstance(node.id, hardware.spec.gpu_hardware, model, max_batch_size=1)
    ]
    node.decode_instances = [
        DecodeInstance(node.id, hardware.spec.gpu_hardware, 1, model)
    ]
    users = 2
    max_session_turns = 5
    return DistributedScenario(
        name="sla_test",
        nodes=[node],
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
            think_time_ms=0.0,
        ),
    )


def test_default_sla_runs():
    scenario = _separate_scenario(isl=128, osl=4, sessions_per_user=1)
    result = simulate_run_distributed(
        scenario, should_print=False, sla={"ttft_ms": 10000.0, "tpot_ms": 100.0}
    )
    assert result is not None
    assert len(result.per_request_stats) == 10
    assert result.ram_cache_usage_bytes >= 0
    assert result.ssd_cache_usage_bytes >= 0
    assert result.s3_cache_usage_bytes >= 0


def test_inf_sla_rejected():
    scenario = _separate_scenario(isl=128, osl=4, sessions_per_user=1)
    with pytest.raises(
        ValueError, match="ttft_ms must be a finite positive number, got inf"
    ):
        simulate_run_distributed(
            scenario,
            should_print=False,
            sla={"ttft_ms": float("inf"), "tpot_ms": float("inf")},
        )


def test_ttft_sla_violation_raises():
    scenario = _separate_scenario(isl=128, osl=4, sessions_per_user=1)
    with pytest.raises(PrefillLatencyError) as exc_info:
        simulate_run_distributed(
            scenario,
            should_print=False,
            sla={"ttft_ms": 0.1, "tpot_ms": 100.0},
        )
    assert "TTFT SLA violated" in str(exc_info.value)


def test_tpot_sla_violation_raises():
    # Use many output tokens so the tiny fake model's decode_time exceeds the
    # extremely tight 0.1 ms per-token SLA.
    scenario = _separate_scenario(isl=128, osl=500, sessions_per_user=1)
    with pytest.raises(DecodeLatencyError) as exc_info:
        simulate_run_distributed(
            scenario,
            should_print=False,
            sla={"ttft_ms": 10000.0, "tpot_ms": 0.1},
        )
    assert "TPOT SLA violated" in str(exc_info.value)


def test_prefill_latency_error_is_picklable():
    """PrefillLatencyError must round-trip through pickle for process pools."""
    exc = PrefillLatencyError("TTFT SLA violated", 1234.5, 1000.0)
    data = pickle.dumps(exc)
    restored = pickle.loads(data)
    assert isinstance(restored, PrefillLatencyError)
    assert str(restored) == "TTFT SLA violated"
    assert restored.prefill_only_ttft_ms == 1234.5
    assert restored.ttft_sla_ms == 1000.0
