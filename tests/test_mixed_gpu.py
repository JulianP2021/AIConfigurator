"""Tests for heterogeneous GPU types on a single mixed-GPU node."""

from src.hardware.hardware import GPUHardwareSpec, Hardware, HardwareSpec
from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.model.model import Model
from src.node.node import Node
from src.request.request import RequestScenario, TokenDistribution
from src.result import SimulationResult
from src.simulations.simulation_distributed import (
    DistributedScenario,
    simulate_run_distributed,
)


def _fake_model() -> Model:
    from tests.fakes import make_fake_model

    return make_fake_model()


def _make_hardware(
    name: str,
    prefill_gpu: GPUHardwareSpec,
    decode_gpu: GPUHardwareSpec,
    node_price: float = 1.0,
) -> Hardware:
    """Build a 2-GPU Hardware node whose per-instance specs differ."""
    spec = HardwareSpec(
        gpu_hardware=prefill_gpu,
        num_gpus=2,
        nvme_mem=1_000_000_000,
        nvme_bw=1_000_000_000,
        network_inet_up=10_000_000_000,
        network_inet_down=10_000_000_000,
        network_inter_node_up=10_000_000_000,
        network_inter_node_down=10_000_000_000,
        cpu_ram=25_000_000_000,
        dph_base=node_price,
        pcie_bw=1_000_000_000_000,
    )
    return Hardware(name=name, spec=spec)


def _mixed_scenario() -> DistributedScenario:
    """One colocated node with a fast prefill GPU and a slow decode GPU."""
    fast_prefill = GPUHardwareSpec(
        flops=10_000_000_000,
        gpu_mem=10_000_000_000,
        gpu_bw=1_000_000_000,
    )
    slow_decode = GPUHardwareSpec(
        flops=1_000_000_000,
        gpu_mem=10_000_000_000,
        gpu_bw=1_000_000_000,
    )
    hardware = _make_hardware("mixed", fast_prefill, slow_decode, node_price=2.0)
    model = _fake_model()
    node = Node(
        hardware=hardware,
        model_name="Qwen/Qwen3-8B",
        batch_size=2,
        prefill_instances=1,
        decode_instances=1,
        prefill_gpu_hardware=fast_prefill,
        decode_gpu_hardware=slow_decode,
    )
    # Patch instances to share the same fake model so compute is deterministic.
    node.prefill_instances = [
        PrefillInstance(node.id, fast_prefill, model, max_batch_size=2)
    ]
    node.decode_instances = [DecodeInstance(node.id, slow_decode, 2, model)]

    return DistributedScenario(
        name="mixed_gpu_test",
        nodes=[node],
        requests=RequestScenario(
            token_distribution=TokenDistribution(
                min_input_tokens=128,
                max_input_tokens=128,
                min_output_tokens=4,
                max_output_tokens=4,
            ),
            sessions_per_user=1,
            users=2,
            max_session_turns=1,
            think_time_ms=0.0,
        ),
    )


def _uniform_scenario() -> DistributedScenario:
    """Identical setup using the slow GPU for both prefill and decode."""
    slow = GPUHardwareSpec(
        flops=1_000_000_000,
        gpu_mem=10_000_000_000,
        gpu_bw=1_000_000_000,
    )
    hardware = _make_hardware("uniform", slow, slow, node_price=2.0)
    model = _fake_model()
    node = Node(
        hardware=hardware,
        model_name="Qwen/Qwen3-8B",
        batch_size=2,
        prefill_instances=1,
        decode_instances=1,
    )
    node.prefill_instances = [PrefillInstance(node.id, slow, model, max_batch_size=2)]
    node.decode_instances = [DecodeInstance(node.id, slow, 2, model)]

    return DistributedScenario(
        name="uniform_gpu_test",
        nodes=[node],
        requests=RequestScenario(
            token_distribution=TokenDistribution(
                min_input_tokens=128,
                max_input_tokens=128,
                min_output_tokens=4,
                max_output_tokens=4,
            ),
            sessions_per_user=1,
            users=2,
            max_session_turns=1,
            think_time_ms=0.0,
        ),
    )


def test_node_uses_distinct_gpu_specs():
    fast = GPUHardwareSpec(flops=10, gpu_mem=100, gpu_bw=10)
    slow = GPUHardwareSpec(flops=1, gpu_mem=100, gpu_bw=10)
    spec = HardwareSpec(
        gpu_hardware=fast,
        num_gpus=2,
        nvme_mem=1,
        nvme_bw=1,
        network_inet_up=1,
        network_inet_down=1,
        network_inter_node_up=1,
        network_inter_node_down=1,
        cpu_ram=250,
        dph_base=1,
        pcie_bw=1.0,
    )
    hardware = Hardware(name="distinct", spec=spec)
    model = _fake_model()
    node = Node(
        hardware=hardware,
        model_name="Qwen/Qwen3-8B",
        batch_size=1,
        prefill_instances=1,
        decode_instances=1,
        prefill_gpu_hardware=fast,
        decode_gpu_hardware=slow,
    )
    # Patch instances to use the fake model to avoid HF lookups.
    node.prefill_instances = [PrefillInstance(node.id, fast, model, max_batch_size=1)]
    node.decode_instances = [DecodeInstance(node.id, slow, 1, model)]
    assert node.prefill_instances[0].hardware == fast
    assert node.decode_instances[0].hardware == slow


def test_mixed_gpu_changes_latency():
    sla = {"ttft_ms": 10000.0, "tpot_ms": 100.0}
    mixed_result = simulate_run_distributed(
        _mixed_scenario(), should_print=False, sla=sla
    )
    uniform_result = simulate_run_distributed(
        _uniform_scenario(), should_print=False, sla=sla
    )

    assert isinstance(mixed_result, SimulationResult)
    assert isinstance(uniform_result, SimulationResult)
    # The mixed node uses a faster prefill GPU, so prefill active time drops.
    assert mixed_result.avg_prefill_time_ms < uniform_result.avg_prefill_time_ms
    # The mixed node uses a slower decode GPU, so decode active time rises.
    assert mixed_result.avg_decode_time_ms >= uniform_result.avg_decode_time_ms - 1e-9
