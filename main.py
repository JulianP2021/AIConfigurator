from dataclasses import dataclass

from src.hardware.hardware import Hardware
from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.model.model import Model
from src.node.node import Node
from src.request.request import TokenDistribution
from src.simulations.simulation_distributed import (
    DistributedScenario,
    simulate_run_distributed,
)


@dataclass
class Result:
    prefill_cache_s: float
    prefill_computation_s: float
    prefill_s: float
    decode_s: float
    total_s: float
    prefill_req_s: float
    decode_req_s: float
    req_s: float
    decode_instances: int


if __name__ == "__main__":
    hardware = Hardware("DGX SPARK")
    simulate_run_distributed(
        DistributedScenario(
            name="test",
            total_requests=2,
            nodes=[
                Node(
                    prefill_instances=[
                        PrefillInstance(
                            hardware=Hardware("DGX SPARK"), model=Model("Qwen/Qwen3-8B")
                        )
                        for _ in range(2)
                    ],
                    decode_instances=[
                        DecodeInstance(
                            hardware=Hardware("DGX SPARK"),
                            max_batch_size=10,
                            model=Model("Qwen/Qwen3-8B"),
                        )
                        for _ in range(2)
                    ],
                ),
                Node(
                    prefill_instances=[
                        PrefillInstance(
                            hardware=Hardware("DGX SPARK"), model=Model("Qwen/Qwen3-8B")
                        )
                        for _ in range(2)
                    ],
                    decode_instances=[
                        DecodeInstance(
                            hardware=Hardware("DGX SPARK"),
                            max_batch_size=10,
                            model=Model("Qwen/Qwen3-8B"),
                        )
                        for _ in range(2)
                    ],
                ),
            ],
            req_s=10,
            batch_size=10,
            token_dist=TokenDistribution(
                min_input_tokens=10,
                max_input_tokens=100,
                min_output_tokens=10,
                max_output_tokens=100,
                cache_percentage=0.5,
            ),
        ),
    )
