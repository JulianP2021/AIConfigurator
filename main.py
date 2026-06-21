import sys

from src.hardware.hardware import Hardware
from src.logger import set_debug
from src.node.node import Node
from src.request.request import RequestScenario, TokenDistribution
from src.simulations.simulation_distributed import (
    DistributedScenario,
    simulate_run_distributed,
)

if __name__ == "__main__":
    # Toggle debug logging via --debug CLI flag
    if "--debug" in sys.argv:
        set_debug(True)

    result = simulate_run_distributed(
        DistributedScenario(
            name="test",
            nodes=[
                Node(
                    hardware=Hardware.from_name("DGX SPARK"),
                    model_name="Qwen/Qwen3-8B",
                    batch_size=10,
                    prefill_instances=1,
                    decode_instances=0,
                ),
                Node(
                    hardware=Hardware.from_name("DGX SPARK"),
                    model_name="Qwen/Qwen3-8B",
                    batch_size=10,
                    prefill_instances=0,
                    decode_instances=1,
                ),
            ],
            requests=RequestScenario(
                total_requests=100,
                min_users=10000,
                max_users=100000000,
                req_s=2,
                token_distribution=TokenDistribution(
                    min_input_tokens=10,
                    max_input_tokens=10,
                    min_output_tokens=100,
                    max_output_tokens=100,
                    cache_percentage=0.5,
                ),
            ),
        ),
    )

    # result is a SimulationResult can be serialized or compared here
    print("Result dict:", result.to_dict())
