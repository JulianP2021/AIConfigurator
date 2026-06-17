from dataclasses import dataclass

from src.simulations.simulation_distributed import DistributedScenario, simulate_run_distributed
from src.model.model import Model
from src.request.request import TokenDistribution



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
    simulate_run_distributed(
        DistributedScenario(
            name="test",
            total_requests=2,
            num_prefill_instances=2,
            num_decode_instances=2,
            req_s=10,
            batch_size=10,
            token_dist=TokenDistribution(
                min_input_tokens=10,
                max_input_tokens=100,
                min_output_tokens=10,
                max_output_tokens=100,
                cache_percentage=0.5
            )
        ),
        Model("Qwen/Qwen3-8B")
    )
