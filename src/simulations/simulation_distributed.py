from dataclasses import dataclass

from src.request.request import Request, RequestGenerator
from src.router.router import Router
from src.instances.decode import DecodeInstance
from src.hardware.hardware import Hardware
from src.instances.prefill import PrefillInstance
from src.model.model import Model
from src.request.request import TokenDistribution

@dataclass
class DistributedScenario:
    name: str
    total_requests: int
    num_prefill_instances: int
    num_decode_instances: int
    req_s: float
    batch_size: int
    token_dist: TokenDistribution

def simulate_run_distributed(scenario: DistributedScenario, model: Model) -> None:

    hardware = Hardware("DGX SPARK")

    prefill_instances: list[PrefillInstance] = []
    for _ in range(scenario.num_prefill_instances):
        prefill_instances.append(PrefillInstance(hardware=hardware, model=model))

    decode_instances: list[DecodeInstance] = []
    for _ in range(scenario.num_decode_instances):
        decode_instances.append(DecodeInstance(hardware=hardware, max_batch_size=scenario.batch_size, model=model))


    router = Router(queue=[], prefill_instances=prefill_instances, decode_instnces=decode_instances)

    request_generator = RequestGenerator(scenario.req_s)
    total_time = 0

    finished_requests: list[Request] = []
    for _ in range(scenario.total_requests):
        time_till_next = request_generator.time_till_next_request()
        total_time += time_till_next * 1000
        router.route_requests()
        prefilled_requests: list[Request] = []
        for instance in prefill_instances:
            print(f"Processing prefill instance with queue length {len(instance.queue)}")
            prefilled_requests.extend(instance.process_queue(int(time_till_next * 1000)))
        for instance in decode_instances:
            print(f"Processing decode instance with queue length {len(instance.queue)}")
            finished_requests.extend(instance.process_queue(int(time_till_next * 1000)))
        new_request = request_generator.generate_request(scenario.token_dist)
        router.queue.append(new_request)
        router.queue.extend(prefilled_requests)
        print(f"Generated new request with id: {new_request.id} after {total_time / 1000} seconds")

    (total_time, reqs) = router.finish_requests()
    for req in reqs:
        print(f"Finishing request with id: {req.id} after {total_time / 1000} seconds")
    finished_requests.extend(reqs)

    print(f"Total time for scenario {scenario.name}: {total_time / 1000} seconds")
    print(f"Finished {len(finished_requests)} requests out of {scenario.total_requests}")

    for req in finished_requests:
        print(f"Request {req.id} - Prefill time: {req.prefill_time_ms / 1000} seconds, Decode time: {req.decode_time_ms / 1000} seconds, Total time: {(req.prefill_time_ms + req.decode_time_ms) / 1000} seconds")
