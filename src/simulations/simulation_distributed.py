from dataclasses import dataclass

from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.node.node import Node
from src.request.request import Request, RequestGenerator, TokenDistribution
from src.router.router import Router


@dataclass
class DistributedScenario:
    name: str
    total_requests: int
    nodes: list[Node]
    req_s: float
    batch_size: int
    token_dist: TokenDistribution


def simulate_run_distributed(scenario: DistributedScenario) -> None:
    prefill_instances: list[PrefillInstance] = []
    decode_instances: list[DecodeInstance] = []

    for node in scenario.nodes:
        prefill_instances.extend(node.prefill_instances)
        decode_instances.extend(node.decode_instances)
    router = Router(
        queue=[], prefill_instances=prefill_instances, decode_instnces=decode_instances
    )

    request_generator = RequestGenerator(scenario.req_s)
    total_time = 0

    finished_requests: list[Request] = []
    for _ in range(scenario.total_requests):
        time_till_next_ms = int(request_generator.time_till_next_request() * 1000)
        total_time += time_till_next_ms

        router.route_requests()

        passed_time = 0
        while passed_time < time_till_next_ms:
            prefilled_requests: list[Request] = []
            time_to_next_completion = min(
                [
                    instance.time_to_next_completion()
                    for instance in prefill_instances
                    if instance.queue
                ]
                +
                # no decode, as it is way shorter than prefill [instance.time_to_next_completion() for instance in decode_instances if instance.queue]
                [time_till_next_ms - passed_time]
            )
            for instance in prefill_instances:
                print(
                    f"Processing prefill instance with queue length {len(instance.queue)}"
                )
                prefilled_requests.extend(
                    instance.process_queue(time_to_next_completion)
                )
            for instance in decode_instances:
                print(
                    f"Processing decode instance with queue length {len(instance.queue)}"
                )
                decoded_requests = instance.process_queue(time_to_next_completion)
                finished_requests.extend(decoded_requests)
            passed_time += time_to_next_completion
            router.queue.extend(prefilled_requests)
            prefilled_requests = []
            router.route_requests()

        new_request = request_generator.generate_request(scenario.token_dist)
        router.queue.append(new_request)
        print(
            f"Generated new request with id: {new_request.id} after {total_time / 1000} seconds"
        )

    router.route_requests()
    print(
        f"Already finished requests: {len(finished_requests)} out of {scenario.total_requests}"
    )
    print("Finished generating requests, finishing remaining requests in queue")
    router.log()

    (total_time, reqs) = router.finish_requests()
    print(finished_requests, [req.id for req in reqs])
    finished_requests.extend(reqs)

    assert len(finished_requests) == scenario.total_requests

    print(f"Total time for scenario {scenario.name}: {total_time / 1000} seconds")
    print(
        f"Finished {len(finished_requests)} requests out of {scenario.total_requests}"
    )

    for req in finished_requests:
        print(
            f"Request {req.id} - Prefill time: {req.prefill_time_ms / 1000} seconds, KV Upload time: {req.kv_upload_time_ms / 1000} seconds, KV Download time: {req.kv_download_time_ms / 1000} seconds, Decode time: {req.decode_time_ms / 1000} seconds, Total time: {(req.prefill_time_ms + req.decode_time_ms + req.kv_upload_time_ms + req.kv_download_time_ms) / 1000} seconds"
        )
