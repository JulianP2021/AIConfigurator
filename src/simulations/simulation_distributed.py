from dataclasses import dataclass

from src.cache.cache import Cache
from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.logger import debug_print
from src.node.node import Node
from src.request.request import Request, RequestGenerator, RequestScenario
from src.result import SimulationResult
from src.router.router import Router


@dataclass
class DistributedScenario:
    name: str
    nodes: list[Node]
    requests: RequestScenario


def simulate_run_distributed(scenario: DistributedScenario) -> SimulationResult:
    prefill_instances: list[PrefillInstance] = []
    decode_instances: list[DecodeInstance] = []

    node_hardware_specs = {node.id: node.hardware for node in scenario.nodes}

    model = (
        scenario.nodes[0].prefill_instances[0].model
        if scenario.nodes[0].prefill_instances
        else scenario.nodes[0].decode_instances[0].model
    )

    cache = Cache(layers={}, node_hardware=node_hardware_specs, model=model)

    for node in scenario.nodes:
        prefill_instances.extend(node.prefill_instances)
        decode_instances.extend(node.decode_instances)

        for prefill_instance in node.prefill_instances:
            prefill_instance.set_cache(cache)
        for decode_instance in node.decode_instances:
            decode_instance.set_cache(cache)

    router = Router(
        queue=[], prefill_instances=prefill_instances, decode_instances=decode_instances
    )

    request_generator = RequestGenerator(scenario.requests.req_s)
    wall_time_ms = 0
    drain_time_ms = 0

    finished_requests: list[Request] = []
    current_requests: list[Request] = []
    num_reqs = 0
    time_to_next_completion = 0
    while (
        time_to_next_completion < float("inf")
        or num_reqs < scenario.requests.total_requests
    ):
        if num_reqs < scenario.requests.total_requests:
            time_till_next_ms = int(request_generator.time_till_next_request() * 1000)
            wall_time_ms += time_till_next_ms
        else:
            time_till_next_ms = float("inf")

        router.route_requests()

        passed_time = 0
        while passed_time < time_till_next_ms:
            prefilled_requests: list[Request] = []
            time_to_next_completion = min(
                [instance.time_to_next_completion() for instance in prefill_instances]
                + [instance.time_to_next_completion() for instance in decode_instances]
                + [time_till_next_ms - passed_time]
            )
            debug_print(
                f"Time to next completion: {time_to_next_completion} ms, passed time: {passed_time} ms, time till next request: {time_till_next_ms} ms {[instance.time_to_next_completion() for instance in prefill_instances]},{[instance.time_to_next_completion() for instance in decode_instances]},{[time_till_next_ms - passed_time]},"
            )

            for instance in prefill_instances:
                debug_print(
                    f"Processing prefill instance with download queue length {len(instance.download_queue)}, queue length {len(instance.queue)}, upload queue length {len(instance.upload_queue)}"
                )
                prefilled_requests.extend(
                    instance.process_queue(time_to_next_completion)
                )
            for instance in decode_instances:
                debug_print(
                    f"Processing decode instance with download queue length {len(instance.download_queue)}, queue length {len(instance.queue)}, upload queue length {len(instance.upload_queue)}"
                )
                decoded_requests = instance.process_queue(time_to_next_completion)
                finished_requests.extend(decoded_requests)
                current_requests = [
                    r for r in current_requests if r not in decoded_requests
                ]
            passed_time += time_to_next_completion
            router.queue.extend(prefilled_requests)
            prefilled_requests = []
            router.route_requests()

        if num_reqs < scenario.requests.total_requests:
            new_request = request_generator.generate_request(
                scenario.requests, current_requests, finished_requests
            )
            current_requests.append(new_request)
            router.queue.append(new_request)
            num_reqs += 1

            debug_print(
                f"Generated new request with id: {new_request.id} after {wall_time_ms / 1000} seconds, user_id: {new_request.user_id}, isl: {new_request.isl}, osl: {new_request.osl}, cached: {new_request.prefilled_tokens}"
            )
        else:
            drain_time_ms += passed_time

    debug_print(f"Finished requests: {finished_requests}")

    assert len(finished_requests) == scenario.requests.total_requests

    wall_time_ms += drain_time_ms
    total_time_s = wall_time_ms / 1000.0

    # Per-request stats
    per_request_stats: list[dict[str, float]] = []
    ttft_list: list[float] = []
    tpot_list: list[float] = []
    latency_list: list[float] = []

    total_decode_time_ms = 0.0
    total_prefill_time_ms = 0.0

    for req in finished_requests:
        # TTFT = prefill_time_ms + kv_upload + kv_download
        ttft_val = float(
            req.prefill_time_ms + req.kv_upload_time_ms + req.kv_download_time_ms
        )

        debug_print(
            f"Request {req.id} TTFT: {ttft_val} ms, prefill: {req.prefill_time_ms} ms, kv_upload: {req.kv_upload_time_ms} ms, kv_download: {req.kv_download_time_ms} ms"
        )

        # TPOT = decode_time_ms / output tokens (guard against div0)
        tpot_val = float(req.decode_time_ms) / (req.osl - 1) if req.osl > 1 else 0.0
        # End-to-end latency
        latency_val = float(
            req.prefill_time_ms
            + req.decode_time_ms
            + req.kv_upload_time_ms
            + req.kv_download_time_ms
        )

        ttft_list.append(ttft_val)
        tpot_list.append(tpot_val)
        latency_list.append(latency_val)
        total_decode_time_ms += float(req.decode_time_ms)
        total_prefill_time_ms += float(req.prefill_time_ms)

        per_request_stats.append({
            "id": req.id,
            "user_id": req.user_id,
            "isl": req.isl,
            "osl": req.osl,
            "prefill_time_ms": req.prefill_time_ms,
            "decode_time_ms": req.decode_time_ms,
            "kv_upload_time_ms": req.kv_upload_time_ms,
            "kv_download_time_ms": req.kv_download_time_ms,
            "ttft_ms": ttft_val,
            "tpot_ms": tpot_val,
            "latency_ms": latency_val,
        })

    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0.0
    avg_tpot = sum(tpot_list) / len(tpot_list) if tpot_list else 0.0
    max_ttft_val = max(ttft_list) if ttft_list else 0.0
    max_tpot_val = max(tpot_list) if tpot_list else 0.0
    avg_latency = sum(latency_list) / len(latency_list) if latency_list else 0.0
    max_latency_val = max(latency_list) if latency_list else 0.0

    sequence_per_second = (
        len(finished_requests) / total_time_s if total_time_s > 0 else 0.0
    )
    _concurrency = total_decode_time_ms / wall_time_ms if wall_time_ms > 0 else 0.0

    # Approximate tokens/s per gpu: total generated tokens / total gpu seconds
    num_gpus = sum(node.hardware.spec.num_gpus for node in scenario.nodes)
    tokens_per_second = (
        sequence_per_second
        * sum(req.osl for req in finished_requests)
        / len(finished_requests)
    )
    tokens_per_second_per_gpu = tokens_per_second / num_gpus if num_gpus > 0 else 0.0
    batch_size = max(
        node.decode_instances[0].max_batch_size if node.decode_instances else 0
        for node in scenario.nodes
    )

    tokens_per_second_per_user = 1000 / avg_tpot if avg_tpot > 0 else 0.0

    # Topology extraction
    num_prefill_workers = sum([
        1 if len(node.prefill_instances) > 0 else 0 for node in scenario.nodes
    ])
    num_decode_workers = sum([
        1 if len(node.decode_instances) > 0 else 0 for node in scenario.nodes
    ])

    prefill_gpus = max([len(node.prefill_instances) for node in scenario.nodes])
    decode_gpus = max([len(node.decode_instances) for node in scenario.nodes])

    batch_size = max(
        node.decode_instances[0].max_batch_size if node.decode_instances else 0
        for node in scenario.nodes
    )

    # Pricing (hourly rate only)
    total_price_per_hour = sum(
        node.hardware.spec.price_usd_per_hour for node in scenario.nodes
    )

    result = SimulationResult(
        scenario_name=scenario.name,
        total_gpus=sum(node.hardware.spec.num_gpus for node in scenario.nodes),
        num_prefill_workers=num_prefill_workers,
        num_decode_workers=num_decode_workers,
        prefill_gpus_per_worker=prefill_gpus,
        decode_gpus_per_worker=decode_gpus,
        batch_size=batch_size,
        ttft=avg_ttft,
        tpot=avg_tpot,
        request_latency=avg_latency,
        max_request_latency=max_latency_val,
        max_ttft=max_ttft_val,
        max_tpot=max_tpot_val,
        tokens_per_second=tokens_per_second,
        tokens_per_second_per_gpu=tokens_per_second_per_gpu,
        tokens_per_second_per_user=tokens_per_second_per_user,
        seq_per_second=sequence_per_second,
        concurrency=batch_size,
        memory_gb=0,
        price_usd_per_hour=total_price_per_hour,
        per_request_stats=per_request_stats,
    )

    # Summary print (always shown regardless of debug flag)
    print(f"\n{'=' * 60}")
    print(f"  Simulation Result: {result.scenario_name}")
    print(f"{'=' * 60}")
    print(f"  Total GPUs:        {result.total_gpus}")
    print(
        f"  Prefill workers:   {result.num_prefill_workers} x {result.prefill_gpus_per_worker} GPU(s)"
    )
    print(
        f"  Decode workers:    {result.num_decode_workers} x {result.decode_gpus_per_worker} GPU(s)"
    )
    print(f"  Batch size:        {result.batch_size}")
    print(f"{'-' * 60}")
    print(f"  TTFT:              {result.ttft:.2f} ms")
    print(f"  max TTFT:          {result.max_ttft:.2f} ms")
    print(f"  TPOT:              {result.tpot:.2f} ms")
    print(f"  max TPOT:          {result.max_tpot:.2f} ms")
    print(f"  Request Latency:   {result.request_latency:.2f} ms")
    print(f"{'-' * 60}")
    print(f"  tokens/s:          {result.tokens_per_second:,.2f}")
    print(f"  tokens/s/gpu:      {result.tokens_per_second_per_gpu:,.2f}")
    print(f"  tokens/s/user:     {result.tokens_per_second_per_user:,.2f}")
    print(f"  seq/s:             {result.seq_per_second:.3f}")
    print(f"  concurrency:       {result.concurrency:.1f}")
    print(f"{'-' * 60}")
    print(f"  Memory (peak):     {result.memory_gb:.2f} GB")
    print(f"{'-' * 60}")
    print(f"  Price/hour:        ${result.price_usd_per_hour:.4f}")
    print(f"{'=' * 60}\n")

    return result
