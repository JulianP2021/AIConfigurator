from dataclasses import dataclass

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

    for node in scenario.nodes:
        prefill_instances.extend(node.prefill_instances)
        decode_instances.extend(node.decode_instances)

    router = Router(
        queue=[], prefill_instances=prefill_instances, decode_instnces=decode_instances
    )

    request_generator = RequestGenerator(scenario.requests.req_s)
    wall_time_ms = 0

    finished_requests: list[Request] = []
    current_requests: list[Request] = []

    for _ in range(scenario.requests.total_requests):
        time_till_next_ms = int(request_generator.time_till_next_request() * 1000)
        wall_time_ms += time_till_next_ms

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
                + [time_till_next_ms - passed_time]
            )
            for instance in prefill_instances:
                debug_print(
                    f"Processing prefill instance with queue length {len(instance.queue)}"
                )
                prefilled_requests.extend(
                    instance.process_queue(time_to_next_completion)
                )
            for instance in decode_instances:
                debug_print(
                    f"Processing decode instance with queue length {len(instance.queue)}"
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

        new_request = request_generator.generate_request(
            scenario.requests, current_requests, finished_requests
        )
        current_requests.append(new_request)
        router.queue.append(new_request)
        debug_print(
            f"Generated new request with id: {new_request.id} after {wall_time_ms / 1000} seconds, user_id: {new_request.user_id}, isl: {new_request.isl}, osl: {new_request.osl}, cached: {new_request.prefilled_tokens}"
        )

    router.route_requests()
    debug_print(
        f"Already finished requests: {len(finished_requests)} out of {scenario.requests.total_requests}"
    )
    debug_print("Finished generating requests, finishing remaining requests in queue")
    router.log()

    (drain_time_ms, reqs) = router.finish_requests()
    debug_print(
        f"Finished requests: {finished_requests}, Final finished requests: {[req.id for req in reqs]}"
    )
    finished_requests.extend(reqs)

    assert len(finished_requests) == scenario.requests.total_requests

    wall_time_ms += drain_time_ms
    total_time_s = wall_time_ms / 1000.0

    # ---- Aggregate metrics -------------------------------------------------
    total_tokens_generated = sum(req.isl + req.osl for req in finished_requests)

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
        # TPOT = decode_time_ms / output tokens (guard against div0)
        tpot_val = float(req.decode_time_ms) / req.osl if req.osl > 0 else 0.0
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

    request_rate = len(finished_requests) / total_time_s if total_time_s > 0 else 0.0
    _concurrency = total_decode_time_ms / wall_time_ms if wall_time_ms > 0 else 0.0

    # Approximate tokens/s per gpu: total generated tokens / total gpu seconds
    total_gpu_seconds = total_time_s * sum(
        node.hardware.spec.num_gpus for node in scenario.nodes
    )
    tokens_per_second = (
        total_tokens_generated / total_time_s if total_time_s > 0 else 0.0
    )
    tokens_per_second_per_gpu = (
        total_tokens_generated / total_gpu_seconds if total_gpu_seconds > 0 else 0.0
    )
    batch_size = max(
        node.decode_instances[0].max_batch_size if node.decode_instances else 0
        for node in scenario.nodes
    )

    tokens_per_second_per_user = (
        total_tokens_generated / (total_time_s * batch_size)
        if total_time_s > 0 and batch_size > 0
        else 0.0
    )

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
        max_ttft=max_ttft_val,
        max_tpot=max_tpot_val,
        tokens_per_second=tokens_per_second,
        tokens_per_second_per_gpu=tokens_per_second_per_gpu,
        tokens_per_second_per_user=tokens_per_second_per_user,
        request_rate=request_rate,
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
    print(f"  req/s:             {result.request_rate:.3f}")
    print(f"  concurrency:       {result.concurrency:.1f}")
    print(f"{'-' * 60}")
    print(f"  Memory (peak):     {result.memory_gb:.2f} GB")
    print(f"{'-' * 60}")
    print(f"  Price/hour:        ${result.price_usd_per_hour:.4f}")
    print(f"{'=' * 60}\n")

    return result
