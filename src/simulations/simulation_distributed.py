from dataclasses import dataclass

from src.cache.cache import Cache
from src.hardware.hardware import S3Spec
from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.logger import LOG_SIMULATION, log
from src.node.node import Node
from src.request.request import Request, RequestGenerator, RequestScenario
from src.result import SimulationResult
from src.router.router import Router
from src.scheduler.bandwidth_scheduler import BandwidthScheduler


@dataclass
class DistributedScenario:
    name: str
    nodes: list[Node]
    requests: RequestScenario


def simulate_run_distributed(
    scenario: DistributedScenario,
    ram_usage_fraction: float = 0.8,
    ssd_usage_fraction: float = 0.8,
    s3_spec: S3Spec | None = None,
    should_print: bool = True,
) -> SimulationResult:
    prefill_instances: list[PrefillInstance] = []
    decode_instances: list[DecodeInstance] = []

    node_hardware_specs = {node.id: node.hardware for node in scenario.nodes}

    model = (
        scenario.nodes[0].prefill_instances[0].model
        if scenario.nodes[0].prefill_instances
        else scenario.nodes[0].decode_instances[0].model
    )

    cache = Cache(
        layers={},
        node_hardware=node_hardware_specs,
        model=model,
        ram_usage_fraction=ram_usage_fraction,
        ssd_usage_fraction=ssd_usage_fraction,
        s3_spec=s3_spec,
    )
    scheduler = BandwidthScheduler(scenario.nodes, s3_spec=s3_spec)

    for node in scenario.nodes:
        prefill_instances.extend(node.prefill_instances)
        decode_instances.extend(node.decode_instances)

        for prefill_instance in node.prefill_instances:
            prefill_instance.set_cache(cache)
            prefill_instance.set_scheduler(scheduler)
        for decode_instance in node.decode_instances:
            decode_instance.set_cache(cache)
            decode_instance.set_scheduler(scheduler)

    router = Router(
        queue=[], prefill_instances=prefill_instances, decode_instances=decode_instances
    )

    request_generator = RequestGenerator(
        req_rate=scenario.requests.req_s,
        max_session_turns=scenario.requests.max_session_turns,
    )
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
            transfer_event_ms = scheduler.next_event_ms()
            compute_event_ms = min(
                [instance.time_to_next_completion() for instance in prefill_instances]
                + [instance.time_to_next_completion() for instance in decode_instances]
                + [float("inf")]
            )
            time_to_next_completion = min(
                compute_event_ms,
                transfer_event_ms,
                time_till_next_ms - passed_time,
            )

            # If there are no more compute or transfer events before the next
            # request arrival, stop the inner loop immediately without advancing
            # time.
            if time_to_next_completion == float("inf"):
                break

            # Guard against zero-length steps that could stall the loop.
            if time_to_next_completion <= 0:
                time_to_next_completion = 1e-9

            log(
                LOG_SIMULATION,
                f"Time to next completion: {time_to_next_completion} ms, "
                f"passed time: {passed_time} ms, "
                f"time till next request: {time_till_next_ms} ms, "
                f"compute_event={compute_event_ms}, transfer_event={transfer_event_ms}",
            )

            # Advance all transfers globally using the scheduler.  It returns
            # fully completed transfers; instances drain their own queues by
            # checking ``active_leg is None`` at the top of ``process_queue``.
            scheduler.advance_time(time_to_next_completion)

            for instance in prefill_instances:
                log(
                    LOG_SIMULATION,
                    f"Processing prefill instance with download queue length "
                    f"{len(instance.download_queue)}, queue length {len(instance.queue)}, "
                    f"upload queue length {len(instance.upload_queue)}",
                )
                prefilled_requests.extend(
                    instance.process_queue(time_to_next_completion)
                )
            for instance in decode_instances:
                log(
                    LOG_SIMULATION,
                    f"Processing decode instance with download queue length "
                    f"{len(instance.download_queue)}, queue length {len(instance.queue)}, "
                    f"upload queue length {len(instance.upload_queue)}",
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

            # If there are no more compute or transfer events before the next
            # request arrival, we can stop the inner loop early.
            if time_to_next_completion == float("inf"):
                break

        if (
            num_reqs < scenario.requests.total_requests
            and time_till_next_ms == time_to_next_completion
        ):
            new_request = request_generator.generate_request(
                scenario.requests, current_requests, finished_requests
            )
            current_requests.append(new_request)
            router.queue.append(new_request)
            num_reqs += 1

            log(
                LOG_SIMULATION,
                f"Generated new request with id: {new_request.id} after {wall_time_ms / 1000} seconds, user_id: {new_request.user_id}, isl: {new_request.isl}, osl: {new_request.osl}, cached: {new_request.prefilled_tokens}",
            )
        else:
            drain_time_ms += passed_time

    log(LOG_SIMULATION, f"Finished requests: {finished_requests}")

    print("Finished generating requests, now draining remaining instance queues...")
    print(router.queue)
    for instance in prefill_instances:
        print(
            f"Prefill instance {instance.node_id} queue length: {len(instance.queue)}, "
            f"download queue length: {len(instance.download_queue)}, "
            f"upload queue length: {len(instance.upload_queue)}"
        )
    for instance in decode_instances:
        print(
            f"Decode instance {instance.node_id} queue length: {len(instance.queue)}, "
            f"download queue length: {len(instance.download_queue)}, "
            f"upload queue length: {len(instance.upload_queue)}"
        )
    # Drain any remaining instance upload queues.  When a decode request finishes
    # during the final event step it appends its KV upload to the instance's
    # upload queue, but the main loop may have already decided there were no
    # more events and exited.  Keep stepping until all pending uploads are
    # flushed and requests are moved to finished_requests.
    while len(finished_requests) < scenario.requests.total_requests:
        print(len(finished_requests))
        transfer_event_ms = scheduler.next_event_ms()
        compute_event_ms = min(
            [instance.time_to_next_completion() for instance in prefill_instances]
            + [instance.time_to_next_completion() for instance in decode_instances]
            + [float("inf")]
        )
        time_to_next_completion = min(compute_event_ms, transfer_event_ms)
        if time_to_next_completion == float("inf"):
            break
        if time_to_next_completion <= 0:
            time_to_next_completion = 1e-9

        log(
            LOG_SIMULATION,
            f"Drain step: time to next completion {time_to_next_completion} ms, "
            f"compute_event={compute_event_ms}, transfer_event={transfer_event_ms}",
        )

        scheduler.advance_time(time_to_next_completion)
        for instance in prefill_instances:
            instance.process_queue(time_to_next_completion)
        for instance in decode_instances:
            decoded_requests = instance.process_queue(time_to_next_completion)
            finished_requests.extend(decoded_requests)
            current_requests = [
                r for r in current_requests if r not in decoded_requests
            ]
        drain_time_ms += time_to_next_completion

    assert len(finished_requests) == scenario.requests.total_requests

    wall_time_ms += drain_time_ms
    total_time_s = wall_time_ms / 1000.0

    # Per-request stats
    per_request_stats: list[dict[str, float]] = []
    ttft_list: list[float] = []
    kv_upload_list: list[float] = []
    kv_download_list: list[float] = []
    tpot_list: list[float] = []
    latency_list: list[float] = []

    # Phase-level timing lists
    prefill_time_list: list[float] = []
    prefill_wait_list: list[float] = []
    prefill_download_active_list: list[float] = []
    prefill_download_wait_list: list[float] = []
    prefill_upload_active_list: list[float] = []
    prefill_upload_wait_list: list[float] = []
    decode_download_active_list: list[float] = []
    decode_download_wait_list: list[float] = []
    decode_time_list: list[float] = []
    decode_wait_list: list[float] = []
    decode_upload_active_list: list[float] = []
    decode_upload_wait_list: list[float] = []
    clean_ttft_list: list[float] = []
    clean_latency_list: list[float] = []

    total_decode_time_ms = 0.0
    total_prefill_time_ms = 0.0

    for req in finished_requests:
        # Derived clean/wait-inclusive metrics.
        req.clean_ttft_ms = float(
            req.prefill_time_ms
            + req.prefill_download_active_ms
            + req.prefill_upload_active_ms
            + req.decode_download_active_ms
        )
        req.wait_inclusive_ttft_ms = float(
            req.clean_ttft_ms
            + req.prefill_wait_ms
            + req.prefill_download_wait_ms
            + req.prefill_upload_wait_ms
            + req.decode_download_wait_ms
        )
        req.clean_latency_ms = float(req.clean_ttft_ms + req.decode_time_ms)
        req.wait_inclusive_latency_ms = float(req.clean_latency_ms + req.decode_wait_ms)

        # Backward-compatible total-style TTFT / latency (wait-inclusive).
        ttft_val = req.wait_inclusive_ttft_ms
        latency_val = req.wait_inclusive_latency_ms

        log(
            LOG_SIMULATION,
            f"Request {req.id} TTFT: {ttft_val:.3f} ms (clean {req.clean_ttft_ms:.3f}), "
            f"latency: {latency_val:.3f} ms (clean {req.clean_latency_ms:.3f}), "
            f"prefill: {req.prefill_time_ms:.3f} ms, wait: {req.prefill_wait_ms:.3f}, "
            f"kv_down: active={req.decode_download_active_ms:.3f} wait={req.decode_download_wait_ms:.3f}",
        )

        # TPOT = decode_time_ms / output tokens (guard against div0)
        tpot_val = float(req.decode_time_ms) / (req.osl - 1) if req.osl > 1 else 0.0

        ttft_list.append(ttft_val)
        tpot_list.append(tpot_val)
        kv_upload_list.append(req.kv_upload_time_ms)
        kv_download_list.append(req.kv_download_time_ms)
        latency_list.append(latency_val)
        total_decode_time_ms += float(req.decode_time_ms)
        total_prefill_time_ms += float(req.prefill_time_ms)

        prefill_time_list.append(req.prefill_time_ms)
        prefill_wait_list.append(req.prefill_wait_ms)
        prefill_download_active_list.append(req.prefill_download_active_ms)
        prefill_download_wait_list.append(req.prefill_download_wait_ms)
        prefill_upload_active_list.append(req.prefill_upload_active_ms)
        prefill_upload_wait_list.append(req.prefill_upload_wait_ms)
        decode_download_active_list.append(req.decode_download_active_ms)
        decode_download_wait_list.append(req.decode_download_wait_ms)
        decode_time_list.append(req.decode_time_ms)
        decode_wait_list.append(req.decode_wait_ms)
        decode_upload_active_list.append(req.decode_upload_active_ms)
        decode_upload_wait_list.append(req.decode_upload_wait_ms)
        clean_ttft_list.append(req.clean_ttft_ms)
        clean_latency_list.append(req.clean_latency_ms)

        per_request_stats.append({
            "id": req.id,
            "user_id": req.user_id,
            "isl": req.isl,
            "osl": req.osl,
            "prefill_time_ms": req.prefill_time_ms,
            "prefill_wait_ms": req.prefill_wait_ms,
            "prefill_download_active_ms": req.prefill_download_active_ms,
            "prefill_download_wait_ms": req.prefill_download_wait_ms,
            "prefill_upload_active_ms": req.prefill_upload_active_ms,
            "prefill_upload_wait_ms": req.prefill_upload_wait_ms,
            "decode_download_active_ms": req.decode_download_active_ms,
            "decode_download_wait_ms": req.decode_download_wait_ms,
            "decode_time_ms": req.decode_time_ms,
            "decode_wait_ms": req.decode_wait_ms,
            "decode_upload_active_ms": req.decode_upload_active_ms,
            "decode_upload_wait_ms": req.decode_upload_wait_ms,
            "kv_upload_time_ms": req.kv_upload_time_ms,
            "kv_download_time_ms": req.kv_download_time_ms,
            "clean_ttft_ms": req.clean_ttft_ms,
            "ttft_ms": ttft_val,
            "clean_latency_ms": req.clean_latency_ms,
            "latency_ms": latency_val,
            "tpot_ms": tpot_val,
        })

    def _avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    def _max(lst: list[float]) -> float:
        return max(lst) if lst else 0.0

    avg_ttft = _avg(ttft_list)
    avg_tpot = _avg(tpot_list)
    avg_kv_upload = _avg(kv_upload_list)
    avg_kv_download = _avg(kv_download_list)
    max_ttft_val = _max(ttft_list)
    max_tpot_val = _max(tpot_list)
    avg_latency = _avg(latency_list)
    max_latency_val = _max(latency_list)

    avg_prefill_time = _avg(prefill_time_list)
    avg_prefill_wait = _avg(prefill_wait_list)
    max_prefill_wait = _max(prefill_wait_list)
    avg_prefill_download_active = _avg(prefill_download_active_list)
    avg_prefill_download_wait = _avg(prefill_download_wait_list)
    avg_prefill_upload_active = _avg(prefill_upload_active_list)
    avg_prefill_upload_wait = _avg(prefill_upload_wait_list)
    avg_decode_download_active = _avg(decode_download_active_list)
    avg_decode_download_wait = _avg(decode_download_wait_list)
    avg_decode_time = _avg(decode_time_list)
    avg_decode_wait = _avg(decode_wait_list)
    max_decode_wait = _max(decode_wait_list)
    avg_decode_upload_active = _avg(decode_upload_active_list)
    avg_decode_upload_wait = _avg(decode_upload_wait_list)
    avg_clean_ttft = _avg(clean_ttft_list)
    max_clean_ttft = _max(clean_ttft_list)
    avg_clean_latency = _avg(clean_latency_list)
    max_clean_latency = _max(clean_latency_list)

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
    total_price_per_hour = (
        sum(node.hardware.spec.dph_base for node in scenario.nodes)
        + cache.cost_usd * 3600.0 / total_time_s
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
        kv_upload_time=avg_kv_upload,
        kv_download_time=avg_kv_download,
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
        avg_prefill_time_ms=avg_prefill_time,
        avg_prefill_wait_ms=avg_prefill_wait,
        max_prefill_wait_ms=max_prefill_wait,
        avg_prefill_download_active_ms=avg_prefill_download_active,
        avg_prefill_download_wait_ms=avg_prefill_download_wait,
        avg_prefill_upload_active_ms=avg_prefill_upload_active,
        avg_prefill_upload_wait_ms=avg_prefill_upload_wait,
        avg_decode_time_ms=avg_decode_time,
        avg_decode_wait_ms=avg_decode_wait,
        max_decode_wait_ms=max_decode_wait,
        avg_decode_download_active_ms=avg_decode_download_active,
        avg_decode_download_wait_ms=avg_decode_download_wait,
        avg_decode_upload_active_ms=avg_decode_upload_active,
        avg_decode_upload_wait_ms=avg_decode_upload_wait,
        avg_clean_ttft_ms=avg_clean_ttft,
        max_clean_ttft_ms=max_clean_ttft,
        avg_clean_latency_ms=avg_clean_latency,
        max_clean_latency_ms=max_clean_latency,
    )

    if not should_print:
        return result

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
    print(f"  TTFT:                 {result.ttft:.2f} ms   (wait-inclusive)")
    print(f"  max TTFT:             {result.max_ttft:.2f} ms")
    print(
        f"  Clean TTFT:           {result.avg_clean_ttft_ms:.2f} ms   (max {result.max_clean_ttft_ms:.2f})"
    )
    print(f"  TPOT:                 {result.tpot:.2f} ms")
    print(f"  max TPOT:             {result.max_tpot:.2f} ms")
    print(f"  KV Upload Time:       {result.kv_upload_time:.2f} ms")
    print(f"  KV Download Time:     {result.kv_download_time:.2f} ms")
    print(f"  Request Latency:      {result.request_latency:.2f} ms   (wait-inclusive)")
    print(
        f"  Clean Latency:        {result.avg_clean_latency_ms:.2f} ms   (max {result.max_clean_latency_ms:.2f})"
    )
    print(f"{'-' * 60}")
    print(f"  Prefill active:       {result.avg_prefill_time_ms:.2f} ms")
    print(
        f"  Prefill wait:         {result.avg_prefill_wait_ms:.2f} ms   (max {result.max_prefill_wait_ms:.2f})"
    )
    print(
        f"  Prefill download:     active {result.avg_prefill_download_active_ms:.2f} ms   wait {result.avg_prefill_download_wait_ms:.2f}"
    )
    print(
        f"  Prefill upload:       active {result.avg_prefill_upload_active_ms:.2f} ms   wait {result.avg_prefill_upload_wait_ms:.2f}"
    )
    print(f"  Decode active:        {result.avg_decode_time_ms:.2f} ms")
    print(
        f"  Decode wait:          {result.avg_decode_wait_ms:.2f} ms   (max {result.max_decode_wait_ms:.2f})"
    )
    print(
        f"  Decode download:      active {result.avg_decode_download_active_ms:.2f} ms   wait {result.avg_decode_download_wait_ms:.2f}"
    )
    print(
        f"  Decode upload:        active {result.avg_decode_upload_active_ms:.2f} ms   wait {result.avg_decode_upload_wait_ms:.2f}"
    )
    print(f"{'-' * 60}")
    print(f"  tokens/s:             {result.tokens_per_second:,.2f}")
    print(f"  tokens/s/gpu:         {result.tokens_per_second_per_gpu:,.2f}")
    print(f"  tokens/s/user:        {result.tokens_per_second_per_user:,.2f}")
    print(f"  seq/s:                {result.seq_per_second:.3f}")
    print(f"  concurrency:          {result.concurrency:.1f}")
    print(f"{'-' * 60}")
    print(f"  Memory (peak):        {result.memory_gb:.2f} GB")
    print(f"{'-' * 60}")
    print(f"  Price/hour:           ${result.price_usd_per_hour:.4f}")
    print(f"{'=' * 60}\n")

    return result
