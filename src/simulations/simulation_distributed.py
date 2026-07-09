from dataclasses import dataclass

from src.cache.cache import Cache
from src.eroors.errors import DecodeLatencyError, PrefillLatencyError
from src.hardware.hardware import S3Spec
from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.logger import LOG_SIMULATION, log
from src.node.node import Node
from src.request.request import Request, RequestGenerator, RequestScenario
from src.result import SimulationResult
from src.router.router import Router, RouterCostConfig
from src.scheduler.bandwidth_scheduler import BandwidthScheduler


@dataclass
class DistributedScenario:
    name: str
    nodes: list[Node]
    requests: RequestScenario


def _duration(start_ms: float | None, end_ms: float | None) -> float:
    if start_ms is None or end_ms is None:
        return 0.0
    return max(0.0, end_ms - start_ms)


def _finish_request(
    request: Request,
    finished_requests: list[Request],
    sla: dict[str, float] | None,
    request_generator: RequestGenerator | None = None,
    now_ms: float = 0.0,
) -> None:
    """Append a completed request to finished_requests and enforce per-request SLAs.

    Computes the wait-inclusive TTFT from the request's phase timestamps and
    raises a latency-specific exception if any configured SLA is exceeded.
    """
    request.prefill_time_ms = _duration(
        request.prefill_start_ms, request.prefill_end_ms
    )
    request.prefill_wait_ms = _duration(
        request.prefill_queue_start_ms, request.prefill_start_ms
    )
    request.prefill_download_wait_ms = max(
        0.0,
        _duration(request.prefill_download_start_ms, request.prefill_download_end_ms)
        - request.prefill_download_active_ms,
    )
    request.prefill_upload_wait_ms = max(
        0.0,
        _duration(request.prefill_upload_start_ms, request.prefill_upload_end_ms)
        - request.prefill_upload_active_ms,
    )
    request.decode_time_ms = _duration(request.decode_start_ms, request.decode_end_ms)
    request.decode_wait_ms = _duration(
        request.decode_queue_start_ms, request.decode_start_ms
    )
    request.decode_download_wait_ms = max(
        0.0,
        _duration(request.decode_download_start_ms, request.decode_download_end_ms)
        - request.decode_download_active_ms,
    )
    request.decode_upload_wait_ms = max(
        0.0,
        _duration(request.decode_upload_start_ms, request.decode_upload_end_ms)
        - request.decode_upload_active_ms,
    )

    clean_ttft_ms = (
        request.prefill_time_ms
        + request.prefill_download_active_ms
        + request.prefill_upload_active_ms
        + request.decode_download_active_ms
    )
    wait_inclusive_ttft_ms = (
        clean_ttft_ms
        + request.prefill_wait_ms
        + request.prefill_download_wait_ms
        + request.prefill_upload_wait_ms
        + request.decode_download_wait_ms
    )

    if sla is not None:
        ttft_sla = sla.get("ttft_ms")
        if ttft_sla is not None and wait_inclusive_ttft_ms > ttft_sla:
            raise PrefillLatencyError(
                f"Request {request.id} TTFT SLA violated: "
                f"wait-inclusive TTFT {wait_inclusive_ttft_ms:.2f} ms > "
                f"SLA {ttft_sla:.2f} ms"
            )

        tpot_sla = sla.get("tpot_ms")
        if tpot_sla is not None and request.osl > 1:
            tpot_ms = request.decode_time_ms / (request.osl - 1)
            if tpot_ms > tpot_sla:
                raise DecodeLatencyError(
                    f"Request {request.id} TPOT SLA violated: "
                    f"TPOT {tpot_ms:.2f} ms > SLA {tpot_sla:.2f} ms"
                )

    finished_requests.append(request)
    if request_generator is not None:
        request_generator.finish_request(request, now_ms)


def _generate_ready_requests(
    request_scenario: RequestScenario,
    request_generator: RequestGenerator,
    now_ms: float,
    router: Router,
    current_requests: list[Request],
    num_reqs: int,
) -> int:
    """Create requests for all users that are idle and past their think time.

    Returns the updated request count.
    """
    while num_reqs < request_scenario.total_requests:
        new_request = request_generator.generate_request(request_scenario, now_ms)
        if new_request is None:
            break
        current_requests.append(new_request)
        router.queue.append(new_request)
        num_reqs += 1
        log(
            LOG_SIMULATION,
            f"Generated new request with id: {new_request.id} at "
            f"{now_ms / 1000:.3f} seconds, user_id: {new_request.user_id}, "
            f"isl: {new_request.isl}, osl: {new_request.osl}, "
            f"cached: {new_request.prefilled_tokens}",
        )
    return num_reqs


def simulate_run_distributed(
    scenario: DistributedScenario,
    ram_usage_fraction: float = 0.8,
    ssd_usage_fraction: float = 0.8,
    s3_spec: S3Spec | None = None,
    router_cost_config: RouterCostConfig | None = None,
    should_print: bool = True,
    sla: dict[str, float] | None = None,
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
        queue=[],
        prefill_instances=prefill_instances,
        decode_instances=decode_instances,
        cache=cache,
        cost_config=router_cost_config,
    )

    request_generator = RequestGenerator(
        users=scenario.requests.users,
        max_session_turns=scenario.requests.max_session_turns,
        think_time_ms=scenario.requests.think_time_ms,
        sessions_per_user=scenario.requests.sessions_per_user,
    )
    finished_requests: list[Request] = []
    current_requests: list[Request] = []
    num_reqs = 0
    time_to_next_completion = 0

    # Generate initial requests for all ready users.
    while num_reqs < scenario.requests.total_requests:
        new_request = request_generator.generate_request(
            scenario.requests, scheduler.time_ms
        )
        if new_request is None:
            break
        current_requests.append(new_request)
        router.queue.append(new_request)
        num_reqs += 1

        log(
            LOG_SIMULATION,
            f"Generated new request with id: {new_request.id} at "
            f"{scheduler.time_ms / 1000:.3f} seconds, user_id: {new_request.user_id}, "
            f"isl: {new_request.isl}, osl: {new_request.osl}, "
            f"cached: {new_request.prefilled_tokens}",
        )

    max_iterations = max(1000, scenario.requests.total_requests * 100)
    iterations = 0
    heartbeat_every = max(100, scenario.requests.total_requests // 10)
    while len(finished_requests) < scenario.requests.total_requests:
        iterations += 1
        if iterations > max_iterations:
            raise RuntimeError(
                f"Simulation loop exceeded {max_iterations} iterations "
                f"(finished={len(finished_requests)}, num_reqs={num_reqs}, "
                f"global_time={scheduler.time_ms:.3f} ms). This usually means "
                f"the event loop is not making progress."
            )
        if iterations % heartbeat_every == 0:
            log(
                LOG_SIMULATION,
                f"Heartbeat: iteration={iterations}, finished={len(finished_requests)}"
                f"/ total={scenario.requests.total_requests}, global_time="
                f"{scheduler.time_ms:.3f} ms, active={len(current_requests)},"
                f" queue={len(router.queue)}.",
            )
            if should_print:
                print(
                    f"[simulate] iteration={iterations}, "
                    f"finished={len(finished_requests)}/"
                    f"{scenario.requests.total_requests}, "
                    f"global_time={scheduler.time_ms:.3f} ms"
                )
        router.route_requests()

        prefilled_requests: list[Request] = []
        transfer_event_ms = scheduler.next_event_ms()
        compute_event_ms = min(
            [instance.time_to_next_completion() for instance in prefill_instances]
            + [instance.time_to_next_completion() for instance in decode_instances]
            + [float("inf")]
        )
        time_to_next_completion = min(compute_event_ms, transfer_event_ms)

        # If nothing else is happening, jump to the next moment a user becomes
        # ready to send its next request.
        if time_to_next_completion == float("inf"):
            next_ready_ms = request_generator.next_ready_time_ms(scheduler.time_ms)
            if next_ready_ms != float("inf") and next_ready_ms > scheduler.time_ms:
                scheduler.advance_time(next_ready_ms - scheduler.time_ms)
                # Routing newly-ready requests may create compute/transfer work,
                # so recompute instead of forcing a zero-length step.
                continue
            if next_ready_ms == float("inf"):
                raise RuntimeError(
                    "No compute/transfer events and no user will become ready, "
                    "but not all requests are finished."
                )
            # next_ready_ms <= scheduler.time_ms: users are already ready now.
            time_to_next_completion = 0

        # Zero-length events can occur when a transfer/compute finishes
        # exactly at the current wall-clock time.  Process them immediately
        # without advancing the clock or logging a bogus 0 ms step.
        if time_to_next_completion <= 0:
            for instance in prefill_instances:
                prefilled_requests.extend(instance.process_queue(0))
            for instance in decode_instances:
                decoded_requests = instance.process_queue(0)
                for decoded_request in decoded_requests:
                    _finish_request(
                        decoded_request,
                        finished_requests,
                        sla,
                        request_generator,
                        scheduler.time_ms,
                    )
                current_requests = [
                    r for r in current_requests if r not in decoded_requests
                ]
            router.queue.extend(prefilled_requests)
            prefilled_requests = []
            router.route_requests()
            num_reqs = _generate_ready_requests(
                scenario.requests,
                request_generator,
                scheduler.time_ms,
                router,
                current_requests,
                num_reqs,
            )
            continue

        log(
            LOG_SIMULATION,
            f"Time to next completion: {time_to_next_completion} ms, "
            f"global_time={scheduler.time_ms:.3f} ms, "
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
            prefilled_requests.extend(instance.process_queue(time_to_next_completion))
        for instance in decode_instances:
            log(
                LOG_SIMULATION,
                f"Processing decode instance with download queue length "
                f"{len(instance.download_queue)}, queue length {len(instance.queue)}, "
                f"upload queue length {len(instance.upload_queue)}",
            )
            decoded_requests = instance.process_queue(time_to_next_completion)
            for decoded_request in decoded_requests:
                _finish_request(
                    decoded_request,
                    finished_requests,
                    sla,
                    request_generator,
                    scheduler.time_ms,
                )
            current_requests = [
                r for r in current_requests if r not in decoded_requests
            ]
        router.queue.extend(prefilled_requests)
        prefilled_requests = []
        router.route_requests()

        # Generate new requests for any users that just became idle and ready.
        num_reqs = _generate_ready_requests(
            scenario.requests,
            request_generator,
            scheduler.time_ms,
            router,
            current_requests,
            num_reqs,
        )

    log(LOG_SIMULATION, f"Finished requests: {finished_requests}")

    # Drain any remaining instance upload queues.  When a decode request finishes
    # during the final event step it appends its KV upload to the instance's
    # upload queue, but the main loop may have already decided there were no
    # more events and exited.  Keep stepping until all pending uploads are
    # flushed and requests are moved to finished_requests.
    while len(finished_requests) < scenario.requests.total_requests:
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
            time_to_next_completion = 10

        log(
            LOG_SIMULATION,
            f"Drain step: time to next completion {time_to_next_completion} ms, "
            f"global_time={scheduler.time_ms:.3f} ms, "
            f"compute_event={compute_event_ms}, transfer_event={transfer_event_ms}",
        )

        scheduler.advance_time(time_to_next_completion)
        for instance in prefill_instances:
            instance.process_queue(time_to_next_completion)
        for instance in decode_instances:
            decoded_requests = instance.process_queue(time_to_next_completion)
            for decoded_request in decoded_requests:
                _finish_request(
                    decoded_request,
                    finished_requests,
                    sla,
                    request_generator,
                    scheduler.time_ms,
                )
            current_requests = [
                r for r in current_requests if r not in decoded_requests
            ]

    assert len(finished_requests) == scenario.requests.total_requests

    # Total elapsed wall-clock time is now owned by the global scheduler clock.
    wall_time_ms = scheduler.time_ms
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
        # Phase timings were already computed when the request was added to
        # finished_requests by _finish_request. Re-use them here so the final
        # metrics and per-request stats are consistent.

        req.kv_download_time_ms = (
            req.prefill_download_active_ms + req.decode_download_active_ms
        )
        req.kv_upload_time_ms = (
            req.prefill_upload_active_ms + req.decode_upload_active_ms
        )

        req.clean_ttft_ms = (
            req.prefill_time_ms
            + req.prefill_download_active_ms
            + req.prefill_upload_active_ms
            + req.decode_download_active_ms
        )
        req.wait_inclusive_ttft_ms = (
            req.clean_ttft_ms
            + req.prefill_wait_ms
            + req.prefill_download_wait_ms
            + req.prefill_upload_wait_ms
            + req.decode_download_wait_ms
        )
        req.clean_latency_ms = req.clean_ttft_ms + req.decode_time_ms
        req.wait_inclusive_latency_ms = req.clean_latency_ms + req.decode_wait_ms

        # Reported TTFT/latency include waiting time.
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

    # Cache usage summary at simulation end
    cache_usage = cache.usage_summary()

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
        ram_cache_usage_bytes=cache_usage["ram_usage_bytes"],
        ssd_cache_usage_bytes=cache_usage["ssd_usage_bytes"],
        s3_cache_usage_bytes=cache_usage["s3_usage_bytes"],
        ram_cache_capacity_bytes=cache_usage["ram_capacity_bytes"],
        ssd_cache_capacity_bytes=cache_usage["ssd_capacity_bytes"],
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
    print(
        f"  RAM cache usage:      {result.ram_cache_usage_bytes:,.0f} / {result.ram_cache_capacity_bytes:,.0f} bytes"
    )
    print(
        f"  SSD cache usage:      {result.ssd_cache_usage_bytes:,.0f} / {result.ssd_cache_capacity_bytes:,.0f} bytes"
    )
    print(f"  S3 cache usage:       {result.s3_cache_usage_bytes:,.0f} bytes")
    print(f"{'-' * 60}")
    print(f"  Price/hour:           ${result.price_usd_per_hour:.4f}")
    print(f"{'=' * 60}\n")

    return result
