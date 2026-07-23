"""Dynamo-style cost-based router for distributed LLM inference.

The router scores candidate workers using a cost function that combines:

* Active load on the worker (prefill + decode tokens already assigned).
* KV cache locality: cached prefix tokens reduce the effective prefill load.
* Tier-aware credits: device-local RAM is cheapest, then SSD, then S3, then
  remote RAM.

The lowest-cost eligible worker is selected deterministically.
"""

from dataclasses import dataclass

from src.cache.cache import S3_NODE_ID, Cache
from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.logger import LOG_ROUTER, log, should_log
from src.request.request import Request


class RouterCostConfig:
    """Tunable knobs for the Dynamo-style routing cost function."""

    def __init__(
        self,
        prefill_load_scale: float = 1.0,
        active_work_scale: float = 0.0001,
        device_credit: float = 1.0,
        remote_ram_credit: float = 0.5,
        remote_ssd_credit: float = 0.3,
        s3_credit: float = 0.1,
        busy_threshold_tokens: float = 1_000_000.0,
    ) -> None:

        self.prefill_load_scale = prefill_load_scale
        self.active_work_scale = active_work_scale
        self.device_credit = device_credit
        self.remote_ram_credit = remote_ram_credit
        self.remote_ssd_credit = remote_ssd_credit
        self.s3_credit = s3_credit
        self.busy_threshold_tokens = busy_threshold_tokens


@dataclass
class Router:
    queue: list[Request]
    prefill_instances: list[PrefillInstance]
    decode_instances: list[DecodeInstance]
    cache: Cache | None = None
    cost_config: RouterCostConfig | None = None

    def __post_init__(self):
        if self.cost_config is None:
            self.cost_config = RouterCostConfig()

    def route_requests(self):
        """Route every request in ``self.queue`` to the lowest-cost worker.

        Active-token totals are maintained incrementally as decisions are made so
        that the cost of each successive route reflects requests already routed
        in this batch.
        """
        active_prefill: dict[int, float] = self._compute_active_prefill_tokens()
        active_decode: dict[int, float] = self._compute_active_decode_tokens()
        while self.queue:
            req = self.queue.pop(0)
            if should_log(LOG_ROUTER):
                log(LOG_ROUTER, f"Routing request {req.id} with stage {req.stage}")
            if req.stage == "prefill":
                instance = self._choose_prefill_instance(
                    req, active_prefill, active_decode
                )
                node_id = self._node_id(instance)
                active_prefill[node_id] = active_prefill.get(node_id, 0.0) + max(
                    0.0, req.isl - req.prefilled_tokens
                )
            else:
                instance = self._choose_decode_instance(
                    req, active_prefill, active_decode
                )
                node_id = self._node_id(instance)
                active_decode[node_id] = (
                    active_decode.get(node_id, 0.0) + req.isl + req.osl
                )
            instance.add_request(req)

    def _node_id(self, instance: PrefillInstance | DecodeInstance) -> int:
        return instance.node_id

    def _instances_by_node(
        self, instances: list[PrefillInstance] | list[DecodeInstance]
    ) -> dict[int, list[PrefillInstance] | list[DecodeInstance]]:
        mapping: dict[int, list[PrefillInstance] | list[DecodeInstance]] = {}
        for inst in instances:
            mapping.setdefault(inst.node_id, []).append(inst)
        return mapping

    def _prefill_tokens_for_instance(self, inst: PrefillInstance) -> float:
        """Sum of uncached prefill tokens assigned to one prefill instance."""
        total = 0.0
        for req, _ in inst.queue:
            total += max(0.0, req.isl - req.prefilled_tokens)
        for download_req, _ in inst.download_queue:
            total += max(
                0.0,
                download_req.request.isl - download_req.request.prefilled_tokens,
            )
        return total

    def _compute_active_prefill_tokens(self) -> dict[int, float]:
        """Return per-node uncached prefill token totals by scanning once."""
        totals: dict[int, float] = {}
        for inst in self.prefill_instances:
            node_id = inst.node_id
            totals[node_id] = totals.get(
                node_id, 0.0
            ) + self._prefill_tokens_for_instance(inst)
        return totals

    def _compute_active_decode_tokens(self) -> dict[int, float]:
        """Return per-node remaining decode token totals by scanning once.

        The full sequence length (ISL + remaining OSL) is counted so that decode
        routing is aware of both the input context and the output tokens yet to
        be generated.
        """
        totals: dict[int, float] = {}
        for inst in self.decode_instances:
            node_id = inst.node_id
            total = 0.0
            for req, _ in inst.queue:
                total += req.isl + req.osl - req.decoded_tokens
            for download_req, _ in inst.download_queue:
                total += (
                    download_req.request.isl
                    + download_req.request.osl
                    - download_req.request.decoded_tokens
                )
            if inst.current_batch:
                for req in inst.current_batch:
                    total += req.isl + req.osl - req.decoded_tokens
            totals[node_id] = totals.get(node_id, 0.0) + total
        return totals

    def _cached_prefix(self, req: Request, node_id: int) -> int:
        if self.cache is None:
            return 0
        return self.cache.cached_prefix_on_node((req.user_id, req.session_id), node_id)

    def _has_decode_on_node(self, node_id: int) -> bool:
        """Return True if ``node_id`` also hosts decode instances."""
        return any(inst.node_id == node_id for inst in self.decode_instances)

    def _overlap_credit(self, req: Request, node_id: int) -> float:
        """Return the weighted cache-overlap credit for routing ``req`` to ``node_id``."""
        if self.cache is None:
            return 0.0

        cfg = self.cost_config
        assert cfg is not None, "Router cost config must be set"
        session_id = (req.user_id, req.session_id)

        # If the node also hosts decode instances, assume the KV produced here
        # will be consumed locally.  This colocated-affinity bonus uses the
        # existing device_credit parameter to reward same-node placement.
        if self._has_decode_on_node(node_id):
            local_output_credit = req.osl * cfg.device_credit
        else:
            local_output_credit = 0.0

        items = self.cache.find_cache(session_id)
        if not items:
            return local_output_credit

        overlap = min(items[-1].token_end, req.isl)
        if overlap <= 0:
            return local_output_credit

        credit = 0.0
        covered = 0
        for item in items:
            if covered >= overlap:
                break
            layer = item.layer
            if layer is None:
                continue
            seg_start = max(item.token_start, covered)
            seg_end = min(item.token_end, overlap)

            if seg_end <= seg_start:
                continue
            seg_len = seg_end - seg_start
            if (layer.node_id == node_id and layer.name == "RAM") or (
                layer.node_id == node_id and layer.name == "SSD"
            ):
                tier_credit = cfg.device_credit
            elif layer.node_id == S3_NODE_ID:
                tier_credit = cfg.s3_credit
            elif layer.name == "SSD":
                tier_credit = cfg.remote_ssd_credit
            else:
                tier_credit = cfg.remote_ram_credit
            credit += seg_len * tier_credit
            covered = seg_end

        return credit + local_output_credit

    def _prefill_cost(
        self, req: Request, node_id: int, active_prefill: dict[int, float]
    ) -> float:
        cfg = self.cost_config
        assert cfg is not None, "Router cost config must be set"
        # Ignore cached-prefix overlap when routing prefills.  The prefill has
        # to run somewhere and the produced KV will be cached on that node, so
        # giving credit for existing cache entries encourages unhealthy stacking
        # on nodes that happen to hold prior context.  Load-balancing across
        # prefill workers dominates TTFT.
        return cfg.prefill_load_scale * (
            active_prefill.get(node_id, 0.0) * cfg.active_work_scale + req.isl
        )

    def _decode_cost(
        self,
        req: Request,
        node_id: int,
        active_decode: dict[int, float],
    ) -> float:
        cfg = self.cost_config
        assert cfg is not None, "Router cost config must be set"
        overlap = self._overlap_credit(req, node_id)
        adjusted_prefill = max(0.0, req.isl - overlap)
        return (
            active_decode.get(node_id, 0.0) * cfg.active_work_scale
            + adjusted_prefill
            + req.osl
        )

    def _total_cost(
        self,
        req: Request,
        node_id: int,
        is_prefill: bool,
        active_prefill: dict[int, float],
        active_decode: dict[int, float],
    ) -> float:
        decode_cost = self._decode_cost(req, node_id, active_decode)
        if is_prefill:
            prefill_cost = self._prefill_cost(req, node_id, active_prefill)
            # Charge the full decode-stage cost when prefill routing so that
            # colocated workers that will later decode the same request are
            # strongly preferred.  This prevents fast local bandwidth from
            # steering prefills away from the nodes that will consume the KV.
            return prefill_cost + decode_cost
        # Decode routing: only decode-stage costs (load + cache-adjusted work).
        return decode_cost

    def _choose_prefill_instance(
        self,
        req: Request,
        active_prefill: dict[int, float] | None = None,
        active_decode: dict[int, float] | None = None,
    ) -> PrefillInstance:
        cfg = self.cost_config
        assert cfg is not None, "Router cost config must be set"
        candidates = self._instances_by_node(self.prefill_instances)
        assert candidates, "No prefill instances available"

        if active_prefill is None:
            active_prefill = self._compute_active_prefill_tokens()
        if active_decode is None:
            active_decode = self._compute_active_decode_tokens()

        best_node_id: int | None = None
        best_cost = float("inf")
        for node_id in candidates:
            if (
                active_prefill.get(node_id, 0.0) + active_decode.get(node_id, 0.0)
                > cfg.busy_threshold_tokens
            ):
                if should_log(LOG_ROUTER):
                    log(
                        LOG_ROUTER,
                        f"Skipping node {node_id}: above busy threshold",
                    )
                continue
            cost = self._total_cost(
                req,
                node_id,
                is_prefill=True,
                active_prefill=active_prefill,
                active_decode=active_decode,
            )
            if should_log(LOG_ROUTER):
                log(
                    LOG_ROUTER,
                    f"Prefill cost for request {req.id} on node {node_id}: {cost:.1f} "
                    f"(active_prefill={active_prefill.get(node_id, 0.0):.0f}, "
                    f"active_decode={active_decode.get(node_id, 0.0):.0f}, "
                    f"cached_prefix={self.cache.find_cache((req.user_id, req.session_id)) if self.cache else 'No cache set'})"
                    f"totoal cost {cost:.1f})",
                )
            if cost < best_cost:
                best_cost = cost
                best_node_id = node_id

        if best_node_id is None:
            raise RuntimeError("Nodes too busy, extend busy threshhold")

        # Pick the least-loaded prefill instance on the chosen node by uncached
        # prefill tokens, falling back to queue depth for ties.
        node_instances = candidates[best_node_id]
        assert node_instances, f"No prefill instances found for node {best_node_id}"
        assert all(isinstance(inst, PrefillInstance) for inst in node_instances), (
            "All instances must be PrefillInstance"
        )
        return min(
            node_instances,
            key=lambda inst: (
                self._prefill_tokens_for_instance(inst),
                len(inst.queue),
            ),
        )

    def _choose_decode_instance(
        self,
        req: Request,
        active_prefill: dict[int, float] | None = None,
        active_decode: dict[int, float] | None = None,
    ) -> DecodeInstance:
        cfg = self.cost_config
        candidates = self._instances_by_node(self.decode_instances)
        assert candidates, "No decode instances available"

        if active_prefill is None:
            active_prefill = self._compute_active_prefill_tokens()
        if active_decode is None:
            active_decode = self._compute_active_decode_tokens()

        best_node_id: int | None = None
        best_cost = float("inf")
        for node_id in candidates:
            if (
                active_prefill.get(node_id, 0.0) + active_decode.get(node_id, 0.0)
                > cfg.busy_threshold_tokens
            ):
                if should_log(LOG_ROUTER):
                    log(
                        LOG_ROUTER,
                        f"Skipping node {node_id}: above busy threshold",
                    )
                continue
            cost = self._total_cost(
                req,
                node_id,
                is_prefill=False,
                active_prefill=active_prefill,
                active_decode=active_decode,
            )
            if should_log(LOG_ROUTER):
                log(
                    LOG_ROUTER,
                    f"Decode cost for request {req.id} on node {node_id}: {cost:.1f} "
                    f"(active_prefill={active_prefill.get(node_id, 0.0):.0f}, "
                    f"active_decode={active_decode.get(node_id, 0.0):.0f}, "
                    f"cached_prefix={self._cached_prefix(req, node_id)})",
                )
            if cost < best_cost:
                best_cost = cost
                best_node_id = node_id

        if best_node_id is None:
            best_node_id = min(
                candidates,
                key=lambda nid: (
                    active_prefill.get(nid, 0.0) + active_decode.get(nid, 0.0)
                ),
            )

        node_instances = candidates[best_node_id]
        assert node_instances, f"No decode instances found for node {best_node_id}"
        assert all(isinstance(inst, DecodeInstance) for inst in node_instances), (
            "All instances must be DecodeInstance"
        )
        return min(node_instances, key=lambda inst: len(inst.queue))

    def log(self):
        if should_log(LOG_ROUTER):
            log(LOG_ROUTER, f"Router state: {len(self.queue)} requests in router queue")
            for i, instance in enumerate(self.prefill_instances):
                log(
                    LOG_ROUTER,
                    f"Prefill instance {i} queue length: {len(instance.queue)}",
                )
                instance.log()
            for i, instance in enumerate(self.decode_instances):
                log(
                    LOG_ROUTER,
                    f"Decode instance {i} queue length: {len(instance.queue)}",
                )
                instance.log()
