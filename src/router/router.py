"""Dynamo-style cost-based router for distributed LLM inference.

The router scores candidate workers using a cost function that combines:

* Active load on the worker (prefill + decode tokens already assigned).
* KV cache locality: cached prefix tokens reduce the effective prefill load.
* Tier-aware credits: device-local RAM is cheapest, then SSD, then S3, then
  remote RAM.

The lowest-cost eligible worker is selected deterministically.
"""

from dataclasses import dataclass

from src.cache.cache import Cache
from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.logger import LOG_ROUTER, log
from src.request.request import Request


@dataclass
class RouterCostConfig:
    """Tunable knobs for the Dynamo-style routing cost function."""

    prefill_load_scale: float = 1.0
    device_credit: float = 1.0
    remote_ram_credit: float = 0.5
    ssd_credit: float = 0.3
    s3_credit: float = 0.1
    busy_threshold_tokens: float = 1_000_000.0


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
        """Route every request in ``self.queue`` to the lowest-cost worker."""
        while self.queue:
            req = self.queue.pop(0)
            log(LOG_ROUTER, f"Routing request {req.id} with stage {req.stage}")
            if req.stage == "prefill":
                instance = self._choose_prefill_instance(req)
            else:
                instance = self._choose_decode_instance(req)
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

    def _active_prefill_tokens(self, node_id: int | None = None) -> float:
        """Sum of uncached prefill tokens assigned to ``node_id``.

        If ``node_id`` is None, return the sum for every prefill instance.
        """
        total = 0.0
        for inst in self.prefill_instances:
            if node_id is not None and inst.node_id != node_id:
                continue
            total += self._active_prefill_tokens_for_instance(inst)
        return total

    def _active_prefill_tokens_for_instance(self, inst: PrefillInstance) -> float:
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

    def _active_decode_tokens(self, node_id: int) -> float:
        """Sum of output tokens assigned to ``node_id``."""
        total = 0.0
        for inst in self.decode_instances:
            if inst.node_id != node_id:
                continue
            for req, _ in inst.queue:
                total += req.osl - req.decoded_tokens
            for download_req, _ in inst.download_queue:
                total += download_req.request.osl - download_req.request.decoded_tokens
            if inst.current_batch:
                for req in inst.current_batch:
                    total += req.osl - req.decoded_tokens
        return total

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
        effective_end, segments = self.cache._find_download_segments(
            session_id, node_id, req.isl
        )
        overlap = min(effective_end, req.isl)

        # If the node also hosts decode instances, assume the KV produced here
        # will be consumed locally.  This colocated-affinity bonus uses the
        # existing device_credit parameter to reward same-node placement.
        if self._has_decode_on_node(node_id):
            local_output_credit = req.osl * cfg.device_credit
        else:
            local_output_credit = 0.0

        if overlap <= 0:
            return local_output_credit

        # If the node already has the entire prefix locally, full device credit.
        local_prefix = self.cache.cached_prefix_on_node(session_id, node_id)
        if local_prefix >= overlap:
            return overlap * cfg.device_credit + local_output_credit

        # Otherwise weight by the tiers present in the required prefix.
        credit = 0.0
        covered = 0
        for start, end, source_node_id, source_layer_name in segments:
            if covered >= overlap:
                break
            seg_start = max(start, covered)
            seg_end = min(end, overlap)
            if seg_end <= seg_start:
                continue
            seg_len = seg_end - seg_start
            if source_node_id == node_id:
                tier_credit = cfg.device_credit
            elif source_node_id == -1:
                tier_credit = cfg.s3_credit
            elif source_layer_name == "SSD":
                tier_credit = cfg.ssd_credit
            else:
                tier_credit = cfg.remote_ram_credit
            credit += seg_len * tier_credit
            covered = seg_end

        return credit + local_output_credit

    def _prefill_cost(self, req: Request, node_id: int) -> float:
        cfg = self.cost_config
        assert cfg is not None, "Router cost config must be set"
        overlap = self._overlap_credit(req, node_id)
        adjusted_prefill = max(0.0, req.isl - overlap)
        active_prefill = self._active_prefill_tokens(node_id)
        return cfg.prefill_load_scale * (active_prefill + adjusted_prefill)

    def _decode_cost(self, req: Request, node_id: int) -> float:
        active_decode = self._active_decode_tokens(node_id)
        return active_decode + req.osl

    def _total_cost(self, req: Request, node_id: int, is_prefill: bool) -> float:
        prefill_cost = self._prefill_cost(req, node_id)
        decode_cost = self._decode_cost(req, node_id)
        if is_prefill:
            # When prefill routing, only charge the decode stage lightly so
            # colocated workers that will later decode the same request are
            # preferred, but the dominant term remains prefill locality.
            return prefill_cost + 0.1 * decode_cost
        # Decode routing: full prefill + decode cost.
        return prefill_cost + decode_cost

    def _choose_prefill_instance(self, req: Request) -> PrefillInstance:
        cfg = self.cost_config
        assert cfg is not None, "Router cost config must be set"
        candidates = self._instances_by_node(self.prefill_instances)
        assert candidates, "No prefill instances available"

        best_node_id: int | None = None
        best_cost = float("inf")
        for node_id in candidates:
            if (
                self._active_prefill_tokens(node_id)
                + self._active_decode_tokens(node_id)
                > cfg.busy_threshold_tokens
            ):
                log(
                    LOG_ROUTER,
                    f"Skipping node {node_id}: above busy threshold",
                )
                continue
            cost = self._total_cost(req, node_id, is_prefill=True)
            log(
                LOG_ROUTER,
                f"Prefill cost for request {req.id} on node {node_id}: {cost:.1f} "
                f"(active_prefill={self._active_prefill_tokens(node_id):.0f}, "
                f"active_decode={self._active_decode_tokens(node_id):.0f}, "
                f"totoal cost {self._total_cost(req, node_id, is_prefill=True):.1f})",
            )
            if cost < best_cost:
                best_cost = cost
                best_node_id = node_id

        if best_node_id is None:
            # All workers were busy; fall back to lowest raw load.
            best_node_id = min(
                candidates,
                key=lambda nid: (
                    self._active_prefill_tokens(nid) + self._active_decode_tokens(nid)
                ),
            )

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
                self._active_prefill_tokens_for_instance(inst),
                len(inst.queue),
            ),
        )

    def _choose_decode_instance(self, req: Request) -> DecodeInstance:
        cfg = self.cost_config
        candidates = self._instances_by_node(self.decode_instances)
        assert candidates, "No decode instances available"

        best_node_id: int | None = None
        best_cost = float("inf")
        for node_id in candidates:
            if (
                self._active_prefill_tokens(node_id)
                + self._active_decode_tokens(node_id)
                > cfg.busy_threshold_tokens
            ):
                log(
                    LOG_ROUTER,
                    f"Skipping node {node_id}: above busy threshold",
                )
                continue
            cost = self._total_cost(req, node_id, is_prefill=False)
            log(
                LOG_ROUTER,
                f"Decode cost for request {req.id} on node {node_id}: {cost:.1f} "
                f"(active_prefill={self._active_prefill_tokens(node_id):.0f}, "
                f"active_decode={self._active_decode_tokens(node_id):.0f}, "
                f"cached_prefix={self._cached_prefix(req, node_id)})",
            )
            if cost < best_cost:
                best_cost = cost
                best_node_id = node_id

        if best_node_id is None:
            best_node_id = min(
                candidates,
                key=lambda nid: (
                    self._active_prefill_tokens(nid) + self._active_decode_tokens(nid)
                ),
            )

        node_instances = candidates[best_node_id]
        assert node_instances, f"No decode instances found for node {best_node_id}"
        assert all(isinstance(inst, DecodeInstance) for inst in node_instances), (
            "All instances must be DecodeInstance"
        )
        return min(node_instances, key=lambda inst: len(inst.queue))

    def log(self):
        log(LOG_ROUTER, f"Router state: {len(self.queue)} requests in router queue")
        for i, instance in enumerate(self.prefill_instances):
            log(LOG_ROUTER, f"Prefill instance {i} queue length: {len(instance.queue)}")
            instance.log()
        for i, instance in enumerate(self.decode_instances):
            log(LOG_ROUTER, f"Decode instance {i} queue length: {len(instance.queue)}")
            instance.log()
