"""Bandwidth-aware router for distributed LLM inference.

The router estimates the wall-clock time a request needs to finish on each
candidate node and picks the node with the lowest expected completion time.
The estimate is composed of:

* Compute time for the remaining prefill/decode work.
* Queue wait based on currently assigned active tokens.
* KV download time from the cached prefix location to the destination node,
  using full (unshared) bandwidth for each leg.

This makes routing sensitive to the physical cost of moving KV across tiers and
nodes, not just to token counts or static credit weights.

When multiple nodes produce exactly the same estimated completion time, the tie
is broken by RAM fill factor: the node with the most free RAM is preferred so
that newly produced or fetched KV blocks are less likely to be forced to SSD or
to trigger a cross-node transfer.  If the fill factor is also tied, a random
choice is made using the configured ``random_seed`` so repeated runs with the
same seed produce identical routing decisions.
"""

import random

from dataclasses import dataclass, field

from src.cache.cache import S3_NODE_ID, Cache
from src.hardware.hardware import Hardware, HardwareSpec
from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.logger import LOG_ROUTER, log, should_log
from src.model.model import Model
from src.request.request import Request


class RouterCostConfig:
    """Deprecated container kept for backward compatibility.

    The bandwidth-aware router no longer uses these scalar credit weights; it
    estimates per-node completion time from compute, queue wait, and full-bandwidth
    KV transfer time.  The class remains so existing callers and serialized
    configs can still construct a RouterCostConfig without crashing.
    """

    def __init__(
        self,
        prefill_load_scale: float = 1.0,
        active_work_scale: float = 0.0001,
        device_credit: float = 1.0,
        remote_ram_credit: float = 0.5,
        remote_ssd_credit: float = 0.3,
        s3_credit: float = 0.1,
    ) -> None:

        self.prefill_load_scale = prefill_load_scale
        self.active_work_scale = active_work_scale
        self.device_credit = device_credit
        self.remote_ram_credit = remote_ram_credit
        self.remote_ssd_credit = remote_ssd_credit
        self.s3_credit = s3_credit


@dataclass
class Router:
    queue: list[Request]
    prefill_instances: list[PrefillInstance]
    decode_instances: list[DecodeInstance]
    cache: Cache | None = None
    cost_config: RouterCostConfig | None = None
    random_seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    model: Model | None = None

    def __post_init__(self):
        if self.cost_config is None:
            self.cost_config = RouterCostConfig()
        self._rng = random.Random(self.random_seed)
        if self.model is None:
            self.model = self._infer_model()

    def _infer_model(self) -> Model | None:
        """Pick a representative model from the attached instances."""
        for inst in list(self.prefill_instances) + list(self.decode_instances):
            model = getattr(inst, "model", None)
            if isinstance(model, Model):
                return model
        return None

    @property
    def _node_hardware(self) -> dict[int, Hardware]:
        if self.cache is not None:
            return self.cache.node_hardware
        # Fall back to instance.hardware when no cache is present. If the
        # instance itself only has a raw spec object, wrap it minimally.
        result: dict[int, Hardware] = {}
        for inst in list(self.prefill_instances) + list(self.decode_instances):
            if isinstance(inst.hardware, Hardware):
                result[inst.node_id] = inst.hardware
            elif hasattr(inst.hardware, "spec"):
                result[inst.node_id] = Hardware(name="test", spec=inst.hardware.spec)
        return result

    def route_requests(self):
        """Route every request in ``self.queue`` to the lowest-cost worker.

        Active-token totals are maintained incrementally as decisions are made so
        that the cost of each successive route reflects requests already routed
        in this batch.
        """
        if not self.queue:
            return
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
                log(
                    LOG_ROUTER,
                    f"Routed request {req.id} with stage {req.stage} to PREFILL {instance.instance_id} (node {node_id})",
                )
            else:
                instance = self._choose_decode_instance(
                    req, active_prefill, active_decode
                )
                node_id = self._node_id(instance)
                active_decode[node_id] = (
                    active_decode.get(node_id, 0.0) + req.isl + req.osl
                )
                log(
                    LOG_ROUTER,
                    f"Routed request {req.id} with stage {req.stage} to DECODE {instance.instance_id} (node {node_id})",
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

    def _node_spec(self, node_id: int) -> HardwareSpec | None:
        hw = self._node_hardware.get(node_id)
        return hw.spec if hw is not None else None

    def _model(self) -> Model | None:
        """Return the explicitly supplied model, or infer one from instances."""
        if self.model is not None:
            return self.model
        return self._infer_model()

    def _compute_active_prefill_tokens(self) -> dict[int, float]:
        """Return per-node uncached prefill token totals.

        Uses each instance's maintained ``active_prefill_tokens`` counter so the
        router avoids scanning queues for every routing decision.  Instances
        built without ``__init__`` (e.g. test doubles) fall back to scanning.
        """
        totals: dict[int, float] = {}
        for inst in self.prefill_instances:
            node_id = inst.node_id
            total = getattr(inst, "active_prefill_tokens", None)
            if total is None:
                total = 0.0
                for req, _ in inst.queue:
                    total += max(0.0, req.isl - req.prefilled_tokens)
                for download_req, _ in inst.download_queue:
                    req = download_req.request
                    total += max(0.0, req.isl - req.prefilled_tokens)
            totals[node_id] = totals.get(node_id, 0.0) + total
        return totals

    def _compute_active_decode_tokens(self) -> dict[int, float]:
        """Return per-node queued decode token totals.

        Uses each instance's maintained ``active_decode_tokens`` counter so the
        router avoids scanning queues for every routing decision.  Instances
        built without ``__init__`` (e.g. test doubles) fall back to scanning.
        """
        totals: dict[int, float] = {}
        for inst in self.decode_instances:
            node_id = inst.node_id
            total = getattr(inst, "active_decode_tokens", None)
            if total is None:
                total = 0.0
                for req, _ in inst.queue:
                    total += req.isl + req.osl
                for download_req, _ in inst.download_queue:
                    total += download_req.request.isl + download_req.request.osl
            totals[node_id] = totals.get(node_id, 0.0) + total
        return totals

    def _cached_prefix(self, req: Request, node_id: int) -> int:
        if self.cache is None:
            return 0
        return self.cache.cached_prefix_on_node((req.user_id, req.session_id), node_id)

    def _ram_fill_factor(self, node_id: int) -> float:
        """Return the fraction of node RAM capacity currently in use.

        Returns 0.0 when no cache is attached or the node has zero RAM capacity,
        so the tie-breaker is neutral in those cases.
        """
        if self.cache is None:
            return 0.0
        capacity = self.cache.ram_capacity_bytes.get(node_id, 0)
        if capacity <= 0:
            return 0.0
        return float(self.cache.ram_usage_bytes.get(node_id, 0)) / float(capacity)

    def _has_decode_on_node(self, node_id: int) -> bool:
        """Return True if ``node_id`` also hosts decode instances."""
        return any(inst.node_id == node_id for inst in self.decode_instances)

    def _tiebreak_by_ram_fill(self, node_ids: list[int]) -> int:
        """Return the node with the lowest RAM fill factor.

        When several nodes have the same routing cost, prefer the one with the
        most free RAM. This reduces the chance that a newly produced or fetched
        KV block is forced to SSD or triggers an expensive cross-node transfer.
        Falls back to the previous random tie-breaker if no cache is available
        or all nodes have the same fill factor.
        """
        if not node_ids:
            raise ValueError("Cannot tie-break an empty node list")
        if len(node_ids) == 1:
            return node_ids[0]
        by_fill = sorted(
            node_ids,
            key=lambda nid: self._ram_fill_factor(nid),
        )
        log(
            LOG_ROUTER,
            f"{by_fill}, {node_ids}, {[(self._has_decode_on_node(nid), self._ram_fill_factor(nid)) for nid in node_ids]}",
        )
        lowest_fill = self._ram_fill_factor(by_fill[0])
        # Include every node that shares the lowest fill factor.
        tied = [nid for nid in by_fill if self._ram_fill_factor(nid) == lowest_fill]
        log(
            LOG_ROUTER,
            f"Tied fill decision: {tied}",
        )
        tied = sorted(
            tied,
            key=lambda nid: self._has_decode_on_node(nid),
        )
        return tied[0]

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

        items = self.cache.find_cache(session_id, node_id=node_id)
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

    def _num_prefill_instances_on_node(self, node_id: int) -> int:
        return sum(1 for inst in self.prefill_instances if inst.node_id == node_id)

    def _num_decode_instances_on_node(self, node_id: int) -> int:
        return sum(1 for inst in self.decode_instances if inst.node_id == node_id)

    def _prefill_cost(
        self, req: Request, node_id: int, active_prefill: dict[int, float]
    ) -> float:
        """Dynamo-style prefill cost using active load and overlap credit."""
        cfg = self.cost_config
        assert cfg is not None, "Router cost config must be set"
        overlap = self._overlap_credit(req, node_id)
        return cfg.prefill_load_scale * max(
            0.0, req.isl - req.prefilled_tokens - overlap
        ) + cfg.active_work_scale * active_prefill.get(node_id, 0.0)

    def _decode_cost(
        self,
        req: Request,
        node_id: int,
        active_decode: dict[int, float],
    ) -> float:
        """Dynamo-style decode cost using active load and overlap credit."""
        cfg = self.cost_config
        assert cfg is not None, "Router cost config must be set"
        overlap = self._overlap_credit(req, node_id)
        adjusted_prefill = max(0.0, req.isl - overlap)
        return (
            cfg.active_work_scale
            * cfg.active_work_scale
            * cfg.active_work_scale
            * active_decode.get(node_id, 0.0)
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
        # Dynamo-style cost model from the original implementation.
        decode_cost = self._decode_cost(req, node_id, active_decode)
        if is_prefill:
            prefill_cost = self._prefill_cost(req, node_id, active_prefill)
            return prefill_cost + 0.1 * decode_cost
        return decode_cost

    def _choose_prefill_instance(
        self,
        req: Request,
        active_prefill: dict[int, float] | None = None,
        active_decode: dict[int, float] | None = None,
    ) -> PrefillInstance:
        candidates = self._instances_by_node(self.prefill_instances)
        assert candidates, "No prefill instances available"

        if active_prefill is None:
            active_prefill = self._compute_active_prefill_tokens()
        if active_decode is None:
            active_decode = self._compute_active_decode_tokens()

        best_cost = float("inf")
        best_nodes: list[int] = []
        for node_id in candidates:
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
                    f"cached_prefix={self.cache.find_cache((req.user_id, req.session_id), node_id=node_id) if self.cache else 'No cache set'})"
                    f"totoal cost {cost:.1f})",
                )
            if cost < best_cost:
                best_cost = cost
                best_nodes = [node_id]
            elif cost == best_cost:
                best_nodes.append(node_id)
        best_node_id = self._tiebreak_by_ram_fill(best_nodes)

        # Pick the least-loaded prefill instance on the chosen node by remaining
        # prefill compute time, falling back to queue depth for ties.
        node_instances = candidates[best_node_id]
        assert node_instances, f"No prefill instances found for node {best_node_id}"
        assert all(isinstance(inst, PrefillInstance) for inst in node_instances), (
            "All instances must be PrefillInstance"
        )
        return min(
            node_instances,
            key=lambda inst: len(inst.queue) + len(inst.download_queue),
        )

    def _choose_decode_instance(
        self,
        req: Request,
        active_prefill: dict[int, float] | None = None,
        active_decode: dict[int, float] | None = None,
    ) -> DecodeInstance:
        candidates = self._instances_by_node(self.decode_instances)
        assert candidates, "No decode instances available"

        if active_prefill is None:
            active_prefill = self._compute_active_prefill_tokens()
        if active_decode is None:
            active_decode = self._compute_active_decode_tokens()

        best_cost = float("inf")
        best_nodes: list[int] = []
        for node_id in candidates:
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
                best_nodes = [node_id]
            elif cost == best_cost:
                best_nodes.append(node_id)

        best_node_id = self._tiebreak_by_ram_fill(best_nodes)

        node_instances = candidates[best_node_id]
        assert node_instances, f"No decode instances found for node {best_node_id}"
        assert all(isinstance(inst, DecodeInstance) for inst in node_instances), (
            "All instances must be DecodeInstance"
        )
        return min(
            node_instances,
            key=lambda inst: len(inst.queue) + len(inst.download_queue),
        )

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
