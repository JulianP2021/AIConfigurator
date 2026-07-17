from __future__ import annotations
import math

from dataclasses import dataclass
from heapq import heappop, heappush

from sortedcontainers import SortedDict

from src.hardware.hardware import Hardware, S3Spec
from src.logger import LOG_CACHE, log, should_log
from src.model.model import Model
from src.request.request import DownloadRequest, Request, TransferLeg, UploadRequest
from src.scheduler.global_clock import GlobalClock


# Sentinel node id used for the shared S3/object-store tier.
S3_NODE_ID = -1
CHUNK_SIZE = 4096


class CacheItem:
    __slots__ = (
        "last_access_ms",
        "last_access_tick",
        "layer",
        "node_id",
        "session_id",
        "token_end",
        "token_start",
    )

    session_id: tuple[int, int]
    token_start: int
    token_end: int
    last_access_tick: int
    last_access_ms: float
    # Back-pointer to the layer holding this item, maintained by CacheLayer
    # insertion/removal.  This makes layer lookup O(1) instead of scanning all
    # layers per item.
    layer: CacheLayer | None

    def __init__(
        self,
        session_id: tuple[int, int],
        token_start: int,
        token_end: int,
    ):
        self.session_id = session_id
        self.token_start = token_start
        self.token_end = token_end
        self.last_access_tick = 0
        self.last_access_ms = 0.0
        self.layer = None

    @property
    def tokens(self):
        return self.token_end - self.token_start


class CacheLayer:
    node_id: int
    name: str

    def __init__(
        self,
        node_id: int,
        name: str,
        content: dict[tuple[int, int], SortedDict[tuple[int, int], CacheItem]]
        | None = None,
    ) -> None:
        self.node_id = node_id
        self.name = name
        # Mapping from session_id (user_id, session_id) to a SortedDict keyed
        # by (token_start, token_end) and sorted by token_start.  This gives
        # O(log N) item lookup and removal, plus O(log N + k) range iteration.
        self.content: dict[tuple[int, int], SortedDict[tuple[int, int], CacheItem]] = (
            content or {}
        )

        # Lazy LRU index.  The heap stores (last_access_tick, id(item), session_id,
        # token_start) tuples.  _lru_tick maps id(item) to its current valid tick;
        # stale heap entries are skipped during pop_lru().  Carrying session_id
        # and token_start lets pop_lru look up the live item directly via the
        # sorted-dict key instead of scanning the session bucket.
        self._lru_heap: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
        self._lru_tick: dict[int, int] = {}

    def _add_item(self, item: CacheItem) -> None:
        """Add ``item`` to this layer's content and set its back-pointer."""
        item_dict = self.content.setdefault(item.session_id, SortedDict())
        item_dict[(item.token_start, item.token_end)] = item
        item.layer = self
        item.node_id = self.node_id

    def _remove_item(self, item: CacheItem) -> None:
        """Remove ``item`` from this layer's content and clear its back-pointer."""
        item_dict = self.content[item.session_id]
        del item_dict[(item.token_start, item.token_end)]
        if not item_dict:
            del self.content[item.session_id]
        item.layer = None
        item.node_id = -1

    def _get_item(
        self, session_id: tuple[int, int], token_start: int, token_end: int
    ) -> CacheItem | None:
        """Return the item with the exact range in this layer, or None."""
        return self.content.get(session_id, SortedDict()).get((token_start, token_end))

    def touch(self, item: CacheItem, tick: int) -> None:
        """Record that ``item`` was accessed at ``tick``."""
        heappush(
            self._lru_heap,
            (tick, id(item), item.session_id, (item.token_start, item.token_end)),
        )
        self._lru_tick[id(item)] = tick
        item.last_access_tick = tick

    def remove_from_lru(self, item: CacheItem) -> None:
        """Mark ``item`` as removed from the LRU index (lazy deletion)."""
        self._lru_tick.pop(id(item), None)

    def pop_lru(self) -> CacheItem | None:
        """Return and remove the least-recently-used live item, or None."""
        while self._lru_heap:
            tick, item_id, session_id, key = heappop(self._lru_heap)
            current_tick = self._lru_tick.get(item_id)
            if current_tick is None or current_tick != tick:
                # Stale heap entry; the item was removed or re-touched.
                continue
            item_dict = self.content.get(session_id)
            if item_dict is not None:
                item = item_dict.get(key)
                if item is not None and id(item) == item_id:
                    self._lru_tick.pop(item_id)
                    return item
            # Item is no longer in content; clean up the tick entry.
            self._lru_tick.pop(item_id, None)
        return None


@dataclass
class Cache:
    layers: dict[int, list[CacheLayer]]
    node_hardware: dict[int, Hardware]
    model: Model
    ram_usage_fraction: float
    ssd_usage_fraction: float
    s3_spec: S3Spec
    ram_capacity_bytes: dict[int, int]
    ssd_capacity_bytes: dict[int, int]
    ram_usage_bytes: dict[int, int]
    ssd_usage_bytes: dict[int, int]
    s3_usage_bytes: int
    _access_tick: int
    _clock: GlobalClock

    # Peak S3 usage observed during the simulation.
    s3_peak_usage_bytes: int = 0

    cost_usd: float = 0.0

    # S3 transfer diagnostics (number of API operations, not bytes).
    s3_upload_requests: int = 0
    s3_download_requests: int = 0

    def __init__(
        self,
        layers: dict,
        node_hardware: dict[int, Hardware],
        model: Model,
        ram_usage_fraction: float = 0.8,
        ssd_usage_fraction: float = 0.8,
        s3_spec: S3Spec | None = None,
        clock: GlobalClock | None = None,
    ):
        self.layers = layers
        self.node_hardware = node_hardware
        self.model = model
        self.ram_usage_fraction = ram_usage_fraction
        self.ssd_usage_fraction = ssd_usage_fraction
        self.s3_spec = s3_spec or S3Spec.from_gbps(enabled=False)
        self.ram_capacity_bytes = {
            node_id: int(hardware.spec.cpu_ram * ram_usage_fraction)
            for node_id, hardware in node_hardware.items()
        }
        self.ssd_capacity_bytes = {
            node_id: int(hardware.spec.nvme_mem * ssd_usage_fraction)
            for node_id, hardware in node_hardware.items()
        }
        self.ram_usage_bytes = dict.fromkeys(node_hardware, 0)
        self.ssd_usage_bytes = dict.fromkeys(node_hardware, 0)
        self.s3_usage_bytes = 0
        self.s3_peak_usage_bytes = 0
        self.s3_upload_requests = 0
        self.s3_download_requests = 0
        self._access_tick = 0
        self._clock = clock or GlobalClock()

        # S3 is a single shared layer keyed by S3_NODE_ID.
        if self.s3_spec.enabled:
            self.layers[S3_NODE_ID] = [CacheLayer(S3_NODE_ID, "S3", {})]

        self._validate_capacity()

    def _validate_capacity(self) -> None:
        """Raise if a node cannot store a minimal 512-token KV item in RAM.

        SSD validation is skipped for nodes that have no local NVMe storage;
        those nodes will evict RAM directly to S3 (if enabled) or drop the item.
        """
        min_item_bytes = self.kv_size(self.model, 512)
        for node_id, hardware in self.node_hardware.items():
            ram_cap = int(hardware.spec.cpu_ram * self.ram_usage_fraction)
            if ram_cap < min_item_bytes:
                raise ValueError(
                    f"Node {node_id} RAM capacity ({ram_cap} bytes with "
                    f"ram_usage_fraction={self.ram_usage_fraction}) is smaller than "
                    f"a 512-token KV item ({min_item_bytes} bytes)"
                )
            raw_ssd_cap = hardware.spec.nvme_mem * self.ssd_usage_fraction
            if 0 < raw_ssd_cap < min_item_bytes:
                raise ValueError(
                    f"Node {node_id} SSD capacity ({int(raw_ssd_cap)} bytes with "
                    f"ssd_usage_fraction={self.ssd_usage_fraction}) is smaller than "
                    f"a 512-token KV item ({min_item_bytes} bytes)"
                )

    @property
    def _min_item_bytes(self) -> int:
        """Bytes required to store the smallest allowed KV item (512 tokens)."""
        return self.kv_size(self.model, 512)

    def _touch(self, item: CacheItem, layer: CacheLayer | None = None) -> None:
        """Update LRU ordering for an accessed item.

        If ``layer`` is given, the layer's lazy LRU index is updated as well.
        The current simulation wall-clock time is recorded so S3 age-based
        eviction can use real elapsed ms instead of logical access ticks.
        """
        self._access_tick += 1
        now_ms = self._clock.time_ms
        if layer is not None:
            layer.touch(item, self._access_tick)
        else:
            item.last_access_tick = self._access_tick
        item.last_access_ms = now_ms

    def _item_size(self, item: CacheItem) -> int:
        return self.kv_size(self.model, item.tokens)

    def _ram_layer(self, node_id: int) -> CacheLayer:
        return self.get_layer(node_id, "RAM")

    def _ssd_layer(self, node_id: int) -> CacheLayer | None:
        if self.ssd_capacity_bytes.get(node_id, 0) == 0:
            return None
        return self.get_layer(node_id, "SSD")

    def _s3_layer(self) -> CacheLayer:
        return self.get_layer(S3_NODE_ID, "S3")

    def usage_summary(self) -> dict[str, int]:
        """Return aggregate cache usage across all nodes/tiers.

        Uses the maintained per-node byte counters and the single S3 counter,
        avoiding an expensive re-walk of every cached item.

        Keys:
            ram_usage_bytes, ssd_usage_bytes, s3_usage_bytes,
            ram_capacity_bytes, ssd_capacity_bytes
        """
        return {
            "ram_usage_bytes": sum(self.ram_usage_bytes.values()),
            "ssd_usage_bytes": sum(self.ssd_usage_bytes.values()),
            "s3_usage_bytes": self.s3_usage_bytes,
            "s3_peak_usage_bytes": self.s3_peak_usage_bytes,
            "ram_capacity_bytes": sum(self.ram_capacity_bytes.values()),
            "ssd_capacity_bytes": sum(self.ssd_capacity_bytes.values()),
        }

    def _s3_covers(self, item: CacheItem) -> bool:
        """Return True if the shared S3 layer already covers ``item``'s range.

        The range is covered when an existing S3 item for the same session fully
        contains ``[item.token_start, item.token_end)``.
        """
        if not self.s3_spec.enabled:
            return False
        s3_layer = self._s3_layer()
        item_dict = s3_layer.content.get(item.session_id)
        if item_dict is None:
            return False
        for existing in item_dict.values():
            if (
                existing.token_start <= item.token_start
                and existing.token_end >= item.token_end
            ):
                return True
        return False

    def _evict_s3_stale(self) -> None:
        """Remove S3 items whose last access is older than the eviction window.

        Called after every upload to S3 so peak S3 memory reflects only recently
        accessed ("hot") objects.
        """
        if not self.s3_spec.enabled or self.s3_spec.eviction_time_ms <= 0:
            return
        s3_layer = self._s3_layer()
        now_ms = self._clock.time_ms
        cutoff_ms = now_ms - self.s3_spec.eviction_time_ms
        for session_id, items in list(s3_layer.content.items()):
            for item in list(items.values()):
                if item.last_access_ms < cutoff_ms:
                    victim_size = self._item_size(item)
                    self.s3_usage_bytes -= victim_size
                    s3_layer._remove_item(item)
                    s3_layer.remove_from_lru(item)
                    if should_log(LOG_CACHE):
                        log(
                            LOG_CACHE,
                            f"Evicted stale S3 KV for request {item.session_id} "
                            f"({(item.token_start, item.token_end)} tokens, {victim_size} bytes), "
                            f"last_access_ms={item.last_access_ms:.3f}, cutoff_ms={cutoff_ms:.3f}",
                        )
            if session_id in s3_layer.content and not s3_layer.content[session_id]:
                s3_layer.content.pop(session_id, None)
        if self.s3_usage_bytes > self.s3_peak_usage_bytes:
            self.s3_peak_usage_bytes = self.s3_usage_bytes

    def upload_to_s3(self, victim: CacheItem, node_id: int) -> TransferLeg | None:
        s3_leg: TransferLeg | None = None
        victim_size = self._item_size(victim)
        node = self.node_hardware.get(node_id)
        s3_upload_ok = (
            self.s3_spec.enabled and node is not None and node.spec.network_inet_up > 0
        )
        if s3_upload_ok and not self._s3_covers(victim):
            s3_layer = self._s3_layer()
            # Merge with any connected existing S3 item for the same session so
            # we keep a single contiguous S3 entry per (user_id, session_id).
            copied = CacheItem(victim.session_id, victim.token_start, victim.token_end)
            self._merge_with_layer_items(s3_layer, copied)
            s3_layer._add_item(copied)
            copied_size = self._item_size(copied)
            self.s3_usage_bytes += copied_size
            if self.s3_usage_bytes > self.s3_peak_usage_bytes:
                self.s3_peak_usage_bytes = self.s3_usage_bytes
            self._touch(copied, s3_layer)
            s3_leg = TransferLeg(victim_size, node_id, S3_NODE_ID, "S3_UPLOAD")
            self.s3_upload_requests += math.ceil(victim.tokens / CHUNK_SIZE)
            self.cost_usd += (
                float(victim_size) / 1024 / 1024 / 1024 * self.s3_spec.S3_UPLOAD_COST_GB
            )
            self.cost_usd += (
                self.s3_spec.S3_UPLOAD_REQ_COSTS
                / 1000
                * math.ceil(victim.tokens / CHUNK_SIZE)
            )

            if should_log(LOG_CACHE):
                log(
                    LOG_CACHE,
                    f"Uploaded SSD-evicted KV for request {copied.session_id} "
                    f"({copied.tokens} tokens, {copied_size} bytes) from node {node_id} to S3",
                )
            # Run S3 stale-object eviction after every upload so the reported
            # peak S3 memory only counts recently-accessed objects.
            self._evict_s3_stale()
        elif s3_upload_ok and self._s3_covers(victim):
            if should_log(LOG_CACHE):
                log(
                    LOG_CACHE,
                    f"Skipped S3 upload for request {victim.session_id} "
                    f"({victim.tokens} tokens, {victim_size} bytes) from node {node_id}: "
                    f"already covered in S3",
                )

        if should_log(LOG_CACHE):
            log(
                LOG_CACHE,
                f"Deleted SSD LRU KV for request {victim.session_id} "
                f"({victim.tokens} tokens, {victim_size} bytes) from node {node_id} SSD",
            )
        return s3_leg

    def _evict_ssd_lru(self, node_id: int) -> TransferLeg | None:
        """Delete the least-recently-used item from a node's SSD layer.

        If S3 is enabled, the victim is copied to the shared S3 layer before
        deletion (unless an equivalent copy already exists there).  Returns an
        S3 upload leg if an upload happened, otherwise None.
        """
        layer = self._ssd_layer(node_id)
        if layer is None:
            return None
        victim = layer.pop_lru()
        if victim is None:
            return None
        victim_size = self._item_size(victim)
        layer._remove_item(victim)
        self.ssd_usage_bytes[node_id] -= victim_size
        if should_log(LOG_CACHE):
            log(
                LOG_CACHE,
                f"Evicted SSD LRU victim for request {victim.session_id} "
                f"({(victim.token_start, victim.token_end)} tokens, {victim_size} bytes) from node {node_id} SSD; "
                f"will attempt S3 upload (enabled={self.s3_spec.enabled})",
            )

        s3_leg = self.upload_to_s3(victim, node_id)
        node = self.node_hardware.get(node_id)
        s3_upload_ok = (
            self.s3_spec.enabled and node is not None and node.spec.network_inet_up > 0
        )
        if s3_leg is None and s3_upload_ok and not self._s3_covers(victim):
            raise RuntimeError(
                f"SSD victim {victim.session_id} range {(victim.token_start, victim.token_end)} "
                f"on node {node_id} was deleted without an S3 copy: S3 does not cover it"
            )
        return s3_leg

    def _evict_ram_to_ssd(self, node_id: int) -> tuple[CacheItem, list[TransferLeg]]:
        """Move the least-recently-used item from RAM to SSD.

        If the SSD layer is full, its LRU item is evicted to S3 (when enabled)
        before deletion.  Returns the moved item and any S3 upload legs produced
        by SSD overflow evictions.
        """
        ram_layer = self._ram_layer(node_id)
        victim = ram_layer.pop_lru()
        if victim is None:
            raise RuntimeError("No RAM item available to evict")
        victim_size = self._item_size(victim)
        ram_layer._remove_item(victim)
        self.ram_usage_bytes[node_id] -= victim_size

        ssd_layer = self._ssd_layer(node_id)
        s3_legs: list[TransferLeg] = []

        if ssd_layer is None:
            # No local SSD: evict RAM directly to S3 (if enabled and upload
            # bandwidth exists) or drop.
            s3_leg = self.upload_to_s3(victim, node_id)
            if not s3_leg:
                raise RuntimeError("No S3 legs!!!")
            return victim, [s3_leg]

        # Merge with any connected existing SSD item for the same session so we
        # keep a single contiguous SSD entry per (user_id, session_id).  This may
        # delete old SSD items and free bytes, so it must happen before we
        # compute the size for the capacity check.
        self._merge_with_layer_items(ssd_layer, victim)
        merged_size = self._item_size(victim)

        # Make room on SSD for the merged victim, evicting SSD LRU to S3
        # synchronously if needed.
        while self.ssd_usage_bytes[node_id] + merged_size > self.ssd_capacity_bytes[
            node_id
        ] and any(item_dict for item_dict in ssd_layer.content.values()):
            s3_leg = self._evict_ssd_lru(node_id)
            if s3_leg is not None:
                s3_legs.append(s3_leg)

        ssd_layer._add_item(victim)
        self.ssd_usage_bytes[node_id] += merged_size
        ssd_layer.touch(victim, self._access_tick)
        if should_log(LOG_CACHE):
            log(
                LOG_CACHE,
                f"Evicted RAM LRU KV for request {victim.session_id} "
                f"([({victim.token_start}, {victim.token_end})] tokens, {merged_size} bytes) "
                f"to node {node_id} SSD",
            )
        return victim, s3_legs

    def _make_room_ram(self, node_id: int, size: int) -> list[TransferLeg]:
        """Evict RAM items to SSD until ``size`` bytes fit.

        Returns the eviction legs generated by the process: SSD_LOCAL write
        legs for the RAM->SSD evictions and S3_UPLOAD legs for any SSD overflow
        evictions that were pushed to S3.
        """
        eviction_legs: list[TransferLeg] = []
        while any(item_dict for item_dict in self._ram_layer(node_id).content.values()):
            # Stop if the new item already fits in the remaining RAM budget.
            if self.ram_usage_bytes[node_id] + size <= self.ram_capacity_bytes[node_id]:
                break
            # Evict the RAM LRU. Capacity validation guarantees each eviction
            # frees at least a 512-token-sized slot, so the loop converges.
            victim, s3_legs = self._evict_ram_to_ssd(node_id)
            victim_size = self._item_size(victim)
            # Only emit a local SSD leg if this node actually has an SSD tier.
            # Nodes without SSD (e.g., Inferentia) evict directly to S3 or drop,
            # and a zero-bandwidth SSD_LOCAL leg would deadlock the scheduler.
            if self._ssd_layer(node_id) is not None:
                eviction_legs.append(
                    TransferLeg(victim_size, node_id, node_id, "SSD_LOCAL")
                )
            eviction_legs.extend(s3_legs)
        return eviction_legs

    def _ensure_node_accounting(self, node_id: int) -> None:
        """Create usage/capacity/peak entries for a node if they do not exist."""
        if node_id == S3_NODE_ID:
            return
        if node_id not in self.ram_capacity_bytes:
            hardware = self.node_hardware.get(node_id)
            if hardware is None:
                raise ValueError(f"Unknown node_id: {node_id}")
            self.ram_capacity_bytes[node_id] = int(
                hardware.spec.cpu_ram * self.ram_usage_fraction
            )
            self.ssd_capacity_bytes[node_id] = int(
                hardware.spec.nvme_mem * self.ssd_usage_fraction
            )
            self.ram_usage_bytes[node_id] = 0
            self.ssd_usage_bytes[node_id] = 0

    def get_layer(self, node_id: int, layer_name: str) -> CacheLayer:
        self._ensure_node_accounting(node_id)
        if self.layers.get(node_id) is None:
            self.layers[node_id] = []
            self.layers[node_id].append(
                CacheLayer(node_id=node_id, name=layer_name, content={})
            )
            layer = self.layers[node_id][0]
        else:
            node_layers = self.layers[node_id]
            if not any(layer.name == layer_name for layer in node_layers):
                node_layers.append(
                    CacheLayer(node_id=node_id, name=layer_name, content={})
                )
                layer = next(
                    (
                        candidate
                        for candidate in node_layers
                        if candidate.name == layer_name
                    ),
                    None,
                )
            else:
                layer = next(
                    (
                        candidate
                        for candidate in node_layers
                        if candidate.name == layer_name
                    ),
                    None,
                )
        assert layer is not None, f"Layer {layer_name} not found for node {node_id}"
        return layer

    @staticmethod
    def _contiguous_prefix_from_sorted(items) -> list[CacheItem]:
        """Return the items that form the longest contiguous prefix [0, N).

        ``items`` must already be sorted by ``token_start``.  Overlapping and
        duplicate ranges are handled by extending coverage to the furthest
        ``token_end`` seen.
        """
        prefix: list[CacheItem] = []
        coverage_end = 0
        for item in items:
            if item.token_start > coverage_end:
                break
            if item.token_end <= coverage_end:
                continue
            prefix.append(item)
            coverage_end = item.token_end
        return prefix

    def _find_all_items(self, session_id: tuple[int, int]) -> list[CacheItem]:
        """Return every cached item for ``session_id`` across all nodes/tiers.

        Items are returned sorted by ``token_start`` so callers can compute the
        contiguous prefix without re-sorting.
        """
        items: list[CacheItem] = []
        for node_layers in self.layers.values():
            for layer in node_layers:
                item_dict = layer.content.get(session_id)
                if item_dict is not None:
                    items.extend(item_dict.values())
        items.sort(key=lambda item: item.token_start)
        return items

    def _contiguous_prefix(self, items: list[CacheItem]) -> list[CacheItem]:
        """Return the items that form the longest contiguous prefix [0, N).

        Items are sorted by ``token_start``.  Overlapping and duplicate ranges
        are handled by extending coverage to the furthest ``token_end`` seen.
        """
        return self._contiguous_prefix_from_sorted(
            sorted(items, key=lambda item: item.token_start)
        )

    def find_cache(
        self, session_id: tuple[int, int], node_id: int | None = None
    ) -> list[CacheItem]:
        """Return the cached items forming the longest contiguous prefix [0, N) sorted.

        If ``node_id`` is given, only items on that node are considered.
        """
        if node_id is None:
            return self._contiguous_prefix_from_sorted(self._find_all_items(session_id))
        layer = self.get_layer(node_id, "RAM")
        ram_dict = layer.content.get(session_id)
        ssd_layer = self._ssd_layer(node_id)
        ssd_dict = ssd_layer.content.get(session_id) if ssd_layer else None
        if ram_dict is None and ssd_dict is None:
            return []
        if ram_dict is None:
            return self._contiguous_prefix_from_sorted(ssd_dict.values())
        if ssd_dict is None:
            return self._contiguous_prefix_from_sorted(ram_dict.values())
        merged: list[CacheItem] = []
        # Merge two sorted views without materialising the full sorted union.
        ram_iter = iter(ram_dict.values())
        ssd_iter = iter(ssd_dict.values())
        ram_next = next(ram_iter, None)
        ssd_next = next(ssd_iter, None)
        while ram_next is not None or ssd_next is not None:
            if ssd_next is None or (
                ram_next is not None and ram_next.token_start <= ssd_next.token_start
            ):
                merged.append(ram_next)
                ram_next = next(ram_iter, None)
            else:
                merged.append(ssd_next)
                ssd_next = next(ssd_iter, None)
        return self._contiguous_prefix_from_sorted(merged)

    def find_cache_layer(self, item: CacheItem) -> CacheLayer | None:
        if item.layer is not None:
            return item.layer
        # Fallback for items created outside CacheLayer._add_item (e.g. tests).
        for _, layers in self.layers.items():
            for layer in layers:
                item_dict = layer.content.get(item.session_id)
                if (
                    item_dict is not None
                    and (item.token_start, item.token_end) in item_dict
                ):
                    return layer
        return None

    def cached_prefix_on_node(self, session_id: tuple[int, int], node_id: int) -> int:
        """Return the longest contiguous cached prefix length on ``node_id``.

        This is a read-only helper used by the router for locality-aware cost
        scoring.  It does not mutate cache state.
        """
        prefix = self.find_cache(session_id, node_id=node_id)
        return prefix[-1].token_end if prefix else 0

    def delete_item(self, item: CacheItem):
        layer = item.layer
        if layer is None:
            raise ValueError("Item not found in cache")
        if layer.name == "RAM":
            self.ram_usage_bytes[layer.node_id] -= self._item_size(item)
        elif layer.name == "SSD":
            self.ssd_usage_bytes[layer.node_id] -= self._item_size(item)
        elif layer.name == "S3":
            self.s3_usage_bytes -= self._item_size(item)
        layer._remove_item(item)
        layer.remove_from_lru(item)

    def _merge_with_layer_items(
        self,
        layer: CacheLayer,
        item: CacheItem,
    ) -> CacheItem:
        """Merge ``item`` with connected existing items for its session in ``layer``.

        The connected component is computed from ``item``'s range: any existing
        item that overlaps or touches the growing cluster is absorbed.  Items
        that are disjoint are left untouched.  Absorbed items are deleted, their
        usage counters are decremented, and they are removed from the layer's
        LRU index.

        ``item`` itself is mutated in place (its token range is expanded to the
        merged cluster) and returned, so callers can preserve the original object
        identity when no merge is needed.

        Under normal session growth successive KV ranges are nested or adjacent
        prefixes, so this maintains a single contiguous entry per
        ``(user_id, session_id)`` in RAM, SSD, and S3.
        """
        item_dict = layer.content.get(item.session_id)
        if item_dict is None or not item_dict:
            return item

        items = list(item_dict.values())
        cluster_start = item.token_start
        cluster_end = item.token_end
        to_delete: set[CacheItem] = set()

        # Fixed-point expansion: keep absorbing items connected to the cluster.
        while True:
            new_connected = [
                existing
                for existing in items
                if existing not in to_delete
                and not (
                    existing.token_end < cluster_start - 1
                    or existing.token_start > cluster_end
                )
            ]
            if not new_connected:
                break
            for existing in new_connected:
                to_delete.add(existing)
                cluster_start = min(cluster_start, existing.token_start)
                cluster_end = max(cluster_end, existing.token_end)

        if not to_delete:
            return item

        for existing in to_delete:
            item_size = self._item_size(existing)
            if layer.name == "RAM":
                self.ram_usage_bytes[layer.node_id] -= item_size
            elif layer.name == "SSD":
                self.ssd_usage_bytes[layer.node_id] -= item_size
            elif layer.name == "S3":
                self.s3_usage_bytes -= item_size
            layer._remove_item(existing)
            layer.remove_from_lru(existing)

        item.token_start = cluster_start
        item.token_end = cluster_end
        return item

    def _merge_into_ram(
        self,
        session_id: tuple[int, int],
        node_id: int,
        token_start: int,
        token_end: int,
    ) -> tuple[CacheItem, list[TransferLeg]]:
        """Create a single contiguous RAM item, deleting overlapping local copies.

        Local source items (same node) are removed because their data becomes
        part of the new merged RAM item.  Source copies on other nodes are kept.
        The inserted item is expanded to cover any local item that overlaps or
        touches it so the resulting RAM entry is the union of the downloaded
        prefix and any local cached data.  Returns the merged ``CacheItem`` and
        any destination-RAM eviction legs produced while making room.
        """
        ssd_layer = self._ssd_layer(node_id)
        layers = [self._ram_layer(node_id)]
        if ssd_layer is not None:
            layers.append(ssd_layer)

        cluster_start = token_start
        cluster_end = token_end
        to_delete: set[CacheItem] = set()

        # Fixed-point expansion: absorb any local item connected to the cluster.
        changed = True
        while changed:
            changed = False
            for layer in layers:
                item_dict = layer.content.get(session_id)
                if item_dict is None:
                    continue
                for existing in list(item_dict.values()):
                    if existing in to_delete:
                        continue
                    if (
                        existing.token_end < cluster_start - 1
                        or existing.token_start > cluster_end
                    ):
                        continue
                    to_delete.add(existing)
                    cluster_start = min(cluster_start, existing.token_start)
                    cluster_end = max(cluster_end, existing.token_end)
                    changed = True

        for existing in to_delete:
            # Items inserted by tests may not have a layer back-pointer.
            if existing.layer is None:
                existing.layer = self.find_cache_layer(existing)
            if existing.layer is not None:
                self.delete_item(existing)

        merged = CacheItem(session_id, cluster_start, cluster_end)
        eviction_legs = self.insert_cache_item(merged, node_id)
        return merged, eviction_legs

    def _find_download_segments(
        self, session_id: tuple[int, int], dest_node_id: int, required_end: int
    ) -> tuple[int, list[tuple[int, int, int, str]]]:
        """Find source segments needed to assemble [0, required_end) on ``dest_node_id``.

        Returns ``(effective_end, segments)`` where each segment is
        ``(start, end, source_node_id, source_layer_name)``.  ``effective_end``
        is the largest token index that can actually be downloaded given the
        globally cached contiguous prefix.  Local SSD items on ``dest_node_id``
        are promoted; remote items and the shared S3 tier are copied.
        """
        all_items = self._find_all_items(session_id)
        global_prefix = self._contiguous_prefix_from_sorted(all_items)

        if should_log(LOG_CACHE):
            all_ranges = [
                (
                    item.token_start,
                    item.token_end,
                    self.find_cache_layer(item).name
                    if self.find_cache_layer(item)
                    else "?",
                    self.find_cache_layer(item).node_id
                    if self.find_cache_layer(item)
                    else -2,
                )
                for item in all_items
            ]
            log(
                LOG_CACHE,
                f"_find_download_segments for session {session_id} dest={dest_node_id} "
                f"required_end={required_end}: all_ranges={all_ranges} "
                f"global_prefix_end={global_prefix[-1].token_end if global_prefix else 0}",
            )

        if not global_prefix:
            return 0, []

        # If the destination already holds a prefix starting at 0 and the global
        # prefix extends further than the request requires, download the full
        # contiguous prefix so the local RAM item can grow by merging with
        # adjacent remote data.
        local_prefix_end = 0
        if local_items := [
            item
            for item in global_prefix
            if (layer := self.find_cache_layer(item)) is not None
            and layer.node_id == dest_node_id
        ]:
            local_prefix = self._contiguous_prefix_from_sorted(local_items)
            if local_prefix and local_prefix[0].token_start <= 0:
                local_prefix_end = local_prefix[-1].token_end

        if local_prefix_end > 0 and global_prefix[-1].token_end > required_end:
            effective_end = global_prefix[-1].token_end
        else:
            effective_end = min(required_end, global_prefix[-1].token_end)

        local_items = [
            item
            for item in all_items
            if (layer := self.find_cache_layer(item)) is not None
            and layer.node_id == dest_node_id
        ]

        remote_items = [
            item
            for item in all_items
            if (layer := self.find_cache_layer(item)) is not None
            and layer.node_id not in (dest_node_id, S3_NODE_ID)
        ]
        s3_items = [
            item
            for item in all_items
            if (layer := self.find_cache_layer(item)) is not None
            and layer.node_id == S3_NODE_ID
        ]
        # Build covering indexes keyed by (token_start, token_end).  Make sure
        # items carry a layer back-pointer, otherwise _covering_item cannot
        # determine the source layer for the returned segment.
        local_index = SortedDict()
        for item in local_items:
            if (
                item.layer is None
                and (layer := self.find_cache_layer(item)) is not None
            ):
                item.layer = layer
            local_index[(item.token_start, item.token_end)] = item

        remote_index = SortedDict()
        for item in remote_items:
            if (
                item.layer is None
                and (layer := self.find_cache_layer(item)) is not None
            ):
                item.layer = layer
            remote_index[(item.token_start, item.token_end)] = item

        s3_index = SortedDict()
        for item in s3_items:
            if (
                item.layer is None
                and (layer := self.find_cache_layer(item)) is not None
            ):
                item.layer = layer
            s3_index[(item.token_start, item.token_end)] = item

        def _covering_item(
            index: SortedDict[tuple[int, int], CacheItem], pos: int
        ) -> CacheItem | None:
            """Return the covering item in ``index`` that extends furthest past ``pos``.

            Several items may cover ``pos``; the longest one is chosen so gaps
            are closed as far as possible.  Returns None if no item covers ``pos``.
            """
            if not index:
                return None
            # All candidates have token_start <= pos.  Scan backwards over those
            # candidates and pick the item with the maximum token_end that
            # actually covers pos.
            idx = index.bisect_right((pos, float("inf")))
            best: CacheItem | None = None
            while idx > 0:
                idx -= 1
                _, item = index.peekitem(idx)
                if item.token_start > pos:
                    continue
                if pos < item.token_end and (
                    best is None or item.token_end > best.token_end
                ):
                    best = item
            return best

        segments: list[tuple[int, int, int, str]] = []

        node = self.node_hardware.get(dest_node_id)
        s3_download_ok = (
            self.s3_spec.enabled
            and node is not None
            and node.spec.network_inet_down > 0
        )

        miss_start = 0
        s3_layer = self._s3_layer()

        while miss_start < effective_end:
            item = _covering_item(local_index, miss_start)
            if item is not None:
                layer = item.layer
                assert layer is not None
                seg_end = min(item.token_end, effective_end)
                segments.append((miss_start, seg_end, layer.node_id, layer.name))
                miss_start = seg_end
                self._touch(item, layer)
                continue
            if miss_start < effective_end:
                item = _covering_item(remote_index, miss_start)
                if item is not None:
                    layer = item.layer
                    assert layer is not None
                    seg_end = min(item.token_end, effective_end)
                    segments.append((miss_start, seg_end, layer.node_id, layer.name))
                    miss_start = seg_end
                    self._touch(item, layer)
                    continue
            if miss_start < effective_end and s3_download_ok:
                item = _covering_item(s3_index, miss_start)
                if item is not None:
                    seg_end = min(item.token_end, effective_end)
                    seg_tokens = seg_end - miss_start
                    segments.append((miss_start, seg_end, S3_NODE_ID, "S3"))
                    miss_start = seg_end
                    self._touch(item, s3_layer)
                    bytes_to_transfer = self.kv_size(self.model, seg_tokens)
                    self.s3_download_requests += 1
                    self.cost_usd += (
                        float(bytes_to_transfer)
                        / 1024
                        / 1024
                        / 1024
                        * self.s3_spec.S3_DOWNLOAD_COST_GB
                    )
                    self.cost_usd += (
                        self.s3_spec.S3_DOWNLOAD_REQ_COSTS
                        / 1000
                        * seg_tokens
                        / CHUNK_SIZE
                    )
                    continue

            break

        if miss_start < effective_end:
            raise Exception(
                f"{[(i.token_start, i.token_end, i.layer.name, i.layer.node_id) for i in all_items], [(i.token_start, i.token_end, i.layer.name, i.layer.node_id) for i in local_items], [(i.token_start, i.token_end, i.layer.name, i.layer.node_id) for i in remote_items], [(i.token_start, i.token_end, i.layer.name, i.layer.node_id) for i in s3_items], miss_start, effective_end}"
            )
        return effective_end, segments

    def insert_cache_item(self, item: CacheItem, node_id: int) -> list[TransferLeg]:
        """Insert an item into a node's RAM layer, evicting to SSD if needed.

        Items for the same ``(user_id, session_id)`` are merged into a single
        contiguous RAM entry.  Returns the synchronous SSD write legs generated
        by evictions.
        """
        layer = self._ram_layer(node_id)

        # Merge with any connected existing RAM item for this session so we keep
        # a single contiguous entry per (user_id, session_id).
        self._merge_with_layer_items(layer, item)
        item_size = self._item_size(item)
        eviction_legs = self._make_room_ram(node_id, item_size)

        layer._add_item(item)
        self.ram_usage_bytes[node_id] += item_size
        self._touch(item, layer)

        if should_log(LOG_CACHE):
            log(
                LOG_CACHE,
                f"Inserted cache item for request {item.session_id} on node {node_id} "
                f"([({item.token_start}, {item.token_end})] tokens, {item_size} bytes), "
                f"RAM usage: {self.ram_usage_bytes[node_id]} / "
                f"{self.ram_capacity_bytes[node_id]} bytes, "
                f"SSD usage: {self.ssd_usage_bytes[node_id]} / "
                f"{self.ssd_capacity_bytes[node_id]} bytes"
                if self.ssd_capacity_bytes.get(node_id, 0) > 0
                else f"RAM usage: {self.ram_usage_bytes[node_id]} / "
                f"{self.ram_capacity_bytes[node_id]} bytes (no SSD)",
            )
        return eviction_legs

    def _build_data_legs(
        self,
        source_layer_name: str,
        source_node_id: int,
        dest_node_id: int,
        bytes_to_transfer: int,
    ) -> list[TransferLeg]:
        """Build physical transfer legs for moving KV from its cache location.

        A remote read from RAM does not need a source-side RAM staging leg: the
        data is already in the source node's host RAM.  A remote read from SSD
        still needs SSD_LOCAL to model reading the SSD into source RAM.  A read
        from S3 is modeled as a single S3_DOWNLOAD leg to the destination node.
        """
        legs: list[TransferLeg] = []

        if source_node_id == S3_NODE_ID:
            return [
                TransferLeg(bytes_to_transfer, S3_NODE_ID, dest_node_id, "S3_DOWNLOAD")
            ]
        if source_layer_name == "SSD":
            # SSD -> source RAM
            legs.append(
                TransferLeg(
                    bytes_to_transfer, source_node_id, source_node_id, "SSD_LOCAL"
                )
            )
        if source_node_id != dest_node_id:
            if source_layer_name == "SSD":
                # SSD -> source RAM before network egress.
                legs.append(
                    TransferLeg(
                        bytes_to_transfer,
                        source_node_id,
                        source_node_id,
                        "RAM_LOCAL",
                    )
                )
            # Inter-node transfer and destination-side placement.
            legs.append(
                TransferLeg(bytes_to_transfer, source_node_id, dest_node_id, "NETWORK")
            )
            legs.append(
                TransferLeg(bytes_to_transfer, dest_node_id, dest_node_id, "RAM_LOCAL")
            )

        return legs

    def upload_kv(self, node_id: int, request: Request) -> UploadRequest:
        """Move KV from GPU RAM to RAM and insert into cache.

        Only the incremental KV bytes (new tokens beyond what was already cached
        on this node) are uploaded. If the node already has the full KV in its
        RAM, no upload transfer is generated.
        """
        cache_key = (request.user_id, request.session_id)
        prior_cache = self.find_cache(cache_key, node_id=node_id)
        prior_cached_tokens = 0
        if prior_cache:
            cache_layer = self.find_cache_layer(prior_cache[-1])

            assert cache_layer is not None, "Cache layer should not be None"
            assert cache_layer.node_id == node_id, (
                f"Cache layer node_id should match the provided node_id for request {request.id}"
            )
            prior_cached_tokens = prior_cache[-1].token_end
            cache_item = CacheItem(
                cache_key,
                prior_cached_tokens,
                request.prefilled_tokens + request.decoded_tokens,
            )
            eviction_legs = self.insert_cache_item(cache_item, node_id)
        else:
            cache_item = CacheItem(
                cache_key, 0, request.prefilled_tokens + request.decoded_tokens
            )
            eviction_legs = self.insert_cache_item(cache_item, node_id)

        current_total_tokens = request.prefilled_tokens + request.decoded_tokens
        new_tokens = max(0, current_total_tokens - prior_cached_tokens)
        bytes_to_transfer = self.kv_size(self.model, new_tokens)

        if should_log(LOG_CACHE):
            prior_layers = (
                [
                    (
                        item.token_start,
                        item.token_end,
                        self.find_cache_layer(item).name
                        if self.find_cache_layer(item)
                        else "?",
                    )
                    for item in prior_cache
                ]
                if prior_cache
                else []
            )
            log(
                LOG_CACHE,
                f"Uploading KV for request {request.id} (user {request.user_id}, session {request.session_id}) to node {node_id}, "
                f"bytes: {bytes_to_transfer}, cache size: {current_total_tokens} tokens, "
                f"new tokens uploaded: {new_tokens}, prior_cached_tokens: {prior_cached_tokens}, "
                f"prior_cache_ranges: {prior_layers}, inserted_range: [({cache_item.token_start}, {cache_item.token_end})]",
            )

        if bytes_to_transfer <= 0:
            raise ValueError(
                f"Zero-byte upload for request {request.id} (user {request.user_id}, "
                f"session {request.session_id}): node {node_id} already has "
                f"{prior_cached_tokens} tokens, current total {current_total_tokens}"
            )

        if eviction_legs and not any(leg.remaining_bytes > 0 for leg in eviction_legs):
            eviction_legs = []

        upload_track = [TransferLeg(bytes_to_transfer, node_id, node_id, "RAM_LOCAL")]
        # The eviction legs are the physical work of making room.  They are
        # returned as a separate track that runs in parallel with the actual
        # upload; however, the request is considered uploaded once the last
        # (upload) track finishes.
        tracks = [eviction_legs, upload_track] if eviction_legs else [upload_track]
        return UploadRequest(request, tracks)

    def download_kv(self, node_id: int, request: Request) -> DownloadRequest:
        """Assemble KV for ``request`` on ``node_id`` RAM from all cached sources.

        The destination receives one contiguous RAM item covering the longest
        cached prefix that can be satisfied.  Source copies on other nodes are
        retained; local SSD sources are promoted (removed from SSD).
        """
        cache_key = (request.user_id, request.session_id)
        required_end = request.isl
        effective_end, segments = self._find_download_segments(
            cache_key, node_id, required_end
        )
        if effective_end == 0:
            if should_log(LOG_CACHE):
                log(
                    LOG_CACHE,
                    f"No cache found for request {request.id} (user {request.user_id}, session {request.session_id})",
                )
            return DownloadRequest(request, [])

        # Verify source segments still exist before we optimistically mutate
        # state.  A segment that disappeared (e.g. evicted by a concurrent upload)
        # would make the computed transfer legs stale and corrupt the cache.
        for start, end, source_node_id, source_layer_name in segments:
            if source_layer_name == "S3":
                layer = self._s3_layer()
            else:
                layer = self.get_layer(source_node_id, source_layer_name)
            item_dict = layer.content.get(cache_key)
            covering_item = None
            if item_dict is not None:
                for item in item_dict.values():
                    if item.token_start <= start and item.token_end >= end:
                        covering_item = item
                        break
            assert covering_item is not None, (
                f"Download source segment [{start},{end}) for request {request.id} "
                f"(user {request.user_id}, session {request.session_id}) no longer "
                f"present on {source_layer_name} node {source_node_id}"
            )

        # Optimistically update cache state: merge everything into one RAM item.
        merged_item, eviction_legs = self._merge_into_ram(
            cache_key, node_id, 0, effective_end
        )

        if merged_item.token_end < request.prefilled_tokens:
            raise RuntimeError(
                f"Download merged_item token_end {merged_item.token_end} < "
                f"request.prefilled_tokens {request.prefilled_tokens} for request {request.id}; "
                f"segments={segments}, all_item={[(i.token_start, i.token_end, i.layer.name, i.layer.node_id) for i in self._find_all_items(cache_key)]}"
            )

        tracks: list[list[TransferLeg]] = []
        if eviction_legs:
            tracks.append(eviction_legs)
        for start, end, source_node_id, source_layer_name in segments:
            if source_node_id == node_id and source_layer_name == "RAM":
                # Already in destination host RAM; still need a local RAM leg
                # to model moving the KV into GPU memory for compute.
                bytes_to_transfer = self.kv_size(self.model, end - start)
                tracks.append([
                    TransferLeg(bytes_to_transfer, node_id, node_id, "RAM_LOCAL")
                ])
                continue
            bytes_to_transfer = self.kv_size(self.model, end - start)
            track = self._build_data_legs(
                source_layer_name, source_node_id, node_id, bytes_to_transfer
            )
            if track:
                tracks.append(track)

        # Use the actual merged cache end, which may be larger than effective_end
        # if the inserted item merged with an existing overlapping cached prefix.
        request.prefilled_tokens = merged_item.token_end
        if should_log(LOG_CACHE):
            log(
                LOG_CACHE,
                f"Downloading KV for request {request.id} (user {request.user_id}, session {request.session_id}) to node {node_id}, "
                f"effective tokens: {effective_end}/{required_end}, "
                f"segments: {segments}, "
                f"tracks: {[[leg.bottleneck for leg in track] for track in tracks]}",
            )
        return DownloadRequest(request, tracks)

    def kv_size(self, model: Model, tokens: int) -> int:
        return model.kv_size_per_token * tokens
