from __future__ import annotations
from dataclasses import dataclass
from heapq import heappop, heappush

from src.hardware.hardware import Hardware, S3Spec
from src.logger import LOG_CACHE, log
from src.model.model import Model
from src.request.request import DownloadRequest, Request, TransferLeg, UploadRequest
from src.scheduler.global_clock import GlobalClock


# Sentinel node id used for the shared S3/object-store tier.
S3_NODE_ID = -1
CHUNK_SIZE = 4096


@dataclass
class CacheItem:
    session_id: tuple[int, int]
    token_start: int
    token_end: int = 0
    last_access_tick: int = 0
    last_access_ms: float = 0.0
    # Back-pointer to the layer holding this item, maintained by CacheLayer
    # insertion/removal.  This makes layer lookup O(1) instead of scanning all
    # layers per item.
    layer: CacheLayer | None = None

    def __init__(
        self, session_id: tuple[int, int], token_start: int, token_end: int = 0
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
        content: dict[tuple[int, int], dict[tuple[int, int], CacheItem]] | None = None,
    ) -> None:
        self.node_id = node_id
        self.name = name
        # Mapping from session_id (user_id, session_id) to a dict keyed by
        # (token_start, token_end).  This gives O(1) item lookup and removal
        # within a session.  Callers sort when needed.
        self.content: dict[tuple[int, int], dict[tuple[int, int], CacheItem]] = (
            content or {}
        )

        # Lazy LRU index.  The heap stores (last_access_tick, id(item), session_id)
        # tuples.  _lru_tick maps id(item) to its current valid tick; stale heap
        # entries are skipped during pop_lru().  Carrying session_id lets pop_lru
        # look up the live item directly in self.content[session_id] instead of
        # scanning every bucket in the layer.
        self._lru_heap: list[tuple[int, int, tuple[int, int]]] = []
        self._lru_tick: dict[int, int] = {}

    def _add_item(self, item: CacheItem) -> None:
        """Add ``item`` to this layer's content and set its back-pointer."""
        item_dict = self.content.setdefault(item.session_id, {})
        item_dict[(item.token_start, item.token_end)] = item
        item.layer = self

    def _remove_item(self, item: CacheItem) -> None:
        """Remove ``item`` from this layer's content and clear its back-pointer."""
        item_dict = self.content[item.session_id]
        del item_dict[(item.token_start, item.token_end)]
        if not item_dict:
            del self.content[item.session_id]
        item.layer = None

    def _get_item(
        self, session_id: tuple[int, int], token_start: int, token_end: int
    ) -> CacheItem | None:
        """Return the item with the exact range in this layer, or None."""
        return self.content.get(session_id, {}).get((token_start, token_end))

    def touch(self, item: CacheItem, tick: int) -> None:
        """Record that ``item`` was accessed at ``tick``."""
        heappush(self._lru_heap, (tick, id(item), item.session_id))
        self._lru_tick[id(item)] = tick
        item.last_access_tick = tick

    def remove_from_lru(self, item: CacheItem) -> None:
        """Mark ``item`` as removed from the LRU index (lazy deletion)."""
        self._lru_tick.pop(id(item), None)

    def pop_lru(self) -> CacheItem | None:
        """Return and remove the least-recently-used live item, or None."""
        while self._lru_heap:
            tick, item_id, session_id = heappop(self._lru_heap)
            current_tick = self._lru_tick.get(item_id)
            if current_tick is None or current_tick != tick:
                # Stale heap entry; the item was removed or re-touched.
                continue
            # Look up the item directly in its session bucket by id.
            for item in self.content.get(session_id, {}).values():
                if id(item) == item_id:
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
        """Raise if a node cannot store a minimal 512-token KV item in RAM/SSD."""
        min_item_bytes = self.kv_size(self.model, 512)
        for node_id, hardware in self.node_hardware.items():
            ram_cap = int(hardware.spec.cpu_ram * self.ram_usage_fraction)
            ssd_cap = int(hardware.spec.nvme_mem * self.ssd_usage_fraction)
            if ram_cap < min_item_bytes:
                raise ValueError(
                    f"Node {node_id} RAM capacity ({ram_cap} bytes with "
                    f"ram_usage_fraction={self.ram_usage_fraction}) is smaller than "
                    f"a 512-token KV item ({min_item_bytes} bytes)"
                )
            if ssd_cap < min_item_bytes:
                raise ValueError(
                    f"Node {node_id} SSD capacity ({ssd_cap} bytes with "
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

    def _ssd_layer(self, node_id: int) -> CacheLayer:
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

    def _has_s3_equivalent(self, item: CacheItem) -> bool:
        """Return True if an equivalent item already exists in the shared S3 layer."""
        if not self.s3_spec.enabled:
            return False
        s3_layer = self._s3_layer()
        return any(
            existing.session_id == item.session_id
            and existing.token_start == item.token_start
            and existing.token_end == item.token_end
            for existing in s3_layer.content.get(item.session_id, {}).values()
        )

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
                    log(
                        LOG_CACHE,
                        f"Evicted stale S3 KV for request {item.session_id} "
                        f"({item.tokens} tokens, {victim_size} bytes), "
                        f"last_access_ms={item.last_access_ms:.3f}, cutoff_ms={cutoff_ms:.3f}",
                    )
            if not s3_layer.content.get(session_id):
                s3_layer.content.pop(session_id, None)
        if self.s3_usage_bytes > self.s3_peak_usage_bytes:
            self.s3_peak_usage_bytes = self.s3_usage_bytes

    def _evict_ssd_lru(self, node_id: int) -> TransferLeg | None:
        """Delete the least-recently-used item from a node's SSD layer.

        If S3 is enabled, the victim is copied to the shared S3 layer before
        deletion (unless an equivalent copy already exists there).  Returns an
        S3 upload leg if an upload happened, otherwise None.
        """
        layer = self._ssd_layer(node_id)
        victim = layer.pop_lru()
        if victim is None:
            return None
        victim_size = self._item_size(victim)
        layer._remove_item(victim)
        self.ssd_usage_bytes[node_id] -= victim_size

        s3_leg: TransferLeg | None = None
        if self.s3_spec.enabled and not self._has_s3_equivalent(victim):
            s3_layer = self._s3_layer()
            copied = CacheItem(victim.session_id, victim.token_start, victim.token_end)
            s3_layer._add_item(copied)
            self._touch(copied, s3_layer)
            self.s3_usage_bytes += victim_size
            if self.s3_usage_bytes > self.s3_peak_usage_bytes:
                self.s3_peak_usage_bytes = self.s3_usage_bytes
            s3_leg = TransferLeg(victim_size, node_id, S3_NODE_ID, "S3_UPLOAD")
            self.s3_upload_requests += 1
            self.cost_usd += (
                float(victim_size) / 1024 / 1024 / 1024 * self.s3_spec.S3_UPLOAD_COST_GB
            )
            self.cost_usd += (
                self.s3_spec.S3_UPLOAD_REQ_COSTS / 1000 * victim.tokens / CHUNK_SIZE
            )

            log(
                LOG_CACHE,
                f"Uploaded SSD-evicted KV for request {victim.session_id} "
                f"({victim.tokens} tokens, {victim_size} bytes) from node {node_id} to S3",
            )
            # Run S3 stale-object eviction after every upload so the reported
            # peak S3 memory only counts recently-accessed objects.
            self._evict_s3_stale()

        log(
            LOG_CACHE,
            f"Deleted SSD LRU KV for request {victim.session_id} "
            f"({victim.tokens} tokens, {victim_size} bytes) from node {node_id} SSD",
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

        # Make room on SSD, evicting SSD LRU to S3 synchronously if needed.
        s3_legs: list[TransferLeg] = []
        while self.ssd_usage_bytes[node_id] + victim_size > self.ssd_capacity_bytes[
            node_id
        ] and any(item_dict for item_dict in self._ssd_layer(node_id).content.values()):
            s3_leg = self._evict_ssd_lru(node_id)
            if s3_leg is not None:
                s3_legs.append(s3_leg)

        ssd_layer = self._ssd_layer(node_id)
        ssd_layer._add_item(victim)
        self.ssd_usage_bytes[node_id] += victim_size
        ssd_layer.touch(victim, self._access_tick)
        log(
            LOG_CACHE,
            f"Evicted RAM LRU KV for request {victim.session_id} "
            f"({victim.tokens} tokens, {victim_size} bytes) to node {node_id} SSD",
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

    def _find_all_items(self, session_id: tuple[int, int]) -> list[CacheItem]:
        """Return every cached item for ``session_id`` across all nodes/tiers."""
        items: list[CacheItem] = []
        for node_layers in self.layers.values():
            for layer in node_layers:
                for item in layer.content.get(session_id, {}).values():
                    items.append(item)
        return items

    def _contiguous_prefix(self, items: list[CacheItem]) -> list[CacheItem]:
        """Return the items that form the longest contiguous prefix [0, N).

        Items are sorted by ``token_start``.  Overlapping and duplicate ranges
        are handled by extending coverage to the furthest ``token_end`` seen.
        """
        sorted_items = sorted(items, key=lambda item: item.token_start)
        prefix: list[CacheItem] = []
        coverage_end = 0
        for item in sorted_items:
            if item.token_start > coverage_end:
                break
            if item.token_end <= coverage_end:
                continue
            prefix.append(item)
            coverage_end = item.token_end
        return prefix

    def find_cache(
        self, session_id: tuple[int, int], node_id: int | None = None
    ) -> list[CacheItem]:
        """Return the cached items forming the longest contiguous prefix [0, N).

        If ``node_id`` is given, only items on that node are considered.
        """
        items = self._find_all_items(session_id)
        if node_id is not None:
            items = [
                item
                for item in items
                if (layer := self.find_cache_layer(item)) is not None
                and layer.node_id == node_id
            ]
        return self._contiguous_prefix(items)

    def find_cache_layer(self, item: CacheItem) -> CacheLayer | None:
        if item.layer is not None:
            return item.layer
        # Fallback for items created outside CacheLayer._add_item (e.g. tests).
        for _, layers in self.layers.items():
            for layer in layers:
                if (
                    item.session_id in layer.content
                    and (item.token_start, item.token_end)
                    in layer.content[item.session_id]
                ):
                    return layer
        return None

    def cached_prefix_on_node(self, session_id: tuple[int, int], node_id: int) -> int:
        """Return the longest contiguous cached prefix length on ``node_id``.

        This is a read-only helper used by the router for locality-aware cost
        scoring.  It does not mutate cache state.
        """
        items = [
            item
            for item in self._find_all_items(session_id)
            if (layer := self.find_cache_layer(item)) is not None
            and layer.node_id == node_id
        ]
        prefix = self._contiguous_prefix(items)
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

    def _merge_into_ram(
        self,
        session_id: tuple[int, int],
        node_id: int,
        token_start: int,
        token_end: int,
    ) -> list[TransferLeg]:
        """Create a single contiguous RAM item, deleting overlapping local copies.

        Local source items (same node) are removed because their data becomes
        part of the new merged RAM item.  Source copies on other nodes are kept.
        Returns any destination-RAM eviction legs produced while making room.
        """
        for layer in (self._ram_layer(node_id), self._ssd_layer(node_id)):
            for existing in list(layer.content.get(session_id, {}).values()):
                if (
                    existing.token_end <= token_start
                    or existing.token_start >= token_end
                ):
                    continue
                # Items inserted by tests may not have a layer back-pointer.
                if existing.layer is None:
                    existing.layer = layer
                self.delete_item(existing)

        merged = CacheItem(session_id, token_start, token_end)
        return self.insert_cache_item(merged, node_id)

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
        global_prefix = self._contiguous_prefix(all_items)
        if not global_prefix:
            return 0, []
        effective_end = min(required_end, global_prefix[-1].token_end)

        local_items = [
            item
            for item in all_items
            if (layer := self.find_cache_layer(item)) is not None
            and layer.node_id == dest_node_id
        ]
        local_prefix = self._contiguous_prefix(local_items)
        local_end = local_prefix[-1].token_end if local_prefix else 0
        local_end = min(local_end, effective_end)

        segments: list[tuple[int, int, int, str]] = []

        # Promote local SSD portions that are already on the destination node.
        for item in local_prefix:
            layer = self.find_cache_layer(item)
            if layer is None or layer.name != "SSD":
                continue
            seg_end = min(item.token_end, effective_end)
            segments.append((item.token_start, seg_end, layer.node_id, layer.name))

        # Fetch any remaining ranges from remote node sources first.
        miss_start = local_end
        while miss_start < effective_end:
            source: tuple[CacheItem, CacheLayer] | None = None
            for item in all_items:
                layer = self.find_cache_layer(item)
                if (
                    layer is None
                    or layer.node_id == dest_node_id
                    or layer.node_id == S3_NODE_ID
                ):
                    continue
                if item.token_start <= miss_start < item.token_end:
                    source = (item, layer)
                    break
            if source is None:
                break
            item, layer = source
            seg_end = min(item.token_end, effective_end)
            segments.append((miss_start, seg_end, layer.node_id, layer.name))
            miss_start = seg_end

        # S3 fallback for anything still missing.
        if miss_start < effective_end and self.s3_spec.enabled:
            s3_layer = self._s3_layer()
            for item in all_items:
                layer = self.find_cache_layer(item)
                if layer is None or layer.node_id != S3_NODE_ID:
                    continue
                if item.token_start <= miss_start < item.token_end:
                    seg_end = min(item.token_end, effective_end)
                    segments.append((miss_start, seg_end, S3_NODE_ID, "S3"))
                    # Reading from S3 refreshes the object's access time so it
                    # is not evicted while still being actively downloaded.
                    self._touch(item, s3_layer)

                    tokens = seg_end - miss_start
                    bytes_to_transfer = self.kv_size(self.model, tokens)
                    self.s3_download_requests += 1
                    self.cost_usd += (
                        float(bytes_to_transfer)
                        / 1024
                        / 1024
                        / 1024
                        * self.s3_spec.S3_DOWNLOAD_COST_GB
                    )
                    self.cost_usd += (
                        self.s3_spec.S3_DOWNLOAD_REQ_COSTS / 1000 * tokens / CHUNK_SIZE
                    )
                    miss_start = seg_end
                    break

        if miss_start < effective_end:
            # Coverage gap: stop at what we can satisfy.
            effective_end = miss_start

        return effective_end, segments

    def insert_cache_item(self, item: CacheItem, node_id: int) -> list[TransferLeg]:
        """Insert an item into a node's RAM layer, evicting to SSD if needed.

        Returns the synchronous SSD write legs generated by evictions.
        """
        layer = self._ram_layer(node_id)
        item_size = self._item_size(item)
        eviction_legs = self._make_room_ram(node_id, item_size)

        layer._add_item(item)
        self.ram_usage_bytes[node_id] += item_size
        self._touch(item, layer)

        log(
            LOG_CACHE,
            f"Inserted cache item for request {item.session_id} on node {node_id} "
            f"({item.tokens} tokens, {item_size} bytes), "
            f"RAM usage: {self.ram_usage_bytes[node_id]} / "
            f"{self.ram_capacity_bytes[node_id]} bytes, "
            f"SSD usage: {self.ssd_usage_bytes[node_id]} / "
            f"{self.ssd_capacity_bytes[node_id]} bytes",
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

        log(
            LOG_CACHE,
            f"Uploading KV for request {request.id} (user {request.user_id}, session {request.session_id}) to node {node_id}, "
            f"bytes: {bytes_to_transfer}, cache size: {current_total_tokens} tokens, "
            f"new tokens uploaded: {new_tokens}",
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
            log(
                LOG_CACHE,
                f"No cache found for request {request.id} (user {request.user_id}, session {request.session_id})",
            )
            return DownloadRequest(request, [])

        # Optimistically update cache state: merge everything into one RAM item.
        eviction_legs = self._merge_into_ram(cache_key, node_id, 0, effective_end)

        tracks: list[list[TransferLeg]] = []
        if eviction_legs:
            tracks.append(eviction_legs)
        for start, end, source_node_id, source_layer_name in segments:
            if source_node_id == node_id and source_layer_name == "RAM":
                # Already in destination RAM; no physical transfer needed.
                continue
            bytes_to_transfer = self.kv_size(self.model, end - start)
            track = self._build_data_legs(
                source_layer_name, source_node_id, node_id, bytes_to_transfer
            )
            if track:
                tracks.append(track)

        request.prefilled_tokens = effective_end
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
