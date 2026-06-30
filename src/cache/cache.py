from dataclasses import dataclass

from src.hardware.hardware import Hardware
from src.logger import LOG_CACHE, log
from src.model.model import Model
from src.request.request import DownloadRequest, Request, TransferLeg, UploadRequest


@dataclass
class CacheItem:
    req_id: int
    token_start: int
    token_end: int = 0
    last_access_tick: int = 0

    def __init__(self, req_id: int, token_start: int, token_end: int = 0):
        self.req_id = req_id
        self.token_start = token_start
        self.token_end = token_end
        self.last_access_tick = 0

    @property
    def tokens(self):
        return self.token_end - self.token_start


@dataclass
class CacheLayer:
    node_id: int
    name: str
    # list req_id, tokens
    content: list[CacheItem]


@dataclass
class Cache:
    layers: dict[int, list[CacheLayer]]
    node_hardware: dict[int, Hardware]
    model: Model
    ram_usage_fraction: float
    ssd_usage_fraction: float
    ram_capacity_bytes: dict[int, int]
    ssd_capacity_bytes: dict[int, int]
    ram_usage_bytes: dict[int, int]
    ssd_usage_bytes: dict[int, int]
    _access_tick: int

    def __init__(
        self,
        layers: dict,
        node_hardware: dict[int, Hardware],
        model: Model,
        ram_usage_fraction: float = 0.8,
        ssd_usage_fraction: float = 0.8,
    ):
        self.layers = layers
        self.node_hardware = node_hardware
        self.model = model
        self.ram_usage_fraction = ram_usage_fraction
        self.ssd_usage_fraction = ssd_usage_fraction
        self.ram_capacity_bytes = {
            node_id: int(hardware.spec.ram_mem * ram_usage_fraction)
            for node_id, hardware in node_hardware.items()
        }
        self.ssd_capacity_bytes = {
            node_id: int(hardware.spec.nvme_mem * ssd_usage_fraction)
            for node_id, hardware in node_hardware.items()
        }
        self.ram_usage_bytes = dict.fromkeys(node_hardware, 0)
        self.ssd_usage_bytes = dict.fromkeys(node_hardware, 0)
        self._access_tick = 0

        self._validate_capacity()

    def _validate_capacity(self) -> None:
        """Raise if a node cannot store a minimal 512-token KV item in RAM/SSD."""
        min_item_bytes = self.kv_size(self.model, 512)
        for node_id, hardware in self.node_hardware.items():
            ram_cap = int(hardware.spec.ram_mem * self.ram_usage_fraction)
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

    def _touch(self, item: CacheItem) -> None:
        """Update LRU ordering for an accessed item."""
        self._access_tick += 1
        item.last_access_tick = self._access_tick

    def _item_size(self, item: CacheItem) -> int:
        return self.kv_size(self.model, item.tokens)

    def _ram_layer(self, node_id: int) -> CacheLayer:
        return self.get_layer(node_id, "RAM")

    def _ssd_layer(self, node_id: int) -> CacheLayer:
        return self.get_layer(node_id, "SSD")

    def _evict_ssd_lru(self, node_id: int) -> None:
        """Permanently delete the least-recently-used item from a node's SSD layer."""
        layer = self._ssd_layer(node_id)
        if not layer.content:
            return
        victim = min(layer.content, key=lambda item: item.last_access_tick)
        victim_size = self._item_size(victim)
        layer.content.remove(victim)
        self.ssd_usage_bytes[node_id] -= victim_size
        log(
            LOG_CACHE,
            f"Deleted SSD LRU KV for request {victim.req_id} "
            f"({victim.tokens} tokens, {victim_size} bytes) from node {node_id} SSD",
        )

    def _evict_ram_to_ssd(self, node_id: int) -> CacheItem:
        """Move the least-recently-used item from RAM to SSD.

        If the SSD layer is full, its LRU item is deleted first.
        Returns the moved item so the caller can build a transfer leg.
        """
        ram_layer = self._ram_layer(node_id)
        if not ram_layer.content:
            raise RuntimeError("No RAM item available to evict")

        victim = min(ram_layer.content, key=lambda item: item.last_access_tick)
        victim_size = self._item_size(victim)
        ram_layer.content.remove(victim)
        self.ram_usage_bytes[node_id] -= victim_size

        # Make room on SSD, deleting SSD LRU synchronously if needed.
        while (
            self.ssd_usage_bytes[node_id] + victim_size
            > self.ssd_capacity_bytes[node_id]
            and self._ssd_layer(node_id).content
        ):
            self._evict_ssd_lru(node_id)

        ssd_layer = self._ssd_layer(node_id)
        ssd_layer.content.append(victim)
        self.ssd_usage_bytes[node_id] += victim_size
        self._touch(victim)
        log(
            LOG_CACHE,
            f"Evicted RAM LRU KV for request {victim.req_id} "
            f"({victim.tokens} tokens, {victim_size} bytes) to node {node_id} SSD",
        )
        return victim

    def _make_room_ram(self, node_id: int, size: int) -> list[TransferLeg]:
        """Evict RAM items to SSD until ``size`` bytes fit.

        Returns the synchronous SSD write legs generated by the evictions.
        """
        eviction_legs: list[TransferLeg] = []
        while self._ram_layer(node_id).content:
            # Stop if the new item already fits in the remaining RAM budget.
            if self.ram_usage_bytes[node_id] + size <= self.ram_capacity_bytes[node_id]:
                break
            # Evict the RAM LRU. Capacity validation guarantees each eviction
            # frees at least a 512-token-sized slot, so the loop converges.
            victim = self._evict_ram_to_ssd(node_id)
            victim_size = self._item_size(victim)
            eviction_legs.append(
                TransferLeg(victim_size, node_id, node_id, "SSD_LOCAL")
            )
        return eviction_legs

    def _ensure_node_accounting(self, node_id: int) -> None:
        """Create usage/capacity entries for a node if they do not exist."""
        if node_id not in self.ram_capacity_bytes:
            hardware = self.node_hardware.get(node_id)
            if hardware is None:
                raise ValueError(f"Unknown node_id: {node_id}")
            self.ram_capacity_bytes[node_id] = int(
                hardware.spec.ram_mem * self.ram_usage_fraction
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
                CacheLayer(node_id=node_id, name=layer_name, content=[])
            )
            layer = self.layers[node_id][0]
        else:
            node_layers = self.layers[node_id]
            if not any(layer.name == layer_name for layer in node_layers):
                node_layers.append(
                    CacheLayer(node_id=node_id, name=layer_name, content=[])
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

    def find_cache(self, req_id: int) -> list[CacheItem]:
        items: list[CacheItem] = []
        for _, layers in self.layers.items():
            for layer in layers:
                for item in layer.content:
                    if item.req_id == req_id:
                        items.append(item)
        hit_tokens = 0
        for item in items:
            if item.token_start > hit_tokens:
                break
            hit_tokens = max(hit_tokens, item.token_end)

        return [item for item in items if item.token_start < hit_tokens]

    def find_cache_layer(self, item: CacheItem) -> CacheLayer | None:
        for _, layers in self.layers.items():
            for layer in layers:
                if item in layer.content:
                    return layer
        return None

    def delete_item(self, item: CacheItem):
        layer = self.find_cache_layer(item)
        if layer:
            if layer.name == "RAM":
                self.ram_usage_bytes[layer.node_id] -= self._item_size(item)
            elif layer.name == "SSD":
                self.ssd_usage_bytes[layer.node_id] -= self._item_size(item)
            layer.content.remove(item)
            return
        raise ValueError("Item not found in cache")

    def cost_move_tokens(
        self, item: CacheItem, dest_node_id: int, layer_name: str
    ) -> None:
        """Move an item logically to ``dest_node_id``/``layer_name``.

        This only updates cache bookkeeping. The physical transfer legs are
        generated separately by the caller that knows the original location.
        """
        found_at_layer = self.find_cache_layer(item)
        if not found_at_layer:
            raise ValueError("Item not found in cache")

        if found_at_layer.name == "RAM":
            self.ram_usage_bytes[found_at_layer.node_id] -= self._item_size(item)
        elif found_at_layer.name == "SSD":
            self.ssd_usage_bytes[found_at_layer.node_id] -= self._item_size(item)

        found_at_layer.content.remove(item)

        layer = self.get_layer(dest_node_id, layer_name)
        if layer.name == "RAM":
            # Note: destination RAM eviction legs are handled by the caller.
            self.ram_usage_bytes[dest_node_id] += self._item_size(item)
        elif layer.name == "SSD":
            self.ssd_usage_bytes[dest_node_id] += self._item_size(item)

        layer.content.append(item)
        self._touch(item)

    def cost_move_cache(
        self, req_id: int, node_id: int, layer_name: str
    ) -> list[TransferLeg]:
        """Move all cache items for ``req_id`` to the destination layer.

        Returns any destination-RAM eviction legs produced while making room.
        """
        items = sorted(self.find_cache(req_id), key=lambda x: -x.token_start)

        eviction_legs: list[TransferLeg] = []
        hit_tokens = 0
        for item in items:
            if item.token_start > hit_tokens:
                break
            if layer_name == "RAM" and node_id in self.ram_capacity_bytes:
                eviction_legs.extend(
                    self._make_room_ram(node_id, self._item_size(item))
                )
            self.cost_move_tokens(item, node_id, layer_name)

        return eviction_legs

    def insert_cache_item(self, item: CacheItem, node_id: int) -> list[TransferLeg]:
        """Insert an item into a node's RAM layer, evicting to SSD if needed.

        Returns the synchronous SSD write legs generated by evictions.
        """
        layer = self._ram_layer(node_id)
        item_size = self._item_size(item)
        eviction_legs = self._make_room_ram(node_id, item_size)

        layer.content.append(item)
        self.ram_usage_bytes[node_id] += item_size
        self._touch(item)

        log(
            LOG_CACHE,
            f"Inserted cache item for request {item.req_id} on node {node_id} "
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
        """Build physical transfer legs for moving KV from its cache location."""
        legs: list[TransferLeg] = []

        if source_layer_name == "SSD":
            # SSD -> source RAM
            legs.append(
                TransferLeg(
                    bytes_to_transfer, source_node_id, source_node_id, "SSD_LOCAL"
                )
            )

        if source_node_id != dest_node_id:
            # Source-side RAM staging for network egress.
            legs.append(
                TransferLeg(
                    bytes_to_transfer, source_node_id, source_node_id, "RAM_LOCAL"
                )
            )
            # Inter-node transfer.
            legs.append(
                TransferLeg(bytes_to_transfer, source_node_id, dest_node_id, "NETWORK")
            )
            # Destination-side RAM placement from network ingress.
            legs.append(
                TransferLeg(bytes_to_transfer, dest_node_id, dest_node_id, "RAM_LOCAL")
            )
        else:
            # Local transfer: just place into RAM.
            legs.append(
                TransferLeg(bytes_to_transfer, dest_node_id, dest_node_id, "RAM_LOCAL")
            )

        return legs

    def upload_kv(self, node_id: int, request: Request) -> UploadRequest:
        """Move KV from GPU RAM to RAM and insert into cache."""
        prior_cache = self.find_cache(request.id)
        if prior_cache:
            cache_layer = self.find_cache_layer(prior_cache[-1])

            assert cache_layer is not None, "Cache layer should not be None"
            assert cache_layer.node_id == node_id, (
                f"Cache layer node_id should match the provided node_id for request {request.id}"
            )
            assert cache_layer.name == "RAM", "Cache layer name should be 'RAM'"
            cache_item = CacheItem(
                request.id,
                prior_cache[-1].token_end,
                request.prefilled_tokens + request.decoded_tokens,
            )
            eviction_legs = self.insert_cache_item(cache_item, node_id)
        else:
            cache_item = CacheItem(
                request.id, 0, request.prefilled_tokens + request.decoded_tokens
            )
            eviction_legs = self.insert_cache_item(cache_item, node_id)

        bytes_to_transfer = self.kv_size(
            self.model, request.prefilled_tokens + request.decoded_tokens
        )
        log(
            LOG_CACHE,
            f"Uploading KV for request {request.id} to node {node_id}, "
            f"bytes: {bytes_to_transfer}, cache size: {request.prefilled_tokens + request.decoded_tokens} tokens",
        )

        upload_leg = TransferLeg(bytes_to_transfer, node_id, node_id, "RAM_LOCAL")
        return UploadRequest(request, [*eviction_legs, upload_leg])

    def download_kv(self, node_id: int, request: Request) -> DownloadRequest:
        """Move KV from current location to node RAM."""
        current_cache = self.find_cache(request.id)
        if not current_cache:
            log(LOG_CACHE, f"No cache found for request {request.id}")
            return DownloadRequest(request, [])

        source_layer = self.find_cache_layer(current_cache[-1])
        assert source_layer is not None, "Cache layer should not be None"
        source_node_id = source_layer.node_id
        source_layer_name = source_layer.name

        bytes_to_transfer = self.kv_size(self.model, request.isl)

        # Build physical transfer legs from the *original* cache location.
        data_legs = self._build_data_legs(
            source_layer_name, source_node_id, node_id, bytes_to_transfer
        )

        # Move cache items logically to destination RAM. This may produce
        # destination-RAM eviction legs that must complete first.
        dest_eviction_legs = self.cost_move_cache(request.id, node_id, "RAM")

        log(
            LOG_CACHE,
            f"Downloading KV for request {request.id} to node {node_id} "
            f"from node {source_node_id} {source_layer_name}, bytes: {bytes_to_transfer}, "
            f"legs: {[leg.bottleneck for leg in dest_eviction_legs + data_legs]}",
        )
        return DownloadRequest(request, dest_eviction_legs + data_legs)

    def kv_size(self, model: Model, tokens: int) -> int:
        return model.kv_size_per_token * tokens
