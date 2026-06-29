from dataclasses import dataclass

from src.hardware.hardware import Hardware
from src.logger import debug_print
from src.model.model import Model
from src.request.request import Request


@dataclass
class CacheItem:
    req_id: int
    token_start: int
    token_end: int = 0

    def __init__(self, req_id: int, token_start: int, token_end: int = 0):
        self.req_id = req_id
        self.token_start = token_start
        self.token_end = token_end

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

    def get_layer(self, node_id: int, layer_name: str) -> CacheLayer:
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
            layer.content.remove(item)
            return
        raise ValueError("Item not found in cache")

    def possible_eviction_time(self, _node_id: int, _layer_name: str) -> float:
        # @TODO: Implement eviction policy if needed
        return 0.0

    def cost_move_tokens(
        self, item: CacheItem, _node_id: int, layer_name: str
    ) -> float:
        cost = 1
        found_at_layer = self.find_cache_layer(item)
        if not found_at_layer:
            raise ValueError("Item not found in cache")

        found_at_layer.content.remove(item)

        layer = self.get_layer(_node_id, layer_name)
        layer.content.append(item)
        self.possible_eviction_time(_node_id, layer_name)

        # if find_layer.name == "SSD":
        #     cost += self.kv_size(self.model, item.tokens) * self.node_hardware[find_layer.node_id]
        # # network transfer to new node
        # if find_layer.node_id != node_id:
        #     cost += self.kv_size(self.model, item.tokens) * self.node_hardware[node_id].networkGB_BW
        return cost

    def cost_move_cache(self, req_id: int, node_id: int, layer_name: str) -> float:
        items = sorted(self.find_cache(req_id), key=lambda x: -x.token_start)

        hit_tokens = 0
        cost = 0
        for item in items:
            if item.token_start > hit_tokens:
                break
            cost += self.cost_move_tokens(item, node_id, layer_name)

        return cost

    def insert_cache_item(self, item: CacheItem, node_id: int) -> float:
        layer = self.get_layer(node_id, "RAM")

        layer.content.append(item)
        size = self.kv_size(self.model, item.tokens)
        time_ms = float((float(size) / self.node_hardware[node_id].spec.ram_bw) * 1000)

        time_ms += self.possible_eviction_time(node_id, "RAM")
        return time_ms

    def upload_kv(self, node_id: int, request: Request) -> float:
        """Move KV from GPU RAM to RAM and insert into cache."""
        kv_size = self.model.kv_size_per_token * request.isl
        time_ms = 0

        print(self.layers)

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
            time_ms += self.insert_cache_item(cache_item, node_id)
        else:
            cache_item = CacheItem(
                request.id, 0, request.prefilled_tokens + request.decoded_tokens
            )
            time_ms += self.insert_cache_item(cache_item, node_id)
        debug_print(
            f"Uploading KV for request {request.id} to node {node_id}, size: {kv_size} bytes, total time: {time_ms} ms"
        )

        return max(time_ms, 1)

    def download_kv(self, node_id: int, request: Request) -> float:
        """Move KV from current location to node RAM, then scatter to GPU RAM."""
        current_cache = self.find_cache(request.id)
        if not current_cache:
            debug_print(f"No cache found for request {request.id}")
            return 1

        time_ms = self.cost_move_cache(request.id, node_id, "RAM")

        time_ms += float(
            (
                float(self.model.kv_size_per_token * request.isl)
                / self.node_hardware[node_id].spec.ram_bw
            )
            * 1000
        )
        return max(time_ms, 1)

    def kv_size(self, model: Model, tokens: int) -> int:
        return model.kv_size_per_token * tokens
