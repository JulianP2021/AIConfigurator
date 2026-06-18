from dataclasses import dataclass

from src.hardware.hardware import Hardware
from src.model.model import Model


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
    layers: list[CacheLayer]
    node_hardware: dict[int, Hardware]
    model: Model

    def find_cache(self, req_id: int) -> list[CacheItem]:
        items: list[CacheItem] = []
        for layer in self.layers:
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
        for layer in self.layers:
            if item in layer.content:
                return layer
        return None

    def delete_item(self, item: CacheItem):
        layer = self.find_cache_layer(item)
        if layer:
            layer.content.remove(item)
            return
        raise ValueError("Item not found in cache")

    def cost_move_tokens(self, item: CacheItem, _node_id: int, _layer: str) -> int:
        cost = 0
        find_layer = self.find_cache_layer(item)
        if not find_layer:
            raise ValueError("Item not found in cache")

        # if find_layer.name == "SSD":
        #     cost += self.kv_size(self.model, item.tokens) * self.node_hardware[find_layer.node_id]
        # # network transfer to new node
        # if find_layer.node_id != node_id:
        #     cost += self.kv_size(self.model, item.tokens) * self.node_hardware[node_id].networkGB_BW
        return cost

    def cost_move_cache(self, req_id: int, node_id: int, layer: str) -> int:
        items = sorted(self.find_cache(req_id), key=lambda x: -x.token_start)

        hit_tokens = 0
        cost = 0
        for item in items:
            if item.token_start > hit_tokens:
                break
            cost += self.cost_move_tokens(item, node_id, layer)

        return cost

    def kv_size(self, model: Model, tokens: int) -> int:
        return model.kv_size_per_token * tokens
