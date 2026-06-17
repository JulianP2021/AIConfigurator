from model.model import Model


class CacheLayer:
    name: str
    # list req_id, tokens
    content: list[dict[int, int]]

class Cache:
    layers: list[CacheLayer]


    def costs_load_cache_ms(self, model:Model, tokens: int) -> float:
        # TOKENS_KV_CACHE_SIZE = model.KV_SIZE_PER_TOKEN * tokens
        # TOTAL_KV_CACHE_SIZE = TOKENS_KV_CACHE_SIZE * 100 # cache for 100 seconds
        # TOTAL_CPU_PERCENTAGE  = min(CPU_CACHE_SIZE_BYTES / TOTAL_KV_CACHE_SIZE, 1)
        # AVERAGE_BW = TOTAL_CPU_PERCENTAGE * CPU_BW + (1-TOTAL_CPU_PERCENTAGE) * DISK_BW

        # print("COST CACHE: ", TOKENS_KV_CACHE_SIZE / AVERAGE_BW, TOKENS_KV_CACHE_SIZE, TOTAL_KV_CACHE_SIZE, TOTAL_CPU_PERCENTAGE, AVERAGE_BW)
        # return TOKENS_KV_CACHE_SIZE / AVERAGE_BW
        return 1