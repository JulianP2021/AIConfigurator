id_counter = 0

class Request:
    isl: int
    osl: int
    prefilled_tokens: int
    decoded_tokens: int
    remaining_prefill_time_ms: int # for requests that were not fully filled till next request, but are currently processed

    prefill_time_ms: int
    decode_time_ms: int
    id: int

    def __init__(self, isl: int, osl: int, cached: int):
        global id_counter
        self.isl = isl
        self.osl = osl
        self.prefilled_tokens = cached
        self.decoded_tokens = 0
        self.remaining_prefill_time_ms = -1
        self.id = id_counter
        id_counter += 1

        self.prefill_time_ms = 0
        self.decode_time_ms = 0

        assert self.isl >= 0, "ISL must be non-negative"
        assert self.osl >= 0, "OSL must be non-negative"
        assert self.prefilled_tokens < self.isl, "Prefilled tokens must be less than ISL"

    @property
    def remaining_tokens_prefill(self):
        return self.isl - self.prefilled_tokens
    @property
    def remaining_tokens_decode(self):
        return self.osl - self.decoded_tokens

    @property
    def stage(self) -> str:
        return "decode" if self.remaining_prefill_time_ms == 0 else "prefill"

from dataclasses import dataclass
import random

@dataclass
class TokenDistribution:
    min_input_tokens: int
    min_output_tokens: int
    max_input_tokens: int
    max_output_tokens: int
    cache_percentage: float

@dataclass
class RequestGenerator:
    req_rate: float

    def time_till_next_request(self) -> float:
        return random.expovariate(self.req_rate)

    def generate_request(self, token_distribution: TokenDistribution) -> Request:
        input_tokens = random.randint(token_distribution.min_input_tokens, token_distribution.max_input_tokens)
        output_tokens = random.randint(token_distribution.min_output_tokens, token_distribution.max_output_tokens)
        cached_tokens = int(input_tokens * token_distribution.cache_percentage)
        return Request(isl=input_tokens, osl=output_tokens, cached=cached_tokens)