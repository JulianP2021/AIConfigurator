import random

from dataclasses import dataclass


request_id_counter = 0


class Request:
    isl: int
    osl: int
    prefilled_tokens: int
    decoded_tokens: int = 0
    remaining_prefill_time_ms: float = -1  # for requests that were not fully filled till next request, but are currently processed

    prefill_time_ms: float = 0
    decode_time_ms: float = 0

    kv_download_time_ms: float = 0
    kv_downloaded: bool = False
    kv_upload_time_ms: float = 0
    kv_uploaded: bool = False
    id: int
    user_id: int

    def __init__(self, isl: int, osl: int, cached: int, user_id: int = -1):
        global request_id_counter
        self.isl = isl
        self.osl = osl
        self.prefilled_tokens = cached
        self.id = request_id_counter
        self.user_id = user_id
        request_id_counter += 1

        assert self.isl >= 0, "ISL must be non-negative"
        assert self.osl >= 0, "OSL must be non-negative"
        assert self.prefilled_tokens < self.isl, (
            "Prefilled tokens must be less than ISL"
        )

    @property
    def remaining_tokens_prefill(self):
        return self.isl - self.prefilled_tokens

    @property
    def remaining_tokens_decode(self):
        return self.osl - self.decoded_tokens

    @property
    def stage(self) -> str:
        return "decode" if self.remaining_prefill_time_ms == 0 else "prefill"


@dataclass
class TokenDistribution:
    min_input_tokens: int
    min_output_tokens: int
    max_input_tokens: int
    max_output_tokens: int
    cache_percentage: float


@dataclass
class RequestScenario:
    token_distribution: TokenDistribution
    total_requests: int
    min_users: int
    max_users: int
    req_s: float


@dataclass
class RequestGenerator:
    req_rate: float

    def time_till_next_request(self) -> float:
        return random.expovariate(self.req_rate)

    def _get_random_user_id(
        self,
        request_scenario: RequestScenario,
        current_requests: list[Request],
        finished_requests: list[Request],
    ) -> int:
        current_users = {r.user_id for r in current_requests}
        finished_users = {r.user_id for r in finished_requests}.difference(
            current_users
        )
        users = current_users.union(finished_users)

        if len(finished_users) < request_scenario.min_users:
            user_id = max(users) + 1 if users else 0
        else:
            if len(finished_users) < request_scenario.max_users:
                if random.random() < 0.5:
                    user_id = max(finished_users) + 1
                else:
                    user_id = random.choice(list(finished_users))
            else:
                user_id = random.choice(list(finished_users))
        return user_id

    def generate_request(
        self,
        request_scenario: RequestScenario,
        current_requests: list[Request],
        finished_requests: list[Request],
    ) -> Request:
        user_id = self._get_random_user_id(
            request_scenario, current_requests, finished_requests
        )
        past_requests = [r for r in current_requests if r.user_id == user_id]
        min_input_tokens = max(
            0,
            max((r.isl + r.osl for r in past_requests), default=0),
        )

        input_tokens = min_input_tokens + random.randint(
            request_scenario.token_distribution.min_input_tokens,
            request_scenario.token_distribution.max_input_tokens,
        )

        output_tokens = random.randint(
            request_scenario.token_distribution.min_output_tokens,
            request_scenario.token_distribution.max_output_tokens,
        )

        cached = min(
            min_input_tokens,
            int(input_tokens * request_scenario.token_distribution.cache_percentage),
        )

        return Request(
            isl=input_tokens, osl=output_tokens, cached=cached, user_id=user_id
        )
