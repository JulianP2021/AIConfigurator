import random

from dataclasses import dataclass


request_id_counter = 0


class TransferLeg:
    """One sequential segment of a physical KV transfer.

    A transfer is a list of legs. Only the active leg is scheduled at any
    moment. Bottleneck values:
      * ``RAM_LOCAL``  : shares the node's ``ram_bw``.
      * ``SSD_LOCAL``  : shares the node's ``nvme_bw``.
      * ``NETWORK``    : shares ``network_inet_up`` at source and
        ``network_inet_down`` at destination.
    """

    remaining_bytes: int
    source_node_id: int
    dest_node_id: int
    bottleneck: str
    bandwidth_bytes_per_ms: float = 0.0

    def __init__(
        self,
        remaining_bytes: int,
        source_node_id: int,
        dest_node_id: int,
        bottleneck: str,
    ):
        self.remaining_bytes = remaining_bytes
        self.source_node_id = source_node_id
        self.dest_node_id = dest_node_id
        self.bottleneck = bottleneck


class UploadRequest:
    request: Request
    legs: list[TransferLeg]
    current_leg: int

    def __init__(self, request: Request, legs: list[TransferLeg]):
        self.request = request
        self.legs = legs
        self.current_leg = 0

    @property
    def active_leg(self) -> TransferLeg | None:
        if self.current_leg < len(self.legs):
            return self.legs[self.current_leg]
        return None

    @property
    def remaining_bytes(self) -> int:
        return self.active_leg.remaining_bytes if self.active_leg else 0

    @property
    def source_node_id(self) -> int:
        return self.active_leg.source_node_id if self.active_leg else 0

    @property
    def dest_node_id(self) -> int:
        return self.active_leg.dest_node_id if self.active_leg else 0

    @property
    def bandwidth_bytes_per_ms(self) -> float:
        return self.active_leg.bandwidth_bytes_per_ms if self.active_leg else 0.0

    @bandwidth_bytes_per_ms.setter
    def bandwidth_bytes_per_ms(self, value: float) -> None:
        if self.active_leg:
            self.active_leg.bandwidth_bytes_per_ms = value

    @property
    def bottleneck(self) -> str:
        return self.active_leg.bottleneck if self.active_leg else "RAM_LOCAL"

    def advance_leg(self) -> bool:
        """Move to the next leg. Returns True if there is another leg."""
        self.current_leg += 1
        return self.current_leg < len(self.legs)


class DownloadRequest:
    request: Request
    legs: list[TransferLeg]
    current_leg: int

    def __init__(self, request: Request, legs: list[TransferLeg]):
        self.request = request
        self.legs = legs
        self.current_leg = 0

    @property
    def active_leg(self) -> TransferLeg | None:
        if self.current_leg < len(self.legs):
            return self.legs[self.current_leg]
        return None

    @property
    def remaining_bytes(self) -> int:
        return self.active_leg.remaining_bytes if self.active_leg else 0

    @property
    def source_node_id(self) -> int:
        return self.active_leg.source_node_id if self.active_leg else 0

    @property
    def dest_node_id(self) -> int:
        return self.active_leg.dest_node_id if self.active_leg else 0

    @property
    def bandwidth_bytes_per_ms(self) -> float:
        return self.active_leg.bandwidth_bytes_per_ms if self.active_leg else 0.0

    @bandwidth_bytes_per_ms.setter
    def bandwidth_bytes_per_ms(self, value: float) -> None:
        if self.active_leg:
            self.active_leg.bandwidth_bytes_per_ms = value

    @property
    def bottleneck(self) -> str:
        return self.active_leg.bottleneck if self.active_leg else "RAM_LOCAL"

    def advance_leg(self) -> bool:
        """Move to the next leg. Returns True if there is another leg."""
        self.current_leg += 1
        return self.current_leg < len(self.legs)


class Request:
    isl: int
    osl: int
    prefix: int
    prefilled_tokens: int
    decoded_tokens: int = 0
    remaining_prefill_time_ms: float = -1

    prefill_time_ms: float = 0
    decode_time_ms: float = 0

    kv_download_time_ms: float = 0
    kv_upload_time_ms: float = 0
    id: int
    user_id: int

    def __init__(self, isl: int, osl: int, cached: int, user_id: int = -1):
        global request_id_counter
        self.isl = isl
        self.osl = osl
        self.prefix = cached
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
        return "decode" if self.remaining_tokens_prefill == 0 else "prefill"


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
