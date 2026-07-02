import random

from dataclasses import dataclass
from typing import ClassVar


request_id_counter = 0


class TransferLeg:
    """One sequential segment of a physical KV transfer.

    A transfer is made of independent *tracks*, each a list of legs.  Within a
    track legs run sequentially; across tracks they run in parallel.
    Bottleneck values:
      * ``RAM_LOCAL``  : shares the node's ``ram_bw``.
      * ``SSD_LOCAL``  : shares the node's ``nvme_bw``.
      * ``NETWORK``    : shares ``network_inet_up`` at source and
        ``network_inet_down`` at destination.
      * ``S3_UPLOAD``  : shared S3 upload link.
      * ``S3_DOWNLOAD``: shared S3 download link.

    Each leg carries a fixed startup ``latency_ms`` that must elapse before
    bytes begin to move.  Default latencies:
      * ``RAM_LOCAL``  : 0 ms
      * ``SSD_LOCAL``  : 0.1 ms
      * ``NETWORK``    : 0 ms
      * ``S3_UPLOAD``  : 50 ms
      * ``S3_DOWNLOAD``: 50 ms
    """

    remaining_bytes: int
    source_node_id: int
    dest_node_id: int
    bottleneck: str
    bandwidth_bytes_per_ms: float = 0.0
    remaining_latency_ms: float = 0.0

    _DEFAULT_LATENCY_MS: ClassVar[dict[str, float]] = {
        "RAM_LOCAL": 0.0,
        "SSD_LOCAL": 0.1,
        "NETWORK": 0.0,
        "S3_UPLOAD": 50.0,
        "S3_DOWNLOAD": 50.0,
    }

    def __init__(
        self,
        remaining_bytes: int,
        source_node_id: int,
        dest_node_id: int,
        bottleneck: str,
        latency_ms: float | None = None,
    ):
        self.remaining_bytes = remaining_bytes
        self.source_node_id = source_node_id
        self.dest_node_id = dest_node_id
        self.bottleneck = bottleneck
        self.remaining_latency_ms = (
            latency_ms
            if latency_ms is not None
            else self._DEFAULT_LATENCY_MS.get(bottleneck, 0.0)
        )


class _MultiTrackTransfer:
    """Mixin-like base for upload/download requests with parallel leg tracks."""

    request: Request
    tracks: list[list[TransferLeg]]
    current_legs: list[int]

    def __init__(self, request: Request, tracks: list[list[TransferLeg]]):
        self.request = request
        self.tracks = tracks
        self.current_legs = [0] * len(tracks)
        self._skip_zero_legs()

    def _skip_zero_legs(self) -> None:
        """Advance tracks past any zero-byte legs created by no-op sources."""
        for track_idx in range(len(self.tracks)):
            while (
                self.current_legs[track_idx] < len(self.tracks[track_idx])
                and self.tracks[track_idx][self.current_legs[track_idx]].remaining_bytes
                <= 0
            ):
                self.current_legs[track_idx] += 1

    @property
    def active_legs(self) -> list[TransferLeg]:
        """All currently active legs, one per non-exhausted track."""
        return [
            self.tracks[track_idx][self.current_legs[track_idx]]
            for track_idx in range(len(self.tracks))
            if self.current_legs[track_idx] < len(self.tracks[track_idx])
        ]

    @property
    def remaining_bytes(self) -> int:
        """Sum of remaining bytes across all active legs."""
        return sum(leg.remaining_bytes for leg in self.active_legs)

    def advance_track(self, track_idx: int) -> bool:
        """Advance a track after its current leg finished.

        Returns True if the track still has an active leg.
        """
        self.current_legs[track_idx] += 1
        while (
            self.current_legs[track_idx] < len(self.tracks[track_idx])
            and self.tracks[track_idx][self.current_legs[track_idx]].remaining_bytes
            <= 0
        ):
            self.current_legs[track_idx] += 1
        return self.current_legs[track_idx] < len(self.tracks[track_idx])

    def is_complete(self) -> bool:
        """True when every track has been exhausted."""
        return not self.active_legs


class UploadRequest(_MultiTrackTransfer):
    """Tracks for an upload.  Eviction and upload tracks run in parallel."""


class DownloadRequest(_MultiTrackTransfer):
    """Tracks for a download.  Eviction and per-source data tracks run in parallel."""


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
