import random

from dataclasses import dataclass
from functools import cached_property
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
    processed_time_ms: float = 0.0  # scheduler time spent advancing this leg

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

    @property
    def active_transfer_duration_ms(self) -> float:
        """Total scheduler-processed time for the longest parallel track.

        Because tracks run in parallel, the transfer wall-clock time is the
        maximum sum of per-leg processed times across tracks.  This includes
        startup latency and byte-transfer time, but excludes any queueing time
        at the instance level.
        """
        if not self.tracks:
            return 0.0
        return max(sum(leg.processed_time_ms for leg in track) for track in self.tracks)


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

    # Phase-level timing (active / wait / total) for detailed analytics.
    # Active time is compute or scheduler-processed transfer time. Wait time is
    # queueing time before that phase becomes active. Total = active + wait.
    prefill_wait_ms: float = 0.0

    prefill_download_total_ms: float = 0.0
    prefill_download_active_ms: float = 0.0
    prefill_download_wait_ms: float = 0.0

    prefill_upload_total_ms: float = 0.0
    prefill_upload_active_ms: float = 0.0
    prefill_upload_wait_ms: float = 0.0

    decode_download_total_ms: float = 0.0
    decode_download_active_ms: float = 0.0
    decode_download_wait_ms: float = 0.0

    decode_total_ms: float = 0.0
    decode_wait_ms: float = 0.0

    decode_upload_total_ms: float = 0.0
    decode_upload_active_ms: float = 0.0
    decode_upload_wait_ms: float = 0.0

    # Derived user-facing metrics
    clean_ttft_ms: float = 0.0
    wait_inclusive_ttft_ms: float = 0.0
    clean_latency_ms: float = 0.0
    wait_inclusive_latency_ms: float = 0.0

    id: int
    user_id: int
    session_id: int

    def __init__(
        self, isl: int, osl: int, cached: int, user_id: int = -1, session_id: int = -1
    ):
        global request_id_counter
        self.isl = isl
        self.osl = osl
        self.prefix = cached
        self.prefilled_tokens = cached
        self.id = request_id_counter
        self.user_id = user_id
        self.session_id = session_id
        request_id_counter += 1

        assert self.isl >= 0, "ISL must be non-negative"
        assert self.osl >= 0, "OSL must be non-negative"
        assert self.prefilled_tokens <= self.isl, (
            "Prefilled tokens must be less than or equal to ISL"
        )

    @cached_property
    def cache_length(self):
        return self.prefilled_tokens + self.decoded_tokens

    @cached_property
    def remaining_tokens_prefill(self):
        return self.isl - self.prefilled_tokens

    @cached_property
    def remaining_tokens_decode(self):
        return self.osl - self.decoded_tokens

    @property
    def stage(self) -> str:
        return "decode" if self.remaining_tokens_prefill == 0 else "prefill"


@dataclass
class TokenDistribution:
    min_input_tokens: int
    max_input_tokens: int
    min_output_tokens: int
    max_output_tokens: int


@dataclass
class RequestScenario:
    token_distribution: TokenDistribution
    total_requests: int
    min_users: int
    max_users: int
    max_session_turns: int
    req_s: float


@dataclass
class RequestGenerator:
    req_rate: float
    max_session_turns: int = 5

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
            # Need more finished users; create a brand-new user not in flight.
            user_id = max(users, default=-1) + 1
        else:
            if len(users) < request_scenario.max_users:
                if random.random() < 0.5:
                    # New user must not collide with an in-flight user.
                    user_id = max(users, default=-1) + 1
                else:
                    user_id = (
                        random.choice(list(finished_users))
                        if finished_users
                        else max(users, default=-1) + 1
                    )
            else:
                if not finished_users:
                    raise ValueError(
                        "No finished users available to select from, but already at maximum user limit."
                    )
                user_id = random.choice(list(finished_users))
        return user_id

    def _get_session_id(
        self,
        user_id: int,
        current_requests: list[Request],
        finished_requests: list[Request],
    ) -> int:
        """Return a session id for the user, creating a new one if needed.

        A session is retired once it has reached ``max_session_turns`` requests
        (counted among finished requests). Requests that are still in flight are
        not counted.
        """
        all_user_requests = [
            r for r in current_requests + finished_requests if r.user_id == user_id
        ]
        if not all_user_requests:
            return 0

        # Group finished requests by session and retire the most recent session
        # if it has reached the turn limit.
        sessions: dict[int, int] = {}
        for r in all_user_requests:
            sessions[r.session_id] = sessions.get(r.session_id, 0) + 1

        latest_session = max(sessions.keys())
        if sessions[latest_session] >= self.max_session_turns:
            return latest_session + 1
        return latest_session

    def generate_request(
        self,
        request_scenario: RequestScenario,
        current_requests: list[Request],
        finished_requests: list[Request],
    ) -> Request:
        user_id = self._get_random_user_id(
            request_scenario, current_requests, finished_requests
        )
        session_id = self._get_session_id(user_id, current_requests, finished_requests)

        # Within a session, the input is the cumulative conversation length so far.
        past_session_requests = [
            r
            for r in current_requests + finished_requests
            if r.user_id == user_id and r.session_id == session_id
        ]
        min_input_tokens = max(
            0,
            max((r.isl + r.osl for r in past_session_requests), default=0),
        )

        input_tokens = min_input_tokens + random.randint(
            request_scenario.token_distribution.min_input_tokens,
            request_scenario.token_distribution.max_input_tokens,
        )

        output_tokens = random.randint(
            request_scenario.token_distribution.min_output_tokens,
            request_scenario.token_distribution.max_output_tokens,
        )

        # All prior session tokens are considered prefetched from cache.
        cached = min_input_tokens

        return Request(
            isl=input_tokens,
            osl=output_tokens,
            cached=cached,
            user_id=user_id,
            session_id=session_id,
        )
