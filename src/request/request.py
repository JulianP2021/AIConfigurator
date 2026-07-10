import random

from dataclasses import dataclass, field
from typing import ClassVar


request_id_counter = 0


class TransferLeg:
    """One sequential segment of a physical KV transfer.

    A transfer is made of independent *tracks*, each a list of legs.  Within a
    track legs run sequentially; across tracks they run in parallel.
    Bottleneck values:
      * ``RAM_LOCAL``  : shares the node's ``ram_bw``.
      * ``SSD_LOCAL``  : shares the node's ``nvme_bw``.
      * ``NETWORK``    : shares ``network_inter_node_up`` at source and
        ``network_inter_node_down`` at destination.
      * ``S3_UPLOAD``  : uses the source node's ``network_inet_up`` link.
      * ``S3_DOWNLOAD``: uses the destination node's ``network_inet_down`` link.

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
    prefilled_tokens: int
    decoded_tokens: int = 0
    remaining_prefill_time_ms: float = -1

    # Event timestamps for each phase, all driven by the global simulation clock.
    prefill_queue_start_ms: float | None = None
    prefill_start_ms: float | None = None
    prefill_end_ms: float | None = None

    prefill_download_start_ms: float | None = None
    prefill_download_end_ms: float | None = None

    prefill_upload_start_ms: float | None = None
    prefill_upload_end_ms: float | None = None

    decode_queue_start_ms: float | None = None
    decode_start_ms: float | None = None
    decode_end_ms: float | None = None

    decode_download_start_ms: float | None = None
    decode_download_end_ms: float | None = None

    decode_upload_start_ms: float | None = None
    decode_upload_end_ms: float | None = None

    # Scheduler-reported active transfer durations for each phase.
    prefill_download_active_ms: float = 0.0
    prefill_upload_active_ms: float = 0.0
    decode_download_active_ms: float = 0.0
    decode_upload_active_ms: float = 0.0

    # Derived user-facing metrics (computed once at simulation end).
    prefill_time_ms: float = 0.0
    prefill_wait_ms: float = 0.0
    decode_time_ms: float = 0.0
    decode_wait_ms: float = 0.0
    kv_download_time_ms: float = 0.0
    kv_upload_time_ms: float = 0.0
    clean_ttft_ms: float = 0.0
    wait_inclusive_ttft_ms: float = 0.0
    clean_latency_ms: float = 0.0
    wait_inclusive_latency_ms: float = 0.0

    id: int
    user_id: int
    session_id: int

    def __init__(self, isl: int, osl: int, user_id: int = -1, session_id: int = -1):
        global request_id_counter
        self.isl = isl
        self.osl = osl
        self.prefilled_tokens = 0
        self.id = request_id_counter
        self.user_id = user_id
        self.session_id = session_id
        request_id_counter += 1

        assert self.isl >= 0, "ISL must be non-negative"
        assert self.osl >= 0, "OSL must be non-negative"
        assert self.prefilled_tokens <= self.isl, (
            "Prefilled tokens must be less than or equal to ISL"
        )

    @property
    def cache_length(self):
        return self.prefilled_tokens + self.decoded_tokens

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
    max_input_tokens: int
    min_output_tokens: int
    max_output_tokens: int


@dataclass
class RequestScenario:
    token_distribution: TokenDistribution
    sessions_per_user: int
    users: int
    max_session_turns: int
    think_time_ms: float

    @property
    def total_requests(self) -> int:
        return self.users * self.sessions_per_user * self.max_session_turns


@dataclass
class RequestGenerator:
    users: int
    max_session_turns: int
    think_time_ms: float
    sessions_per_user: int

    # One active session per user.  A user is "active" while it has any request
    # in flight; otherwise it is "idle".  Session ids are monotonic per user.
    _active_users: set[int] = field(default_factory=set)
    _idle_users: set[int] = field(default_factory=set)
    _user_session_id: dict[int, int] = field(default_factory=dict)
    _user_session_turns: dict[int, int] = field(default_factory=dict)
    _last_total_tokens: dict[tuple[int, int], int] = field(default_factory=dict)
    _next_available_ms: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize all users as idle with a random startup offset.

        Users are spread uniformly across ``[0, max_startup_offset_ms]`` so the
        first batch of requests does not all arrive at ``t=0``.
        """
        self._idle_users.update(range(self.users))
        # Space initial arrivals over several think-time windows so the startup
        # burst is not bunched within a single interval.  With ``think_time_ms``
        # this keeps the long-term average rate intact while giving a much
        # smoother initial arrival pattern.
        startup_windows = 5
        max_offset_ms = max(0.0, self.think_time_ms * startup_windows)
        for user_id in range(self.users):
            self._next_available_ms[user_id] = random.random() * max_offset_ms

    @property
    def total_requests(self) -> int:
        return self.users * self.sessions_per_user * self.max_session_turns

    def start_request(self, request: Request) -> None:
        """Record that ``request`` has been generated and is now in flight."""
        user_id = request.user_id
        session_id = request.session_id
        self._active_users.add(user_id)
        self._idle_users.discard(user_id)
        self._user_session_id[user_id] = session_id
        self._user_session_turns[user_id] = self._user_session_turns.get(user_id, 0) + 1

    def finish_request(self, request: Request, now_ms: float) -> None:
        """Record that ``request`` has completed and is no longer in flight.

        ``now_ms`` is the current simulation time; the user's next request will
        not be generated until ``now_ms + think_time_ms`` has elapsed.
        """
        user_id = request.user_id
        session_id = request.session_id
        total = request.isl + request.osl
        key = (user_id, session_id)
        self._last_total_tokens[key] = max(self._last_total_tokens.get(key, 0), total)
        self._active_users.discard(user_id)
        self._idle_users.add(user_id)
        self._next_available_ms[user_id] = now_ms + self.think_time_ms

    def ready_users(self, now_ms: float) -> list[int]:
        """Return idle users whose think time has elapsed, shuffled."""
        ready = [
            user_id
            for user_id in self._idle_users
            if now_ms >= self._next_available_ms.get(user_id, 0.0)
        ]
        random.shuffle(ready)
        return ready

    def next_ready_time_ms(self, _now_ms: float) -> float:
        """Return the earliest absolute time at which an idle user becomes ready."""
        min_time = float("inf")
        for user_id in self._idle_users:
            available = self._next_available_ms.get(user_id, 0.0)
            if available < min_time:
                min_time = available
        return min_time

    def generate_request(
        self,
        request_scenario: RequestScenario,
        now_ms: float,
    ) -> Request | None:
        """Generate the next request for a ready idle user, if any.

        Returns ``None`` when every user is still active or within its think time.
        """
        ready_users = self.ready_users(now_ms)
        if not ready_users:
            return None

        user_id = ready_users[0]
        session_id = self._user_session_id.get(user_id, 0)

        # If the user's current session is full, roll over to a new session.
        if self._user_session_turns.get(user_id, 0) >= self.max_session_turns:
            session_id += 1
            # Reset cached state for the new session so its first request has
            # no prior context (turn 1, no accumulated tokens).
            self._user_session_id[user_id] = session_id
            self._user_session_turns[user_id] = 0
            key = (user_id, session_id)
            self._last_total_tokens.pop(key, None)

        key = (user_id, session_id)
        min_input_tokens = self._last_total_tokens.get(key, 0)

        input_tokens = min_input_tokens + random.randint(
            request_scenario.token_distribution.min_input_tokens,
            request_scenario.token_distribution.max_input_tokens,
        )

        output_tokens = random.randint(
            request_scenario.token_distribution.min_output_tokens,
            request_scenario.token_distribution.max_output_tokens,
        )

        request = Request(
            isl=input_tokens,
            osl=output_tokens,
            user_id=user_id,
            session_id=session_id,
        )
        self.start_request(request)
        return request
