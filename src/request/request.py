import math
import random

from dataclasses import dataclass
from typing import ClassVar


_rng: random.Random = random.Random()


def set_request_rng(seed: int | None) -> None:
    """Seed the global request generator RNG for reproducible user delays.

    Calling this before constructing a ``RequestGenerator`` makes startup
    offsets and per-user delays deterministic across runs.
    """
    global _rng
    _rng = random.Random(seed)


request_id_counter: int = 0


def reset_request_state(seed: int | None = None) -> None:
    """Reset module-level mutable state for a fresh simulation run.

    Resets both the request ID counter and the global RNG so that repeated
    simulations in the same process produce deterministic results.
    """
    global request_id_counter
    request_id_counter = 0
    set_request_rng(seed)


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

    __slots__ = (
        "bandwidth_bytes_per_ms",
        "bottleneck",
        "dest_node_id",
        "processed_time_ms",
        "remaining_bytes",
        "remaining_latency_ms",
        "source_node_id",
    )

    remaining_bytes: int
    source_node_id: int
    dest_node_id: int
    bottleneck: str
    bandwidth_bytes_per_ms: float
    remaining_latency_ms: float
    processed_time_ms: float  # scheduler time spent advancing this leg

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
        self.bandwidth_bytes_per_ms = 0.0
        self.remaining_latency_ms = (
            latency_ms
            if latency_ms is not None
            else self._DEFAULT_LATENCY_MS.get(bottleneck, 0.0)
        )
        self.processed_time_ms = 0.0


class _MultiTrackTransfer:
    """Mixin-like base for upload/download requests with parallel leg tracks."""

    __slots__ = ("_cached_active_legs", "current_legs", "request", "tracks")

    request: Request
    tracks: list[list[TransferLeg]]
    current_legs: list[int]

    def __init__(self, request: Request, tracks: list[list[TransferLeg]]):
        self.request = request
        self.tracks = tracks
        self.current_legs = [0] * len(tracks)
        self._cached_active_legs: list[TransferLeg] | None = None
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
        self._invalidate_active_legs_cache()

    @property
    def active_legs(self) -> list[TransferLeg]:
        """All currently active legs, one per non-exhausted track."""
        if self._cached_active_legs is None:
            self._cached_active_legs = [
                self.tracks[track_idx][self.current_legs[track_idx]]
                for track_idx in range(len(self.tracks))
                if self.current_legs[track_idx] < len(self.tracks[track_idx])
            ]
        return self._cached_active_legs

    def _invalidate_active_legs_cache(self) -> None:
        self._cached_active_legs = None

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
        self._invalidate_active_legs_cache()
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
    """Tracks for an upload.  Eviction and upload tracks run in parallel.

    The last track is the actual upload leg; the request is considered
    finished once that track completes, while any eviction tracks keep running
    in the background.
    """

    __slots__ = ()

    def is_upload_done(self) -> bool:
        """Return True when the actual upload leg (last track) is exhausted."""
        if not self.tracks:
            return True
        last_idx = len(self.tracks) - 1
        return self.current_legs[last_idx] >= len(self.tracks[last_idx])

    def upload_active_duration_ms(self) -> float:
        """Active duration of the actual upload leg (last track) only."""
        if not self.tracks:
            return 0.0
        return sum(leg.processed_time_ms for leg in self.tracks[-1])

    def background_active_duration_ms(self) -> float:
        """Active duration of all tracks except the actual upload leg.

        This measures the background eviction work that continues after the
        request itself is considered uploaded.
        """
        if not self.tracks:
            return 0.0
        return (
            max(
                sum(leg.processed_time_ms for leg in track)
                for track in self.tracks[:-1]
            )
            if len(self.tracks) > 1
            else 0.0
        )


class DownloadRequest(_MultiTrackTransfer):
    """Tracks for a download.  Eviction and per-source data tracks run in parallel.

    The first ``eviction_track_count`` tracks are background work (e.g. moving
    evicted data to SSD/S3) that is triggered by making room for the download.
    They must not block the request from entering the decode queue: the request
    is considered downloaded once all non-eviction tracks finish.
    """

    __slots__ = ("eviction_track_count",)

    def __init__(
        self,
        request: Request,
        tracks: list[list[TransferLeg]],
        eviction_track_count: int = 0,
    ):
        super().__init__(request, tracks)
        self.eviction_track_count = eviction_track_count

    def is_download_done(self) -> bool:
        """True when all data tracks (non-eviction) are exhausted."""
        first_data_idx = self.eviction_track_count
        for track_idx in range(first_data_idx, len(self.tracks)):
            if self.current_legs[track_idx] < len(self.tracks[track_idx]):
                return False
        return True

    def download_active_duration_ms(self) -> float:
        """Wall-clock duration of the data tracks only.

        Because data tracks run in parallel, this is the maximum sum of
        processed times across the non-eviction tracks.
        """
        first_data_idx = self.eviction_track_count
        if first_data_idx >= len(self.tracks):
            return 0.0
        return max(
            sum(leg.processed_time_ms for leg in track)
            for track in self.tracks[first_data_idx:]
        )

    def download_background_active_duration_ms(self) -> float:
        """Wall-clock duration of the background eviction tracks only."""
        if self.eviction_track_count == 0:
            return 0.0
        return max(
            sum(leg.processed_time_ms for leg in track)
            for track in self.tracks[: self.eviction_track_count]
        )


class Request:
    # Core request state and event timestamps.  All fields are initialized to
    # None / 0 / -1 and are populated by instances as the request progresses.
    __slots__ = (
        "clean_latency_ms",
        "clean_ttft_ms",
        "decode_download_active_ms",
        "decode_download_background_active_ms",
        "decode_download_end_ms",
        "decode_download_start_ms",
        "decode_download_wait_ms",
        "decode_end_ms",
        "decode_queue_start_ms",
        "decode_start_ms",
        "decode_time_ms",
        "decode_upload_active_ms",
        "decode_upload_background_active_ms",
        "decode_upload_end_ms",
        "decode_upload_start_ms",
        "decode_upload_wait_ms",
        "decode_wait_ms",
        "decoded_tokens",
        "generated_ms",
        "id",
        "initial_prefilled_tokens",
        "isl",
        "kv_download_time_ms",
        "kv_upload_time_ms",
        "osl",
        "prefill_download_active_ms",
        "prefill_download_background_active_ms",
        "prefill_download_end_ms",
        "prefill_download_start_ms",
        "prefill_download_wait_ms",
        "prefill_end_ms",
        "prefill_queue_start_ms",
        "prefill_start_ms",
        "prefill_time_ms",
        "prefill_upload_active_ms",
        "prefill_upload_background_active_ms",
        "prefill_upload_end_ms",
        "prefill_upload_start_ms",
        "prefill_upload_wait_ms",
        "prefill_wait_ms",
        "prefilled_tokens",
        "remaining_prefill_time_ms",
        "session_id",
        "user_id",
        "wait_inclusive_latency_ms",
        "wait_inclusive_ttft_ms",
    )

    def __init__(self, isl: int, osl: int, user_id: int = -1, session_id: int = -1):
        global request_id_counter

        self.id = request_id_counter
        self.user_id = user_id
        self.session_id = session_id
        self.isl = isl
        self.osl = osl
        request_id_counter += 1

        assert self.isl >= 0, "ISL must be non-negative"
        assert self.osl >= 0, "OSL must be non-negative"

        # Scalar state.
        self.prefilled_tokens = 0
        self.decoded_tokens = 0
        self.remaining_prefill_time_ms = -1.0
        self.initial_prefilled_tokens = None

        # Timestamps and durations.  Grouped into None-initialized floats and
        # zero-initialized floats for compact assignment.
        for name in (
            "generated_ms",
            "prefill_queue_start_ms",
            "prefill_start_ms",
            "prefill_end_ms",
            "prefill_download_start_ms",
            "prefill_download_end_ms",
            "prefill_upload_start_ms",
            "prefill_upload_end_ms",
            "decode_queue_start_ms",
            "decode_start_ms",
            "decode_end_ms",
            "decode_download_start_ms",
            "decode_download_end_ms",
            "decode_upload_start_ms",
            "decode_upload_end_ms",
        ):
            setattr(self, name, None)

        for name in (
            "prefill_download_active_ms",
            "prefill_download_background_active_ms",
            "prefill_upload_active_ms",
            "prefill_upload_background_active_ms",
            "decode_download_active_ms",
            "decode_download_background_active_ms",
            "decode_upload_active_ms",
            "decode_upload_background_active_ms",
            "prefill_time_ms",
            "prefill_wait_ms",
            "prefill_download_wait_ms",
            "prefill_upload_wait_ms",
            "decode_time_ms",
            "decode_wait_ms",
            "decode_download_wait_ms",
            "decode_upload_wait_ms",
            "kv_download_time_ms",
            "kv_upload_time_ms",
            "clean_ttft_ms",
            "wait_inclusive_ttft_ms",
            "clean_latency_ms",
            "wait_inclusive_latency_ms",
        ):
            setattr(self, name, 0.0)

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


class RequestGenerator:
    users: int
    max_session_turns: int
    think_time_ms: float
    sessions_per_user: int
    delay_fraction: float
    delay_min_ms: float
    delay_max_ms: float
    ttft_sla_ms: float
    tpot_sla_ms: float

    def __init__(
        self,
        users: int,
        max_session_turns: int,
        think_time_ms: float,
        sessions_per_user: int,
        delay_fraction: float,
        delay_min_ms: float,
        delay_max_ms: float,
        ttft_sla_ms: float,
        tpot_sla_ms: float,
        startup_arrival_mean_ms: float = 0.0,
    ) -> None:
        if not math.isfinite(ttft_sla_ms) or ttft_sla_ms <= 0:
            raise ValueError(
                f"ttft_sla_ms must be a finite positive number, got {ttft_sla_ms}"
            )
        if not math.isfinite(tpot_sla_ms) or tpot_sla_ms <= 0:
            raise ValueError(
                f"tpot_sla_ms must be a finite positive number, got {tpot_sla_ms}"
            )

        self.users = users
        self.max_session_turns = max_session_turns
        self.think_time_ms = think_time_ms
        self.sessions_per_user = sessions_per_user
        self.delay_fraction = delay_fraction
        self.delay_min_ms = delay_min_ms
        self.delay_max_ms = delay_max_ms
        self.startup_arrival_mean_ms = startup_arrival_mean_ms
        self.ttft_sla_ms = ttft_sla_ms
        self.tpot_sla_ms = tpot_sla_ms

        # One active session per user.  A user is "active" while it has any request
        # in flight; otherwise it is "idle".  Session ids are monotonic per user.
        self._active_users: set[int] = set()
        self._idle_users: set[int] = set()
        self._user_session_id: dict[int, int] = {}
        self._user_session_turns: dict[int, int] = {}
        self._last_total_tokens: dict[tuple[int, int], int] = {}
        self._last_request_generated_ms: dict[tuple[int, int], float] = {}
        self._next_available_ms: dict[int, float] = {}

        self._init_startup_offsets()

    def _init_startup_offsets(self) -> None:
        """Initialize all users as idle with a random startup offset.

        Offsets are drawn from an exponential distribution with mean
        ``startup_arrival_mean_ms`` (or 0 if not set), seeded by the global RNG
        so the same seed always produces the same schedule.
        """
        self._idle_users.update(range(self.users))
        if self.startup_arrival_mean_ms > 0.0:
            for user_id in range(self.users):
                self._next_available_ms[user_id] = _rng.expovariate(
                    1.0 / self.startup_arrival_mean_ms
                )
        else:
            self._next_available_ms.update((uid, 0.0) for uid in range(self.users))

    @property
    def total_requests(self) -> int:
        return self.users * self.sessions_per_user * self.max_session_turns

    def start_request(self, request: Request) -> None:
        """Record that ``request`` has been generated and is now in flight."""
        user_id = request.user_id
        session_id = request.session_id
        key = (user_id, session_id)
        self._active_users.add(user_id)
        self._idle_users.discard(user_id)
        self._user_session_id[user_id] = session_id
        self._user_session_turns[user_id] = self._user_session_turns.get(user_id, 0) + 1
        # Track the most recent request generation time per (user, session) so
        # that SLA violation messages can report inter-request delay.
        self._last_request_generated_ms[key] = request.generated_ms

    def finish_request(self, request: Request, now_ms: float) -> None:
        """Record that ``request`` has completed and is no longer in flight.

        the user's next request is generated after
        ``now_ms + think_time_ms`` has elapsed.  With probability ``delay_fraction`` an
        extra uniform delay in [``delay_min_ms``, ``delay_max_ms``] is added on
        top of that base interval.

        If the user has completed all allowed sessions, it is removed from the
        idle pool permanently so the simulator does not wait for it.
        """
        user_id = request.user_id
        session_id = request.session_id
        total = request.isl + request.osl
        key = (user_id, session_id)
        self._last_total_tokens[key] = max(self._last_total_tokens.get(key, 0), total)
        # Keep the generation timestamp around so the SLA message can compute
        # the gap from the previous request in the same session.
        self._active_users.discard(user_id)

        # Determine whether this was the user's final request.  A session is
        # the last one if it equals ``sessions_per_user`` and it has just
        # reached ``max_session_turns``.
        turns_after_this = self._user_session_turns.get(user_id, 0)
        is_final_session = session_id >= self.sessions_per_user
        is_final_turn = turns_after_this >= self.max_session_turns
        user_exhausted = is_final_session and is_final_turn

        if user_exhausted:
            self._idle_users.discard(user_id)
            self._next_available_ms.pop(user_id, None)
            return

        self._idle_users.add(user_id)
        next_ready_ms = now_ms + self.think_time_ms
        if (
            self.delay_fraction > 0.0
            and _rng.random() < self.delay_fraction
            and self.delay_max_ms > 0.0
        ):
            extra_ms = _rng.uniform(self.delay_min_ms, self.delay_max_ms)
            next_ready_ms += extra_ms

        self._next_available_ms[user_id] = next_ready_ms

    def ready_users(self, now_ms: float) -> list[int]:
        """Return idle users whose think time has elapsed and can still generate requests."""
        return self._ready_users(now_ms)

    def get_last_request_generated_ms(
        self, user_id: int, session_id: int
    ) -> float | None:
        """Return the generation time of the previous request in this session."""
        return self._last_request_generated_ms.get((user_id, session_id))

    def next_ready_time_ms(self, _now_ms: float) -> float:
        """Return the earliest absolute time at which an idle user becomes ready."""
        min_time = float("inf")
        for user_id in self._idle_users:
            available = self._next_available_ms.get(user_id, 0.0)
            if available < min_time:
                min_time = available
        return min_time

    def _current_session_id(self, user_id: int) -> int:
        """Return the session id currently in progress for ``user_id``.

        ``0`` means the user has not started any session yet; the first session
        will be numbered 1.
        """
        return self._user_session_id.get(user_id, 0)

    def _session_count(self, user_id: int) -> int:
        """Return the number of sessions already started by ``user_id``."""
        return self._current_session_id(user_id)

    def _can_generate_requests(self, user_id: int) -> bool:
        """Return True if ``user_id`` has not exhausted its request budget.

        A user is done when it has started ``sessions_per_user`` sessions *and*
        completed ``max_session_turns`` turns in the final session.
        """
        session_id = self._current_session_id(user_id)
        turns = self._user_session_turns.get(user_id, 0)
        if session_id < self.sessions_per_user:
            return True
        return bool(
            session_id == self.sessions_per_user and turns < self.max_session_turns
        )

    def _ready_users(self, now_ms: float) -> list[int]:
        """Return idle users past their think time who can still generate requests."""
        ready = [
            user_id
            for user_id in self._idle_users
            if now_ms >= self._next_available_ms.get(user_id, 0.0)
            and self._can_generate_requests(user_id)
        ]
        _rng.shuffle(ready)
        return ready

    def generate_request(
        self,
        request_scenario: RequestScenario,
        now_ms: float,
    ) -> Request | None:
        """Generate the next request for a ready idle user, if any.

        Sessions are numbered starting at 1.  Each user can start at most
        ``sessions_per_user`` sessions; once that cap is reached the user is
        removed from the ready pool and will not generate further requests.

        Returns ``None`` when every user is still active, within its think time,
        or has exhausted its session budget.
        """
        ready_users = self._ready_users(now_ms)
        if not ready_users:
            return None

        user_id = ready_users[0]
        session_id = self._current_session_id(user_id)

        # If the user's current session is full, roll over to a new session.
        if self._user_session_turns.get(user_id, 0) >= self.max_session_turns:
            if session_id + 1 > self.sessions_per_user:
                # This user has exhausted all allowed sessions.  Remove them
                # from the idle pool permanently.
                self._idle_users.discard(user_id)
                self._next_available_ms.pop(user_id, None)
                return self.generate_request(request_scenario, now_ms)
            session_id += 1
            # Reset cached state for the new session so its first request has
            # no prior context (turn 1, no accumulated tokens).
            self._user_session_id[user_id] = session_id
            self._user_session_turns[user_id] = 0
            key = (user_id, session_id)
            self._last_total_tokens.pop(key, None)

        # First request for a brand-new user starts session 1.
        if session_id == 0:
            session_id = 1
            self._user_session_id[user_id] = 1

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
        request.generated_ms = now_ms
        self.start_request(request)
        return request
