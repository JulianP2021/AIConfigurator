"""Static global simulation clock.

The clock is owned by the ``BandwidthScheduler`` because every simulation step
advances time through ``BandwidthScheduler.advance_time``.  Other components
read the shared ``time_ms`` property when they need an authoritative wall-clock
timestamp.
"""


class GlobalClock:
    """Monotonically increasing simulation time in milliseconds."""

    _time_ms: float

    def __init__(self) -> None:
        self._time_ms = 0.0

    @property
    def time_ms(self) -> float:
        """Return the current simulation time in milliseconds."""
        return self._time_ms

    def advance(self, delta_ms: float) -> None:
        """Advance the clock by ``delta_ms`` milliseconds."""
        if delta_ms < 0:
            raise ValueError("Clock cannot advance by a negative amount")
        self._time_ms += delta_ms
