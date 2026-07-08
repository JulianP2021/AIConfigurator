class PrefillError(Exception):
    """Exception raised for too large prefill queue."""


class PrefillLatencyError(Exception):
    """Exception raised for too large prefill latency."""


class DecodeError(Exception):
    """Exception raised for too large decode queue."""


class DecodeLatencyError(Exception):
    """Exception raised for too large decode latency."""
