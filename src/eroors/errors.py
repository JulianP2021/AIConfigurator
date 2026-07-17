class PrefillError(Exception):
    """Exception raised for too large prefill queue."""


class PrefillLatencyError(Exception):
    """Exception raised for too large prefill latency.

    Attributes:
        prefill_only_ttft_ms: TTFT contribution from prefill-side phases only
            (prefill compute + prefill KV download + prefill KV upload).
            This excludes the decode KV download leg.
        ttft_sla_ms: The per-request TTFT SLA that was violated.
    """

    def __init__(
        self,
        message: str,
        prefill_only_ttft_ms: float | None = None,
        ttft_sla_ms: float | None = None,
    ):
        super().__init__(message)
        self.prefill_only_ttft_ms = prefill_only_ttft_ms
        self.ttft_sla_ms = ttft_sla_ms


class DecodeError(Exception):
    """Exception raised for too large decode queue."""


class DecodeLatencyError(Exception):
    """Exception raised for too large decode latency."""


class KVStoreTooSmallError(Exception):
    """Exception raised if decode can no longer download full KV after prefill upload."""
