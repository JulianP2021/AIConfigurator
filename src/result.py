from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SimulationResult:
    """Aggregated metrics for a distributed simulation run, designed to be
    comparable with NVIDIA AI Configurator outputs.
    """

    # Scenario identity
    scenario_name: str

    # Topology (extracted from scenario.nodes)
    total_gpus: int
    num_prefill_workers: int
    num_decode_workers: int
    prefill_gpus_per_worker: int
    decode_gpus_per_worker: int
    batch_size: int

    # Latency metrics (milliseconds)
    ttft: float  # avg wait-inclusive TTFT
    tpot: float  # avg decode_time_ms / osl
    kv_upload_time: float  # avg kv upload time per request
    kv_download_time: float  # avg kv download time per request
    request_latency: float  # avg wait-inclusive end-to-end latency
    max_request_latency: float  # max end-to-end latency across all requests
    max_ttft: float  # max TTFT across all requests
    max_tpot: float  # max TPOT across all requests

    # Throughput metrics
    tokens_per_second: float
    tokens_per_second_per_gpu: float
    tokens_per_second_per_user: float
    seq_per_second: float  # seq/s (completed sequences per second)
    concurrency: float  # average requests in flight

    # Resource
    memory_gb: float

    # Pricing
    compute_price_usd_per_hour: float
    s3_cost_usd_per_hour: float = 0.0
    s3_storage_cost_usd_per_hour: float = 0.0
    total_cost_usd_per_hour: float = 0.0

    # KV download diagnostics (operation counts)
    ram_download_requests: int = 0
    ssd_download_requests: int = 0
    s3_upload_requests: int = 0
    s3_download_requests: int = 0

    # Cache usage statistics (bytes)
    ram_cache_usage_bytes: float = 0.0
    ssd_cache_usage_bytes: float = 0.0
    s3_cache_usage_bytes: float = 0.0
    s3_peak_cache_usage_bytes: float = 0.0
    ram_cache_capacity_bytes: float = 0.0
    ssd_cache_capacity_bytes: float = 0.0

    # Raw per-request data for deeper analysis
    per_request_stats: list[dict[str, Any]] = field(default_factory=list)

    # Phase-level timing analytics (active / wait)
    avg_prefill_time_ms: float = 0.0
    avg_prefill_wait_ms: float = 0.0
    max_prefill_wait_ms: float = 0.0
    avg_prefill_download_active_ms: float = 0.0
    avg_prefill_download_wait_ms: float = 0.0
    avg_prefill_upload_active_ms: float = 0.0
    avg_prefill_upload_background_active_ms: float = 0.0
    avg_prefill_upload_wait_ms: float = 0.0
    avg_decode_time_ms: float = 0.0
    avg_decode_wait_ms: float = 0.0
    max_decode_wait_ms: float = 0.0
    avg_decode_download_active_ms: float = 0.0
    avg_decode_download_wait_ms: float = 0.0
    avg_decode_upload_active_ms: float = 0.0
    avg_decode_upload_background_active_ms: float = 0.0
    avg_decode_upload_wait_ms: float = 0.0
    avg_clean_ttft_ms: float = 0.0
    max_clean_ttft_ms: float = 0.0
    avg_clean_latency_ms: float = 0.0
    max_clean_latency_ms: float = 0.0

    # Rounding configuration for JSON export.  Keys match attribute names.
    _ROUND: dict[str, int] = field(
        default_factory=lambda: {
            "ttft": 3,
            "tpot": 3,
            "kv_upload_time": 3,
            "kv_download_time": 3,
            "request_latency": 3,
            "max_request_latency": 3,
            "max_ttft": 3,
            "max_tpot": 3,
            "tokens_per_second": 2,
            "tokens_per_second_per_gpu": 2,
            "tokens_per_second_per_user": 2,
            "seq_per_second": 3,
            "memory_gb": 2,
            "ram_cache_usage_bytes": 0,
            "ssd_cache_usage_bytes": 0,
            "s3_cache_usage_bytes": 0,
            "s3_peak_cache_usage_bytes": 0,
            "ram_cache_capacity_bytes": 0,
            "ssd_cache_capacity_bytes": 0,
            "compute_price_usd_per_hour": 4,
            "s3_cost_usd_per_hour": 6,
            "s3_storage_cost_usd_per_hour": 6,
            "total_cost_usd_per_hour": 4,
            "ram_download_requests": 0,
            "ssd_download_requests": 0,
            "s3_upload_requests": 0,
            "s3_download_requests": 0,
            "avg_prefill_time_ms": 3,
            "avg_prefill_wait_ms": 3,
            "max_prefill_wait_ms": 3,
            "avg_prefill_download_active_ms": 3,
            "avg_prefill_download_wait_ms": 3,
            "avg_prefill_upload_active_ms": 3,
            "avg_prefill_upload_background_active_ms": 3,
            "avg_prefill_upload_wait_ms": 3,
            "avg_decode_time_ms": 3,
            "avg_decode_wait_ms": 3,
            "max_decode_wait_ms": 3,
            "avg_decode_download_active_ms": 3,
            "avg_decode_download_wait_ms": 3,
            "avg_decode_upload_active_ms": 3,
            "avg_decode_upload_background_active_ms": 3,
            "avg_decode_upload_wait_ms": 3,
            "avg_clean_ttft_ms": 3,
            "max_clean_ttft_ms": 3,
            "avg_clean_latency_ms": 3,
            "max_clean_latency_ms": 3,
        },
        repr=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dict using the dataclass attribute names.

        Fields listed in ``_ROUND`` are rounded; everything else is exported
        as-is (e.g. ints, strings, unrounded lists).
        """
        raw = asdict(self)
        rounded: dict[str, Any] = {}
        for key, value in raw.items():
            if key.startswith("_"):
                continue
            if key == "per_request_stats":
                rounded[key] = value
                continue
            decimals = self._ROUND.get(key)
            if decimals is not None:
                rounded[key] = round(value, decimals)
            else:
                rounded[key] = value
        return rounded

    def __repr__(self) -> str:
        return (
            f"SimulationResult({self.scenario_name!r}, "
            f"GPUs={self.total_gpus}, "
            f"prefill_workers={self.num_prefill_workers}, "
            f"decode_workers={self.num_decode_workers}, "
            f"ttft={self.ttft:.2f}ms (clean {self.avg_clean_ttft_ms:.2f}), "
            f"max_ttft={self.max_ttft:.2f}ms (clean {self.max_clean_ttft_ms:.2f}), "
            f"tpot={self.tpot:.2f}ms, max_tpot={self.max_tpot:.2f}ms, "
            f"latency={self.request_latency:.2f}ms (clean {self.avg_clean_latency_ms:.2f}), "
            f"tok/s/gpu={self.tokens_per_second_per_gpu:.2f}, "
            f"seq/s={self.seq_per_second:.3f}, "
            f"concurrency={self.concurrency:.1f})"
        )
