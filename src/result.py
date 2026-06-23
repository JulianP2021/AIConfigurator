from dataclasses import dataclass, field
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
    ttft: float  # avg prefill_time_ms + kv transfer overhead
    tpot: float  # avg decode_time_ms / osl
    request_latency: float  # end-to-end per request
    max_ttft: float  # max TTFT across all requests
    max_tpot: float  # max TPOT across all requests

    # Throughput metrics
    tokens_per_second: float
    tokens_per_second_per_gpu: float
    tokens_per_second_per_user: float
    request_rate: float  # req/s
    concurrency: float  # average requests in flight

    # Resource
    memory_gb: float

    # Pricing
    price_usd_per_hour: float

    # Raw per-request data for deeper analysis
    per_request_stats: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # Sanity checks
        assert self.total_gpus > 0, "total_gpus must be > 0"
        assert self.request_rate >= 0, "request_rate must be >= 0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "total_gpus": self.total_gpus,
            "num_prefill_workers": self.num_prefill_workers,
            "num_decode_workers": self.num_decode_workers,
            "prefill_gpus_per_worker": self.prefill_gpus_per_worker,
            "decode_gpus_per_worker": self.decode_gpus_per_worker,
            "batch_size": self.batch_size,
            "ttft_ms": round(self.ttft, 3),
            "tpot_ms": round(self.tpot, 3),
            "request_latency_ms": round(self.request_latency, 3),
            "max_ttft_ms": round(self.max_ttft, 3),
            "max_tpot_ms": round(self.max_tpot, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "tokens_per_second_per_gpu": round(self.tokens_per_second_per_gpu, 2),
            "tokens_per_second_per_user": round(self.tokens_per_second_per_user, 2),
            "request_rate": round(self.request_rate, 3),
            "concurrency": round(self.concurrency, 2),
            "memory_gb": round(self.memory_gb, 2),
            "price_usd_per_hour": round(self.price_usd_per_hour, 4),
            "num_requests": len(self.per_request_stats),
        }

    def __repr__(self) -> str:
        return (
            f"SimulationResult({self.scenario_name!r}, "
            f"GPUs={self.total_gpus}, "
            f"prefill_workers={self.num_prefill_workers}, "
            f"decode_workers={self.num_decode_workers}, "
            f"ttft={self.ttft:.2f}ms, max_ttft={self.max_ttft:.2f}ms, "
            f"tpot={self.tpot:.2f}ms, max_tpot={self.max_tpot:.2f}ms, "
            f"latency={self.request_latency:.2f}ms, "
            f"tok/s/gpu={self.tokens_per_second_per_gpu:.2f}, "
            f"req/s={self.request_rate:.3f}, "
            f"concurrency={self.concurrency:.1f})"
        )
