from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class HardwareSpec:
    flops: int
    gpu_mem: int
    gpu_bw: int
    ram_mem: int
    ram_bw: int
    nvme_mem: int
    nvme_bw: int
    network_bw: int
    price_ms: float


@dataclass
class Hardware:
    name: str
    spec: HardwareSpec

    def __init__(
        self,
        name: str,
        flops: int,
        gpu_mem: int,
        gpu_bw: int,
        ram_mem: int,
        ram_bw: int,
        nvme_mem: int,
        nvme_bw: int,
        network_bw: int,
        price_ms: float,
    ):
        self.name = name
        spec = HardwareSpec(
            flops=flops,
            gpu_mem=gpu_mem,
            gpu_bw=gpu_bw,
            ram_mem=ram_mem,
            ram_bw=ram_bw,
            nvme_mem=nvme_mem,
            nvme_bw=nvme_bw,
            network_bw=network_bw,
            price_ms=price_ms,
        )
        self.spec = spec

    PRESETS: ClassVar[dict[str, HardwareSpec]] = {
        "DGX SPARK": HardwareSpec(
            flops=213 * 10**12,
            gpu_mem=128 * 10**9,
            gpu_bw=273 * 10**9,
            ram_mem=128 * 10**9,  # example value
            ram_bw=100 * 10**9,  # example value
            nvme_mem=1 * 10**12,  # example value
            nvme_bw=3 * 10**12,  # example value
            network_bw=10 * 10**9,
            price_ms=10000.0,
        ),
    }
