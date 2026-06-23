from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class GPUHardwareSpec:
    flops: int
    gpu_mem: int
    gpu_bw: int


@dataclass(frozen=True)
class HardwareSpec:
    gpu_hardware: GPUHardwareSpec
    num_gpus: int
    ram_mem: int
    ram_bw: int
    nvme_mem: int
    nvme_bw: int
    network_bw: int
    network_inet_up: int
    network_inet_down: int
    price_usd_per_hour: float
    price_inet_up: float
    price_inet_down: float


@dataclass(frozen=True)
class Hardware:
    name: str
    spec: HardwareSpec

    @classmethod
    def from_name(cls, name: str):
        if name not in cls.PRESETS:
            raise ValueError(f"Unknown hardware preset: {name}")
        spec = cls.PRESETS[name]
        return cls(name=name, spec=spec)

    PRESETS: ClassVar[dict[str, HardwareSpec]] = {
        "DGX SPARK": HardwareSpec(
            gpu_hardware=GPUHardwareSpec(
                flops=213 * 10**12, gpu_mem=128 * 10**9, gpu_bw=273 * 10**9
            ),
            num_gpus=1,
            ram_mem=128 * 10**9,
            ram_bw=100 * 10**9,
            nvme_mem=1 * 10**12,
            nvme_bw=3 * 10**9,
            network_bw=10 * 10**9,
            network_inet_up=0,
            network_inet_down=0,
            price_usd_per_hour=4.0,
            price_inet_up=0.0,
            price_inet_down=0.0,
        ),
        "B200": HardwareSpec(
            gpu_hardware=GPUHardwareSpec(
                flops=4500 * 10**12, gpu_mem=192 * 10**9, gpu_bw=8000 * 10**9
            ),
            num_gpus=1,
            ram_mem=0,
            ram_bw=0,
            nvme_mem=0,
            nvme_bw=0,
            network_bw=0,
            network_inet_up=0,
            network_inet_down=0,
            price_usd_per_hour=12.0,
            price_inet_up=0.0,
            price_inet_down=0.0,
        ),
        "H100SXM": HardwareSpec(
            gpu_hardware=GPUHardwareSpec(
                flops=1979 * 10**12, gpu_mem=80 * 10**9, gpu_bw=3350 * 10**9
            ),
            num_gpus=1,
            ram_mem=0,
            ram_bw=0,
            nvme_mem=0,
            nvme_bw=0,
            network_bw=0,
            network_inet_up=0,
            network_inet_down=0,
            price_usd_per_hour=8.0,
            price_inet_up=0.0,
            price_inet_down=0.0,
        ),
    }
