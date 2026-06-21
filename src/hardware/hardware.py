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
    price_usd_per_hour: float


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
            ram_mem=128 * 10**9,  # example value
            ram_bw=100 * 10**9,  # example value
            nvme_mem=1 * 10**12,  # example value
            nvme_bw=3 * 10**9,  # example value
            network_bw=10 * 10**9,
            price_usd_per_hour=4.0,
        ),
    }
