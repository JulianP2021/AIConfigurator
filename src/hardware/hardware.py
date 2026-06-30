import json
import pathlib

from dataclasses import dataclass
from typing import ClassVar

from src.logger import LOG_SIMULATION, log


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
    network_inet_up: int
    network_inet_down: int
    price_usd_per_hour: float
    price_inet_up: float
    price_inet_down: float


def _load_machine_presets() -> dict[str, HardwareSpec]:
    """Load machine specs from the cached Vast.ai machine database."""
    db_path = pathlib.Path(__file__).parent / "_machine_db.json"
    if not db_path.exists():
        return {}

    db: dict[str, dict[str, int | float | str]] = json.loads(
        db_path.read_text(encoding="utf-8")
    )
    presets: dict[str, HardwareSpec] = {}
    for entry in db.values():
        num_gpus = int(entry.get("num_gpus", 1))
        flops = entry.get("flops", 0)
        # Some entries report total FLOPS for all GPUs; normalise per-GPU.
        per_gpu_flops = int(flops) // max(num_gpus, 1)

        gpu_spec = GPUHardwareSpec(
            flops=int(per_gpu_flops) * 10**12,
            gpu_mem=int(entry.get("gpu_mem", 0)) * 10**9,
            gpu_bw=int(entry.get("gpu_bw", 0)) * 10**9,
        )
        spec = HardwareSpec(
            gpu_hardware=gpu_spec,
            num_gpus=num_gpus,
            ram_mem=int(entry.get("ram_mem", 0)) * 10**9,
            ram_bw=int(entry.get("ram_bw", 0)) * 10**9,
            nvme_mem=int(entry.get("nvme_mem", 0)) * 10**9,
            nvme_bw=int(entry.get("nvme_bw", 0)) * 10**9,
            network_inet_up=int(entry.get("network_inet_up", 0)) * 10**6,
            network_inet_down=int(entry.get("network_inet_down", 0)) * 10**6,
            price_usd_per_hour=float(entry.get("price_usd_per_hour", 0.0)),  # $/hour
            price_inet_up=float(entry.get("price_inet_up", 0.0)),  # $/TB
            price_inet_down=float(entry.get("price_inet_down", 0.0)),  # $/TB
        )
        presets[str(entry["name"])] = spec
    return presets


@dataclass(frozen=True)
class Hardware:
    name: str
    spec: HardwareSpec

    @classmethod
    def from_name(cls, name: str):
        if name not in cls.PRESETS:
            raise ValueError(f"Unknown hardware preset: {name}")
        spec = cls.PRESETS[name]

        log(LOG_SIMULATION, f"Loaded hardware preset '{name}': {spec}")
        return cls(name=name, spec=spec)

    PRESETS: ClassVar[dict[str, HardwareSpec]] = _load_machine_presets()
