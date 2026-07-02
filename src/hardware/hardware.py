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
    nvme_mem: int
    nvme_bw: int
    network_inet_up: int  # in bytes per second
    network_inet_down: int  # in bytes per second
    cpu_cores: int
    cpu_cores_effective: float
    cpu_ghz: float
    cpu_name: str
    cpu_ram: int
    disk_bw: float
    disk_name: str
    disk_space: float
    dlperf: float
    dlperf_per_dphtotal: float
    dph_base: float
    dph_total: float
    geolocation: str
    gpu_display_active: bool
    gpu_frac: float
    gpu_lanes: int
    gpu_max_power: float
    gpu_max_temp: float
    has_avx: int
    host_id: int
    inet_down_cost: float
    inet_up_cost: float
    mobo_name: str
    os_version: str
    pci_gen: float
    pcie_bw: float
    network_bw: float
    reliability: float
    reliability_mult: float
    score: float
    storage_cost: float
    storage_total_cost: float
    verification: str


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
            flops=int(per_gpu_flops),
            gpu_mem=int(entry.get("gpu_mem", 0)),
            gpu_bw=int(entry.get("gpu_bw", 0)),
        )
        spec = HardwareSpec(
            gpu_hardware=gpu_spec,
            num_gpus=num_gpus,
            nvme_mem=int(entry.get("nvme_mem", 0)),
            nvme_bw=int(entry.get("nvme_bw", 0)),
            network_inet_up=int(entry.get("network_inet_up", 0)),
            network_inet_down=int(entry.get("network_inet_down", 0)),
            cpu_cores=int(entry.get("cpu_cores", 0)),
            cpu_cores_effective=float(entry.get("cpu_cores_effective", 0.0)),
            cpu_ghz=float(entry.get("cpu_ghz", 0.0)),
            cpu_name=str(entry.get("cpu_name", "")),
            cpu_ram=int(entry.get("cpu_ram", 0)),
            disk_bw=float(entry.get("disk_bw", 0.0)),
            disk_name=str(entry.get("disk_name", "")),
            disk_space=float(entry.get("disk_space", 0.0)),
            dlperf=float(entry.get("dlperf", 0.0)),
            dlperf_per_dphtotal=float(entry.get("dlperf_per_dphtotal", 0.0)),
            dph_base=float(entry.get("dph_base", 0.0)),
            dph_total=float(entry.get("dph_total", 0.0)),
            geolocation=str(entry.get("geolocation", "")),
            gpu_display_active=bool(entry.get("gpu_display_active", False)),
            gpu_frac=float(entry.get("gpu_frac", 0.0)),
            gpu_lanes=int(entry.get("gpu_lanes", 0)),
            gpu_max_power=float(entry.get("gpu_max_power", 0.0)),
            gpu_max_temp=float(entry.get("gpu_max_temp", 0.0)),
            has_avx=int(entry.get("has_avx", 0)),
            host_id=int(entry.get("host_id", 0)),
            inet_down_cost=float(entry.get("inet_down_cost", 0.0)),
            inet_up_cost=float(entry.get("inet_up_cost", 0.0)),
            mobo_name=str(entry.get("mobo_name", "")),
            os_version=str(entry.get("os_version", "")),
            pci_gen=float(entry.get("pci_gen", 0.0)),
            pcie_bw=float(entry.get("pcie_bw", 0.0)),
            network_bw=float(entry.get("network_bw", 0.0)),
            reliability=float(entry.get("reliability", 0.0)),
            reliability_mult=float(entry.get("reliability_mult", 0.0)),
            score=float(entry.get("score", 0.0)),
            storage_cost=float(entry.get("storage_cost", 0.0)),
            storage_total_cost=float(entry.get("storage_total_cost", 0.0)),
            verification=str(entry.get("verification", "")),
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
