from dataclasses import dataclass


@dataclass(frozen=True)
class GPUHardwareSpec:
    flops: int
    gpu_mem: int
    gpu_bw: int


@dataclass(frozen=True)
class S3Spec:
    """Shared S3/object-store bandwidth configuration.

    S3 is modeled as a single shared pool with independent upload/download
    bandwidths.  Values are stored in bytes per second.
    """

    enabled: bool
    up_bw_bytes_per_s: int
    down_bw_bytes_per_s: int

    S3_DOWNLOAD_REQ_COSTS = 0.0004  # USD per 1k requests
    S3_DOWNLOAD_COST_GB = 0.09  # USD per GB
    S3_UPLOAD_REQ_COSTS = 0.0005  # USD per 1k requests
    S3_UPLOAD_COST_GB = 0.09  # USD per GB

    @classmethod
    def from_gbps(
        cls, enabled: bool = True, up_gbps: float = 25.0, down_gbps: float = 25.0
    ) -> S3Spec:
        """Build an S3Spec from gigabits-per-second values."""
        gbps_to_bytes_per_s = 1e9 / 8.0
        return cls(
            enabled=enabled,
            up_bw_bytes_per_s=int(up_gbps * gbps_to_bytes_per_s),
            down_bw_bytes_per_s=int(down_gbps * gbps_to_bytes_per_s),
        )


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


@dataclass(frozen=True)
class Hardware:
    name: str
    spec: HardwareSpec
