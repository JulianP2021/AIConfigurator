"""Global bandwidth scheduler for shared node/network links."""

from collections import defaultdict

from src.hardware.hardware import HardwareSpec
from src.request.request import DownloadRequest, UploadRequest


class BandwidthScheduler:
    """Schedules KV upload/download bandwidth across all instances.

    Bandwidth is shared with equal-share fairness over three independent
    bottlenecks:
      * ``RAM_LOCAL``  : shares the node's ``ram_bw`` bus;
      * ``SSD_LOCAL``  : shares the node's ``nvme_bw`` bus;
      * ``NETWORK``    : uses the source node's ``network_inet_up`` and the
        destination node's ``network_inet_down``; effective rate is the
        minimum of the two shares.

    All bandwidth values are stored as bytes/second in ``HardwareSpec``.
    The scheduler exposes bytes/ms rates to instances so they can decrement
    ``remaining_bytes`` directly.
    """

    node_specs: dict[int, HardwareSpec]
    transfers: list[DownloadRequest | UploadRequest]

    def __init__(self, nodes: list):
        self.node_specs = {}
        for node in nodes:
            self.node_specs[node.id] = node.hardware.spec
        self.transfers = []

    def register(self, transfer: DownloadRequest | UploadRequest) -> None:
        """Register an active transfer so it receives a bandwidth share."""
        if transfer.active_leg and transfer.remaining_bytes > 0:
            self.transfers.append(transfer)

    def unregister(self, transfer: DownloadRequest | UploadRequest) -> None:
        """Remove a finished transfer from scheduling."""
        self.transfers.remove(transfer)

    def update_shares(self) -> None:
        """Recompute every active leg's ``bandwidth_bytes_per_ms``."""
        for transfer in self.transfers:
            transfer.bandwidth_bytes_per_ms = 0.0

        if not self.transfers:
            return

        ram_counts: dict[int, int] = defaultdict(int)
        ssd_counts: dict[int, int] = defaultdict(int)
        remote_up_counts: dict[int, int] = defaultdict(int)
        remote_down_counts: dict[int, int] = defaultdict(int)

        for transfer in self.transfers:
            leg = transfer.active_leg
            if leg is None:
                continue
            if leg.bottleneck == "RAM_LOCAL":
                ram_counts[leg.source_node_id] += 1
            elif leg.bottleneck == "SSD_LOCAL":
                ssd_counts[leg.source_node_id] += 1
            elif leg.bottleneck == "NETWORK":
                remote_up_counts[leg.source_node_id] += 1
                remote_down_counts[leg.dest_node_id] += 1

        for transfer in self.transfers:
            leg = transfer.active_leg
            if leg is None:
                continue
            if leg.bottleneck == "RAM_LOCAL":
                node_bw = (
                    self.node_specs[leg.source_node_id].ram_bw
                    / ram_counts[leg.source_node_id]
                )
                transfer.bandwidth_bytes_per_ms = node_bw / 1000.0
            elif leg.bottleneck == "SSD_LOCAL":
                node_bw = (
                    self.node_specs[leg.source_node_id].nvme_bw
                    / ssd_counts[leg.source_node_id]
                )
                transfer.bandwidth_bytes_per_ms = node_bw / 1000.0
            elif leg.bottleneck == "NETWORK":
                source_bw = (
                    self.node_specs[leg.source_node_id].network_inet_up
                    / remote_up_counts[leg.source_node_id]
                )
                dest_bw = (
                    self.node_specs[leg.dest_node_id].network_inet_down
                    / remote_down_counts[leg.dest_node_id]
                )
                transfer.bandwidth_bytes_per_ms = min(source_bw, dest_bw) / 1000.0

    def next_event_ms(self) -> float:
        """Return the earliest time at which any active transfer leg finishes."""
        self.update_shares()
        min_time = float("inf")
        for transfer in self.transfers:
            leg = transfer.active_leg
            if leg is None:
                continue
            if leg.bandwidth_bytes_per_ms > 0 and leg.remaining_bytes > 0:
                time_remaining = leg.remaining_bytes / leg.bandwidth_bytes_per_ms
                if time_remaining < min_time:
                    min_time = time_remaining
        return min_time
