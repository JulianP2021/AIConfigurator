"""Global bandwidth scheduler for shared node/network links."""

from collections import defaultdict
from typing import Any

from src.hardware.hardware import HardwareSpec
from src.logger import LOG_BANDWIDTH, log
from src.request.request import DownloadRequest, UploadRequest


# Minimum time granularity for transfer-driven event steps.  Steps smaller than
# this are rounded up to avoid excessive event-loop iterations caused by tiny
# remaining transfer bytes.
MIN_STEP_MS = 0.1


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

    def __init__(self, nodes: list[Any]):
        self.node_specs = {}
        for node in nodes:
            self.node_specs[node.id] = node.hardware.spec
            print(
                f"Node {node.id} bandwidths: "
                f"RAM {node.hardware.spec.pcie_bw / 1e6:.2f} MB/s, "
                f"SSD {node.hardware.spec.nvme_bw / 1e6:.2f} MB/s, "
                f"Network Up {node.hardware.spec.network_inet_up / 1e6:.2f} MB/s, "
                f"Network Down {node.hardware.spec.network_inet_down / 1e6:.2f} MB/s"
            )
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
                    self.node_specs[leg.source_node_id].pcie_bw
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
        """Return the earliest time at which any active transfer leg finishes.

        The returned time is bounded below by ``MIN_STEP_MS`` so that tiny
        remaining byte slivers do not drive the simulation through millions of
        infinitesimal event-loop iterations.
        """
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
        if min_time == float("inf"):
            return min_time
        return max(min_time, MIN_STEP_MS)

    def advance_time(self, time_ms: float) -> list[DownloadRequest | UploadRequest]:
        """Advance every active transfer by ``time_ms`` and return fully completed ones.

        This centralizes transfer progression so that the same bandwidth shares
        are used to both size and execute a step.  Multi-leg transfers whose
        current leg finishes are advanced to the next leg and re-registered;
        transfers with no remaining legs are returned as completed.
        """
        self.update_shares()
        completed: list[DownloadRequest | UploadRequest] = []

        # Snapshot the list because register/unregister mutate ``self.transfers``.
        for transfer in list(self.transfers):
            leg = transfer.active_leg
            if leg is None:
                continue

            log(
                LOG_BANDWIDTH,
                f"Transfer from {transfer.source_node_id} to {transfer.dest_node_id} on leg {leg.bottleneck} advanced by {time_ms:.3f} ms, "
                f"Bandwidth={leg.bandwidth_bytes_per_ms:.3f} B/ms, "
                f"remaining_bytes={leg.remaining_bytes:.3f}, "
                f"transfering {leg.bandwidth_bytes_per_ms * time_ms:.3f} bytes",
            )
            leg.remaining_bytes -= leg.bandwidth_bytes_per_ms * time_ms
            # A leg is considered complete when no bytes remain.  Use a tiny
            # tolerance so floating-point drift does not leave near-zero slivers.
            if leg.remaining_bytes <= 1e-9:
                self.unregister(transfer)
                if transfer.advance_leg():
                    self.register(transfer)
                else:
                    completed.append(transfer)

        return completed
