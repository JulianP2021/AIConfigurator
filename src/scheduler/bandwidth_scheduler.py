"""Global bandwidth scheduler for shared node/network links."""

from collections import defaultdict
from typing import Any

from src.hardware.hardware import HardwareSpec, S3Spec
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
    s3_spec: S3Spec
    transfers: list[DownloadRequest | UploadRequest]

    def __init__(self, nodes: list[Any], s3_spec: S3Spec | None = None):
        from src.hardware.hardware import S3Spec

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
        self.s3_spec = s3_spec or S3Spec.from_gbps(enabled=False)
        if self.s3_spec.enabled:
            print(
                f"S3 bandwidths: Up {self.s3_spec.up_bw_bytes_per_s / 1e6:.2f} MB/s, "
                f"Down {self.s3_spec.down_bw_bytes_per_s / 1e6:.2f} MB/s"
            )
        self.transfers = []

    def register(self, transfer: DownloadRequest | UploadRequest) -> None:
        """Register an active transfer so it receives a bandwidth share."""
        if transfer.active_legs and transfer.remaining_bytes > 0:
            self.transfers.append(transfer)

    def unregister(self, transfer: DownloadRequest | UploadRequest) -> None:
        """Remove a finished transfer from scheduling."""
        self.transfers.remove(transfer)

    def update_shares(self) -> None:
        """Recompute every active leg's ``bandwidth_bytes_per_ms``."""
        for transfer in self.transfers:
            for leg in transfer.active_legs:
                leg.bandwidth_bytes_per_ms = 0.0

        if not self.transfers:
            return

        ram_counts: dict[int, int] = defaultdict(int)
        ssd_counts: dict[int, int] = defaultdict(int)
        remote_up_counts: dict[int, int] = defaultdict(int)
        remote_down_counts: dict[int, int] = defaultdict(int)
        s3_upload_count = 0
        s3_download_count = 0

        for transfer in self.transfers:
            for leg in transfer.active_legs:
                if leg.bottleneck == "RAM_LOCAL":
                    ram_counts[leg.source_node_id] += 1
                elif leg.bottleneck == "SSD_LOCAL":
                    ssd_counts[leg.source_node_id] += 1
                elif leg.bottleneck == "NETWORK":
                    remote_up_counts[leg.source_node_id] += 1
                    remote_down_counts[leg.dest_node_id] += 1
                elif leg.bottleneck == "S3_UPLOAD":
                    s3_upload_count += 1
                elif leg.bottleneck == "S3_DOWNLOAD":
                    s3_download_count += 1

        for transfer in self.transfers:
            for leg in transfer.active_legs:
                if leg.bottleneck == "RAM_LOCAL":
                    node_bw = (
                        self.node_specs[leg.source_node_id].pcie_bw
                        / ram_counts[leg.source_node_id]
                    )
                    leg.bandwidth_bytes_per_ms = node_bw / 1000.0
                elif leg.bottleneck == "SSD_LOCAL":
                    node_bw = (
                        self.node_specs[leg.source_node_id].nvme_bw
                        / ssd_counts[leg.source_node_id]
                    )
                    leg.bandwidth_bytes_per_ms = node_bw / 1000.0
                elif leg.bottleneck == "NETWORK":
                    source_bw = (
                        self.node_specs[leg.source_node_id].network_inet_up
                        / remote_up_counts[leg.source_node_id]
                    )
                    dest_bw = (
                        self.node_specs[leg.dest_node_id].network_inet_down
                        / remote_down_counts[leg.dest_node_id]
                    )
                    leg.bandwidth_bytes_per_ms = min(source_bw, dest_bw) / 1000.0
                elif leg.bottleneck == "S3_UPLOAD":
                    leg.bandwidth_bytes_per_ms = (
                        self.s3_spec.up_bw_bytes_per_s
                        / max(s3_upload_count, 1)
                        / 1000.0
                    )
                elif leg.bottleneck == "S3_DOWNLOAD":
                    leg.bandwidth_bytes_per_ms = (
                        self.s3_spec.down_bw_bytes_per_s
                        / max(s3_download_count, 1)
                        / 1000.0
                    )

    def next_event_ms(self) -> float:
        """Return the earliest time at which any active transfer leg finishes.

        A leg finishes when either its fixed startup latency expires or its
        remaining bytes are transferred.  The returned time is bounded below
        by ``MIN_STEP_MS`` so that tiny remaining byte slivers do not drive the
        simulation through millions of infinitesimal event-loop iterations.
        """
        self.update_shares()
        min_time = float("inf")
        for transfer in self.transfers:
            for leg in transfer.active_legs:
                if leg.remaining_latency_ms > 0:
                    if leg.remaining_latency_ms < min_time:
                        min_time = leg.remaining_latency_ms
                elif leg.bandwidth_bytes_per_ms > 0 and leg.remaining_bytes > 0:
                    time_remaining = leg.remaining_bytes / leg.bandwidth_bytes_per_ms
                    if time_remaining < min_time:
                        min_time = time_remaining
        if min_time == float("inf"):
            return min_time
        return max(min_time, MIN_STEP_MS)

    def advance_time(self, time_ms: float) -> list[DownloadRequest | UploadRequest]:
        """Advance every active transfer by ``time_ms`` and return fully completed ones.

        This centralizes transfer progression so that the same bandwidth shares
        are used to both size and execute a step.  Each leg consumes its fixed
        startup latency before any bytes move.  Tracks whose current leg finishes
        are advanced to the next leg in that track; a transfer is returned as
        completed only once every track has been exhausted.
        """
        self.update_shares()
        completed: list[DownloadRequest | UploadRequest] = []

        # Snapshot the list because register/unregister mutate ``self.transfers``.
        for transfer in list(self.transfers):
            active_indices = [
                track_idx
                for track_idx in range(len(transfer.tracks))
                if transfer.current_legs[track_idx] < len(transfer.tracks[track_idx])
            ]
            if not active_indices:
                continue

            for track_idx in active_indices:
                leg = transfer.tracks[track_idx][transfer.current_legs[track_idx]]
                if leg.remaining_latency_ms > 0:
                    leg.remaining_latency_ms -= time_ms
                    if leg.remaining_latency_ms < 0:
                        leg.remaining_latency_ms = 0.0
                    log(
                        LOG_BANDWIDTH,
                        f"Transfer for request {transfer.request.id} on leg {leg.bottleneck} "
                        f"(track {track_idx}) latency advanced by {time_ms:.3f} ms, "
                        f"remaining_latency={leg.remaining_latency_ms:.3f} ms",
                    )
                else:
                    log(
                        LOG_BANDWIDTH,
                        f"Transfer for request {transfer.request.id} on leg {leg.bottleneck} "
                        f"(track {track_idx}) advanced by {time_ms:.3f} ms, "
                        f"Bandwidth={leg.bandwidth_bytes_per_ms:.3f} B/ms, "
                        f"remaining_bytes={leg.remaining_bytes:.3f}, "
                        f"transfering {leg.bandwidth_bytes_per_ms * time_ms:.3f} bytes",
                    )
                    leg.remaining_bytes -= leg.bandwidth_bytes_per_ms * time_ms

            # A leg is considered complete when no bytes remain.  Use a tiny
            # tolerance so floating-point drift does not leave near-zero slivers.
            finished_tracks = [
                track_idx
                for track_idx in active_indices
                if transfer.tracks[track_idx][
                    transfer.current_legs[track_idx]
                ].remaining_bytes
                <= 1e-9
                and transfer.tracks[track_idx][
                    transfer.current_legs[track_idx]
                ].remaining_latency_ms
                <= 0
            ]

            if finished_tracks:
                self.unregister(transfer)
                for track_idx in finished_tracks:
                    transfer.advance_track(track_idx)
                if transfer.active_legs:
                    self.register(transfer)
                else:
                    completed.append(transfer)

        return completed
