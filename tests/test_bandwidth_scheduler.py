"""Tests for the global BandwidthScheduler."""

import pytest

from src.hardware.hardware import Hardware, S3Spec
from src.request.request import DownloadRequest, TransferLeg, UploadRequest
from src.scheduler.bandwidth_scheduler import BandwidthScheduler


class FakeRequest:
    def __init__(self, req_id: int = 0):
        self.id = req_id
        self.kv_download_time_ms = 0.0
        self.kv_upload_time_ms = 0.0


class FakeNode:
    def __init__(self, node_id: int, hardware: Hardware):
        self.id = node_id
        self.hardware = hardware


class TestBandwidthScheduler:
    def test_empty_scheduler_returns_infinity(self, tiny_hardware: Hardware):
        scheduler = BandwidthScheduler([FakeNode(0, tiny_hardware)])
        assert scheduler.next_event_ms() == float("inf")

    def test_ram_local_share_single_transfer(self, tiny_hardware: Hardware):
        scheduler = BandwidthScheduler([FakeNode(0, tiny_hardware)])
        req = FakeRequest()
        ur = UploadRequest(req, [[TransferLeg(10_000_000, 0, 0, "RAM_LOCAL")]])
        scheduler.register(ur)

        # RAM_LOCAL has 0 ms latency, so 10 MB / (10 GB/s) = 1 ms
        assert scheduler.next_event_ms() == pytest.approx(1.0, rel=1e-3)
        assert (
            ur.active_legs[0].bandwidth_bytes_per_ms
            == tiny_hardware.spec.pcie_bw / 1000.0
        )

    def test_ram_local_share_two_transfers(self, tiny_hardware: Hardware):
        scheduler = BandwidthScheduler([FakeNode(0, tiny_hardware)])
        req_a = FakeRequest()
        req_b = FakeRequest()
        ur_a = UploadRequest(req_a, [[TransferLeg(5_000_000, 0, 0, "RAM_LOCAL")]])
        ur_b = UploadRequest(req_b, [[TransferLeg(5_000_000, 0, 0, "RAM_LOCAL")]])
        scheduler.register(ur_a)
        scheduler.register(ur_b)

        # Equal share: each gets 5 GB/s, so 5 MB / 5 GB/s = 1 ms
        assert scheduler.next_event_ms() == pytest.approx(1.0, rel=1e-3)
        assert (
            ur_a.active_legs[0].bandwidth_bytes_per_ms
            == tiny_hardware.spec.pcie_bw / 2 / 1000.0
        )
        assert (
            ur_b.active_legs[0].bandwidth_bytes_per_ms
            == tiny_hardware.spec.pcie_bw / 2 / 1000.0
        )

    def test_network_share_is_min_of_up_and_down(self, tiny_hardware: Hardware):
        scheduler = BandwidthScheduler([
            FakeNode(0, tiny_hardware),
            FakeNode(1, tiny_hardware),
        ])
        req = FakeRequest()
        dr = DownloadRequest(req, [[TransferLeg(100_000_000, 0, 1, "NETWORK")]])
        scheduler.register(dr)

        # NETWORK has 0 ms latency. Source up = 100 MB/s, dest down = 200 MB/s => bottleneck is up.
        expected_ms = 100_000_000 / (tiny_hardware.spec.network_inet_up / 1000.0)
        assert scheduler.next_event_ms() == pytest.approx(expected_ms, rel=1e-3)

    def test_ssd_local_uses_nvme_bw(self, tiny_hardware: Hardware):
        scheduler = BandwidthScheduler([FakeNode(0, tiny_hardware)])
        req = FakeRequest()
        dr = DownloadRequest(req, [[TransferLeg(1_000_000_000, 0, 0, "SSD_LOCAL")]])
        scheduler.register(dr)

        # First event is the 0.1 ms SSD latency, then the transfer time.
        assert scheduler.next_event_ms() == pytest.approx(0.1, rel=1e-3)

    def test_ssd_local_uses_nvme_bw_after_latency(self, tiny_hardware: Hardware):
        scheduler = BandwidthScheduler([FakeNode(0, tiny_hardware)])
        req = FakeRequest()
        dr = DownloadRequest(req, [[TransferLeg(1_000_000_000, 0, 0, "SSD_LOCAL")]])
        scheduler.register(dr)

        scheduler.advance_time(0.1)
        expected_ms = 1_000_000_000 / (tiny_hardware.spec.nvme_bw / 1000.0)
        assert scheduler.next_event_ms() == pytest.approx(expected_ms, rel=1e-3)

    def test_unregister_removes_transfer(self, tiny_hardware: Hardware):
        scheduler = BandwidthScheduler([FakeNode(0, tiny_hardware)])
        req = FakeRequest()
        ur = UploadRequest(req, [[TransferLeg(10_000_000, 0, 0, "RAM_LOCAL")]])
        scheduler.register(ur)
        scheduler.unregister(ur)
        assert scheduler.next_event_ms() == float("inf")

    def test_register_skips_zero_byte_transfer(self, tiny_hardware: Hardware):
        scheduler = BandwidthScheduler([FakeNode(0, tiny_hardware)])
        req = FakeRequest()
        ur = UploadRequest(req, [[TransferLeg(0, 0, 0, "RAM_LOCAL")]])
        scheduler.register(ur)
        assert scheduler.next_event_ms() == float("inf")

    def test_advance_track_recomputes_bandwidth(self, tiny_hardware: Hardware):
        scheduler = BandwidthScheduler([FakeNode(0, tiny_hardware)])
        req = FakeRequest()
        # Single track with two sequential legs; after the first leg the second
        # should get the same full-bus bandwidth.
        ur = UploadRequest(
            req,
            [
                [
                    TransferLeg(5_000_000, 0, 0, "RAM_LOCAL"),
                    TransferLeg(5_000_000, 0, 0, "RAM_LOCAL"),
                ],
            ],
        )
        scheduler.register(ur)
        # update_shares zeroes all active-leg bandwidths first, then recomputes
        scheduler.update_shares()
        first_bw = ur.active_legs[0].bandwidth_bytes_per_ms
        ur.advance_track(0)
        scheduler.update_shares()
        assert ur.active_legs[0].bandwidth_bytes_per_ms == first_bw

    def test_parallel_tracks_share_bottleneck(self, tiny_hardware: Hardware):
        scheduler = BandwidthScheduler([FakeNode(0, tiny_hardware)])
        req = FakeRequest()
        # Two parallel RAM tracks on the same node share pcie_bw.
        ur = UploadRequest(
            req,
            [
                [TransferLeg(5_000_000, 0, 0, "RAM_LOCAL")],
                [TransferLeg(5_000_000, 0, 0, "RAM_LOCAL")],
            ],
        )
        scheduler.register(ur)
        assert scheduler.next_event_ms() == pytest.approx(1.0, rel=1e-3)
        for leg in ur.active_legs:
            assert leg.bandwidth_bytes_per_ms == tiny_hardware.spec.pcie_bw / 2 / 1000.0

    def test_s3_download_bandwidth(self, tiny_hardware: Hardware):
        s3_spec = S3Spec.from_gbps(enabled=True, up_gbps=25.0, down_gbps=25.0)
        scheduler = BandwidthScheduler([FakeNode(0, tiny_hardware)], s3_spec=s3_spec)
        req = FakeRequest()
        dr = DownloadRequest(req, [[TransferLeg(1_000_000_000, -1, 0, "S3_DOWNLOAD")]])
        scheduler.register(dr)

        # First event is the 50 ms S3 latency.
        assert scheduler.next_event_ms() == pytest.approx(50.0, rel=1e-3)

    def test_s3_download_bandwidth_after_latency(self, tiny_hardware: Hardware):
        s3_spec = S3Spec.from_gbps(enabled=True, up_gbps=25.0, down_gbps=25.0)
        scheduler = BandwidthScheduler([FakeNode(0, tiny_hardware)], s3_spec=s3_spec)
        req = FakeRequest()
        dr = DownloadRequest(req, [[TransferLeg(1_000_000_000, -1, 0, "S3_DOWNLOAD")]])
        scheduler.register(dr)

        scheduler.advance_time(50.0)
        expected_ms = 1_000_000_000 / (s3_spec.down_bw_bytes_per_s / 1000.0)
        assert scheduler.next_event_ms() == pytest.approx(expected_ms, rel=1e-3)

    def test_s3_upload_shared(self, tiny_hardware: Hardware):
        s3_spec = S3Spec.from_gbps(enabled=True, up_gbps=25.0, down_gbps=25.0)
        scheduler = BandwidthScheduler([FakeNode(0, tiny_hardware)], s3_spec=s3_spec)
        req_a = FakeRequest()
        req_b = FakeRequest()
        ur_a = UploadRequest(req_a, [[TransferLeg(1_000_000_000, 0, -1, "S3_UPLOAD")]])
        ur_b = UploadRequest(req_b, [[TransferLeg(1_000_000_000, 0, -1, "S3_UPLOAD")]])
        scheduler.register(ur_a)
        scheduler.register(ur_b)

        # First event is the shared 50 ms S3 latency.
        assert scheduler.next_event_ms() == pytest.approx(50.0, rel=1e-3)

    def test_s3_upload_shared_after_latency(self, tiny_hardware: Hardware):
        s3_spec = S3Spec.from_gbps(enabled=True, up_gbps=25.0, down_gbps=25.0)
        scheduler = BandwidthScheduler([FakeNode(0, tiny_hardware)], s3_spec=s3_spec)
        req_a = FakeRequest()
        req_b = FakeRequest()
        ur_a = UploadRequest(req_a, [[TransferLeg(1_000_000_000, 0, -1, "S3_UPLOAD")]])
        ur_b = UploadRequest(req_b, [[TransferLeg(1_000_000_000, 0, -1, "S3_UPLOAD")]])
        scheduler.register(ur_a)
        scheduler.register(ur_b)

        scheduler.advance_time(50.0)
        # Equal share of 25 Gbps -> each gets half.
        expected_ms = 1_000_000_000 / (s3_spec.up_bw_bytes_per_s / 2 / 1000.0)
        assert scheduler.next_event_ms() == pytest.approx(expected_ms, rel=1e-3)
