"""Tests for the global BandwidthScheduler."""

import pytest

from src.hardware.hardware import Hardware, HardwareSpec, S3Spec
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

        # NETWORK has 0 ms latency. Source inter-node up = 100 Mb/s, dest inter-node down = 200 Mb/s => bottleneck is up.
        expected_ms = 100_000_000 / (tiny_hardware.spec.network_inter_node_up / 8000.0)
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

    def test_ram_local_pcie_not_scaled_by_gpu_count(self, tiny_hardware: Hardware):
        """PCIe bandwidth is total node bandwidth and must not scale with num_gpus."""
        spec = tiny_hardware.spec
        multi_gpu_spec = HardwareSpec(
            gpu_hardware=spec.gpu_hardware,
            num_gpus=4,
            nvme_mem=spec.nvme_mem,
            nvme_bw=spec.nvme_bw,
            network_inet_up=spec.network_inet_up,
            network_inet_down=spec.network_inet_down,
            network_inter_node_up=spec.network_inter_node_up,
            network_inter_node_down=spec.network_inter_node_down,
            cpu_cores=spec.cpu_cores,
            cpu_cores_effective=spec.cpu_cores_effective,
            cpu_ghz=spec.cpu_ghz,
            cpu_name=spec.cpu_name,
            cpu_ram=spec.cpu_ram,
            disk_name=spec.disk_name,
            dlperf=spec.dlperf,
            dlperf_per_dphtotal=spec.dlperf_per_dphtotal,
            dph_base=spec.dph_base,
            geolocation=spec.geolocation,
            gpu_display_active=spec.gpu_display_active,
            gpu_frac=spec.gpu_frac,
            gpu_lanes=spec.gpu_lanes,
            gpu_max_power=spec.gpu_max_power,
            gpu_max_temp=spec.gpu_max_temp,
            has_avx=spec.has_avx,
            host_id=spec.host_id,
            inet_down_cost=spec.inet_down_cost,
            inet_up_cost=spec.inet_up_cost,
            mobo_name=spec.mobo_name,
            os_version=spec.os_version,
            pci_gen=spec.pci_gen,
            pcie_bw=spec.pcie_bw,
            network_bw=spec.network_bw,
            reliability=spec.reliability,
            reliability_mult=spec.reliability_mult,
            score=spec.score,
            storage_cost=spec.storage_cost,
            storage_total_cost=spec.storage_total_cost,
            verification=spec.verification,
            nvlink_bw=0.0,
        )
        multi_gpu_hardware = Hardware(name="multi-gpu-pcie", spec=multi_gpu_spec)
        scheduler = BandwidthScheduler([FakeNode(0, multi_gpu_hardware)])

        # A single transfer gets the full node PCIe bandwidth.
        req_single = FakeRequest()
        ur_single = UploadRequest(
            req_single, [[TransferLeg(10_000_000, 0, 0, "RAM_LOCAL")]]
        )
        scheduler.register(ur_single)
        assert scheduler.next_event_ms() == pytest.approx(4.0, rel=1e-3)
        assert ur_single.active_legs[0].bandwidth_bytes_per_ms == pytest.approx(
            multi_gpu_hardware.spec.pcie_bw / 1000.0 / 4, rel=1e-3
        )

        # Multiple transfers share the same total node PCIe bandwidth but are capped at per gpu bw
        scheduler.unregister(ur_single)
        req_a = FakeRequest()
        req_b = FakeRequest()
        ur_a = UploadRequest(req_a, [[TransferLeg(10_000_000, 0, 0, "RAM_LOCAL")]])
        ur_b = UploadRequest(req_b, [[TransferLeg(10_000_000, 0, 0, "RAM_LOCAL")]])
        scheduler.register(ur_a)
        scheduler.register(ur_b)
        scheduler.update_shares()
        expected_share = multi_gpu_hardware.spec.pcie_bw / 4 / 1000.0
        assert ur_a.active_legs[0].bandwidth_bytes_per_ms == expected_share
        assert ur_b.active_legs[0].bandwidth_bytes_per_ms == expected_share

    def test_ram_local_nvlink_scales_by_gpu_count(self, tiny_hardware: Hardware):
        """NVLink bandwidth is per-GPU, so aggregate bandwidth scales with num_gpus."""
        spec = tiny_hardware.spec
        nvlink = spec.pcie_bw / 4  # arbitrary per-GPU NVLink value
        multi_gpu_spec = HardwareSpec(
            gpu_hardware=spec.gpu_hardware,
            num_gpus=4,
            nvme_mem=spec.nvme_mem,
            nvme_bw=spec.nvme_bw,
            network_inet_up=spec.network_inet_up,
            network_inet_down=spec.network_inet_down,
            network_inter_node_up=spec.network_inter_node_up,
            network_inter_node_down=spec.network_inter_node_down,
            cpu_cores=spec.cpu_cores,
            cpu_cores_effective=spec.cpu_cores_effective,
            cpu_ghz=spec.cpu_ghz,
            cpu_name=spec.cpu_name,
            cpu_ram=spec.cpu_ram,
            disk_name=spec.disk_name,
            dlperf=spec.dlperf,
            dlperf_per_dphtotal=spec.dlperf_per_dphtotal,
            dph_base=spec.dph_base,
            geolocation=spec.geolocation,
            gpu_display_active=spec.gpu_display_active,
            gpu_frac=spec.gpu_frac,
            gpu_lanes=spec.gpu_lanes,
            gpu_max_power=spec.gpu_max_power,
            gpu_max_temp=spec.gpu_max_temp,
            has_avx=spec.has_avx,
            host_id=spec.host_id,
            inet_down_cost=spec.inet_down_cost,
            inet_up_cost=spec.inet_up_cost,
            mobo_name=spec.mobo_name,
            os_version=spec.os_version,
            pci_gen=spec.pci_gen,
            pcie_bw=0.0,
            network_bw=spec.network_bw,
            reliability=spec.reliability,
            reliability_mult=spec.reliability_mult,
            score=spec.score,
            storage_cost=spec.storage_cost,
            storage_total_cost=spec.storage_total_cost,
            verification=spec.verification,
            nvlink_bw=nvlink,
        )
        multi_gpu_hardware = Hardware(name="multi-gpu-nvlink", spec=multi_gpu_spec)
        scheduler = BandwidthScheduler([FakeNode(0, multi_gpu_hardware)])

        # A single NVLink transfer is capped at one GPU's link bandwidth.
        req_single = FakeRequest()
        ur_single = UploadRequest(
            req_single, [[TransferLeg(10_000_000, 0, 0, "RAM_LOCAL")]]
        )
        scheduler.register(ur_single)
        expected_single_ms = 10_000_000 / (nvlink / 1000.0) * 4
        assert scheduler.next_event_ms() == pytest.approx(expected_single_ms, rel=1e-3)
        assert ur_single.active_legs[0].bandwidth_bytes_per_ms == pytest.approx(
            nvlink / 1000.0 / 4, rel=1e-3
        )

        # Many transfers share the aggregate NVLink bandwidth.
        scheduler.unregister(ur_single)
        transfers = [
            UploadRequest(
                FakeRequest(i), [[TransferLeg(10_000_000, 0, 0, "RAM_LOCAL")]]
            )
            for i in range(8)
        ]
        for ur in transfers:
            scheduler.register(ur)
        scheduler.update_shares()
        # 8 transfers share 4 NVLink links -> each gets half a link.
        expected_share = (nvlink / 8) / 1000.0
        for ur in transfers:
            assert ur.active_legs[0].bandwidth_bytes_per_ms == pytest.approx(
                expected_share, rel=1e-3
            )

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
        # S3 download is now limited by the smaller of the S3 link and the
        # destination node's inet downlink.
        expected_bw = min(
            s3_spec.down_bw_bytes_per_s, tiny_hardware.spec.network_inet_down
        )
        expected_ms = 1_000_000_000 / (expected_bw / 8000.0)
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
        # S3 uploads share the node's inet uplink (and the S3 uplink), so each
        # transfer gets an equal share of the bottleneck.
        expected_bw = min(
            s3_spec.up_bw_bytes_per_s,
            tiny_hardware.spec.network_inet_up / 2,
        )
        expected_ms = 1_000_000_000 / (expected_bw / 8000.0)
        assert scheduler.next_event_ms() == pytest.approx(expected_ms, rel=1e-3)
