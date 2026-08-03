"""Tests for the new phase-level timing analytics."""

from unittest.mock import MagicMock, patch

import pytest

from src.cache.cache import Cache
from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.request.request import DownloadRequest, Request, TransferLeg, UploadRequest
from src.scheduler.bandwidth_scheduler import BandwidthScheduler


class FakeScheduler:
    """Minimal scheduler that owns a mutable global clock."""

    def __init__(self):
        self._time_ms = 0.0

    @property
    def time_ms(self):
        return self._time_ms

    def advance_time(self, delta_ms: float):
        self._time_ms += delta_ms

    def register(self, transfer):
        pass

    def unregister(self, transfer):
        pass


@pytest.fixture
def fake_prefill_instance():
    model = MagicMock()
    model.name = "test-model"
    hardware = MagicMock()
    hardware.gpu_mem = 1_000_000
    hardware.gpu_bw = 1_000_000_000
    hardware.flops = 1_000_000_000_000
    instance = PrefillInstance.__new__(PrefillInstance)
    instance.node_id = 0
    instance.hardware = hardware
    instance.queue = []
    instance.upload_queue = []
    instance.background_upload_queue = []
    instance.download_queue = []
    instance.cache = MagicMock(spec=Cache)
    instance.cache.download_kv.return_value = MagicMock(
        active_legs=[], tracks=[], remaining_bytes=0
    )
    instance.cache.upload_kv.return_value = MagicMock(
        active_legs=[MagicMock()], tracks=[[MagicMock()]]
    )
    instance.scheduler = FakeScheduler()
    instance.session = MagicMock()
    instance._kv_cache_bytes = 0
    instance.max_batch_size = 10
    return instance


@pytest.fixture
def fake_decode_instance():
    model = MagicMock()
    model.name = "test-model"
    model.kv_size_per_token = 1
    hardware = MagicMock()
    hardware.gpu_mem = 1_000_000
    hardware.gpu_bw = 1_000_000_000
    hardware.flops = 1_000_000_000_000
    instance = DecodeInstance.__new__(DecodeInstance)
    instance.node_id = 1
    instance.hardware = hardware
    instance.max_batch_size = 4
    instance.model = model
    instance.queue = []
    instance.download_queue = []
    instance.upload_queue = []
    instance.background_upload_queue = []
    instance.cache = MagicMock(spec=Cache)
    instance.cache.upload_kv.return_value = MagicMock(
        active_legs=[MagicMock()], tracks=[[MagicMock()]]
    )
    instance.cache.download_kv.return_value = MagicMock(
        active_legs=[], tracks=[], remaining_bytes=0
    )
    instance.scheduler = FakeScheduler()
    instance.session = MagicMock()
    instance.current_batch = None
    instance.remaining_batch_time_ms = None
    instance.current_batch_decode_time_ms = None
    instance._kv_cache_bytes = 0
    return instance


class TestTransferActiveDuration:
    def test_active_duration_sums_longest_parallel_track(self):
        req = Request(10, 5, 0)
        leg1 = TransferLeg(100, 0, 0, "RAM_LOCAL")
        leg1.processed_time_ms = 3.0
        leg2 = TransferLeg(100, 0, 0, "RAM_LOCAL")
        leg2.processed_time_ms = 7.0
        ur = UploadRequest(req, [[leg1], [leg2]])
        assert ur.active_transfer_duration_ms == pytest.approx(7.0)


class TestPrefillTiming:
    def test_prefill_wait_accumulates_for_non_head_requests(
        self, fake_prefill_instance: PrefillInstance
    ):
        inst = fake_prefill_instance
        scheduler = inst.scheduler

        with patch.object(inst, "calculate_prefill_time", return_value=10.0):
            req_a = Request(10, 5, 0)
            req_b = Request(10, 5, 0)
            inst.add_request(req_a)
            inst.add_request(req_b)

            # Both are in queue; step 3 ms with req_a at the head.
            scheduler.advance_time(3.0)
            inst.process_queue(3.0)
            assert req_a.prefill_start_ms == pytest.approx(0.0)
            assert req_a.prefill_end_ms is None
            assert req_b.prefill_start_ms is None
            assert req_b.prefill_queue_start_ms == pytest.approx(0.0)

            # Finish req_a and continue with req_b.
            scheduler.advance_time(7.0)
            inst.process_queue(7.0)
            assert req_a.prefill_end_ms - req_a.prefill_start_ms == pytest.approx(10.0)
            assert req_b.prefill_start_ms == pytest.approx(10.0)
            assert req_b.prefill_end_ms is None

            # Run req_b to completion.
            scheduler.advance_time(10.0)
            inst.process_queue(10.0)
            assert req_b.prefill_end_ms == pytest.approx(20.0)


class TestDecodeTiming:
    def test_decode_wait_accumulates_for_requests_not_in_batch(
        self, fake_decode_instance: DecodeInstance
    ):
        inst = fake_decode_instance
        scheduler = inst.scheduler
        inst.max_batch_size = 2  # keep room small so new arrivals wait

        with patch.object(inst, "calculate_decode_time", return_value=4.0):
            req_a = Request(10, 2, 0)
            req_b = Request(10, 2, 0)
            inst.add_request(req_a)
            inst.add_request(req_b)

            # Freeze the batch at t=0, just like the main simulation loop does
            # before advancing time.
            inst.time_to_next_completion()
            assert req_a.decode_start_ms == pytest.approx(0.0)
            assert req_b.decode_start_ms == pytest.approx(0.0)
            assert req_a.decode_end_ms is None
            assert req_b.decode_end_ms is None

            # First full token step: batch is [req_a, req_b]; each completes
            # one decode token (osl=2 means two tokens total).
            scheduler.advance_time(4.0)
            inst.process_queue(4.0)
            assert req_a.decode_end_ms is None
            assert req_b.decode_end_ms is None

            # Add a third request; with max_batch_size=2 it will wait while the
            # first two finish their last token.
            req_c = Request(10, 2, 0)
            inst.add_request(req_c)
            scheduler.advance_time(4.0)
            inst.process_queue(4.0)
            assert req_a.decode_end_ms == pytest.approx(8.0)
            assert req_b.decode_end_ms == pytest.approx(8.0)
            assert req_c.decode_start_ms is None
            assert req_c.decode_queue_start_ms == pytest.approx(4.0)

            # Re-form the next batch at t=8 (mimics the main simulation loop).
            inst.time_to_next_completion()
            assert req_c.decode_start_ms == pytest.approx(8.0)

            # req_c needs two 4 ms tokens to finish (osl=2).
            scheduler.advance_time(4.0)
            inst.process_queue(4.0)
            assert req_c.decode_end_ms is None

            scheduler.advance_time(4.0)
            inst.process_queue(4.0)
            assert req_c.decode_end_ms == pytest.approx(16.0)


class TestSchedulerCreditsProcessedTime:
    def test_processed_time_tracked_per_leg(self):
        from src.hardware.hardware import (
            GPUHardwareSpec,
            Hardware,
            HardwareSpec,
        )
        from src.node.node import Node

        spec = HardwareSpec(
            gpu_hardware=GPUHardwareSpec(flops=1, gpu_mem=1, gpu_bw=1_000_000_000),
            num_gpus=1,
            nvme_mem=1,
            nvme_bw=1,
            network_inet_up=1,
            network_inet_down=1,
            network_inter_node_up=1,
            network_inter_node_down=1,
            cpu_ram=1,
            dph_base=1.0,
            pcie_bw=1_000_000.0,
        )
        node = Node.__new__(Node)
        node.id = 0
        node.hardware = Hardware(name="test", spec=spec)
        node.prefill_instances = []
        node.decode_instances = []

        scheduler = BandwidthScheduler([node])
        req = Request(10, 5, 0)
        leg = TransferLeg(100, 0, 0, "RAM_LOCAL")
        dr = DownloadRequest(req, [[leg]])
        scheduler.register(dr)

        scheduler.advance_time(0.05)
        assert leg.processed_time_ms == pytest.approx(0.05)

        scheduler.advance_time(0.05)
        assert leg.processed_time_ms == pytest.approx(0.1)
        assert dr.active_transfer_duration_ms == pytest.approx(0.1)
