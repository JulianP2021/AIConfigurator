"""Tests for request and transfer-leg data structures."""

import pytest

from src.request.request import DownloadRequest, Request, TransferLeg, UploadRequest


class TestTransferLeg:
    def test_active_leg_starts_at_zero(self):
        leg = TransferLeg(100, 0, 1, "NETWORK")
        ur = UploadRequest(Request(10, 2, 0), [leg])
        assert ur.active_leg is leg
        assert ur.remaining_bytes == 100
        assert ur.source_node_id == 0
        assert ur.dest_node_id == 1
        assert ur.bottleneck == "NETWORK"

    def test_advance_leg_moves_to_next(self):
        legs = [
            TransferLeg(100, 0, 0, "RAM_LOCAL"),
            TransferLeg(100, 0, 1, "NETWORK"),
        ]
        dr = DownloadRequest(Request(10, 2, 0), legs)
        assert dr.active_leg == legs[0]
        assert dr.advance_leg() is True
        assert dr.active_leg == legs[1]

    def test_advance_leg_returns_false_at_end(self):
        leg = TransferLeg(100, 0, 1, "NETWORK")
        ur = UploadRequest(Request(10, 2, 0), [leg])
        assert ur.advance_leg() is False
        assert ur.active_leg is None
        assert ur.remaining_bytes == 0

    def test_bandwidth_setter_updates_active_leg(self):
        leg = TransferLeg(100, 0, 1, "NETWORK")
        dr = DownloadRequest(Request(10, 2, 0), [leg])
        dr.bandwidth_bytes_per_ms = 1_000.0
        assert leg.bandwidth_bytes_per_ms == 1_000.0


class TestRequest:
    def test_stage_prefill_when_remaining_tokens(self):
        req = Request(isl=10, osl=2, cached=0)
        assert req.stage == "prefill"

    def test_stage_decode_when_prefill_complete(self):
        req = Request(isl=10, osl=2, cached=0)
        req.prefilled_tokens = 10
        assert req.stage == "decode"

    def test_prefilled_tokens_must_be_less_than_isl(self):
        with pytest.raises(AssertionError):
            Request(isl=10, osl=2, cached=10)

    def test_ids_increment_globally(self):
        from src.request import request

        before = request.request_id_counter
        Request(isl=10, osl=2, cached=0)
        after = request.request_id_counter
        assert after == before + 1
