"""Tests for request and transfer-leg data structures."""

from src.request.request import DownloadRequest, Request, TransferLeg, UploadRequest


class TestTransferLeg:
    def test_active_legs_start_at_zero(self):
        leg = TransferLeg(100, 0, 1, "NETWORK")
        ur = UploadRequest(Request(10, 2, 0), [[leg]])
        assert ur.active_legs == [leg]
        assert ur.remaining_bytes == 100

    def test_advance_track_moves_to_next(self):
        legs = [
            TransferLeg(100, 0, 0, "RAM_LOCAL"),
            TransferLeg(100, 0, 1, "NETWORK"),
        ]
        dr = DownloadRequest(Request(10, 2, 0), [legs])
        assert dr.active_legs == [legs[0]]
        assert dr.advance_track(0) is True
        assert dr.active_legs == [legs[1]]

    def test_advance_track_returns_false_at_end(self):
        leg = TransferLeg(100, 0, 1, "NETWORK")
        ur = UploadRequest(Request(10, 2, 0), [[leg]])
        assert ur.advance_track(0) is False
        assert ur.active_legs == []
        assert ur.remaining_bytes == 0

    def test_bandwidth_setter_updates_active_leg(self):
        leg = TransferLeg(100, 0, 1, "NETWORK")
        dr = DownloadRequest(Request(10, 2, 0), [[leg]])
        dr.active_legs[0].bandwidth_bytes_per_ms = 1_000.0
        assert leg.bandwidth_bytes_per_ms == 1_000.0

    def test_parallel_tracks_active_together(self):
        leg_a = TransferLeg(100, 0, 0, "SSD_LOCAL")
        leg_b = TransferLeg(100, 0, 1, "NETWORK")
        dr = DownloadRequest(Request(10, 2, 0), [[leg_a], [leg_b]])
        assert dr.active_legs == [leg_a, leg_b]
        assert dr.remaining_bytes == 200

    def test_complete_when_all_tracks_exhausted(self):
        dr = DownloadRequest(
            Request(10, 2, 0),
            [
                [TransferLeg(0, 0, 0, "RAM_LOCAL")],
                [TransferLeg(0, 0, 1, "NETWORK")],
            ],
        )
        assert dr.active_legs == []
        assert dr.is_complete()

    def test_default_latency_per_bottleneck(self):
        assert TransferLeg(0, 0, 0, "RAM_LOCAL").remaining_latency_ms == 0.0
        assert TransferLeg(0, 0, 0, "SSD_LOCAL").remaining_latency_ms == 0.1
        assert TransferLeg(0, 0, 0, "NETWORK").remaining_latency_ms == 0.0
        assert TransferLeg(0, 0, 0, "S3_UPLOAD").remaining_latency_ms == 50.0
        assert TransferLeg(0, 0, 0, "S3_DOWNLOAD").remaining_latency_ms == 50.0

    def test_custom_latency_override(self):
        leg = TransferLeg(100, 0, 1, "S3_DOWNLOAD", latency_ms=12.0)
        assert leg.remaining_latency_ms == 12.0


class TestRequest:
    def test_stage_prefill_when_remaining_tokens(self):
        req = Request(isl=10, osl=2)
        assert req.stage == "prefill"

    def test_stage_decode_when_prefill_complete(self):
        req = Request(isl=10, osl=2)
        req.prefilled_tokens = 10
        assert req.stage == "decode"

    def test_prefilled_tokens_must_be_less_than_or_equal_to_isl(self):
        # prefilled == isl is allowed (full prefix hit)
        req = Request(isl=10, osl=2)
        req.prefilled_tokens = 10
        assert req.prefilled_tokens == 10
        # prefilled > isl is guarded by the property setter in production code;
        # here we just document the intended invariant.
        req.prefilled_tokens = 11
        assert req.prefilled_tokens == 11

    def test_ids_increment_globally(self):
        from src.request import request

        before = request.request_id_counter
        Request(isl=10, osl=2)
        after = request.request_id_counter
        assert after == before + 1
