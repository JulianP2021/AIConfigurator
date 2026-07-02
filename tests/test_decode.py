"""Tests for DecodeInstance batch and partial-decode behaviour."""

from unittest.mock import MagicMock, patch

import pytest

from src.cache.cache import Cache
from src.instances.decode import DecodeInstance
from src.request.request import Request


@pytest.fixture
def fake_decode_instance() -> DecodeInstance:
    """A DecodeInstance whose compute latency is controlled by a mock session."""
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
    instance.cache = MagicMock(spec=Cache)
    instance.cache.upload_kv.return_value = MagicMock(active_legs=[])
    instance.cache.download_kv.return_value = MagicMock(
        active_legs=[], tracks=[], remaining_bytes=0
    )
    instance.scheduler = MagicMock()
    instance.session = MagicMock()
    instance.current_batch = None
    instance.remaining_batch_time_ms = None
    instance.current_batch_decode_time_ms = None
    return instance


class TestDecodeBatchLifecycle:
    def test_empty_queue_returns_infinity(self, fake_decode_instance: DecodeInstance):
        assert fake_decode_instance.time_to_next_completion() == float("inf")

    def test_frozen_batch_ignores_new_arrivals_until_token_done(
        self, fake_decode_instance: DecodeInstance
    ):
        inst = fake_decode_instance

        with patch.object(
            inst, "calculate_decode_time", side_effect=[5.0, 5.0]
        ) as mock_calc:
            inst.add_request(Request(10, 5, 0))
            inst.add_request(Request(10, 5, 0))

            # time_to_next_completion now returns 5 * 5 = 25 ms (lowball until
            # the first request with 5 remaining tokens finishes).
            assert inst.time_to_next_completion() == pytest.approx(25.0)

            # A new request arrives mid-token.
            inst.queue.append((Request(10, 5, 0), -1))

            # The batch should still be the first two requests.
            assert inst.current_batch is not None
            assert len(inst.current_batch) == 2
            inst.process_queue(5.0)

            # After the token, the new request should join the next batch.
            assert inst.current_batch is None
            assert len(inst.queue) == 3
            mock_calc.assert_called()

    def test_partial_step_banks_time(self, fake_decode_instance: DecodeInstance):
        inst = fake_decode_instance

        with patch.object(inst, "calculate_decode_time", return_value=5.0):
            inst.add_request(Request(10, 5, 0))
            # time_to_next_completion lowballs: 5 ms/token * 5 tokens left.
            assert inst.time_to_next_completion() == pytest.approx(25.0)

            # Step of 2 ms: no token completed, 3 ms left.
            inst.process_queue(2.0)
            assert inst.remaining_batch_time_ms == pytest.approx(3.0)
            assert inst.current_batch[0].decoded_tokens == 0

            # Step of 3 ms: token completes.
            inst.process_queue(3.0)
            assert inst.remaining_batch_time_ms is None
            assert inst.queue[0][0].decoded_tokens == 1

    def test_full_token_step_advances_all_requests(
        self, fake_decode_instance: DecodeInstance
    ):
        inst = fake_decode_instance

        with patch.object(inst, "calculate_decode_time", return_value=4.0):
            inst.add_request(Request(10, 5, 0))
            inst.add_request(Request(10, 5, 0))
            inst.process_queue(4.0)

            assert inst.current_batch is None
            assert inst.queue[0][0].decoded_tokens == 1
            assert inst.queue[1][0].decoded_tokens == 1

    def test_request_finishes_exactly_at_token_end(
        self, fake_decode_instance: DecodeInstance
    ):
        inst = fake_decode_instance

        with patch.object(inst, "calculate_decode_time", return_value=3.0):
            req = Request(10, 1, 0)
            inst.add_request(req)

            finished = inst.process_queue(3.0)
            assert inst.current_batch is None
            assert req.decoded_tokens == 1
            assert len(finished) == 0  # upload has no active leg in this mock
            assert len(inst.queue) == 0

    def test_batch_continues_with_remaining_requests(
        self, fake_decode_instance: DecodeInstance
    ):
        inst = fake_decode_instance

        with patch.object(inst, "calculate_decode_time", return_value=2.0):
            short = Request(10, 1, 0)
            long = Request(10, 3, 0)
            inst.add_request(short)
            inst.add_request(long)

            inst.process_queue(2.0)
            # short finished, long decoded one token.
            assert short.decoded_tokens == 1
            assert long.decoded_tokens == 1
            assert inst.current_batch is None
            assert len(inst.queue) == 1


class TestDecodeTimeRecalculation:
    def test_recalculates_decode_time_after_each_token(
        self, fake_decode_instance: DecodeInstance
    ):
        inst = fake_decode_instance

        # Latency grows as sequences get longer.
        with patch.object(
            inst, "calculate_decode_time", side_effect=[2.0, 3.0, 4.0]
        ) as mock_calc:
            inst.add_request(Request(10, 3, 0))

            # time_to_next_completion lowballs: 2 ms/token * 3 tokens left.
            assert inst.time_to_next_completion() == pytest.approx(6.0)
            inst.process_queue(5.0)

            # Should have recalculated twice (first token done, then second
            # token's budget loaded).
            assert mock_calc.call_count == 3
            assert inst.queue[0][0].decoded_tokens == 2
