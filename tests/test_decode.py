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
    model.kv_size_tokens = lambda tokens: tokens
    hardware = MagicMock()
    hardware.gpu_mem = 1_000_000
    hardware.gpu_bw = 1_000_000_000
    hardware.flops = 1_000_000_000_000
    instance = DecodeInstance.__new__(DecodeInstance)
    instance.instance_id = 0
    instance.node_id = 1
    instance.hardware = hardware
    instance.max_batch_size = 4
    instance.model = model
    instance.queue = []
    instance.download_queue = []
    instance.background_download_queue = []
    instance.upload_queue = []
    instance.background_upload_queue = []
    instance.cache = MagicMock(spec=Cache)
    instance.cache.upload_kv.return_value = MagicMock(
        active_legs=[MagicMock()], tracks=[[MagicMock()]]
    )
    instance.cache.download_kv.return_value = MagicMock(
        active_legs=[], tracks=[], remaining_bytes=0
    )
    instance.scheduler = MagicMock()
    instance.session = MagicMock()
    instance.current_batch = None
    instance.remaining_batch_time_ms = None
    instance.current_batch_decode_time_ms = None
    instance._kv_cache_bytes = 0
    return instance


class TestDecodeBatchLifecycle:
    def test_empty_queue_returns_infinity(self, fake_decode_instance: DecodeInstance):
        assert fake_decode_instance.time_to_next_completion() == float("inf")

    def test_frozen_batch_ignores_new_arrivals_until_token_done(
        self, fake_decode_instance: DecodeInstance
    ):
        inst = fake_decode_instance

        with patch.object(
            inst, "calculate_decode_time", side_effect=[5.0, 5.0, 5.0]
        ) as mock_calc:
            inst.add_request(Request(10, 5, 0))
            inst.add_request(Request(10, 5, 0))

            # time_to_next_completion reports the full committed-stride time.
            assert inst.time_to_next_completion() == pytest.approx(25.0)

            # A new request arrives mid-token.
            inst.queue.append((Request(10, 5, 0), -1))

            # The frozen batch should still be the first two requests.
            assert inst.current_batch is not None
            assert len(inst.current_batch) == 2
            inst.process_queue(5.0)

            # The batch is unfrozen only after its token commitment is
            # exhausted (or the batch empties), not after a single token.
            assert inst.current_batch is not None
            assert len(inst.current_batch) == 2
            # The new arrival is queued but has not joined the frozen batch.
            assert len(inst.queue) == 3
            mock_calc.assert_called()

    def test_partial_step_banks_time(self, fake_decode_instance: DecodeInstance):
        inst = fake_decode_instance

        with patch.object(inst, "calculate_decode_time", return_value=5.0):
            inst.add_request(Request(10, 5, 0))
            # time_to_next_completion reports the full committed-stride time.
            assert inst.time_to_next_completion() == pytest.approx(25.0)

            # Step of 2 ms: no token completed, 3 ms left.
            inst.process_queue(2.0)
            assert inst.remaining_batch_time_ms == pytest.approx(23.0)
            assert inst.current_batch[0].decoded_tokens == 0

            # Step of 3 ms: one token completes. The batch remains frozen
            # because the fixed token commitment has not been exhausted.
            inst.process_queue(3.0)
            assert inst.remaining_batch_time_ms is not None
            assert inst.current_batch is not None
            assert inst.queue[0][0].decoded_tokens == 0

    def test_full_token_step_advances_all_requests(
        self, fake_decode_instance: DecodeInstance
    ):
        inst = fake_decode_instance

        with patch.object(inst, "calculate_decode_time", return_value=4.0):
            inst.add_request(Request(10, 5, 0))
            inst.add_request(Request(10, 5, 0))
            inst.process_queue(4.0)

            # One token completed for all requests, but the batch stays frozen
            # for the remainder of the fixed token commitment.
            assert inst.current_batch is not None
            assert len(inst.current_batch) == 2
            assert inst.queue[0][0].decoded_tokens == 0
            assert inst.queue[1][0].decoded_tokens == 0

    def test_request_finishes_exactly_at_token_end(
        self, fake_decode_instance: DecodeInstance
    ):
        inst = fake_decode_instance

        with patch.object(inst, "calculate_decode_time", return_value=3.0):
            req = Request(10, 1, 0)
            inst.add_request(req)

            finished = inst.process_queue(3.0)
            assert inst.current_batch is not None
            assert len(inst.current_batch) == 0
            assert req.decoded_tokens == 1
            assert len(finished) == 0  # upload stays in queue, drained later
            assert len(inst.queue) == 0
            assert len(inst.upload_queue) == 1

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
            assert inst.current_batch is not None
            assert len(inst.current_batch) == 1
            assert len(inst.queue) == 1

    def test_finished_upload_drains_when_last_track_done(
        self, fake_decode_instance: DecodeInstance
    ):
        inst = fake_decode_instance
        inst.model.kv_size_per_token = 1

        # Build a concrete UploadRequest-like object that simulates an upload
        # with one eviction track and one actual upload track.
        class FakeUpload:
            def __init__(
                self, request: Request, done: bool = False, complete: bool = False
            ):
                self.request = request
                self.tracks: list[list] = [[MagicMock()], [MagicMock()]]
                self._upload_active = 1.5
                self._background_active = 3.0
                self._done = done
                self._complete = complete

            @property
            def active_legs(self) -> list:
                active: list = []
                if not self._done:
                    active.append(self.tracks[-1][0])
                if not self._complete:
                    active.append(self.tracks[0][0])
                return active

            def is_upload_done(self) -> bool:
                return self._done

            def is_complete(self) -> bool:
                return self._complete

            def upload_active_duration_ms(self) -> float:
                return self._upload_active

            def background_active_duration_ms(self) -> float:
                return self._background_active

        with patch.object(inst, "calculate_decode_time", return_value=2.0):
            req = Request(10, 1, 0)
            inst.cache.upload_kv.return_value = FakeUpload(req)
            inst.add_request(req)
            inst.process_queue(2.0)
            assert len(inst.queue) == 0
            assert len(inst.upload_queue) == 1
            assert len(inst.background_upload_queue) == 0

            # Mark only the upload leg done; request should finish, background
            # duration is not recorded yet.
            inst.upload_queue[0][0]._done = True
            finished = inst.process_queue(0.0)
            assert len(finished) == 1
            assert req.decode_upload_active_ms == pytest.approx(1.5)
            assert req.decode_upload_background_active_ms == pytest.approx(0.0)
            assert len(inst.upload_queue) == 0
            assert len(inst.background_upload_queue) == 1

            # Mark the eviction track done; background duration is now captured.
            inst.background_upload_queue[0]._complete = True
            inst.process_queue(0.0)
            assert req.decode_upload_background_active_ms == pytest.approx(3.0)
            assert len(inst.background_upload_queue) == 0


# class TestDecodeTimeRecalculation:
#     def test_recalculates_decode_time_after_each_token(
#         self, fake_decode_instance: DecodeInstance
#     ):
#         inst = fake_decode_instance

#         # Latency grows as sequences get longer.  Provide enough values for the
#         # fixed token commitment: initial batch decode time plus one recompute
#         # per token generated before the request finishes.
#         with patch.object(
#             inst, "calculate_decode_time", side_effect=[2.0, 3.0, 4.0]
#         ) as mock_calc:
#             inst.add_request(Request(10, 3, 0))

#             # time_to_next_completion reports the full committed-stride time.
#             assert inst.time_to_next_completion() == pytest.approx(6.0)
#             # A 5 ms step completes the first token (using 2 ms), banks 3 ms,
#             # and completes the second token (using 3 ms).
#             inst.process_queue(5.0)

#             assert mock_calc.call_count == 3
#             assert inst.queue[0][0].decoded_tokens == 2
