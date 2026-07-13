"""Tests for Dynamo-style cost-based routing."""

from unittest.mock import MagicMock

from src.cache.cache import Cache, CacheItem, CacheLayer
from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.request.request import Request
from src.router.router import Router, RouterCostConfig


def _make_prefill_instance(node_id: int) -> PrefillInstance:
    inst = PrefillInstance.__new__(PrefillInstance)
    inst.node_id = node_id
    inst.queue = []
    inst.download_queue = []
    inst.upload_queue = []
    inst.background_upload_queue = []
    inst.hardware = MagicMock()
    inst.model = MagicMock()
    inst.cache = None
    inst.scheduler = None
    return inst


def _make_decode_instance(node_id: int) -> DecodeInstance:
    inst = DecodeInstance.__new__(DecodeInstance)
    inst.node_id = node_id
    inst.queue = []
    inst.download_queue = []
    inst.upload_queue = []
    inst.current_batch = None
    inst.remaining_batch_time_ms = None
    inst.current_batch_decode_time_ms = None
    inst.hardware = MagicMock()
    inst.model = MagicMock()
    inst.model.kv_size_per_token = 1
    inst.max_batch_size = 4
    inst.cache = None
    inst.scheduler = None
    return inst


def _make_cache() -> Cache:
    """Return a tiny cache with two nodes (0 and 1)."""
    model = MagicMock()
    model.kv_size_per_token = 1
    hw0 = MagicMock()
    hw0.spec.cpu_ram = 1_000_000
    hw0.spec.nvme_mem = 1_000_000
    hw1 = MagicMock()
    hw1.spec.cpu_ram = 1_000_000
    hw1.spec.nvme_mem = 1_000_000
    return Cache(
        layers={},
        node_hardware={0: hw0, 1: hw1},
        model=model,
        ram_usage_fraction=0.8,
        ssd_usage_fraction=0.8,
    )


class TestRouterCostFunction:
    def test_prefill_prefers_node_with_cached_prefix(self):
        cache = _make_cache()
        cache.layers = {
            0: [CacheLayer(0, "RAM", {(1, 0): [CacheItem((1, 0), 0, 500)]})],
            1: [CacheLayer(1, "RAM", {})],
        }

        prefill_0 = _make_prefill_instance(0)
        prefill_1 = _make_prefill_instance(1)
        decode_0 = _make_decode_instance(0)
        decode_1 = _make_decode_instance(1)

        router = Router(
            queue=[],
            prefill_instances=[prefill_0, prefill_1],
            decode_instances=[decode_0, decode_1],
            cache=cache,
            cost_config=RouterCostConfig(device_credit=1.0),
        )

        req = Request(isl=1000, osl=100, user_id=1, session_id=0)
        chosen = router._choose_prefill_instance(req)
        assert chosen.node_id == 0

    def test_prefill_load_balances_when_no_cache(self):
        prefill_0 = _make_prefill_instance(0)
        prefill_1 = _make_prefill_instance(1)
        decode_0 = _make_decode_instance(0)
        decode_1 = _make_decode_instance(1)

        # Put one active prefill request on node 0.
        prefill_0.queue.append((Request(isl=500, osl=10), -1))

        router = Router(
            queue=[],
            prefill_instances=[prefill_0, prefill_1],
            decode_instances=[decode_0, decode_1],
            cache=None,
            cost_config=RouterCostConfig(prefill_load_scale=1.0),
        )

        req = Request(isl=1000, osl=100)
        chosen = router._choose_prefill_instance(req)
        assert chosen.node_id == 1

    def test_prefill_picks_least_loaded_instance_by_tokens(self):
        # Two instances on the same node: same queue depth but very different
        # token loads. The router should pick the one with fewer tokens.
        prefill_a = _make_prefill_instance(0)
        prefill_b = _make_prefill_instance(0)
        decode_0 = _make_decode_instance(0)

        # Both queues have one request, but instance a has far more tokens.
        prefill_a.queue.append((Request(isl=5000, osl=10), -1))
        prefill_b.queue.append((Request(isl=10, osl=10), -1))

        router = Router(
            queue=[],
            prefill_instances=[prefill_a, prefill_b],
            decode_instances=[decode_0],
            cache=None,
            cost_config=RouterCostConfig(prefill_load_scale=1.0),
        )

        req = Request(isl=100, osl=100)
        chosen = router._choose_prefill_instance(req)
        assert chosen is prefill_b

    def test_decode_prefers_node_with_full_kv(self):
        cache = _make_cache()
        cache.layers = {
            0: [CacheLayer(0, "RAM", {(1, 0): [CacheItem((1, 0), 0, 1000)]})],
            1: [CacheLayer(1, "RAM", {})],
        }

        prefill_0 = _make_prefill_instance(0)
        prefill_1 = _make_prefill_instance(1)
        decode_0 = _make_decode_instance(0)
        decode_1 = _make_decode_instance(1)

        router = Router(
            queue=[],
            prefill_instances=[prefill_0, prefill_1],
            decode_instances=[decode_0, decode_1],
            cache=cache,
            cost_config=RouterCostConfig(device_credit=1.0),
        )

        req = Request(isl=1000, osl=100, user_id=1, session_id=0)
        chosen = router._choose_decode_instance(req)
        assert chosen.node_id == 0

    def test_decode_prefers_local_over_remote(self):
        cache = _make_cache()
        cache.layers = {
            0: [CacheLayer(0, "RAM", {})],
            1: [CacheLayer(1, "RAM", {(1, 0): [CacheItem((1, 0), 0, 1000)]})],
        }

        prefill_0 = _make_prefill_instance(0)
        prefill_1 = _make_prefill_instance(1)
        decode_0 = _make_decode_instance(0)
        decode_1 = _make_decode_instance(1)

        router = Router(
            queue=[],
            prefill_instances=[prefill_0, prefill_1],
            decode_instances=[decode_0, decode_1],
            cache=cache,
            cost_config=RouterCostConfig(device_credit=1.0, remote_ram_credit=0.0),
        )

        req = Request(isl=1000, osl=100, user_id=1, session_id=0)
        # Both nodes can satisfy the prefix, but node 1 holds it locally so it
        # should win despite the remote option on node 0.
        chosen = router._choose_decode_instance(req)
        assert chosen.node_id == 1

    def test_decode_load_balances_when_no_cache(self):
        prefill_0 = _make_prefill_instance(0)
        prefill_1 = _make_prefill_instance(1)
        decode_0 = _make_decode_instance(0)
        decode_1 = _make_decode_instance(1)

        # Node 0 is already decoding one request.
        decode_0.queue.append((Request(isl=1000, osl=100), -1))

        router = Router(
            queue=[],
            prefill_instances=[prefill_0, prefill_1],
            decode_instances=[decode_0, decode_1],
            cache=None,
        )

        req = Request(isl=1000, osl=100)
        chosen = router._choose_decode_instance(req)
        assert chosen.node_id == 1

    def test_route_requests_routes_by_stage(self):
        prefill_0 = _make_prefill_instance(0)
        prefill_1 = _make_prefill_instance(1)
        decode_0 = _make_decode_instance(0)
        decode_1 = _make_decode_instance(1)

        # Patch add_request to record routed requests without hitting instance logic.
        prefill_calls: list[tuple[int, Request]] = []
        decode_calls: list[tuple[int, Request]] = []

        def make_prefill_add(inst: PrefillInstance):
            def _add(req: Request):
                prefill_calls.append((inst.node_id, req))

            return _add

        def make_decode_add(inst: DecodeInstance):
            def _add(req: Request):
                decode_calls.append((inst.node_id, req))

            return _add

        for inst in (prefill_0, prefill_1):
            inst.add_request = make_prefill_add(inst)  # type: ignore[method-assign]
        for inst in (decode_0, decode_1):
            inst.add_request = make_decode_add(inst)  # type: ignore[method-assign]

        router = Router(
            queue=[],
            prefill_instances=[prefill_0, prefill_1],
            decode_instances=[decode_0, decode_1],
            cache=None,
        )

        prefill_req = Request(isl=1000, osl=100)
        decode_req = Request(isl=1000, osl=100)
        decode_req.prefilled_tokens = 1000  # forces stage == "decode"
        router.queue = [prefill_req, decode_req]
        router.route_requests()

        assert len(prefill_calls) == 1
        assert len(decode_calls) == 1
        assert prefill_calls[0][1] is prefill_req
        assert decode_calls[0][1] is decode_req
