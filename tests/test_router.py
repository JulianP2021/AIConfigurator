"""Tests for Dynamo-style cost-based routing."""

from unittest.mock import MagicMock

from src.cache.cache import Cache, CacheItem, CacheLayer
from src.instances.decode import DecodeInstance
from src.instances.prefill import PrefillInstance
from src.model.model import Model
from src.request.request import Request
from src.router.router import Router, RouterCostConfig


def _make_prefill_instance(node_id: int) -> PrefillInstance:
    from src.hardware.hardware import GPUHardwareSpec, Hardware, HardwareSpec

    inst = PrefillInstance.__new__(PrefillInstance)
    inst.node_id = node_id
    inst.queue = []
    inst.download_queue = []
    inst.upload_queue = []
    inst.background_upload_queue = []
    inst.hardware = Hardware(
        name="test",
        spec=HardwareSpec(
            gpu_hardware=GPUHardwareSpec(
                flops=1e15, gpu_mem=1_000_000_000, gpu_bw=1e12
            ),
            num_gpus=1,
            nvme_mem=1_000_000_000,
            nvme_bw=1_000_000_000.0,
            network_inet_up=1_000_000_000.0,
            network_inet_down=1_000_000_000.0,
            network_inter_node_up=1_000_000_000.0,
            network_inter_node_down=1_000_000_000.0,
            cpu_cores=1,
            cpu_cores_effective=1.0,
            cpu_ghz=1.0,
            cpu_name="test",
            cpu_ram=1_000_000_000,
            disk_name="test",
            dlperf=1.0,
            dlperf_per_dphtotal=1.0,
            dph_base=1.0,
            geolocation="test",
            gpu_display_active=False,
            gpu_frac=1.0,
            gpu_lanes=1,
            gpu_max_power=1.0,
            gpu_max_temp=1.0,
            has_avx=1,
            host_id=node_id,
            inet_down_cost=0.0,
            inet_up_cost=0.0,
            mobo_name="test",
            os_version="test",
            pci_gen=4.0,
            pcie_bw=1_000_000_000.0,
            network_bw=1_000_000_000.0,
            reliability=1.0,
            reliability_mult=1.0,
            score=1.0,
            storage_cost=0.0,
            storage_total_cost=0.0,
            verification="test",
            nvlink_bw=0.0,
        ),
    )
    inst.model = _make_test_model()
    inst.cache = None
    inst.scheduler = None
    return inst


def _make_test_model() -> Model:
    """Return a tiny model that the compute estimators can use."""
    model = Model.__new__(Model)
    model.name = "test"
    model.config = {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "dtype": "bfloat16",
        "head_size": 128,
    }
    model.dtype_size = 2.0
    model.cost_constants = {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32000,
        "d_kv": 128,
        "dtype_size": 2.0,
        "output_flops": 1,
        "matrices": 1,
        "embedding_memory": 1,
    }
    return model


def _make_decode_instance(node_id: int) -> DecodeInstance:
    from src.hardware.hardware import GPUHardwareSpec, Hardware, HardwareSpec

    inst = DecodeInstance.__new__(DecodeInstance)
    inst.node_id = node_id
    inst.queue = []
    inst.download_queue = []
    inst.upload_queue = []
    inst.current_batch = None
    inst.remaining_batch_time_ms = None
    inst.current_batch_decode_time_ms = None
    inst.hardware = Hardware(
        name="test",
        spec=HardwareSpec(
            gpu_hardware=GPUHardwareSpec(
                flops=1e15, gpu_mem=1_000_000_000, gpu_bw=1e12
            ),
            num_gpus=1,
            nvme_mem=1_000_000_000,
            nvme_bw=1_000_000_000.0,
            network_inet_up=1_000_000_000.0,
            network_inet_down=1_000_000_000.0,
            network_inter_node_up=1_000_000_000.0,
            network_inter_node_down=1_000_000_000.0,
            cpu_cores=1,
            cpu_cores_effective=1.0,
            cpu_ghz=1.0,
            cpu_name="test",
            cpu_ram=1_000_000_000,
            disk_name="test",
            dlperf=1.0,
            dlperf_per_dphtotal=1.0,
            dph_base=1.0,
            geolocation="test",
            gpu_display_active=False,
            gpu_frac=1.0,
            gpu_lanes=1,
            gpu_max_power=1.0,
            gpu_max_temp=1.0,
            has_avx=1,
            host_id=node_id,
            inet_down_cost=0.0,
            inet_up_cost=0.0,
            mobo_name="test",
            os_version="test",
            pci_gen=4.0,
            pcie_bw=1_000_000_000.0,
            network_bw=1_000_000_000.0,
            reliability=1.0,
            reliability_mult=1.0,
            score=1.0,
            storage_cost=0.0,
            storage_total_cost=0.0,
            verification="test",
            nvlink_bw=0.0,
        ),
    )
    inst.model = _make_test_model()
    inst.max_batch_size = 4
    inst.cache = None
    inst.scheduler = None
    return inst


def _make_cache() -> Cache:
    """Return a tiny cache with two nodes (0 and 1)."""
    model = MagicMock()
    model.kv_size_per_token = 1
    model.config = {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "dtype": "bfloat16",
    }
    model.dtype_size = 2.0
    model.cost_constants = {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32000,
        "d_kv": 128,
        "dtype_size": 2.0,
        "output_flops": 1,
        "matrices": 1,
        "embedding_memory": 1,
    }
    hw0 = MagicMock()
    hw0.spec.cpu_ram = 1_000_000
    hw0.spec.nvme_mem = 1_000_000
    hw0.spec.nvlink_bw = 0.0
    hw0.spec.pcie_bw = 1_000_000_000.0
    hw0.spec.nvme_bw = 1_000_000_000.0
    hw0.spec.network_inter_node_up = 1_000_000_000.0
    hw0.spec.network_inter_node_down = 1_000_000_000.0
    hw0.spec.num_gpus = 1
    hw0.spec.gpu_hardware.flops = 1e15
    hw0.spec.gpu_hardware.gpu_bw = 1e12
    hw1 = MagicMock()
    hw1.spec.cpu_ram = 1_000_000
    hw1.spec.nvme_mem = 1_000_000
    hw1.spec.nvlink_bw = 0.0
    hw1.spec.pcie_bw = 1_000_000_000.0
    hw1.spec.nvme_bw = 1_000_000_000.0
    hw1.spec.network_inter_node_up = 1_000_000_000.0
    hw1.spec.network_inter_node_down = 1_000_000_000.0
    hw1.spec.num_gpus = 1
    hw1.spec.gpu_hardware.flops = 1e15
    hw1.spec.gpu_hardware.gpu_bw = 1e12
    return Cache(
        layers={},
        node_hardware={0: hw0, 1: hw1},
        model=model,
        ram_usage_fraction=0.8,
        ssd_usage_fraction=0.8,
    )


class TestRouterCostFunction:
    def prefer_cache_over_tokens(self):
        cache = _make_cache()
        item_0_1000 = CacheItem((1, 0), 0, 1000)
        layer_0 = CacheLayer(0, "RAM")
        layer_0._add_item(item_0_1000)
        cache.layers = {
            0: [layer_0],
            1: [CacheLayer(1, "RAM")],
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
            cost_config=RouterCostConfig(active_work_scale=0.001),
        )

        # Put one active prefill request on node 0.
        prefill_0.queue.append((Request(isl=50000, osl=10), -1))

        req = Request(isl=1000, osl=100, user_id=1, session_id=0)
        chosen = router._choose_decode_instance(req)
        assert chosen.node_id == 0

    def test_prefill_prefers_node_with_cached_prefix(self):
        cache = _make_cache()
        item_0_500 = CacheItem((1, 0), 0, 500)
        layer_0 = CacheLayer(0, "RAM")
        layer_0._add_item(item_0_500)
        cache.layers = {
            0: [layer_0],
            1: [CacheLayer(1, "RAM")],
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
        item_0_1000 = CacheItem((1, 0), 0, 1000)
        layer_0 = CacheLayer(0, "RAM")
        layer_0._add_item(item_0_1000)
        cache.layers = {
            0: [layer_0],
            1: [CacheLayer(1, "RAM")],
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
        item_0_1000 = CacheItem((1, 0), 0, 1000)
        layer_1 = CacheLayer(1, "RAM")
        layer_1._add_item(item_0_1000)
        cache.layers = {
            0: [CacheLayer(0, "RAM", {})],
            1: [layer_1],
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


class TestRouterTieBreaking:
    def test_prefill_tie_breaking_deterministic_with_seed(self):
        """When two prefill nodes have identical cost, the chosen node is a
        deterministic function of the supplied seed.
        """
        prefill_0 = _make_prefill_instance(0)
        prefill_1 = _make_prefill_instance(1)
        decode_0 = _make_decode_instance(0)
        decode_1 = _make_decode_instance(1)

        def sequence(seed: int | None) -> list[int]:
            router = Router(
                queue=[],
                prefill_instances=[prefill_0, prefill_1],
                decode_instances=[decode_0, decode_1],
                cache=None,
                random_seed=seed,
            )
            active_prefill = {0: 100.0, 1: 100.0}
            active_decode = {0: 100.0, 1: 100.0}
            req = Request(isl=1000, osl=100)
            return [
                router._choose_prefill_instance(
                    req, active_prefill, active_decode
                ).node_id
                for _ in range(20)
            ]

        assert sequence(42) == sequence(42)
        assert sequence(0) == sequence(0)
        # Different seeds should usually produce different sequences.
        assert sequence(42) != sequence(43)

    def test_decode_tie_breaking_deterministic_with_seed(self):
        """When two decode nodes have identical cost, the chosen node is a
        deterministic function of the supplied seed.
        """
        prefill_0 = _make_prefill_instance(0)
        prefill_1 = _make_prefill_instance(1)
        decode_0 = _make_decode_instance(0)
        decode_1 = _make_decode_instance(1)

        def sequence(seed: int | None) -> list[int]:
            router = Router(
                queue=[],
                prefill_instances=[prefill_0, prefill_1],
                decode_instances=[decode_0, decode_1],
                cache=None,
                random_seed=seed,
            )
            active_prefill = {0: 100.0, 1: 100.0}
            active_decode = {0: 100.0, 1: 100.0}
            req = Request(isl=1000, osl=100)
            req.prefilled_tokens = 1000
            return [
                router._choose_decode_instance(
                    req, active_prefill, active_decode
                ).node_id
                for _ in range(20)
            ]

        assert sequence(42) == sequence(42)
        assert sequence(42) != sequence(43)

    def test_route_requests_prefill_tie_breaking_is_seeded(self):
        """At the public route_requests level, prefill tie-breaking follows
        the configured seed.
        """

        def sequence(seed: int) -> list[int]:
            prefill_0 = _make_prefill_instance(0)
            prefill_1 = _make_prefill_instance(1)
            decode_0 = _make_decode_instance(0)
            decode_1 = _make_decode_instance(1)

            routed: list[int] = []

            def make_add(inst):
                def _add(_req: Request):
                    routed.append(inst.node_id)

                return _add

            for inst in (prefill_0, prefill_1):
                inst.add_request = make_add(inst)  # type: ignore[method-assign]
            for inst in (decode_0, decode_1):
                inst.add_request = make_add(inst)  # type: ignore[method-assign]

            router = Router(
                queue=[Request(isl=1000, osl=100) for _ in range(10)],
                prefill_instances=[prefill_0, prefill_1],
                decode_instances=[decode_0, decode_1],
                cache=None,
                random_seed=seed,
            )
            router.route_requests()
            return routed

        assert sequence(42) == sequence(42)
        assert sequence(42) != sequence(43)

    def test_route_requests_decode_tie_breaking_is_seeded(self):
        """At the public route_requests level, decode tie-breaking follows
        the configured seed.
        """

        def sequence(seed: int) -> list[int]:
            prefill_0 = _make_prefill_instance(0)
            prefill_1 = _make_prefill_instance(1)
            decode_0 = _make_decode_instance(0)
            decode_1 = _make_decode_instance(1)

            routed: list[int] = []

            def make_add(inst):
                def _add(_req: Request):
                    routed.append(inst.node_id)

                return _add

            for inst in (prefill_0, prefill_1):
                inst.add_request = make_add(inst)  # type: ignore[method-assign]
            for inst in (decode_0, decode_1):
                inst.add_request = make_add(inst)  # type: ignore[method-assign]

            requests: list[Request] = []
            for _ in range(10):
                req = Request(isl=1000, osl=100)
                req.prefilled_tokens = 1000
                requests.append(req)

            router = Router(
                queue=requests,
                prefill_instances=[prefill_0, prefill_1],
                decode_instances=[decode_0, decode_1],
                cache=None,
                random_seed=seed,
            )
            router.route_requests()
            return routed

        assert sequence(42) == sequence(42)
        assert sequence(42) != sequence(43)


class TestRouterRamTieBreaking:
    def test_tie_break_prefers_node_with_more_free_ram(self):
        """When two nodes have identical routing cost, the one with the lower
        RAM fill factor is chosen.
        """
        cache = _make_cache()
        # Fill node 0 RAM to 50% and leave node 1 RAM empty.
        cache.ram_usage_bytes[0] = cache.ram_capacity_bytes[0] // 2
        cache.ram_usage_bytes[1] = 0

        prefill_0 = _make_prefill_instance(0)
        prefill_1 = _make_prefill_instance(1)
        decode_0 = _make_decode_instance(0)
        decode_1 = _make_decode_instance(1)

        router = Router(
            queue=[],
            prefill_instances=[prefill_0, prefill_1],
            decode_instances=[decode_0, decode_1],
            cache=cache,
            cost_config=RouterCostConfig(),
        )

        # Identical cost on both nodes: active prefill/decode the same, no cache.
        active_prefill = {0: 100.0, 1: 100.0}
        active_decode = {0: 100.0, 1: 100.0}
        req = Request(isl=1000, osl=100)
        chosen_prefill = router._choose_prefill_instance(
            req, active_prefill, active_decode
        )
        assert chosen_prefill.node_id == 1

        decode_req = Request(isl=1000, osl=100)
        decode_req.prefilled_tokens = 1000
        chosen_decode = router._choose_decode_instance(
            decode_req, active_prefill, active_decode
        )
        assert chosen_decode.node_id == 1

    def test_ram_tie_break_falls_back_to_random_when_fill_equal(self):
        """When RAM fill is also tied, the router falls back to the seeded
        random choice.
        """
        cache = _make_cache()
        cache.ram_usage_bytes[0] = cache.ram_capacity_bytes[0] // 4
        cache.ram_usage_bytes[1] = cache.ram_capacity_bytes[1] // 4

        prefill_0 = _make_prefill_instance(0)
        prefill_1 = _make_prefill_instance(1)
        decode_0 = _make_decode_instance(0)
        decode_1 = _make_decode_instance(1)

        def sequence(seed: int | None) -> list[int]:
            router = Router(
                queue=[],
                prefill_instances=[prefill_0, prefill_1],
                decode_instances=[decode_0, decode_1],
                cache=cache,
                random_seed=seed,
            )
            active_prefill = {0: 100.0, 1: 100.0}
            active_decode = {0: 100.0, 1: 100.0}
            req = Request(isl=1000, osl=100)
            return [
                router._choose_prefill_instance(
                    req, active_prefill, active_decode
                ).node_id
                for _ in range(20)
            ]

        assert sequence(42) == sequence(42)
        assert sequence(42) != sequence(43)

    def test_ram_tie_break_ignores_cache_when_no_cache_set(self):
        """Without a cache, the router uses the old random tie-breaker."""
        prefill_0 = _make_prefill_instance(0)
        prefill_1 = _make_prefill_instance(1)
        decode_0 = _make_decode_instance(0)
        decode_1 = _make_decode_instance(1)

        router = Router(
            queue=[],
            prefill_instances=[prefill_0, prefill_1],
            decode_instances=[decode_0, decode_1],
            cache=None,
            random_seed=42,
        )
        active_prefill = {0: 100.0, 1: 100.0}
        active_decode = {0: 100.0, 1: 100.0}
        req = Request(isl=1000, osl=100)
        first = router._choose_prefill_instance(
            req, active_prefill, active_decode
        ).node_id
        second = router._choose_prefill_instance(
            req, active_prefill, active_decode
        ).node_id
        # Same seed and same state -> random tie-breaker is deterministic.
        assert first == second
