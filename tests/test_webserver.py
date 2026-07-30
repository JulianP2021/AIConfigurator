"""Regression tests for the webserver simulation helpers."""

import pytest


@pytest.fixture
def webserver_module():
    """Import the webserver module (heavy, so only do it in tests that need it)."""
    import sys

    from pathlib import Path

    # src/webserver/server.py adds the project root to sys.path, so import via
    # the same path the server uses.
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.webserver import server

    return server


class TestBuildNodesInterNodeBandwidth:
    def test_build_nodes_applies_inter_node_bandwidth_override(
        self, webserver_module, monkeypatch
    ):
        """_build_nodes must use the provided inter-node bandwidth values."""
        monkeypatch.setenv("INTER_NODE_NETWORK_UP_GBPS", "100.0")
        monkeypatch.setenv("INTER_NODE_NETWORK_DOWN_GBPS", "100.0")

        nodes = webserver_module._build_nodes(
            prefill_hardware_name="H200 x4 #0a43645c",
            decode_hardware_name="H200 x4 #0a43645c",
            prefill_nodes=1,
            decode_nodes=1,
            prefill_gpus_per_node=4,
            decode_gpus_per_node=4,
            batch_size=2,
            model="Qwen/Qwen3-8B",
            colocated=False,
            inter_node_network_up_gbps=2.0,
            inter_node_network_down_gbps=3.0,
        )

        assert len(nodes) == 2
        for node in nodes:
            assert node.hardware.spec.network_inter_node_up == int(2.0 * 1e9 / 8.0)
            assert node.hardware.spec.network_inter_node_down == int(3.0 * 1e9 / 8.0)

        # _build_nodes writes the values into os.environ as a side effect so
        # fetch_machine_hardware picks them up.  Verify they were updated.
        import os

        assert os.environ["INTER_NODE_NETWORK_UP_GBPS"] == "2.0"
        assert os.environ["INTER_NODE_NETWORK_DOWN_GBPS"] == "3.0"

    def test_build_nodes_uses_default_inter_node_bandwidth(
        self, webserver_module, monkeypatch
    ):
        """When no override is supplied, the default 100 Gbps is used."""
        monkeypatch.setenv("INTER_NODE_NETWORK_UP_GBPS", "100.0")
        monkeypatch.setenv("INTER_NODE_NETWORK_DOWN_GBPS", "100.0")

        nodes = webserver_module._build_nodes(
            prefill_hardware_name="H200 x4 #0a43645c",
            decode_hardware_name="H200 x4 #0a43645c",
            prefill_nodes=1,
            decode_nodes=1,
            prefill_gpus_per_node=4,
            decode_gpus_per_node=4,
            batch_size=2,
            model="Qwen/Qwen3-8B",
            colocated=False,
        )

        expected = int(100.0 * 1e9 / 8.0)
        for node in nodes:
            assert node.hardware.spec.network_inter_node_up == expected
            assert node.hardware.spec.network_inter_node_down == expected


class TestTTFTDelayPlots:
    def test_build_ttft_cost_plots_groups_by_user_delay(self, webserver_module):
        rows = [
            {
                "label": "cfg-a | TTFT=25ms | delay=0ms",
                "kv_upload_time": 10.0,
                "kv_download_time": 12.0,
                "total_cost_usd_per_hour": 1.25,
                "user_delay_ms": 0.0,
                "has_error": False,
                "focus": "nodes",
                "focus_value": "2",
                "color": "#d29922",
            },
            {
                "label": "cfg-b | TTFT=50ms | delay=0ms",
                "kv_upload_time": 10.0,
                "kv_download_time": 12.0,
                "total_cost_usd_per_hour": 1.50,
                "user_delay_ms": 0.0,
                "has_error": False,
                "focus": "nodes",
                "focus_value": "2",
                "color": "#d29922",
            },
            {
                "label": "cfg-c | TTFT=25ms | delay=10ms",
                "kv_upload_time": 10.0,
                "kv_download_time": 12.0,
                "total_cost_usd_per_hour": 1.75,
                "user_delay_ms": 10.0,
                "has_error": False,
                "focus": "nodes",
                "focus_value": "4",
                "color": "#d29922",
            },
        ]

        plots_by_ttft = webserver_module._build_ttft_economics_plots_by_delay(rows)

        # With TTFT grouping the outer dict has one key per TTFT value.
        assert set(plots_by_ttft.keys()) == {"TTFT=0.025s", "TTFT=0.05s"}
        all_urls = [url for urls in plots_by_ttft.values() for url in urls]
        # Two TTFTs, one plot per (ttft, delay, focus) bucket => 3 total.
        assert len(all_urls) == 3
        assert all(url.startswith("/plot/") for url in all_urls)

    def test_extract_user_delay_from_label(self, webserver_module):
        assert (
            webserver_module._extract_user_delay_ms({
                "label": "cfg | TTFT=100ms | delay=50ms"
            })
            == 50.0
        )

    def test_load_results_from_dir_keeps_rows_without_users(
        self, webserver_module, tmp_path
    ):
        path = tmp_path / "results_nodes_2.json"
        path.write_text(
            '{"results": [{"label": "cfg", "ttft": 10.0, "total_cost_usd_per_hour": 1.0}]}',
            encoding="utf-8",
        )

        rows, benchmark = webserver_module._load_results_from_dir(tmp_path)

        assert len(rows) == 1
        assert rows[0]["label"] == "cfg"
        assert benchmark == "hardware_economics"
