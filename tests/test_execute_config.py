"""Tests for execute_config.py batching helpers.

These tests focus on the config-grouping logic that splits a flat list of
configs into batches of compatible hardware for parallel execution.
"""

import pytest

from execute_config import (
    _group_colocated_configs,
    _group_separate_configs,
    extreme_first_eytzinger_layout,
    parse_users_arg,
)


def _cfg(
    label: str,
    prefill_hw: str,
    decode_hw: str,
    prefill_nodes: int,
    decode_nodes: int,
    batch_size: int,
    colocated: str = "false",
    prefill_gpus: int = 0,
    decode_gpus: int = 0,
) -> dict:
    return {
        "label": label,
        "prefill_hardware": prefill_hw,
        "decode_hardware": decode_hw,
        "prefill_nodes": str(prefill_nodes),
        "decode_nodes": str(decode_nodes),
        "batch_size": str(batch_size),
        "colocated": colocated,
        "prefill_gpus_per_node": str(prefill_gpus),
        "decode_gpus_per_node": str(decode_gpus),
    }


class TestExtremeFirstEytzingerLayout:
    def test_empty(self) -> None:
        assert extreme_first_eytzinger_layout([], key=lambda x: x) == []

    def test_single(self) -> None:
        assert extreme_first_eytzinger_layout([5], key=lambda x: x) == [5]

    def test_two_descending(self) -> None:
        assert extreme_first_eytzinger_layout([1, 5], key=lambda x: x) == [5, 1]

    def test_three_extremes_first(self) -> None:
        # sorted: [1, 3, 5]; largest=5, smallest=1, middle=[3]
        assert extreme_first_eytzinger_layout([3, 1, 5], key=lambda x: x) == [5, 1, 3]

    def test_seven_order(self) -> None:
        arr = list(range(7))  # 0..6
        result = extreme_first_eytzinger_layout(arr, key=lambda x: x)
        # largest=6, smallest=0, middle=[1,2,3,4,5] -> Eytzinger of middle
        # sorted middle [1,2,3,4,5]; Eytzinger build gives [4,2,5,1,3]
        assert result[0] == 6
        assert result[1] == 0
        assert result[2:] == [4, 2, 5, 1, 3]

    def test_key_uses_field(self) -> None:
        cfg = [
            {"k": 1, "label": "a"},
            {"k": 5, "label": "b"},
            {"k": 3, "label": "c"},
        ]
        result = extreme_first_eytzinger_layout(cfg, key=lambda c: c["k"])
        assert [c["k"] for c in result] == [5, 1, 3]

    def test_outer_prefill_group_order(self) -> None:
        """Outer prefill groups should be ordered largest-first, smallest-second, then Eytzinger."""
        groups = [
            [[{"prefill_hardware": "A", "prefill_nodes": "1"}]],
            [[{"prefill_hardware": "A", "prefill_nodes": "8"}]],
            [[{"prefill_hardware": "A", "prefill_nodes": "4"}]],
            [[{"prefill_hardware": "A", "prefill_nodes": "2"}]],
        ]
        result = extreme_first_eytzinger_layout(
            groups,
            key=lambda pb: (
                pb[0][0]["prefill_hardware"],
                int(pb[0][0]["prefill_nodes"]),
            ),
        )
        # largest=8, smallest=1, middle=[2,4] -> Eytzinger of middle = [4,2]
        assert [pb[0][0]["prefill_nodes"] for pb in result] == ["8", "1", "4", "2"]


class TestParseUsersArg:
    def test_none_returns_none(self) -> None:
        assert parse_users_arg(None) is None

    def test_comma_separated_list(self) -> None:
        assert parse_users_arg("1,10,100") == [1, 10, 100]

    def test_comma_separated_unsorted_returns_sorted_unique(self) -> None:
        assert parse_users_arg("100,10,10,1") == [1, 10, 100]

    def test_rejects_range_notation(self) -> None:
        with pytest.raises(ValueError, match=r".*"):
            parse_users_arg("[1,1000]")

    def test_rejects_negative_values(self) -> None:
        with pytest.raises(ValueError, match=r".*"):
            parse_users_arg("-10,10")


class TestColocatedGrouping:
    def test_empty_list(self) -> None:
        assert _group_colocated_configs([]) == []

    def test_single_config(self) -> None:
        cfg = _cfg(
            "c1",
            "H200 x8 #a",
            "H200 x8 #a",
            1,
            1,
            64,
            colocated="true",
            prefill_gpus=4,
            decode_gpus=4,
        )
        groups = _group_colocated_configs([cfg])
        assert groups == [[cfg]]

    def test_same_hardware_grouped_together(self) -> None:
        base = ("H200 x8 #a", "H200 x8 #a")
        c1 = _cfg(
            "c1", *base, 1, 1, 64, colocated="true", prefill_gpus=4, decode_gpus=4
        )
        c2 = _cfg(
            "c2", *base, 2, 2, 64, colocated="true", prefill_gpus=4, decode_gpus=4
        )
        c3 = _cfg(
            "c3", *base, 1, 1, 128, colocated="true", prefill_gpus=6, decode_gpus=2
        )
        groups = _group_colocated_configs([c1, c2, c3])
        assert len(groups) == 2
        assert sorted(groups[0], key=lambda c: c["label"]) == sorted(
            [c1, c2], key=lambda c: c["label"]
        )
        assert sorted(groups[1], key=lambda c: c["label"]) == sorted(
            [c3], key=lambda c: c["label"]
        )

    def test_different_prefill_hardware_split(self) -> None:
        c1 = _cfg(
            "c1",
            "H200 x8 #a",
            "H200 x8 #a",
            1,
            1,
            64,
            colocated="true",
            prefill_gpus=4,
            decode_gpus=4,
        )
        c2 = _cfg(
            "c2",
            "B200 x8 #b",
            "B200 x8 #b",
            1,
            1,
            64,
            colocated="true",
            prefill_gpus=4,
            decode_gpus=4,
        )
        groups = _group_colocated_configs([c1, c2])
        assert len(groups) == 2
        assert sorted(groups[0], key=lambda c: c["label"]) == [c1]
        assert sorted(groups[1], key=lambda c: c["label"]) == [c2]

    def test_matters_for_colocated(self) -> None:
        # Colocated nodes must have identical prefill and decode hardware, but
        # the grouping key still checks both fields.
        c1 = _cfg(
            "c1",
            "H200 x8 #a",
            "H200 x8 #a",
            1,
            1,
            64,
            colocated="true",
            prefill_gpus=6,
            decode_gpus=2,
        )
        c2 = _cfg(
            "c2",
            "H200 x8 #a",
            "H200 x8 #a",
            1,
            1,
            64,
            colocated="true",
            prefill_gpus=4,
            decode_gpus=4,
        )
        groups = _group_colocated_configs([c1, c2])
        assert len(groups) == 2


class TestSingleNodeGrouping:
    def test_empty_list(self) -> None:
        assert _group_separate_configs([]) == []

    def test_single_config(self) -> None:
        cfg = _cfg("c1", "H200 x8 #a", "H200 x8 #b", 1, 2, 64)
        groups = _group_separate_configs([cfg])
        assert groups == [[[cfg]]]

    def test_grouped_by_prefill_hardware_then_prefill_nodes(self) -> None:
        # Same prefill hw + nodes, different decode hw -> two decode batches
        # inside one prefill batch.
        c1 = _cfg("c1", "H200 x8 #a", "H200 x8 #b", 1, 2, 64)
        c2 = _cfg("c2", "H200 x8 #a", "H200 x8 #c", 1, 2, 64)
        groups = _group_separate_configs([c1, c2])
        assert len(groups) == 1
        # Two decode batches within the single prefill batch.
        assert len(groups[0]) == 2
        assert groups[0][0] == [c1]
        assert groups[0][1] == [c2]

    def test_same_prefill_and_decode_hardware_grouped_together(self) -> None:
        c1 = _cfg("c1", "H200 x8 #a", "H200 x8 #b", 1, 2, 64)
        c2 = _cfg("c2", "H200 x8 #a", "H200 x8 #b", 1, 4, 64)
        c3 = _cfg("c3", "H200 x8 #a", "H200 x8 #b", 1, 2, 128)
        groups = _group_separate_configs([c1, c2, c3])
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert sorted(groups[0][0], key=lambda c: c["label"]) == sorted(
            [c1, c2, c3], key=lambda c: c["label"]
        )

    def test_different_prefill_nodes_create_separate_prefill_batches(self) -> None:
        c1 = _cfg("c1", "H200 x8 #a", "H200 x8 #b", 1, 2, 64)
        c2 = _cfg("c2", "H200 x8 #a", "H200 x8 #b", 2, 2, 64)
        groups = _group_separate_configs([c1, c2])
        assert len(groups) == 2
        assert len(groups[0]) == 1
        assert len(groups[1]) == 1
        assert groups[0][0] == [c1]
        assert groups[1][0] == [c2]

    def test_different_prefill_hardware_create_separate_prefill_batches(self) -> None:
        c1 = _cfg("c1", "H200 x8 #a", "H200 x8 #b", 1, 2, 64)
        c2 = _cfg("c2", "B200 x8 #b", "H200 x8 #b", 1, 2, 64)
        groups = _group_separate_configs([c1, c2])
        assert len(groups) == 2
        assert groups[0][0] == [c1]
        assert groups[1][0] == [c2]

    def test_no_duplicate_configs(self) -> None:
        # Regression: the original implementation appended a config to a matching
        # decode batch and then unconditionally appended it again as a new batch.
        c1 = _cfg("c1", "H200 x8 #a", "H200 x8 #b", 1, 2, 64)
        c2 = _cfg("c2", "H200 x8 #a", "H200 x8 #b", 1, 4, 64)
        groups = _group_separate_configs([c1, c2])
        flat = [
            cfg
            for prefill_batch in groups
            for decode_batch in prefill_batch
            for cfg in decode_batch
        ]
        assert flat == [c1, c2]

    def test_complex_mix(self) -> None:
        c1 = _cfg("c1", "H200 x8 #a", "H200 x8 #b", 1, 2, 64)
        c2 = _cfg("c2", "H200 x8 #a", "H200 x8 #b", 2, 2, 64)
        c3 = _cfg("c3", "H200 x8 #a", "H200 x8 #c", 1, 2, 64)
        c4 = _cfg("c4", "B200 x8 #b", "H200 x8 #b", 1, 2, 64)
        groups = _group_separate_configs([c1, c2, c3, c4])

        # Three prefill batches keyed by (prefill_hardware, prefill_nodes):
        #   (H200 x8 #a, 1): c1 and c3 -> two decode batches
        #   (H200 x8 #a, 2): c2        -> one decode batch
        #   (B200 x8 #b, 1): c4        -> one decode batch
        assert len(groups) == 3

        h200_pn1 = next(
            pb
            for pb in groups
            if pb[0][0]["prefill_hardware"] == "H200 x8 #a"
            and pb[0][0]["prefill_nodes"] == "1"
        )
        assert len(h200_pn1) == 2
        decode_batches = {
            tuple(sorted(cfg["decode_hardware"] for cfg in db)): db for db in h200_pn1
        }
        assert ("H200 x8 #b",) in decode_batches
        assert ("H200 x8 #c",) in decode_batches
        assert decode_batches[("H200 x8 #b",)] == [c1]
        assert decode_batches[("H200 x8 #c",)] == [c3]

        h200_pn2 = next(
            pb
            for pb in groups
            if pb[0][0]["prefill_hardware"] == "H200 x8 #a"
            and pb[0][0]["prefill_nodes"] == "2"
        )
        assert len(h200_pn2) == 1
        assert h200_pn2[0] == [c2]

        b200_prefill = next(
            pb for pb in groups if pb[0][0]["prefill_hardware"] == "B200 x8 #b"
        )
        assert len(b200_prefill) == 1
        assert b200_prefill[0] == [c4]
