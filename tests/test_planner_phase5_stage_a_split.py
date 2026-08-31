"""Tests for deterministic Stage-A semantic_group holdout splits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from tiergraph.enums import QueryType
from tiergraph.planner.annotation_step_a import (
    DEFAULT_STEP_A_ANNOTATIONS_PATH,
    EXPECTED_STAGE_A_COUNT,
    fingerprint_file,
)
from tiergraph.planner.annotation_step_b import DEFAULT_STEP_B_ANNOTATIONS_PATH
from tiergraph.planner.stage_a_split import (
    DEFAULT_SPLIT_SEED,
    FINAL_BUCKETS,
    group_holdout_split,
    validate_split_result,
)
from tiergraph.planner.stage_a_to_corpus import (
    count_explicit_h7_edges,
    load_stage_a_planner_examples,
)


ROOT = Path(__file__).resolve().parent.parent
STEP_A_PATH = ROOT / DEFAULT_STEP_A_ANNOTATIONS_PATH
STEP_B_PATH = ROOT / DEFAULT_STEP_B_ANNOTATIONS_PATH
ALT_SEED = 42


@pytest.fixture(scope="module")
def examples():
    return load_stage_a_planner_examples(STEP_A_PATH, STEP_B_PATH)


@pytest.fixture(scope="module")
def default_split(examples):
    return group_holdout_split(examples, seed=DEFAULT_SPLIT_SEED)


def _assignment_map(result) -> dict[str, str]:
    return dict(result.report["example_to_split"])


def test_exact_split_sizes(default_split):
    assert len(default_split.train) == 96
    assert len(default_split.dev) == 12
    assert len(default_split.test) == 12
    assert default_split.report["sizes"] == {
        "train": 96,
        "dev": 12,
        "test": 12,
    }
    assert validate_split_result(default_split) == []


def test_zero_semantic_group_leakage(default_split):
    assert default_split.report["semantic_group_leakage"] == []
    group_to_splits: dict[str, set[str]] = {}
    for split_name, items in (
        ("train", default_split.train),
        ("dev", default_split.dev),
        ("test", default_split.test),
    ):
        for example in items:
            group = str(example.metadata["semantic_group"])
            group_to_splits.setdefault(group, set()).add(split_name)
    assert all(len(splits) == 1 for splits in group_to_splits.values())


def test_same_seed_same_assignment(examples, default_split):
    again = group_holdout_split(examples, seed=DEFAULT_SPLIT_SEED)
    assert again.fingerprint == default_split.fingerprint
    assert _assignment_map(again) == _assignment_map(default_split)
    assert [ex.example_id for ex in again.train] == [
        ex.example_id for ex in default_split.train
    ]
    assert [ex.example_id for ex in again.dev] == [
        ex.example_id for ex in default_split.dev
    ]
    assert [ex.example_id for ex in again.test] == [
        ex.example_id for ex in default_split.test
    ]


def test_input_order_perturbation_same_assignment(examples, default_split):
    shuffled = list(reversed(examples))
    mid = len(shuffled) // 2
    shuffled = shuffled[mid:] + shuffled[:mid]
    again = group_holdout_split(shuffled, seed=DEFAULT_SPLIT_SEED)
    assert again.fingerprint == default_split.fingerprint
    assert _assignment_map(again) == _assignment_map(default_split)


def test_different_seed_can_differ(examples, default_split):
    alt = group_holdout_split(examples, seed=ALT_SEED)
    assert validate_split_result(alt) == []
    assert alt.report["sizes"] == {"train": 96, "dev": 12, "test": 12}
    assert alt.report["semantic_group_leakage"] == []
    assert alt.fingerprint != default_split.fingerprint
    assert _assignment_map(alt) != _assignment_map(default_split)


def test_each_split_has_h7_positives(default_split):
    by_split = default_split.report["by_split"]
    # Feasible Stage-A target: ≥2 H7 in each eval split (prefer ~11/2/2).
    assert int(by_split["train"]["h7_positive"]) >= 1
    assert int(by_split["dev"]["h7_positive"]) >= 2
    assert int(by_split["test"]["h7_positive"]) >= 2
    assert (
        sum(
            1
            for ex in default_split.train
            if count_explicit_h7_edges(ex) > 0
        )
        == by_split["train"]["h7_positive"]
    )


def test_train_has_majority_of_h7_positives(default_split):
    total = int(default_split.report["n_h7_positive_total"])
    assert total == 15
    train_h7 = int(default_split.report["by_split"]["train"]["h7_positive"])
    dev_h7 = int(default_split.report["by_split"]["dev"]["h7_positive"])
    test_h7 = int(default_split.report["by_split"]["test"]["h7_positive"])
    assert train_h7 + dev_h7 + test_h7 == total
    assert train_h7 >= (total // 2 + total % 2)
    assert train_h7 >= 8
    # Preferred resilient distribution when feasible.
    assert (train_h7, dev_h7, test_h7) == (11, 2, 2) or (
        train_h7 >= 9 and dev_h7 >= 2 and test_h7 >= 2
    )


def test_reasonable_bucket_and_query_type_balance(default_split):
    by_split = default_split.report["by_split"]
    for split, size in (("train", 96), ("dev", 12), ("test", 12)):
        bucket_counts = by_split[split]["final_bucket"]
        assert set(bucket_counts) == set(FINAL_BUCKETS)
        # Each of the five buckets appears in every split.
        assert all(int(bucket_counts[bucket]) >= 1 for bucket in FINAL_BUCKETS)
        # No bucket monopolizes a split.
        assert max(int(v) for v in bucket_counts.values()) <= size - 4

        qt = by_split[split]["query_type"]
        assert qt.get(QueryType.PERSONAL.value, 0) >= 1
        assert qt.get(QueryType.ENVIRONMENTAL.value, 0) >= 1
        assert qt.get(QueryType.MIXED.value, 0) >= 1

    # Global per-bucket totals remain 24.
    global_buckets = Counter()
    for split in ("train", "dev", "test"):
        global_buckets.update(by_split[split]["final_bucket"])
    assert dict(global_buckets) == {bucket: 24 for bucket in FINAL_BUCKETS}


def test_all_ids_appear_exactly_once(examples, default_split):
    all_ids = [
        ex.example_id
        for ex in (*default_split.train, *default_split.dev, *default_split.test)
    ]
    assert len(all_ids) == EXPECTED_STAGE_A_COUNT
    assert len(set(all_ids)) == EXPECTED_STAGE_A_COUNT
    assert set(all_ids) == {ex.example_id for ex in examples}
    assert set(all_ids) == {
        f"sa_{index:04d}" for index in range(1, EXPECTED_STAGE_A_COUNT + 1)
    }


def test_frozen_annotation_files_unchanged(examples):
    step_a_before = fingerprint_file(STEP_A_PATH)
    step_b_before = fingerprint_file(STEP_B_PATH)
    step_a_bytes = STEP_A_PATH.read_bytes()
    step_b_bytes = STEP_B_PATH.read_bytes()

    result = group_holdout_split(examples, seed=DEFAULT_SPLIT_SEED)
    assert len(result.train) + len(result.dev) + len(result.test) == 120

    assert fingerprint_file(STEP_A_PATH) == step_a_before
    assert fingerprint_file(STEP_B_PATH) == step_b_before
    assert STEP_A_PATH.read_bytes() == step_a_bytes
    assert STEP_B_PATH.read_bytes() == step_b_bytes


def test_template_overlap_reported_but_not_hard_failure(default_split):
    # Overlap is expected under semantic_group holdout; it must not fail validation.
    assert "template_group_overlap" in default_split.report
    assert "template_group_overlap_count" in default_split.report
    assert isinstance(default_split.report["template_group_overlap"], list)
    assert int(default_split.report["template_group_overlap_count"]) >= 0
    assert validate_split_result(default_split) == []
    # Overlap entries name template groups that appear in >1 split.
    for item in default_split.report["template_group_overlap"]:
        assert "template_group" in item
        assert len(item["splits"]) >= 2
