"""Tests for deterministic Stage-A v2 publication-oriented holdout splits."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from tiergraph.planner.stage_a_split import group_holdout_split
from tiergraph.planner.stage_a_to_corpus import load_stage_a_planner_examples
from tiergraph.planner.stage_a_v2_spec import (
    H7_SPLIT_FLOOR_DEV,
    H7_SPLIT_FLOOR_TEST,
    LEGAL_H7_FAMILY_LABELS,
    STAGE_A_V2_BUCKETS,
    STAGE_A_V2_CORPUS_SIZE,
    STAGE_A_V2_DEV_SIZE,
    STAGE_A_V2_SPLIT_PATH,
    STAGE_A_V2_SPLIT_REPORT_PATH,
    STAGE_A_V2_SPLIT_SEED,
    STAGE_A_V2_TEST_IDS_PATH,
    STAGE_A_V2_TEST_SIZE,
    STAGE_A_V2_TRAIN_SIZE,
)
from tiergraph.planner.stage_a_v2_split import (
    H5_SPLIT_FLOOR_DEV,
    H5_SPLIT_FLOOR_TEST,
    build_stage_a_v2_split,
    regenerate_stage_a_v2_split_report,
    validate_split_result_v2,
    write_stage_a_v2_split,
    write_stage_a_v2_split_report,
)

ROOT = Path(__file__).resolve().parent.parent
ALT_SEED = 42


@pytest.fixture(scope="module")
def v2_split():
    result, rows_by_id = build_stage_a_v2_split(seed=STAGE_A_V2_SPLIT_SEED)
    return result, rows_by_id


@pytest.fixture(scope="module")
def frozen_report():
    return json.loads(STAGE_A_V2_SPLIT_REPORT_PATH.read_text(encoding="utf-8"))


def _assignment_map(result) -> dict[str, str]:
    return dict(result.report["example_to_split"])


def test_exact_split_sizes(v2_split):
    result, rows_by_id = v2_split
    assert len(result.train) == STAGE_A_V2_TRAIN_SIZE
    assert len(result.dev) == STAGE_A_V2_DEV_SIZE
    assert len(result.test) == STAGE_A_V2_TEST_SIZE
    assert result.report["sizes"] == {
        "train": STAGE_A_V2_TRAIN_SIZE,
        "dev": STAGE_A_V2_DEV_SIZE,
        "test": STAGE_A_V2_TEST_SIZE,
    }
    assert validate_split_result_v2(result, rows_by_id=rows_by_id) == []


def test_zero_semantic_group_leakage(v2_split):
    result, _rows_by_id = v2_split
    assert result.report["semantic_group_leakage"] == {}
    group_to_splits: dict[str, set[str]] = {}
    for split_name, items in (
        ("train", result.train),
        ("dev", result.dev),
        ("test", result.test),
    ):
        for example in items:
            group = str(example.metadata["semantic_group"])
            group_to_splits.setdefault(group, set()).add(split_name)
    assert all(len(splits) == 1 for splits in group_to_splits.values())


def test_zero_authored_holdout_family_leakage(v2_split):
    result, _rows_by_id = v2_split
    assert result.report["authored_holdout_family_leakage"] == {}


def test_quarantine_excluded_from_test(v2_split):
    result, rows_by_id = v2_split
    assert result.report["quarantined_in_test"] == []
    for example_id in result.report["test_ids"]:
        assert bool(rows_by_id[example_id].get("publication_test_eligible"))


def test_same_seed_same_assignment(v2_split):
    result, rows_by_id = v2_split
    again, _ = build_stage_a_v2_split(seed=STAGE_A_V2_SPLIT_SEED)
    assert again.fingerprint == result.fingerprint
    assert _assignment_map(again) == _assignment_map(result)


def test_different_seed_can_differ(v2_split):
    result, rows_by_id = v2_split
    alt, alt_rows = build_stage_a_v2_split(seed=ALT_SEED)
    assert validate_split_result_v2(alt, rows_by_id=alt_rows) == []
    assert alt.report["sizes"] == result.report["sizes"]
    assert alt.report["semantic_group_leakage"] == {}
    assert alt.fingerprint != result.fingerprint
    assert _assignment_map(alt) != _assignment_map(result)


def test_h5_h7_floors_in_dev_and_test(v2_split):
    result, _rows_by_id = v2_split
    by_split = result.report["by_split"]
    assert int(by_split["dev"]["h5_positive_rows"]) >= H5_SPLIT_FLOOR_DEV
    assert int(by_split["test"]["h5_positive_rows"]) >= H5_SPLIT_FLOOR_TEST
    assert int(by_split["dev"]["h7_positive_rows"]) >= H7_SPLIT_FLOOR_DEV
    assert int(by_split["test"]["h7_positive_rows"]) >= H7_SPLIT_FLOOR_TEST


def test_h7_families_are_legal_labels(v2_split):
    result, _rows_by_id = v2_split
    for split in ("train", "dev", "test"):
        families = result.report["by_split"][split]["h7_family_counts"]
        assert set(families).issubset(set(LEGAL_H7_FAMILY_LABELS))


def test_canonical_h1_totals_match_corpus_design(v2_split):
    result, _rows_by_id = v2_split
    assert result.report["h1_classification_label_total"] == {
        "Environmental": 96,
        "Mixed": 288,
        "Personal": 96,
    }
    by_split = result.report["by_split"]
    global_h1 = Counter()
    for split in ("train", "dev", "test"):
        global_h1.update(by_split[split]["h1_classification_label"])
    assert dict(global_h1) == result.report["h1_classification_label_total"]


def test_reasonable_bucket_and_h1_balance(v2_split):
    result, _rows_by_id = v2_split
    by_split = result.report["by_split"]
    for split, size in (
        ("train", STAGE_A_V2_TRAIN_SIZE),
        ("dev", STAGE_A_V2_DEV_SIZE),
        ("test", STAGE_A_V2_TEST_SIZE),
    ):
        bucket_counts = by_split[split]["final_bucket"]
        assert set(bucket_counts) == set(STAGE_A_V2_BUCKETS)
        for bucket in STAGE_A_V2_BUCKETS:
            count = int(bucket_counts[bucket])
            if split == "train":
                assert 76 <= count <= 78, f"{bucket} train={count}"
            else:
                assert 9 <= count <= 11, f"{bucket} {split}={count}"

        h1 = by_split[split]["h1_classification_label"]
        assert h1.get("Personal", 0) >= 1
        assert h1.get("Environmental", 0) >= 1
        assert h1.get("Mixed", 0) >= 1

    global_buckets = Counter()
    for split in ("train", "dev", "test"):
        global_buckets.update(by_split[split]["final_bucket"])
    assert dict(global_buckets) == {bucket: 96 for bucket in STAGE_A_V2_BUCKETS}


def test_all_ids_appear_exactly_once(v2_split):
    result, _rows_by_id = v2_split
    all_ids = [
        ex.example_id
        for ex in (*result.train, *result.dev, *result.test)
    ]
    assert len(all_ids) == STAGE_A_V2_CORPUS_SIZE
    assert len(set(all_ids)) == STAGE_A_V2_CORPUS_SIZE
    assert set(all_ids) == {f"sa_{index:04d}" for index in range(1, STAGE_A_V2_CORPUS_SIZE + 1)}


def test_frozen_artifacts_match_regeneration(v2_split, frozen_report):
    result, _rows_by_id = v2_split
    assert result.seed == STAGE_A_V2_SPLIT_SEED
    assert result.fingerprint == frozen_report["fingerprint"]
    assert result.report["test_ids"] == frozen_report["test_ids"]
    assert STAGE_A_V2_SPLIT_PATH.exists()
    assert STAGE_A_V2_TEST_IDS_PATH.exists()


def test_v2_test_is_not_v1_test_definition(v2_split):
    """Publication v2 test must not reuse the legacy 12-example v1 test set."""
    result, _rows_by_id = v2_split
    v1_examples = load_stage_a_planner_examples()
    v1_result = group_holdout_split(v1_examples, seed=20260831)
    v1_test_ids = {ex.example_id for ex in v1_result.test}
    v2_test_ids = set(result.report["test_ids"])
    assert v1_test_ids != v2_test_ids
    assert len(v2_test_ids) == STAGE_A_V2_TEST_SIZE


def test_template_overlap_reported_but_not_hard_failure(v2_split):
    result, rows_by_id = v2_split
    assert "template_group_overlap" in result.report
    assert "template_group_overlap_count" in result.report
    assert isinstance(result.report["template_group_overlap"], list)
    assert validate_split_result_v2(result, rows_by_id=rows_by_id) == []


def test_regenerate_report_preserves_frozen_assignment(v2_split):
    result, _rows_by_id = v2_split
    frozen_assignment = dict(result.report["example_to_split"])
    regen = regenerate_stage_a_v2_split_report(assignment=frozen_assignment)
    assert regen.fingerprint == result.fingerprint
    assert regen.report["example_to_split"] == frozen_assignment
    assert regen.report["test_ids"] == result.report["test_ids"]
    assert regen.report["h1_classification_label_total"] == {
        "Environmental": 96,
        "Mixed": 288,
        "Personal": 96,
    }


def test_write_stage_a_v2_split_is_idempotent(tmp_path):
    out = tmp_path / "split.jsonl"
    report_path = tmp_path / "report.json"
    test_ids_path = tmp_path / "test_ids.json"
    first = write_stage_a_v2_split(
        output_path=out,
        report_path=report_path,
        test_ids_path=test_ids_path,
        seed=STAGE_A_V2_SPLIT_SEED,
    )
    second = write_stage_a_v2_split(
        output_path=out,
        report_path=report_path,
        test_ids_path=test_ids_path,
        seed=STAGE_A_V2_SPLIT_SEED,
    )
    assert first["fingerprint"] == second["fingerprint"]
    assert first["example_to_split"] == second["example_to_split"]


def test_write_stage_a_v2_split_report_from_frozen_assignment(tmp_path):
    report_path = tmp_path / "report.json"
    test_ids_path = tmp_path / "test_ids.json"
    first = write_stage_a_v2_split_report(
        report_path=report_path,
        test_ids_path=test_ids_path,
        seed=STAGE_A_V2_SPLIT_SEED,
    )
    second = write_stage_a_v2_split_report(
        report_path=report_path,
        test_ids_path=test_ids_path,
        seed=STAGE_A_V2_SPLIT_SEED,
    )
    assert first["fingerprint"] == second["fingerprint"]
    assert first["example_to_split"] == second["example_to_split"]
