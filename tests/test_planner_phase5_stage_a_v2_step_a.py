"""Focused tests for Stage-A v2 Step-A annotations (Todo Step-A only)."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from tiergraph.enums import OperatorType
from tiergraph.planner.annotation_step_a import (
    StageAStepAAnnotation,
    StepAStatus,
    load_step_a_annotations,
)
from tiergraph.planner.stage_a_selection import load_jsonl
from tiergraph.planner.stage_a_v2_step_a import (
    annotate_new_row,
    build_stage_a_v2_step_a,
    validate_step_a_v2_corpus,
)
from tiergraph.planner.stage_a_v2_spec import (
    STAGE_A_V1_SELECTION_PATH,
    STAGE_A_V1_STEP_A_PATH,
    STAGE_A_V1_STEP_B_PATH,
    STAGE_A_V2_CORPUS_SIZE,
    STAGE_A_V2_SELECTION_PATH,
    STAGE_A_V2_STEP_A_PATH,
    STAGE_A_V2_STEP_A_REPORT_PATH,
)


ROOT = Path(__file__).resolve().parent.parent

FROZEN_V1 = (
    STAGE_A_V1_SELECTION_PATH,
    STAGE_A_V1_STEP_A_PATH,
    STAGE_A_V1_STEP_B_PATH,
)

_ALLOWED = frozenset(
    {
        OperatorType.RETRIEVE_PERSONAL,
        OperatorType.IDENTIFY_ENVIRONMENTAL,
        OperatorType.LOCATE_ENVIRONMENTAL,
        OperatorType.NAVIGATE_TO,
        OperatorType.DESCRIBE_ENVIRONMENT,
    }
)


@pytest.fixture(scope="module")
def step_a_batch():
    return build_stage_a_v2_step_a()


def test_v1_step_a_b_unchanged():
    for rel in FROZEN_V1:
        assert (ROOT / rel).is_file()
        diff = subprocess.run(
            ["git", "diff", "--", str(rel).replace("\\", "/")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert diff.stdout == ""


def test_artifact_geometry_and_legacy_copy(step_a_batch):
    records, report = step_a_batch
    assert len(records) == STAGE_A_V2_CORPUS_SIZE
    assert report["legacy_copied"] == 120
    assert report["new_agent_assisted"] == 360
    assert report["complete_count"] == 480
    assert report.get("ontology_blocked_count", 0) == 0
    assert all(r.step_a_status is StepAStatus.COMPLETE for r in records)

    legacy = load_step_a_annotations(ROOT / STAGE_A_V1_STEP_A_PATH)
    by_id = {r.stage_a_id: r for r in records}
    for src in legacy:
        dst = by_id[src.stage_a_id]
        assert dst.query == src.query
        assert dst.final_bucket == src.final_bucket
        assert [
            (o.char_start, o.char_end, o.text, o.operator_type)
            for o in dst.operations
        ] == [
            (o.char_start, o.char_end, o.text, o.operator_type)
            for o in src.operations
        ]
        assert [
            (a.char_start, a.char_end, a.text) for a in dst.anchors
        ] == [(a.char_start, a.char_end, a.text) for a in src.anchors]


def test_validate_corpus_and_disallowed_ops(step_a_batch):
    records, _report = step_a_batch
    errors = validate_step_a_v2_corpus(
        records, selection_path=ROOT / STAGE_A_V2_SELECTION_PATH
    )
    assert errors == []
    for record in records:
        assert record.step_a_status is StepAStatus.COMPLETE
        for op in record.operations:
            assert op.operator_type in _ALLOWED
            assert op.operator_type is not OperatorType.RESOLVE_PERSONAL
            assert op.operator_type is not OperatorType.FUSE


def test_on_disk_artifacts_match_builder(step_a_batch):
    records, report = step_a_batch
    assert (ROOT / STAGE_A_V2_STEP_A_PATH).is_file()
    assert (ROOT / STAGE_A_V2_STEP_A_REPORT_PATH).is_file()
    on_disk = load_step_a_annotations(ROOT / STAGE_A_V2_STEP_A_PATH)
    assert [r.stage_a_id for r in on_disk] == [r.stage_a_id for r in records]
    disk_report = json.loads((ROOT / STAGE_A_V2_STEP_A_REPORT_PATH).read_text())
    assert disk_report["fingerprint"] == report["fingerprint"]
    assert disk_report["total_rows"] == 480


def test_deterministic_rebuild(step_a_batch):
    a, ra = step_a_batch
    b, rb = build_stage_a_v2_step_a()
    assert ra["fingerprint"] == rb["fingerprint"]
    assert [
        (
            r.stage_a_id,
            [(o.char_start, o.char_end, o.operator_type) for o in r.operations],
            [(x.char_start, x.char_end) for x in r.anchors],
        )
        for r in a
    ] == [
        (
            r.stage_a_id,
            [(o.char_start, o.char_end, o.operator_type) for o in r.operations],
            [(x.char_start, x.char_end) for x in r.anchors],
        )
        for r in b
    ]


def test_sequential_triple_and_urgency_boundaries():
    selection = {
        row["stage_a_id"]: row
        for row in load_jsonl(ROOT / STAGE_A_V2_SELECTION_PATH)
    }
    sequential = selection["sa_0427"]
    record, _meta = annotate_new_row(sequential)
    assert [op.operator_type.value for op in record.operations] == [
        "IDENTIFY_ENVIRONMENTAL",
        "LOCATE_ENVIRONMENTAL",
        "NAVIGATE_TO",
    ]
    assert record.operations[0].text == "Identify the stairwell door label"

    urgency = next(
        row
        for row in selection.values()
        if str(row.get("query", "")).startswith("Urgent:")
    )
    urg_rec, _ = annotate_new_row(urgency)
    assert len(urg_rec.operations) == 1
    assert urg_rec.operations[0].text.lower().startswith("describe")
    assert "urgent" not in urg_rec.operations[0].text.lower()


def test_selection_metadata_aligned(step_a_batch):
    records, _ = step_a_batch
    selection = {
        row["stage_a_id"]: row
        for row in load_jsonl(ROOT / STAGE_A_V2_SELECTION_PATH)
    }
    assert len(selection) == STAGE_A_V2_CORPUS_SIZE
    for record in records:
        row = selection[record.stage_a_id]
        assert record.query == row["query"]
        assert record.final_bucket == row["final_bucket"]
        assert record.candidate_id == row.get("candidate_id")
        assert record.source_id == row.get("source_id")


def test_report_counts_consistent(step_a_batch):
    records, report = step_a_batch
    op_counts = Counter(
        op.operator_type.value for r in records for op in r.operations
    )
    assert report["operator_counts"] == dict(op_counts)
    assert report["total_explicit_operations"] == sum(
        len(r.operations) for r in records
    )
    assert report["total_anchors"] == sum(len(r.anchors) for r in records)
    assert report["rows_with_multiple_operations"] == sum(
        1 for r in records if len(r.operations) > 1
    )
