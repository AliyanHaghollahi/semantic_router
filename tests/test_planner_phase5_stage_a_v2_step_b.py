"""Focused tests for Stage-A v2 Step-B annotations (Todo Step-B only)."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from tiergraph.enums import OperatorType
from tiergraph.planner.annotation_step_b import (
    StepBStatus,
    load_step_b_annotations,
)
from tiergraph.planner.annotations import ImplicitResolution
from tiergraph.planner.operator_io import is_h7_pair_eligible
from tiergraph.planner.stage_a_selection import load_jsonl
from tiergraph.planner.stage_a_to_corpus import step_ab_to_planner_example
from tiergraph.planner.annotation_step_a import load_step_a_annotations
from tiergraph.planner.stage_a_v2_step_b import (
    annotate_new_row,
    build_stage_a_v2_step_b,
    validate_step_b_v2_corpus,
)
from tiergraph.planner.stage_a_v2_spec import (
    STAGE_A_V1_STEP_B_PATH,
    STAGE_A_V2_CORPUS_SIZE,
    STAGE_A_V2_SELECTION_PATH,
    STAGE_A_V2_STEP_A_PATH,
    STAGE_A_V2_STEP_B_PATH,
    STAGE_A_V2_STEP_B_REPORT_PATH,
)

ROOT = Path(__file__).resolve().parent.parent

FROZEN_V1_STEP_B = (STAGE_A_V1_STEP_B_PATH,)

MULTI_HOP_IDS = tuple(f"sa_{index:04d}" for index in range(426, 436))


@pytest.fixture(scope="module")
def step_b_batch():
    return build_stage_a_v2_step_b()


def test_v1_step_b_unchanged():
    for rel in FROZEN_V1_STEP_B:
        assert (ROOT / rel).is_file()
        diff = subprocess.run(
            ["git", "diff", "--", str(rel).replace("\\", "/")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert diff.stdout == ""


def test_artifact_geometry_and_legacy_copy(step_b_batch):
    records, report = step_b_batch
    assert len(records) == STAGE_A_V2_CORPUS_SIZE
    assert report["B_legacy_copied"] == 120
    assert report["C_new_agent_assisted"] == 360
    assert report["D_complete_count"] == 480
    assert report["D_ambiguous_count"] == 0
    assert report["D_incompatible_count"] == 0
    assert all(r.step_b_status is StepBStatus.COMPLETE for r in records)

    legacy = load_step_b_annotations(ROOT / STAGE_A_V1_STEP_B_PATH)
    by_id = {r.stage_a_id: r for r in records}
    for src in legacy:
        dst = by_id[src.stage_a_id]
        assert dst.model_dump(mode="json") == src.model_dump(mode="json")


def test_validate_corpus_and_parallel_h7_zero(step_b_batch):
    records, _report = step_b_batch
    step_a = load_step_a_annotations(ROOT / STAGE_A_V2_STEP_A_PATH)
    errors = validate_step_b_v2_corpus(records, step_a_records=step_a)
    assert errors == []

    parallel_with_h7 = [
        r.stage_a_id
        for r in records
        if r.final_bucket == "MIXED_PARALLEL" and r.dependencies
    ]
    assert parallel_with_h7 == []


def test_retrieve_possessive_h5_negative(step_b_batch):
    records, _report = step_b_batch
    for record in records:
        for decision in record.anchor_decisions:
            owner = decision.owner_operation_index
            if owner is None:
                continue
            if record.operation_types[owner] != OperatorType.RETRIEVE_PERSONAL.value:
                continue
            assert decision.implicit_resolution is ImplicitResolution.NONE


def test_multi_hop_sequential_h7_chains(step_b_batch):
    records, report = step_b_batch
    by_id = {r.stage_a_id: r for r in records}
    assert report["H_multi_hop_examples"] == list(MULTI_HOP_IDS)
    for stage_a_id in MULTI_HOP_IDS:
        record = by_id[stage_a_id]
        assert len(record.dependencies) == 2
        assert record.dependencies[0].source_operation_index == 0
        assert record.dependencies[0].target_operation_index == 1
        assert record.dependencies[1].source_operation_index == 1
        assert record.dependencies[1].target_operation_index == 2
        ops = [OperatorType(value) for value in record.operation_types]
        assert is_h7_pair_eligible(ops[0], ops[1])
        assert is_h7_pair_eligible(ops[1], ops[2])


def test_h7_edges_contract_legal(step_b_batch):
    records, _report = step_b_batch
    for record in records:
        for dep in record.dependencies:
            source = OperatorType(record.operation_types[dep.source_operation_index])
            target = OperatorType(record.operation_types[dep.target_operation_index])
            assert is_h7_pair_eligible(source, target)


def test_decoder_validation_all_480(step_b_batch):
    records, report = step_b_batch
    step_a = load_step_a_annotations(ROOT / STAGE_A_V2_STEP_A_PATH)
    by_b = {r.stage_a_id: r for r in records}
    for row in step_a:
        step_ab_to_planner_example(row, by_b[row.stage_a_id])
    assert report["N_decoder_validation"]["decoder_valid_graph_count"] == 480
    assert report["N_decoder_validation"]["decoder_failure_count"] == 0


def test_written_artifact_matches_build(step_b_batch):
    records, report = step_b_batch
    on_disk = load_step_b_annotations(ROOT / STAGE_A_V2_STEP_B_PATH)
    assert len(on_disk) == STAGE_A_V2_CORPUS_SIZE
    assert report["fingerprint"] == json.loads(
        (ROOT / STAGE_A_V2_STEP_B_REPORT_PATH).read_text(encoding="utf-8")
    )["fingerprint"]
    assert {r.stage_a_id for r in on_disk} == {r.stage_a_id for r in records}


def test_annotate_new_row_deterministic():
    step_a = load_step_a_annotations(ROOT / STAGE_A_V2_STEP_A_PATH)
    selection = {
        str(row["stage_a_id"]): row
        for row in load_jsonl(ROOT / STAGE_A_V2_SELECTION_PATH)
    }
    sample = next(r for r in step_a if r.stage_a_id == "sa_0265")
    first, _ = annotate_new_row(sample, selection[sample.stage_a_id])
    second, _ = annotate_new_row(sample, selection[sample.stage_a_id])
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert any(
        d.implicit_resolution is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
        for d in first.anchor_decisions
    )
    assert first.dependencies == ()


def test_environmental_h5_positive_rows_are_defensible(step_b_batch):
    records, _report = step_b_batch
    env_h5 = [
        r
        for r in records
        if r.final_bucket == "Environmental"
        and any(
            d.implicit_resolution is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
            for d in r.anchor_decisions
        )
    ]
    assert {r.stage_a_id for r in env_h5} == {"sa_0229", "sa_0248", "sa_0258"}


def test_mixed_implicit_rows_all_h5_positive(step_b_batch):
    records, _report = step_b_batch
    implicit = [r for r in records if r.final_bucket == "MIXED_IMPLICIT"]
    assert len(implicit) == 96
    zero = [
        r.stage_a_id
        for r in implicit
        if not any(
            d.implicit_resolution is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
            for d in r.anchor_decisions
        )
    ]
    assert zero == [], f"MIXED_IMPLICIT rows without H5+: {zero}"


def test_h7_family_row_percentages_use_h7_positive_denominator(step_b_batch):
    _records, report = step_b_batch
    h7_positive = report["H_h7_positive_examples"]
    row_counts = report["H_counts_by_family_rows"]
    percentages = report["H_family_row_percentages"]
    assert h7_positive == 79
    for label, count in row_counts.items():
        assert percentages[label] == round(100.0 * count / h7_positive, 2)
