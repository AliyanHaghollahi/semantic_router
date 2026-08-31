"""Tests for Phase-5 Stage-A Step-B annotation infrastructure (H5/H6/H7)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tiergraph.enums import OperatorType, QueryType
from tiergraph.planner.annotation_step_a import (
    DEFAULT_STEP_A_ANNOTATIONS_PATH,
    EXPECTED_STAGE_A_COUNT,
    StageAStepAAnnotation,
    StepAAnchor,
    StepAOperation,
    StepAStatus,
    fingerprint_file,
    load_step_a_annotations,
)
from tiergraph.planner.annotation_step_b import (
    DEFAULT_STEP_B_ANNOTATIONS_PATH,
    StageAStepBAnnotation,
    StepBAnnotationSession,
    StepBDependency,
    StepBStatus,
    add_h7_dependency,
    demo_step_b_interaction,
    ensure_step_b_annotations_initialized,
    h7_forms_cycle,
    initialize_step_b_annotations_from_step_a,
    initialize_step_b_from_step_a,
    load_step_b_annotations,
    mark_step_b_complete,
    parse_step_b_command,
    remove_h7_dependency,
    set_anchor_h5,
    set_anchor_h6,
    validate_step_b_corpus,
    write_step_b_annotations,
)
from tiergraph.planner.annotations import ImplicitResolution


ROOT = Path(__file__).resolve().parent.parent
STEP_A_PATH = ROOT / DEFAULT_STEP_A_ANNOTATIONS_PATH
STEP_B_PATH = ROOT / DEFAULT_STEP_B_ANNOTATIONS_PATH


@pytest.fixture(scope="module")
def step_a_records():
    records = load_step_a_annotations(STEP_A_PATH)
    assert len(records) == EXPECTED_STAGE_A_COUNT
    assert all(item.step_a_status is StepAStatus.COMPLETE for item in records)
    return records


def _make_step_a(
    *,
    stage_a_id: str = "sa_demo_0001",
    query: str = "Where is my gate and how do I get there?",
    final_bucket: str = "MIXED_SEQUENTIAL",
    operations: tuple[StepAOperation, ...] | None = None,
    anchors: tuple[StepAAnchor, ...] | None = None,
) -> StageAStepAAnnotation:
    if operations is None:
        operations = (
            StepAOperation(
                operation_index=0,
                text="Where is my gate",
                char_start=0,
                char_end=16,
                operator_type=OperatorType.LOCATE_ENVIRONMENTAL,
            ),
            StepAOperation(
                operation_index=1,
                text="how do I get there",
                char_start=21,
                char_end=39,
                operator_type=OperatorType.NAVIGATE_TO,
            ),
        )
    if anchors is None:
        anchors = (
            StepAAnchor(
                anchor_index=0, text="my gate", char_start=9, char_end=16
            ),
            StepAAnchor(
                anchor_index=1, text="there", char_start=34, char_end=39
            ),
        )
    return StageAStepAAnnotation(
        stage_a_id=stage_a_id,
        source_id="demo_src",
        candidate_id=None,
        query=query,
        final_bucket=final_bucket,
        source_kind="demo",
        semantic_group="demo",
        template_group="demo",
        provenance={"kind": "demo"},
        derived_query_type=QueryType.MIXED,
        operations=operations,
        anchors=anchors,
        step_a_status=StepAStatus.COMPLETE,
    )


def test_initialize_from_frozen_step_a(step_a_records, tmp_path):
    before = fingerprint_file(STEP_A_PATH)
    records = initialize_step_b_annotations_from_step_a(step_a_records)
    assert len(records) == EXPECTED_STAGE_A_COUNT
    assert all(item.step_b_status is StepBStatus.UNREVIEWED for item in records)
    assert all(item.dependencies == () for item in records)
    assert [item.stage_a_id for item in records] == sorted(
        item.stage_a_id for item in step_a_records
    )
    for step_a, step_b in zip(
        sorted(step_a_records, key=lambda r: r.stage_a_id),
        records,
        strict=True,
    ):
        assert step_b.stage_a_id == step_a.stage_a_id
        assert step_b.query == step_a.query
        assert step_b.final_bucket == step_a.final_bucket
        assert step_b.source_id == step_a.source_id
        assert step_b.candidate_id == step_a.candidate_id
        assert step_b.n_operations == len(step_a.operations)
        assert step_b.n_anchors == len(step_a.anchors)
        assert step_b.operation_types == tuple(
            op.operator_type.value for op in step_a.operations
        )
        for decision, anchor in zip(
            step_b.anchor_decisions, step_a.anchors, strict=True
        ):
            assert decision.anchor_index == anchor.anchor_index
            assert decision.text == anchor.text
            assert decision.implicit_resolution is ImplicitResolution.NONE
            assert decision.owner_operation_index is None
    path = tmp_path / "step_b.jsonl"
    write_step_b_annotations(path, records)
    reloaded = load_step_b_annotations(path)
    assert [item.stage_a_id for item in reloaded] == [
        item.stage_a_id for item in records
    ]
    assert fingerprint_file(STEP_A_PATH) == before


def test_step_a_not_mutated_by_init_or_session(step_a_records, tmp_path):
    before = fingerprint_file(STEP_A_PATH)
    step_a_bytes = STEP_A_PATH.read_bytes()
    step_b_path = tmp_path / "step_b.jsonl"
    records = ensure_step_b_annotations_initialized(
        step_a_path=STEP_A_PATH,
        step_b_path=step_b_path,
    )
    session = StepBAnnotationSession(
        records,
        step_a_records=step_a_records,
        step_b_path=step_b_path,
        step_a_path=STEP_A_PATH,
    )
    current = session.current()
    assert current is not None
    if current.n_anchors > 0:
        session.set_h5(0, ImplicitResolution.NONE)
        session.set_h6(0, 0)
    session.save()
    assert STEP_A_PATH.read_bytes() == step_a_bytes
    assert fingerprint_file(STEP_A_PATH) == before


def test_h5_values_and_enum_reuse():
    step_a = _make_step_a()
    record = initialize_step_b_from_step_a(step_a)
    updated = set_anchor_h5(
        record, 0, ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
    )
    assert (
        updated.anchor_decisions[0].implicit_resolution
        is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
    )
    updated = set_anchor_h5(updated, 1, "NONE")
    assert updated.anchor_decisions[1].implicit_resolution is ImplicitResolution.NONE
    with pytest.raises(ValueError, match="invalid H5"):
        set_anchor_h5(record, 0, "NOT_A_CLASS")


def test_h6_owner_validation():
    step_a = _make_step_a()
    record = initialize_step_b_from_step_a(step_a)
    updated = set_anchor_h6(record, 0, 0)
    assert updated.anchor_decisions[0].owner_operation_index == 0
    updated = set_anchor_h6(updated, 1, 1)
    assert updated.anchor_decisions[1].owner_operation_index == 1
    with pytest.raises(ValueError, match="owner_operation_index out of range"):
        set_anchor_h6(record, 0, 99)
    with pytest.raises(ValueError, match="anchor_index out of range"):
        set_anchor_h6(record, 99, 0)


def test_h7_edge_validation_and_illegal_typed_rejection():
    step_a = _make_step_a()
    record = initialize_step_b_from_step_a(step_a)
    ok = add_h7_dependency(record, 0, 1)
    assert len(ok.dependencies) == 1
    assert ok.dependencies[0].source_operation_index == 0
    assert ok.dependencies[0].target_operation_index == 1
    with pytest.raises(ValueError, match="illegal typed H7"):
        add_h7_dependency(record, 1, 0)
    with pytest.raises((ValueError, ValidationError), match="self-loop"):
        add_h7_dependency(record, 0, 0)
    with pytest.raises(ValueError, match="duplicate H7"):
        add_h7_dependency(ok, 0, 1)


def test_h7_cycle_rejection(monkeypatch):
    assert h7_forms_cycle(2, [(0, 1), (1, 0)]) is True
    assert h7_forms_cycle(2, [(0, 1)]) is False
    assert h7_forms_cycle(3, [(0, 1), (1, 2), (2, 0)]) is True

    # Closing NAVIGATE -> IDENTIFY is typed-illegal (and would cycle if allowed).
    with pytest.raises((ValueError, ValidationError), match="illegal typed H7|cycle"):
        StageAStepBAnnotation(
            stage_a_id="cycle_demo",
            source_id="x",
            query="q",
            final_bucket="MIXED_SEQUENTIAL",
            n_operations=3,
            n_anchors=0,
            operation_types=(
                OperatorType.IDENTIFY_ENVIRONMENTAL.value,
                OperatorType.LOCATE_ENVIRONMENTAL.value,
                OperatorType.NAVIGATE_TO.value,
            ),
            anchor_decisions=(),
            dependencies=(
                StepBDependency(
                    source_operation_index=0, target_operation_index=1
                ),
                StepBDependency(
                    source_operation_index=1, target_operation_index=2
                ),
                StepBDependency(
                    source_operation_index=2, target_operation_index=0
                ),
            ),
            step_b_status=StepBStatus.UNREVIEWED,
        )

    # Force typed-eligible edges so the DAG cycle check is exercised.
    import tiergraph.planner.annotation_step_b as step_b_mod

    monkeypatch.setattr(step_b_mod, "is_h7_pair_eligible", lambda *_a, **_k: True)
    cyclic = StageAStepBAnnotation(
        stage_a_id="cycle_forced",
        source_id="x",
        query="q",
        final_bucket="MIXED_SEQUENTIAL",
        n_operations=2,
        n_anchors=0,
        operation_types=(
            OperatorType.LOCATE_ENVIRONMENTAL.value,
            OperatorType.NAVIGATE_TO.value,
        ),
        anchor_decisions=(),
        dependencies=(),
        step_b_status=StepBStatus.UNREVIEWED,
    )
    with pytest.raises(ValueError, match="cycle"):
        cyclic.model_copy(
            update={
                "dependencies": (
                    StepBDependency(
                        source_operation_index=0, target_operation_index=1
                    ),
                    StepBDependency(
                        source_operation_index=1, target_operation_index=0
                    ),
                )
            }
        )


def test_one_operation_h7_must_be_empty():
    step_a = StageAStepAAnnotation(
        stage_a_id="sa_demo_one_op",
        source_id="demo_src",
        candidate_id=None,
        query="Where is my gate?",
        final_bucket="Environmental",
        source_kind="demo",
        semantic_group="demo",
        template_group="demo",
        provenance={"kind": "demo"},
        derived_query_type=QueryType.ENVIRONMENTAL,
        operations=(
            StepAOperation(
                operation_index=0,
                text="Where is my gate",
                char_start=0,
                char_end=16,
                operator_type=OperatorType.LOCATE_ENVIRONMENTAL,
            ),
        ),
        anchors=(
            StepAAnchor(
                anchor_index=0, text="my gate", char_start=9, char_end=16
            ),
        ),
        step_a_status=StepAStatus.COMPLETE,
    )
    record = initialize_step_b_from_step_a(step_a)
    assert record.n_operations == 1
    with pytest.raises(ValueError, match="H7 must be empty"):
        add_h7_dependency(record, 0, 0)
    record = set_anchor_h5(
        record, 0, ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
    )
    record = set_anchor_h6(record, 0, 0)
    complete = mark_step_b_complete(record)
    assert complete.step_b_status is StepBStatus.COMPLETE
    assert complete.dependencies == ()


def test_complete_requires_h6_owners():
    step_a = _make_step_a()
    record = initialize_step_b_from_step_a(step_a)
    with pytest.raises((ValueError, ValidationError), match="H6 owner"):
        mark_step_b_complete(record)
    record = set_anchor_h6(record, 0, 0)
    record = set_anchor_h6(record, 1, 1)
    record = add_h7_dependency(record, 0, 1)
    complete = mark_step_b_complete(record)
    assert complete.step_b_status is StepBStatus.COMPLETE


def test_persistence_and_resume(step_a_records, tmp_path):
    step_b_path = tmp_path / "step_b.jsonl"
    records = ensure_step_b_annotations_initialized(
        step_a_path=STEP_A_PATH,
        step_b_path=step_b_path,
    )
    session = StepBAnnotationSession(
        records,
        step_a_records=step_a_records,
        step_b_path=step_b_path,
        step_a_path=STEP_A_PATH,
    )
    first_id = session.current().stage_a_id
    session.skip(persist=True)
    second_id = session.current().stage_a_id
    assert second_id != first_id
    session.save()

    reloaded = load_step_b_annotations(step_b_path)
    session2 = StepBAnnotationSession(
        reloaded,
        step_a_records=step_a_records,
        step_b_path=step_b_path,
        step_a_path=STEP_A_PATH,
    )
    assert session2.current().stage_a_id == first_id

    cur = session2.current()
    for decision in cur.anchor_decisions:
        session2.set_h6(decision.anchor_index, 0, persist=False)
    session2.mark_complete(persist=True)
    session2.skip(persist=True)
    assert session2.current() is not None
    assert session2.current().stage_a_id != first_id

    reloaded2 = load_step_b_annotations(step_b_path)
    by_id = {item.stage_a_id: item for item in reloaded2}
    assert by_id[first_id].step_b_status is StepBStatus.COMPLETE
    session3 = StepBAnnotationSession(
        reloaded2,
        step_a_records=step_a_records,
        step_b_path=step_b_path,
        step_a_path=STEP_A_PATH,
    )
    assert session3.current().stage_a_id != first_id


def test_stable_stage_a_id_linkage(step_a_records, tmp_path):
    step_b_path = tmp_path / "step_b.jsonl"
    records = ensure_step_b_annotations_initialized(
        step_a_path=STEP_A_PATH,
        step_b_path=step_b_path,
    )
    errors = validate_step_b_corpus(
        records,
        step_a_records=step_a_records,
        step_a_path=STEP_A_PATH,
        require_all_complete=False,
        step_a_fingerprint=fingerprint_file(STEP_A_PATH),
    )
    assert errors == []
    assert {r.stage_a_id for r in records} == {
        r.stage_a_id for r in step_a_records
    }


def test_checked_in_step_b_init_and_validation(step_a_records, tmp_path):
    before = fingerprint_file(STEP_A_PATH)
    step_b_before = fingerprint_file(STEP_B_PATH)

    # Init behavior: missing Step-B path creates UNREVIEWED shells.
    temp_step_b = tmp_path / "step_b_init.jsonl"
    assert not temp_step_b.exists()
    initialized = ensure_step_b_annotations_initialized(
        step_a_path=STEP_A_PATH,
        step_b_path=temp_step_b,
    )
    assert len(initialized) == EXPECTED_STAGE_A_COUNT
    assert all(item.step_b_status is StepBStatus.UNREVIEWED for item in initialized)

    # Checked-in corpus is the completed human-reviewed Step B.
    checked_in = load_step_b_annotations(STEP_B_PATH)
    assert len(checked_in) == EXPECTED_STAGE_A_COUNT
    assert sum(1 for item in checked_in if item.step_b_status is StepBStatus.COMPLETE) == 120
    assert sum(1 for item in checked_in if item.step_b_status is StepBStatus.UNREVIEWED) == 0
    errors = validate_step_b_corpus(
        checked_in,
        step_a_records=step_a_records,
        step_a_path=STEP_A_PATH,
        require_all_complete=True,
        step_a_fingerprint=before,
    )
    assert errors == []
    assert fingerprint_file(STEP_A_PATH) == before
    assert fingerprint_file(STEP_B_PATH) == step_b_before


def test_command_parsing():
    assert parse_step_b_command("i")[0] == "set_h5"
    assert parse_step_b_command("o")[0] == "set_h6"
    assert parse_step_b_command("d")[0] == "add_dependency"
    assert parse_step_b_command("c")[0] == "complete"
    assert parse_step_b_command("q")[0] == "quit"
    with pytest.raises(ValueError):
        parse_step_b_command("zzz")


def test_demo_step_b_interaction():
    text = demo_step_b_interaction()
    assert "IMPLICIT_RESOLVE_PERSONAL" in text
    assert "LOCATE_ENVIRONMENTAL" in text
    assert "NAVIGATE_TO" in text
    assert "0(LOCATE_ENVIRONMENTAL) -> 1(NAVIGATE_TO)" in text
    assert "my gate" in text


def test_remove_h7_dependency():
    step_a = _make_step_a()
    record = initialize_step_b_from_step_a(step_a)
    record = add_h7_dependency(record, 0, 1)
    record = remove_h7_dependency(record, 0, 1)
    assert record.dependencies == ()
    with pytest.raises(ValueError, match="not found"):
        remove_h7_dependency(record, 0, 1)


def test_refuse_init_when_step_a_incomplete():
    incomplete = _make_step_a()
    incomplete = incomplete.model_copy(
        update={
            "step_a_status": StepAStatus.UNREVIEWED,
            "operations": (),
            "anchors": (),
        }
    )
    with pytest.raises(ValueError, match="COMPLETE"):
        initialize_step_b_from_step_a(incomplete)
