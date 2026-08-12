"""Tests for Phase-5 Stage-A Step-A annotation infrastructure."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tiergraph.enums import OperatorType, QueryType
from tiergraph.planner.align import TokenCharSpan
from tiergraph.planner.annotation_step_a import (
    DEFAULT_FROZEN_SELECTION_PATH,
    DEFAULT_STEP_A_ANNOTATIONS_PATH,
    EXPECTED_STAGE_A_COUNT,
    H4_ANCHOR_GUIDANCE,
    RESOLVE_PERSONAL_IMPLICIT_WARNING,
    StageAStepAAnnotation,
    StepAAnnotationSession,
    StepAStatus,
    create_anchor_from_substring,
    create_operation_from_substring,
    demo_step_a_interaction,
    derive_query_type,
    ensure_step_a_annotations_initialized,
    find_substring_occurrences,
    fingerprint_file,
    format_operator_help,
    initialize_step_a_annotations_from_selection,
    load_step_a_annotations,
    parse_step_a_command,
    reindex_operations,
    spans_overlap,
    validate_step_a_corpus,
    write_step_a_annotations,
)
from tiergraph.planner.stage_a_selection import load_jsonl


ROOT = Path(__file__).resolve().parent.parent
SELECTION_PATH = ROOT / DEFAULT_FROZEN_SELECTION_PATH
ANNOTATIONS_PATH = ROOT / DEFAULT_STEP_A_ANNOTATIONS_PATH


def _tokens_for_query(query: str) -> tuple[TokenCharSpan, ...]:
    tokens: list[TokenCharSpan] = [
        TokenCharSpan(None, None, is_special=True, is_padding=False),
    ]
    for index, _char in enumerate(query):
        if index % 2 == 1:
            continue
        end = min(index + 2, len(query))
        tokens.append(
            TokenCharSpan(index, end, is_special=False, is_padding=False)
        )
    tokens.append(TokenCharSpan(None, None, is_special=True, is_padding=False))
    return tuple(tokens)


@pytest.fixture(scope="module")
def selection_rows():
    assert SELECTION_PATH.is_file()
    rows = load_jsonl(SELECTION_PATH)
    assert len(rows) == EXPECTED_STAGE_A_COUNT
    return rows


def test_120_row_initialization_from_frozen_manifest(selection_rows, tmp_path):
    records = initialize_step_a_annotations_from_selection(selection_rows)
    assert len(records) == EXPECTED_STAGE_A_COUNT
    assert all(item.step_a_status is StepAStatus.UNREVIEWED for item in records)
    assert all(item.operations == () for item in records)
    assert all(item.anchors == () for item in records)
    path = tmp_path / "ann.jsonl"
    write_step_a_annotations(path, records)
    reloaded = load_step_a_annotations(path)
    assert [item.stage_a_id for item in reloaded] == [
        item.stage_a_id for item in records
    ]


def test_frozen_metadata_preservation(selection_rows):
    records = initialize_step_a_annotations_from_selection(selection_rows)
    by_id = {item.stage_a_id: item for item in records}
    for row in selection_rows:
        item = by_id[row["stage_a_id"]]
        assert item.query == row["query"]
        assert item.final_bucket == row["final_bucket"]
        assert item.source_kind == row["source_kind"]
        assert item.semantic_group == row["semantic_group"]
        assert item.template_group == row["template_group"]
        assert item.source_id == row.get("source_id")
        assert item.candidate_id == row.get("candidate_id")
        assert item.provenance == row.get("provenance")


def test_h1_derivation_from_final_bucket():
    assert derive_query_type("Personal") is QueryType.PERSONAL
    assert derive_query_type("Environmental") is QueryType.ENVIRONMENTAL
    assert derive_query_type("MIXED_IMPLICIT") is QueryType.MIXED
    assert derive_query_type("MIXED_PARALLEL") is QueryType.MIXED
    assert derive_query_type("MIXED_SEQUENTIAL") is QueryType.MIXED
    with pytest.raises(ValueError):
        derive_query_type("NOT_A_BUCKET")


def test_substring_to_offsets_and_repeated_occurrence():
    query = "this gate and that gate"
    occ = find_substring_occurrences(query, "gate")
    assert occ == [(5, 9), (19, 23)]
    op0 = create_operation_from_substring(
        query, "gate", OperatorType.IDENTIFY_ENVIRONMENTAL, occurrence=0
    )
    assert op0.char_start == 5 and op0.text == "gate"
    op1 = create_operation_from_substring(
        query,
        "gate",
        OperatorType.LOCATE_ENVIRONMENTAL,
        occurrence=1,
        existing_operations=(op0,),
    )
    assert op1.char_start == 19
    with pytest.raises(ValueError, match="out of range"):
        create_operation_from_substring(
            query, "gate", OperatorType.IDENTIFY_ENVIRONMENTAL, occurrence=2
        )


def test_operation_creation_and_invalid_operator():
    query = "Where is my gate?"
    op = create_operation_from_substring(
        query, "Where is my gate", OperatorType.LOCATE_ENVIRONMENTAL
    )
    assert op.operator_type is OperatorType.LOCATE_ENVIRONMENTAL
    with pytest.raises(ValueError, match="invalid OperatorType|FUSE"):
        create_operation_from_substring(query, "Where is my gate", "FUSE")
    with pytest.raises(ValueError, match="invalid OperatorType"):
        create_operation_from_substring(query, "Where is my gate", "NOT_AN_OP")


def test_operation_overlap_rejection():
    query = "Where is my gate and how do I get there?"
    op1 = create_operation_from_substring(
        query, "Where is my gate", OperatorType.LOCATE_ENVIRONMENTAL
    )
    with pytest.raises(ValueError, match="must not overlap"):
        create_operation_from_substring(
            query,
            "my gate and how",
            OperatorType.NAVIGATE_TO,
            existing_operations=(op1,),
        )


def test_anchor_creation_and_overlap_rejection():
    query = "Where is my gate and how do I get there?"
    a1 = create_anchor_from_substring(query, "my gate")
    assert a1.text == "my gate"
    with pytest.raises(ValueError, match="must not overlap"):
        create_anchor_from_substring(
            query, "gate and", existing_anchors=(a1,)
        )


def test_operation_anchor_cross_head_overlap_allowed(selection_rows):
    row = selection_rows[0]
    query = "Where is my gate?"
    op = create_operation_from_substring(
        query, "Where is my gate", OperatorType.LOCATE_ENVIRONMENTAL
    )
    anchor = create_anchor_from_substring(query, "my gate")
    assert spans_overlap(op.char_start, op.char_end, anchor.char_start, anchor.char_end)
    record = StageAStepAAnnotation(
        stage_a_id="demo_cross",
        source_id="demo",
        query=query,
        final_bucket="MIXED_SEQUENTIAL",
        source_kind="demo",
        semantic_group="g",
        template_group="t",
        provenance={},
        derived_query_type=QueryType.MIXED,
        operations=reindex_operations((op,)),
        anchors=(anchor,),
        step_a_status=StepAStatus.COMPLETE,
    )
    assert record.operations[0].text == "Where is my gate"
    assert record.anchors[0].text == "my gate"


def test_zero_anchor_complete_allowed():
    query = "What time is it?"
    op = create_operation_from_substring(
        query, "What time is it", OperatorType.DESCRIBE_ENVIRONMENT
    )
    record = StageAStepAAnnotation(
        stage_a_id="demo_zero_anchor",
        source_id="demo",
        query=query,
        final_bucket="Environmental",
        source_kind="demo",
        semantic_group="g",
        template_group="t",
        provenance={},
        derived_query_type=QueryType.ENVIRONMENTAL,
        operations=reindex_operations((op,)),
        anchors=(),
        step_a_status=StepAStatus.COMPLETE,
    )
    assert record.anchors == ()


def test_save_resume_and_back_edit(selection_rows, tmp_path):
    records = initialize_step_a_annotations_from_selection(selection_rows)
    ann_path = tmp_path / "ann.jsonl"
    write_step_a_annotations(ann_path, records)
    # Copy frozen selection into tmp so fingerprint stays stable for session.
    sel_path = tmp_path / "selection.jsonl"
    sel_path.write_bytes(SELECTION_PATH.read_bytes())

    session = StepAAnnotationSession(
        load_step_a_annotations(ann_path),
        annotations_path=ann_path,
        selection_path=sel_path,
    )
    current = session.current()
    assert current is not None
    first_id = current.stage_a_id
    # Use a substring that exists in the current query.
    query = current.query
    # Prefer a short stable token present in most queries; fall back to whole query.
    substring = query.split()[0]
    session.add_operation(substring, OperatorType.DESCRIBE_ENVIRONMENT)
    session.mark_complete()
    assert session._records[first_id].step_a_status is StepAStatus.COMPLETE

    resumed = StepAAnnotationSession(
        load_step_a_annotations(ann_path),
        annotations_path=ann_path,
        selection_path=sel_path,
    )
    assert resumed.summary()["COMPLETE"] == 1
    assert resumed.current() is not None
    assert resumed.current().stage_a_id != first_id

    # Back/edit completed example via explicit reopen after seeking.
    resumed._cursor = resumed._order.index(first_id)
    reopened = resumed.reopen_for_edit()
    assert reopened.step_a_status is StepAStatus.UNREVIEWED
    assert reopened.operations  # metadata/ops preserved


def test_metadata_preservation_during_editing(selection_rows, tmp_path):
    records = initialize_step_a_annotations_from_selection(selection_rows)
    ann_path = tmp_path / "ann.jsonl"
    sel_path = tmp_path / "selection.jsonl"
    write_step_a_annotations(ann_path, records)
    sel_path.write_bytes(SELECTION_PATH.read_bytes())
    session = StepAAnnotationSession(
        load_step_a_annotations(ann_path),
        annotations_path=ann_path,
        selection_path=sel_path,
    )
    before = session.current()
    assert before is not None
    meta = (
        before.stage_a_id,
        before.query,
        before.final_bucket,
        before.source_kind,
        before.semantic_group,
        before.template_group,
        before.provenance,
        before.source_id,
        before.candidate_id,
        before.derived_query_type,
    )
    substring = before.query.split()[0]
    after = session.add_operation(substring, OperatorType.IDENTIFY_ENVIRONMENTAL)
    assert (
        after.stage_a_id,
        after.query,
        after.final_bucket,
        after.source_kind,
        after.semantic_group,
        after.template_group,
        after.provenance,
        after.source_id,
        after.candidate_id,
        after.derived_query_type,
    ) == meta


def test_deterministic_ordering(selection_rows):
    records = initialize_step_a_annotations_from_selection(selection_rows)
    ids = [item.stage_a_id for item in records]
    assert ids == sorted(ids)
    assert ids[0] == "sa_0001"
    assert ids[-1] == "sa_0120"


def test_bad_character_span_rejection():
    with pytest.raises((ValueError, ValidationError)):
        StageAStepAAnnotation(
            stage_a_id="bad",
            source_id="x",
            query="hello",
            final_bucket="Personal",
            source_kind="demo",
            semantic_group="g",
            template_group="t",
            provenance={},
            derived_query_type=QueryType.PERSONAL,
            operations=(
                {
                    "operation_index": 0,
                    "text": "hello",
                    "char_start": 0,
                    "char_end": 99,
                    "operator_type": "RETRIEVE_PERSONAL",
                },
            ),
            step_a_status=StepAStatus.COMPLETE,
        )


def test_tokenizer_alignment_validation(selection_rows):
    query = "Where is my gate?"
    op = create_operation_from_substring(
        query, "Where is my gate", OperatorType.LOCATE_ENVIRONMENTAL
    )
    record = StageAStepAAnnotation(
        stage_a_id="sa_align_demo",
        source_id="demo",
        query=query,
        final_bucket="MIXED_SEQUENTIAL",
        source_kind="demo",
        semantic_group="g",
        template_group="t",
        provenance={},
        derived_query_type=QueryType.MIXED,
        operations=reindex_operations((op,)),
        anchors=(),
        step_a_status=StepAStatus.COMPLETE,
    )
    # Build a tiny fake corpus around one frozen row replaced conceptually.
    # Directly exercise alignment helper path via validate on a crafted list
    # is awkward; call check helpers through validate with a one-off factory.
    from tiergraph.planner.annotation_step_a import check_span_alignment

    tokens = _tokens_for_query(query)
    issues = check_span_alignment(
        query,
        [(op.char_start, op.char_end, "operation[0]")],
        tokens,
    )
    assert issues == []


def test_frozen_manifest_not_modified(selection_rows, tmp_path):
    before = fingerprint_file(SELECTION_PATH)
    ann_path = tmp_path / "ann.jsonl"
    records = ensure_step_a_annotations_initialized(
        selection_path=SELECTION_PATH,
        annotations_path=ann_path,
    )
    assert len(records) == EXPECTED_STAGE_A_COUNT
    sel_copy = tmp_path / "selection.jsonl"
    sel_copy.write_bytes(SELECTION_PATH.read_bytes())
    session = StepAAnnotationSession(
        records,
        annotations_path=ann_path,
        selection_path=sel_copy,
    )
    current = session.current()
    assert current is not None
    session.add_operation(current.query.split()[0], OperatorType.DESCRIBE_ENVIRONMENT)
    session.save()
    after = fingerprint_file(SELECTION_PATH)
    assert before == after


def test_command_parsing():
    assert parse_step_a_command("a")[0] == "add_operation"
    assert parse_step_a_command("c")[0] == "complete"
    assert parse_step_a_command("q")[0] == "quit"
    with pytest.raises(ValueError):
        parse_step_a_command("zzz")


def test_complete_requires_operation():
    with pytest.raises((ValueError, ValidationError), match="at least one operation"):
        StageAStepAAnnotation(
            stage_a_id="empty_ops",
            source_id="x",
            query="hello",
            final_bucket="Personal",
            source_kind="demo",
            semantic_group="g",
            template_group="t",
            provenance={},
            derived_query_type=QueryType.PERSONAL,
            operations=(),
            anchors=(),
            step_a_status=StepAStatus.COMPLETE,
        )


def test_checked_in_initialization_and_validation():
    before = fingerprint_file(SELECTION_PATH)
    records = ensure_step_a_annotations_initialized(
        selection_path=SELECTION_PATH,
        annotations_path=ANNOTATIONS_PATH,
    )
    assert len(records) == EXPECTED_STAGE_A_COUNT
    errors = validate_step_a_corpus(
        records,
        selection_path=SELECTION_PATH,
        token_view_factory=_tokens_for_query,
        require_all_complete=False,
    )
    assert errors == []
    assert fingerprint_file(SELECTION_PATH) == before


def test_operator_help_contains_implicit_resolve_warning():
    help_text = format_operator_help()
    assert "RESOLVE_PERSONAL" in help_text
    assert RESOLVE_PERSONAL_IMPLICIT_WARNING in help_text
    assert "do NOT annotate RESOLVE_PERSONAL as an H2 operation" in help_text
    assert H4_ANCHOR_GUIDANCE in help_text
    assert '"my gate"' in help_text or "'my gate'" in help_text


def test_demo_omits_resolve_personal_and_uses_my_gate_anchor():
    text = demo_step_a_interaction()
    assert "my gate" in text
    assert "H4 anchor" in text
    assert "synthesized from H5" in text
    assert "RESOLVE_PERSONAL is intentionally absent" in text
    # Demo H2/H3 ops only; no RESOLVE_PERSONAL operation annotation.
    assert "Human operator: RESOLVE_PERSONAL" not in text
    assert "LOCATE_ENVIRONMENTAL" in text
    assert "NAVIGATE_TO" in text
    assert "Tool stored anchor0:" in text
    assert "'my gate'" in text
    # Preview must not list RESOLVE_PERSONAL as an operation.
    preview_ops = [
        line for line in text.splitlines() if line.strip().startswith("[0]")
        or line.strip().startswith("[1]")
    ]
    assert any("LOCATE_ENVIRONMENTAL" in line for line in preview_ops)
    assert any("NAVIGATE_TO" in line for line in preview_ops)
    assert not any("RESOLVE_PERSONAL" in line for line in preview_ops)
    assert ANNOTATIONS_PATH.is_file()
