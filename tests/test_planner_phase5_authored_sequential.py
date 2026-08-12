"""Tests for authored Stage-A MIXED_SEQUENTIAL candidate scaffold."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiergraph.planner.authored_sequential import (
    AUTHORED_SEQUENTIAL_CANDIDATES,
    AuthoredReviewStatus,
    AuthoredSequentialCandidate,
    AuthoredSequentialReviewSession,
    EXPECTED_AUTHORED_SEQUENTIAL_COUNT,
    default_authored_sequential_candidates,
    load_authored_sequential_candidates,
    parse_authored_sequential_review_command,
    parse_dependency_edge,
    validate_authored_sequential_candidate_set,
    validate_dependency_spec,
    write_authored_sequential_candidates,
)
from tiergraph.planner.corpus import normalize_query_key
from tiergraph.planner.operator_io import is_h7_pair_eligible
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = (
    ROOT / "dataset" / "planner" / "stage_a_authored_sequential_candidates.jsonl"
)
TRAIN_PATH = ROOT / "dataset" / "training_data.json"


def test_schema_validation_of_canonical_and_checked_in_file():
    canonical = default_authored_sequential_candidates()
    assert len(canonical) == EXPECTED_AUTHORED_SEQUENTIAL_COUNT
    assert CANDIDATES_PATH.is_file()
    loaded = load_authored_sequential_candidates(CANDIDATES_PATH)
    validate_authored_sequential_candidate_set(loaded, train_path=TRAIN_PATH)
    assert [item.candidate_id for item in loaded] == [
        item.candidate_id for item in canonical
    ]
    for item in loaded:
        assert item.source_kind == "authored_stage_a"
        assert item.planner_bucket == "MIXED_SEQUENTIAL"
        assert item.review_status is AuthoredReviewStatus.UNREVIEWED
        assert item.template_group
        assert item.semantic_group
        assert item.dependency_family
        assert item.intended_operations
        assert item.intended_dependency_edges
        assert item.intended_typed_values
        assert item.intended_personal_requirement
        assert item.intended_environmental_requirement
        assert item.personal_necessity_reason
        assert item.environmental_necessity_reason


def test_duplicate_candidate_id_and_query_rejected():
    base = AUTHORED_SEQUENTIAL_CANDIDATES[0]
    dup_id = base.model_copy(
        update={"candidate_id": "auth_seq_001", "query": "Another unique query?"}
    )
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        validate_authored_sequential_candidate_set(
            (AUTHORED_SEQUENTIAL_CANDIDATES[0], dup_id),
            train_path=None,
            expected_count=2,
        )
    dup_query = base.model_copy(
        update={
            "candidate_id": "auth_seq_099",
            "query": "  " + base.query.upper() + "  ",
        }
    )
    with pytest.raises(ValueError, match="duplicate normalized query"):
        validate_authored_sequential_candidate_set(
            (AUTHORED_SEQUENTIAL_CANDIDATES[0], dup_query),
            train_path=None,
            expected_count=2,
        )


def test_deterministic_ordering():
    shuffled = list(reversed(AUTHORED_SEQUENTIAL_CANDIDATES))
    path = Path("unused")
    session_a = AuthoredSequentialReviewSession(
        shuffled,
        reviews_path=path.with_name("a.jsonl"),
        existing_reviews=(),
        train_path=TRAIN_PATH,
    )
    session_b = AuthoredSequentialReviewSession(
        AUTHORED_SEQUENTIAL_CANDIDATES,
        reviews_path=path.with_name("b.jsonl"),
        existing_reviews=(),
        train_path=TRAIN_PATH,
    )
    assert [item.candidate_id for item in session_a.candidates] == [
        item.candidate_id for item in session_b.candidates
    ]
    assert session_a.candidates[0].candidate_id == "auth_seq_001"
    assert session_a.candidates[-1].candidate_id == "auth_seq_020"


def test_valid_dependency_edges():
    for item in AUTHORED_SEQUENTIAL_CANDIDATES:
        assert item.intended_dependency_edges
        for edge, typed in zip(
            item.intended_dependency_edges,
            item.intended_typed_values,
            strict=True,
        ):
            source, target = parse_dependency_edge(edge)
            assert is_h7_pair_eligible(source, target)
            assert typed
        validate_dependency_spec(
            operations=item.intended_operations,
            edges=item.intended_dependency_edges,
            typed_values=item.intended_typed_values,
        )


def test_invalid_typed_edge_rejection():
    with pytest.raises(ValueError, match="not allowed under OPERATOR_IO_CONTRACT_V1"):
        parse_dependency_edge("RETRIEVE_PERSONAL -> LOCATE_ENVIRONMENTAL")
    with pytest.raises(ValueError, match="FUSE cannot appear|unsupported"):
        parse_dependency_edge("IDENTIFY_ENVIRONMENTAL -> FUSE")
    with pytest.raises(ValueError, match="does not match V1 transfer"):
        validate_dependency_spec(
            operations=("RESOLVE_PERSONAL", "LOCATE_ENVIRONMENTAL"),
            edges=("RESOLVE_PERSONAL -> LOCATE_ENVIRONMENTAL",),
            typed_values=("LOCATION",),
        )


def test_no_fusion_only_candidate_accepted_by_validation():
    with pytest.raises(ValidationError):
        AuthoredSequentialCandidate(
            candidate_id="auth_seq_099",
            query="What medication is this and does it conflict with what I take?",
            source_kind="authored_stage_a",
            planner_bucket="MIXED_SEQUENTIAL",
            template_group="fusion_only_bad",
            semantic_group="fusion_only_bad",
            authoring_reason="invalid fuse-only",
            intended_personal_requirement="my medications",
            intended_environmental_requirement="identify this medication",
            personal_necessity_reason="personal meds list",
            environmental_necessity_reason="observed medication",
            intended_operations=(
                "IDENTIFY_ENVIRONMENTAL",
                "RETRIEVE_PERSONAL",
            ),
            intended_dependency_edges=(),
            intended_typed_values=(),
            dependency_family="invalid_fusion_only",
        )
    with pytest.raises(ValueError, match="at least one non-fusion dependency edge"):
        validate_dependency_spec(
            operations=("IDENTIFY_ENVIRONMENTAL", "RETRIEVE_PERSONAL"),
            edges=(),
            typed_values=(),
        )


def test_resume_behavior(tmp_path):
    reviews_path = tmp_path / "reviews.jsonl"
    session = AuthoredSequentialReviewSession(
        AUTHORED_SEQUENTIAL_CANDIDATES,
        reviews_path=reviews_path,
        existing_reviews=(),
        train_path=TRAIN_PATH,
    )
    first = session.current()
    assert first is not None
    session.apply_choice("1")
    second = session.current()
    assert second is not None
    session.apply_choice("2")

    resumed = AuthoredSequentialReviewSession(
        AUTHORED_SEQUENTIAL_CANDIDATES,
        reviews_path=reviews_path,
        train_path=TRAIN_PATH,
    )
    assert resumed.summary()["reviewed"] == 2
    assert resumed.summary()["ACCEPT"] == 1
    assert resumed.summary()["REJECT"] == 1
    assert resumed.current() is not None
    assert resumed.current().candidate_id not in {
        first.candidate_id,
        second.candidate_id,
    }


def test_back_undo_behavior(tmp_path):
    reviews_path = tmp_path / "reviews.jsonl"
    session = AuthoredSequentialReviewSession(
        AUTHORED_SEQUENTIAL_CANDIDATES,
        reviews_path=reviews_path,
        existing_reviews=(),
        train_path=TRAIN_PATH,
    )
    first = session.current()
    assert first is not None
    session.apply_choice("1")
    assert session.reviews[0].review_status is AuthoredReviewStatus.ACCEPT
    back_to = session.back()
    assert back_to is not None
    assert back_to.candidate_id == first.candidate_id
    assert first.candidate_id not in {item.candidate_id for item in session.reviews}
    session.apply_choice("2")
    assert session.reviews[0].review_status is AuthoredReviewStatus.REJECT


def test_invalid_review_value(tmp_path):
    reviews_path = tmp_path / "reviews.jsonl"
    session = AuthoredSequentialReviewSession(
        AUTHORED_SEQUENTIAL_CANDIDATES,
        reviews_path=reviews_path,
        existing_reviews=(),
        train_path=TRAIN_PATH,
    )
    with pytest.raises(ValueError, match="invalid review value"):
        session.apply_choice("9")
    with pytest.raises(ValueError, match="invalid review value|UNREVIEWED"):
        session.apply_status(AuthoredReviewStatus.UNREVIEWED)
    with pytest.raises(ValueError, match="unknown command"):
        parse_authored_sequential_review_command("x")


def test_candidate_metadata_preservation(tmp_path):
    reviews_path = tmp_path / "reviews.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    write_authored_sequential_candidates(
        candidates_path, AUTHORED_SEQUENTIAL_CANDIDATES
    )
    before = candidates_path.read_text(encoding="utf-8")

    session = AuthoredSequentialReviewSession(
        load_authored_sequential_candidates(candidates_path),
        reviews_path=reviews_path,
        existing_reviews=(),
        train_path=TRAIN_PATH,
    )
    candidate = session.current()
    assert candidate is not None
    record = session.apply_choice("1")
    assert candidates_path.read_text(encoding="utf-8") == before
    assert record.candidate_id == candidate.candidate_id
    assert record.query == candidate.query
    assert record.template_group == candidate.template_group
    assert record.semantic_group == candidate.semantic_group
    assert record.dependency_family == candidate.dependency_family
    assert record.intended_operations == candidate.intended_operations
    assert record.intended_dependency_edges == candidate.intended_dependency_edges
    assert record.intended_typed_values == candidate.intended_typed_values
    assert record.authoring_reason == candidate.authoring_reason
    assert (
        record.intended_personal_requirement
        == candidate.intended_personal_requirement
    )
    assert (
        record.intended_environmental_requirement
        == candidate.intended_environmental_requirement
    )
    assert record.personal_necessity_reason == candidate.personal_necessity_reason
    assert (
        record.environmental_necessity_reason
        == candidate.environmental_necessity_reason
    )
    assert record.source_kind == "authored_stage_a"
    assert record.planner_bucket == "MIXED_SEQUENTIAL"
    reloaded = load_authored_sequential_candidates(candidates_path)
    assert all(
        item.review_status is AuthoredReviewStatus.UNREVIEWED for item in reloaded
    )


def test_not_injected_into_training_data():
    train = {
        normalize_query_key(row["query"])
        for row in json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    }
    for item in AUTHORED_SEQUENTIAL_CANDIDATES:
        assert normalize_query_key(item.query) not in train


def test_choice_mappings():
    assert parse_authored_sequential_review_command("1") == (
        "assign",
        AuthoredReviewStatus.ACCEPT,
    )
    assert parse_authored_sequential_review_command("2") == (
        "assign",
        AuthoredReviewStatus.REJECT,
    )
    assert parse_authored_sequential_review_command("s")[0] == "skip"
    assert parse_authored_sequential_review_command("b")[0] == "back"
    assert parse_authored_sequential_review_command("q")[0] == "quit"


def test_dependency_family_coverage_counts():
    counts: dict[str, int] = {}
    for item in AUTHORED_SEQUENTIAL_CANDIDATES:
        counts[item.dependency_family] = counts.get(item.dependency_family, 0) + 1
    assert counts["resolve_to_identify"] == 5
    assert counts["identify_to_locate"] == 1
    assert counts["resolve_identify_locate"] == 3
    assert counts["resolve_to_locate"] == 2
    assert counts["resolve_locate_navigate"] == 7
    assert counts["resolve_to_describe"] == 2
    assert sum(counts.values()) == 20
    # Motivation-only / incidental patterns removed.
    assert "locate_to_navigate" not in counts
    assert "resolve_to_retrieve" not in counts


def test_necessity_fields_required():
    base = AUTHORED_SEQUENTIAL_CANDIDATES[0]
    with pytest.raises(ValidationError):
        base.model_copy(update={"personal_necessity_reason": "   "})
    with pytest.raises(ValidationError):
        base.model_copy(update={"environmental_necessity_reason": ""})
