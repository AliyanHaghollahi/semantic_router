"""Tests for authored Stage-A MIXED_IMPLICIT candidate scaffold."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiergraph.planner.authored_implicit import (
    AUTHORED_IMPLICIT_CANDIDATES,
    AuthoredImplicitCandidate,
    AuthoredImplicitReview,
    AuthoredImplicitReviewSession,
    AuthoredReviewStatus,
    EXPECTED_AUTHORED_COUNT,
    default_authored_candidates,
    load_authored_candidates,
    load_authored_reviews,
    parse_authored_review_command,
    validate_authored_candidate_set,
    write_authored_candidates,
    write_authored_reviews,
)
from tiergraph.planner.corpus import normalize_query_key


ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = (
    ROOT / "dataset" / "planner" / "stage_a_authored_implicit_candidates.jsonl"
)
TRAIN_PATH = ROOT / "dataset" / "training_data.json"


def test_schema_validation_of_canonical_and_checked_in_file():
    canonical = default_authored_candidates()
    assert len(canonical) == EXPECTED_AUTHORED_COUNT
    assert CANDIDATES_PATH.is_file()
    loaded = load_authored_candidates(CANDIDATES_PATH)
    validate_authored_candidate_set(loaded, train_path=TRAIN_PATH)
    assert [item.candidate_id for item in loaded] == [
        item.candidate_id for item in canonical
    ]
    for item in loaded:
        assert item.source_kind == "authored_stage_a"
        assert item.planner_bucket == "MIXED_IMPLICIT"
        assert item.review_status is AuthoredReviewStatus.UNREVIEWED
        assert item.template_group
        assert item.authoring_reason
        assert item.intended_personal_requirement
        assert item.intended_environmental_requirement


def test_duplicate_candidate_id_and_query_rejected(tmp_path):
    base = AUTHORED_IMPLICIT_CANDIDATES[0]
    dup_id = base.model_copy(
        update={"candidate_id": "auth_imp_001", "query": "Another unique query?"}
    )
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        validate_authored_candidate_set(
            (AUTHORED_IMPLICIT_CANDIDATES[0], dup_id),
            train_path=None,
            expected_count=2,
        )
    dup_query = base.model_copy(
        update={
            "candidate_id": "auth_imp_099",
            "query": "  " + base.query.upper() + "  ",
        }
    )
    with pytest.raises(ValueError, match="duplicate normalized query"):
        validate_authored_candidate_set(
            (AUTHORED_IMPLICIT_CANDIDATES[0], dup_query),
            train_path=None,
            expected_count=2,
        )


def test_deterministic_ordering():
    shuffled = list(reversed(AUTHORED_IMPLICIT_CANDIDATES))
    path = Path("unused")
    session_a = AuthoredImplicitReviewSession(
        shuffled,
        reviews_path=path.with_name("a.jsonl"),
        existing_reviews=(),
        train_path=TRAIN_PATH,
    )
    session_b = AuthoredImplicitReviewSession(
        AUTHORED_IMPLICIT_CANDIDATES,
        reviews_path=path.with_name("b.jsonl"),
        existing_reviews=(),
        train_path=TRAIN_PATH,
    )
    assert [item.candidate_id for item in session_a.candidates] == [
        item.candidate_id for item in session_b.candidates
    ]
    assert session_a.candidates[0].candidate_id == "auth_imp_001"
    assert session_a.candidates[-1].candidate_id == "auth_imp_012"


def test_resume_behavior(tmp_path):
    reviews_path = tmp_path / "reviews.jsonl"
    session = AuthoredImplicitReviewSession(
        AUTHORED_IMPLICIT_CANDIDATES,
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

    resumed = AuthoredImplicitReviewSession(
        AUTHORED_IMPLICIT_CANDIDATES,
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
    session = AuthoredImplicitReviewSession(
        AUTHORED_IMPLICIT_CANDIDATES,
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
    session = AuthoredImplicitReviewSession(
        AUTHORED_IMPLICIT_CANDIDATES,
        reviews_path=reviews_path,
        existing_reviews=(),
        train_path=TRAIN_PATH,
    )
    with pytest.raises(ValueError, match="invalid review value"):
        session.apply_choice("9")
    with pytest.raises(ValueError, match="invalid review value|UNREVIEWED"):
        session.apply_status(AuthoredReviewStatus.UNREVIEWED)
    with pytest.raises(ValueError, match="unknown command"):
        parse_authored_review_command("x")


def test_candidate_metadata_preservation(tmp_path):
    reviews_path = tmp_path / "reviews.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    write_authored_candidates(candidates_path, AUTHORED_IMPLICIT_CANDIDATES)
    before = candidates_path.read_text(encoding="utf-8")

    session = AuthoredImplicitReviewSession(
        load_authored_candidates(candidates_path),
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
    assert record.authoring_reason == candidate.authoring_reason
    assert (
        record.intended_personal_requirement
        == candidate.intended_personal_requirement
    )
    assert (
        record.intended_environmental_requirement
        == candidate.intended_environmental_requirement
    )
    assert record.source_kind == "authored_stage_a"
    assert record.planner_bucket == "MIXED_IMPLICIT"
    # Candidate file still UNREVIEWED.
    reloaded = load_authored_candidates(candidates_path)
    assert all(
        item.review_status is AuthoredReviewStatus.UNREVIEWED for item in reloaded
    )


def test_not_injected_into_training_data():
    train = {
        normalize_query_key(row["query"])
        for row in json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    }
    for item in AUTHORED_IMPLICIT_CANDIDATES:
        assert normalize_query_key(item.query) not in train


def test_choice_mappings():
    assert parse_authored_review_command("1") == (
        "assign",
        AuthoredReviewStatus.ACCEPT,
    )
    assert parse_authored_review_command("2") == (
        "assign",
        AuthoredReviewStatus.REJECT,
    )
    assert parse_authored_review_command("s")[0] == "skip"
    assert parse_authored_review_command("b")[0] == "back"
    assert parse_authored_review_command("q")[0] == "quit"
