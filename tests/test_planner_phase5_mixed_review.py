"""Phase-5 Mixed candidate review tool tests (no terminal input)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tiergraph.planner.corpus import StageACandidate, load_candidates_jsonl
from tiergraph.planner.mixed_review import (
    CHOICE_TO_BUCKET,
    MixedReviewBucket,
    MixedReviewRecord,
    MixedReviewSession,
    ReviewStatus,
    load_mixed_candidates,
    load_mixed_reviews,
    parse_review_command,
    summarize_mixed_reviews,
    write_mixed_reviews,
)


ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = ROOT / "dataset" / "planner" / "stage_a_candidates.jsonl"


def _file_fingerprint(path: Path) -> tuple[int, str]:
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def test_exactly_127_mixed_candidates_loaded():
    mixed = load_mixed_candidates(CANDIDATES_PATH)
    assert len(mixed) == 127
    assert all(item.source_classification_label == "Mixed" for item in mixed)


def test_personal_and_environmental_excluded_from_review():
    all_candidates = load_candidates_jsonl(CANDIDATES_PATH)
    mixed = load_mixed_candidates(CANDIDATES_PATH)
    mixed_ids = {item.source_query_id for item in mixed}
    for item in all_candidates:
        if item.source_classification_label != "Mixed":
            assert item.source_query_id not in mixed_ids


def test_original_query_preserved_in_review_record(tmp_path):
    mixed = load_mixed_candidates(CANDIDATES_PATH)
    candidate = mixed[0]
    reviews_path = tmp_path / "reviews.jsonl"
    session = MixedReviewSession(mixed, reviews_path=reviews_path)
    assert session.current() is not None
    assert session.current().query == candidate.query
    record = session.apply_bucket(MixedReviewBucket.MIXED_IMPLICIT)
    assert record.query == candidate.query
    reloaded = load_mixed_reviews(reviews_path)
    assert reloaded[0].query == candidate.query


def test_choice_mappings_correct():
    assert CHOICE_TO_BUCKET["1"] is MixedReviewBucket.MIXED_IMPLICIT
    assert CHOICE_TO_BUCKET["2"] is MixedReviewBucket.MIXED_PARALLEL
    assert CHOICE_TO_BUCKET["3"] is MixedReviewBucket.MIXED_SEQUENTIAL
    assert CHOICE_TO_BUCKET["4"] is MixedReviewBucket.NOT_SUITABLE
    assert parse_review_command("1") == ("assign", MixedReviewBucket.MIXED_IMPLICIT)
    assert parse_review_command("2") == ("assign", MixedReviewBucket.MIXED_PARALLEL)
    assert parse_review_command("3") == ("assign", MixedReviewBucket.MIXED_SEQUENTIAL)
    assert parse_review_command("4") == ("assign", MixedReviewBucket.NOT_SUITABLE)
    assert parse_review_command("s")[0] == "skip"
    assert parse_review_command("b")[0] == "back"
    assert parse_review_command("q")[0] == "quit"


def test_save_and_resume(tmp_path):
    mixed = load_mixed_candidates(CANDIDATES_PATH)[:5]
    reviews_path = tmp_path / "reviews.jsonl"
    session = MixedReviewSession(mixed, reviews_path=reviews_path)
    first_id = session.current().source_query_id
    session.apply_choice("1")
    second_id = session.current().source_query_id
    session.apply_choice("2")
    assert reviews_path.is_file()

    resumed = MixedReviewSession(mixed, reviews_path=reviews_path)
    assert first_id in {item.source_query_id for item in resumed.reviews}
    assert second_id in {item.source_query_id for item in resumed.reviews}
    assert resumed.current() is not None
    assert resumed.current().source_query_id not in {first_id, second_id}
    assert resumed.summary()["reviewed"] == 2


def test_duplicate_source_query_id_rejected(tmp_path):
    mixed = load_mixed_candidates(CANDIDATES_PATH)[:2]
    reviews_path = tmp_path / "reviews.jsonl"
    record = MixedReviewRecord(
        source_query_id=mixed[0].source_query_id,
        semantic_group_id=mixed[0].semantic_group_id,
        query=mixed[0].query,
        source_classification_label="Mixed",
        planner_bucket=MixedReviewBucket.MIXED_PARALLEL,
        review_status=ReviewStatus.REVIEWED,
    )
    with pytest.raises(ValueError, match="duplicate source_query_id"):
        write_mixed_reviews(reviews_path, [record, record])

    write_mixed_reviews(reviews_path, [record])
    reviews_path.write_text(
        reviews_path.read_text(encoding="utf-8")
        + json.dumps(record.model_dump(mode="json"))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate source_query_id"):
        load_mixed_reviews(reviews_path)


def test_changing_a_decision_with_back(tmp_path):
    mixed = load_mixed_candidates(CANDIDATES_PATH)[:3]
    reviews_path = tmp_path / "reviews.jsonl"
    session = MixedReviewSession(mixed, reviews_path=reviews_path)
    first = session.current()
    assert first is not None
    session.apply_choice("1")
    assert session.reviews[0].planner_bucket is MixedReviewBucket.MIXED_IMPLICIT
    back_to = session.back()
    assert back_to is not None
    assert back_to.source_query_id == first.source_query_id
    assert first.source_query_id not in {
        item.source_query_id for item in session.reviews
    }
    session.apply_choice("3")
    assert session.reviews[0].planner_bucket is MixedReviewBucket.MIXED_SEQUENTIAL
    assert session.reviews[0].source_query_id == first.source_query_id


def test_summary_counts_correct(tmp_path):
    mixed = load_mixed_candidates(CANDIDATES_PATH)[:8]
    reviews_path = tmp_path / "reviews.jsonl"
    session = MixedReviewSession(mixed, reviews_path=reviews_path)
    session.apply_choice("1")
    session.apply_choice("1")
    session.apply_choice("2")
    session.apply_choice("3")
    session.apply_choice("4")
    summary = session.summary()
    assert summary["total_mixed"] == 8
    assert summary["reviewed"] == 5
    assert summary["remaining"] == 3
    assert summary["counts"]["mixed_implicit"] == 2
    assert summary["counts"]["mixed_parallel"] == 1
    assert summary["counts"]["mixed_sequential"] == 1
    assert summary["counts"]["not_suitable"] == 1
    assert summary["targets"]["mixed_implicit"]["reached_target"] is False
    assert summarize_mixed_reviews(mixed, session.reviews) == summary


def test_source_candidate_file_remains_unchanged(tmp_path):
    before = _file_fingerprint(CANDIDATES_PATH)
    mixed = load_mixed_candidates(CANDIDATES_PATH)
    reviews_path = tmp_path / "reviews.jsonl"
    session = MixedReviewSession(mixed, reviews_path=reviews_path)
    session.apply_choice("2")
    session.back()
    session.apply_choice("4")
    session.save()
    after = _file_fingerprint(CANDIDATES_PATH)
    assert after == before
    assert reviews_path.is_file()


def test_session_rejects_non_mixed_candidates(tmp_path):
    personal = StageACandidate.model_validate(
        {
            "source_query_id": "src_p",
            "semantic_group_id": "src_p",
            "query": "What is my blood type?",
            "source_classification_label": "Personal",
            "annotation_status": "unreviewed",
            "planner_bucket": None,
            "split": None,
        }
    )
    with pytest.raises(ValueError, match="non-Mixed"):
        MixedReviewSession([personal], reviews_path=tmp_path / "r.jsonl")
