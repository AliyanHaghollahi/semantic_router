"""Tests for Phase-5 implicit-mixed candidate mining."""

from __future__ import annotations

import json
from pathlib import Path

from tiergraph.planner.corpus import (
    DEFAULT_CANDIDATE_SEED,
    StageACandidate,
    build_unique_query_pool,
    normalize_query_key,
)
from tiergraph.planner.implicit_mining import (
    build_mining_inputs,
    mine_implicit_candidates,
    score_implicit_candidate,
    summarize_implicit_mining,
    write_implicit_candidates_jsonl,
)


ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "dataset" / "training_data.json"
REVIEWS_PATH = ROOT / "dataset" / "planner" / "stage_a_mixed_reviews.jsonl"

_ENVIRONMENTAL_ONLY_FALSE_POSITIVES = (
    "How far am I from the fire hydrant?",
    "How far is the door from where I am sitting?",
    "Is the path from the couch to the door clear?",
    "Can I walk over to the lawns, or is there a barrier?",
    "Is there a clear sidewalk on the other side of the street?",
    "How long will it take to tidy up this room?",
)


def _toy_pool() -> tuple[StageACandidate, ...]:
    rows = [
        {"query": "Am I allergic to anything on this menu?", "label": "Personal"},
        {"query": "am i allergic to anything on this menu?", "label": "Personal"},
        {"query": "What is my blood type?", "label": "Personal"},
        {"query": "What does this sign say?", "label": "Environmental"},
        {
            "query": "What medication am I holding and is there a pharmacy nearby?",
            "label": "Mixed",
        },
        {
            "query": "Is this the right entrance for my appointment?",
            "label": "Environmental",
        },
        {"query": "Is this medication on my list?", "label": "Personal"},
        {
            "query": "Am I supposed to take this on an empty stomach?",
            "label": "Personal",
        },
        {
            "query": "Does this contain anything I'm allergic to?",
            "label": "Personal",
        },
    ]
    for query in _ENVIRONMENTAL_ONLY_FALSE_POSITIVES:
        rows.append({"query": query, "label": "Environmental"})
    return build_unique_query_pool(rows)


def test_deterministic_output():
    pool, excluded = build_mining_inputs(
        train_path=TRAIN_PATH,
        reviews_path=REVIEWS_PATH,
    )
    first = mine_implicit_candidates(
        pool,
        excluded_keys=excluded,
        limit=40,
        seed=DEFAULT_CANDIDATE_SEED,
    )
    second = mine_implicit_candidates(
        pool,
        excluded_keys=excluded,
        limit=40,
        seed=DEFAULT_CANDIDATE_SEED,
    )
    assert [item.to_json_dict() for item in first] == [
        item.to_json_dict() for item in second
    ]


def test_duplicate_removal_in_unique_pool():
    pool = _toy_pool()
    keys = [normalize_query_key(item.query) for item in pool]
    assert len(keys) == len(set(keys))
    allergy = [
        item
        for item in pool
        if normalize_query_key(item.query)
        == normalize_query_key("Am I allergic to anything on this menu?")
    ]
    assert len(allergy) == 1
    assert allergy[0].query == "Am I allergic to anything on this menu?"


def test_no_original_mixed_rows_in_shortlist():
    pool, excluded = build_mining_inputs(
        train_path=TRAIN_PATH,
        reviews_path=REVIEWS_PATH,
    )
    shortlist = mine_implicit_candidates(
        pool,
        excluded_keys=excluded,
        limit=80,
        seed=DEFAULT_CANDIDATE_SEED,
    )
    assert all(
        item.original_label in {"Personal", "Environmental"} for item in shortlist
    )
    mixed_keys = {
        normalize_query_key(item.query)
        for item in pool
        if item.source_classification_label == "Mixed"
    }
    assert not any(item.normalized_query in mixed_keys for item in shortlist)


def test_no_automatic_label_mutation():
    pool = _toy_pool()
    before = {
        item.source_query_id: item.source_classification_label for item in pool
    }
    shortlist = mine_implicit_candidates(pool, limit=10, seed=DEFAULT_CANDIDATE_SEED)
    after = {
        item.source_query_id: item.source_classification_label for item in pool
    }
    assert before == after
    for item in shortlist:
        assert item.original_label == before[item.source_id]
    assert all(
        item.original_label in {"Personal", "Environmental"} for item in shortlist
    )


def test_ranking_and_reasons():
    positive = "Am I allergic to anything on this menu?"
    score, reasons = score_implicit_candidate(positive)
    assert score > 0
    assert any(reason.startswith("personal:") for reason in reasons)
    assert any(reason.startswith("environmental:") for reason in reasons)

    pool = _toy_pool()
    shortlist = mine_implicit_candidates(pool, limit=10, seed=DEFAULT_CANDIDATE_SEED)
    assert shortlist
    scores = [item.score for item in shortlist]
    assert scores == sorted(scores, reverse=True)
    allergy = next(
        item
        for item in shortlist
        if item.query == "Am I allergic to anything on this menu?"
    )
    assert allergy.score >= shortlist[-1].score
    assert allergy.mining_reasons


def test_limit_behavior():
    pool, excluded = build_mining_inputs(
        train_path=TRAIN_PATH,
        reviews_path=REVIEWS_PATH,
    )
    full = mine_implicit_candidates(
        pool,
        excluded_keys=excluded,
        limit=80,
        seed=DEFAULT_CANDIDATE_SEED,
    )
    limited = mine_implicit_candidates(
        pool,
        excluded_keys=excluded,
        limit=5,
        seed=DEFAULT_CANDIDATE_SEED,
    )
    assert len(limited) == min(5, len(full))
    assert [item.source_id for item in limited] == [
        item.source_id for item in full[: len(limited)]
    ]


def test_positive_allergy_menu_example_is_mined():
    score, reasons = score_implicit_candidate(
        "Am I allergic to anything on this menu?"
    )
    assert score > 0
    assert any("allergy" in reason for reason in reasons)
    assert any(reason.startswith("environmental:") for reason in reasons)

    pool = _toy_pool()
    shortlist = mine_implicit_candidates(pool, limit=10, seed=DEFAULT_CANDIDATE_SEED)
    assert any(
        item.query == "Am I allergic to anything on this menu?" for item in shortlist
    )


def test_pure_personal_and_environmental_are_not_mined():
    assert score_implicit_candidate("What is my blood type?")[0] == 0
    assert score_implicit_candidate("What does this sign say?")[0] == 0
    pool = _toy_pool()
    shortlist = mine_implicit_candidates(pool, limit=20, seed=DEFAULT_CANDIDATE_SEED)
    queries = {item.query for item in shortlist}
    assert "What is my blood type?" not in queries
    assert "What does this sign say?" not in queries


def test_generic_first_person_pronouns_alone_are_not_personal_evidence():
    assert score_implicit_candidate("Can I walk over to the lawns?")[0] == 0
    assert score_implicit_candidate("Tell me what is on the shelf.")[0] == 0
    assert score_implicit_candidate("How far am I from the door?")[0] == 0
    assert score_implicit_candidate("Read the warning red label for me right now.")[0] == 0
    score, reasons = score_implicit_candidate(
        "Can I walk over to the lawns, or is there a barrier?"
    )
    assert score == 0
    assert reasons == ()


def test_pure_personal_medication_history_not_mined_without_observation():
    assert score_implicit_candidate("Is my medication covered under my insurance?")[0] == 0
    assert score_implicit_candidate("When did I last take my medication?")[0] == 0
    assert score_implicit_candidate("What pills am I supposed to take right now?")[0] == 0
    assert score_implicit_candidate(
        "Is there anything on my calendar this afternoon?"
    )[0] == 0


def test_personal_place_property_without_observation_is_not_environmental():
    assert score_implicit_candidate(
        "Which terminal is my flight departing from?"
    )[0] == 0
    assert score_implicit_candidate(
        "What is my usual pharmacy location?"
    )[0] == 0


def test_egocentric_spatial_language_is_environmental_not_personal():
    for query in _ENVIRONMENTAL_ONLY_FALSE_POSITIVES:
        score, reasons = score_implicit_candidate(query)
        assert score == 0, query
        assert reasons == (), query


def test_strong_personal_plus_observed_object_qualifies():
    positives = [
        "Is this medication on my list?",
        "Am I supposed to take this on an empty stomach?",
        "Does this contain anything I'm allergic to?",
        "Is this the right entrance for my appointment?",
    ]
    for query in positives:
        score, reasons = score_implicit_candidate(query)
        assert score > 0, query
        assert any(reason.startswith("personal:") for reason in reasons), query
        assert any(reason.startswith("environmental:") for reason in reasons), query

    pool = _toy_pool()
    shortlist = mine_implicit_candidates(pool, limit=20, seed=DEFAULT_CANDIDATE_SEED)
    queries = {item.query for item in shortlist}
    for query in positives:
        assert query in queries


def test_write_jsonl_and_summary(tmp_path):
    pool, excluded = build_mining_inputs(
        train_path=TRAIN_PATH,
        reviews_path=REVIEWS_PATH,
    )
    summary = summarize_implicit_mining(
        pool,
        excluded_keys=excluded,
        limit=80,
        seed=DEFAULT_CANDIDATE_SEED,
    )
    assert summary["unique_personal"] == 148
    assert summary["unique_environmental"] == 425
    assert summary["satisfying_mining_criteria"] >= summary["shortlist_written"]
    assert summary["shortlist_written"] <= 80

    out = tmp_path / "implicit.jsonl"
    candidates = mine_implicit_candidates(
        pool,
        excluded_keys=excluded,
        limit=80,
        seed=DEFAULT_CANDIDATE_SEED,
    )
    write_implicit_candidates_jsonl(out, candidates)
    lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == len(candidates)
    if lines:
        payload = json.loads(lines[0])
        assert set(payload) == {
            "source_id",
            "query",
            "original_label",
            "normalized_query",
            "score",
            "mining_reasons",
        }
