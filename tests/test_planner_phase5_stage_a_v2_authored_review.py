"""Todo-3B tests: Stage-A v2 authored candidate review / approval workflow."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tiergraph.planner.stage_a_v2_authored_review import (
    REVIEW_APPROVE,
    REVIEW_REVISE,
    REVIEW_REJECT,
    REVIEW_UNREVIEWED,
    build_authored_reviews,
    first_pass_review_decision,
    h7_family_shares_among_positive,
    publication_test_eligible_for_row,
    validate_review_status,
    write_authored_reviews,
)
from tiergraph.planner.stage_a_v2_spec import (
    AUTHORED_HOLDOUT_FAMILY_LINKS,
    AUTHORED_REVIEW_METHOD,
    AUTHORED_REVIEW_STATUSES,
    H7_FAMILY_MINIMUMS,
    H7_FAMILY_SHARE_MAX,
    H7_MULTI_HOP_MINIMUM,
    PUBLICATION_TEST_INELIGIBLE_AUTHORED_FAMILIES,
    STAGE_A_V1_SELECTION_PATH,
    STAGE_A_V1_STEP_A_PATH,
    STAGE_A_V1_STEP_B_PATH,
    STAGE_A_V2_AUTHORED_APPROVE_FLOOR_IMPLICIT,
    STAGE_A_V2_AUTHORED_APPROVE_FLOOR_SEQUENTIAL,
    STAGE_A_V2_AUTHORED_REVIEW_REPORT_PATH,
    STAGE_A_V2_AUTHORED_REVIEWS_PATH,
    STAGE_A_V2_SELECTION_PATH,
    example_is_quarantined_for_publication_test,
    hard_holdout_atoms,
    is_legal_h7_pair,
    parse_h7_family_label,
    publication_test_ineligibility_reason,
)


ROOT = Path(__file__).resolve().parent.parent

FROZEN_V1 = (
    STAGE_A_V1_SELECTION_PATH,
    STAGE_A_V1_STEP_A_PATH,
    STAGE_A_V1_STEP_B_PATH,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def reviewed_batch():
    return build_authored_reviews()


def test_v1_frozen_unchanged():
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


def test_review_writer_does_not_mutate_v2_selection():
    """Todo-3B review write must not mutate the Todo-4 frozen selection artifact."""
    assert STAGE_A_V2_SELECTION_PATH.name == "stage_a_v2_final_selection.jsonl"
    assert STAGE_A_V2_SELECTION_PATH != STAGE_A_V1_SELECTION_PATH
    selection_path = ROOT / STAGE_A_V2_SELECTION_PATH
    assert selection_path.is_file()
    before = _sha256_file(selection_path)
    write_authored_reviews()
    after = _sha256_file(selection_path)
    assert after == before


def test_review_status_validity():
    for status in AUTHORED_REVIEW_STATUSES:
        assert validate_review_status(status) == []
    assert validate_review_status("ACCEPT")
    assert validate_review_status("APPROVED")


def test_deterministic_review_generation(reviewed_batch):
    a, ra = reviewed_batch
    b, rb = build_authored_reviews()
    assert [r["candidate_id"] for r in a] == [r["candidate_id"] for r in b]
    assert [r["review_status"] for r in a] == [r["review_status"] for r in b]
    assert [r["query"] for r in a] == [r["query"] for r in b]
    assert ra["A_status_counts"] == rb["A_status_counts"]
    assert ra["approved_total"] == rb["approved_total"]


def test_provenance_and_identity_preserved(reviewed_batch):
    reviews, _report = reviewed_batch
    assert len(reviews) == 199
    for row in reviews:
        assert row["candidate_id"].startswith("auth_v2_")
        assert row["source_kind"] == "authored"
        assert row["authored_template_family"]
        assert row["authored_holdout_family"]
        assert row["provenance"]["authored_template_family"] == row[
            "authored_template_family"
        ]
        assert row["provenance"]["origin"] == "stage_a_v2_authored_instantiate"
        assert row["provenance"]["review_origin"] == (
            "stage_a_v2_authored_review_first_pass"
        )
        assert row["review_method"] == AUTHORED_REVIEW_METHOD
        assert row["provenance"]["review_method"] == AUTHORED_REVIEW_METHOD
        assert row["review_status"] in AUTHORED_REVIEW_STATUSES
        assert row["review_status"] != REVIEW_UNREVIEWED
        assert isinstance(row["publication_test_eligible"], bool)
        assert row["review_reason"]
        assert publication_test_eligible_for_row(row) == (
            not example_is_quarantined_for_publication_test(row)
        )


def test_approved_h7_legality_and_floors(reviewed_batch):
    reviews, report = reviewed_batch
    approved = [r for r in reviews if r["review_status"] == REVIEW_APPROVE]
    for row in approved:
        for label in row.get("h7_families") or []:
            src, tgt = parse_h7_family_label(label)
            assert is_legal_h7_pair(src, tgt)
    for label, need in H7_FAMILY_MINIMUMS.items():
        assert report["D_approved_h7_family_distribution"].get(label, 0) >= need
    assert report["E_approved_multi_hop_count"] >= H7_MULTI_HOP_MINIMUM
    assert report["H_shortfalls"]["illegal_h7_on_approved"] == []
    assert report["G_family_diversity"]["max_h7_family_share"] <= H7_FAMILY_SHARE_MAX
    assert report["review_method"] == AUTHORED_REVIEW_METHOD


def test_h7_family_share_denominator_and_maximum(reviewed_batch):
    reviews, report = reviewed_batch
    approved = [r for r in reviews if r["review_status"] == REVIEW_APPROVE]
    denom, shares, max_label, max_share = h7_family_shares_among_positive(approved)
    h7_pos = [r for r in approved if r.get("h7_positive")]
    assert denom == len(h7_pos) == 70
    assert report["G_family_diversity"]["h7_positive_approved_count"] == 70
    # Share = examples containing label / H7-positive examples (not authored-template share).
    expected = {
        "IDENTIFY_ENVIRONMENTAL->LOCATE_ENVIRONMENTAL": 22 / 70,
        "LOCATE_ENVIRONMENTAL->NAVIGATE_TO": 22 / 70,
        "IDENTIFY_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT": 19 / 70,
        "LOCATE_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT": 17 / 70,
    }
    for label, share in expected.items():
        assert shares[label] == pytest.approx(share)
        assert report["G_family_diversity"]["h7_family_shares"][label] == pytest.approx(
            round(share, 4)
        )
    assert max_label in {
        "IDENTIFY_ENVIRONMENTAL->LOCATE_ENVIRONMENTAL",
        "LOCATE_ENVIRONMENTAL->NAVIGATE_TO",
    }
    assert max_share == pytest.approx(22 / 70)
    assert report["G_family_diversity"]["max_h7_family_share"] == pytest.approx(
        round(22 / 70, 4)
    )
    assert report["D2_approved_h7_family_shares"]["h7_positive_denominator"] == 70
    assert report["D2_approved_h7_family_shares"]["max_share_cap"] == H7_FAMILY_SHARE_MAX
    # Multi-hop can contribute to more than one label → shares need not sum to 1.
    assert sum(shares.values()) > 1.0
    assert max_share <= H7_FAMILY_SHARE_MAX


def test_approve_floors_implicit_sequential(reviewed_batch):
    _reviews, report = reviewed_batch
    by_bucket = report["B_approved_counts_by_bucket"]
    assert by_bucket.get("MIXED_IMPLICIT", 0) >= STAGE_A_V2_AUTHORED_APPROVE_FLOOR_IMPLICIT
    assert (
        by_bucket.get("MIXED_SEQUENTIAL", 0)
        >= STAGE_A_V2_AUTHORED_APPROVE_FLOOR_SEQUENTIAL
    )
    assert report["H_shortfalls"]["approve_floor_shortfalls"] == {
        "MIXED_IMPLICIT": 0,
        "MIXED_SEQUENTIAL": 0,
    }


def test_linked_holdout_families_preserved(reviewed_batch):
    reviews, _report = reviewed_batch
    for leaf, parent in AUTHORED_HOLDOUT_FAMILY_LINKS.items():
        rows = [r for r in reviews if r["authored_template_family"] == leaf]
        assert rows
        for row in rows:
            assert row["authored_holdout_family"] == parent
            atoms = hard_holdout_atoms(row)
            assert f"authored_holdout_family:{parent}" in atoms


def test_publication_test_eligibility_obeys_quarantine(reviewed_batch):
    reviews, report = reviewed_batch
    for row in reviews:
        eligible = publication_test_eligible_for_row(row)
        assert eligible == (not example_is_quarantined_for_publication_test(row))
        assert row["publication_test_eligible"] is eligible
        if not eligible:
            reason = publication_test_ineligibility_reason(row)
            assert reason
            assert row["provenance"].get("publication_test_ineligibility_reason") == reason

    urgency = [
        r
        for r in reviews
        if r["authored_template_family"] == "urgency_distractor_scene"
    ]
    assert len(urgency) == 4
    assert all(r["review_status"] == REVIEW_APPROVE for r in urgency)
    assert all(r["publication_test_eligible"] is False for r in urgency)
    assert all(
        example_is_quarantined_for_publication_test(r) for r in urgency
    )
    assert urgency[0]["authored_template_family"] in (
        PUBLICATION_TEST_INELIGIBLE_AUTHORED_FAMILIES
    )

    train_dev = report["F_publication_test_eligibility"]["train_dev_only_candidates"]
    assert report["F_publication_test_eligibility"]["approved_train_dev_only"] == 4
    assert report["F_publication_test_eligibility"][
        "approved_publication_test_eligible"
    ] == 191
    assert {row["candidate_id"] for row in train_dev} == {
        r["candidate_id"] for r in urgency
    }
    for item in train_dev:
        assert item["authored_template_family"] == "urgency_distractor_scene"
        assert "publication_test_ineligible_authored_family" in item["reason"]


def test_cleanup_families_approved_and_contrast_family_revised(reviewed_batch):
    reviews, report = reviewed_batch
    train = [
        r
        for r in reviews
        if r["authored_template_family"] == "my_train_platform_reservation"
    ]
    assert len(train) == 5
    assert all(r["review_status"] == REVIEW_APPROVE for r in train)
    assert all(r["operator_family"] == ["IDENTIFY_ENVIRONMENTAL"] for r in train)

    lab = [
        r
        for r in reviews
        if r["authored_template_family"]
        == "resolve_locate_navigate_lab_draw_station"
    ]
    assert len(lab) == 4
    assert all(r["review_status"] == REVIEW_APPROVE for r in lab)
    assert "resolve_locate_navigate_order_locker" not in {
        r["authored_template_family"] for r in reviews
    }

    contrast = [
        r
        for r in reviews
        if r["authored_template_family"] == "retrieve_vs_describe_personal"
    ]
    assert len(contrast) == 4
    assert all(r["review_status"] == REVIEW_REVISE for r in contrast)
    assert all(r["h5_positive"] is False for r in contrast)
    assert report["A_status_counts"][REVIEW_REJECT] == 0
    assert report["A_status_counts"][REVIEW_REVISE] == 4
    assert report["A_status_counts"][REVIEW_APPROVE] == 195


def test_written_artifact_matches_when_present():
    path = ROOT / STAGE_A_V2_AUTHORED_REVIEWS_PATH
    report_path = ROOT / STAGE_A_V2_AUTHORED_REVIEW_REPORT_PATH
    write_authored_reviews()
    assert path.is_file()
    assert report_path.is_file()
    reviews, report = build_authored_reviews()
    disk = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r["candidate_id"] for r in disk] == [r["candidate_id"] for r in reviews]
    assert [r["review_status"] for r in disk] == [r["review_status"] for r in reviews]
    disk_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert disk_report["A_status_counts"] == report["A_status_counts"]
    assert disk_report["approved_total"] == report["approved_total"]


def test_first_pass_rejects_illegal_h7():
    bad = {
        "candidate_id": "auth_v2_fake_01",
        "query": "Identify this sign then navigate somehow.",
        "source_kind": "authored",
        "authored_template_family": "identify_locate_plaque_vs_desk",
        "authored_holdout_family": "identify_locate_plaque_vs_desk",
        "semantic_group": "other__identify_locate_plaque_vs_desk",
        "template_group": "identify_locate_plaque_vs_desk",
        "proposed_final_bucket": "MIXED_SEQUENTIAL",
        "final_bucket": "MIXED_SEQUENTIAL",
        "operator_family": ["IDENTIFY_ENVIRONMENTAL", "NAVIGATE_TO"],
        "h5_positive": False,
        "h7_positive": True,
        "h7_families": ["IDENTIFY_ENVIRONMENTAL->NAVIGATE_TO"],
        "multi_hop": False,
        "provenance": {
            "origin": "stage_a_v2_authored_instantiate",
            "authored_template_family": "identify_locate_plaque_vs_desk",
        },
    }
    decision = first_pass_review_decision(bad)
    assert decision["review_status"] == REVIEW_REJECT
