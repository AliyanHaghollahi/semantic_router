"""Focused Todo-4 tests: Stage-A v2 480-example final selection."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from tiergraph.planner.corpus import normalize_query_key
from tiergraph.planner.stage_a_v2_selection import (
    SELECTION_ORDERING_DOC,
    build_stage_a_v2_selection,
    selection_fingerprint,
    validate_stage_a_v2_selection,
    write_stage_a_v2_selection,
)
from tiergraph.planner.stage_a_v2_spec import (
    H7_FAMILY_MINIMUMS,
    H7_MULTI_HOP_MINIMUM,
    STAGE_A_V1_SELECTION_PATH,
    STAGE_A_V1_STEP_A_PATH,
    STAGE_A_V1_STEP_B_PATH,
    STAGE_A_V2_AUTHORED_REVIEWS_PATH,
    STAGE_A_V2_BUCKETS,
    STAGE_A_V2_SELECTION_PATH,
    STAGE_A_V2_SELECTION_REPORT_PATH,
    STAGE_A_V2_SELECTION_SEED,
    example_is_quarantined_for_publication_test,
    hard_holdout_atoms,
    is_legal_h7_pair,
    parse_h7_family_label,
)


ROOT = Path(__file__).resolve().parent.parent

FROZEN_V1 = (
    STAGE_A_V1_SELECTION_PATH,
    STAGE_A_V1_STEP_A_PATH,
    STAGE_A_V1_STEP_B_PATH,
)


@pytest.fixture(scope="module")
def selection_batch():
    return build_stage_a_v2_selection()


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


def test_deterministic_selection_and_fingerprint(selection_batch):
    a, ra = selection_batch
    b, rb = build_stage_a_v2_selection()
    assert [r["stage_a_id"] for r in a] == [r["stage_a_id"] for r in b]
    assert [r["query"] for r in a] == [r["query"] for r in b]
    assert ra["I_selection_fingerprint"] == rb["I_selection_fingerprint"]
    assert ra["I_selection_fingerprint"] == selection_fingerprint(a)
    assert STAGE_A_V2_SELECTION_SEED == 20260901
    assert "sa_0121" in SELECTION_ORDERING_DOC


def test_geometry_and_legacy_carryforward(selection_batch):
    selected, report = selection_batch
    assert len(selected) == 480
    by_bucket = Counter(r["final_bucket"] for r in selected)
    assert dict(by_bucket) == {b: 96 for b in STAGE_A_V2_BUCKETS}
    assert report["H1"] == {"Personal": 96, "Environmental": 96, "Mixed": 288}
    assert report["B_legacy_vs_new"]["legacy"] == 120
    assert report["B_legacy_vs_new"]["new"] == 360

    v1 = [
        json.loads(line)
        for line in (ROOT / STAGE_A_V1_SELECTION_PATH)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    by_id = {r["stage_a_id"]: r for r in selected}
    for row in v1:
        got = by_id[row["stage_a_id"]]
        assert got["query"] == row["query"]
        assert got["final_bucket"] == row["final_bucket"]
        assert got["source_kind"] == row["source_kind"]
        assert got["source_id"] == row["source_id"]
        assert got["semantic_group"] == row["semantic_group"]
        assert got["template_group"] == row["template_group"]
        assert got["provenance"] == row["provenance"]


def test_unique_queries_and_validation_gates(selection_batch):
    selected, _report = selection_batch
    keys = [normalize_query_key(r["query"]) for r in selected]
    assert len(keys) == len(set(keys))
    assert validate_stage_a_v2_selection(selected) == []

    reviews = [
        json.loads(line)
        for line in (ROOT / STAGE_A_V2_AUTHORED_REVIEWS_PATH)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    status = {r["candidate_id"]: r["review_status"] for r in reviews if r.get("candidate_id")}
    for row in selected:
        cid = row.get("candidate_id")
        if cid and str(cid).startswith("auth_v2_"):
            assert status[cid] == "APPROVE"
        if row.get("final_bucket") == "MIXED_PARALLEL" and row.get("h7_families"):
            raise AssertionError(row["stage_a_id"])
        for label in row.get("h7_families") or []:
            src, tgt = parse_h7_family_label(label)
            assert is_legal_h7_pair(src, tgt)


def test_new_ids_and_bucket_composition(selection_batch):
    selected, report = selection_batch
    new_rows = [r for r in selected if r["stage_a_id"] >= "sa_0121"]
    assert [r["stage_a_id"] for r in new_rows] == [f"sa_{i:04d}" for i in range(121, 481)]
    for bucket in STAGE_A_V2_BUCKETS:
        assert report["B_legacy_vs_new"]["new_per_bucket"][bucket] == 72

    personal_new = [
        r
        for r in new_rows
        if r["final_bucket"] == "Personal"
    ]
    possessive = [
        r
        for r in personal_new
        if r.get("authored_template_family") == "retrieve_possessive_h5_none"
    ]
    assert len(possessive) == 16

    implicit_new = [r for r in new_rows if r["final_bucket"] == "MIXED_IMPLICIT"]
    assert all(r.get("h5_positive") is True for r in implicit_new)
    assert all(not (r.get("h7_families") or []) for r in implicit_new)
    fam_counts = Counter(r["authored_template_family"] for r in implicit_new)
    assert len(fam_counts) == 15
    assert max(fam_counts.values()) <= 5

    sequential_new = [r for r in new_rows if r["final_bucket"] == "MIXED_SEQUENTIAL"]
    assert sum(1 for r in sequential_new if r.get("multi_hop")) >= H7_MULTI_HOP_MINIMUM
    h7 = Counter(label for r in sequential_new for label in (r.get("h7_families") or []))
    for label, need in H7_FAMILY_MINIMUMS.items():
        assert h7.get(label, 0) >= need


def test_holdout_links_and_publication_eligibility_preserved(selection_batch):
    selected, report = selection_batch
    seat = [
        r
        for r in selected
        if r.get("authored_template_family")
        in {"my_reservation_seat_marker", "resolve_only_identify_seat_marker"}
    ]
    assert seat
    for row in seat:
        assert row["authored_holdout_family"] == "holdout_seat_reservation_match"
        assert "authored_holdout_family:holdout_seat_reservation_match" in hard_holdout_atoms(
            row
        )

    urgency = [
        r
        for r in selected
        if r.get("authored_template_family") == "urgency_distractor_scene"
    ]
    assert len(urgency) == 4
    assert all(r.get("publication_test_eligible") is False for r in urgency)
    assert all(example_is_quarantined_for_publication_test(r) for r in urgency)
    assert report["G_publication_test_eligibility"]["train_dev_only"] >= 4


def test_h5_h7_report_separates_legacy_gold_from_new_provisional(selection_batch):
    selected, report = selection_batch
    h5 = report["E_h5_accounting"]
    assert h5["legacy_gold"]["h5_positive"] == 58
    assert h5["legacy_gold"]["h5_negative"] == 62
    assert h5["legacy_gold"]["h5_positive_by_bucket"] == {
        "MIXED_IMPLICIT": 24,
        "MIXED_PARALLEL": 10,
        "MIXED_SEQUENTIAL": 24,
    }
    assert h5["new_provisional"]["h5_positive"] == 97
    assert h5["new_provisional"]["h5_unknown"] == 20
    assert h5["combined_projected"]["h5_positive"] == 155
    assert h5["combined_projected"]["within_target_range"] is True

    h7 = report["F_h7_accounting"]
    assert h7["legacy_gold"]["h7_positive_examples"] == 15
    assert h7["legacy_gold"]["multi_hop"] == 0
    assert h7["legacy_gold"]["family_example_counts"] == {
        "IDENTIFY_ENVIRONMENTAL->LOCATE_ENVIRONMENTAL": 4,
        "LOCATE_ENVIRONMENTAL->NAVIGATE_TO": 7,
        "IDENTIFY_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT": 2,
        "LOCATE_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT": 2,
    }
    assert h7["new_provisional"]["h7_positive_examples"] == 65
    assert h7["new_provisional"]["multi_hop"] == 10
    assert h7["combined_projected"]["h7_positive_examples"] == 80
    assert h7["combined_projected"]["multi_hop"] == 10
    assert h7["combined_projected"]["family_example_counts"][
        "IDENTIFY_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT"
    ] == 19
    assert h7["combined_projected"]["family_example_counts"][
        "LOCATE_ENVIRONMENTAL->NAVIGATE_TO"
    ] == 28
    assert h7["combined_projected"]["family_shares"][
        "LOCATE_ENVIRONMENTAL->NAVIGATE_TO"
    ] == pytest.approx(0.35)
    assert h7["combined_projected"]["max_h7_family_share"] <= 0.35
    assert h7["combined_projected"]["max_share_within_cap"] is True
    assert all(h7["combined_projected"]["floors_met"].values())

    # Exact H7 diversity-cap swap
    new_seq = [
        r
        for r in selected
        if r["final_bucket"] == "MIXED_SEQUENTIAL" and r["stage_a_id"] >= "sa_0121"
    ]
    ids = {r.get("candidate_id") for r in new_seq}
    assert "auth_v2_locate_navigate_clinic_corridor_03" not in ids
    assert "auth_v2_identify_describe_device_status_05" in ids


def test_write_artifact_matches(selection_batch):
    selected, report = selection_batch
    write_stage_a_v2_selection()
    disk = [
        json.loads(line)
        for line in (ROOT / STAGE_A_V2_SELECTION_PATH)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [r["stage_a_id"] for r in disk] == [r["stage_a_id"] for r in selected]
    assert [r["query"] for r in disk] == [r["query"] for r in selected]
    disk_report = json.loads(
        (ROOT / STAGE_A_V2_SELECTION_REPORT_PATH).read_text(encoding="utf-8")
    )
    assert disk_report["I_selection_fingerprint"] == report["I_selection_fingerprint"]
    assert len(report["I_selection_fingerprint"]) == 64
    assert len(report["J_excluded_approved"]) == 15  # 10 implicit + 5 sequential
    assert any(
        x.get("candidate") == "auth_v2_locate_navigate_clinic_corridor_03"
        for x in report["J_excluded_approved"]
    )
    assert report["G_publication_test_eligibility"]["train_dev_only"] == 110
    assert (
        report["G_publication_test_eligibility"]["train_dev_only_breakdown"][
            "reason_any_counts"
        ]["template_group_quarantine"]
        == 105
    )
    assert report["F_h7_accounting"]["combined_projected"]["max_share_within_cap"] is True
