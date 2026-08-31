"""Focused Todo-3A tests: authored family instantiation into reviewable candidates."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from tiergraph.planner.corpus import normalize_query_key
from tiergraph.planner.stage_a_v2_authored_instantiate import (
    QUERY_BANKS,
    STAGE_A_V2_AUTHORED_CANDIDATES_PATH,
    instantiate_authored_candidates,
    load_authored_family_specs,
    write_authored_candidates,
)
from tiergraph.planner.stage_a_v2_candidates import (
    load_v1_frozen_index,
    validate_candidate_row,
)
from tiergraph.planner.stage_a_v2_spec import (
    STAGE_A_V1_SELECTION_PATH,
    STAGE_A_V1_STEP_A_PATH,
    STAGE_A_V1_STEP_B_PATH,
    STAGE_A_V2_SELECTION_PATH,
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
def authored_batch():
    return instantiate_authored_candidates()


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


def test_no_final_selection_created():
    assert not (ROOT / STAGE_A_V2_SELECTION_PATH).exists()


def test_deterministic_generation(authored_batch):
    a, ra = authored_batch
    b, rb = instantiate_authored_candidates()
    assert [r["candidate_id"] for r in a] == [r["candidate_id"] for r in b]
    assert [r["query"] for r in a] == [r["query"] for r in b]
    assert ra["concrete_authored_total"] == rb["concrete_authored_total"]


def test_exact_expected_counts(authored_batch):
    candidates, report = authored_batch
    assert len(candidates) == 199
    assert report["counts_by_bucket"]["MIXED_IMPLICIT"] == 82
    assert report["counts_by_bucket"]["MIXED_SEQUENTIAL"] == 77
    assert report["counts_by_bucket"]["Personal"] == 20
    assert report["counts_by_bucket"]["Environmental"] == 20
    assert report["failed_families"] == []
    assert report["duplicate_conflict_count"] == 0

    specs = load_authored_family_specs()
    by_family = Counter(r["authored_template_family"] for r in candidates)
    for spec in specs:
        family = spec["authored_template_family"]
        assert by_family[family] == spec["planned_paraphrases"]
        assert len(QUERY_BANKS[family]) == spec["planned_paraphrases"]


def test_unique_queries_and_no_v1_overlap(authored_batch):
    candidates, _report = authored_batch
    keys = [normalize_query_key(r["query"]) for r in candidates]
    assert len(keys) == len(set(keys))
    v1 = load_v1_frozen_index()
    for key in keys:
        assert key not in v1["query_keys"]


def test_provenance_and_stable_families(authored_batch):
    candidates, report = authored_batch
    for row in candidates:
        assert validate_candidate_row(row) == []
        assert row["source_kind"] == "authored"
        assert row["review_status"] == "needs_review"
        assert row["authored_template_family"]
        assert row["authored_template_family"] == row["scenario_family"]
        assert row["template_group"] == row["authored_template_family"]
        assert row["candidate_id"].startswith("auth_v2_")
    assert len(report["distinct_authored_template_families"]) == 41


def test_legal_h7_only_and_implicit_has_no_h7(authored_batch):
    candidates, report = authored_batch
    for row in candidates:
        if row["proposed_final_bucket"] == "MIXED_IMPLICIT":
            assert row["h5_positive"] is True
            assert row["h7_positive"] is False
            assert row["h7_families"] == []
        for label in row.get("h7_families") or []:
            src, tgt = parse_h7_family_label(label)
            assert is_legal_h7_pair(src, tgt)
    assert report["multi_hop_count"] == 10
    assert report["h7_positive_count"] == 70
    for label, count in report["h7_family_counts"].items():
        src, tgt = parse_h7_family_label(label)
        assert is_legal_h7_pair(src, tgt)
        assert count >= 14


def test_h5_distribution_includes_possessive_none_controls(authored_batch):
    candidates, report = authored_batch
    none_controls = [
        r
        for r in candidates
        if r["authored_template_family"] == "retrieve_possessive_h5_none"
    ]
    assert len(none_controls) == 16
    assert all(r["h5_positive"] is False for r in none_controls)
    assert report["h5_positive_count"] >= 82  # all implicit + sequential H5+
    seq_h5 = [
        r
        for r in candidates
        if r["proposed_final_bucket"] == "MIXED_SEQUENTIAL" and r["h5_positive"] is True
    ]
    assert len(seq_h5) == 25


def test_written_artifact_matches_when_present():
    path = ROOT / STAGE_A_V2_AUTHORED_CANDIDATES_PATH
    if not path.is_file():
        pytest.skip("authored candidates not written")
    candidates, _report = instantiate_authored_candidates()
    disk = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r["candidate_id"] for r in disk] == [r["candidate_id"] for r in candidates]
    # rewrite is deterministic / idempotent
    write_authored_candidates()
    disk2 = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(disk2) == 199


def test_todo3a_cleanup_train_platform_and_lab_family(authored_batch):
    candidates, _report = authored_batch
    train = [
        r
        for r in candidates
        if r["authored_template_family"] == "my_train_platform_reservation"
    ]
    assert len(train) == 5
    for row in train:
        assert row["operator_family"] == ["IDENTIFY_ENVIRONMENTAL"]
        assert "NAVIGATE_TO" not in (row["operator_family"] or [])
        assert row["h7_families"] == []
        assert row["h5_positive"] is True

    assert "resolve_locate_navigate_order_locker" not in QUERY_BANKS
    lab = [
        r
        for r in candidates
        if r["authored_template_family"]
        == "resolve_locate_navigate_lab_draw_station"
    ]
    assert len(lab) == 4
    for row in lab:
        assert row["operator_family"] == ["LOCATE_ENVIRONMENTAL", "NAVIGATE_TO"]
        assert row["h7_families"] == ["LOCATE_ENVIRONMENTAL->NAVIGATE_TO"]
        assert row["h5_positive"] is True
        assert "locker" not in row["query"].lower()
        assert "pickup" not in row["query"].lower()


def test_todo3a_cleanup_h5_negative_and_holdout_links(authored_batch):
    candidates, _report = authored_batch
    retrieve_vs = [
        r
        for r in candidates
        if r["authored_template_family"] == "retrieve_vs_describe_personal"
    ]
    assert len(retrieve_vs) == 4
    assert all(r["h5_positive"] is False for r in retrieve_vs)

    seat_holdouts = {
        hard_holdout_atoms(r)
        for r in candidates
        if r["authored_template_family"]
        in {"my_reservation_seat_marker", "resolve_only_identify_seat_marker"}
    }
    # leaf semantic groups differ; shared authored holdout parent must appear
    parents = {
        atom
        for atoms in seat_holdouts
        for atom in atoms
        if atom.startswith("authored_holdout_family:")
    }
    assert parents == {"authored_holdout_family:holdout_seat_reservation_match"}

    food_parents = {
        atom
        for r in candidates
        if r["authored_template_family"]
        in {"my_allergy_menu_safe", "my_dietary_restriction_dish"}
        for atom in hard_holdout_atoms(r)
        if atom.startswith("authored_holdout_family:")
    }
    assert food_parents == {"authored_holdout_family:holdout_food_profile_safety"}

    for row in candidates:
        assert row["authored_holdout_family"]
        if row["authored_template_family"] == "retrieve_vs_describe_personal":
            assert row["h5_positive"] is False
        if row["authored_template_family"] == "retrieve_possessive_h5_none":
            assert row["h5_positive"] is False