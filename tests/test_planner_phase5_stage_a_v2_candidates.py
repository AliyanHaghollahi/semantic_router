"""Focused Todo-2 tests: Stage-A v2 candidate inventory assembly."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tiergraph.planner.corpus import normalize_query_key
from tiergraph.planner.stage_a_v2_candidates import (
    STAGE_A_V2_AUTHORED_SPECS_PATH,
    STAGE_A_V2_CANDIDATES_PATH,
    STAGE_A_V2_CANDIDATE_REPORT_PATH,
    build_authored_family_specs,
    build_candidate_inventory,
    load_v1_frozen_index,
    validate_candidate_row,
    write_candidate_inventory,
)
from tiergraph.planner.stage_a_v2_spec import (
    STAGE_A_V1_SELECTION_PATH,
    STAGE_A_V1_STEP_A_PATH,
    STAGE_A_V1_STEP_B_PATH,
    STAGE_A_V2_SELECTION_PATH,
    is_legal_h7_pair,
    parse_h7_family_label,
)


ROOT = Path(__file__).resolve().parent.parent

FROZEN_V1_PATHS = (
    STAGE_A_V1_SELECTION_PATH,
    STAGE_A_V1_STEP_A_PATH,
    STAGE_A_V1_STEP_B_PATH,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def inventory():
    return build_candidate_inventory()


def test_v1_frozen_files_unchanged():
    for rel in FROZEN_V1_PATHS:
        path = ROOT / rel
        assert path.is_file(), rel
        diff = subprocess.run(
            ["git", "diff", "--", str(rel).replace("\\", "/")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert diff.stdout == ""
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", str(rel)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert untracked.stdout.strip() == ""


def test_inventory_writer_does_not_mutate_v2_selection():
    """Todo-2 inventory must not mutate the Todo-4 frozen selection artifact."""
    assert STAGE_A_V2_SELECTION_PATH.name == "stage_a_v2_final_selection.jsonl"
    assert STAGE_A_V2_SELECTION_PATH != STAGE_A_V1_SELECTION_PATH
    selection_path = ROOT / STAGE_A_V2_SELECTION_PATH
    assert selection_path.is_file()
    before = _sha256_file(selection_path)
    write_candidate_inventory()
    after = _sha256_file(selection_path)
    assert after == before


def test_no_duplicate_normalized_query_against_frozen_v1(inventory):
    candidates, _specs, _report = inventory
    v1 = load_v1_frozen_index()
    for row in candidates:
        key = normalize_query_key(row["query"])
        assert key not in v1["query_keys"], row["candidate_uid"]
        if row.get("source_id"):
            assert row["source_id"] not in v1["source_ids"]
        if row.get("candidate_id"):
            assert row["candidate_id"] not in v1["candidate_ids"]


def test_inventory_queries_unique(inventory):
    candidates, _specs, _report = inventory
    keys = [row["normalized_query"] for row in candidates]
    assert len(keys) == len(set(keys))


def test_provenance_valid_and_parallel_has_zero_h7(inventory):
    candidates, _specs, _report = inventory
    for row in candidates:
        assert validate_candidate_row(row) == []
        if row["proposed_final_bucket"] == "MIXED_PARALLEL":
            assert row["h7_positive"] is False
            assert row["h7_families"] == []
            assert row["provenance"].get("explicit_h7") is False


def test_no_illegal_h7_on_candidates_or_specs(inventory):
    candidates, specs, _report = inventory
    for row in candidates:
        for label in row.get("h7_families") or []:
            src, tgt = parse_h7_family_label(label)
            assert is_legal_h7_pair(src, tgt)
    for spec in specs:
        for label in spec.get("h7_families") or []:
            src, tgt = parse_h7_family_label(label)
            assert is_legal_h7_pair(src, tgt)
        if spec["source_kind"] == "authored":
            assert spec.get("authored_template_family")
            assert spec["authored_template_family"] == spec.get("scenario_family")


def test_authored_family_ids_stable_when_present(inventory):
    candidates, specs, _report = inventory
    authored = [r for r in candidates if r["source_kind"] == "authored"]
    assert authored  # at least spare(s)
    for row in authored:
        assert row["authored_template_family"]
        assert row["authored_template_family"] == row["template_group"]
    # Spec families are stable identifiers
    families = [s["authored_template_family"] for s in specs]
    assert len(families) == len(set(families))
    assert build_authored_family_specs() == specs


def test_assembly_deterministic(inventory):
    again = build_candidate_inventory()
    c1, s1, r1 = inventory
    c2, s2, r2 = again
    assert [row["candidate_uid"] for row in c1] == [row["candidate_uid"] for row in c2]
    assert s1 == s2
    assert r1["candidate_total"] == r2["candidate_total"]
    assert r1["A_personal_natural_available"] == r2["A_personal_natural_available"]
    assert r1["C_mixed_parallel_accepted"] == r2["C_mixed_parallel_accepted"]
    assert r1["D_natural_true_sequential_available"] == r2[
        "D_natural_true_sequential_available"
    ]


def test_report_sections_and_mined_not_auto_accepted(inventory):
    candidates, specs, report = inventory
    assert report["A_personal_natural_available"] >= 72
    assert report["B_environmental_natural_available"] >= 72
    assert report["C_mixed_parallel_accepted"] >= 72
    assert report["D_natural_true_sequential_available"] >= 1
    assert report["E_implicit_mined_needs_review"] >= 1
    assert len(report["F_authored_sequential_family_specs"]) >= 1
    assert len(report["G_authored_implicit_family_specs"]) >= 1
    assert len(report["H_h2_h3_hard_case_family_specs"]) >= 1
    assert "J_coverage_shortfalls" in report
    assert "K_h7_legal_family_inventory" in report
    mined = [r for r in candidates if r["source_kind"] == "mined"]
    assert mined
    assert all(r["review_status"] == "needs_review" for r in mined)
    assert all(r["provenance"].get("auto_accepted") is False for r in mined)
    assert all(s["review_status"] == "spec_only" for s in specs)


def test_todo2b_spec_capacity_closes_scarce_buckets(inventory):
    _candidates, specs, report = inventory
    planned = report["J_planned_authored_paraphrase_capacity"]
    assert planned["MIXED_IMPLICIT"] >= 80
    assert 75 <= planned["MIXED_SEQUENTIAL"] <= 85
    assert report["J_coverage_shortfalls"]["MIXED_IMPLICIT"][
        "still_short_if_specs_filled"
    ] == 0
    assert report["J_coverage_shortfalls"]["MIXED_SEQUENTIAL"][
        "still_short_if_specs_filled"
    ] == 0

    # Diversified families, not one giant template
    fam_imp = {
        s["authored_template_family"]
        for s in specs
        if s["spec_kind"] == "authored_implicit_family"
    }
    fam_seq = {
        s["authored_template_family"]
        for s in specs
        if s["spec_kind"] == "authored_sequential_family"
    }
    assert len(fam_imp) >= 10
    assert len(fam_seq) >= 12
    assert all(s["h5_positive"] is True for s in specs if s["spec_kind"] == "authored_implicit_family")

    k = report["K_h7_legal_family_inventory"]
    expected = k["expected_final_family_totals_if_specs_filled"]
    for label, minimum in k["family_minimums"].items():
        assert expected[label] >= minimum, label
    assert k["expected_final_multi_hop"] >= 10
    assert k["expected_final_h7_positive_examples"] >= 80
    for label, share in k["expected_family_share_of_h7_positive"].items():
        assert share <= 0.35 + 1e-9, (label, share)

    legacy = report["L_legacy_h5_h7"]
    assert legacy["h5_positive_by_bucket"]["MIXED_IMPLICIT"] == 24
    assert legacy["h7_positive_total"] == 15
    assert report["N_projected_corpus_fills"]["remaining_short_or_impossible"] == []
    assert (
        report["N_projected_corpus_fills"]["retrieve_possessive_h5_none_if_specs_filled"]
        >= 40
    )
    assert (
        report["N_projected_corpus_fills"]["mixed_sequential_h5_positive_if_specs_filled"]
        >= 40
    )


def test_written_artifacts_match_builder_when_present():
    """If inventory was written, it must match a fresh deterministic build."""
    cand_path = ROOT / STAGE_A_V2_CANDIDATES_PATH
    specs_path = ROOT / STAGE_A_V2_AUTHORED_SPECS_PATH
    report_path = ROOT / STAGE_A_V2_CANDIDATE_REPORT_PATH
    if not cand_path.is_file():
        pytest.skip("inventory artifacts not written on disk")
    candidates, specs, report = build_candidate_inventory()
    disk_cands = [
        json.loads(line)
        for line in cand_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    disk_specs = [
        json.loads(line)
        for line in specs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    disk_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [r["candidate_uid"] for r in disk_cands] == [
        r["candidate_uid"] for r in candidates
    ]
    assert disk_specs == specs
    assert disk_report["candidate_total"] == report["candidate_total"]
