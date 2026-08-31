"""Focused Todo-1 tests: Stage-A v2 spec freeze (no selection/split/train)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tiergraph.enums import OperatorType
from tiergraph.planner.annotation_step_a import EXPECTED_STAGE_A_COUNT
from tiergraph.planner.operator_io import is_h7_pair_eligible
from tiergraph.planner.stage_a_v2_spec import (
    AUTHORED_TEMPLATE_FAMILY_ALIASES,
    H5_NEGATIVE_RETRIEVE_POSSESSIVE_MIN,
    H5_POSITIVE_TARGET_RANGE,
    H7_EDGE_TARGET_RANGE,
    H7_FAMILY_MINIMUMS,
    H7_MULTI_HOP_MINIMUM,
    H7_POSITIVE_EXAMPLE_TARGET_RANGE,
    LEGAL_H7_FAMILY_LABELS,
    OPERATOR_TARGET_RANGES,
    PROVENANCE_FIELDS,
    QUARANTINED_AUTHORED_FAMILIES,
    QUARANTINED_EXAMPLE_IDS,
    QUARANTINED_SEMANTIC_GROUPS,
    QUARANTINED_TEMPLATE_GROUPS,
    SOURCE_KINDS,
    STAGE_A_V1_SELECTION_PATH,
    STAGE_A_V1_SPLIT_FINGERPRINT,
    STAGE_A_V1_SPLIT_SEED,
    STAGE_A_V1_STEP_A_PATH,
    STAGE_A_V1_STEP_B_PATH,
    STAGE_A_V2_BUCKETS,
    STAGE_A_V2_CORPUS_SIZE,
    STAGE_A_V2_DEV_SIZE,
    STAGE_A_V2_H1_ENVIRONMENTAL,
    STAGE_A_V2_H1_MIXED,
    STAGE_A_V2_H1_PERSONAL,
    STAGE_A_V2_PER_BUCKET,
    STAGE_A_V2_SELECTION_PATH,
    STAGE_A_V2_SPLIT_SEED,
    STAGE_A_V2_TEST_SIZE,
    STAGE_A_V2_TRAIN_SIZE,
    assert_geometry_consistent,
    derive_legal_h7_family_labels,
    example_is_quarantined_for_publication_test,
    hard_holdout_atoms,
    h7_family_label,
    is_legal_h7_pair,
    normalize_source_kind,
    require_authored_template_family,
    resolve_authored_template_family,
    validate_authored_family_on_row,
    validate_provenance_metadata,
)


ROOT = Path(__file__).resolve().parent.parent

FROZEN_V1_PATHS = (
    STAGE_A_V1_SELECTION_PATH,
    STAGE_A_V1_STEP_A_PATH,
    STAGE_A_V1_STEP_B_PATH,
)


def test_geometry_and_split_sizes_are_frozen():
    assert_geometry_consistent()
    assert STAGE_A_V2_CORPUS_SIZE == 480
    assert STAGE_A_V2_PER_BUCKET == 96
    assert len(STAGE_A_V2_BUCKETS) == 5
    assert STAGE_A_V2_TRAIN_SIZE == 384
    assert STAGE_A_V2_DEV_SIZE == 48
    assert STAGE_A_V2_TEST_SIZE == 48
    assert (
        STAGE_A_V2_H1_PERSONAL,
        STAGE_A_V2_H1_ENVIRONMENTAL,
        STAGE_A_V2_H1_MIXED,
    ) == (96, 96, 288)
    assert STAGE_A_V2_SPLIT_SEED == 20260901
    assert STAGE_A_V2_SPLIT_SEED != STAGE_A_V1_SPLIT_SEED
    assert STAGE_A_V1_SPLIT_FINGERPRINT.startswith("7adb7e6a")
    assert H5_POSITIVE_TARGET_RANGE == (140, 160)
    assert H5_NEGATIVE_RETRIEVE_POSSESSIVE_MIN == 40
    assert H7_POSITIVE_EXAMPLE_TARGET_RANGE == (80, 90)
    assert H7_EDGE_TARGET_RANGE == (100, 120)


def test_v1_pipeline_count_constant_unchanged():
    assert EXPECTED_STAGE_A_COUNT == 120


def test_v1_frozen_files_match_git_head():
    for rel in FROZEN_V1_PATHS:
        path = ROOT / rel
        assert path.is_file(), rel
        result = subprocess.run(
            ["git", "diff", "--", str(rel).replace("\\", "/")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout == "", f"frozen v1 file has local diff: {rel}"
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", str(rel)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert untracked.stdout.strip() == ""


def test_provenance_schema_fields_and_validation():
    assert PROVENANCE_FIELDS == (
        "source_kind",
        "authored_template_family",
        "template_group",
        "semantic_group",
        "final_bucket",
        "operator_family",
        "h5_positive",
        "h7_positive",
        "h7_families",
    )
    assert SOURCE_KINDS == ("natural", "authored", "mined", "legacy_stage_a")
    assert normalize_source_kind("authored_stage_a_sequential") == "authored"
    assert normalize_source_kind("mined_implicit") == "mined"
    assert normalize_source_kind("stage_a_candidate_personal") == "legacy_stage_a"
    assert "scenario_family" in AUTHORED_TEMPLATE_FAMILY_ALIASES

    selection_row = {
        "source_kind": "natural",
        "authored_template_family": None,
        "template_group": "what_is_my_X",
        "semantic_group": "med__what_is_my_X",
        "final_bucket": "Personal",
        "operator_family": None,
        "h5_positive": None,
        "h7_positive": None,
        "h7_families": None,
    }
    assert validate_provenance_metadata(selection_row) == []

    incomplete = {"source_kind": "natural", "final_bucket": "Personal"}
    missing = validate_provenance_metadata(incomplete)
    assert any("missing provenance field" in err for err in missing)

    bad_bucket = {**selection_row, "final_bucket": "NOT_A_BUCKET"}
    assert any("final_bucket" in err for err in validate_provenance_metadata(bad_bucket))

    illegal_h7 = {
        **selection_row,
        "source_kind": "authored",
        "authored_template_family": "fam",
        "final_bucket": "MIXED_SEQUENTIAL",
        "h7_families": ["DESCRIBE_ENVIRONMENT->LOCATE_ENVIRONMENTAL"],
    }
    assert any("illegal h7_families" in err for err in validate_provenance_metadata(illegal_h7))

    annotated = {
        **selection_row,
        "operator_family": ["RETRIEVE_PERSONAL"],
        "h5_positive": False,
        "h7_positive": False,
        "h7_families": [],
    }
    assert validate_provenance_metadata(annotated, require_annotation_flags=True) == []
    assert validate_provenance_metadata(selection_row, require_annotation_flags=True)


def test_authored_template_family_schema_alias_and_require():
    authored = {
        "source_kind": "authored",
        "scenario_family": "locate_navigate_meeting_room",
        "semantic_group": "meeting_room_wayfinding",
    }
    assert resolve_authored_template_family(authored) == "locate_navigate_meeting_room"
    assert require_authored_template_family(authored) == "locate_navigate_meeting_room"
    assert validate_authored_family_on_row(authored) == []

    missing = {
        "source_kind": "authored_stage_a_sequential",
        "stage_a_id": "sa_9999",
        "semantic_group": "x",
    }
    assert validate_authored_family_on_row(missing)
    with pytest.raises(ValueError, match="authored_template_family"):
        require_authored_template_family(missing)

    conflict = {
        "source_kind": "authored",
        "authored_template_family": "family_a",
        "scenario_family": "family_b",
        "semantic_group": "x",
    }
    assert any("disagree" in err for err in validate_authored_family_on_row(conflict))

    natural_ok = {
        "source_kind": "natural",
        "semantic_group": "med__what_is_my_X",
        "template_group": "what_is_my_X",
    }
    assert validate_authored_family_on_row(natural_ok) == []
    natural_bad = {**natural_ok, "authored_template_family": "should_not_set"}
    assert validate_authored_family_on_row(natural_bad)


def test_hard_holdout_atoms_merge_authored_family():
    natural = {"source_kind": "natural", "semantic_group": "med__allergies"}
    assert hard_holdout_atoms(natural) == frozenset({"semantic:med__allergies"})

    authored = {
        "source_kind": "authored",
        "semantic_group": "order_pickup_wayfinding",
        "authored_template_family": "resolve_locate_navigate_order_pickup",
    }
    assert hard_holdout_atoms(authored) == frozenset(
        {
            "semantic:order_pickup_wayfinding",
            "authored_family:resolve_locate_navigate_order_pickup",
        }
    )


def test_quarantine_ids_exact_and_cover_inspected_v1_test():
    assert QUARANTINED_EXAMPLE_IDS == frozenset(
        {
            "sa_0010",
            "sa_0015",
            "sa_0040",
            "sa_0042",
            "sa_0043",
            "sa_0062",
            "sa_0066",
            "sa_0077",
            "sa_0090",
            "sa_0093",
            "sa_0106",
            "sa_0118",
        }
    )
    assert QUARANTINED_AUTHORED_FAMILIES == frozenset(
        {"resolve_locate_navigate_order_pickup"}
    )

    selection_path = ROOT / STAGE_A_V1_SELECTION_PATH
    rows = [
        json.loads(line)
        for line in selection_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {row["stage_a_id"]: row for row in rows}

    for stage_a_id in sorted(QUARANTINED_EXAMPLE_IDS):
        row = by_id[stage_a_id]
        assert row["semantic_group"] in QUARANTINED_SEMANTIC_GROUPS
        assert row["template_group"] in QUARANTINED_TEMPLATE_GROUPS
        assert example_is_quarantined_for_publication_test(row)

    assert example_is_quarantined_for_publication_test(
        {
            "stage_a_id": "sa_v2_new",
            "source_kind": "authored",
            "semantic_group": "unrelated_semantic",
            "template_group": "unrelated_template",
            "authored_template_family": "resolve_locate_navigate_order_pickup",
        }
    )
    assert not example_is_quarantined_for_publication_test(
        {
            "stage_a_id": "sa_0001",
            "source_kind": "natural",
            "semantic_group": "medication_health__am_is_are_X",
            "template_group": "am_is_are_X",
        }
    )


def test_legal_h7_families_derived_from_operator_io_contract():
    derived = derive_legal_h7_family_labels()
    assert LEGAL_H7_FAMILY_LABELS == derived
    assert set(H7_FAMILY_MINIMUMS) == set(LEGAL_H7_FAMILY_LABELS)
    assert set(LEGAL_H7_FAMILY_LABELS) == {
        "IDENTIFY_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT",
        "IDENTIFY_ENVIRONMENTAL->LOCATE_ENVIRONMENTAL",
        "LOCATE_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT",
        "LOCATE_ENVIRONMENTAL->NAVIGATE_TO",
    }

    for label in LEGAL_H7_FAMILY_LABELS:
        src_name, tgt_name = label.split("->", 1)
        assert is_h7_pair_eligible(OperatorType(src_name), OperatorType(tgt_name))
        assert is_legal_h7_pair(src_name, tgt_name)

    assert not is_legal_h7_pair("DESCRIBE_ENVIRONMENT", "LOCATE_ENVIRONMENTAL")
    assert not is_legal_h7_pair("RETRIEVE_PERSONAL", "LOCATE_ENVIRONMENTAL")
    assert not is_h7_pair_eligible(
        OperatorType.DESCRIBE_ENVIRONMENT, OperatorType.LOCATE_ENVIRONMENTAL
    )
    assert H7_FAMILY_MINIMUMS["IDENTIFY_ENVIRONMENTAL->LOCATE_ENVIRONMENTAL"] == 18
    assert H7_FAMILY_MINIMUMS["LOCATE_ENVIRONMENTAL->NAVIGATE_TO"] == 18
    assert H7_FAMILY_MINIMUMS["IDENTIFY_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT"] == 14
    assert H7_FAMILY_MINIMUMS["LOCATE_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT"] == 14
    assert H7_MULTI_HOP_MINIMUM == 10
    assert (
        h7_family_label("LOCATE_ENVIRONMENTAL", "NAVIGATE_TO")
        == "LOCATE_ENVIRONMENTAL->NAVIGATE_TO"
    )


def test_operator_ranges_are_declarative_only():
    assert set(OPERATOR_TARGET_RANGES) == {
        "RETRIEVE_PERSONAL",
        "IDENTIFY_ENVIRONMENTAL",
        "DESCRIBE_ENVIRONMENT",
        "LOCATE_ENVIRONMENTAL",
        "NAVIGATE_TO",
    }
    for low, high in OPERATOR_TARGET_RANGES.values():
        assert 0 < low <= high


def test_v2_paths_are_separate_from_v1():
    assert (ROOT / STAGE_A_V1_SELECTION_PATH).is_file()
    assert (ROOT / STAGE_A_V1_STEP_A_PATH).is_file()
    assert (ROOT / STAGE_A_V1_STEP_B_PATH).is_file()
    assert "v2" in STAGE_A_V2_SELECTION_PATH.name
    assert STAGE_A_V2_SELECTION_PATH != STAGE_A_V1_SELECTION_PATH
    assert not (ROOT / STAGE_A_V2_SELECTION_PATH).exists()
