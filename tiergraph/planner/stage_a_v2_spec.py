"""Frozen Stage-A v2 (~480) corpus-expansion constants and schemas.

Todo 1 only: targets, provenance / authored-family schema, quarantine lists,
and legal H7 family floors. Does not assemble candidates, select 480, annotate,
split, or train.

Do not mutate v1 frozen selection / annotations / split fingerprint.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from tiergraph.enums import OperatorType
from tiergraph.planner.operator_io import is_h7_pair_eligible

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

STAGE_A_V2_CORPUS_SIZE: Final[int] = 480
STAGE_A_V2_PER_BUCKET: Final[int] = 96
STAGE_A_V2_BUCKETS: Final[tuple[str, ...]] = (
    "Personal",
    "Environmental",
    "MIXED_IMPLICIT",
    "MIXED_PARALLEL",
    "MIXED_SEQUENTIAL",
)

STAGE_A_V2_H1_PERSONAL: Final[int] = 96
STAGE_A_V2_H1_ENVIRONMENTAL: Final[int] = 96
STAGE_A_V2_H1_MIXED: Final[int] = 288

STAGE_A_V2_TRAIN_SIZE: Final[int] = 384
STAGE_A_V2_DEV_SIZE: Final[int] = 48
STAGE_A_V2_TEST_SIZE: Final[int] = 48
STAGE_A_V2_SPLIT_SEED: Final[int] = 20260901

# ---------------------------------------------------------------------------
# Paths (v2 selection reserved; v1 refs read-only for freeze checks)
# ---------------------------------------------------------------------------

STAGE_A_V2_SELECTION_PATH: Final[Path] = Path(
    "dataset/planner/stage_a_v2_final_selection.jsonl"
)
STAGE_A_V2_SELECTION_REPORT_PATH: Final[Path] = Path(
    "dataset/planner/stage_a_v2_selection_report.json"
)
STAGE_A_V2_AUTHORED_CANDIDATES_PATH: Final[Path] = Path(
    "dataset/planner/stage_a_v2_authored_candidates.jsonl"
)
STAGE_A_V2_AUTHORED_REVIEWS_PATH: Final[Path] = Path(
    "dataset/planner/stage_a_v2_authored_reviews.jsonl"
)
STAGE_A_V2_AUTHORED_REVIEW_REPORT_PATH: Final[Path] = Path(
    "dataset/planner/stage_a_v2_authored_review_report.json"
)

# First-pass approval floors for NEW authored candidates (Todo 3B).
STAGE_A_V2_AUTHORED_APPROVE_FLOOR_IMPLICIT: Final[int] = 72
STAGE_A_V2_AUTHORED_APPROVE_FLOOR_SEQUENTIAL: Final[int] = 72
STAGE_A_V2_NEW_PER_BUCKET: Final[int] = 72
STAGE_A_V2_LEGACY_PER_BUCKET: Final[int] = 24
STAGE_A_V2_SELECTION_SEED: Final[int] = 20260901

# Soft-adjacent authored families: may train/dev but not publication test.
# Checked by example_is_quarantined_for_publication_test (single eligibility path).
PUBLICATION_TEST_INELIGIBLE_AUTHORED_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        "urgency_distractor_scene",  # adjacent to quarantined urgency_scene_query
    }
)

# Max share of H7-positive approved examples that may carry one H7 family label.
# Share = (# H7-positive examples containing label) / (# H7-positive examples).
# Multi-hop rows can contribute to multiple labels, so shares need not sum to 1.
H7_FAMILY_SHARE_MAX: Final[float] = 0.35

AUTHORED_REVIEW_METHOD: Final[str] = "agent_assisted_first_pass"

AUTHORED_REVIEW_STATUSES: Final[tuple[str, ...]] = (
    "APPROVE",
    "REVISE",
    "REJECT",
    "UNREVIEWED",
)

STAGE_A_V1_SELECTION_PATH: Final[Path] = Path(
    "dataset/planner/stage_a_final_selection.jsonl"
)
STAGE_A_V1_STEP_A_PATH: Final[Path] = Path(
    "dataset/planner/stage_a_step_a_annotations.jsonl"
)
STAGE_A_V1_STEP_B_PATH: Final[Path] = Path(
    "dataset/planner/stage_a_step_b_annotations.jsonl"
)
STAGE_A_V1_SPLIT_FINGERPRINT: Final[str] = (
    "7adb7e6a1f2080d965092097207f2e084d24d4a659c4042c27575fc8fac70478"
)
STAGE_A_V1_SPLIT_SEED: Final[int] = 20260831

# ---------------------------------------------------------------------------
# Provenance schema
# ---------------------------------------------------------------------------

SOURCE_KINDS: Final[tuple[str, ...]] = (
    "natural",
    "authored",
    "mined",
    "legacy_stage_a",
)

LEGACY_SOURCE_KIND_TAGS: Final[tuple[str, ...]] = (
    "stage_a_candidate_personal",
    "stage_a_candidate_environmental",
    "mixed_review",
    "mined_implicit",
    "authored_stage_a",
    "authored_stage_a_sequential",
)

PROVENANCE_FIELDS: Final[tuple[str, ...]] = (
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

AUTHORED_TEMPLATE_FAMILY_ALIASES: Final[tuple[str, ...]] = (
    "authored_template_family",
    "scenario_family",
)

# Leaf authored_template_family → parent holdout component (leakage linking).
# Bucket labels stay distinct; publication split merges via this parent ID.
AUTHORED_HOLDOUT_FAMILY_LINKS: Final[dict[str, str]] = {
    "my_reservation_seat_marker": "holdout_seat_reservation_match",
    "resolve_only_identify_seat_marker": "holdout_seat_reservation_match",
    "my_allergy_menu_safe": "holdout_food_profile_safety",
    "my_dietary_restriction_dish": "holdout_food_profile_safety",
}

# Declarative aspirational operator ranges only — never enforce as quotas.
OPERATOR_TARGET_RANGES: Final[dict[str, tuple[int, int]]] = {
    "RETRIEVE_PERSONAL": (160, 200),
    "IDENTIFY_ENVIRONMENTAL": (160, 200),
    "DESCRIBE_ENVIRONMENT": (140, 180),
    "LOCATE_ENVIRONMENTAL": (140, 180),
    "NAVIGATE_TO": (80, 120),
}

# ---------------------------------------------------------------------------
# H5 / H7 targets
# ---------------------------------------------------------------------------

H5_POSITIVE_TARGET_RANGE: Final[tuple[int, int]] = (140, 160)
H5_NEGATIVE_RETRIEVE_POSSESSIVE_MIN: Final[int] = 40

H7_POSITIVE_EXAMPLE_TARGET_RANGE: Final[tuple[int, int]] = (80, 90)
H7_EDGE_TARGET_RANGE: Final[tuple[int, int]] = (100, 120)
H7_SPLIT_FLOOR_DEV: Final[int] = 8
H7_SPLIT_FLOOR_TEST: Final[int] = 8
H7_MULTI_HOP_MINIMUM: Final[int] = 10

# Stage-A answer ops used when deriving contract-legal explicit H7 pairs
# (no explicit RESOLVE endpoints in Stage-A H7 annotation policy).
_STAGE_A_ANSWER_OPS: Final[tuple[OperatorType, ...]] = (
    OperatorType.RETRIEVE_PERSONAL,
    OperatorType.IDENTIFY_ENVIRONMENTAL,
    OperatorType.DESCRIBE_ENVIRONMENT,
    OperatorType.LOCATE_ENVIRONMENTAL,
    OperatorType.NAVIGATE_TO,
)


def h7_family_label(src_op: str, tgt_op: str) -> str:
    return f"{src_op}->{tgt_op}"


def parse_h7_family_label(label: str) -> tuple[str, str]:
    if "->" not in label:
        raise ValueError(f"invalid H7 family label {label!r}")
    src, tgt = label.split("->", 1)
    return src.strip(), tgt.strip()


def derive_legal_h7_family_labels() -> tuple[str, ...]:
    """Legal Stage-A H7 pairs = contract-eligible answer-op pairs only."""
    labels: list[str] = []
    for src in _STAGE_A_ANSWER_OPS:
        for tgt in _STAGE_A_ANSWER_OPS:
            if is_h7_pair_eligible(src, tgt):
                labels.append(h7_family_label(src.value, tgt.value))
    return tuple(labels)


# Single source of truth: derived from OPERATOR_IO_CONTRACT_V1 via eligibility.
LEGAL_H7_FAMILY_LABELS: Final[tuple[str, ...]] = derive_legal_h7_family_labels()

H7_FAMILY_MINIMUMS: Final[dict[str, int]] = {
    "IDENTIFY_ENVIRONMENTAL->LOCATE_ENVIRONMENTAL": 18,
    "LOCATE_ENVIRONMENTAL->NAVIGATE_TO": 18,
    "IDENTIFY_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT": 14,
    "LOCATE_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT": 14,
}


def is_legal_h7_pair(src_op: str, tgt_op: str) -> bool:
    """True iff the ordered pair is eligible under OPERATOR_IO_CONTRACT_V1."""
    try:
        src = OperatorType(src_op)
        tgt = OperatorType(tgt_op)
    except ValueError:
        return False
    return is_h7_pair_eligible(src, tgt)


def normalize_source_kind(source_kind: str) -> str:
    """Map fine-grained / legacy tags onto the coarse SOURCE_KINDS bucket."""
    kind = source_kind.strip()
    if kind in SOURCE_KINDS:
        return kind
    if kind in {"authored_stage_a", "authored_stage_a_sequential"}:
        return "authored"
    if kind == "mined_implicit":
        return "mined"
    if kind in LEGACY_SOURCE_KIND_TAGS:
        return "legacy_stage_a"
    return kind


def resolve_authored_template_family(row: Mapping[str, Any]) -> str | None:
    """Return authored_template_family, accepting scenario_family alias."""
    for key in AUTHORED_TEMPLATE_FAMILY_ALIASES:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def resolve_authored_holdout_family(row: Mapping[str, Any]) -> str | None:
    """Holdout component ID for authored rows (parent link or leaf family).

    Related leaf families share one parent via ``AUTHORED_HOLDOUT_FAMILY_LINKS``
    so the publication split cannot leak structurally equivalent templates.
    """
    explicit = row.get("authored_holdout_family")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    family = resolve_authored_template_family(row)
    if family is None:
        return None
    return AUTHORED_HOLDOUT_FAMILY_LINKS.get(family, family)


def require_authored_template_family(row: Mapping[str, Any]) -> str:
    """Authored rows must carry a non-empty authored_template_family."""
    family = resolve_authored_template_family(row)
    if family is None:
        raise ValueError(
            "authored example missing required authored_template_family "
            f"(aliases={AUTHORED_TEMPLATE_FAMILY_ALIASES})"
        )
    return family


def is_authored_source(source_kind: str) -> bool:
    coarse = normalize_source_kind(source_kind)
    return coarse == "authored" or source_kind.strip() in {
        "authored_stage_a",
        "authored_stage_a_sequential",
    }


def validate_authored_family_on_row(row: Mapping[str, Any]) -> list[str]:
    """Schema checks for authored_template_family / scenario_family."""
    errors: list[str] = []
    source_kind = str(row.get("source_kind") or "").strip()
    family = resolve_authored_template_family(row)
    if is_authored_source(source_kind):
        if family is None:
            errors.append(
                f"authored row {row.get('stage_a_id') or row.get('candidate_id')!r} "
                "requires authored_template_family (alias scenario_family)"
            )
    elif family is not None and normalize_source_kind(source_kind) == "natural":
        errors.append(
            f"natural row {row.get('stage_a_id') or row.get('source_id')!r} "
            "must not set authored_template_family"
        )
    primary = row.get("authored_template_family")
    alias = row.get("scenario_family")
    if (
        primary is not None
        and alias is not None
        and str(primary).strip()
        and str(alias).strip()
        and str(primary).strip() != str(alias).strip()
    ):
        errors.append(
            "authored_template_family and scenario_family disagree: "
            f"{primary!r} vs {alias!r}"
        )
    return errors


def validate_provenance_metadata(
    row: Mapping[str, Any],
    *,
    require_annotation_flags: bool = False,
) -> list[str]:
    """Validate provenance schema fields on a v2 selection / inventory row."""
    errors: list[str] = []
    for field in PROVENANCE_FIELDS:
        if field not in row:
            errors.append(f"missing provenance field {field!r}")

    source_kind = str(row.get("source_kind") or "").strip()
    if not source_kind:
        errors.append("source_kind must be non-empty")
    elif normalize_source_kind(source_kind) not in SOURCE_KINDS:
        errors.append(f"unknown source_kind {source_kind!r}")

    if not str(row.get("semantic_group") or "").strip():
        errors.append("semantic_group must be non-empty")
    if not str(row.get("template_group") or "").strip():
        errors.append("template_group must be non-empty")

    bucket = str(row.get("final_bucket") or "").strip()
    if bucket not in STAGE_A_V2_BUCKETS:
        errors.append(
            f"final_bucket must be one of {STAGE_A_V2_BUCKETS}, got {bucket!r}"
        )

    errors.extend(validate_authored_family_on_row(row))

    h7_families = row.get("h7_families")
    if h7_families is not None:
        if not isinstance(h7_families, (list, tuple)):
            errors.append("h7_families must be a list when present")
        else:
            for label in h7_families:
                text = str(label).strip()
                if not text:
                    continue
                try:
                    src, tgt = parse_h7_family_label(text)
                except ValueError:
                    errors.append(f"illegal h7_families entry {text!r}")
                    continue
                if not is_legal_h7_pair(src, tgt):
                    errors.append(f"illegal h7_families entry {text!r}")

    if require_annotation_flags:
        for flag in ("h5_positive", "h7_positive"):
            if not isinstance(row.get(flag), bool):
                errors.append(f"{flag} must be bool when annotation flags required")
        if row.get("operator_family") in (None, ""):
            errors.append("operator_family required when annotation flags required")
        if row.get("h7_families") is None:
            errors.append("h7_families required when annotation flags required")

    return errors


def hard_holdout_atoms(row: Mapping[str, Any]) -> frozenset[str]:
    """Atoms that merge into the v2 hard holdout component.

    Natural: semantic_group only.
    Authored: semantic_group ∪ authored_holdout_family (parent-linked when set).
    """
    semantic = str(row.get("semantic_group") or "").strip()
    if not semantic:
        raise ValueError("row missing semantic_group for hard holdout")
    atoms = {f"semantic:{semantic}"}
    holdout = resolve_authored_holdout_family(row)
    if holdout is not None:
        atoms.add(f"authored_holdout_family:{holdout}")
    elif is_authored_source(str(row.get("source_kind") or "")):
        raise ValueError("authored row missing authored_template_family")
    return frozenset(atoms)


# ---------------------------------------------------------------------------
# Inspected v1 free-test quarantine (publication test exclusion)
# ---------------------------------------------------------------------------

QUARANTINED_EXAMPLE_IDS: Final[frozenset[str]] = frozenset(
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

# Frozen from the inspected 12 on stage_a_final_selection.jsonl
QUARANTINED_SEMANTIC_GROUPS: Final[frozenset[str]] = frozenset(
    {
        "appointment_schedule__coord_what_time_and",
        "appointment_schedule__coord_where_and",
        "identity_documents__other_pe",
        "identity_documents__single_clause_mixed",
        "medication_health__allergies",
        "medication_health__coord_how_and",
        "medication_health__coord_is_and",
        "medication_health__what_is_my_X",
        "order_pickup_wayfinding",
        "personal_profile__single_clause_mixed",
        "personal_profile__urgent_scene",
        "shopping_finance__urgent_scene",
    }
)

# Soft linguistic templates: inspected IDs' templates + plan-listed near-duplicates
QUARANTINED_TEMPLATE_GROUPS: Final[frozenset[str]] = frozenset(
    {
        "coord_how_and",
        "coord_is_and",
        "coord_what_time_and",
        "coord_where_and",
        "other_pe",
        "resolve_locate_navigate_order_pickup",
        "single_clause_mixed",
        "urgency_scene_query",
        "what_is_my_X",
        "coord_and",
        "coord_then",
        "coord_comma",
        "where_is_my_X",
        "tell_me_about_my_X",
    }
)

# Grounded authored family from inspected sa_0106 / auth_seq_010 only
QUARANTINED_AUTHORED_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        "resolve_locate_navigate_order_pickup",
    }
)


def example_is_quarantined_for_publication_test(row: Mapping[str, Any]) -> bool:
    """True if this example must not appear in the publication test split.

    Covers inspected v1 quarantine atoms (ids / semantic / template / authored
    family) plus soft-adjacent authored families declared in
    ``PUBLICATION_TEST_INELIGIBLE_AUTHORED_FAMILIES``.
    """
    stage_a_id = str(row.get("stage_a_id") or "").strip()
    if stage_a_id in QUARANTINED_EXAMPLE_IDS:
        return True
    semantic = str(row.get("semantic_group") or "").strip()
    if semantic in QUARANTINED_SEMANTIC_GROUPS:
        return True
    template = str(row.get("template_group") or "").strip()
    if template in QUARANTINED_TEMPLATE_GROUPS:
        return True
    family = resolve_authored_template_family(row)
    if family is not None and family in QUARANTINED_AUTHORED_FAMILIES:
        return True
    if family is not None and family in PUBLICATION_TEST_INELIGIBLE_AUTHORED_FAMILIES:
        return True
    return False


def publication_test_ineligibility_reason(row: Mapping[str, Any]) -> str | None:
    """Human-readable reason when quarantined for publication test; else None."""
    stage_a_id = str(row.get("stage_a_id") or "").strip()
    if stage_a_id in QUARANTINED_EXAMPLE_IDS:
        return f"quarantined_example_id:{stage_a_id}"
    semantic = str(row.get("semantic_group") or "").strip()
    if semantic in QUARANTINED_SEMANTIC_GROUPS:
        return f"quarantined_semantic_group:{semantic}"
    template = str(row.get("template_group") or "").strip()
    if template in QUARANTINED_TEMPLATE_GROUPS:
        return f"quarantined_template_group:{template}"
    family = resolve_authored_template_family(row)
    if family is not None and family in QUARANTINED_AUTHORED_FAMILIES:
        return f"quarantined_authored_family:{family}"
    if family is not None and family in PUBLICATION_TEST_INELIGIBLE_AUTHORED_FAMILIES:
        return (
            f"publication_test_ineligible_authored_family:{family}"
            " (soft-adjacent to quarantined urgency_scene_query)"
        )
    return None


def assert_geometry_consistent() -> None:
    """Internal sanity: 96×5 and H1 shares match corpus size."""
    if STAGE_A_V2_CORPUS_SIZE != STAGE_A_V2_PER_BUCKET * len(STAGE_A_V2_BUCKETS):
        raise AssertionError("corpus size must equal per_bucket × buckets")
    if (
        STAGE_A_V2_H1_PERSONAL
        + STAGE_A_V2_H1_ENVIRONMENTAL
        + STAGE_A_V2_H1_MIXED
        != STAGE_A_V2_CORPUS_SIZE
    ):
        raise AssertionError("H1 targets must sum to corpus size")
    if (
        STAGE_A_V2_TRAIN_SIZE + STAGE_A_V2_DEV_SIZE + STAGE_A_V2_TEST_SIZE
        != STAGE_A_V2_CORPUS_SIZE
    ):
        raise AssertionError("split sizes must sum to corpus size")
    if STAGE_A_V2_SPLIT_SEED == STAGE_A_V1_SPLIT_SEED:
        raise AssertionError("v2 split seed must differ from v1")
    if set(H7_FAMILY_MINIMUMS) != set(LEGAL_H7_FAMILY_LABELS):
        raise AssertionError(
            "H7_FAMILY_MINIMUMS keys must match derived LEGAL_H7_FAMILY_LABELS"
        )
