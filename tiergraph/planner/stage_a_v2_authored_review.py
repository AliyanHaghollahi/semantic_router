"""Stage-A v2 authored-candidate review / approval workflow (Todo 3B).

Produces an explicitly reviewed pool for later 480-example selection.
Does not write final selection, annotations, splits, or train.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from tiergraph.planner.stage_a_selection import load_jsonl
from tiergraph.planner.stage_a_v2_authored_instantiate import (
    STAGE_A_V2_AUTHORED_CANDIDATES_PATH,
)
from tiergraph.planner.stage_a_v2_spec import (
    AUTHORED_HOLDOUT_FAMILY_LINKS,
    AUTHORED_REVIEW_METHOD,
    AUTHORED_REVIEW_STATUSES,
    H5_NEGATIVE_RETRIEVE_POSSESSIVE_MIN,
    H7_FAMILY_MINIMUMS,
    H7_FAMILY_SHARE_MAX,
    H7_MULTI_HOP_MINIMUM,
    STAGE_A_V1_SELECTION_PATH,
    STAGE_A_V1_STEP_A_PATH,
    STAGE_A_V1_STEP_B_PATH,
    STAGE_A_V2_AUTHORED_APPROVE_FLOOR_IMPLICIT,
    STAGE_A_V2_AUTHORED_APPROVE_FLOOR_SEQUENTIAL,
    STAGE_A_V2_AUTHORED_REVIEW_REPORT_PATH,
    STAGE_A_V2_AUTHORED_REVIEWS_PATH,
    STAGE_A_V2_SELECTION_PATH,
    example_is_quarantined_for_publication_test,
    is_legal_h7_pair,
    parse_h7_family_label,
    publication_test_ineligibility_reason,
    resolve_authored_holdout_family,
    resolve_authored_template_family,
)


REVIEW_APPROVE = "APPROVE"
REVIEW_REVISE = "REVISE"
REVIEW_REJECT = "REJECT"
REVIEW_UNREVIEWED = "UNREVIEWED"

# Prior quality-audit family classes (pre-cleanup). Used as review evidence only.
_PRIOR_PASS_FAMILIES: frozenset[str] = frozenset(
    {
        "my_medication_scene_match",
        "my_appointment_entrance_cue",
        "my_order_pickup_window",
        "my_luggage_carousel_tag",
        "my_hotel_room_key_panel",
        "my_package_locker_bank",
        "identify_locate_menu_dish_station",
        "locate_navigate_clinic_corridor",
        "locate_navigate_platform_exit",
        "locate_describe_shelf_contents",
        "locate_describe_room_layout",
        "locate_describe_vehicle_bay",
        "identify_locate_navigate_building_exit",
        "resolve_locate_navigate_rental_stall",
        "resolve_identify_locate_baggage_belt",
        "resolve_identify_describe_prescription_bottle",
        "resolve_locate_describe_appointment_room",
        "resolve_only_describe_cabin_map",
        "hyphenated_entity_identify",
        "identify_vs_describe_minimal_pair",
        "locate_vs_identify_describe",
        "retrieve_possessive_h5_none",
    }
)

_PRIOR_PASS_WITH_MINOR_FAMILIES: frozenset[str] = frozenset(
    {
        "my_allergy_menu_safe",
        "my_reservation_seat_marker",
        "my_boarding_pass_gate_board",
        "my_meeting_room_directory",
        "my_dietary_restriction_dish",
        "my_prescription_label_dose",
        "my_workspace_badge_reader",
        "my_cart_item_shelf_match",
        "identify_locate_plaque_vs_desk",
        "identify_describe_device_status",
        "identify_describe_signage_content",
        "identify_describe_packaging_label",
        "identify_locate_navigate_counter_queue",
        "resolve_only_identify_seat_marker",
        "urgency_distractor_scene",
        "navigate_direct_route",
        "retrieve_vs_describe_personal",
    }
)

# Families fixed/replaced in Todo-3A cleanup; rechecked here.
_CLEANUP_RECHECKED_FAMILIES: frozenset[str] = frozenset(
    {
        "my_train_platform_reservation",
        "resolve_locate_navigate_lab_draw_station",
        "retrieve_vs_describe_personal",
    }
)

# Hard packs approved only when genuinely useful (not forced to fill 20+20).
_HARD_APPROVE_FAMILIES: frozenset[str] = frozenset(
    {
        "hyphenated_entity_identify",
        "identify_vs_describe_minimal_pair",
        "locate_vs_identify_describe",
        "navigate_direct_route",
        "urgency_distractor_scene",
        "retrieve_possessive_h5_none",
        "retrieve_vs_describe_personal",
    }
)

# Explicit first-pass overrides (candidate_id → status, reason).
# Prefer REVISE over REJECT when salvageable; keep H7 floors intact.
_EXPLICIT_DECISIONS: dict[str, tuple[str, str]] = {
    # Weak advertised contrast: all four queries are RETRIEVE-only possessives.
    # Keep as REVISE so selection does not treat them as IDENTIFY/DESCRIBE pairs;
    # H5-NONE capacity is covered by retrieve_possessive_h5_none (16) + legacy.
    "auth_v2_retrieve_vs_describe_personal_01": (
        REVIEW_REVISE,
        "Family claims retrieve-vs-describe contrast but query is RETRIEVE-only; "
        "rewrite with a true DESCRIBE counterpart or fold into possessive H5-NONE pack",
    ),
    "auth_v2_retrieve_vs_describe_personal_02": (
        REVIEW_REVISE,
        "Family claims retrieve-vs-describe contrast but query is RETRIEVE-only; "
        "rewrite with a true DESCRIBE counterpart or fold into possessive H5-NONE pack",
    ),
    "auth_v2_retrieve_vs_describe_personal_03": (
        REVIEW_REVISE,
        "Family claims retrieve-vs-describe contrast but query is RETRIEVE-only; "
        "rewrite with a true DESCRIBE counterpart or fold into possessive H5-NONE pack",
    ),
    "auth_v2_retrieve_vs_describe_personal_04": (
        REVIEW_REVISE,
        "Family claims retrieve-vs-describe contrast but query is RETRIEVE-only; "
        "rewrite with a true DESCRIBE counterpart or fold into possessive H5-NONE pack",
    ),
}


def validate_review_status(status: str) -> list[str]:
    if status not in AUTHORED_REVIEW_STATUSES:
        return [f"invalid review_status {status!r}"]
    return []


def publication_test_eligible_for_row(row: Mapping[str, Any]) -> bool:
    """Train/dev may keep quarantine-adjacent rows; publication test must not.

    Single path: ``example_is_quarantined_for_publication_test`` in
    ``stage_a_v2_spec`` (inspected quarantine atoms + soft-adjacent families).
    """
    return not example_is_quarantined_for_publication_test(row)


def _navigate_words_in_query(query: str) -> bool:
    q = query.lower()
    needles = (
        "navigat",
        "how do i walk",
        "how do i get",
        "guide me",
        "directions",
        "walk to",
        "get to it",
        "get there",
        "reach it",
        "from here",
    )
    return any(n in q for n in needles)


def _coherence_issues(row: Mapping[str, Any]) -> list[str]:
    """Automated semantic/metadata checks used by first-pass review."""
    issues: list[str] = []
    bucket = str(row.get("proposed_final_bucket") or row.get("final_bucket") or "")
    ops = list(row.get("operator_family") or [])
    query = str(row.get("query") or "")
    family = resolve_authored_template_family(row) or ""

    for label in row.get("h7_families") or []:
        src, tgt = parse_h7_family_label(str(label))
        if not is_legal_h7_pair(src, tgt):
            issues.append(f"illegal_h7:{label}")

    if bucket == "MIXED_IMPLICIT":
        if row.get("h5_positive") is not True:
            issues.append("implicit_requires_h5_positive")
        if row.get("h7_positive") or row.get("h7_families"):
            issues.append("implicit_must_not_have_h7")
        if "NAVIGATE_TO" in ops and not _navigate_words_in_query(query):
            issues.append("navigate_ops_without_navigation_ask")

    if bucket == "MIXED_SEQUENTIAL":
        if row.get("h7_positive") and not (row.get("h7_families") or []):
            # resolve-only sequential may be H5 without learned H7
            if ops and set(ops) <= {
                "IDENTIFY_ENVIRONMENTAL",
                "DESCRIBE_ENVIRONMENT",
                "RETRIEVE_PERSONAL",
            }:
                pass
            else:
                issues.append("sequential_h7_positive_without_families")

    holdout = resolve_authored_holdout_family(row)
    expected = AUTHORED_HOLDOUT_FAMILY_LINKS.get(family, family)
    if holdout != expected:
        issues.append(
            f"holdout_mismatch:got={holdout!r}:expected={expected!r}"
        )

    if family == "my_train_platform_reservation":
        if ops != ["IDENTIFY_ENVIRONMENTAL"]:
            issues.append("train_platform_must_be_identify_only")
        if "NAVIGATE_TO" in ops:
            issues.append("train_platform_must_not_navigate")

    if family == "resolve_locate_navigate_order_locker":
        issues.append("quarantined_structural_twin_family_present")

    if family == "resolve_locate_navigate_lab_draw_station":
        if ops != ["LOCATE_ENVIRONMENTAL", "NAVIGATE_TO"]:
            issues.append("lab_draw_ops_incorrect")
        if row.get("h5_positive") is not True:
            issues.append("lab_draw_requires_h5")
        lowered = query.lower()
        if "locker" in lowered or "pickup" in lowered:
            issues.append("lab_draw_still_locker_pickup_skeleton")

    if family == "retrieve_vs_describe_personal":
        if row.get("h5_positive") is not False:
            issues.append("retrieve_vs_describe_requires_explicit_h5_none")

    if family == "retrieve_possessive_h5_none":
        if row.get("h5_positive") is not False:
            issues.append("possessive_h5_none_requires_false")

    return issues


def first_pass_review_decision(row: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic first-pass review for one authored candidate."""
    candidate_id = str(row["candidate_id"])
    family = resolve_authored_template_family(row) or ""
    bucket = str(row.get("proposed_final_bucket") or row.get("final_bucket") or "")
    issues = _coherence_issues(row)
    pub_eligible = publication_test_eligible_for_row(row)

    if candidate_id in _EXPLICIT_DECISIONS:
        status, reason = _EXPLICIT_DECISIONS[candidate_id]
        # Still block APPROVE if hard illegal issues remain.
        hard = [i for i in issues if i.startswith("illegal_h7") or "twin" in i]
        if status == REVIEW_APPROVE and hard:
            status = REVIEW_REJECT
            reason = "; ".join(hard)
        return {
            "review_status": status,
            "review_reason": reason,
            "coherence_issues": issues,
            "publication_test_eligible": pub_eligible,
            "prior_audit_class": _prior_audit_class(family),
        }

    # Hard illegal → REJECT
    hard_reject = [
        i
        for i in issues
        if i.startswith("illegal_h7")
        or i == "quarantined_structural_twin_family_present"
        or i.startswith("holdout_mismatch")
    ]
    if hard_reject:
        return {
            "review_status": REVIEW_REJECT,
            "review_reason": "; ".join(hard_reject),
            "coherence_issues": issues,
            "publication_test_eligible": pub_eligible,
            "prior_audit_class": _prior_audit_class(family),
        }

    # Metadata/coherence problems → REVISE
    if issues:
        return {
            "review_status": REVIEW_REVISE,
            "review_reason": "; ".join(issues),
            "coherence_issues": issues,
            "publication_test_eligible": pub_eligible,
            "prior_audit_class": _prior_audit_class(family),
        }

    if bucket in {"Personal", "Environmental"}:
        if family not in _HARD_APPROVE_FAMILIES:
            return {
                "review_status": REVIEW_REVISE,
                "review_reason": (
                    "hard-case family not on useful-approve list; "
                    "do not force weak hard packs into selection"
                ),
                "coherence_issues": issues,
                "publication_test_eligible": pub_eligible,
                "prior_audit_class": _prior_audit_class(family),
            }
        reason = (
            "confirmed useful hard case after cleanup metadata check"
            if family in _CLEANUP_RECHECKED_FAMILIES
            else "confirmed useful hard case; prior audit PASS/PASS_WITH_MINOR"
        )
        if family == "urgency_distractor_scene":
            reason += (
                "; quarantine-adjacent to urgency_scene_query "
                "(publication_test_eligible=false)"
            )
        return {
            "review_status": REVIEW_APPROVE,
            "review_reason": reason,
            "coherence_issues": issues,
            "publication_test_eligible": pub_eligible,
            "prior_audit_class": _prior_audit_class(family),
        }

    # MIXED buckets: approve after confirming metadata post-cleanup.
    if family in _CLEANUP_RECHECKED_FAMILIES:
        reason = (
            "Todo-3A cleanup rechecked: operators/H5/H7 and family semantics OK"
        )
    elif family in _PRIOR_PASS_FAMILIES:
        reason = "prior PASS confirmed; metadata still coherent after cleanup"
    elif family in _PRIOR_PASS_WITH_MINOR_FAMILIES:
        reason = (
            "prior PASS_WITH_MINOR confirmed; minor cadence/soft-H7 concerns "
            "tolerated for capacity, metadata coherent"
        )
    else:
        reason = "new/replacement family confirmed coherent for approval"

    return {
        "review_status": REVIEW_APPROVE,
        "review_reason": reason,
        "coherence_issues": issues,
        "publication_test_eligible": pub_eligible,
        "prior_audit_class": _prior_audit_class(family),
    }


def _prior_audit_class(family: str) -> str:
    if family in _CLEANUP_RECHECKED_FAMILIES and family not in _PRIOR_PASS_WITH_MINOR_FAMILIES:
        if family == "resolve_locate_navigate_lab_draw_station":
            return "REPLACED_CLEANUP"
        if family == "my_train_platform_reservation":
            return "REVISED_CLEANUP"
        return "CLEANUP_RECHECK"
    if family in _PRIOR_PASS_FAMILIES:
        return "PASS"
    if family in _PRIOR_PASS_WITH_MINOR_FAMILIES:
        return "PASS_WITH_MINOR"
    return "UNCLASSIFIED"


def _copy_candidate_with_review(
    row: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, Any]:
    out = deepcopy(dict(row))
    out["review_status"] = decision["review_status"]
    out["review_reason"] = decision["review_reason"]
    out["coherence_issues"] = list(decision.get("coherence_issues") or [])
    out["publication_test_eligible"] = bool(decision["publication_test_eligible"])
    out["prior_audit_class"] = decision.get("prior_audit_class")
    out["review_method"] = AUTHORED_REVIEW_METHOD
    # Preserve acceptance/rejection slots used by instantiate.
    if decision["review_status"] == REVIEW_APPROVE:
        out["acceptance_reason"] = decision["review_reason"]
        out["rejection_reason"] = None
    elif decision["review_status"] == REVIEW_REJECT:
        out["acceptance_reason"] = None
        out["rejection_reason"] = decision["review_reason"]
    else:
        out["acceptance_reason"] = None
        out["rejection_reason"] = decision["review_reason"]
    provenance = dict(out.get("provenance") or {})
    provenance["review_origin"] = "stage_a_v2_authored_review_first_pass"
    provenance["review_method"] = AUTHORED_REVIEW_METHOD
    provenance["publication_test_eligible"] = out["publication_test_eligible"]
    ineligible_reason = publication_test_ineligibility_reason(out)
    if ineligible_reason is not None:
        provenance["publication_test_ineligibility_reason"] = ineligible_reason
    out["provenance"] = provenance
    return out


def h7_family_shares_among_positive(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, dict[str, float], str | None, float]:
    """H7 family share among H7-positive examples.

    Share(label) = (# H7-positive examples whose h7_families contain label)
                   / (# H7-positive examples).

    Multi-hop examples may belong to more than one label, so shares need not
    sum to 1.0. Returns (denominator, shares_by_label, max_label, max_share).
    """
    h7_positive = [r for r in rows if r.get("h7_positive")]
    denominator = len(h7_positive)
    if denominator == 0:
        return 0, {}, None, 0.0
    label_example_counts: Counter[str] = Counter()
    for row in h7_positive:
        for label in set(row.get("h7_families") or []):
            label_example_counts[str(label)] += 1
    shares = {
        label: count / denominator
        for label, count in sorted(label_example_counts.items())
    }
    max_label = None
    max_share = 0.0
    if shares:
        max_label = max(shares, key=shares.get)
        max_share = shares[max_label]
    return denominator, shares, max_label, max_share


def load_authored_candidates(
    path: str | Path = STAGE_A_V2_AUTHORED_CANDIDATES_PATH,
) -> list[dict[str, Any]]:
    return load_jsonl(path)


def build_authored_reviews(
    *,
    candidates_path: str | Path = STAGE_A_V2_AUTHORED_CANDIDATES_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply deterministic first-pass review to the authored candidate pool."""
    candidates = load_authored_candidates(candidates_path)
    reviews: list[dict[str, Any]] = []
    for row in candidates:
        decision = first_pass_review_decision(row)
        errors = validate_review_status(str(decision["review_status"]))
        if errors:
            raise ValueError(f"{row.get('candidate_id')}: {errors}")
        reviewed = _copy_candidate_with_review(row, decision)
        # Identity / provenance invariants
        if reviewed["candidate_id"] != row["candidate_id"]:
            raise ValueError("candidate_id mutated during review")
        if reviewed.get("provenance", {}).get("authored_template_family") != row.get(
            "provenance", {}
        ).get("authored_template_family"):
            raise ValueError("provenance authored_template_family mutated")
        reviews.append(reviewed)

    reviews = sorted(
        reviews,
        key=lambda r: (
            r["proposed_final_bucket"],
            r["authored_template_family"],
            r["candidate_id"],
        ),
    )
    report = build_review_report(reviews)
    return reviews, report


def build_review_report(reviews: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(r["review_status"]) for r in reviews)
    approved = [r for r in reviews if r["review_status"] == REVIEW_APPROVE]
    revise = [r for r in reviews if r["review_status"] == REVIEW_REVISE]
    rejected = [r for r in reviews if r["review_status"] == REVIEW_REJECT]

    approved_by_bucket = Counter(
        str(r.get("proposed_final_bucket") or r.get("final_bucket")) for r in approved
    )
    h5_pos = sum(1 for r in approved if r.get("h5_positive") is True)
    h5_neg = sum(1 for r in approved if r.get("h5_positive") is False)
    h5_unk = sum(1 for r in approved if r.get("h5_positive") is None)

    h7_family_counts = Counter(
        label for r in approved for label in (r.get("h7_families") or [])
    )
    multi_hop = sum(1 for r in approved if r.get("multi_hop"))

    pub_true = sum(1 for r in approved if r.get("publication_test_eligible") is True)
    pub_false = sum(1 for r in approved if r.get("publication_test_eligible") is False)
    train_dev_only = [
        {
            "candidate_id": r["candidate_id"],
            "authored_template_family": r["authored_template_family"],
            "reason": publication_test_ineligibility_reason(r),
        }
        for r in approved
        if r.get("publication_test_eligible") is False
    ]

    family_counts = Counter(
        str(r["authored_template_family"]) for r in approved
    )
    holdout_counts = Counter(
        str(r.get("authored_holdout_family") or r["authored_template_family"])
        for r in approved
    )

    # H7 dominance: share of H7-positive examples containing each H7 family label.
    (
        h7_positive_count,
        h7_family_shares,
        max_h7_family,
        max_h7_family_share,
    ) = h7_family_shares_among_positive(approved)
    h7_positive_approved = [r for r in approved if r.get("h7_positive")]
    assert h7_positive_count == len(h7_positive_approved)

    # Possessive H5-NONE capacity among approved authored rows
    possessive_h5_none = sum(
        1
        for r in approved
        if r.get("authored_template_family") == "retrieve_possessive_h5_none"
        and r.get("h5_positive") is False
        and "RETRIEVE_PERSONAL" in (r.get("operator_family") or [])
    )

    imp_approved = approved_by_bucket.get("MIXED_IMPLICIT", 0)
    seq_approved = approved_by_bucket.get("MIXED_SEQUENTIAL", 0)
    shortfalls = {
        "MIXED_IMPLICIT": max(
            0, STAGE_A_V2_AUTHORED_APPROVE_FLOOR_IMPLICIT - imp_approved
        ),
        "MIXED_SEQUENTIAL": max(
            0, STAGE_A_V2_AUTHORED_APPROVE_FLOOR_SEQUENTIAL - seq_approved
        ),
    }

    h7_floor_gaps = {
        label: max(0, need - int(h7_family_counts.get(label, 0)))
        for label, need in H7_FAMILY_MINIMUMS.items()
    }
    multi_hop_gap = max(0, H7_MULTI_HOP_MINIMUM - multi_hop)

    illegal_h7 = []
    for r in approved:
        for label in r.get("h7_families") or []:
            src, tgt = parse_h7_family_label(str(label))
            if not is_legal_h7_pair(src, tgt):
                illegal_h7.append(
                    {"candidate_id": r["candidate_id"], "h7_family": label}
                )

    # Linked holdout pairs must share parent among approved+revise retained metadata
    link_check: dict[str, list[str]] = {}
    for leaf, parent in AUTHORED_HOLDOUT_FAMILY_LINKS.items():
        link_check.setdefault(parent, [])
        for r in reviews:
            if r.get("authored_template_family") == leaf:
                link_check[parent].append(
                    f"{r['candidate_id']}:{r.get('authored_holdout_family')}"
                )

    revise_ids = [
        {"candidate_id": r["candidate_id"], "reason": r.get("review_reason")}
        for r in revise
    ]
    reject_ids = [
        {"candidate_id": r["candidate_id"], "reason": r.get("review_reason")}
        for r in rejected
    ]

    return {
        "A_status_counts": {
            status: int(status_counts.get(status, 0))
            for status in AUTHORED_REVIEW_STATUSES
        },
        "B_approved_counts_by_bucket": dict(approved_by_bucket),
        "C_approved_h5_distribution": {
            "h5_positive": h5_pos,
            "h5_negative": h5_neg,
            "h5_unknown": h5_unk,
            "approved_possessive_retrieve_h5_none": possessive_h5_none,
            "corpus_h5_none_floor": H5_NEGATIVE_RETRIEVE_POSSESSIVE_MIN,
            "note": (
                "Full-corpus >=40 H5-NONE uses approved authored possessive "
                "controls plus legacy/natural RETRIEVE+possessive rows"
            ),
        },
        "D_approved_h7_family_distribution": dict(h7_family_counts),
        "D2_approved_h7_family_shares": {
            "definition": (
                "share(label) = (# H7-positive approved examples containing label) "
                "/ (# H7-positive approved examples); multi-hop may count in "
                "multiple labels so shares need not sum to 1"
            ),
            "h7_positive_denominator": h7_positive_count,
            "shares": {
                label: round(share, 4) for label, share in h7_family_shares.items()
            },
            "max_h7_family": max_h7_family,
            "max_h7_family_share": round(max_h7_family_share, 4),
            "max_share_cap": H7_FAMILY_SHARE_MAX,
        },
        "E_approved_multi_hop_count": multi_hop,
        "F_publication_test_eligibility": {
            "approved_publication_test_eligible": pub_true,
            "approved_train_dev_only": pub_false,
            "eligibility_path": (
                "not example_is_quarantined_for_publication_test(row) "
                "from stage_a_v2_spec"
            ),
            "train_dev_only_candidates": train_dev_only,
        },
        "G_family_diversity": {
            "approved_distinct_authored_template_families": len(family_counts),
            "approved_family_counts": dict(sorted(family_counts.items())),
            "approved_holdout_family_counts": dict(sorted(holdout_counts.items())),
            "h7_positive_approved_count": h7_positive_count,
            "max_h7_family": max_h7_family,
            "max_h7_family_share": round(max_h7_family_share, 4),
            "h7_family_shares": {
                label: round(share, 4) for label, share in h7_family_shares.items()
            },
        },
        "review_method": AUTHORED_REVIEW_METHOD,
        "H_shortfalls": {
            "approve_floor_shortfalls": shortfalls,
            "h7_family_floor_gaps": h7_floor_gaps,
            "multi_hop_gap": multi_hop_gap,
            "illegal_h7_on_approved": illegal_h7,
        },
        "I_revise_reject_ids": {
            "revise": revise_ids,
            "reject": reject_ids,
        },
        "holdout_link_audit": link_check,
        "reviewed_total": len(reviews),
        "approved_total": len(approved),
        "targets": {
            "implicit_approve_floor": STAGE_A_V2_AUTHORED_APPROVE_FLOOR_IMPLICIT,
            "sequential_approve_floor": STAGE_A_V2_AUTHORED_APPROVE_FLOOR_SEQUENTIAL,
            "h7_family_minimums": dict(H7_FAMILY_MINIMUMS),
            "multi_hop_minimum": H7_MULTI_HOP_MINIMUM,
        },
        "frozen_v1_paths_untouched": [
            str(STAGE_A_V1_SELECTION_PATH),
            str(STAGE_A_V1_STEP_A_PATH),
            str(STAGE_A_V1_STEP_B_PATH),
        ],
        "final_selection_not_created": str(STAGE_A_V2_SELECTION_PATH),
    }


def write_authored_reviews(
    *,
    candidates_path: str | Path = STAGE_A_V2_AUTHORED_CANDIDATES_PATH,
    reviews_path: str | Path = STAGE_A_V2_AUTHORED_REVIEWS_PATH,
    report_path: str | Path = STAGE_A_V2_AUTHORED_REVIEW_REPORT_PATH,
) -> dict[str, Any]:
    reviews, report = build_authored_reviews(candidates_path=candidates_path)

    # Gate: must clear Implicit/Sequential floors and H7 legality for APPROVE set.
    shortfalls = report["H_shortfalls"]["approve_floor_shortfalls"]
    if any(shortfalls.values()):
        raise ValueError(f"approval floor shortfalls: {shortfalls}")
    if report["H_shortfalls"]["illegal_h7_on_approved"]:
        raise ValueError(
            f"illegal H7 on approved: {report['H_shortfalls']['illegal_h7_on_approved']}"
        )
    for label, gap in report["H_shortfalls"]["h7_family_floor_gaps"].items():
        if gap > 0:
            raise ValueError(f"approved H7 floor gap for {label}: {gap}")
    if report["H_shortfalls"]["multi_hop_gap"] > 0:
        raise ValueError(
            f"approved multi-hop gap: {report['H_shortfalls']['multi_hop_gap']}"
        )
    if report["G_family_diversity"]["max_h7_family_share"] > H7_FAMILY_SHARE_MAX:
        raise ValueError(
            "one H7 family dominates approved H7-positive pool: "
            f"{report['G_family_diversity']['max_h7_family']} "
            f"share={report['G_family_diversity']['max_h7_family_share']} "
            f"(cap={H7_FAMILY_SHARE_MAX})"
        )

    reviews_path = Path(reviews_path)
    reviews_path.parent.mkdir(parents=True, exist_ok=True)
    with reviews_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in reviews:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    Path(report_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = write_authored_reviews()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
