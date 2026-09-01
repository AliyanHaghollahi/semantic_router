"""Stage-A v2 final 480-example selection (Todo 4).

Carries frozen v1 120 examples unchanged and selects 72 NEW per bucket.
Does not annotate Step A/B, split, or train.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Final

from tiergraph.planner.corpus import normalize_query_key
from tiergraph.planner.stage_a_selection import PERSONAL_CONTRACT_EXCLUDE_IDS, load_jsonl
from tiergraph.planner.stage_a_v2_candidates import STAGE_A_V2_CANDIDATES_PATH
from tiergraph.planner.stage_a_v2_spec import (
    H5_POSITIVE_TARGET_RANGE,
    H7_FAMILY_MINIMUMS,
    H7_FAMILY_SHARE_MAX,
    H7_MULTI_HOP_MINIMUM,
    OPERATOR_TARGET_RANGES,
    PUBLICATION_TEST_INELIGIBLE_AUTHORED_FAMILIES,
    QUARANTINED_AUTHORED_FAMILIES,
    QUARANTINED_EXAMPLE_IDS,
    QUARANTINED_SEMANTIC_GROUPS,
    QUARANTINED_TEMPLATE_GROUPS,
    STAGE_A_V1_SELECTION_PATH,
    STAGE_A_V1_STEP_B_PATH,
    STAGE_A_V2_AUTHORED_REVIEWS_PATH,
    STAGE_A_V2_BUCKETS,
    STAGE_A_V2_CORPUS_SIZE,
    STAGE_A_V2_LEGACY_PER_BUCKET,
    STAGE_A_V2_NEW_PER_BUCKET,
    STAGE_A_V2_PER_BUCKET,
    STAGE_A_V2_SELECTION_PATH,
    STAGE_A_V2_SELECTION_REPORT_PATH,
    STAGE_A_V2_SELECTION_SEED,
    example_is_quarantined_for_publication_test,
    h7_family_label,
    is_legal_h7_pair,
    parse_h7_family_label,
    publication_test_ineligibility_reason,
    resolve_authored_holdout_family,
    resolve_authored_template_family,
)


SELECTION_ORDERING_DOC = (
    "1) Carry all frozen v1 rows unchanged, ordered by existing stage_a_id "
    "sa_0001..sa_0120. "
    "2) Append NEW rows ordered by STAGE_A_V2_BUCKETS, then within each bucket "
    "by selection_rank / stable candidate key. "
    "3) Assign new IDs sa_0121..sa_0480 in that append order. "
    f"Selection seed={STAGE_A_V2_SELECTION_SEED} (deterministic greedy diversity). "
    "4) Apply H7_DIVERSITY_CAP_SWAP on NEW MIXED_SEQUENTIAL if needed so "
    "combined projected H7 family share stays <=35%."
)

# Minimal post-round-robin correction so combined projected LOCATE->NAVIGATE
# share stays <= H7_FAMILY_SHARE_MAX after legacy gold is included.
# Remove a non-multi-hop, H5-negative, LOCATE->NAVIGATE-only clinic row (family
# retains 2) and add the first preferred excluded IDENTIFY->DESCRIBE control.
H7_DIVERSITY_CAP_SWAP: Final[dict[str, str]] = {
    "remove_candidate_id": "auth_v2_locate_navigate_clinic_corridor_03",
    "add_candidate_id": "auth_v2_identify_describe_device_status_05",
    "reason": (
        "combined projected LOCATE_ENVIRONMENTAL->NAVIGATE_TO share exceeded "
        "35%; swap one non-multi-hop H5-negative LOCATE->NAVIGATE-only row for "
        "an APPROVED excluded IDENTIFY->DESCRIBE row"
    ),
}

_WHAT_IS_MY_TEMPLATES = frozenset(
    {
        "what_is_my_X",
        "what_is_the_X_of_my_Y",
        "tell_me_about_my_X",
    }
)


# Step-A audit repair: telephony/action personal queries incompatible with V1 ops.
ONTOLOGY_BLOCKED_PERSONAL_SOURCE_IDS: Final[frozenset[str]] = frozenset(
    {"src_0033", "src_0034"}
)
ONTOLOGY_BLOCKED_PERSONAL_STAGE_A_IDS: Final[tuple[str, ...]] = (
    "sa_0172",
    "sa_0184",
)


def _eligible_personal_replacement_pool(
    inventory: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Unused Personal natural rows eligible for ontology repair swaps."""
    sel_keys = {normalize_query_key(str(r["query"])) for r in selected}
    sel_ids = {r.get("source_id") for r in selected}
    pool = [
        dict(r)
        for r in inventory
        if r.get("proposed_final_bucket") == "Personal"
        and r.get("source_kind") == "natural"
        and r.get("review_status") == "available"
        and r.get("source_id") not in sel_ids
        and str(r.get("source_id")) not in PERSONAL_CONTRACT_EXCLUDE_IDS
        and str(r.get("source_id")) not in ONTOLOGY_BLOCKED_PERSONAL_SOURCE_IDS
        and normalize_query_key(str(r["query"])) not in sel_keys
        and r.get("template_group") != "call_my_X"
    ]
    pool.sort(key=_stable_key)
    return pool


def repair_ontology_incompatible_personal(
    selected: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Swap ontology-incompatible Personal rows in place; preserve stage_a_id."""
    out = [deepcopy(r) for r in selected]
    by_id = {str(r["stage_a_id"]): i for i, r in enumerate(out)}
    repair_log: list[dict[str, Any]] = []
    pool = _eligible_personal_replacement_pool(inventory, out)
    needed = [
        sid
        for sid in ONTOLOGY_BLOCKED_PERSONAL_STAGE_A_IDS
        if str(out[by_id[sid]].get("source_id")) in ONTOLOGY_BLOCKED_PERSONAL_SOURCE_IDS
    ]
    if not needed:
        return out, repair_log
    if len(pool) < len(needed):
        raise ValueError(
            f"ontology repair needs {len(needed)} replacements, pool has {len(pool)}"
        )
    for stage_a_id, cand in zip(needed, pool[: len(needed)]):
        idx = by_id[stage_a_id]
        old = out[idx]
        old_source = str(old.get("source_id"))
        rank = int((old.get("provenance") or {}).get("selection_rank") or 0)
        new_row = _new_row_from_candidate(
            cand,
            final_bucket="Personal",
            selection_reason=(
                f"ontology_repair_replace_{old_source}; "
                f"seed={STAGE_A_V2_SELECTION_SEED}"
            ),
            selection_rank=rank,
        )
        new_row["stage_a_id"] = stage_a_id
        prov = dict(new_row.get("provenance") or {})
        prov["ontology_repair"] = {
            "replaced_source_id": old_source,
            "replaced_query": old["query"],
            "replaced_candidate_id": old.get("candidate_id"),
            "repair_reason": (
                "telephony/action request incompatible with V1 Step-A answer operators"
            ),
        }
        new_row["provenance"] = prov
        out[idx] = new_row
        repair_log.append(
            {
                "stage_a_id": stage_a_id,
                "old_source_id": old_source,
                "old_query": old["query"],
                "new_source_id": new_row.get("source_id"),
                "new_query": new_row["query"],
                "new_candidate_id": new_row.get("candidate_id"),
            }
        )
    keys = [normalize_query_key(str(r["query"])) for r in out]
    if len(keys) != len(set(keys)):
        raise ValueError("ontology repair introduced duplicate normalized queries")
    return out, repair_log


def _stable_key(row: Mapping[str, Any]) -> str:
    return str(
        row.get("candidate_id")
        or row.get("candidate_uid")
        or row.get("source_id")
        or normalize_query_key(str(row.get("query") or ""))
    )


def diversify_greedy_select(
    pool: Sequence[Mapping[str, Any]],
    n: int,
    *,
    template_penalty: Mapping[str, int] | None = None,
    soft_template_cap: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedy template/semantic diversity select; deterministic tie-break by key."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n > len(pool):
        raise ValueError(f"cannot select {n} from pool of {len(pool)}")
    remaining = sorted((dict(r) for r in pool), key=_stable_key)
    selected: list[dict[str, Any]] = []
    template_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    penalties = dict(template_penalty or {})

    while len(selected) < n:

        def score(row: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
            tg = str(row.get("template_group") or "")
            sg = str(row.get("semantic_group") or "")
            over = 0
            if soft_template_cap is not None and template_counts[tg] >= soft_template_cap:
                over = 1
            return (
                over,
                template_counts[tg] + int(penalties.get(tg, 0)),
                semantic_counts[sg],
                template_counts[tg],
                _stable_key(row),
            )

        best = min(remaining, key=score)
        remaining.remove(best)
        selected.append(best)
        template_counts[str(best.get("template_group") or "")] += 1
        semantic_counts[str(best.get("semantic_group") or "")] += 1

    return selected, remaining


def round_robin_by_family(
    pool: Sequence[Mapping[str, Any]],
    n: int,
    *,
    family_key: str = "authored_template_family",
    prefer_first: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select n rows by round-robin across families; optional force-include set."""
    prefer = [dict(r) for r in (prefer_first or ())]
    prefer_ids = {_stable_key(r) for r in prefer}
    rest = [dict(r) for r in pool if _stable_key(r) not in prefer_ids]
    if len(prefer) > n:
        raise ValueError("prefer_first larger than n")

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rest, key=_stable_key):
        pools[str(row.get(family_key) or "")].append(row)
    family_order = sorted(pools)
    selected = list(prefer)
    idx = {f: 0 for f in family_order}
    while len(selected) < n:
        progress = False
        for fam in family_order:
            if len(selected) >= n:
                break
            i = idx[fam]
            bucket = pools[fam]
            if i < len(bucket):
                selected.append(bucket[i])
                idx[fam] = i + 1
                progress = True
        if not progress:
            break
    if len(selected) != n:
        raise ValueError(f"round-robin underfilled: got {len(selected)} want {n}")
    selected_ids = {_stable_key(r) for r in selected}
    excluded = [dict(r) for r in pool if _stable_key(r) not in selected_ids]
    excluded.sort(key=_stable_key)
    return selected, excluded


def load_legacy_selection(
    path: str | Path = STAGE_A_V1_SELECTION_PATH,
) -> list[dict[str, Any]]:
    rows = [deepcopy(r) for r in load_jsonl(path)]
    rows.sort(key=lambda r: str(r["stage_a_id"]))
    return rows


def load_inventory(
    path: str | Path = STAGE_A_V2_CANDIDATES_PATH,
) -> list[dict[str, Any]]:
    return load_jsonl(path)


def load_approved_reviews(
    path: str | Path = STAGE_A_V2_AUTHORED_REVIEWS_PATH,
) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    return [r for r in rows if r.get("review_status") == "APPROVE"]


def _new_row_from_candidate(
    row: Mapping[str, Any],
    *,
    final_bucket: str,
    selection_reason: str,
    selection_rank: int,
) -> dict[str, Any]:
    family = row.get("authored_template_family")
    holdout = row.get("authored_holdout_family")
    if holdout is None and family:
        holdout = resolve_authored_holdout_family(row)

    pub_eligible = row.get("publication_test_eligible")
    if pub_eligible is None:
        pub_eligible = not example_is_quarantined_for_publication_test(row)

    provenance = dict(row.get("provenance") or {})
    provenance.setdefault("selection_origin", "stage_a_v2_selection")
    provenance["selection_seed"] = STAGE_A_V2_SELECTION_SEED
    provenance["selection_rank"] = selection_rank
    provenance["publication_test_eligible"] = bool(pub_eligible)
    reason = publication_test_ineligibility_reason(row)
    if reason is not None:
        provenance["publication_test_ineligibility_reason"] = reason

    h7_families = list(row.get("h7_families") or [])
    return {
        "stage_a_id": None,  # filled later
        "query": row["query"],
        "normalized_query": row.get("normalized_query")
        or normalize_query_key(str(row["query"])),
        "final_bucket": final_bucket,
        "source_kind": row.get("source_kind"),
        "source_id": row.get("source_id"),
        "candidate_id": row.get("candidate_id") or row.get("candidate_uid"),
        "semantic_group": row.get("semantic_group"),
        "template_group": row.get("template_group"),
        "authored_template_family": family,
        "authored_holdout_family": holdout,
        "operator_family": row.get("operator_family"),
        "h5_positive": row.get("h5_positive"),
        "h7_positive": bool(h7_families) if row.get("h7_positive") is None else row.get("h7_positive"),
        "h7_families": h7_families,
        "multi_hop": bool(row.get("multi_hop")),
        "publication_test_eligible": bool(pub_eligible),
        "selected": True,
        "selection_reason": selection_reason,
        "legacy_stage_a": False,
        "original_label": row.get("original_label"),
        "original_review_bucket": (row.get("provenance") or {}).get(
            "original_review_bucket"
        )
        or row.get("original_review_bucket"),
        "provenance": provenance,
        "review_status": row.get("review_status"),
        "review_method": row.get("review_method"),
    }


def select_personal_new(
    inventory: Sequence[Mapping[str, Any]],
    approved: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    must = [
        r
        for r in approved
        if r.get("proposed_final_bucket") == "Personal"
        and r.get("authored_template_family") == "retrieve_possessive_h5_none"
    ]
    must = sorted(must, key=_stable_key)
    if len(must) != 16:
        raise ValueError(f"expected 16 approved possessive H5-NONE, got {len(must)}")

    natural = [
        r
        for r in inventory
        if r.get("proposed_final_bucket") == "Personal"
        and r.get("source_kind") == "natural"
    ]
    # Soft-cap what_is_my-like templates so they do not dominate the +56.
    selected_nat, leftover = diversify_greedy_select(
        natural,
        56,
        template_penalty={t: 3 for t in _WHAT_IS_MY_TEMPLATES},
        soft_template_cap=14,
    )
    excluded = [{"candidate": _stable_key(r), "reason": "personal_natural_not_selected"} for r in leftover]
    out: list[dict[str, Any]] = []
    for i, row in enumerate(must):
        out.append(
            _new_row_from_candidate(
                row,
                final_bucket="Personal",
                selection_reason="required_approved_possessive_h5_none_control",
                selection_rank=i,
            )
        )
    for i, row in enumerate(selected_nat):
        out.append(
            _new_row_from_candidate(
                row,
                final_bucket="Personal",
                selection_reason=(
                    f"diversity_select_personal_natural; seed={STAGE_A_V2_SELECTION_SEED}"
                ),
                selection_rank=100 + i,
            )
        )
    out.sort(key=lambda r: (str(r["selection_reason"]), _stable_key(r)))
    return out, excluded


def select_environmental_new(
    inventory: Sequence[Mapping[str, Any]],
    approved: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hard = [
        r
        for r in approved
        if r.get("proposed_final_bucket") == "Environmental"
    ]
    hard = sorted(hard, key=_stable_key)
    if len(hard) != 20:
        raise ValueError(f"expected 20 approved Environmental hard cases, got {len(hard)}")

    natural = [
        r
        for r in inventory
        if r.get("proposed_final_bucket") == "Environmental"
        and r.get("source_kind") == "natural"
    ]
    selected_nat, leftover = diversify_greedy_select(
        natural,
        52,
        soft_template_cap=12,
    )
    excluded = [
        {"candidate": _stable_key(r), "reason": "environmental_natural_not_selected"}
        for r in leftover
    ]
    out: list[dict[str, Any]] = []
    for i, row in enumerate(hard):
        out.append(
            _new_row_from_candidate(
                row,
                final_bucket="Environmental",
                selection_reason="required_approved_environmental_hard_case",
                selection_rank=i,
            )
        )
    for i, row in enumerate(selected_nat):
        out.append(
            _new_row_from_candidate(
                row,
                final_bucket="Environmental",
                selection_reason=(
                    f"diversity_select_environmental_natural; seed={STAGE_A_V2_SELECTION_SEED}"
                ),
                selection_rank=100 + i,
            )
        )
    out.sort(key=lambda r: (str(r["selection_reason"]), _stable_key(r)))
    return out, excluded


def select_parallel_new(
    inventory: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool = [
        r
        for r in inventory
        if r.get("proposed_final_bucket") == "MIXED_PARALLEL"
        and not r.get("h7_positive")
        and not (r.get("h7_families") or [])
    ]
    selected, leftover = diversify_greedy_select(pool, 72, soft_template_cap=16)
    excluded = [
        {
            "candidate": _stable_key(r),
            "reason": "parallel_capacity_spare_after_diversity_select",
        }
        for r in leftover
    ]
    out = [
        _new_row_from_candidate(
            row,
            final_bucket="MIXED_PARALLEL",
            selection_reason=(
                f"diversity_select_parallel_execution_independent; "
                f"seed={STAGE_A_V2_SELECTION_SEED}"
            ),
            selection_rank=i,
        )
        for i, row in enumerate(selected)
    ]
    out.sort(key=_stable_key)
    for i, row in enumerate(out):
        row["selection_rank"] = i
        row["provenance"]["selection_rank"] = i
    return out, excluded


def select_implicit_new(
    approved: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool = [
        r
        for r in approved
        if r.get("proposed_final_bucket") == "MIXED_IMPLICIT"
        and r.get("h5_positive") is True
        and not (r.get("h7_families") or [])
        and not r.get("h7_positive")
    ]
    selected, excluded_rows = round_robin_by_family(pool, 72)
    excluded = [
        {
            "candidate": r.get("candidate_id"),
            "family": r.get("authored_template_family"),
            "reason": "implicit_approved_but_over_family_round_robin_cap",
        }
        for r in excluded_rows
    ]
    out = [
        _new_row_from_candidate(
            row,
            final_bucket="MIXED_IMPLICIT",
            selection_reason="approved_authored_implicit_round_robin_family_diversity",
            selection_rank=i,
        )
        for i, row in enumerate(sorted(selected, key=_stable_key))
    ]
    return out, excluded


def select_sequential_new(
    approved: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool = [
        r
        for r in approved
        if r.get("proposed_final_bucket") == "MIXED_SEQUENTIAL"
    ]
    multi = [r for r in pool if r.get("multi_hop")]
    selected, excluded_rows = round_robin_by_family(
        pool, 72, prefer_first=sorted(multi, key=_stable_key)
    )

    # Deterministic H7 family-share correction (combined corpus <=35%).
    remove_id = H7_DIVERSITY_CAP_SWAP["remove_candidate_id"]
    add_id = H7_DIVERSITY_CAP_SWAP["add_candidate_id"]
    by_id = {str(r.get("candidate_id")): r for r in pool}
    selected_ids = {str(r.get("candidate_id")) for r in selected}
    if remove_id not in selected_ids:
        raise ValueError(f"H7 diversity swap remove id missing from selected: {remove_id}")
    if add_id in selected_ids:
        raise ValueError(f"H7 diversity swap add id already selected: {add_id}")
    if add_id not in by_id:
        raise ValueError(f"H7 diversity swap add id not in approved pool: {add_id}")
    selected = [r for r in selected if str(r.get("candidate_id")) != remove_id]
    selected.append(by_id[add_id])
    excluded_rows = [
        r for r in excluded_rows if str(r.get("candidate_id")) != add_id
    ] + [by_id[remove_id]]

    excluded = [
        {
            "candidate": r.get("candidate_id"),
            "family": r.get("authored_template_family"),
            "reason": (
                H7_DIVERSITY_CAP_SWAP["reason"]
                if str(r.get("candidate_id")) == remove_id
                else "sequential_approved_but_over_round_robin_cap_after_multihop_priority"
            ),
            "h7_families": list(r.get("h7_families") or []),
            "multi_hop": bool(r.get("multi_hop")),
        }
        for r in sorted(excluded_rows, key=_stable_key)
    ]
    out = [
        _new_row_from_candidate(
            row,
            final_bucket="MIXED_SEQUENTIAL",
            selection_reason=(
                "h7_diversity_cap_swap_add"
                if str(row.get("candidate_id")) == add_id
                else (
                    "approved_authored_sequential_multihop_priority_then_family_round_robin"
                )
            ),
            selection_rank=i,
        )
        for i, row in enumerate(sorted(selected, key=_stable_key))
    ]
    return out, excluded


def assign_new_stage_a_ids(new_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Order NEW rows by bucket order then stable key; assign sa_0121.."""
    by_bucket: dict[str, list[dict[str, Any]]] = {b: [] for b in STAGE_A_V2_BUCKETS}
    for row in new_rows:
        by_bucket[str(row["final_bucket"])].append(dict(row))
    ordered: list[dict[str, Any]] = []
    for bucket in STAGE_A_V2_BUCKETS:
        bucket_rows = sorted(by_bucket[bucket], key=_stable_key)
        ordered.extend(bucket_rows)
    if len(ordered) != STAGE_A_V2_NEW_PER_BUCKET * len(STAGE_A_V2_BUCKETS):
        raise ValueError(f"expected 360 new rows, got {len(ordered)}")
    for offset, row in enumerate(ordered):
        row["stage_a_id"] = f"sa_{121 + offset:04d}"
    return ordered


def selection_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = []
    for row in sorted(rows, key=lambda r: str(r["stage_a_id"])):
        payload.append(
            {
                "stage_a_id": row["stage_a_id"],
                "query": row["query"],
                "final_bucket": row["final_bucket"],
                "source_kind": row.get("source_kind"),
                "semantic_group": row.get("semantic_group"),
                "template_group": row.get("template_group"),
                "candidate_id": row.get("candidate_id"),
                "source_id": row.get("source_id"),
            }
        )
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_stage_a_v2_selection(
    *,
    v1_path: str | Path = STAGE_A_V1_SELECTION_PATH,
    inventory_path: str | Path = STAGE_A_V2_CANDIDATES_PATH,
    reviews_path: str | Path = STAGE_A_V2_AUTHORED_REVIEWS_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    legacy = load_legacy_selection(v1_path)
    inventory = load_inventory(inventory_path)
    approved = load_approved_reviews(reviews_path)

    if len(legacy) != 120:
        raise ValueError(f"legacy selection must be 120, got {len(legacy)}")
    legacy_by_bucket = Counter(str(r["final_bucket"]) for r in legacy)
    for bucket in STAGE_A_V2_BUCKETS:
        if legacy_by_bucket.get(bucket) != STAGE_A_V2_LEGACY_PER_BUCKET:
            raise ValueError(
                f"legacy {bucket} count {legacy_by_bucket.get(bucket)} != 24"
            )

    v1_keys = {normalize_query_key(str(r["query"])) for r in legacy}
    excluded_approved: list[dict[str, Any]] = []

    personal, excl_p = select_personal_new(inventory, approved)
    environmental, excl_e = select_environmental_new(inventory, approved)
    parallel, excl_par = select_parallel_new(inventory)
    implicit, excl_i = select_implicit_new(approved)
    sequential, excl_s = select_sequential_new(approved)
    excluded_approved.extend(excl_i)
    excluded_approved.extend(excl_s)

    new_rows = personal + environmental + parallel + implicit + sequential
    for bucket, rows in [
        ("Personal", personal),
        ("Environmental", environmental),
        ("MIXED_PARALLEL", parallel),
        ("MIXED_IMPLICIT", implicit),
        ("MIXED_SEQUENTIAL", sequential),
    ]:
        if len(rows) != STAGE_A_V2_NEW_PER_BUCKET:
            raise ValueError(f"{bucket} new count {len(rows)} != 72")

    # Uniqueness vs v1 and within new
    new_keys = [normalize_query_key(str(r["query"])) for r in new_rows]
    if len(new_keys) != len(set(new_keys)):
        raise ValueError("duplicate normalized queries among NEW selections")
    overlap = set(new_keys) & v1_keys
    if overlap:
        raise ValueError(f"NEW selection overlaps frozen v1 queries: {sorted(overlap)[:5]}")

    # Authored must be APPROVE only (already filtered); double-check no REVISE ids
    review_status = {
        r["candidate_id"]: r["review_status"]
        for r in load_jsonl(reviews_path)
        if r.get("candidate_id")
    }
    for row in new_rows:
        cid = row.get("candidate_id")
        if cid and str(cid).startswith("auth_v2_"):
            if review_status.get(cid) != "APPROVE":
                raise ValueError(f"non-APPROVE authored selected: {cid}")

    for row in new_rows:
        if row["final_bucket"] == "MIXED_PARALLEL" and (
            row.get("h7_positive") or row.get("h7_families")
        ):
            raise ValueError(f"parallel row has H7: {row.get('candidate_id')}")
        for label in row.get("h7_families") or []:
            src, tgt = parse_h7_family_label(str(label))
            if not is_legal_h7_pair(src, tgt):
                raise ValueError(f"illegal H7 on selected row: {label}")

    ordered_new = assign_new_stage_a_ids(new_rows)

    # Legacy carried forward unchanged (exact copies).
    legacy_out = [deepcopy(r) for r in legacy]
    for row in legacy_out:
        row["selected"] = True

    selected = legacy_out + ordered_new
    if len(selected) != STAGE_A_V2_CORPUS_SIZE:
        raise ValueError(f"selection size {len(selected)} != 480")

    selected, ontology_repairs = repair_ontology_incompatible_personal(
        selected, inventory
    )

    report = build_selection_report(
        selected,
        legacy_out,
        ordered_new,
        excluded_approved=excluded_approved,
        excluded_natural={
            "personal_leftover_count": len(excl_p),
            "environmental_leftover_count": len(excl_e),
            "parallel_leftover_count": len(excl_par),
            "parallel_leftover": excl_par,
        },
        ontology_repairs=ontology_repairs,
    )
    return selected, report


def _is_legacy_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("stage_a_id") or "") < "sa_0121"


def _legacy_step_b_h5_positive(rec: Mapping[str, Any]) -> bool:
    return any(
        str(d.get("implicit_resolution") or "") == "IMPLICIT_RESOLVE_PERSONAL"
        for d in (rec.get("anchor_decisions") or [])
    )


def _legacy_step_b_h7_labels(rec: Mapping[str, Any]) -> list[str]:
    ops = [str(x) for x in (rec.get("operation_types") or [])]
    labels: list[str] = []
    for dep in rec.get("dependencies") or []:
        src_i = int(dep["source_operation_index"])
        tgt_i = int(dep["target_operation_index"])
        labels.append(h7_family_label(ops[src_i], ops[tgt_i]))
    return labels


def _publication_exclusion_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    stage_a_id = str(row.get("stage_a_id") or "").strip()
    if stage_a_id in QUARANTINED_EXAMPLE_IDS:
        reasons.append("inspected_legacy_id")
    semantic = str(row.get("semantic_group") or "").strip()
    if semantic in QUARANTINED_SEMANTIC_GROUPS:
        reasons.append("semantic_group_quarantine")
    template = str(row.get("template_group") or "").strip()
    if template in QUARANTINED_TEMPLATE_GROUPS:
        reasons.append("template_group_quarantine")
    family = resolve_authored_template_family(row)
    if family is not None and family in QUARANTINED_AUTHORED_FAMILIES:
        reasons.append("quarantined_authored_family")
    if family is not None and family in PUBLICATION_TEST_INELIGIBLE_AUTHORED_FAMILIES:
        reasons.append("urgency_adjacent")
    return reasons


def _row_publication_eligible(row: Mapping[str, Any]) -> bool:
    if (
        "publication_test_eligible" in row
        and row.get("publication_test_eligible") is not None
    ):
        return bool(row["publication_test_eligible"])
    return not example_is_quarantined_for_publication_test(row)


def _h7_example_shares(
    h7_positive_rows: Sequence[Mapping[str, Any]],
    *,
    labels_for_row,
) -> tuple[int, dict[str, int], dict[str, float], str | None, float]:
    """Share(label) = examples containing label / H7-positive examples."""
    denominator = len(h7_positive_rows)
    if denominator == 0:
        return 0, {}, {}, None, 0.0
    example_counts: Counter[str] = Counter()
    for row in h7_positive_rows:
        for label in set(labels_for_row(row)):
            example_counts[str(label)] += 1
    shares = {lab: cnt / denominator for lab, cnt in sorted(example_counts.items())}
    max_label = max(shares, key=shares.get) if shares else None
    max_share = shares[max_label] if max_label is not None else 0.0
    return denominator, dict(example_counts), shares, max_label, max_share


def build_selection_report(
    selected: Sequence[Mapping[str, Any]],
    legacy: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
    *,
    excluded_approved: Sequence[Mapping[str, Any]],
    excluded_natural: Mapping[str, Any],
    ontology_repairs: Sequence[Mapping[str, Any]] | None = None,
    step_b_path: str | Path = STAGE_A_V1_STEP_B_PATH,
) -> dict[str, Any]:
    by_bucket = Counter(str(r["final_bucket"]) for r in selected)
    legacy_vs_new = {
        "legacy": len(legacy),
        "new": len(new_rows),
    }
    source_by_bucket: dict[str, dict[str, int]] = {}
    for bucket in STAGE_A_V2_BUCKETS:
        rows = [r for r in selected if r["final_bucket"] == bucket]
        source_by_bucket[bucket] = dict(Counter(str(r.get("source_kind")) for r in rows))

    semantics = {str(r.get("semantic_group")) for r in selected if r.get("semantic_group")}
    templates = {str(r.get("template_group")) for r in selected if r.get("template_group")}
    authored_families = {
        str(r.get("authored_template_family"))
        for r in selected
        if r.get("authored_template_family")
    }
    holdouts = {
        str(r.get("authored_holdout_family"))
        for r in selected
        if r.get("authored_holdout_family")
    }

    # ---- H5: legacy gold vs new provisional (do not treat unknown as negative) ----
    step_b = {r["stage_a_id"]: r for r in load_jsonl(step_b_path)}
    legacy_h5_pos = 0
    legacy_h5_neg = 0
    legacy_h5_by_bucket: Counter[str] = Counter()
    for row in legacy:
        gold = step_b[str(row["stage_a_id"])]
        if _legacy_step_b_h5_positive(gold):
            legacy_h5_pos += 1
            legacy_h5_by_bucket[str(gold.get("final_bucket") or row["final_bucket"])] += 1
        else:
            legacy_h5_neg += 1

    new_h5_pos = sum(1 for r in new_rows if r.get("h5_positive") is True)
    new_h5_neg = sum(1 for r in new_rows if r.get("h5_positive") is False)
    new_h5_unk = sum(
        1
        for r in new_rows
        if r.get("h5_positive") is None or "h5_positive" not in r
    )
    combined_h5_pos = legacy_h5_pos + new_h5_pos
    combined_h5_neg = legacy_h5_neg + new_h5_neg
    h5_lo, h5_hi = H5_POSITIVE_TARGET_RANGE

    # ---- H7: legacy gold vs new provisional ----
    legacy_h7_edge_counts: Counter[str] = Counter()
    legacy_h7_positive_rows: list[dict[str, Any]] = []
    legacy_multi = 0
    for row in legacy:
        gold = step_b[str(row["stage_a_id"])]
        labels = _legacy_step_b_h7_labels(gold)
        for lab in labels:
            legacy_h7_edge_counts[lab] += 1
        if labels:
            legacy_h7_positive_rows.append(
                {"stage_a_id": row["stage_a_id"], "h7_families": labels}
            )
        if len(gold.get("dependencies") or []) >= 2:
            legacy_multi += 1

    new_h7_edge_counts: Counter[str] = Counter()
    new_h7_positive_rows = [
        r
        for r in new_rows
        if r.get("h7_positive") or (r.get("h7_families") or [])
    ]
    for row in new_h7_positive_rows:
        for lab in row.get("h7_families") or []:
            new_h7_edge_counts[str(lab)] += 1
    new_multi = sum(1 for r in new_rows if r.get("multi_hop"))

    combined_h7_edge = Counter(legacy_h7_edge_counts)
    combined_h7_edge.update(new_h7_edge_counts)
    combined_h7_positive_rows = list(legacy_h7_positive_rows) + list(new_h7_positive_rows)
    combined_multi = legacy_multi + new_multi

    def _labels(row: Mapping[str, Any]) -> list[str]:
        return [str(x) for x in (row.get("h7_families") or [])]

    (
        _den_new,
        new_example_counts,
        new_shares,
        new_max_lab,
        new_max_share,
    ) = _h7_example_shares(new_h7_positive_rows, labels_for_row=_labels)
    (
        den_comb,
        comb_example_counts,
        comb_shares,
        comb_max_lab,
        comb_max_share,
    ) = _h7_example_shares(combined_h7_positive_rows, labels_for_row=_labels)
    (
        _den_leg,
        leg_example_counts,
        leg_shares,
        _leg_max_lab,
        _leg_max_share,
    ) = _h7_example_shares(legacy_h7_positive_rows, labels_for_row=_labels)

    # ---- publication-test eligibility breakdown ----
    pub_true = 0
    pub_false = 0
    ineligible_rows: list[Mapping[str, Any]] = []
    for row in selected:
        if _row_publication_eligible(row):
            pub_true += 1
        else:
            pub_false += 1
            ineligible_rows.append(row)

    reason_any: Counter[str] = Counter()
    reason_combos: Counter[str] = Counter()
    multi_reason = 0
    for row in ineligible_rows:
        reasons = _publication_exclusion_reasons(row)
        if not reasons and example_is_quarantined_for_publication_test(row):
            reasons = ["quarantine_helper_other"]
        if len(reasons) > 1:
            multi_reason += 1
            reason_any["multiple_reasons"] += 1
        for reason in reasons:
            reason_any[reason] += 1
        reason_combos[",".join(sorted(reasons)) or "unspecified"] += 1

    template_only = Counter(
        str(r.get("template_group"))
        for r in ineligible_rows
        if _publication_exclusion_reasons(r) == ["template_group_quarantine"]
    )

    op_counts: Counter[str] = Counter()
    for r in selected:
        for op in r.get("operator_family") or []:
            op_counts[str(op)] += 1
    op_shortfalls = {}
    for op, (lo, hi) in OPERATOR_TARGET_RANGES.items():
        got = int(op_counts.get(op, 0))
        op_shortfalls[op] = {
            "observed_on_rows_with_operator_family": got,
            "target_range": [lo, hi],
            "below_range": max(0, lo - got),
            "note": (
                "legacy/natural rows often lack operator_family until Step A/B; "
                "shortfall here is informational only"
            ),
        }

    fingerprint = selection_fingerprint(selected)
    new_by_bucket = Counter(str(r["final_bucket"]) for r in new_rows)

    return {
        "A_counts": {
            "total": len(selected),
            "per_bucket": dict(by_bucket),
            "target_per_bucket": STAGE_A_V2_PER_BUCKET,
        },
        "B_legacy_vs_new": {
            **legacy_vs_new,
            "new_per_bucket": dict(new_by_bucket),
            "legacy_ids": [r["stage_a_id"] for r in legacy],
            "new_id_range": ["sa_0121", "sa_0480"],
        },
        "C_source_kind_per_bucket": source_by_bucket,
        "D_diversity": {
            "distinct_semantic_groups": len(semantics),
            "distinct_template_groups": len(templates),
            "distinct_authored_template_families": len(authored_families),
            "distinct_authored_holdout_families": len(holdouts),
            "authored_template_families": sorted(authored_families),
            "authored_holdout_families": sorted(holdouts),
        },
        "E_h5_accounting": {
            "note": (
                "Legacy H5 from frozen Step-B gold; new H5 from provisional "
                "selection metadata. Unknown natural/hard rows are NOT counted "
                "as negative."
            ),
            "legacy_gold": {
                "h5_positive": legacy_h5_pos,
                "h5_negative": legacy_h5_neg,
                "h5_positive_by_bucket": dict(legacy_h5_by_bucket),
            },
            "new_provisional": {
                "h5_positive": new_h5_pos,
                "h5_negative": new_h5_neg,
                "h5_unknown": new_h5_unk,
            },
            "combined_projected": {
                "h5_positive": combined_h5_pos,
                "h5_negative_known": combined_h5_neg,
                "h5_unknown": new_h5_unk,
                "target_range": list(H5_POSITIVE_TARGET_RANGE),
                "within_target_range": h5_lo <= combined_h5_pos <= h5_hi,
            },
            # Backward-compatible aliases (new provisional only; do not use as corpus totals)
            "legacy_compatible_aliases_new_provisional_only": {
                "h5_positive": new_h5_pos,
                "h5_negative": new_h5_neg,
                "h5_unknown_or_absent": new_h5_unk,
            },
        },
        "F_h7_accounting": {
            "note": (
                "Legacy H7 from frozen Step-B gold dependencies; new H7 from "
                "provisional authored expected families. Shares use "
                "H7-positive examples as denominator; multi-hop may appear in "
                "multiple families."
            ),
            "legacy_gold": {
                "h7_positive_examples": len(legacy_h7_positive_rows),
                "family_edge_counts": dict(legacy_h7_edge_counts),
                "family_example_counts": leg_example_counts,
                "family_shares": {k: round(v, 4) for k, v in leg_shares.items()},
                "multi_hop": legacy_multi,
            },
            "new_provisional": {
                "h7_positive_examples": len(new_h7_positive_rows),
                "family_edge_counts": dict(new_h7_edge_counts),
                "family_example_counts": new_example_counts,
                "family_shares": {k: round(v, 4) for k, v in new_shares.items()},
                "max_h7_family": new_max_lab,
                "max_h7_family_share": round(new_max_share, 4),
                "multi_hop": new_multi,
            },
            "combined_projected": {
                "h7_positive_examples": den_comb,
                "family_edge_counts": dict(combined_h7_edge),
                "family_example_counts": comb_example_counts,
                "family_shares": {k: round(v, 4) for k, v in comb_shares.items()},
                "max_h7_family": comb_max_lab,
                "max_h7_family_share": round(comb_max_share, 4),
                "max_share_cap": H7_FAMILY_SHARE_MAX,
                "max_share_within_cap": comb_max_share <= H7_FAMILY_SHARE_MAX,
                "multi_hop": combined_multi,
                "family_minimums": dict(H7_FAMILY_MINIMUMS),
                "floors_met": {
                    lab: int(comb_example_counts.get(lab, 0)) >= need
                    for lab, need in H7_FAMILY_MINIMUMS.items()
                },
                "multi_hop_minimum": H7_MULTI_HOP_MINIMUM,
                "multi_hop_floor_met": combined_multi >= H7_MULTI_HOP_MINIMUM,
            },
        },
        "G_publication_test_eligibility": {
            "eligible": pub_true,
            "train_dev_only": pub_false,
            "train_dev_only_breakdown": {
                "reason_any_counts": dict(reason_any),
                "reason_combo_counts": dict(reason_combos),
                "multiple_reasons_rows": multi_reason,
                "template_only_template_counts": dict(template_only.most_common()),
                "legacy_ineligible": sum(1 for r in ineligible_rows if _is_legacy_row(r)),
                "new_ineligible": sum(1 for r in ineligible_rows if not _is_legacy_row(r)),
                "scientific_note": (
                    "Primary driver is soft template_group quarantine "
                    "(what_is_my_X / other_pe / coord_*), intentionally broader than "
                    "the 12 inspected legacy IDs. Do not weaken solely to raise "
                    "eligible count."
                ),
            },
        },
        "H_operator_coverage": {
            "counts": dict(op_counts),
            "range_shortfalls": op_shortfalls,
        },
        "I_selection_fingerprint": fingerprint,
        "J_excluded_approved": list(excluded_approved),
        "excluded_natural_summary": dict(excluded_natural),
        "ontology_repairs": list(ontology_repairs or []),
        "ordering_doc": SELECTION_ORDERING_DOC,
        "selection_seed": STAGE_A_V2_SELECTION_SEED,
        "H1": {
            "Personal": by_bucket.get("Personal", 0),
            "Environmental": by_bucket.get("Environmental", 0),
            "Mixed": sum(
                by_bucket.get(b, 0)
                for b in ("MIXED_IMPLICIT", "MIXED_PARALLEL", "MIXED_SEQUENTIAL")
            ),
        },
        # Keep old keys as pointers for older readers
        "E_h5_expected": {
            "DEPRECATED": "use E_h5_accounting",
            "new_provisional_only_h5_positive": new_h5_pos,
            "new_provisional_only_h5_negative": new_h5_neg,
            "new_provisional_only_h5_unknown": new_h5_unk,
            "combined_projected_h5_positive": combined_h5_pos,
        },
        "F_h7_expected": {
            "DEPRECATED": "use F_h7_accounting",
            "new_provisional_family_counts": dict(new_h7_edge_counts),
            "new_provisional_multi_hop": new_multi,
            "combined_projected_family_example_counts": comb_example_counts,
            "combined_projected_h7_positive_examples": den_comb,
            "combined_projected_multi_hop": combined_multi,
        },
    }


def validate_stage_a_v2_selection(
    selected: Sequence[Mapping[str, Any]],
    *,
    v1_path: str | Path = STAGE_A_V1_SELECTION_PATH,
    reviews_path: str | Path = STAGE_A_V2_AUTHORED_REVIEWS_PATH,
) -> list[str]:
    errors: list[str] = []
    if len(selected) != STAGE_A_V2_CORPUS_SIZE:
        errors.append(f"count {len(selected)} != 480")
    by_bucket = Counter(str(r.get("final_bucket")) for r in selected)
    for bucket in STAGE_A_V2_BUCKETS:
        if by_bucket.get(bucket) != STAGE_A_V2_PER_BUCKET:
            errors.append(f"{bucket}={by_bucket.get(bucket)} != 96")

    legacy = load_legacy_selection(v1_path)
    legacy_by_id = {r["stage_a_id"]: r for r in legacy}
    selected_by_id = {r["stage_a_id"]: r for r in selected}
    for stage_a_id, v1_row in legacy_by_id.items():
        got = selected_by_id.get(stage_a_id)
        if got is None:
            errors.append(f"missing legacy {stage_a_id}")
            continue
        for key in (
            "query",
            "final_bucket",
            "source_kind",
            "source_id",
            "semantic_group",
            "template_group",
            "provenance",
            "candidate_id",
        ):
            if got.get(key) != v1_row.get(key):
                errors.append(f"legacy {stage_a_id} field {key} mutated")

    keys = [normalize_query_key(str(r["query"])) for r in selected]
    if len(keys) != len(set(keys)):
        errors.append("duplicate normalized queries")

    review_status = {
        r["candidate_id"]: r["review_status"]
        for r in load_jsonl(reviews_path)
        if r.get("candidate_id")
    }
    for row in selected:
        cid = row.get("candidate_id")
        if cid and str(cid).startswith("auth_v2_"):
            status = review_status.get(cid)
            if status != "APPROVE":
                errors.append(f"{cid} selected with status {status}")
        if row.get("final_bucket") == "MIXED_PARALLEL":
            if row.get("h7_positive") or row.get("h7_families"):
                # legacy parallel may lack these fields
                if row.get("h7_families"):
                    errors.append(f"{row.get('stage_a_id')} parallel has h7_families")
        for label in row.get("h7_families") or []:
            src, tgt = parse_h7_family_label(str(label))
            if not is_legal_h7_pair(src, tgt):
                errors.append(f"illegal H7 {label} on {row.get('stage_a_id')}")

    ids = [r.get("stage_a_id") for r in selected]
    if len(ids) != len(set(ids)):
        errors.append("duplicate stage_a_id")
    expected_new = [f"sa_{i:04d}" for i in range(121, 481)]
    new_ids = sorted(i for i in ids if i and i >= "sa_0121")
    if new_ids != expected_new:
        errors.append("new stage_a_id sequence is not sa_0121..sa_0480 contiguous")

    return errors


def write_stage_a_v2_selection(
    *,
    output_path: str | Path = STAGE_A_V2_SELECTION_PATH,
    report_path: str | Path = STAGE_A_V2_SELECTION_REPORT_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    selected, report = build_stage_a_v2_selection(**kwargs)
    errors = validate_stage_a_v2_selection(selected, **{
        k: kwargs[k]
        for k in ("v1_path", "reviews_path")
        if k in kwargs
    })
    if errors:
        raise ValueError(f"selection validation failed: {errors[:10]}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    Path(report_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = write_stage_a_v2_selection()
    print(json.dumps({
        "A_counts": report["A_counts"],
        "B_legacy_vs_new": {
            k: report["B_legacy_vs_new"][k]
            for k in ("legacy", "new", "new_per_bucket", "new_id_range")
        },
        "I_selection_fingerprint": report["I_selection_fingerprint"],
        "F_h7_expected": report["F_h7_expected"],
        "E_h5_expected": report["E_h5_expected"],
        "G_publication_test_eligibility": report["G_publication_test_eligibility"],
        "J_excluded_approved_count": len(report["J_excluded_approved"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
