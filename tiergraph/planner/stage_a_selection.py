"""Deterministic Stage-A final selection (120) with execution-based Mixed labels.

Rebuilds Personal / Environmental / Mixed buckets using:
- human Mixed reviews
- execution-dependency reclassification for Parallel vs Sequential
- fixed authored implicit / sequential ID sets

Does not invent train/dev/test split fields.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tiergraph.planner.authored_implicit import (
    DEFAULT_AUTHORED_CANDIDATES_PATH,
    DEFAULT_AUTHORED_REVIEWS_PATH,
    AuthoredReviewStatus as ImplicitReviewStatus,
    load_authored_candidates,
    load_authored_reviews,
)
from tiergraph.planner.authored_sequential import (
    DEFAULT_AUTHORED_SEQUENTIAL_CANDIDATES_PATH,
    DEFAULT_AUTHORED_SEQUENTIAL_REVIEWS_PATH,
    AuthoredReviewStatus as SequentialReviewStatus,
    load_authored_sequential_candidates,
    load_authored_sequential_reviews,
)
from tiergraph.planner.corpus import (
    DEFAULT_CANDIDATE_SEED,
    StageACandidate,
    load_candidates_jsonl,
    normalize_query_key,
)
from tiergraph.planner.implicit_mining import (
    DEFAULT_OUTPUT_PATH as DEFAULT_IMPLICIT_CANDIDATES_PATH,
    load_implicit_candidates_jsonl,
)
from tiergraph.planner.mixed_review import (
    MixedReviewBucket,
    MixedReviewRecord,
    load_mixed_reviews,
)


DEFAULT_CANDIDATES_PATH = Path("dataset/planner/stage_a_candidates.jsonl")
DEFAULT_MIXED_REVIEWS_PATH = Path("dataset/planner/stage_a_mixed_reviews.jsonl")
DEFAULT_SELECTION_PATH = Path("dataset/planner/stage_a_final_selection.jsonl")
DEFAULT_SPARES_PATH = Path("dataset/planner/stage_a_spares.jsonl")

STAGE_A_TOTAL = 120
PER_BUCKET = 24
SELECTION_SEED = DEFAULT_CANDIDATE_SEED

FINAL_BUCKET_PERSONAL = "Personal"
FINAL_BUCKET_ENVIRONMENTAL = "Environmental"
FINAL_BUCKET_MIXED_IMPLICIT = "MIXED_IMPLICIT"
FINAL_BUCKET_MIXED_PARALLEL = "MIXED_PARALLEL"
FINAL_BUCKET_MIXED_SEQUENTIAL = "MIXED_SEQUENTIAL"

BUCKET_ORDER: tuple[str, ...] = (
    FINAL_BUCKET_PERSONAL,
    FINAL_BUCKET_ENVIRONMENTAL,
    FINAL_BUCKET_MIXED_IMPLICIT,
    FINAL_BUCKET_MIXED_PARALLEL,
    FINAL_BUCKET_MIXED_SEQUENTIAL,
)

SOURCE_KIND_PERSONAL_POOL = "stage_a_candidate_personal"
SOURCE_KIND_ENVIRONMENTAL_POOL = "stage_a_candidate_environmental"
SOURCE_KIND_MIXED_REVIEW = "mixed_review"
SOURCE_KIND_MINED_IMPLICIT = "mined_implicit"
SOURCE_KIND_AUTHORED_IMPLICIT = "authored_stage_a"
SOURCE_KIND_AUTHORED_SEQUENTIAL = "authored_stage_a_sequential"

# Exact recovered MIXED_IMPLICIT source IDs.
MINED_IMPLICIT_SELECTED_IDS: tuple[str, ...] = (
    "src_0264",
    "src_0141",
    "src_0008",
)

# Exact authored MIXED_IMPLICIT core / spare IDs.
AUTHORED_IMPLICIT_SELECTED_IDS: tuple[str, ...] = (
    "auth_imp_001",
    "auth_imp_002",
    "auth_imp_003",
    "auth_imp_004",
    "auth_imp_005",
    "auth_imp_006",
    "auth_imp_009",
    "auth_imp_010",
    "auth_imp_011",
)
AUTHORED_IMPLICIT_SPARE_IDS: tuple[str, ...] = (
    "auth_imp_007",
    "auth_imp_008",
    "auth_imp_012",
)

# Raw true sequential IDs from execution audit (9).
TRUE_SEQUENTIAL_RAW_IDS: frozenset[str] = frozenset(
    {
        "src_0350",
        "src_0456",
        "src_0458",
        "src_0644",
        "src_0645",
        "src_0682",
        "src_0683",
        "src_0684",
        "src_0687",
    }
)

# Diversity-core natural sequential (7).
NATURAL_SEQUENTIAL_SELECTED_IDS: tuple[str, ...] = (
    "src_0350",
    "src_0456",
    "src_0458",
    "src_0644",
    "src_0682",
    "src_0683",
    "src_0687",
)
NATURAL_SEQUENTIAL_SPARE_IDS: tuple[str, ...] = (
    "src_0645",
    "src_0684",
)

# Exact authored sequential core / spare IDs.
AUTHORED_SEQUENTIAL_SELECTED_IDS: tuple[str, ...] = (
    "auth_seq_001",
    "auth_seq_002",
    "auth_seq_003",
    "auth_seq_004",
    "auth_seq_005",
    "auth_seq_006",
    "auth_seq_007",
    "auth_seq_008",
    "auth_seq_009",
    "auth_seq_010",
    "auth_seq_011",
    "auth_seq_012",
    "auth_seq_013",
    "auth_seq_014",
    "auth_seq_017",
    "auth_seq_019",
    "auth_seq_020",
)
AUTHORED_SEQUENTIAL_SPARE_IDS: tuple[str, ...] = (
    "auth_seq_015",
    "auth_seq_016",
    "auth_seq_018",
)

RECLASSIFY_REASON_FUSION_ONLY = "fusion_only_no_nonfusion_dependency"

_PARALLEL_TEMPLATE_SOFT_CAP = 3
_PE_TEMPLATE_SOFT_CAP = 4

_DOMAIN_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "medication_health",
        (
            "medication",
            "prescription",
            "pill",
            "pharmacy",
            "dosage",
            "dose",
            "metformin",
            "blood thinner",
            "blood pressure",
            "cardiologist",
            "health condition",
            "diabetic",
            "allergen",
            "allerg",
            "shellfish",
            "gluten",
            "lactose",
            "dietary",
            "nutrition",
            "ingredients",
            "empty stomach",
        ),
    ),
    (
        "food_dining",
        (
            "menu",
            "dish",
            "food",
            "eat",
            "restaurant",
            "tray",
            "snack",
            "takeout",
            "coupon",
        ),
    ),
    (
        "travel_transit",
        (
            "flight",
            "gate",
            "boarding",
            "airport",
            "terminal",
            "plane",
            "taxi",
            "bus",
            "train",
            "platform",
            "timetable",
            "departure",
            "connection",
            "luggage",
            "suitcase",
            "boarding pass",
            "hotel",
            "check-in",
            "check-out",
            "seat",
            "aisle",
            "window",
            "carousel",
            "belt",
        ),
    ),
    (
        "appointment_schedule",
        (
            "appointment",
            "meeting",
            "doctor",
            "clinic",
            "waiting area",
            "reservation",
            "booking",
            "schedule",
        ),
    ),
    (
        "shopping_finance",
        (
            "price",
            "afford",
            "shopping",
            "store",
            "brand",
            "atm",
            "cash",
            "card",
            "balance",
            "invoice",
            "receipt",
            "account",
            "reward points",
            "insurance",
            "tax id",
            "shipping",
            "coupon",
        ),
    ),
    (
        "identity_documents",
        (
            "name",
            "form",
            "contract",
            "letter",
            "document",
            "ticket",
            "label",
            "tag",
            "directory",
            "sign",
            "map",
            "chart",
            "screen",
            "board",
            "newspaper",
            "magazine",
            "page",
        ),
    ),
    (
        "personal_profile",
        (
            "my ",
            " i ",
            "brother",
            "mom",
            "mother",
            "emergency contact",
            "supervisor",
            "membership",
            "library",
            "wifi",
            "wi-fi",
            "clothing size",
            "usual size",
        ),
    ),
    (
        "scene_objects",
        (
            "object",
            "item",
            "identify",
            "pinpoint",
            "species",
            "tree",
            "charger",
            "backpack",
            "bench",
            "toys",
            "floor",
            "ground",
            "sidewalk",
            "material",
            "lights",
            "pool",
            "classroom",
            "cars",
            "people",
            "crowded",
            "path",
            "door",
            "private",
            "public",
        ),
    ),
)


def infer_domain(query: str) -> str:
    text = f" {normalize_query_key(query)} "
    for domain, keywords in _DOMAIN_RULES:
        for keyword in keywords:
            if keyword in text or keyword in text.strip():
                return domain
    return "other"


def infer_pe_template_group(query: str) -> str:
    q = normalize_query_key(query)
    if q.startswith("what is my ") or q.startswith("what's my "):
        return "what_is_my_X"
    if q.startswith("who is my "):
        return "who_is_my_X"
    if q.startswith("what is the ") and " my " in q:
        return "what_is_the_X_of_my_Y"
    if q.startswith("call my ") or q.startswith("call my"):
        return "call_my_X"
    if q.startswith("do i have ") or q.startswith("did i "):
        return "do_did_i_X"
    if q.startswith("am i ") or q.startswith("are the ") or q.startswith("is the "):
        return "am_is_are_X"
    if q.startswith("is my ") or q.startswith("is this "):
        return "is_my_or_this_X"
    if q.startswith("how many ") or q.startswith("how much ") or q.startswith("how far "):
        return "how_quantity_X"
    if q.startswith("can you ") or q.startswith("identify ") or q.startswith("name the "):
        return "identify_or_name_X"
    if q.startswith("read "):
        return "read_X"
    if q.startswith("what does ") or q.startswith("what is written ") or q.startswith(
        "which languages "
    ):
        return "read_or_interpret_text"
    if q.startswith("what material ") or q.startswith("what physical ") or q.startswith(
        "summarize the objects "
    ):
        return "enumerate_objects"
    if q.startswith("is there ") or q.startswith("are there "):
        return "is_there_X"
    if "this second" in q or "at once" in q or "immediately" in q or "without delay" in q:
        return "urgency_scene_query"
    if q.startswith("let me know ") or q.startswith("tell me "):
        return "tell_me_scene_X"
    if q.startswith("pinpoint "):
        return "identify_or_name_X"
    return "other_pe"


def infer_mixed_template_group(query: str) -> str:
    q = normalize_query_key(query)
    if q.startswith("read ") and " and " in q:
        if " and check " in q:
            return "read_X_and_check"
        if " and tell me " in q:
            return "read_X_and_tell_me"
        return "read_X_and_Y"
    if " and " not in q:
        return "single_clause_mixed"
    left, _right = q.split(" and ", 1)
    if left.startswith("what does ") and (" say" in left or " show" in left):
        return "coord_what_does_say_and"
    if left.startswith("what is written"):
        return "coord_what_is_written_and"
    if left.startswith("what is on ") or left.startswith("what is in "):
        return "coord_what_is_on_in_and"
    if left.startswith("what is my ") or left.startswith("what is the "):
        return "coord_what_is_and"
    if left.startswith("what time"):
        return "coord_what_time_and"
    if left.startswith("what food") or left.startswith("what dish") or left.startswith(
        "what medication"
    ):
        return "coord_what_entity_and"
    if left.startswith("what "):
        return "coord_what_other_and"
    if left.startswith("where "):
        return "coord_where_and"
    if left.startswith("how "):
        return "coord_how_and"
    if left.startswith("can "):
        return "coord_can_and"
    if left.startswith("is this ") or left.startswith("is my ") or left.startswith(
        "is the "
    ):
        return "coord_is_and"
    if left.startswith("am i "):
        return "coord_am_i_and"
    return "coord_other_and"


def infer_mixed_semantic_group(query: str, domain: str) -> str:
    q = normalize_query_key(query)
    template = infer_mixed_template_group(query)
    if "sign say" in q and ("gate" in q or "route" in q or "access" in q):
        return f"{domain}__sign_text_navigation"
    if "close" in q and "appointment" in q:
        return f"{domain}__closing_time_vs_appointment"
    if ("can i eat" in q or "allowed to eat" in q or "anything i can eat" in q) and (
        "allerg" in q or "diet" in q or "menu" in q or "food" in q or "dish" in q
    ):
        return f"{domain}__food_safety_personal"
    if "prescription" in q and ("label" in q or "match" in q or "bottle" in q):
        return f"{domain}__prescription_label_match"
    if "medication" in q and ("mine" in q or "my " in q or "i take" in q):
        return f"{domain}__medication_identity_personal"
    if "gate" in q and ("how do i get" in q or "way to" in q or "path" in q):
        return f"{domain}__gate_wayfinding"
    if "flight" in q and (
        "delayed" in q or "listed" in q or "on time" in q or "board" in q
    ):
        return f"{domain}__flight_status_board"
    if template.startswith("read_"):
        object_m = re.match(r"read (?:the |this )?(.+?) and ", q)
        obj = object_m.group(1).replace(" ", "_")[:40] if object_m else "text"
        return f"{domain}__read_{obj}"
    return f"{domain}__{template}"


def infer_pe_semantic_group(query: str, domain: str, template_group: str) -> str:
    q = normalize_query_key(query)
    if "call my" in q:
        return f"{domain}__contact_call"
    if "medication" in q or "prescription" in q or "dosage" in q or "metformin" in q:
        return f"{domain}__meds_profile"
    if "flight" in q or "boarding" in q or "gate" in q or "hotel" in q:
        return f"{domain}__travel_profile"
    if "allerg" in q:
        return f"{domain}__allergies"
    if "password" in q or "account" in q or "tax id" in q or "card number" in q:
        return f"{domain}__credentials_ids"
    if template_group in {"identify_or_name_X", "enumerate_objects"}:
        return f"{domain}__object_enumeration"
    if template_group in {"read_X", "read_or_interpret_text"}:
        return f"{domain}__text_reading"
    if template_group == "urgency_scene_query":
        return f"{domain}__urgent_scene"
    return f"{domain}__{template_group}"


def _soft_cap_for_bucket(final_bucket: str) -> int:
    if final_bucket == FINAL_BUCKET_MIXED_PARALLEL:
        return _PARALLEL_TEMPLATE_SOFT_CAP
    return _PE_TEMPLATE_SOFT_CAP


def diversify_select(
    items: Sequence[Mapping[str, Any]],
    n: int,
    *,
    final_bucket: str,
) -> list[dict[str, Any]]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n > len(items):
        raise ValueError(
            f"cannot select {n} from {len(items)} items for {final_bucket}"
        )
    remaining = sorted(items, key=lambda row: row["source_key"])
    selected: list[dict[str, Any]] = []
    template_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    soft_cap = _soft_cap_for_bucket(final_bucket)

    while len(selected) < n:

        def score(row: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
            tg = str(row["template_group"])
            sg = str(row["semantic_group"])
            over_cap = 1 if template_counts[tg] >= soft_cap else 0
            return (
                over_cap,
                template_counts[tg],
                semantic_counts[sg],
                sum(1 for s in selected if s["domain"] == row["domain"]),
                str(row["source_key"]),
            )

        best = min(remaining, key=score)
        remaining.remove(best)
        selected.append(dict(best))
        template_counts[str(best["template_group"])] += 1
        semantic_counts[str(best["semantic_group"])] += 1
    return selected


def _review_bucket_label(bucket: MixedReviewBucket | str) -> str:
    value = bucket.value if isinstance(bucket, MixedReviewBucket) else str(bucket)
    mapping = {
        MixedReviewBucket.MIXED_IMPLICIT.value: FINAL_BUCKET_MIXED_IMPLICIT,
        MixedReviewBucket.MIXED_PARALLEL.value: FINAL_BUCKET_MIXED_PARALLEL,
        MixedReviewBucket.MIXED_SEQUENTIAL.value: FINAL_BUCKET_MIXED_SEQUENTIAL,
        MixedReviewBucket.NOT_SUITABLE.value: "NOT_SUITABLE",
    }
    if value not in mapping:
        raise ValueError(f"unknown mixed review bucket: {value!r}")
    return mapping[value]


def _pool_rows_for_label(
    candidates: Sequence[StageACandidate],
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_kind = (
        SOURCE_KIND_PERSONAL_POOL
        if label == "Personal"
        else SOURCE_KIND_ENVIRONMENTAL_POOL
    )
    final_bucket = (
        FINAL_BUCKET_PERSONAL if label == "Personal" else FINAL_BUCKET_ENVIRONMENTAL
    )
    for item in candidates:
        if item.source_classification_label != label:
            continue
        domain = infer_domain(item.query)
        template_group = infer_pe_template_group(item.query)
        semantic_group = infer_pe_semantic_group(item.query, domain, template_group)
        rows.append(
            {
                "source_key": item.source_query_id,
                "source_id": item.source_query_id,
                "candidate_id": None,
                "query": item.query,
                "final_bucket": final_bucket,
                "source_kind": source_kind,
                "original_label": item.source_classification_label,
                "original_review_bucket": None,
                "domain": domain,
                "semantic_group": semantic_group,
                "template_group": template_group,
                "provenance": {
                    "kind": source_kind,
                    "path": str(DEFAULT_CANDIDATES_PATH).replace("\\", "/"),
                    "source_query_id": item.source_query_id,
                    "semantic_group_id": item.semantic_group_id,
                },
            }
        )
    rows.sort(key=lambda row: row["source_key"])
    return rows


def _mixed_row_from_review(
    item: MixedReviewRecord,
    *,
    final_bucket: str,
    reclassified: bool = False,
) -> dict[str, Any]:
    domain = infer_domain(item.query)
    template_group = infer_mixed_template_group(item.query)
    semantic_group = infer_mixed_semantic_group(item.query, domain)
    original_review_bucket = _review_bucket_label(item.planner_bucket)
    provenance: dict[str, Any] = {
        "kind": SOURCE_KIND_MIXED_REVIEW,
        "path": str(DEFAULT_MIXED_REVIEWS_PATH).replace("\\", "/"),
        "source_query_id": item.source_query_id,
        "original_review_bucket": original_review_bucket,
        "planner_bucket": item.planner_bucket.value,
        "review_status": item.review_status.value,
    }
    if reclassified:
        provenance["reclassification_reason"] = RECLASSIFY_REASON_FUSION_ONLY
        provenance["final_bucket"] = final_bucket
    return {
        "source_key": item.source_query_id,
        "source_id": item.source_query_id,
        "candidate_id": None,
        "query": item.query,
        "final_bucket": final_bucket,
        "source_kind": SOURCE_KIND_MIXED_REVIEW,
        "original_label": item.source_classification_label,
        "original_review_bucket": original_review_bucket,
        "domain": domain,
        "semantic_group": semantic_group,
        "template_group": template_group,
        "provenance": provenance,
        "reclassified_from_sequential": reclassified,
    }


def _mined_implicit_rows(path: Path) -> list[dict[str, Any]]:
    by_id = {item.source_id: item for item in load_implicit_candidates_jsonl(path)}
    rows: list[dict[str, Any]] = []
    for source_id in MINED_IMPLICIT_SELECTED_IDS:
        if source_id not in by_id:
            raise ValueError(f"missing mined implicit candidate: {source_id}")
        item = by_id[source_id]
        domain = infer_domain(item.query)
        template_group = infer_pe_template_group(item.query)
        semantic_group = infer_pe_semantic_group(item.query, domain, template_group)
        rows.append(
            {
                "source_key": source_id,
                "source_id": source_id,
                "candidate_id": None,
                "query": item.query,
                "final_bucket": FINAL_BUCKET_MIXED_IMPLICIT,
                "source_kind": SOURCE_KIND_MINED_IMPLICIT,
                "original_label": item.original_label,
                "original_review_bucket": None,
                "domain": domain,
                "semantic_group": f"{semantic_group}__mined_implicit",
                "template_group": template_group,
                "selection_reason": (
                    "mined_implicit_recovered; conservative Personal/Environmental mining"
                ),
                "provenance": {
                    "kind": SOURCE_KIND_MINED_IMPLICIT,
                    "path": str(path).replace("\\", "/"),
                    "source_id": source_id,
                    "score": item.score,
                    "mining_reasons": list(item.mining_reasons),
                },
            }
        )
    return rows


def _authored_implicit_rows(
    candidates_path: Path,
    reviews_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = {
        item.candidate_id: item for item in load_authored_candidates(candidates_path)
    }
    reviews = {
        item.candidate_id: item for item in load_authored_reviews(reviews_path)
    }
    selected: list[dict[str, Any]] = []
    spares: list[dict[str, Any]] = []
    for candidate_id in AUTHORED_IMPLICIT_SELECTED_IDS + AUTHORED_IMPLICIT_SPARE_IDS:
        item = candidates.get(candidate_id)
        review = reviews.get(candidate_id)
        if item is None:
            raise ValueError(f"missing authored implicit candidate: {candidate_id}")
        if review is None or review.review_status is not ImplicitReviewStatus.ACCEPT:
            raise ValueError(
                f"authored implicit {candidate_id} must be ACCEPT-reviewed"
            )
        domain = infer_domain(item.query)
        row = {
            "source_key": candidate_id,
            "source_id": None,
            "candidate_id": candidate_id,
            "query": item.query,
            "final_bucket": FINAL_BUCKET_MIXED_IMPLICIT,
            "source_kind": SOURCE_KIND_AUTHORED_IMPLICIT,
            "original_label": None,
            "original_review_bucket": None,
            "domain": domain,
            "semantic_group": f"{domain}__{item.template_group}",
            "template_group": item.template_group,
            "provenance": {
                "kind": SOURCE_KIND_AUTHORED_IMPLICIT,
                "path": str(candidates_path).replace("\\", "/"),
                "reviews_path": str(reviews_path).replace("\\", "/"),
                "candidate_id": candidate_id,
                "review_status": review.review_status.value,
                "authoring_reason": item.authoring_reason,
            },
        }
        if candidate_id in AUTHORED_IMPLICIT_SELECTED_IDS:
            row["selection_reason"] = (
                "authored_implicit_core; fixed Stage-A ACCEPT set "
                "(001-006,009-011)"
            )
            selected.append(row)
        else:
            row["selection_reason"] = (
                "authored_implicit_spare; ACCEPT retained as spare (007,008,012)"
            )
            spares.append(row)
    return selected, spares


def _authored_sequential_rows(
    candidates_path: Path,
    reviews_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = {
        item.candidate_id: item
        for item in load_authored_sequential_candidates(candidates_path)
    }
    reviews = {
        item.candidate_id: item
        for item in load_authored_sequential_reviews(reviews_path)
    }
    selected: list[dict[str, Any]] = []
    spares: list[dict[str, Any]] = []
    for candidate_id in (
        AUTHORED_SEQUENTIAL_SELECTED_IDS + AUTHORED_SEQUENTIAL_SPARE_IDS
    ):
        item = candidates.get(candidate_id)
        review = reviews.get(candidate_id)
        if item is None:
            raise ValueError(f"missing authored sequential candidate: {candidate_id}")
        if review is None or review.review_status is not SequentialReviewStatus.ACCEPT:
            raise ValueError(
                f"authored sequential {candidate_id} must be ACCEPT-reviewed"
            )
        row = {
            "source_key": candidate_id,
            "source_id": None,
            "candidate_id": candidate_id,
            "query": item.query,
            "final_bucket": FINAL_BUCKET_MIXED_SEQUENTIAL,
            "source_kind": SOURCE_KIND_AUTHORED_SEQUENTIAL,
            "original_label": None,
            "original_review_bucket": None,
            "domain": infer_domain(item.query),
            "semantic_group": item.semantic_group,
            "template_group": item.template_group,
            "dependency_family": item.dependency_family,
            "intended_operations": list(item.intended_operations),
            "intended_dependency_edges": list(item.intended_dependency_edges),
            "intended_typed_values": list(item.intended_typed_values),
            "provenance": {
                "kind": SOURCE_KIND_AUTHORED_SEQUENTIAL,
                "path": str(candidates_path).replace("\\", "/"),
                "reviews_path": str(reviews_path).replace("\\", "/"),
                "candidate_id": candidate_id,
                "review_status": review.review_status.value,
                "dependency_family": item.dependency_family,
                "intended_dependency_edges": list(item.intended_dependency_edges),
                "intended_typed_values": list(item.intended_typed_values),
                "personal_necessity_reason": item.personal_necessity_reason,
                "environmental_necessity_reason": item.environmental_necessity_reason,
            },
        }
        if candidate_id in AUTHORED_SEQUENTIAL_SELECTED_IDS:
            row["selection_reason"] = (
                "authored_sequential_core; fixed Stage-A ACCEPT set "
                "(001-014,017,019,020)"
            )
            selected.append(row)
        else:
            row["selection_reason"] = (
                "authored_sequential_spare; ACCEPT retained as spare (015,016,018)"
            )
            spares.append(row)
    return selected, spares


def _assign_stage_a_ids(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            BUCKET_ORDER.index(str(row["final_bucket"])),
            str(row["source_key"]),
        ),
    )
    out: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        record = dict(row)
        record["stage_a_id"] = f"sa_{index:04d}"
        record["selected"] = True
        out.append(record)
    return out


def _finalize_record(row: Mapping[str, Any], *, selected: bool) -> dict[str, Any]:
    record = {
        "stage_a_id": row.get("stage_a_id"),
        "source_id": row.get("source_id"),
        "candidate_id": row.get("candidate_id"),
        "query": row["query"],
        "final_bucket": row["final_bucket"],
        "source_kind": row["source_kind"],
        "original_label": row.get("original_label"),
        "original_review_bucket": row.get("original_review_bucket"),
        "semantic_group": row["semantic_group"],
        "template_group": row["template_group"],
        "selection_reason": row["selection_reason"],
        "provenance": row["provenance"],
        "selected": selected,
    }
    if row.get("reclassified_from_sequential"):
        record["reclassified_from_sequential"] = True
    if row.get("dependency_family"):
        record["dependency_family"] = row["dependency_family"]
    if row.get("intended_operations"):
        record["intended_operations"] = row["intended_operations"]
    if row.get("intended_dependency_edges"):
        record["intended_dependency_edges"] = row["intended_dependency_edges"]
    if row.get("intended_typed_values"):
        record["intended_typed_values"] = row["intended_typed_values"]
    return record


def build_stage_a_selection(
    *,
    candidates_path: str | Path = DEFAULT_CANDIDATES_PATH,
    reviews_path: str | Path = DEFAULT_MIXED_REVIEWS_PATH,
    mined_path: str | Path = DEFAULT_IMPLICIT_CANDIDATES_PATH,
    authored_implicit_candidates_path: str | Path = DEFAULT_AUTHORED_CANDIDATES_PATH,
    authored_implicit_reviews_path: str | Path = DEFAULT_AUTHORED_REVIEWS_PATH,
    authored_sequential_candidates_path: str
    | Path = DEFAULT_AUTHORED_SEQUENTIAL_CANDIDATES_PATH,
    authored_sequential_reviews_path: str
    | Path = DEFAULT_AUTHORED_SEQUENTIAL_REVIEWS_PATH,
    seed: int = SELECTION_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    del seed  # recorded for provenance compatibility; selection is non-random
    candidates = load_candidates_jsonl(candidates_path)
    reviews = load_mixed_reviews(reviews_path)
    reviews_by_id = {item.source_query_id: item for item in reviews}

    # ---- MIXED_IMPLICIT ----
    reviewed_implicit = [
        _mixed_row_from_review(item, final_bucket=FINAL_BUCKET_MIXED_IMPLICIT)
        for item in reviews
        if item.planner_bucket is MixedReviewBucket.MIXED_IMPLICIT
    ]
    if len(reviewed_implicit) != 12:
        raise ValueError(
            f"expected 12 reviewed MIXED_IMPLICIT, got {len(reviewed_implicit)}"
        )
    for row in reviewed_implicit:
        row["selection_reason"] = (
            "all_reviewed_mixed_implicit; human-reviewed MIXED_IMPLICIT retained"
        )
    mined_selected = _mined_implicit_rows(Path(mined_path))
    authored_imp_selected, authored_imp_spares = _authored_implicit_rows(
        Path(authored_implicit_candidates_path),
        Path(authored_implicit_reviews_path),
    )
    mixed_implicit_selected = (
        reviewed_implicit + mined_selected + authored_imp_selected
    )
    if len(mixed_implicit_selected) != PER_BUCKET:
        raise ValueError(
            "MIXED_IMPLICIT must be 12 reviewed + 3 mined + 9 authored "
            f"(got {len(mixed_implicit_selected)})"
        )

    reserved_query_keys = {
        normalize_query_key(row["query"]) for row in mixed_implicit_selected
    }

    # ---- Personal / Environmental ----
    personal_pool = _pool_rows_for_label(candidates, "Personal")
    environmental_pool = _pool_rows_for_label(candidates, "Environmental")
    if len(personal_pool) != 30 or len(environmental_pool) != 30:
        raise ValueError(
            "expected 30 Personal and 30 Environmental Stage-A candidates, "
            f"got {len(personal_pool)} / {len(environmental_pool)}"
        )
    personal_eligible = [
        row
        for row in personal_pool
        if normalize_query_key(row["query"]) not in reserved_query_keys
    ]
    environmental_eligible = [
        row
        for row in environmental_pool
        if normalize_query_key(row["query"]) not in reserved_query_keys
    ]
    personal_selected = diversify_select(
        personal_eligible, PER_BUCKET, final_bucket=FINAL_BUCKET_PERSONAL
    )
    for row in personal_selected:
        row["selection_reason"] = (
            "diversity_select_personal; greedy template/semantic coverage "
            f"(seed={SELECTION_SEED})"
        )
    environmental_selected = diversify_select(
        environmental_eligible,
        PER_BUCKET,
        final_bucket=FINAL_BUCKET_ENVIRONMENTAL,
    )
    for row in environmental_selected:
        row["selection_reason"] = (
            "diversity_select_environmental; greedy template/semantic coverage "
            f"(seed={SELECTION_SEED})"
        )

    # ---- MIXED_SEQUENTIAL ----
    natural_seq_selected: list[dict[str, Any]] = []
    for source_id in NATURAL_SEQUENTIAL_SELECTED_IDS:
        item = reviews_by_id.get(source_id)
        if item is None:
            raise ValueError(f"missing natural sequential review: {source_id}")
        if item.planner_bucket is not MixedReviewBucket.MIXED_SEQUENTIAL:
            raise ValueError(
                f"{source_id} expected original MIXED_SEQUENTIAL review"
            )
        row = _mixed_row_from_review(item, final_bucket=FINAL_BUCKET_MIXED_SEQUENTIAL)
        row["selection_reason"] = (
            "natural_true_sequential_core; execution-audit non-fusion dependency "
            "(diversity-deduped 7)"
        )
        natural_seq_selected.append(row)

    natural_seq_spares: list[dict[str, Any]] = []
    for source_id in NATURAL_SEQUENTIAL_SPARE_IDS:
        item = reviews_by_id[source_id]
        row = _mixed_row_from_review(item, final_bucket=FINAL_BUCKET_MIXED_SEQUENTIAL)
        row["selection_reason"] = (
            "natural_true_sequential_spare; near-paraphrase of selected core"
        )
        natural_seq_spares.append(row)

    authored_seq_selected, authored_seq_spares = _authored_sequential_rows(
        Path(authored_sequential_candidates_path),
        Path(authored_sequential_reviews_path),
    )
    mixed_sequential_selected = natural_seq_selected + authored_seq_selected
    if len(mixed_sequential_selected) != PER_BUCKET:
        raise ValueError(
            "MIXED_SEQUENTIAL must be 7 natural + 17 authored "
            f"(got {len(mixed_sequential_selected)})"
        )

    # ---- MIXED_PARALLEL pool ----
    original_parallel = [
        item
        for item in reviews
        if item.planner_bucket is MixedReviewBucket.MIXED_PARALLEL
    ]
    if len(original_parallel) != 31:
        raise ValueError(
            f"expected 31 reviewed MIXED_PARALLEL, got {len(original_parallel)}"
        )
    reclassified_parallel = [
        item
        for item in reviews
        if item.planner_bucket is MixedReviewBucket.MIXED_SEQUENTIAL
        and item.source_query_id not in TRUE_SEQUENTIAL_RAW_IDS
    ]
    if len(reclassified_parallel) != 70:
        raise ValueError(
            "expected 70 reclassified fusion-only old-sequential as Parallel, "
            f"got {len(reclassified_parallel)}"
        )

    parallel_pool: list[dict[str, Any]] = []
    for item in original_parallel:
        row = _mixed_row_from_review(
            item, final_bucket=FINAL_BUCKET_MIXED_PARALLEL, reclassified=False
        )
        parallel_pool.append(row)
    for item in reclassified_parallel:
        row = _mixed_row_from_review(
            item, final_bucket=FINAL_BUCKET_MIXED_PARALLEL, reclassified=True
        )
        parallel_pool.append(row)
    if len(parallel_pool) != 101:
        raise ValueError(f"expected 101 natural Parallel pool, got {len(parallel_pool)}")

    parallel_selected = diversify_select(
        parallel_pool, PER_BUCKET, final_bucket=FINAL_BUCKET_MIXED_PARALLEL
    )
    for row in parallel_selected:
        if row.get("reclassified_from_sequential"):
            row["selection_reason"] = (
                "diversity_select_mixed_parallel; reclassified from old "
                f"MIXED_SEQUENTIAL ({RECLASSIFY_REASON_FUSION_ONLY}; "
                f"seed={SELECTION_SEED})"
            )
        else:
            row["selection_reason"] = (
                "diversity_select_mixed_parallel; originally reviewed "
                f"MIXED_PARALLEL (seed={SELECTION_SEED})"
            )

    selected_raw = (
        personal_selected
        + environmental_selected
        + mixed_implicit_selected
        + parallel_selected
        + mixed_sequential_selected
    )
    selected = [
        _finalize_record(row, selected=True)
        for row in _assign_stage_a_ids(selected_raw)
    ]

    personal_selected_keys = {row["source_key"] for row in personal_selected}
    environmental_selected_keys = {
        row["source_key"] for row in environmental_selected
    }
    parallel_selected_keys = {row["source_key"] for row in parallel_selected}
    selected_query_keys = {normalize_query_key(row["query"]) for row in selected}

    def _spare_from(row: Mapping[str, Any], reason: str) -> dict[str, Any]:
        spare = dict(row)
        spare["selection_reason"] = reason
        spare["stage_a_id"] = None
        return _finalize_record(spare, selected=False)

    spares: list[dict[str, Any]] = []
    for row in personal_pool:
        if row["source_key"] in personal_selected_keys:
            continue
        if normalize_query_key(row["query"]) in selected_query_keys:
            continue
        spares.append(
            _spare_from(row, "personal_pool_spare; not selected under diversity budget")
        )
    for row in environmental_pool:
        if row["source_key"] in environmental_selected_keys:
            continue
        if normalize_query_key(row["query"]) in selected_query_keys:
            continue
        spares.append(
            _spare_from(
                row,
                "environmental_pool_spare; not selected under diversity budget",
            )
        )
    for row in parallel_pool:
        if row["source_key"] in parallel_selected_keys:
            continue
        reason = (
            "mixed_parallel_spare; reclassified fusion-only old-sequential unused"
            if row.get("reclassified_from_sequential")
            else "mixed_parallel_spare; originally reviewed Parallel unused"
        )
        spares.append(_spare_from(row, reason))
    for row in natural_seq_spares:
        spares.append(_finalize_record(row, selected=False))
    for row in authored_imp_spares:
        spares.append(_finalize_record(row, selected=False))
    for row in authored_seq_spares:
        spares.append(_finalize_record(row, selected=False))

    spares.sort(
        key=lambda row: (
            BUCKET_ORDER.index(str(row["final_bucket"]))
            if row["final_bucket"] in BUCKET_ORDER
            else 99,
            str(row.get("source_id") or row.get("candidate_id") or ""),
        )
    )

    summary = summarize_selection(selected, spares, not_suitable_count=5)
    summary["selection_seed"] = SELECTION_SEED
    summary["personal_pool_reassigned_to_mixed_implicit"] = sorted(
        str(row["source_id"])
        for row in personal_pool
        if row["source_key"] not in personal_selected_keys
        and normalize_query_key(row["query"]) in selected_query_keys
    )
    return selected, spares, summary


def summarize_selection(
    selected: Sequence[Mapping[str, Any]],
    spares: Sequence[Mapping[str, Any]],
    *,
    not_suitable_count: int = 0,
) -> dict[str, Any]:
    bucket_counts = Counter(row["final_bucket"] for row in selected)
    source_kind_counts = Counter(row["source_kind"] for row in selected)
    spare_bucket_counts = Counter(row["final_bucket"] for row in spares)
    unique_semantic: dict[str, int] = {}
    unique_template: dict[str, int] = {}
    top_templates: dict[str, list[tuple[str, int]]] = {}
    for bucket in BUCKET_ORDER:
        bucket_rows = [row for row in selected if row["final_bucket"] == bucket]
        unique_semantic[bucket] = len({row["semantic_group"] for row in bucket_rows})
        unique_template[bucket] = len({row["template_group"] for row in bucket_rows})
        top_templates[bucket] = Counter(
            row["template_group"] for row in bucket_rows
        ).most_common(8)

    parallel_rows = [
        row for row in selected if row["final_bucket"] == FINAL_BUCKET_MIXED_PARALLEL
    ]
    parallel_original = sum(
        1
        for row in parallel_rows
        if row.get("original_review_bucket") == FINAL_BUCKET_MIXED_PARALLEL
    )
    parallel_reclassified = sum(
        1
        for row in parallel_rows
        if row.get("reclassified_from_sequential")
        or row.get("original_review_bucket") == FINAL_BUCKET_MIXED_SEQUENTIAL
    )

    seq_rows = [
        row for row in selected if row["final_bucket"] == FINAL_BUCKET_MIXED_SEQUENTIAL
    ]
    natural_seq = [
        row for row in seq_rows if row["source_kind"] == SOURCE_KIND_MIXED_REVIEW
    ]
    authored_seq = [
        row
        for row in seq_rows
        if row["source_kind"] == SOURCE_KIND_AUTHORED_SEQUENTIAL
    ]
    dependency_family_counts = Counter(
        row.get("dependency_family") or row.get("provenance", {}).get("dependency_family")
        for row in authored_seq
    )
    typed_edge_coverage = sorted(
        {
            f"{edge} [{typed}]"
            for row in authored_seq
            for edge, typed in zip(
                row.get("intended_dependency_edges")
                or row.get("provenance", {}).get("intended_dependency_edges", []),
                row.get("intended_typed_values")
                or row.get("provenance", {}).get("intended_typed_values", []),
                strict=False,
            )
        }
    )

    return {
        "selected_total": len(selected),
        "spares_total": len(spares),
        "not_suitable_excluded": not_suitable_count,
        "counts_by_final_bucket": dict(bucket_counts),
        "counts_by_source_kind": dict(source_kind_counts),
        "spare_counts_by_final_bucket": dict(spare_bucket_counts),
        "unique_semantic_groups_per_bucket": unique_semantic,
        "unique_template_groups_per_bucket": unique_template,
        "top_template_families_per_bucket": top_templates,
        "mixed_parallel_from_original_review": parallel_original,
        "mixed_parallel_reclassified_from_sequential": parallel_reclassified,
        "mixed_sequential_natural": len(natural_seq),
        "mixed_sequential_authored": len(authored_seq),
        "authored_sequential_dependency_family_counts": dict(
            sorted(dependency_family_counts.items())
        ),
        "authored_sequential_typed_edge_coverage": typed_edge_coverage,
        "authored_implicit_selected_ids": [
            row["candidate_id"]
            for row in selected
            if row["source_kind"] == SOURCE_KIND_AUTHORED_IMPLICIT
        ],
        "authored_implicit_spare_ids": [
            row["candidate_id"]
            for row in spares
            if row["source_kind"] == SOURCE_KIND_AUTHORED_IMPLICIT
        ],
        "authored_sequential_selected_ids": [
            row["candidate_id"] for row in authored_seq
        ],
        "authored_sequential_spare_ids": [
            row["candidate_id"]
            for row in spares
            if row["source_kind"] == SOURCE_KIND_AUTHORED_SEQUENTIAL
        ],
        "natural_sequential_selected_ids": [
            row["source_id"] for row in natural_seq
        ],
        "selected_vs_spare": {
            "selected": len(selected),
            "spares": len(spares),
        },
    }


def format_selection_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "Stage-A final selection summary",
        f"selected_total: {summary['selected_total']}",
        f"spares_total: {summary['spares_total']}",
        f"not_suitable_excluded: {summary['not_suitable_excluded']}",
        "counts_by_final_bucket:",
    ]
    for bucket in BUCKET_ORDER:
        lines.append(
            f"  {bucket}: {summary['counts_by_final_bucket'].get(bucket, 0)}"
        )
    lines.append("counts_by_source_kind:")
    for kind, count in sorted(summary["counts_by_source_kind"].items()):
        lines.append(f"  {kind}: {count}")
    lines.append("unique_semantic_groups_per_bucket:")
    for bucket in BUCKET_ORDER:
        lines.append(
            f"  {bucket}: {summary['unique_semantic_groups_per_bucket'][bucket]}"
        )
    lines.append("unique_template_groups_per_bucket:")
    for bucket in BUCKET_ORDER:
        lines.append(
            f"  {bucket}: {summary['unique_template_groups_per_bucket'][bucket]}"
        )
    lines.append(
        "mixed_parallel_from_original_review: "
        f"{summary['mixed_parallel_from_original_review']}"
    )
    lines.append(
        "mixed_parallel_reclassified_from_sequential: "
        f"{summary['mixed_parallel_reclassified_from_sequential']}"
    )
    lines.append(
        "mixed_sequential_natural/authored: "
        f"{summary['mixed_sequential_natural']}/"
        f"{summary['mixed_sequential_authored']}"
    )
    lines.append("authored_sequential_dependency_family_counts:")
    for family, count in summary[
        "authored_sequential_dependency_family_counts"
    ].items():
        lines.append(f"  {family}: {count}")
    lines.append("authored_sequential_typed_edge_coverage:")
    for edge in summary["authored_sequential_typed_edge_coverage"]:
        lines.append(f"  - {edge}")
    lines.append(
        "natural_sequential_selected_ids: "
        + ", ".join(summary["natural_sequential_selected_ids"])
    )
    lines.append(
        "authored_sequential_selected_ids: "
        + ", ".join(summary["authored_sequential_selected_ids"])
    )
    lines.append(
        "authored_sequential_spare_ids: "
        + ", ".join(summary["authored_sequential_spare_ids"])
    )
    lines.append(
        "authored_implicit_selected_ids: "
        + ", ".join(summary["authored_implicit_selected_ids"])
    )
    lines.append(
        "authored_implicit_spare_ids: "
        + ", ".join(summary["authored_implicit_spare_ids"])
    )
    lines.append("top_template_families_per_bucket:")
    for bucket in BUCKET_ORDER:
        families = summary["top_template_families_per_bucket"][bucket]
        rendered = ", ".join(f"{name}={count}" for name, count in families) or "(none)"
        lines.append(f"  {bucket}: {rendered}")
    return "\n".join(lines)


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(payload)
    return rows


def validate_stage_a_selection(
    selected: Sequence[Mapping[str, Any]],
    spares: Sequence[Mapping[str, Any]],
    *,
    reviews_path: str | Path = DEFAULT_MIXED_REVIEWS_PATH,
) -> list[str]:
    errors: list[str] = []
    if len(selected) != STAGE_A_TOTAL:
        errors.append(f"selected count {len(selected)} != {STAGE_A_TOTAL}")

    bucket_counts = Counter(row.get("final_bucket") for row in selected)
    for bucket in BUCKET_ORDER:
        if bucket_counts.get(bucket) != PER_BUCKET:
            errors.append(
                f"{bucket} count {bucket_counts.get(bucket, 0)} != {PER_BUCKET}"
            )

    query_keys = [normalize_query_key(str(row.get("query", ""))) for row in selected]
    if len(query_keys) != len(set(query_keys)):
        errors.append("duplicate normalized queries in selected set")
    if any(not key for key in query_keys):
        errors.append("blank normalized query in selected set")

    source_ids = [row.get("source_id") or row.get("candidate_id") for row in selected]
    if any(item is None for item in source_ids):
        errors.append("selected row missing source_id/candidate_id")
    if len(source_ids) != len(set(source_ids)):
        errors.append("duplicate selected source/candidate IDs")

    for row in selected:
        if row.get("selected") is not True:
            errors.append(f"{row.get('stage_a_id')}: selected flag is not true")
        if not row.get("provenance"):
            errors.append(f"{row.get('stage_a_id')}: missing provenance")
        if not row.get("semantic_group"):
            errors.append(f"{row.get('stage_a_id')}: missing semantic_group")
        if not row.get("template_group"):
            errors.append(f"{row.get('stage_a_id')}: missing template_group")
        if row.get("final_bucket") == "NOT_SUITABLE":
            errors.append("NOT_SUITABLE example selected")
        if "split" in row:
            errors.append(f"{row.get('stage_a_id')}: unexpected split field")

    # Implicit composition.
    reviewed_imp = [
        row
        for row in selected
        if row.get("final_bucket") == FINAL_BUCKET_MIXED_IMPLICIT
        and row.get("source_kind") == SOURCE_KIND_MIXED_REVIEW
    ]
    mined_imp = [
        row
        for row in selected
        if row.get("source_kind") == SOURCE_KIND_MINED_IMPLICIT
    ]
    authored_imp = [
        row
        for row in selected
        if row.get("source_kind") == SOURCE_KIND_AUTHORED_IMPLICIT
    ]
    authored_imp_spares = [
        row
        for row in spares
        if row.get("source_kind") == SOURCE_KIND_AUTHORED_IMPLICIT
    ]
    if len(reviewed_imp) != 12:
        errors.append(f"reviewed mixed implicit selected {len(reviewed_imp)} != 12")
    mined_ids = tuple(sorted(str(row.get("source_id")) for row in mined_imp))
    if mined_ids != tuple(sorted(MINED_IMPLICIT_SELECTED_IDS)):
        errors.append(
            f"mined implicit IDs {mined_ids} != {tuple(sorted(MINED_IMPLICIT_SELECTED_IDS))}"
        )
    authored_imp_ids = tuple(row.get("candidate_id") for row in authored_imp)
    if authored_imp_ids != AUTHORED_IMPLICIT_SELECTED_IDS:
        # tolerate sort differences by comparing sets + count
        if sorted(authored_imp_ids) != sorted(AUTHORED_IMPLICIT_SELECTED_IDS) or len(
            authored_imp_ids
        ) != 9:
            errors.append(
                f"authored implicit selected IDs {authored_imp_ids} "
                f"!= {AUTHORED_IMPLICIT_SELECTED_IDS}"
            )
    spare_imp_ids = tuple(sorted(str(row.get("candidate_id")) for row in authored_imp_spares))
    if spare_imp_ids != tuple(sorted(AUTHORED_IMPLICIT_SPARE_IDS)):
        errors.append(
            f"authored implicit spare IDs {spare_imp_ids} "
            f"!= {tuple(sorted(AUTHORED_IMPLICIT_SPARE_IDS))}"
        )

    # Sequential composition.
    natural_seq = [
        row
        for row in selected
        if row.get("final_bucket") == FINAL_BUCKET_MIXED_SEQUENTIAL
        and row.get("source_kind") == SOURCE_KIND_MIXED_REVIEW
    ]
    authored_seq = [
        row
        for row in selected
        if row.get("source_kind") == SOURCE_KIND_AUTHORED_SEQUENTIAL
    ]
    authored_seq_spares = [
        row
        for row in spares
        if row.get("source_kind") == SOURCE_KIND_AUTHORED_SEQUENTIAL
    ]
    natural_seq_ids = tuple(sorted(str(row.get("source_id")) for row in natural_seq))
    if natural_seq_ids != tuple(sorted(NATURAL_SEQUENTIAL_SELECTED_IDS)):
        errors.append(
            f"natural sequential IDs {natural_seq_ids} "
            f"!= {tuple(sorted(NATURAL_SEQUENTIAL_SELECTED_IDS))}"
        )
    if len(authored_seq) != 17:
        errors.append(f"authored sequential selected {len(authored_seq)} != 17")
    authored_seq_ids = tuple(sorted(str(row.get("candidate_id")) for row in authored_seq))
    if authored_seq_ids != tuple(sorted(AUTHORED_SEQUENTIAL_SELECTED_IDS)):
        errors.append(
            f"authored sequential selected IDs {authored_seq_ids} "
            f"!= {tuple(sorted(AUTHORED_SEQUENTIAL_SELECTED_IDS))}"
        )
    spare_seq_ids = tuple(
        sorted(str(row.get("candidate_id")) for row in authored_seq_spares)
    )
    if spare_seq_ids != tuple(sorted(AUTHORED_SEQUENTIAL_SPARE_IDS)):
        errors.append(
            f"authored sequential spare IDs {spare_seq_ids} "
            f"!= {tuple(sorted(AUTHORED_SEQUENTIAL_SPARE_IDS))}"
        )
    selected_ids = {row.get("source_id") for row in selected}
    for banned in NATURAL_SEQUENTIAL_SPARE_IDS:
        if banned in selected_ids:
            errors.append(f"{banned} must not be selected as core sequential")

    # Parallel constraints.
    parallel_rows = [
        row for row in selected if row.get("final_bucket") == FINAL_BUCKET_MIXED_PARALLEL
    ]
    if any(
        row.get("source_kind")
        in {SOURCE_KIND_AUTHORED_IMPLICIT, SOURCE_KIND_AUTHORED_SEQUENTIAL}
        for row in parallel_rows
    ):
        errors.append("MIXED_PARALLEL must be natural/source examples only")
    for row in parallel_rows:
        sid = row.get("source_id")
        if sid in TRUE_SEQUENTIAL_RAW_IDS:
            errors.append(f"true sequential {sid} classified as Parallel")

    # src_0264 must not also appear as Personal.
    personal_ids = {
        row.get("source_id")
        for row in selected
        if row.get("final_bucket") == FINAL_BUCKET_PERSONAL
    }
    if "src_0264" in personal_ids:
        errors.append("src_0264 must not be selected as Personal")

    # Mixed review consistency for non-reclassified rows.
    reviews = {
        item.source_query_id: item for item in load_mixed_reviews(reviews_path)
    }
    for row in selected:
        if row.get("source_kind") != SOURCE_KIND_MIXED_REVIEW:
            continue
        source_id = str(row.get("source_id"))
        review = reviews.get(source_id)
        if review is None:
            errors.append(f"selected mixed {source_id} missing from reviews")
            continue
        if review.planner_bucket is MixedReviewBucket.NOT_SUITABLE:
            errors.append(f"NOT_SUITABLE review selected: {source_id}")
            continue
        original = _review_bucket_label(review.planner_bucket)
        if row.get("original_review_bucket") != original:
            errors.append(
                f"{source_id} original_review_bucket mismatch "
                f"{row.get('original_review_bucket')!r} vs {original!r}"
            )
        if row.get("final_bucket") == FINAL_BUCKET_MIXED_PARALLEL:
            if original == FINAL_BUCKET_MIXED_PARALLEL:
                continue
            if (
                original == FINAL_BUCKET_MIXED_SEQUENTIAL
                and source_id not in TRUE_SEQUENTIAL_RAW_IDS
            ):
                continue
            errors.append(
                f"parallel {source_id} has incompatible original review {original}"
            )
        elif row.get("final_bucket") == FINAL_BUCKET_MIXED_SEQUENTIAL:
            if source_id not in NATURAL_SEQUENTIAL_SELECTED_IDS:
                errors.append(
                    f"natural sequential selected unexpected id {source_id}"
                )
        elif row.get("final_bucket") == FINAL_BUCKET_MIXED_IMPLICIT:
            if original != FINAL_BUCKET_MIXED_IMPLICIT:
                errors.append(
                    f"implicit {source_id} original review is {original}"
                )

    for row in spares:
        if row.get("final_bucket") == "NOT_SUITABLE":
            errors.append("NOT_SUITABLE present in spares")
        if not row.get("provenance"):
            errors.append(
                f"spare {row.get('source_id') or row.get('candidate_id')}: "
                "missing provenance"
            )
        if not row.get("semantic_group") or not row.get("template_group"):
            errors.append(
                f"spare {row.get('source_id') or row.get('candidate_id')}: "
                "missing group metadata"
            )

    return errors


def build_and_write_stage_a_selection(
    *,
    candidates_path: str | Path = DEFAULT_CANDIDATES_PATH,
    reviews_path: str | Path = DEFAULT_MIXED_REVIEWS_PATH,
    mined_path: str | Path = DEFAULT_IMPLICIT_CANDIDATES_PATH,
    authored_implicit_candidates_path: str | Path = DEFAULT_AUTHORED_CANDIDATES_PATH,
    authored_implicit_reviews_path: str | Path = DEFAULT_AUTHORED_REVIEWS_PATH,
    authored_sequential_candidates_path: str
    | Path = DEFAULT_AUTHORED_SEQUENTIAL_CANDIDATES_PATH,
    authored_sequential_reviews_path: str
    | Path = DEFAULT_AUTHORED_SEQUENTIAL_REVIEWS_PATH,
    selection_path: str | Path = DEFAULT_SELECTION_PATH,
    spares_path: str | Path = DEFAULT_SPARES_PATH,
) -> dict[str, Any]:
    selected, spares, summary = build_stage_a_selection(
        candidates_path=candidates_path,
        reviews_path=reviews_path,
        mined_path=mined_path,
        authored_implicit_candidates_path=authored_implicit_candidates_path,
        authored_implicit_reviews_path=authored_implicit_reviews_path,
        authored_sequential_candidates_path=authored_sequential_candidates_path,
        authored_sequential_reviews_path=authored_sequential_reviews_path,
    )
    errors = validate_stage_a_selection(
        selected, spares, reviews_path=reviews_path
    )
    if errors:
        raise ValueError("Stage-A selection invalid:\n- " + "\n- ".join(errors))
    write_jsonl(selection_path, selected)
    write_jsonl(spares_path, spares)
    summary["selection_path"] = str(selection_path).replace("\\", "/")
    summary["spares_path"] = str(spares_path).replace("\\", "/")
    return summary


__all__ = [
    "AUTHORED_IMPLICIT_SELECTED_IDS",
    "AUTHORED_IMPLICIT_SPARE_IDS",
    "AUTHORED_SEQUENTIAL_SELECTED_IDS",
    "AUTHORED_SEQUENTIAL_SPARE_IDS",
    "BUCKET_ORDER",
    "DEFAULT_CANDIDATES_PATH",
    "DEFAULT_MIXED_REVIEWS_PATH",
    "DEFAULT_SELECTION_PATH",
    "DEFAULT_SPARES_PATH",
    "FINAL_BUCKET_ENVIRONMENTAL",
    "FINAL_BUCKET_MIXED_IMPLICIT",
    "FINAL_BUCKET_MIXED_PARALLEL",
    "FINAL_BUCKET_MIXED_SEQUENTIAL",
    "FINAL_BUCKET_PERSONAL",
    "MINED_IMPLICIT_SELECTED_IDS",
    "NATURAL_SEQUENTIAL_SELECTED_IDS",
    "NATURAL_SEQUENTIAL_SPARE_IDS",
    "PER_BUCKET",
    "RECLASSIFY_REASON_FUSION_ONLY",
    "SELECTION_SEED",
    "STAGE_A_TOTAL",
    "TRUE_SEQUENTIAL_RAW_IDS",
    "build_and_write_stage_a_selection",
    "build_stage_a_selection",
    "diversify_select",
    "format_selection_summary",
    "infer_domain",
    "infer_mixed_semantic_group",
    "infer_mixed_template_group",
    "infer_pe_semantic_group",
    "infer_pe_template_group",
    "load_jsonl",
    "summarize_selection",
    "validate_stage_a_selection",
    "write_jsonl",
]
