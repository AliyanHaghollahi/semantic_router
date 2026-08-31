"""Stage-A v2 candidate inventory assembly (Todo 2).

Assembles reviewable candidate pools and authored-family *specifications*
that will later feed the 480-example selection. Does **not** write the final
selection, annotations, or split.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tiergraph.planner.authored_implicit import (
    DEFAULT_AUTHORED_CANDIDATES_PATH,
    load_authored_candidates,
)
from tiergraph.planner.authored_sequential import (
    DEFAULT_AUTHORED_SEQUENTIAL_CANDIDATES_PATH,
    load_authored_sequential_candidates,
)
from tiergraph.planner.corpus import (
    DEFAULT_CANDIDATE_SEED,
    build_unique_query_pool,
    load_classification_rows,
    normalize_query_key,
)
from tiergraph.planner.implicit_mining import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V1_IMPLICIT_MINE_PATH,
    DEFAULT_REVIEWS_PATH,
    DEFAULT_TRAIN_PATH,
    excluded_normalized_queries,
    load_implicit_candidates_jsonl,
    mine_implicit_candidates,
)
from tiergraph.planner.mixed_review import MixedReviewBucket, load_mixed_reviews
from tiergraph.planner.stage_a_selection import (
    AUTHORED_IMPLICIT_SELECTED_IDS,
    AUTHORED_IMPLICIT_SPARE_IDS,
    AUTHORED_SEQUENTIAL_SELECTED_IDS,
    AUTHORED_SEQUENTIAL_SPARE_IDS,
    DEFAULT_MIXED_REVIEWS_PATH,
    DEFAULT_SPARES_PATH,
    MIXED_PARALLEL_REQUIRED_IDS,
    NATURAL_SEQUENTIAL_SELECTED_IDS,
    TRUE_SEQUENTIAL_RAW_IDS,
    infer_domain,
    infer_mixed_semantic_group,
    infer_mixed_template_group,
    infer_pe_semantic_group,
    infer_pe_template_group,
    load_jsonl,
)
from tiergraph.planner.stage_a_v2_spec import (
    AUTHORED_HOLDOUT_FAMILY_LINKS,
    H5_NEGATIVE_RETRIEVE_POSSESSIVE_MIN,
    H7_FAMILY_MINIMUMS,
    H7_MULTI_HOP_MINIMUM,
    H7_POSITIVE_EXAMPLE_TARGET_RANGE,
    LEGAL_H7_FAMILY_LABELS,
    STAGE_A_V1_SELECTION_PATH,
    STAGE_A_V2_PER_BUCKET,
    STAGE_A_V2_BUCKETS,
    h7_family_label,
    is_legal_h7_pair,
    parse_h7_family_label,
    validate_authored_family_on_row,
    validate_provenance_metadata,
)


STAGE_A_V2_CANDIDATES_PATH = Path("dataset/planner/stage_a_v2_candidate_inventory.jsonl")
STAGE_A_V2_CANDIDATE_REPORT_PATH = Path(
    "dataset/planner/stage_a_v2_candidate_report.json"
)
STAGE_A_V2_AUTHORED_SPECS_PATH = Path(
    "dataset/planner/stage_a_v2_authored_family_specs.jsonl"
)
STAGE_A_V2_IMPLICIT_MINE_PATH = Path(
    "dataset/planner/stage_a_v2_implicit_mine_review.jsonl"
)

V2_CANDIDATE_ASSEMBLY_SEED = DEFAULT_CANDIDATE_SEED
V2_IMPLICIT_MINE_LIMIT = 40

CANDIDATE_PROVENANCE_FIELDS: tuple[str, ...] = (
    "source_kind",
    "source_id",
    "semantic_group",
    "template_group",
    "authored_template_family",
    "proposed_final_bucket",
    "review_status",
    "operator_family",
    "h5_positive",
    "h7_positive",
    "h7_families",
)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _candidate_uid(*, source_kind: str, source_key: str, bucket: str) -> str:
    digest = hashlib.sha1(
        f"{source_kind}|{source_key}|{bucket}".encode("utf-8")
    ).hexdigest()[:12]
    return f"v2c_{digest}"


def load_v1_frozen_index(
    selection_path: str | Path = STAGE_A_V1_SELECTION_PATH,
) -> dict[str, Any]:
    """Read-only index of frozen v1 selection (queries, source IDs, candidate IDs)."""
    rows = load_jsonl(selection_path)
    query_keys = {normalize_query_key(row["query"]) for row in rows}
    source_ids = {
        str(row["source_id"]) for row in rows if row.get("source_id") is not None
    }
    candidate_ids = {
        str(row["candidate_id"])
        for row in rows
        if row.get("candidate_id") is not None
    }
    return {
        "rows": rows,
        "query_keys": query_keys,
        "source_ids": source_ids,
        "candidate_ids": candidate_ids,
        "by_bucket": Counter(str(row["final_bucket"]) for row in rows),
    }


def _base_candidate(
    *,
    source_kind: str,
    source_id: str | None,
    candidate_id: str | None,
    query: str,
    proposed_final_bucket: str,
    semantic_group: str,
    template_group: str,
    authored_template_family: str | None,
    original_label: str | None,
    review_status: str,
    acceptance_reason: str | None,
    rejection_reason: str | None,
    operator_family: list[str] | None,
    h5_positive: bool | None,
    h7_positive: bool | None,
    h7_families: list[str] | None,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    source_key = candidate_id or source_id or normalize_query_key(query)
    row = {
        "candidate_uid": _candidate_uid(
            source_kind=source_kind,
            source_key=str(source_key),
            bucket=proposed_final_bucket,
        ),
        "source_kind": source_kind,
        "source_id": source_id,
        "candidate_id": candidate_id,
        "query": query,
        "normalized_query": normalize_query_key(query),
        "semantic_group": semantic_group,
        "template_group": template_group,
        "authored_template_family": authored_template_family,
        "final_bucket": proposed_final_bucket,  # provenance alias
        "proposed_final_bucket": proposed_final_bucket,
        "original_label": original_label,
        "review_status": review_status,
        "acceptance_reason": acceptance_reason,
        "rejection_reason": rejection_reason,
        "operator_family": operator_family,
        "h5_positive": h5_positive,
        "h7_positive": h7_positive,
        "h7_families": h7_families if h7_families is not None else [],
        "provenance": dict(provenance),
    }
    return row


def assemble_personal_environmental_candidates(
    *,
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    v1_index: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Unused Personal/Environmental uniques from training_data.json."""
    pool = build_unique_query_pool(load_classification_rows(train_path))
    blocked_keys: set[str] = set(v1_index["query_keys"])
    blocked_ids: set[str] = set(v1_index["source_ids"])
    conflicts: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []

    for item in sorted(pool, key=lambda c: c.source_query_id):
        if item.source_classification_label not in {"Personal", "Environmental"}:
            continue
        key = normalize_query_key(item.query)
        bucket = item.source_classification_label
        if key in blocked_keys or item.source_query_id in blocked_ids:
            conflicts.append(
                {
                    "source_id": item.source_query_id,
                    "normalized_query": key,
                    "reason": "duplicate_against_frozen_v1",
                    "proposed_final_bucket": bucket,
                }
            )
            continue
        domain = infer_domain(item.query)
        template_group = infer_pe_template_group(item.query)
        semantic_group = infer_pe_semantic_group(item.query, domain, template_group)
        accepted.append(
            _base_candidate(
                source_kind="natural",
                source_id=item.source_query_id,
                candidate_id=None,
                query=item.query,
                proposed_final_bucket=bucket,
                semantic_group=semantic_group,
                template_group=template_group,
                authored_template_family=None,
                original_label=item.source_classification_label,
                review_status="available",
                acceptance_reason=(
                    "unused_natural_pe_unique; excluded frozen v1 query/source"
                ),
                rejection_reason=None,
                operator_family=None,
                h5_positive=False,
                h7_positive=False,
                h7_families=[],
                provenance={
                    "origin": "training_data_unique_pool",
                    "path": str(Path(train_path)).replace("\\", "/"),
                    "domain": domain,
                },
            )
        )
    return accepted, conflicts


def assemble_parallel_candidates(
    *,
    spares_path: str | Path = DEFAULT_SPARES_PATH,
    reviews_path: str | Path = DEFAULT_MIXED_REVIEWS_PATH,
    v1_index: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """TRUE execution-independent Parallel inventory (no hidden sequential)."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    def _reject(row: Mapping[str, Any], reason: str) -> None:
        rejected.append(
            {
                "source_id": row.get("source_id") or row.get("source_query_id"),
                "query": row.get("query"),
                "normalized_query": normalize_query_key(str(row.get("query") or "")),
                "reason": reason,
                "proposed_final_bucket": "MIXED_PARALLEL",
            }
        )

    spares = load_jsonl(spares_path)
    for row in sorted(spares, key=lambda r: str(r.get("source_id") or r.get("query"))):
        if row.get("final_bucket") != "MIXED_PARALLEL":
            continue
        source_id = str(row.get("source_id") or "")
        key = normalize_query_key(str(row["query"]))
        if source_id in TRUE_SEQUENTIAL_RAW_IDS:
            _reject(row, "rejected_true_sequential_not_parallel")
            continue
        if key in v1_index["query_keys"] or source_id in v1_index["source_ids"]:
            _reject(row, "duplicate_against_frozen_v1")
            continue
        if source_id in MIXED_PARALLEL_REQUIRED_IDS:
            _reject(row, "already_required_in_v1_parallel_core")
            continue
        if key in seen_keys:
            _reject(row, "duplicate_within_v2_inventory")
            continue
        seen_keys.add(key)
        reclassified = bool(row.get("reclassified_from_sequential"))
        accepted.append(
            _base_candidate(
                source_kind="natural",
                source_id=source_id or None,
                candidate_id=None,
                query=str(row["query"]),
                proposed_final_bucket="MIXED_PARALLEL",
                semantic_group=str(row["semantic_group"]),
                template_group=str(row["template_group"]),
                authored_template_family=None,
                original_label=row.get("original_label"),
                review_status="available",
                acceptance_reason=(
                    "parallel_spare_execution_independent"
                    + ("; fusion_only_reclassified_audit" if reclassified else "")
                ),
                rejection_reason=None,
                operator_family=None,
                h5_positive=False,
                h7_positive=False,
                h7_families=[],
                provenance={
                    "origin": "stage_a_spares",
                    "path": str(Path(spares_path)).replace("\\", "/"),
                    "reclassified_from_sequential": reclassified,
                    "original_review_bucket": row.get("original_review_bucket"),
                    "explicit_h7": False,
                },
            )
        )

    # Unused human Parallel reviews not already in spares/selection
    reviews = load_mixed_reviews(reviews_path)
    spare_ids = {
        str(r.get("source_id"))
        for r in spares
        if r.get("final_bucket") == "MIXED_PARALLEL" and r.get("source_id")
    }
    for review in sorted(reviews, key=lambda r: r.source_query_id):
        if review.planner_bucket is not MixedReviewBucket.MIXED_PARALLEL:
            continue
        key = normalize_query_key(review.query)
        sid = review.source_query_id
        if sid in TRUE_SEQUENTIAL_RAW_IDS:
            _reject(
                {"source_id": sid, "query": review.query},
                "rejected_true_sequential_not_parallel",
            )
            continue
        if (
            key in v1_index["query_keys"]
            or sid in v1_index["source_ids"]
            or sid in spare_ids
            or key in seen_keys
        ):
            continue
        seen_keys.add(key)
        domain = infer_domain(review.query)
        template_group = infer_mixed_template_group(review.query)
        semantic_group = infer_mixed_semantic_group(review.query, domain)
        accepted.append(
            _base_candidate(
                source_kind="natural",
                source_id=sid,
                candidate_id=None,
                query=review.query,
                proposed_final_bucket="MIXED_PARALLEL",
                semantic_group=semantic_group,
                template_group=template_group,
                authored_template_family=None,
                original_label=review.source_classification_label,
                review_status="available",
                acceptance_reason="unused_mixed_review_parallel",
                rejection_reason=None,
                operator_family=None,
                h5_positive=False,
                h7_positive=False,
                h7_families=[],
                provenance={
                    "origin": "mixed_review",
                    "path": str(Path(reviews_path)).replace("\\", "/"),
                    "review_bucket": MixedReviewBucket.MIXED_PARALLEL.value,
                    "explicit_h7": False,
                },
            )
        )

    return accepted, rejected


def assemble_natural_sequential_candidates(
    *,
    reviews_path: str | Path = DEFAULT_MIXED_REVIEWS_PATH,
    spares_path: str | Path = DEFAULT_SPARES_PATH,
    v1_index: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Audited natural TRUE_SEQUENTIAL still available (no mass authoring)."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reviews_by_id = {r.source_query_id: r for r in load_mixed_reviews(reviews_path)}
    spares_by_id = {
        str(r["source_id"]): r
        for r in load_jsonl(spares_path)
        if r.get("source_id")
    }

    for source_id in sorted(TRUE_SEQUENTIAL_RAW_IDS):
        if (
            source_id in v1_index["source_ids"]
            or source_id in NATURAL_SEQUENTIAL_SELECTED_IDS
        ):
            rejected.append(
                {
                    "source_id": source_id,
                    "reason": "already_in_frozen_v1_or_core_natural_sequential",
                    "proposed_final_bucket": "MIXED_SEQUENTIAL",
                }
            )
            continue
        review = reviews_by_id.get(source_id)
        spare = spares_by_id.get(source_id)
        if review is None and spare is None:
            rejected.append(
                {
                    "source_id": source_id,
                    "reason": "true_sequential_id_missing_from_reviews_and_spares",
                    "proposed_final_bucket": "MIXED_SEQUENTIAL",
                }
            )
            continue
        query = review.query if review is not None else str(spare["query"])
        key = normalize_query_key(query)
        if key in v1_index["query_keys"]:
            rejected.append(
                {
                    "source_id": source_id,
                    "normalized_query": key,
                    "reason": "duplicate_against_frozen_v1",
                    "proposed_final_bucket": "MIXED_SEQUENTIAL",
                }
            )
            continue
        domain = infer_domain(query)
        if spare is not None:
            semantic_group = str(spare["semantic_group"])
            template_group = str(spare["template_group"])
        else:
            template_group = infer_mixed_template_group(query)
            semantic_group = infer_mixed_semantic_group(query, domain)
        accepted.append(
            _base_candidate(
                source_kind="natural",
                source_id=source_id,
                candidate_id=None,
                query=query,
                proposed_final_bucket="MIXED_SEQUENTIAL",
                semantic_group=semantic_group,
                template_group=template_group,
                authored_template_family=None,
                original_label=(
                    review.source_classification_label if review else "Mixed"
                ),
                review_status="available",
                acceptance_reason=(
                    "audited_true_sequential_unused; eligible for v2 sequential"
                ),
                rejection_reason=None,
                operator_family=None,
                h5_positive=None,
                h7_positive=True,
                h7_families=[],  # concrete pair labels require Step-B annotation
                provenance={
                    "origin": "true_sequential_audit",
                    "true_sequential_raw": True,
                    "path": str(Path(reviews_path)).replace("\\", "/"),
                },
            )
        )
    return accepted, rejected


def _explicit_h7_families_from_edges(edges: Sequence[str]) -> list[str]:
    families: list[str] = []
    for edge in edges:
        if "->" not in edge:
            continue
        src, tgt = (part.strip() for part in edge.split("->", 1))
        if src == "RESOLVE_PERSONAL":
            continue  # H5 structure, not learned H7
        if is_legal_h7_pair(src, tgt):
            label = f"{src}->{tgt}"
            if label not in families:
                families.append(label)
        else:
            raise ValueError(f"illegal explicit H7 edge in authored candidate: {edge}")
    return families


def assemble_authored_spare_candidates(
    *,
    v1_index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Small leftover authored spares only (no mass generation)."""
    out: list[dict[str, Any]] = []

    seq = {
        c.candidate_id: c
        for c in load_authored_sequential_candidates(
            DEFAULT_AUTHORED_SEQUENTIAL_CANDIDATES_PATH
        )
    }
    for cid in AUTHORED_SEQUENTIAL_SPARE_IDS:
        if cid in v1_index["candidate_ids"] or cid in AUTHORED_SEQUENTIAL_SELECTED_IDS:
            continue
        item = seq[cid]
        family = item.template_group  # stable scenario family ID
        h7_families = _explicit_h7_families_from_edges(item.intended_dependency_edges)
        out.append(
            _base_candidate(
                source_kind="authored",
                source_id=None,
                candidate_id=cid,
                query=item.query,
                proposed_final_bucket="MIXED_SEQUENTIAL",
                semantic_group=item.semantic_group,
                template_group=item.template_group,
                authored_template_family=family,
                original_label="Mixed",
                review_status="available",
                acceptance_reason="authored_sequential_spare_unused",
                rejection_reason=None,
                operator_family=[
                    op
                    for op in item.intended_operations
                    if op != "RESOLVE_PERSONAL"
                ],
                h5_positive=any(
                    "RESOLVE_PERSONAL" in edge for edge in item.intended_dependency_edges
                ),
                h7_positive=bool(h7_families),
                h7_families=h7_families,
                provenance={
                    "origin": "authored_sequential_spare",
                    "dependency_family": item.dependency_family,
                    "path": str(DEFAULT_AUTHORED_SEQUENTIAL_CANDIDATES_PATH).replace(
                        "\\", "/"
                    ),
                },
            )
        )

    imp = {
        c.candidate_id: c
        for c in load_authored_candidates(DEFAULT_AUTHORED_CANDIDATES_PATH)
    }
    for cid in AUTHORED_IMPLICIT_SPARE_IDS:
        if cid in v1_index["candidate_ids"] or cid in AUTHORED_IMPLICIT_SELECTED_IDS:
            continue
        item = imp[cid]
        domain = infer_domain(item.query)
        semantic_group = f"{domain}__{item.template_group}"
        out.append(
            _base_candidate(
                source_kind="authored",
                source_id=None,
                candidate_id=cid,
                query=item.query,
                proposed_final_bucket="MIXED_IMPLICIT",
                semantic_group=semantic_group,
                template_group=item.template_group,
                authored_template_family=item.template_group,
                original_label="Mixed",
                review_status="available",
                acceptance_reason="authored_implicit_spare_unused",
                rejection_reason=None,
                operator_family=None,
                h5_positive=True,
                h7_positive=False,
                h7_families=[],
                provenance={
                    "origin": "authored_implicit_spare",
                    "path": str(DEFAULT_AUTHORED_CANDIDATES_PATH).replace("\\", "/"),
                },
            )
        )
    return out


def assemble_mined_implicit_review_pool(
    *,
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    reviews_path: str | Path = DEFAULT_REVIEWS_PATH,
    v1_mine_path: str | Path = DEFAULT_V1_IMPLICIT_MINE_PATH,
    v1_index: Mapping[str, Any],
    limit: int = V2_IMPLICIT_MINE_LIMIT,
    seed: int = V2_CANDIDATE_ASSEMBLY_SEED,
) -> list[dict[str, Any]]:
    """Expand mining into a reviewable pool; never auto-accept."""
    pool = build_unique_query_pool(load_classification_rows(train_path))
    excluded = excluded_normalized_queries(
        unique_pool=pool, reviews_path=reviews_path
    )
    excluded |= set(v1_index["query_keys"])

    existing_v1 = {
        c.source_id: c for c in load_implicit_candidates_jsonl(v1_mine_path)
    }
    mined = mine_implicit_candidates(
        pool, excluded_keys=excluded, limit=limit, seed=seed
    )

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Prior v1 mine leftovers (not selected) still need human review.
    for source_id, cand in sorted(existing_v1.items()):
        if source_id in v1_index["source_ids"]:
            continue
        key = normalize_query_key(cand.query)
        if key in seen:
            continue
        seen.add(key)
        domain = infer_domain(cand.query)
        template_group = infer_pe_template_group(cand.query)
        semantic_group = infer_pe_semantic_group(cand.query, domain, template_group)
        out.append(
            _base_candidate(
                source_kind="mined",
                source_id=cand.source_id,
                candidate_id=None,
                query=cand.query,
                proposed_final_bucket="MIXED_IMPLICIT",
                semantic_group=semantic_group,
                template_group=template_group,
                authored_template_family=None,
                original_label=cand.original_label,
                review_status="needs_review",
                acceptance_reason=None,
                rejection_reason=None,
                operator_family=None,
                h5_positive=True,  # hypothesized; requires human confirm
                h7_positive=False,
                h7_families=[],
                provenance={
                    "origin": "v1_implicit_mine_leftover",
                    "path": str(Path(v1_mine_path)).replace("\\", "/"),
                    "score": cand.score,
                    "mining_reasons": list(cand.mining_reasons),
                    "auto_accepted": False,
                },
            )
        )

    for cand in mined:
        key = normalize_query_key(cand.query)
        if key in seen or cand.source_id in v1_index["source_ids"]:
            continue
        if cand.source_id in existing_v1:
            continue
        seen.add(key)
        domain = infer_domain(cand.query)
        template_group = infer_pe_template_group(cand.query)
        semantic_group = infer_pe_semantic_group(cand.query, domain, template_group)
        out.append(
            _base_candidate(
                source_kind="mined",
                source_id=cand.source_id,
                candidate_id=None,
                query=cand.query,
                proposed_final_bucket="MIXED_IMPLICIT",
                semantic_group=semantic_group,
                template_group=template_group,
                authored_template_family=None,
                original_label=cand.original_label,
                review_status="needs_review",
                acceptance_reason=None,
                rejection_reason=None,
                operator_family=None,
                h5_positive=True,
                h7_positive=False,
                h7_families=[],
                provenance={
                    "origin": "v2_implicit_mine",
                    "path": str(STAGE_A_V2_IMPLICIT_MINE_PATH).replace("\\", "/"),
                    "score": cand.score,
                    "mining_reasons": list(cand.mining_reasons),
                    "auto_accepted": False,
                },
            )
        )
    return out


def compute_legacy_h5_h7_inventory() -> dict[str, Any]:
    """Read-only H5/H7 distribution over the frozen legacy Stage-A 120."""
    import re

    from tiergraph.enums import OperatorType
    from tiergraph.planner.annotation_step_a import DEFAULT_STEP_A_ANNOTATIONS_PATH
    from tiergraph.planner.annotation_step_b import DEFAULT_STEP_B_ANNOTATIONS_PATH
    from tiergraph.planner.stage_a_to_corpus import (
        count_explicit_h7_edges,
        load_stage_a_planner_examples,
    )

    answer_ops = {
        OperatorType.RETRIEVE_PERSONAL,
        OperatorType.IDENTIFY_ENVIRONMENTAL,
        OperatorType.DESCRIBE_ENVIRONMENT,
        OperatorType.LOCATE_ENVIRONMENTAL,
        OperatorType.NAVIGATE_TO,
    }
    h5_by_bucket: Counter[str] = Counter()
    h7_family_examples: Counter[str] = Counter()
    h7_pos_by_bucket: Counter[str] = Counter()
    multi_hop = 0
    examples = load_stage_a_planner_examples()
    for ex in examples:
        bucket = str(ex.metadata["final_bucket"])
        nodes = {node.node_id: node for node in ex.graph.nodes}
        if any(n.operator == OperatorType.RESOLVE_PERSONAL for n in nodes.values()):
            h5_by_bucket[bucket] += 1
        families: set[str] = set()
        for edge in ex.graph.edges:
            src = nodes.get(edge.source_node_id)
            tgt = nodes.get(edge.target_node_id)
            if src is None or tgt is None:
                continue
            if src.operator not in answer_ops or tgt.operator not in answer_ops:
                continue
            if is_legal_h7_pair(src.operator.value, tgt.operator.value):
                families.add(h7_family_label(src.operator.value, tgt.operator.value))
        if families:
            h7_pos_by_bucket[bucket] += 1
            for label in families:
                h7_family_examples[label] += 1
        if count_explicit_h7_edges(ex) >= 2:
            multi_hop += 1

    poss = re.compile(r"\b(my|mine|our)\b", re.IGNORECASE)
    step_a = {
        row["stage_a_id"]: row for row in load_jsonl(DEFAULT_STEP_A_ANNOTATIONS_PATH)
    }
    step_b = {
        row["stage_a_id"]: row for row in load_jsonl(DEFAULT_STEP_B_ANNOTATIONS_PATH)
    }
    h5_none_controls = 0
    h5_none_by_bucket: Counter[str] = Counter()
    for stage_a_id, row_a in step_a.items():
        row_b = step_b[stage_a_id]
        has_retrieve = any(
            op["operator_type"] == "RETRIEVE_PERSONAL" for op in row_a["operations"]
        )
        if not has_retrieve or not poss.search(str(row_a["query"])):
            continue
        decisions = row_b.get("anchor_decisions") or []
        if decisions and all(
            d.get("implicit_resolution") == "NONE" for d in decisions
        ):
            h5_none_controls += 1
            h5_none_by_bucket[str(row_a["final_bucket"])] += 1

    return {
        "h5_positive_total": sum(h5_by_bucket.values()),
        "h5_positive_by_bucket": dict(h5_by_bucket),
        "h7_positive_total": sum(h7_pos_by_bucket.values()),
        "h7_positive_by_bucket": dict(h7_pos_by_bucket),
        "h7_family_example_counts": dict(h7_family_examples),
        "h7_multi_hop_examples": multi_hop,
        "h5_negative_retrieve_possessive_controls": h5_none_controls,
        "h5_negative_retrieve_possessive_by_bucket": dict(h5_none_by_bucket),
    }


def build_authored_family_specs() -> list[dict[str, Any]]:
    """Declarative authored pack specs (Todo 2b capacity closure; no examples)."""
    specs: list[dict[str, Any]] = []

    def add_seq(
        family: str,
        *,
        h7_families: Sequence[str],
        h5_positive: bool,
        operator_family: Sequence[str],
        planned_paraphrases: int,
        rationale: str,
        multi_hop: bool = False,
    ) -> None:
        for label in h7_families:
            src, tgt = parse_h7_family_label(label)
            if not is_legal_h7_pair(src, tgt):
                raise ValueError(f"spec proposes illegal H7 family {label}")
        specs.append(
            {
                "spec_id": f"spec_seq_{family}",
                "spec_kind": "authored_sequential_family",
                "authored_template_family": family,
                "scenario_family": family,
                "authored_holdout_family": AUTHORED_HOLDOUT_FAMILY_LINKS.get(
                    family, family
                ),
                "proposed_final_bucket": "MIXED_SEQUENTIAL",
                "source_kind": "authored",
                "planned_paraphrases": planned_paraphrases,
                "operator_family": list(operator_family),
                "h5_positive": h5_positive,
                "h7_positive": bool(h7_families),
                "h7_families": list(h7_families),
                "multi_hop": multi_hop,
                "review_status": "spec_only",
                "rationale": rationale,
            }
        )

    # --- MIXED_SEQUENTIAL (~78 planned) ---
    # Pure legal H7 singles (no H5)
    add_seq(
        "identify_locate_plaque_vs_desk",
        h7_families=["IDENTIFY_ENVIRONMENTAL->LOCATE_ENVIRONMENTAL"],
        h5_positive=False,
        operator_family=["IDENTIFY_ENVIRONMENTAL", "LOCATE_ENVIRONMENTAL"],
        planned_paraphrases=4,
        rationale="Identify object then locate it in workplace scenes",
    )
    add_seq(
        "identify_locate_menu_dish_station",
        h7_families=["IDENTIFY_ENVIRONMENTAL->LOCATE_ENVIRONMENTAL"],
        h5_positive=False,
        operator_family=["IDENTIFY_ENVIRONMENTAL", "LOCATE_ENVIRONMENTAL"],
        planned_paraphrases=4,
        rationale="Identify dish/sign then locate serving station",
    )
    add_seq(
        "locate_navigate_clinic_corridor",
        h7_families=["LOCATE_ENVIRONMENTAL->NAVIGATE_TO"],
        h5_positive=False,
        operator_family=["LOCATE_ENVIRONMENTAL", "NAVIGATE_TO"],
        planned_paraphrases=3,
        rationale="Locate clinic room then navigate",
    )
    add_seq(
        "locate_navigate_platform_exit",
        h7_families=["LOCATE_ENVIRONMENTAL->NAVIGATE_TO"],
        h5_positive=False,
        operator_family=["LOCATE_ENVIRONMENTAL", "NAVIGATE_TO"],
        planned_paraphrases=2,
        rationale="Locate platform/gate then navigate without RESOLVE",
    )
    add_seq(
        "identify_describe_device_status",
        h7_families=["IDENTIFY_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT"],
        h5_positive=False,
        operator_family=["IDENTIFY_ENVIRONMENTAL", "DESCRIBE_ENVIRONMENT"],
        planned_paraphrases=5,
        rationale="Name device then describe status/state",
    )
    add_seq(
        "identify_describe_signage_content",
        h7_families=["IDENTIFY_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT"],
        h5_positive=False,
        operator_family=["IDENTIFY_ENVIRONMENTAL", "DESCRIBE_ENVIRONMENT"],
        planned_paraphrases=5,
        rationale="Identify sign/panel then describe readable content",
    )
    add_seq(
        "identify_describe_packaging_label",
        h7_families=["IDENTIFY_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT"],
        h5_positive=False,
        operator_family=["IDENTIFY_ENVIRONMENTAL", "DESCRIBE_ENVIRONMENT"],
        planned_paraphrases=5,
        rationale="Identify package then describe label attributes",
    )
    add_seq(
        "locate_describe_shelf_contents",
        h7_families=["LOCATE_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT"],
        h5_positive=False,
        operator_family=["LOCATE_ENVIRONMENTAL", "DESCRIBE_ENVIRONMENT"],
        planned_paraphrases=5,
        rationale="Locate shelf/bin then describe contents",
    )
    add_seq(
        "locate_describe_room_layout",
        h7_families=["LOCATE_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT"],
        h5_positive=False,
        operator_family=["LOCATE_ENVIRONMENTAL", "DESCRIBE_ENVIRONMENT"],
        planned_paraphrases=5,
        rationale="Locate room area then describe layout/atmosphere",
    )
    add_seq(
        "locate_describe_vehicle_bay",
        h7_families=["LOCATE_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT"],
        h5_positive=False,
        operator_family=["LOCATE_ENVIRONMENTAL", "DESCRIBE_ENVIRONMENT"],
        planned_paraphrases=4,
        rationale="Locate bay/stall then describe what is there",
    )
    # Multi-hop legal chains (count toward multi-hop floor + two families)
    add_seq(
        "identify_locate_navigate_building_exit",
        h7_families=[
            "IDENTIFY_ENVIRONMENTAL->LOCATE_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL->NAVIGATE_TO",
        ],
        h5_positive=False,
        operator_family=[
            "IDENTIFY_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL",
            "NAVIGATE_TO",
        ],
        planned_paraphrases=5,
        multi_hop=True,
        rationale="Multi-hop IDENTIFY→LOCATE→NAVIGATE for exits/landmarks",
    )
    add_seq(
        "identify_locate_navigate_counter_queue",
        h7_families=[
            "IDENTIFY_ENVIRONMENTAL->LOCATE_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL->NAVIGATE_TO",
        ],
        h5_positive=False,
        operator_family=[
            "IDENTIFY_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL",
            "NAVIGATE_TO",
        ],
        planned_paraphrases=5,
        multi_hop=True,
        rationale="Multi-hop find counter/desk then navigate in queue scenes",
    )
    # H5-positive sequential (~22 planned) toward 40–50 sequential H5 total
    add_seq(
        "resolve_locate_navigate_lab_draw_station",
        h7_families=["LOCATE_ENVIRONMENTAL->NAVIGATE_TO"],
        h5_positive=True,
        operator_family=["LOCATE_ENVIRONMENTAL", "NAVIGATE_TO"],
        planned_paraphrases=4,
        rationale=(
            "H5 lab/appointment resolve → locate blood-draw/imaging station → "
            "navigate (not a locker/pickup twin of quarantined order_pickup)"
        ),
    )
    add_seq(
        "resolve_locate_navigate_rental_stall",
        h7_families=["LOCATE_ENVIRONMENTAL->NAVIGATE_TO"],
        h5_positive=True,
        operator_family=["LOCATE_ENVIRONMENTAL", "NAVIGATE_TO"],
        planned_paraphrases=3,
        rationale="H5 reservation → locate stall → navigate",
    )
    add_seq(
        "resolve_identify_locate_baggage_belt",
        h7_families=["IDENTIFY_ENVIRONMENTAL->LOCATE_ENVIRONMENTAL"],
        h5_positive=True,
        operator_family=["IDENTIFY_ENVIRONMENTAL", "LOCATE_ENVIRONMENTAL"],
        planned_paraphrases=4,
        rationale="H5 flight/bag resolve → identify belt → locate",
    )
    add_seq(
        "resolve_identify_describe_prescription_bottle",
        h7_families=["IDENTIFY_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT"],
        h5_positive=True,
        operator_family=["IDENTIFY_ENVIRONMENTAL", "DESCRIBE_ENVIRONMENT"],
        planned_paraphrases=4,
        rationale="H5 med resolve → identify bottle → describe label",
    )
    add_seq(
        "resolve_locate_describe_appointment_room",
        h7_families=["LOCATE_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT"],
        h5_positive=True,
        operator_family=["LOCATE_ENVIRONMENTAL", "DESCRIBE_ENVIRONMENT"],
        planned_paraphrases=3,
        rationale="H5 appointment resolve → locate room → describe entry cues",
    )
    add_seq(
        "resolve_only_identify_seat_marker",
        h7_families=[],
        h5_positive=True,
        operator_family=["IDENTIFY_ENVIRONMENTAL"],
        planned_paraphrases=4,
        rationale="H5→IDENTIFY only (no learned H7); sequential via implicit resolve",
    )
    add_seq(
        "resolve_only_describe_cabin_map",
        h7_families=[],
        h5_positive=True,
        operator_family=["DESCRIBE_ENVIRONMENT"],
        planned_paraphrases=3,
        rationale="H5→DESCRIBE only; keeps sequential H5 without inflating one H7 family",
    )

    def add_imp(
        family: str,
        *,
        planned: int,
        rationale: str,
        operator_hint: Sequence[str] | None = None,
    ) -> None:
        specs.append(
            {
                "spec_id": f"spec_imp_{family}",
                "spec_kind": "authored_implicit_family",
                "authored_template_family": family,
                "scenario_family": family,
                "authored_holdout_family": AUTHORED_HOLDOUT_FAMILY_LINKS.get(
                    family, family
                ),
                "proposed_final_bucket": "MIXED_IMPLICIT",
                "source_kind": "authored",
                "planned_paraphrases": planned,
                "operator_family": list(operator_hint) if operator_hint else None,
                "h5_positive": True,
                "h7_positive": False,
                "h7_families": [],
                "review_status": "spec_only",
                "rationale": rationale,
            }
        )

    # --- MIXED_IMPLICIT (~82 planned), all require H5 IMPLICIT_RESOLVE_PERSONAL ---
    add_imp(
        "my_medication_scene_match",
        planned=7,
        operator_hint=["IDENTIFY_ENVIRONMENTAL"],
        rationale="my med/prescription must resolve before matching visible pill/bottle",
    )
    add_imp(
        "my_allergy_menu_safe",
        planned=6,
        operator_hint=["DESCRIBE_ENVIRONMENT", "IDENTIFY_ENVIRONMENTAL"],
        rationale="allergy profile resolve before judging dish/menu item",
    )
    add_imp(
        "my_appointment_entrance_cue",
        planned=6,
        operator_hint=["LOCATE_ENVIRONMENTAL", "IDENTIFY_ENVIRONMENTAL"],
        rationale="appointment resolve into finding the correct entrance/door",
    )
    add_imp(
        "my_reservation_seat_marker",
        planned=6,
        operator_hint=["IDENTIFY_ENVIRONMENTAL"],
        rationale="reservation/seat resolve into identifying labeled marker",
    )
    add_imp(
        "my_order_pickup_window",
        planned=6,
        operator_hint=["LOCATE_ENVIRONMENTAL"],
        rationale="order account resolve into locating pickup window",
    )
    add_imp(
        "my_boarding_pass_gate_board",
        planned=6,
        operator_hint=["LOCATE_ENVIRONMENTAL", "IDENTIFY_ENVIRONMENTAL"],
        rationale="boarding/gate facts resolve into reading gate board",
    )
    add_imp(
        "my_luggage_carousel_tag",
        planned=5,
        operator_hint=["IDENTIFY_ENVIRONMENTAL"],
        rationale="bag ownership resolve into identifying tagged luggage",
    )
    add_imp(
        "my_meeting_room_directory",
        planned=5,
        operator_hint=["DESCRIBE_ENVIRONMENT", "LOCATE_ENVIRONMENTAL"],
        rationale="calendar/meeting resolve into directory/room description",
    )
    add_imp(
        "my_dietary_restriction_dish",
        planned=5,
        operator_hint=["DESCRIBE_ENVIRONMENT"],
        rationale="dietary profile resolve before describing dish ingredients",
    )
    add_imp(
        "my_prescription_label_dose",
        planned=5,
        operator_hint=["DESCRIBE_ENVIRONMENT"],
        rationale="prescription resolve into reading dose text on label",
    )
    add_imp(
        "my_hotel_room_key_panel",
        planned=5,
        operator_hint=["IDENTIFY_ENVIRONMENTAL", "LOCATE_ENVIRONMENTAL"],
        rationale="hotel reservation resolve into key/panel identification",
    )
    add_imp(
        "my_package_locker_bank",
        planned=5,
        operator_hint=["LOCATE_ENVIRONMENTAL"],
        rationale="delivery/package resolve into locating locker bank",
    )
    add_imp(
        "my_train_platform_reservation",
        planned=5,
        operator_hint=["IDENTIFY_ENVIRONMENTAL"],
        rationale="Rail reservation resolve into identifying/matching the correct platform",
    )
    add_imp(
        "my_workspace_badge_reader",
        planned=5,
        operator_hint=["IDENTIFY_ENVIRONMENTAL", "DESCRIBE_ENVIRONMENT"],
        rationale="badge/access profile resolve into reader/panel identify-describe",
    )
    add_imp(
        "my_cart_item_shelf_match",
        planned=5,
        operator_hint=["IDENTIFY_ENVIRONMENTAL"],
        rationale="shopping-list ownership resolve into matching shelf item",
    )

    hard_cases = [
        (
            "urgency_distractor_scene",
            "Environmental",
            ["DESCRIBE_ENVIRONMENT"],
            4,
            None,
            "Urgency/distractor clauses; H2 span discipline",
        ),
        (
            "hyphenated_entity_identify",
            "Environmental",
            ["IDENTIFY_ENVIRONMENTAL"],
            4,
            None,
            "Hyphenated expressions as single H2/H3 unit",
        ),
        (
            "identify_vs_describe_minimal_pair",
            "Environmental",
            ["IDENTIFY_ENVIRONMENTAL", "DESCRIBE_ENVIRONMENT"],
            4,
            None,
            "Minimal-pair contrast IDENTIFY vs DESCRIBE",
        ),
        (
            "retrieve_vs_describe_personal",
            "Personal",
            ["RETRIEVE_PERSONAL"],
            4,
            False,
            "RETRIEVE_PERSONAL vs DESCRIBE contrast on personal facts; H5=NONE",
        ),
        (
            "locate_vs_identify_describe",
            "Environmental",
            ["LOCATE_ENVIRONMENTAL"],
            4,
            None,
            "LOCATE vs IDENTIFY/DESCRIBE contrast",
        ),
        (
            "navigate_direct_route",
            "Environmental",
            ["NAVIGATE_TO"],
            4,
            None,
            "NAVIGATE coverage without illegal H7 padding",
        ),
        (
            "retrieve_possessive_h5_none",
            "Personal",
            ["RETRIEVE_PERSONAL"],
            16,
            False,
            "RETRIEVE + my X with H5=NONE controls toward >=40 corpus-wide",
        ),
    ]
    for family, bucket, ops, planned, h5_flag, rationale in hard_cases:
        specs.append(
            {
                "spec_id": f"spec_hard_{family}",
                "spec_kind": "h2_h3_hard_case_family",
                "authored_template_family": family,
                "scenario_family": family,
                "authored_holdout_family": AUTHORED_HOLDOUT_FAMILY_LINKS.get(
                    family, family
                ),
                "proposed_final_bucket": bucket,
                "source_kind": "authored",
                "planned_paraphrases": planned,
                "operator_family": list(ops),
                "h5_positive": h5_flag,
                "h7_positive": False,
                "h7_families": [],
                "review_status": "spec_only",
                "rationale": rationale,
            }
        )

    return sorted(specs, key=lambda s: s["spec_id"])


def summarize_spec_capacity(specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate planned paraphrase capacity and projected H7/H5 fills."""
    planned_by_bucket: Counter[str] = Counter()
    families_by_bucket: dict[str, set[str]] = {b: set() for b in STAGE_A_V2_BUCKETS}
    h7_planned: Counter[str] = Counter()
    multi_hop_planned = 0
    seq_h5_planned = 0
    seq_h7_pos_planned = 0
    h5_none_planned = 0
    for spec in specs:
        bucket = str(spec["proposed_final_bucket"])
        n = int(spec["planned_paraphrases"])
        planned_by_bucket[bucket] += n
        families_by_bucket.setdefault(bucket, set()).add(
            str(spec["authored_template_family"])
        )
        if spec.get("spec_kind") == "authored_sequential_family":
            if spec.get("h5_positive"):
                seq_h5_planned += n
            if spec.get("h7_positive"):
                seq_h7_pos_planned += n
            if spec.get("multi_hop"):
                multi_hop_planned += n
            for label in spec.get("h7_families") or []:
                h7_planned[str(label)] += n
        if (
            spec.get("spec_kind") == "h2_h3_hard_case_family"
            and spec.get("h5_positive") is False
            and "possessive" in str(spec.get("authored_template_family"))
        ):
            h5_none_planned += n
    return {
        "planned_paraphrases_by_bucket": dict(planned_by_bucket),
        "distinct_authored_families_by_bucket": {
            bucket: len(families) for bucket, families in families_by_bucket.items() if families
        },
        "h7_planned_family_counts": dict(h7_planned),
        "multi_hop_planned": multi_hop_planned,
        "sequential_h5_positive_planned": seq_h5_planned,
        "sequential_h7_positive_planned": seq_h7_pos_planned,
        "retrieve_possessive_h5_none_planned": h5_none_planned,
    }


def validate_candidate_row(row: Mapping[str, Any]) -> list[str]:
    """Validate inventory-row provenance + H7 legality constraints."""
    errors: list[str] = []
    for field in CANDIDATE_PROVENANCE_FIELDS:
        if field not in row:
            errors.append(f"missing candidate field {field!r}")

    # Map onto Stage-A provenance validator (uses final_bucket).
    prov_row = {
        "source_kind": row.get("source_kind"),
        "authored_template_family": row.get("authored_template_family"),
        "template_group": row.get("template_group"),
        "semantic_group": row.get("semantic_group"),
        "final_bucket": row.get("proposed_final_bucket") or row.get("final_bucket"),
        "operator_family": row.get("operator_family"),
        "h5_positive": row.get("h5_positive"),
        "h7_positive": row.get("h7_positive"),
        "h7_families": row.get("h7_families"),
    }
    errors.extend(validate_provenance_metadata(prov_row))
    errors.extend(validate_authored_family_on_row(prov_row))

    if row.get("proposed_final_bucket") == "MIXED_PARALLEL":
        if row.get("h7_positive") is True or row.get("h7_families"):
            errors.append("MIXED_PARALLEL candidate must not propose explicit H7")

    for label in row.get("h7_families") or []:
        src, tgt = parse_h7_family_label(str(label))
        if not is_legal_h7_pair(src, tgt):
            errors.append(f"illegal H7 family on candidate: {label}")

    if row.get("source_kind") == "authored" and not row.get("authored_template_family"):
        errors.append("authored candidate missing authored_template_family")

    return errors


def _coverage_shortfalls(
    available_by_bucket: Mapping[str, int],
    *,
    v1_by_bucket: Mapping[str, int],
) -> dict[str, Any]:
    shortfalls: dict[str, Any] = {}
    for bucket in STAGE_A_V2_BUCKETS:
        need_total = STAGE_A_V2_PER_BUCKET
        have_v1 = int(v1_by_bucket.get(bucket, 0))
        need_new = max(0, need_total - have_v1)
        available = int(available_by_bucket.get(bucket, 0))
        shortfalls[bucket] = {
            "target": need_total,
            "already_in_v1": have_v1,
            "new_needed": need_new,
            "available_accepted_or_reviewable": available,
            "shortfall_vs_new_needed": max(0, need_new - available),
        }
    return shortfalls


def build_candidate_inventory(
    *,
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    selection_path: str | Path = STAGE_A_V1_SELECTION_PATH,
    spares_path: str | Path = DEFAULT_SPARES_PATH,
    reviews_path: str | Path = DEFAULT_MIXED_REVIEWS_PATH,
    mine_limit: int = V2_IMPLICIT_MINE_LIMIT,
    seed: int = V2_CANDIDATE_ASSEMBLY_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Assemble deterministic candidate inventory + authored specs + report."""
    v1_index = load_v1_frozen_index(selection_path)

    pe, pe_conflicts = assemble_personal_environmental_candidates(
        train_path=train_path, v1_index=v1_index
    )
    parallel, parallel_rejected = assemble_parallel_candidates(
        spares_path=spares_path, reviews_path=reviews_path, v1_index=v1_index
    )
    sequential, sequential_rejected = assemble_natural_sequential_candidates(
        reviews_path=reviews_path, spares_path=spares_path, v1_index=v1_index
    )
    authored_spares = assemble_authored_spare_candidates(v1_index=v1_index)
    mined = assemble_mined_implicit_review_pool(
        train_path=train_path,
        reviews_path=reviews_path,
        v1_index=v1_index,
        limit=mine_limit,
        seed=seed,
    )
    specs = build_authored_family_specs()

    candidates = pe + parallel + sequential + authored_spares + mined
    candidates = sorted(
        candidates,
        key=lambda row: (
            str(row["proposed_final_bucket"]),
            str(row["source_kind"]),
            str(row.get("source_id") or ""),
            str(row.get("candidate_id") or ""),
            str(row["candidate_uid"]),
        ),
    )

    # Dedup normalized queries across buckets (keep first in sort order)
    seen_keys: set[str] = set()
    deduped: list[dict[str, Any]] = []
    dup_conflicts: list[dict[str, Any]] = []
    for row in candidates:
        key = str(row["normalized_query"])
        if key in seen_keys or key in v1_index["query_keys"]:
            dup_conflicts.append(
                {
                    "candidate_uid": row["candidate_uid"],
                    "normalized_query": key,
                    "reason": (
                        "duplicate_against_frozen_v1"
                        if key in v1_index["query_keys"]
                        else "duplicate_within_v2_inventory"
                    ),
                }
            )
            continue
        seen_keys.add(key)
        deduped.append(row)
    candidates = deduped

    validation_errors = []
    for row in candidates:
        errs = validate_candidate_row(row)
        if errs:
            validation_errors.append({"candidate_uid": row["candidate_uid"], "errors": errs})
    if validation_errors:
        raise ValueError(
            f"candidate inventory failed validation ({len(validation_errors)} rows): "
            f"{validation_errors[:3]}"
        )

    for spec in specs:
        for label in spec.get("h7_families") or []:
            src, tgt = parse_h7_family_label(label)
            if not is_legal_h7_pair(src, tgt):
                raise ValueError(f"authored spec has illegal H7: {label}")

    available_by_bucket: Counter[str] = Counter()
    for row in candidates:
        if row["review_status"] in {"available", "needs_review"}:
            available_by_bucket[str(row["proposed_final_bucket"])] += 1

    capacity = summarize_spec_capacity(specs)
    planned_by_bucket = Counter(capacity["planned_paraphrases_by_bucket"])
    h7_planned = Counter(capacity["h7_planned_family_counts"])

    h7_available: Counter[str] = Counter()
    for row in candidates:
        for label in row.get("h7_families") or []:
            h7_available[str(label)] += 1

    legacy = compute_legacy_h5_h7_inventory()
    concrete_seq = [
        r for r in candidates if r["proposed_final_bucket"] == "MIXED_SEQUENTIAL"
    ]
    concrete_imp = [
        r for r in candidates if r["proposed_final_bucket"] == "MIXED_IMPLICIT"
    ]
    expected_h7_family_totals = {
        label: int(legacy["h7_family_example_counts"].get(label, 0))
        + int(h7_available.get(label, 0))
        + int(h7_planned.get(label, 0))
        for label in LEGAL_H7_FAMILY_LABELS
    }
    expected_h7_positive = (
        int(legacy["h7_positive_total"])
        + sum(1 for r in concrete_seq if r.get("h7_positive"))
        + int(capacity["sequential_h7_positive_planned"])
    )
    expected_multi_hop = int(legacy["h7_multi_hop_examples"]) + int(
        capacity["multi_hop_planned"]
    )
    expected_seq_h5 = (
        int(legacy["h5_positive_by_bucket"].get("MIXED_SEQUENTIAL", 0))
        + sum(1 for r in concrete_seq if r.get("h5_positive"))
        + int(capacity["sequential_h5_positive_planned"])
    )
    expected_imp_h5 = (
        int(legacy["h5_positive_by_bucket"].get("MIXED_IMPLICIT", 0))
        + len(concrete_imp)
        + int(planned_by_bucket.get("MIXED_IMPLICIT", 0))
    )
    expected_h5_none = int(legacy["h5_negative_retrieve_possessive_controls"]) + int(
        capacity["retrieve_possessive_h5_none_planned"]
    )
    h7_share = {
        label: (count / expected_h7_positive if expected_h7_positive else 0.0)
        for label, count in expected_h7_family_totals.items()
    }
    remaining_impossible: list[str] = []
    for label, minimum in H7_FAMILY_MINIMUMS.items():
        if expected_h7_family_totals.get(label, 0) < minimum:
            remaining_impossible.append(
                f"{label}: expected {expected_h7_family_totals.get(label, 0)} "
                f"< min {minimum}"
            )
    if expected_multi_hop < H7_MULTI_HOP_MINIMUM:
        remaining_impossible.append(
            f"multi_hop: expected {expected_multi_hop} < min {H7_MULTI_HOP_MINIMUM}"
        )
    lo, hi = H7_POSITIVE_EXAMPLE_TARGET_RANGE
    if expected_h7_positive < lo:
        remaining_impossible.append(
            f"h7_positive_examples: expected {expected_h7_positive} < target low {lo}"
        )
    for label, share in h7_share.items():
        if share > 0.35 + 1e-9:
            remaining_impossible.append(
                f"{label} share {share:.1%} exceeds 35% of expected H7-positive"
            )
    if expected_h5_none < H5_NEGATIVE_RETRIEVE_POSSESSIVE_MIN:
        remaining_impossible.append(
            f"H5-NONE retrieve+possessive: expected {expected_h5_none} "
            f"< min {H5_NEGATIVE_RETRIEVE_POSSESSIVE_MIN}"
        )

    personal = [r for r in candidates if r["proposed_final_bucket"] == "Personal"]
    environmental = [
        r for r in candidates if r["proposed_final_bucket"] == "Environmental"
    ]
    report = {
        "A_personal_natural_available": len(personal),
        "B_environmental_natural_available": len(environmental),
        "C_mixed_parallel_accepted": len(
            [r for r in candidates if r["proposed_final_bucket"] == "MIXED_PARALLEL"]
        ),
        "D_natural_true_sequential_available": len(
            [
                r
                for r in candidates
                if r["proposed_final_bucket"] == "MIXED_SEQUENTIAL"
                and r["source_kind"] == "natural"
            ]
        ),
        "E_implicit_mined_needs_review": len(
            [r for r in candidates if r["source_kind"] == "mined"]
        ),
        "F_authored_sequential_family_specs": [
            s
            for s in specs
            if s["spec_kind"] == "authored_sequential_family"
        ],
        "G_authored_implicit_family_specs": [
            s for s in specs if s["spec_kind"] == "authored_implicit_family"
        ],
        "H_h2_h3_hard_case_family_specs": [
            s for s in specs if s["spec_kind"] == "h2_h3_hard_case_family"
        ],
        "I_duplicate_conflicts_with_frozen_v1": pe_conflicts
        + [
            c
            for c in parallel_rejected + sequential_rejected + dup_conflicts
            if "duplicate_against_frozen_v1" in str(c.get("reason"))
        ],
        "J_coverage_shortfalls": _coverage_shortfalls(
            available_by_bucket,
            v1_by_bucket=v1_index["by_bucket"],
        ),
        "J_planned_authored_paraphrase_capacity": dict(planned_by_bucket),
        "K_h7_legal_family_inventory": {
            "legal_families": list(LEGAL_H7_FAMILY_LABELS),
            "family_minimums": dict(H7_FAMILY_MINIMUMS),
            "legacy_family_counts": legacy["h7_family_example_counts"],
            "available_on_concrete_candidates": dict(h7_available),
            "planned_via_authored_specs": dict(h7_planned),
            "expected_final_family_totals_if_specs_filled": expected_h7_family_totals,
            "expected_final_h7_positive_examples": expected_h7_positive,
            "expected_final_multi_hop": expected_multi_hop,
            "expected_family_share_of_h7_positive": {
                k: round(v, 4) for k, v in h7_share.items()
            },
            "natural_true_sequential_h7_pair_labels_unknown_until_annotation": len(
                [
                    r
                    for r in candidates
                    if r["proposed_final_bucket"] == "MIXED_SEQUENTIAL"
                    and r["source_kind"] == "natural"
                ]
            ),
        },
        "L_legacy_h5_h7": legacy,
        "M_spec_capacity_summary": capacity,
        "N_projected_corpus_fills": {
            "mixed_implicit_h5_positive_if_specs_filled": expected_imp_h5,
            "mixed_sequential_h5_positive_if_specs_filled": expected_seq_h5,
            "retrieve_possessive_h5_none_if_specs_filled": expected_h5_none,
            "remaining_short_or_impossible": remaining_impossible,
        },
        "parallel_rejections": parallel_rejected,
        "sequential_rejections": sequential_rejected,
        "counts_by_bucket_and_source": {
            bucket: dict(
                Counter(
                    r["source_kind"]
                    for r in candidates
                    if r["proposed_final_bucket"] == bucket
                )
            )
            for bucket in STAGE_A_V2_BUCKETS
        },
        "candidate_total": len(candidates),
        "authored_spare_concrete": len(authored_spares),
        "assembly_seed": seed,
        "mine_limit": mine_limit,
    }

    for bucket, row in report["J_coverage_shortfalls"].items():
        planned = planned_by_bucket.get(bucket, 0)
        concrete = int(available_by_bucket.get(bucket, 0))
        row["planned_authored_paraphrases_not_yet_generated"] = planned
        row["reviewable_capacity_concrete_plus_planned"] = concrete + planned
        row["still_short_if_specs_filled"] = max(
            0, row["shortfall_vs_new_needed"] - planned
        )

    return candidates, specs, report


def write_candidate_inventory(
    *,
    candidates_path: str | Path = STAGE_A_V2_CANDIDATES_PATH,
    specs_path: str | Path = STAGE_A_V2_AUTHORED_SPECS_PATH,
    report_path: str | Path = STAGE_A_V2_CANDIDATE_REPORT_PATH,
    mine_review_path: str | Path = STAGE_A_V2_IMPLICIT_MINE_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write inventory artifacts (never writes stage_a_v2_final_selection)."""
    candidates, specs, report = build_candidate_inventory(**kwargs)
    _write_jsonl(Path(candidates_path), candidates)
    _write_jsonl(Path(specs_path), specs)
    mined_rows = [r for r in candidates if r["source_kind"] == "mined"]
    _write_jsonl(Path(mine_review_path), mined_rows)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = write_candidate_inventory()
    print(json.dumps({k: report[k] for k in report if k[0] in "ABCDEFGHIJK" or k.startswith("J_") or k.startswith("K_") or k in {"candidate_total", "authored_spare_concrete"}}, indent=2))


if __name__ == "__main__":
    main()
