#!/usr/bin/env python3
"""Validate Phase-5 planner semantic annotations / candidate pools.

Checks identity, span, ownership, H7 eligibility, and GraphDecoder→PlannerExample
conversion. Prints corpus summary counts.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.corpus import (
    PlannerSemanticAnnotation,
    load_candidates_jsonl,
    load_semantic_annotations_jsonl,
    normalize_query_key,
    semantic_annotation_to_planner_example,
)
from tiergraph.planner.decode import PlannerDecodeError
from tiergraph.planner.operator_io import is_h7_pair_eligible


def _validate_annotation(
    annotation: PlannerSemanticAnnotation,
    *,
    seen_ids: set[str],
    seen_groups: dict[str, str],
    seen_query_keys: dict[str, str],
) -> None:
    if annotation.source_query_id in seen_ids:
        raise ValueError(f"duplicate source_query_id: {annotation.source_query_id}")
    seen_ids.add(annotation.source_query_id)

    if not annotation.semantic_group_id.strip():
        raise ValueError(
            f"missing semantic_group_id for {annotation.source_query_id}"
        )

    prior_group_query = seen_groups.get(annotation.semantic_group_id)
    key = normalize_query_key(annotation.query)
    if prior_group_query is None:
        seen_groups[annotation.semantic_group_id] = key
    # Groups may contain paraphrases later; do not force identical query text.

    prior_id = seen_query_keys.get(key)
    if prior_id is not None and prior_id != annotation.source_query_id:
        # Same normalized query with a different source id / conflicting annotation.
        raise ValueError(
            "duplicate normalized query assigned to multiple source_query_id "
            f"values: {prior_id!r} and {annotation.source_query_id!r}"
        )
    seen_query_keys[key] = annotation.source_query_id

    n_ops = len(annotation.operations)
    for index, operation in enumerate(annotation.operations):
        if operation.char_end > len(annotation.query):
            raise ValueError(
                f"{annotation.source_query_id}: operation span out of bounds "
                f"at {index}"
            )
        if operation.char_end <= operation.char_start:
            raise ValueError(
                f"{annotation.source_query_id}: empty operation span at {index}"
            )

    for index, anchor in enumerate(annotation.anchors):
        if anchor.char_end > len(annotation.query):
            raise ValueError(
                f"{annotation.source_query_id}: anchor span out of bounds at {index}"
            )
        if not (0 <= anchor.owner_operation_index < n_ops):
            raise ValueError(
                f"{annotation.source_query_id}: invalid owner index at anchor {index}"
            )

    for index, dependency in enumerate(annotation.dependencies):
        if not (
            0 <= dependency.source_operation_index < n_ops
            and 0 <= dependency.target_operation_index < n_ops
        ):
            raise ValueError(
                f"{annotation.source_query_id}: dependency index out of range "
                f"at {index}"
            )
        if dependency.source_operation_index == dependency.target_operation_index:
            raise ValueError(
                f"{annotation.source_query_id}: dependency self-loop at {index}"
            )
        source_op = annotation.operations[
            dependency.source_operation_index
        ].operator_type
        target_op = annotation.operations[
            dependency.target_operation_index
        ].operator_type
        if not is_h7_pair_eligible(source_op, target_op):
            raise ValueError(
                f"{annotation.source_query_id}: ineligible H7 pair "
                f"{source_op.value} -> {target_op.value}"
            )

    # Loud GraphDecoder / PlannerExample failure.
    semantic_annotation_to_planner_example(annotation)


def _print_annotation_summary(
    annotations: tuple[PlannerSemanticAnnotation, ...],
) -> None:
    by_class = Counter(item.source_classification_label for item in annotations)
    by_bucket = Counter(item.planner_bucket.value for item in annotations)
    by_operator = Counter(
        operation.operator_type.value
        for item in annotations
        for operation in item.operations
    )
    by_implicit = Counter(
        anchor.implicit_resolution.value
        for item in annotations
        for anchor in item.anchors
    )
    anchor_counts = Counter(len(item.anchors) for item in annotations)
    op_counts = Counter(len(item.operations) for item in annotations)
    h7_positive = sum(1 for item in annotations if item.dependencies)

    print(f"annotations: {len(annotations)}")
    print("by source classification label:")
    for label, count in sorted(by_class.items()):
        print(f"  {label}: {count}")
    print("by planner bucket:")
    for label, count in sorted(by_bucket.items()):
        print(f"  {label}: {count}")
    print("by operator type:")
    for label, count in sorted(by_operator.items()):
        print(f"  {label}: {count}")
    print("by implicit resolution:")
    for label, count in sorted(by_implicit.items()):
        print(f"  {label}: {count}")
    print("by number of anchors:")
    for n, count in sorted(anchor_counts.items()):
        print(f"  {n}: {count}")
    print("by number of explicit operations:")
    for n, count in sorted(op_counts.items()):
        print(f"  {n}: {count}")
    print(f"H7-positive examples: {h7_positive}")


def _print_candidate_summary(path: Path) -> None:
    candidates = load_candidates_jsonl(path)
    by_class = Counter(item.source_classification_label for item in candidates)
    print(f"candidates: {len(candidates)}")
    for label in ("Personal", "Environmental", "Mixed"):
        print(f"  {label}: {by_class[label]}")
    ids = [item.source_query_id for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source_query_id in candidates")
    for item in candidates:
        if not item.semantic_group_id:
            raise ValueError(f"missing semantic_group_id on {item.source_query_id}")
        if item.planner_bucket is not None:
            raise ValueError(
                "candidate planner_bucket must remain null until annotation: "
                f"{item.source_query_id}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        help="optional Stage-A candidates JSONL to summarize/validate",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="optional semantic annotations JSONL to validate end-to-end",
    )
    args = parser.parse_args(argv)

    if args.candidates is None and args.annotations is None:
        parser.error("provide --candidates and/or --annotations")

    if args.candidates is not None:
        _print_candidate_summary(args.candidates)

    if args.annotations is not None:
        annotations = load_semantic_annotations_jsonl(args.annotations)
        seen_ids: set[str] = set()
        seen_groups: dict[str, str] = {}
        seen_query_keys: dict[str, str] = {}
        for annotation in annotations:
            try:
                _validate_annotation(
                    annotation,
                    seen_ids=seen_ids,
                    seen_groups=seen_groups,
                    seen_query_keys=seen_query_keys,
                )
            except (ValueError, PlannerDecodeError) as exc:
                print(f"FAIL {annotation.source_query_id}: {exc}", file=sys.stderr)
                return 1
        _print_annotation_summary(annotations)
        print("all annotations validated")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
