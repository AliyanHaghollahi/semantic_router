"""Deterministic training targets from validated ``PlannerExample`` annotations.

Downstream heads (operator, implicit flag, ownership, H7) are supervised on
GOLD operation and anchor structures. Token BIO heads use aligned gold spans.
Truncated or otherwise unrepresentable spans are masked and reported rather
than clipped into alternate targets.

Index spaces
------------
* ``owner_index_full`` indexes into the full gold ``operation_spans`` tuple.
* ``owner_index_supervised`` / H7 ``source_index`` / ``target_index`` index
  into the filtered ``representable`` / supervised operation list used by
  H3/H5/H6/H7 training. Never treat these spaces as interchangeable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tiergraph.enums import OperatorType, QueryType
from tiergraph.graph import ExecutionGraph
from tiergraph.planner.align import (
    AlignmentStats,
    BioEncoding,
    SpanAlignment,
    TokenCharSpan,
    align_char_span,
    encode_bio_labels,
)
from tiergraph.planner.annotations import (
    ImplicitResolution,
    PlannerExample,
)
from tiergraph.planner.operator_io import is_h7_pair_eligible


@dataclass(frozen=True, slots=True)
class OperationTarget:
    """Gold explicit-operation structure for downstream head supervision."""

    node_id: str
    operator: OperatorType
    start: int
    end: int
    alignment: SpanAlignment
    representable: bool
    # Index into the full gold operation_spans tuple.
    index_full: int
    # Index into representable_ops when representable; else None.
    index_supervised: int | None


@dataclass(frozen=True, slots=True)
class AnchorTarget:
    """Gold slot-anchor structure for downstream head supervision."""

    anchor_id: str
    start: int
    end: int
    text: str
    normalized_name: str
    owner_node_id: str
    # Index into the full gold operation_spans tuple.
    owner_index_full: int
    # Index into representable_ops when both anchor and owner are supervised.
    owner_index_supervised: int | None
    implicit_resolution: ImplicitResolution
    implicit_node_id: str | None
    alignment: SpanAlignment
    representable: bool


@dataclass(frozen=True, slots=True)
class H7PairTarget:
    """One ordered explicit-operation pair for dependency supervision.

    ``source_index`` / ``target_index`` are always in the supervised
    (representable_ops) index space.
    """

    source_index: int
    target_index: int
    source_node_id: str
    target_node_id: str
    eligible: bool
    label: float | None
    masked: bool
    mask_reason: str | None


@dataclass(frozen=True, slots=True)
class PlannerTargets:
    """All deterministic supervision tensors/structures for one example."""

    example_id: str
    query: str
    query_type: QueryType
    operation_bio: BioEncoding
    anchor_bio: BioEncoding
    operations: tuple[OperationTarget, ...]
    anchors: tuple[AnchorTarget, ...]
    # Downstream labels over GOLD representable structures only.
    # Indices below refer to this supervised/representable operation list.
    supervised_operations: tuple[OperationTarget, ...]
    operator_labels: tuple[OperatorType, ...]
    operator_node_ids: tuple[str, ...]
    supervised_anchors: tuple[AnchorTarget, ...]
    implicit_labels: tuple[ImplicitResolution, ...]
    ownership_owner_indices: tuple[int, ...]
    ownership_anchor_ids: tuple[str, ...]
    h7_pairs: tuple[H7PairTarget, ...]
    alignment_stats: AlignmentStats
    n_masked_operations: int
    n_masked_anchors: int


def build_planner_targets(
    example: PlannerExample,
    tokens: Sequence[TokenCharSpan],
) -> PlannerTargets:
    """Convert a validated planner example into training targets and masks."""
    if not isinstance(example, PlannerExample):
        raise TypeError("example must be a PlannerExample")

    labels = example.planner_labels
    operations = labels.operation_spans
    anchors = labels.slot_anchors
    op_id_to_full_index = {
        span.node_id: index for index, span in enumerate(operations)
    }

    operation_targets: list[OperationTarget] = []
    for full_index, span in enumerate(operations):
        alignment = align_char_span(span.start, span.end, tokens)
        operation_targets.append(
            OperationTarget(
                node_id=span.node_id,
                operator=span.operator,
                start=span.start,
                end=span.end,
                alignment=alignment,
                representable=alignment.representable,
                index_full=full_index,
                index_supervised=None,
            )
        )

    supervised_operations: list[OperationTarget] = []
    for item in operation_targets:
        if not item.representable:
            continue
        supervised_index = len(supervised_operations)
        updated = OperationTarget(
            node_id=item.node_id,
            operator=item.operator,
            start=item.start,
            end=item.end,
            alignment=item.alignment,
            representable=True,
            index_full=item.index_full,
            index_supervised=supervised_index,
        )
        operation_targets[item.index_full] = updated
        supervised_operations.append(updated)

    representable_op_ids = {item.node_id for item in supervised_operations}
    supervised_op_index = {
        item.node_id: item.index_supervised
        for item in supervised_operations
        if item.index_supervised is not None
    }

    anchor_targets: list[AnchorTarget] = []
    for anchor in anchors:
        owner_index_full = op_id_to_full_index[anchor.owner_node_id]
        alignment = align_char_span(anchor.start, anchor.end, tokens)
        owner_index_supervised = None
        if (
            alignment.representable
            and anchor.owner_node_id in representable_op_ids
        ):
            owner_index_supervised = supervised_op_index[anchor.owner_node_id]
        anchor_targets.append(
            AnchorTarget(
                anchor_id=anchor.anchor_id,
                start=anchor.start,
                end=anchor.end,
                text=anchor.text,
                normalized_name=anchor.normalized_name,
                owner_node_id=anchor.owner_node_id,
                owner_index_full=owner_index_full,
                owner_index_supervised=owner_index_supervised,
                implicit_resolution=anchor.implicit_resolution,
                implicit_node_id=anchor.implicit_node_id,
                alignment=alignment,
                representable=alignment.representable,
            )
        )

    operation_bio = encode_bio_labels(
        tuple(item.alignment for item in operation_targets),
        tokens,
    )
    anchor_bio = encode_bio_labels(
        tuple(item.alignment for item in anchor_targets),
        tokens,
    )

    operator_labels = tuple(item.operator for item in supervised_operations)
    operator_node_ids = tuple(item.node_id for item in supervised_operations)

    supervised_anchors = tuple(
        item
        for item in anchor_targets
        if item.representable and item.owner_index_supervised is not None
    )
    implicit_labels = tuple(item.implicit_resolution for item in supervised_anchors)
    ownership_owner_indices = tuple(
        item.owner_index_supervised
        for item in supervised_anchors
        if item.owner_index_supervised is not None
    )
    ownership_anchor_ids = tuple(item.anchor_id for item in supervised_anchors)

    h7_pairs = _build_h7_pair_targets(
        graph=example.graph,
        operations=operation_targets,
        supervised_operations=supervised_operations,
        anchors=anchor_targets,
    )

    alignment_stats = operation_bio.stats.merge(
        AlignmentStats(
            n_spans=anchor_bio.stats.n_spans,
            n_representable=anchor_bio.stats.n_representable,
            n_fully_truncated=anchor_bio.stats.n_fully_truncated,
            n_partially_truncated=anchor_bio.stats.n_partially_truncated,
            n_empty=anchor_bio.stats.n_empty,
        )
    )

    return PlannerTargets(
        example_id=example.example_id,
        query=example.query,
        query_type=labels.query_type,
        operation_bio=operation_bio,
        anchor_bio=anchor_bio,
        operations=tuple(operation_targets),
        anchors=tuple(anchor_targets),
        supervised_operations=tuple(supervised_operations),
        operator_labels=operator_labels,
        operator_node_ids=operator_node_ids,
        supervised_anchors=supervised_anchors,
        implicit_labels=implicit_labels,
        ownership_owner_indices=ownership_owner_indices,
        ownership_anchor_ids=ownership_anchor_ids,
        h7_pairs=tuple(h7_pairs),
        alignment_stats=alignment_stats,
        n_masked_operations=sum(
            1 for item in operation_targets if not item.representable
        ),
        n_masked_anchors=sum(1 for item in anchor_targets if not item.representable),
    )


def _build_h7_pair_targets(
    *,
    graph: ExecutionGraph,
    operations: Sequence[OperationTarget],
    supervised_operations: Sequence[OperationTarget],
    anchors: Sequence[AnchorTarget],
) -> list[H7PairTarget]:
    explicit_ids = {item.node_id for item in operations}
    implicit_ids = {
        item.implicit_node_id
        for item in anchors
        if item.implicit_resolution is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
        and item.implicit_node_id is not None
    }
    mandatory_pairs = {
        (item.implicit_node_id, item.owner_node_id)
        for item in anchors
        if item.implicit_resolution is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
        and item.implicit_node_id is not None
    }

    positive_explicit_pairs: set[tuple[str, str]] = set()
    for edge in graph.edges:
        source = edge.source_node_id
        target = edge.target_node_id
        if (source, target) in mandatory_pairs:
            continue
        if source in implicit_ids:
            continue
        if source not in explicit_ids or target not in explicit_ids:
            continue
        positive_explicit_pairs.add((source, target))

    pairs: list[H7PairTarget] = []
    n = len(supervised_operations)
    for source_index in range(n):
        for target_index in range(n):
            if source_index == target_index:
                continue
            source = supervised_operations[source_index]
            target = supervised_operations[target_index]
            eligible = is_h7_pair_eligible(source.operator, target.operator)
            if not eligible:
                pairs.append(
                    H7PairTarget(
                        source_index=source_index,
                        target_index=target_index,
                        source_node_id=source.node_id,
                        target_node_id=target.node_id,
                        eligible=False,
                        label=None,
                        masked=True,
                        mask_reason="structurally_ineligible",
                    )
                )
                continue
            if (source.node_id, target.node_id) in mandatory_pairs:
                pairs.append(
                    H7PairTarget(
                        source_index=source_index,
                        target_index=target_index,
                        source_node_id=source.node_id,
                        target_node_id=target.node_id,
                        eligible=True,
                        label=None,
                        masked=True,
                        mask_reason="mandatory_implicit_owner",
                    )
                )
                continue
            label = (
                1.0
                if (source.node_id, target.node_id) in positive_explicit_pairs
                else 0.0
            )
            pairs.append(
                H7PairTarget(
                    source_index=source_index,
                    target_index=target_index,
                    source_node_id=source.node_id,
                    target_node_id=target.node_id,
                    eligible=True,
                    label=label,
                    masked=False,
                    mask_reason=None,
                )
            )

    return pairs


__all__ = [
    "AnchorTarget",
    "H7PairTarget",
    "OperationTarget",
    "PlannerTargets",
    "build_planner_targets",
]
