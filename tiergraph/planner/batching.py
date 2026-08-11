"""Batch collation for Phase-4 learned planner heads.

Gold ownership and dependency indices use the supervised/representable
operation index space from :mod:`tiergraph.planner.targets`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tiergraph.enums import OperatorType, QueryType
from tiergraph.planner.align import (
    BIO_IGNORE,
    TokenCharSpan,
)
from tiergraph.planner.annotations import ImplicitResolution, PlannerExample
from tiergraph.planner.encoder import EncoderBatch
from tiergraph.planner.operator_io import is_h7_pair_eligible
from tiergraph.planner.targets import PlannerTargets, build_planner_targets


ANSWER_OPERATORS: tuple[OperatorType, ...] = (
    OperatorType.RESOLVE_PERSONAL,
    OperatorType.RETRIEVE_PERSONAL,
    OperatorType.IDENTIFY_ENVIRONMENTAL,
    OperatorType.LOCATE_ENVIRONMENTAL,
    OperatorType.NAVIGATE_TO,
    OperatorType.DESCRIBE_ENVIRONMENT,
)
OPERATOR_TO_INDEX = {operator: index for index, operator in enumerate(ANSWER_OPERATORS)}
INDEX_TO_OPERATOR = {index: operator for operator, index in OPERATOR_TO_INDEX.items()}

QUERY_TYPES: tuple[QueryType, ...] = (
    QueryType.PERSONAL,
    QueryType.ENVIRONMENTAL,
    QueryType.MIXED,
)
QUERY_TYPE_TO_INDEX = {query_type: index for index, query_type in enumerate(QUERY_TYPES)}

IMPLICIT_TO_INDEX = {
    ImplicitResolution.NONE: 0,
    ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL: 1,
}
INDEX_TO_IMPLICIT = {
    index: value for value, index in IMPLICIT_TO_INDEX.items()
}


@dataclass(frozen=True, slots=True)
class GoldStructureBatch:
    """Padded gold supervision tensors for one encoder batch."""

    query_type_labels: torch.Tensor  # [B]
    op_bio_labels: torch.Tensor  # [B, T]
    anc_bio_labels: torch.Tensor  # [B, T]
    token_loss_mask: torch.Tensor  # [B, T] bool
    op_span_mask: torch.Tensor  # [B, O, T] bool
    op_valid: torch.Tensor  # [B, O] bool
    op_type_labels: torch.Tensor  # [B, O]
    anc_span_mask: torch.Tensor  # [B, A, T] bool
    anc_valid: torch.Tensor  # [B, A] bool
    impl_labels: torch.Tensor  # [B, A]
    own_labels: torch.Tensor  # [B, A]
    own_mask: torch.Tensor  # [B, A, O] bool
    dep_labels: torch.Tensor  # [B, O, O]
    dep_mask: torch.Tensor  # [B, O, O] bool
    example_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructureBatch:
    """Operation/anchor span masks used by H3/H5/H6/H7 (gold or predicted)."""

    op_span_mask: torch.Tensor  # [B, O, T]
    op_valid: torch.Tensor  # [B, O]
    anc_span_mask: torch.Tensor  # [B, A, T]
    anc_valid: torch.Tensor  # [B, A]
    # Optional operator types for H7 eligibility masking (train: gold; infer: pred).
    op_type_indices: torch.Tensor | None = None  # [B, O] long; -1 pad


def content_token_loss_mask(tokens: Sequence[TokenCharSpan]) -> list[bool]:
    """True on retained content tokens only."""
    return [token.is_content for token in tokens]


def span_token_mask(
    tokens: Sequence[TokenCharSpan],
    token_indices: Sequence[int],
) -> list[bool]:
    mask = [False] * len(tokens)
    for index in token_indices:
        if not (0 <= index < len(tokens)):
            raise ValueError(f"span token index out of range: {index}")
        mask[index] = True
    return mask


def collate_gold_structure_batch(
    *,
    features: EncoderBatch,
    targets: Sequence[PlannerTargets],
    token_views: Sequence[Sequence[TokenCharSpan]],
) -> GoldStructureBatch:
    """Collate per-example planner targets into padded training tensors."""
    if len(targets) != features.batch_size:
        raise ValueError("targets batch size must match EncoderBatch")
    if len(token_views) != features.batch_size:
        raise ValueError("token_views batch size must match EncoderBatch")

    batch_size = features.batch_size
    token_length = features.input_ids.shape[1]
    max_ops = max((len(item.supervised_operations) for item in targets), default=0)
    max_anc = max((len(item.supervised_anchors) for item in targets), default=0)

    query_type_labels = torch.zeros(batch_size, dtype=torch.long)
    op_bio_labels = torch.full(
        (batch_size, token_length),
        BIO_IGNORE,
        dtype=torch.long,
    )
    anc_bio_labels = torch.full(
        (batch_size, token_length),
        BIO_IGNORE,
        dtype=torch.long,
    )
    token_loss_mask = torch.zeros(batch_size, token_length, dtype=torch.bool)
    op_span_mask = torch.zeros(batch_size, max_ops, token_length, dtype=torch.bool)
    op_valid = torch.zeros(batch_size, max_ops, dtype=torch.bool)
    op_type_labels = torch.zeros(batch_size, max_ops, dtype=torch.long)
    anc_span_mask = torch.zeros(batch_size, max_anc, token_length, dtype=torch.bool)
    anc_valid = torch.zeros(batch_size, max_anc, dtype=torch.bool)
    impl_labels = torch.zeros(batch_size, max_anc, dtype=torch.long)
    own_labels = torch.zeros(batch_size, max_anc, dtype=torch.long)
    own_mask = torch.zeros(batch_size, max_anc, max_ops, dtype=torch.bool)
    dep_labels = torch.zeros(batch_size, max_ops, max_ops, dtype=torch.float32)
    dep_mask = torch.zeros(batch_size, max_ops, max_ops, dtype=torch.bool)

    device = features.token_embeddings.device
    for batch_index, (target, tokens) in enumerate(zip(targets, token_views, strict=True)):
        if len(tokens) != token_length:
            raise ValueError(
                f"token view length {len(tokens)} != encoder token length {token_length}"
            )
        query_type_labels[batch_index] = QUERY_TYPE_TO_INDEX[target.query_type]

        loss_mask = content_token_loss_mask(tokens)
        token_loss_mask[batch_index] = torch.tensor(loss_mask, dtype=torch.bool)
        op_labels = list(target.operation_bio.labels)
        anc_labels = list(target.anchor_bio.labels)
        if len(op_labels) != token_length or len(anc_labels) != token_length:
            raise ValueError("BIO label length must match encoder token length")
        op_bio_labels[batch_index] = torch.tensor(op_labels, dtype=torch.long)
        anc_bio_labels[batch_index] = torch.tensor(anc_labels, dtype=torch.long)

        for op_index, operation in enumerate(target.supervised_operations):
            op_valid[batch_index, op_index] = True
            op_type_labels[batch_index, op_index] = OPERATOR_TO_INDEX[operation.operator]
            span_mask = span_token_mask(tokens, operation.alignment.token_indices)
            op_span_mask[batch_index, op_index] = torch.tensor(span_mask, dtype=torch.bool)

        for anc_index, anchor in enumerate(target.supervised_anchors):
            anc_valid[batch_index, anc_index] = True
            impl_labels[batch_index, anc_index] = IMPLICIT_TO_INDEX[
                anchor.implicit_resolution
            ]
            if anchor.owner_index_supervised is None:
                raise ValueError(
                    "supervised anchor missing owner_index_supervised"
                )
            own_labels[batch_index, anc_index] = anchor.owner_index_supervised
            span_mask = span_token_mask(tokens, anchor.alignment.token_indices)
            anc_span_mask[batch_index, anc_index] = torch.tensor(
                span_mask,
                dtype=torch.bool,
            )
            for op_index in range(len(target.supervised_operations)):
                own_mask[batch_index, anc_index, op_index] = True

        for pair in target.h7_pairs:
            if pair.masked or pair.label is None:
                continue
            dep_mask[batch_index, pair.source_index, pair.target_index] = True
            dep_labels[batch_index, pair.source_index, pair.target_index] = float(
                pair.label
            )

    return GoldStructureBatch(
        query_type_labels=query_type_labels.to(device),
        op_bio_labels=op_bio_labels.to(device),
        anc_bio_labels=anc_bio_labels.to(device),
        token_loss_mask=token_loss_mask.to(device),
        op_span_mask=op_span_mask.to(device),
        op_valid=op_valid.to(device),
        op_type_labels=op_type_labels.to(device),
        anc_span_mask=anc_span_mask.to(device),
        anc_valid=anc_valid.to(device),
        impl_labels=impl_labels.to(device),
        own_labels=own_labels.to(device),
        own_mask=own_mask.to(device),
        dep_labels=dep_labels.to(device),
        dep_mask=dep_mask.to(device),
        example_ids=tuple(item.example_id for item in targets),
    )


def gold_structure_as_head_structure(gold: GoldStructureBatch) -> StructureBatch:
    """Expose gold span masks for training head routing."""
    return StructureBatch(
        op_span_mask=gold.op_span_mask,
        op_valid=gold.op_valid,
        anc_span_mask=gold.anc_span_mask,
        anc_valid=gold.anc_valid,
        op_type_indices=gold.op_type_labels.masked_fill(~gold.op_valid, -1),
    )


def build_gold_batch_from_examples(
    examples: Sequence[PlannerExample],
    features: EncoderBatch,
    token_views: Sequence[Sequence[TokenCharSpan]],
) -> tuple[tuple[PlannerTargets, ...], GoldStructureBatch]:
    """Build per-example targets and a collated gold batch."""
    targets = tuple(
        build_planner_targets(example, tokens)
        for example, tokens in zip(examples, token_views, strict=True)
    )
    return targets, collate_gold_structure_batch(
        features=features,
        targets=targets,
        token_views=token_views,
    )


def dependency_eligibility_mask(
    *,
    op_valid: torch.Tensor,
    op_type_indices: torch.Tensor,
) -> torch.Tensor:
    """Build ``[B, O, O]`` H7 eligibility mask from operator type indices."""
    batch_size, max_ops = op_valid.shape
    mask = torch.zeros(
        batch_size,
        max_ops,
        max_ops,
        dtype=torch.bool,
        device=op_valid.device,
    )
    for batch_index in range(batch_size):
        for source in range(max_ops):
            if not bool(op_valid[batch_index, source]):
                continue
            source_type = INDEX_TO_OPERATOR.get(
                int(op_type_indices[batch_index, source].item())
            )
            if source_type is None:
                continue
            for target in range(max_ops):
                if source == target or not bool(op_valid[batch_index, target]):
                    continue
                target_type = INDEX_TO_OPERATOR.get(
                    int(op_type_indices[batch_index, target].item())
                )
                if target_type is None:
                    continue
                if is_h7_pair_eligible(source_type, target_type):
                    mask[batch_index, source, target] = True
    return mask


__all__ = [
    "ANSWER_OPERATORS",
    "GoldStructureBatch",
    "IMPLICIT_TO_INDEX",
    "INDEX_TO_IMPLICIT",
    "INDEX_TO_OPERATOR",
    "OPERATOR_TO_INDEX",
    "QUERY_TYPES",
    "QUERY_TYPE_TO_INDEX",
    "StructureBatch",
    "build_gold_batch_from_examples",
    "collate_gold_structure_batch",
    "content_token_loss_mask",
    "dependency_eligibility_mask",
    "gold_structure_as_head_structure",
    "span_token_mask",
]
