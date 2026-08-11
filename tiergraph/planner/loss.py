"""Equal-weight multi-task losses for the Phase-4 planner heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tiergraph.planner.align import BIO_IGNORE
from tiergraph.planner.batching import GoldStructureBatch
from tiergraph.planner.model import PlannerHeadOutputs


@dataclass(frozen=True, slots=True)
class PlannerLossBreakdown:
    """Per-head losses with fixed V1 weights of 1.0."""

    total: torch.Tensor
    h1: torch.Tensor
    h2: torch.Tensor
    h3: torch.Tensor
    h4: torch.Tensor
    h5: torch.Tensor
    h6: torch.Tensor
    h7: torch.Tensor


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros(())


def _masked_token_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_loss_mask: torch.Tensor,
) -> torch.Tensor:
    """CE over content tokens; labels use BIO_IGNORE on non-supervised positions."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    # Combine ignore index with content mask.
    safe_labels = flat_labels.clone()
    safe_labels = safe_labels.masked_fill(~token_loss_mask.reshape(-1), BIO_IGNORE)
    if not bool((safe_labels != BIO_IGNORE).any()):
        return _zero_like_loss(logits)
    return F.cross_entropy(flat_logits, safe_labels, ignore_index=BIO_IGNORE)


def _masked_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if logits.numel() == 0 or not bool(valid.any()):
        return _zero_like_loss(logits if logits.numel() else labels)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    flat_valid = valid.reshape(-1)
    if not bool(flat_valid.any()):
        return _zero_like_loss(logits)
    return F.cross_entropy(flat_logits[flat_valid], flat_labels[flat_valid])


def _masked_ownership_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    own_mask: torch.Tensor,
    anc_valid: torch.Tensor,
) -> torch.Tensor:
    """Pointer CE; rows without any valid operation are skipped."""
    if logits.numel() == 0 or logits.shape[-1] == 0:
        return _zero_like_loss(logits if logits.numel() else labels)
    losses: list[torch.Tensor] = []
    batch_size, max_anc, _max_ops = logits.shape
    for batch_index in range(batch_size):
        for anc_index in range(max_anc):
            if not bool(anc_valid[batch_index, anc_index]):
                continue
            row_mask = own_mask[batch_index, anc_index]
            if not bool(row_mask.any()):
                continue
            row_logits = logits[batch_index, anc_index].masked_fill(
                ~row_mask,
                float("-inf"),
            )
            target = labels[batch_index, anc_index]
            if not bool(row_mask[target]):
                continue
            losses.append(F.cross_entropy(row_logits.unsqueeze(0), target.unsqueeze(0)))
    if not losses:
        return _zero_like_loss(logits)
    return torch.stack(losses).mean()


def _masked_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if logits.numel() == 0 or not bool(mask.any()):
        return _zero_like_loss(logits if logits.numel() else labels)
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def planner_loss(
    outputs: PlannerHeadOutputs,
    gold: GoldStructureBatch,
) -> PlannerLossBreakdown:
    """Equal-weight multi-task loss. Empty heads contribute scalar 0, never NaN."""
    h1 = F.cross_entropy(outputs.query_type_logits, gold.query_type_labels)
    h2 = _masked_token_ce(
        outputs.op_bio_logits,
        gold.op_bio_labels,
        gold.token_loss_mask,
    )
    h4 = _masked_token_ce(
        outputs.anc_bio_logits,
        gold.anc_bio_labels,
        gold.token_loss_mask,
    )
    h3 = _masked_ce(outputs.op_type_logits, gold.op_type_labels, gold.op_valid)
    h5 = _masked_ce(outputs.impl_logits, gold.impl_labels, gold.anc_valid)
    h6 = _masked_ownership_ce(
        outputs.own_logits,
        gold.own_labels,
        gold.own_mask,
        gold.anc_valid,
    )
    # Use gold dep_mask (already excludes ineligible / mandatory / pad).
    h7 = _masked_bce(outputs.dep_logits, gold.dep_labels, gold.dep_mask)

    total = h1 + h2 + h3 + h4 + h5 + h6 + h7
    return PlannerLossBreakdown(
        total=total,
        h1=h1,
        h2=h2,
        h3=h3,
        h4=h4,
        h5=h5,
        h6=h6,
        h7=h7,
    )


__all__ = [
    "PlannerLossBreakdown",
    "planner_loss",
]
