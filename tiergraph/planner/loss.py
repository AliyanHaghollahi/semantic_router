"""Equal-weight multi-task losses for the Phase-4 planner heads."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tiergraph.planner.align import BIO_B, BIO_I, BIO_IGNORE
from tiergraph.planner.batching import GoldStructureBatch
from tiergraph.planner.model import PlannerHeadOutputs

HEAD_KEYS: tuple[str, ...] = ("h1", "h2", "h3", "h4", "h5", "h6", "h7")
# Explicit O/B/I order for optional H2/H4 class weights (matches BIO_O/B/I indices).
BIO_CLASS_WEIGHT_LABELS: tuple[str, str, str] = ("O", "B", "I")


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


def validate_bio_class_weights(
    weights: Sequence[float] | None,
    *,
    name: str = "bio_class_weights",
) -> tuple[float, float, float] | None:
    """Validate optional length-3 positive O/B/I weights; ``None`` means unweighted."""
    if weights is None:
        return None
    if len(weights) != 3:
        raise ValueError(
            f"{name} must be length 3 (O, B, I), got length {len(weights)}"
        )
    validated = (float(weights[0]), float(weights[1]), float(weights[2]))
    if any(value <= 0.0 for value in validated):
        raise ValueError(
            f"{name} values must be strictly positive (O, B, I), got {validated!r}"
        )
    if any(not (value == value) for value in validated):  # NaN check
        raise ValueError(f"{name} values must be finite, got {validated!r}")
    return validated


def _masked_token_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_loss_mask: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """CE over content tokens; labels use BIO_IGNORE on non-supervised positions.

    ``weight=None`` preserves the historical unweighted CE call.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    # Combine ignore index with content mask.
    safe_labels = flat_labels.clone()
    safe_labels = safe_labels.masked_fill(~token_loss_mask.reshape(-1), BIO_IGNORE)
    if not bool((safe_labels != BIO_IGNORE).any()):
        return _zero_like_loss(logits)
    if weight is None:
        return F.cross_entropy(flat_logits, safe_labels, ignore_index=BIO_IGNORE)
    return F.cross_entropy(
        flat_logits,
        safe_labels,
        weight=weight,
        ignore_index=BIO_IGNORE,
    )


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


def _weight_tensor(
    weights: Sequence[float] | None,
    *,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor | None:
    validated = validate_bio_class_weights(weights, name=name)
    if validated is None:
        return None
    return torch.tensor(
        validated,
        dtype=reference.dtype,
        device=reference.device,
    )


def _validate_boundary_lambda(value: float, *, name: str) -> float:
    """Boundary-auxiliary scales must be finite and non-negative."""
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return numeric


def _next_content_index(token_loss_mask: torch.Tensor) -> torch.Tensor:
    """Index of the next content token, or -1 when this is not content or is last.

    "Next" walks true positions in ``token_loss_mask``, not tensor index ``t+1``.
    """
    batch_size, length = token_loss_mask.shape
    next_index = torch.full(
        (batch_size, length),
        -1,
        dtype=torch.long,
        device=token_loss_mask.device,
    )
    for batch_index in range(batch_size):
        positions = token_loss_mask[batch_index].nonzero(as_tuple=False).flatten()
        if positions.numel() < 2:
            continue
        next_index[batch_index, positions[:-1]] = positions[1:]
    return next_index


def h4_start_logits(anc_bio_logits: torch.Tensor) -> torch.Tensor:
    """Binary start logit ``B - logsumexp(O, I)`` from 3-way BIO logits."""
    outside = torch.logsumexp(anc_bio_logits[..., [0, 2]], dim=-1)
    return anc_bio_logits[..., 1] - outside


def h4_end_log_terms(
    anc_bio_logits: torch.Tensor,
    token_loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact ``(log p_end, log(1 - p_end))`` of the parameter-free end surrogate.

    Non-final content tokens use
    ``p_end = p_inside * p_stop`` and
    ``1 - p_end = p_outside + p_inside * p_continue``.
    The final content token uses ``log p_end = log p_inside`` and
    ``log(1 - p_end) = log p_outside``.
    """
    logits = anc_bio_logits
    log_all = torch.logsumexp(logits, dim=-1)
    log_inside = torch.logsumexp(logits[..., 1:], dim=-1) - log_all
    log_outside = logits[..., 0] - log_all
    next_index = _next_content_index(token_loss_mask)
    has_next = next_index >= 0
    next_logits = logits.gather(
        1,
        next_index.clamp(min=0).unsqueeze(-1).expand(-1, -1, 3),
    )
    log_next_all = torch.logsumexp(next_logits, dim=-1)
    log_stop = torch.logsumexp(next_logits[..., [0, 1]], dim=-1) - log_next_all
    log_continue = next_logits[..., 2] - log_next_all
    log_p_end = torch.where(has_next, log_inside + log_stop, log_inside)
    log_p_not_end = torch.where(
        has_next,
        torch.logaddexp(log_outside, log_inside + log_continue),
        log_outside,
    )
    return log_p_end, log_p_not_end


def h4_end_log_probability(
    anc_bio_logits: torch.Tensor,
    token_loss_mask: torch.Tensor,
) -> torch.Tensor:
    """``log p_end`` on every position.

    Non-final content tokens use ``log p_inside + log p_stop``. The last
    content token uses ``log p_inside``. Non-content positions are unused.
    """
    log_p_end, _log_p_not_end = h4_end_log_terms(anc_bio_logits, token_loss_mask)
    return log_p_end


def _mean_over_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return _zero_like_loss(values)
    return values[mask].mean()


def h4_boundary_targets(
    anc_bio_labels: torch.Tensor,
    token_loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gold start and end masks on content tokens.

    Start: gold BIO is ``B``.
    End: gold BIO is ``B`` or ``I``, and the next content token is absent or
    not ``I``. Non-content positions are false.
    """
    next_index = _next_content_index(token_loss_mask)
    has_next = next_index >= 0
    next_labels = anc_bio_labels.gather(1, next_index.clamp(min=0))
    next_is_i = has_next & (next_labels == BIO_I)
    is_inside = (anc_bio_labels == BIO_B) | (anc_bio_labels == BIO_I)
    start = (anc_bio_labels == BIO_B) & token_loss_mask
    end = is_inside & token_loss_mask & ~next_is_i
    return start, end


def h4_start_loss(
    anc_bio_logits: torch.Tensor,
    anc_bio_labels: torch.Tensor,
    token_loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked BCE of the B-versus-rest start score. Content tokens only."""
    start_target, _end_target = h4_boundary_targets(anc_bio_labels, token_loss_mask)
    return _masked_bce(
        h4_start_logits(anc_bio_logits),
        start_target.to(anc_bio_logits.dtype),
        token_loss_mask,
    )


def h4_end_loss(
    anc_bio_logits: torch.Tensor,
    anc_bio_labels: torch.Tensor,
    token_loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Binary NLL of the parameter-free end surrogate, in exact log space."""
    _start_target, end_target = h4_boundary_targets(anc_bio_labels, token_loss_mask)
    log_p_end, log_p_not_end = h4_end_log_terms(anc_bio_logits, token_loss_mask)
    token_loss = torch.where(
        end_target,
        -log_p_end,
        -log_p_not_end,
    )
    return _mean_over_mask(token_loss, token_loss_mask)


def planner_loss(
    outputs: PlannerHeadOutputs,
    gold: GoldStructureBatch,
    *,
    active_heads: frozenset[str] | None = None,
    h2_bio_class_weights: Sequence[float] | None = None,
    h4_bio_class_weights: Sequence[float] | None = None,
    h4_boundary_lambda_start: float = 0.0,
    h4_boundary_lambda_end: float = 0.0,
) -> PlannerLossBreakdown:
    """Equal-weight multi-task loss. Empty heads contribute scalar 0, never NaN.

    Optional ``h2_bio_class_weights`` / ``h4_bio_class_weights`` are length-3
    ``(w_O, w_B, w_I)`` for class-weighted BIO CE. ``None`` keeps the
    historical unweighted CE path.

    H4 boundary auxiliaries are added only when ``h4`` is active and the
    corresponding lambda is strictly positive. Lambda 0 skips those graphs
    entirely, so the H4 path matches the previous BIO cross-entropy.
    """
    active = active_heads or frozenset(HEAD_KEYS)
    unknown = active - frozenset(HEAD_KEYS)
    if unknown:
        raise ValueError(f"unknown active_heads: {sorted(unknown)}")
    lambda_start = _validate_boundary_lambda(
        h4_boundary_lambda_start,
        name="h4_boundary_lambda_start",
    )
    lambda_end = _validate_boundary_lambda(
        h4_boundary_lambda_end,
        name="h4_boundary_lambda_end",
    )

    h1 = F.cross_entropy(outputs.query_type_logits, gold.query_type_labels)
    h2 = _masked_token_ce(
        outputs.op_bio_logits,
        gold.op_bio_labels,
        gold.token_loss_mask,
        weight=_weight_tensor(
            h2_bio_class_weights,
            reference=outputs.op_bio_logits,
            name="h2_bio_class_weights",
        ),
    )
    h4 = _masked_token_ce(
        outputs.anc_bio_logits,
        gold.anc_bio_labels,
        gold.token_loss_mask,
        weight=_weight_tensor(
            h4_bio_class_weights,
            reference=outputs.anc_bio_logits,
            name="h4_bio_class_weights",
        ),
    )
    if "h4" in active and lambda_start > 0.0:
        h4 = h4 + lambda_start * h4_start_loss(
            outputs.anc_bio_logits,
            gold.anc_bio_labels,
            gold.token_loss_mask,
        )
    if "h4" in active and lambda_end > 0.0:
        h4 = h4 + lambda_end * h4_end_loss(
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

    per_head = {
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
        "h5": h5,
        "h6": h6,
        "h7": h7,
    }
    total = sum(per_head[key] for key in HEAD_KEYS if key in active)
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
    "BIO_CLASS_WEIGHT_LABELS",
    "HEAD_KEYS",
    "PlannerLossBreakdown",
    "h4_boundary_targets",
    "h4_end_log_probability",
    "h4_end_log_terms",
    "h4_end_loss",
    "h4_start_logits",
    "h4_start_loss",
    "planner_loss",
    "validate_bio_class_weights",
]
