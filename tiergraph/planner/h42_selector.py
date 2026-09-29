"""H4.2 per-operation multi-label anchor candidate selector.

Candidates are contiguous word spans of length 1 through 8, generated from the
query text alone. Gold spans are used only to label training targets. Scoring
is one bilinear product plus a NONE vector, applied to token embeddings from
an already completed frozen MiniLM forward. This module is not registered on
``PlannerModel``, so the H4 BIO head and the frozen H4.1 checkpoint stay
unchanged.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from tiergraph.planner.align import TokenCharSpan, align_char_span
from tiergraph.planner.model import masked_mean_pool

H42_MAX_SPAN_TOKENS = 8
_WORD = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


@dataclass(frozen=True, slots=True)
class WordSpan:
    """Half-open character span of one candidate in the original query."""

    start: int
    end: int

    def text(self, query: str) -> str:
        return query[self.start : self.end]


def enumerate_word_spans(
    query: str,
    *,
    max_tokens: int = H42_MAX_SPAN_TOKENS,
) -> tuple[WordSpan, ...]:
    """Every deduplicated contiguous word span of length 1..``max_tokens``.

    The query text is the only input. Gold anchors are not consulted.
    """
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    words = [(match.start(), match.end()) for match in _WORD.finditer(query)]
    seen: set[tuple[int, int]] = set()
    spans: list[WordSpan] = []
    count = len(words)
    for start_index in range(count):
        last = min(count, start_index + max_tokens)
        for end_index in range(start_index + 1, last + 1):
            key = (words[start_index][0], words[end_index - 1][1])
            if key in seen:
                continue
            seen.add(key)
            spans.append(WordSpan(key[0], key[1]))
    return tuple(spans)


def candidate_targets(
    spans: Sequence[WordSpan],
    n_operations: int,
    gold_anchors: Sequence[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary multi-label targets for one example.

    ``gold_anchors`` entries are ``(char_start, char_end, owner_index)``.
    A candidate is positive only for the operation that owns that exact span.
    NONE is positive exactly when that operation owns zero gold anchors.
    """
    if n_operations < 0:
        raise ValueError("n_operations must be non-negative")
    by_span = {(span.start, span.end): index for index, span in enumerate(spans)}
    candidates = torch.zeros(len(spans), n_operations, dtype=torch.float32)
    owned = torch.zeros(n_operations, dtype=torch.float32)
    for start, end, owner in gold_anchors:
        if not (0 <= owner < n_operations):
            raise ValueError(f"owner index out of range: {owner}")
        index = by_span.get((start, end))
        if index is None:
            raise ValueError(
                f"gold span {(start, end)} is not in the candidate set"
            )
        candidates[index, owner] = 1.0
        owned[owner] += 1.0
    none_targets = (owned == 0).to(dtype=torch.float32)
    return candidates, none_targets


def select_candidates(
    candidate_logits: torch.Tensor,
    none_logits: torch.Tensor,
) -> tuple[tuple[int, ...], ...]:
    """Keep every candidate that strictly beats NONE, independently per operation.

    ``candidate_logits`` is ``[C, O]`` and ``none_logits`` is ``[O]``.
    An operation may receive zero, one, or several spans. Exactly one span
    is never forced.
    """
    if candidate_logits.ndim != 2:
        raise ValueError("candidate_logits must have shape [C, O]")
    if none_logits.ndim != 1 or none_logits.shape[0] != candidate_logits.shape[1]:
        raise ValueError("none_logits must have shape [O]")
    chosen: list[tuple[int, ...]] = []
    for operation in range(candidate_logits.shape[1]):
        keep = candidate_logits[:, operation] > none_logits[operation]
        chosen.append(tuple(index for index, flag in enumerate(keep.tolist()) if flag))
    return tuple(chosen)


def candidate_token_mask(
    spans: Sequence[WordSpan],
    tokens: Sequence[TokenCharSpan],
) -> torch.Tensor:
    """``[C, T]`` bool mask. Unrepresentable spans are all False."""
    rows = []
    width = len(tokens)
    for span in spans:
        alignment = align_char_span(span.start, span.end, tokens)
        mask = [False] * width
        if alignment.representable:
            for index in alignment.token_indices:
                mask[index] = True
        rows.append(mask)
    if not rows:
        return torch.zeros(0, width, dtype=torch.bool)
    return torch.tensor(rows, dtype=torch.bool)


def operation_token_mask(
    operation_spans: Sequence[tuple[int, int]],
    tokens: Sequence[TokenCharSpan],
) -> torch.Tensor:
    """``[O, T]`` bool mask for already decoded operation character spans."""
    return candidate_token_mask(
        tuple(WordSpan(start, end) for start, end in operation_spans),
        tokens,
    )


class H42CandidateSelector(nn.Module):
    """Bilinear span-versus-operation ranker plus one learned NONE vector."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        self.hidden_size = hidden_size
        self.anchor_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.operation_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.none_vector = nn.Parameter(torch.empty(hidden_size))
        nn.init.xavier_uniform_(self.anchor_proj.weight)
        nn.init.xavier_uniform_(self.operation_proj.weight)
        nn.init.normal_(self.none_vector, std=0.02)
        self._score_scale = math.sqrt(float(hidden_size))

    def score(
        self,
        token_embeddings: torch.Tensor,
        operation_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score one example from embeddings that already exist.

        ``token_embeddings`` is ``[T, H]`` from a single MiniLM forward.
        ``operation_mask`` is ``[O, T]`` and ``candidate_mask`` is ``[C, T]``.
        Returns candidate logits ``[C, O]`` and NONE logits ``[O]``.
        Unrepresentable rows (all-False masks) receive the dtype minimum
        so they cannot beat a finite NONE score. The minimum stays finite so
        masked binary cross-entropy does not see infinity.
        """
        if token_embeddings.ndim != 2:
            raise ValueError("token_embeddings must have shape [T, H]")
        if token_embeddings.shape[1] != self.hidden_size:
            raise ValueError("token embedding width must match hidden_size")
        if operation_mask.ndim != 2 or candidate_mask.ndim != 2:
            raise ValueError("masks must be rank 2")
        if operation_mask.shape[1] != token_embeddings.shape[0]:
            raise ValueError("operation mask token length mismatch")
        if candidate_mask.shape[1] != token_embeddings.shape[0]:
            raise ValueError("candidate mask token length mismatch")

        embeddings = token_embeddings.unsqueeze(0)
        op_repr = masked_mean_pool(embeddings, operation_mask.unsqueeze(0)).squeeze(0)
        cand_repr = masked_mean_pool(embeddings, candidate_mask.unsqueeze(0)).squeeze(0)
        op_hidden = self.operation_proj(op_repr)
        cand_hidden = self.anchor_proj(cand_repr)
        candidate_logits = cand_hidden @ op_hidden.transpose(0, 1) / self._score_scale
        none_logits = op_hidden @ self.none_vector / self._score_scale
        lowest = torch.finfo(candidate_logits.dtype).min
        unrepresentable = candidate_mask.sum(dim=1) == 0
        if unrepresentable.any():
            candidate_logits = candidate_logits.masked_fill(
                unrepresentable.unsqueeze(1),
                lowest,
            )
        empty_operation = operation_mask.sum(dim=1) == 0
        if empty_operation.any():
            candidate_logits = candidate_logits.masked_fill(
                empty_operation.unsqueeze(0),
                lowest,
            )
            none_logits = none_logits.masked_fill(empty_operation, -lowest)
        return candidate_logits, none_logits


def h42_loss(
    candidate_logits: torch.Tensor,
    none_logits: torch.Tensor,
    candidate_targets: torch.Tensor,
    none_targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    operation_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean multi-label BCE over representable candidates and valid operations."""
    if candidate_logits.shape != candidate_targets.shape:
        raise ValueError("candidate logit and target shapes must match")
    if none_logits.shape != none_targets.shape:
        raise ValueError("NONE logit and target shapes must match")
    if operation_mask.shape != none_logits.shape:
        raise ValueError("operation mask must match NONE logits")
    pair_mask = candidate_mask.unsqueeze(1) & operation_mask.unsqueeze(0)
    candidate_loss = F.binary_cross_entropy_with_logits(
        candidate_logits,
        candidate_targets,
        reduction="none",
    )
    pair_weight = pair_mask.to(dtype=candidate_loss.dtype)
    pair_total = pair_weight.sum().clamp_min(1.0)
    none_loss = F.binary_cross_entropy_with_logits(
        none_logits,
        none_targets,
        reduction="none",
    )
    op_weight = operation_mask.to(dtype=none_loss.dtype)
    op_total = op_weight.sum().clamp_min(1.0)
    return (candidate_loss * pair_weight).sum() / pair_total + (
        none_loss * op_weight
    ).sum() / op_total
