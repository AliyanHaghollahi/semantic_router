"""Shared-encoder multi-head learned planner (H1–H7).

Uses one frozen :class:`MiniLMFeatureEncoder` forward per batch. Heads are
independent linear / bilinear modules; do not overclaim cross-task
representation learning.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from tiergraph.enums import QueryType
from tiergraph.planner.align import BIO_B, BIO_I, BIO_O, TokenCharSpan
from tiergraph.planner.annotations import ImplicitResolution
from tiergraph.planner.batching import (
    ANSWER_OPERATORS,
    INDEX_TO_IMPLICIT,
    INDEX_TO_OPERATOR,
    QUERY_TYPES,
    GoldStructureBatch,
    StructureBatch,
    dependency_eligibility_mask,
    gold_structure_as_head_structure,
)
from tiergraph.planner.decode import (
    PlannerPredictions,
    PredictedAnchor,
    PredictedOperation,
)
from tiergraph.planner.encoder import EncoderBatch, MiniLMFeatureEncoder


@dataclass(frozen=True, slots=True)
class PlannerHeadOutputs:
    """Raw head logits and masks for one forward pass."""

    query_type_logits: torch.Tensor  # [B, 3]
    op_bio_logits: torch.Tensor  # [B, T, 3]
    anc_bio_logits: torch.Tensor  # [B, T, 3]
    op_type_logits: torch.Tensor  # [B, O, 6]
    impl_logits: torch.Tensor  # [B, A, 2]
    own_logits: torch.Tensor  # [B, A, O]
    dep_logits: torch.Tensor  # [B, O, O]
    op_valid: torch.Tensor
    anc_valid: torch.Tensor
    own_mask: torch.Tensor
    dep_mask: torch.Tensor
    token_loss_mask: torch.Tensor
    op_repr: torch.Tensor
    anc_repr: torch.Tensor


@dataclass(frozen=True, slots=True)
class PlannerPredictionsBatch:
    """Per-example predicted structures for GraphDecoder (not decoded graphs)."""

    items: tuple[PlannerPredictions, ...]
    token_views: tuple[tuple[TokenCharSpan, ...], ...]
    head_outputs: PlannerHeadOutputs


def masked_mean_pool(
    token_embeddings: torch.Tensor,
    span_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked mean pool.

    ``token_embeddings``: ``[B, T, H]``
    ``span_mask``: ``[B, N, T]`` bool
    returns ``[B, N, H]``
    """
    if token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must have shape [B, T, H]")
    if span_mask.ndim != 3:
        raise ValueError("span_mask must have shape [B, N, T]")
    if span_mask.shape[0] != token_embeddings.shape[0]:
        raise ValueError("batch size mismatch for masked_mean_pool")
    if span_mask.shape[2] != token_embeddings.shape[1]:
        raise ValueError("token length mismatch for masked_mean_pool")

    mask = span_mask.to(dtype=token_embeddings.dtype).unsqueeze(-1)
    summed = (token_embeddings.unsqueeze(1) * mask).sum(dim=2)
    counts = mask.sum(dim=2).clamp_min(1.0)
    return summed / counts


def decode_bio_spans(
    labels: Sequence[int],
    tokens: Sequence[TokenCharSpan],
) -> tuple[tuple[int, int], ...]:
    """Deterministic BIO→char-span decode over content tokens.

    Convention
    ----------
    * Only content tokens participate.
    * ``B`` starts a new span (closes any open span).
    * ``I`` continues an open span; an ``I`` with no open span is treated as ``B``.
    * ``O`` closes an open span.
    * No semantic repair is applied.
    """
    if len(labels) != len(tokens):
        raise ValueError("BIO labels length must match token view length")

    spans: list[tuple[int, int]] = []
    open_start: int | None = None
    open_end: int | None = None

    def _close() -> None:
        nonlocal open_start, open_end
        if open_start is not None and open_end is not None and open_end > open_start:
            spans.append((open_start, open_end))
        open_start = None
        open_end = None

    for label, token in zip(labels, tokens, strict=True):
        if not token.is_content:
            continue
        assert token.char_start is not None and token.char_end is not None
        if label == BIO_O:
            _close()
            continue
        if label == BIO_B or (label == BIO_I and open_start is None):
            _close()
            open_start = token.char_start
            open_end = token.char_end
            continue
        if label == BIO_I:
            assert open_start is not None
            open_end = token.char_end
            continue
        # Unknown label: close without extending.
        _close()
    _close()
    return tuple(spans)


class PlannerModel(nn.Module):
    """Frozen-encoder multi-head planner."""

    def __init__(
        self,
        encoder: MiniLMFeatureEncoder,
        *,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, MiniLMFeatureEncoder):
            raise TypeError("encoder must be a MiniLMFeatureEncoder instance")
        self.encoder = encoder
        if hidden_size is None:
            # Prefer configured size after load; allow override for unit tests.
            try:
                hidden_size = encoder.hidden_size
            except RuntimeError:
                hidden_size = 384
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        self.hidden_size = hidden_size
        scale = math.sqrt(float(hidden_size))

        self.query_type_head = nn.Linear(hidden_size, len(QUERY_TYPES))
        self.op_bio_head = nn.Linear(hidden_size, 3)
        self.anc_bio_head = nn.Linear(hidden_size, 3)
        self.op_type_head = nn.Linear(hidden_size, len(ANSWER_OPERATORS))
        self.impl_head = nn.Linear(hidden_size, 2)
        self.anchor_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.operation_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dep_source_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dep_target_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self._score_scale = scale

        if self.encoder.is_loaded and self.encoder._model is not None:
            for parameter in self.encoder._model.parameters():
                parameter.requires_grad_(False)

    def encode(self, texts: str | Sequence[str]) -> EncoderBatch:
        """Exactly one MiniLM forward for the provided texts."""
        return self.encoder.encode(texts)

    def forward_heads(
        self,
        features: EncoderBatch,
        *,
        structure: StructureBatch,
        token_loss_mask: torch.Tensor | None = None,
    ) -> PlannerHeadOutputs:
        """Run H1–H7 on one encoder batch and a structure mask batch."""
        # Encoder may emit inference_mode tensors; clone so trainable heads can
        # backprop without attaching the frozen encoder to the graph.
        token_embeddings = features.token_embeddings.detach().clone()
        pooled = features.pooled_embeddings.detach().clone()
        batch_size, token_length, hidden = token_embeddings.shape
        if hidden != self.hidden_size:
            raise ValueError(
                f"feature hidden size {hidden} != model hidden_size {self.hidden_size}"
            )

        if token_loss_mask is None:
            attention = features.attention_mask
            if attention.dtype == torch.bool:
                token_loss_mask = attention
            else:
                token_loss_mask = attention != 0
            # Default content mask approximation: attended tokens. Callers should
            # pass an explicit content-token mask from TokenCharSpan views.
        if token_loss_mask.shape != (batch_size, token_length):
            raise ValueError("token_loss_mask shape must be [B, T]")

        op_span_mask = structure.op_span_mask
        anc_span_mask = structure.anc_span_mask
        op_valid = structure.op_valid
        anc_valid = structure.anc_valid
        if op_span_mask.shape[:2] != op_valid.shape:
            raise ValueError("op_span_mask / op_valid shape mismatch")
        if anc_span_mask.shape[:2] != anc_valid.shape:
            raise ValueError("anc_span_mask / anc_valid shape mismatch")
        if op_span_mask.shape[2] != token_length or anc_span_mask.shape[2] != token_length:
            raise ValueError("structure token length must match features")

        max_ops = op_valid.shape[1]
        max_anc = anc_valid.shape[1]

        query_type_logits = self.query_type_head(pooled)
        op_bio_logits = self.op_bio_head(token_embeddings)
        anc_bio_logits = self.anc_bio_head(token_embeddings)

        if max_ops == 0:
            op_repr = token_embeddings.new_zeros(batch_size, 0, hidden)
            op_type_logits = token_embeddings.new_zeros(
                batch_size,
                0,
                len(ANSWER_OPERATORS),
            )
            dep_logits = token_embeddings.new_zeros(batch_size, 0, 0)
            dep_mask = torch.zeros(batch_size, 0, 0, dtype=torch.bool, device=token_embeddings.device)
            own_logits = token_embeddings.new_zeros(batch_size, max_anc, 0)
            own_mask = torch.zeros(
                batch_size,
                max_anc,
                0,
                dtype=torch.bool,
                device=token_embeddings.device,
            )
        else:
            op_repr = masked_mean_pool(token_embeddings, op_span_mask)
            op_type_logits = self.op_type_head(op_repr)
            source = self.dep_source_proj(op_repr)
            target = self.dep_target_proj(op_repr)
            dep_logits = torch.matmul(source, target.transpose(-1, -2)) / self._score_scale
            if structure.op_type_indices is None:
                # Fall back to predicted types for eligibility when not provided.
                pred_types = op_type_logits.argmax(dim=-1)
                op_type_indices = pred_types.masked_fill(~op_valid, -1)
            else:
                op_type_indices = structure.op_type_indices
            dep_mask = dependency_eligibility_mask(
                op_valid=op_valid,
                op_type_indices=op_type_indices,
            )
            own_logits = token_embeddings.new_zeros(batch_size, max_anc, max_ops)
            own_mask = op_valid.unsqueeze(1).expand(batch_size, max_anc, max_ops).clone()
            own_mask = own_mask & anc_valid.unsqueeze(-1)

        if max_anc == 0:
            anc_repr = token_embeddings.new_zeros(batch_size, 0, hidden)
            impl_logits = token_embeddings.new_zeros(batch_size, 0, 2)
            if max_ops != 0:
                own_logits = token_embeddings.new_zeros(batch_size, 0, max_ops)
                own_mask = torch.zeros(
                    batch_size,
                    0,
                    max_ops,
                    dtype=torch.bool,
                    device=token_embeddings.device,
                )
        else:
            anc_repr = masked_mean_pool(token_embeddings, anc_span_mask)
            impl_logits = self.impl_head(anc_repr)
            if max_ops != 0:
                anchor_h = self.anchor_proj(anc_repr)
                operation_h = self.operation_proj(op_repr)
                own_logits = torch.matmul(
                    anchor_h,
                    operation_h.transpose(-1, -2),
                ) / self._score_scale
                own_mask = op_valid.unsqueeze(1).expand_as(own_logits).clone()
                own_mask = own_mask & anc_valid.unsqueeze(-1)

        return PlannerHeadOutputs(
            query_type_logits=query_type_logits,
            op_bio_logits=op_bio_logits,
            anc_bio_logits=anc_bio_logits,
            op_type_logits=op_type_logits,
            impl_logits=impl_logits,
            own_logits=own_logits,
            dep_logits=dep_logits,
            op_valid=op_valid,
            anc_valid=anc_valid,
            own_mask=own_mask,
            dep_mask=dep_mask,
            token_loss_mask=token_loss_mask,
            op_repr=op_repr,
            anc_repr=anc_repr,
        )

    def forward_train(
        self,
        features: EncoderBatch,
        gold: GoldStructureBatch,
    ) -> PlannerHeadOutputs:
        """Training forward: H3/H5/H6/H7 consume GOLD structure masks."""
        structure = gold_structure_as_head_structure(gold)
        return self.forward_heads(
            features,
            structure=structure,
            token_loss_mask=gold.token_loss_mask,
        )

    def predict_structures(
        self,
        features: EncoderBatch,
        *,
        token_views: Sequence[Sequence[TokenCharSpan]] | None = None,
    ) -> PlannerPredictionsBatch:
        """Inference structure prediction (no GraphDecoder).

        Uses one already-computed ``EncoderBatch``. Builds token views from the
        encoder tokenizer when ``token_views`` is omitted.
        """
        if token_views is None:
            token_views = self.encoder.token_char_spans_for_batch(features)
        if len(token_views) != features.batch_size:
            raise ValueError("token_views batch size must match features")

        batch_size, token_length, _ = features.token_embeddings.shape
        token_loss_mask = torch.zeros(
            batch_size,
            token_length,
            dtype=torch.bool,
            device=features.token_embeddings.device,
        )
        for batch_index, tokens in enumerate(token_views):
            if len(tokens) != token_length:
                raise ValueError("token view length must match encoder tokens")
            token_loss_mask[batch_index] = torch.tensor(
                [token.is_content for token in tokens],
                dtype=torch.bool,
                device=features.token_embeddings.device,
            )

        # First pass: BIO heads only need dummy empty structure for shapes.
        empty_structure = StructureBatch(
            op_span_mask=torch.zeros(
                batch_size,
                0,
                token_length,
                dtype=torch.bool,
                device=features.token_embeddings.device,
            ),
            op_valid=torch.zeros(
                batch_size,
                0,
                dtype=torch.bool,
                device=features.token_embeddings.device,
            ),
            anc_span_mask=torch.zeros(
                batch_size,
                0,
                token_length,
                dtype=torch.bool,
                device=features.token_embeddings.device,
            ),
            anc_valid=torch.zeros(
                batch_size,
                0,
                dtype=torch.bool,
                device=features.token_embeddings.device,
            ),
        )
        bio_outputs = self.forward_heads(
            features,
            structure=empty_structure,
            token_loss_mask=token_loss_mask,
        )

        pred_op_spans: list[list[tuple[int, int]]] = []
        pred_anc_spans: list[list[tuple[int, int]]] = []
        for batch_index, tokens in enumerate(token_views):
            op_labels = bio_outputs.op_bio_logits[batch_index].argmax(dim=-1).tolist()
            anc_labels = bio_outputs.anc_bio_logits[batch_index].argmax(dim=-1).tolist()
            # Force non-content tokens to O for decoding stability.
            op_labels = [
                BIO_O if not token.is_content else int(label)
                for label, token in zip(op_labels, tokens, strict=True)
            ]
            anc_labels = [
                BIO_O if not token.is_content else int(label)
                for label, token in zip(anc_labels, tokens, strict=True)
            ]
            pred_op_spans.append(list(decode_bio_spans(op_labels, tokens)))
            pred_anc_spans.append(list(decode_bio_spans(anc_labels, tokens)))

        max_ops = max((len(spans) for spans in pred_op_spans), default=0)
        max_anc = max((len(spans) for spans in pred_anc_spans), default=0)
        op_span_mask = torch.zeros(
            batch_size,
            max_ops,
            token_length,
            dtype=torch.bool,
            device=features.token_embeddings.device,
        )
        op_valid = torch.zeros(
            batch_size,
            max_ops,
            dtype=torch.bool,
            device=features.token_embeddings.device,
        )
        anc_span_mask = torch.zeros(
            batch_size,
            max_anc,
            token_length,
            dtype=torch.bool,
            device=features.token_embeddings.device,
        )
        anc_valid = torch.zeros(
            batch_size,
            max_anc,
            dtype=torch.bool,
            device=features.token_embeddings.device,
        )

        for batch_index, tokens in enumerate(token_views):
            for op_index, (start, end) in enumerate(pred_op_spans[batch_index]):
                op_valid[batch_index, op_index] = True
                for token_index, token in enumerate(tokens):
                    if not token.is_content:
                        continue
                    assert token.char_start is not None and token.char_end is not None
                    if token.char_start < end and token.char_end > start:
                        op_span_mask[batch_index, op_index, token_index] = True
            for anc_index, (start, end) in enumerate(pred_anc_spans[batch_index]):
                anc_valid[batch_index, anc_index] = True
                for token_index, token in enumerate(tokens):
                    if not token.is_content:
                        continue
                    assert token.char_start is not None and token.char_end is not None
                    if token.char_start < end and token.char_end > start:
                        anc_span_mask[batch_index, anc_index, token_index] = True

        structure = StructureBatch(
            op_span_mask=op_span_mask,
            op_valid=op_valid,
            anc_span_mask=anc_span_mask,
            anc_valid=anc_valid,
            op_type_indices=None,
        )
        outputs = self.forward_heads(
            features,
            structure=structure,
            token_loss_mask=token_loss_mask,
        )

        items: list[PlannerPredictions] = []
        for batch_index, tokens in enumerate(token_views):
            text = features.texts[batch_index]
            aux_index = int(outputs.query_type_logits[batch_index].argmax().item())
            aux_query_type = QUERY_TYPES[aux_index]

            operations: list[PredictedOperation] = []
            n_ops = int(op_valid[batch_index].sum().item())
            for op_index in range(n_ops):
                start, end = pred_op_spans[batch_index][op_index]
                type_index = int(outputs.op_type_logits[batch_index, op_index].argmax().item())
                operations.append(
                    PredictedOperation(
                        start=start,
                        end=end,
                        operator=INDEX_TO_OPERATOR[type_index],
                    )
                )

            anchors: list[PredictedAnchor] = []
            n_anc = int(anc_valid[batch_index].sum().item())
            for anc_index in range(n_anc):
                start, end = pred_anc_spans[batch_index][anc_index]
                if not (0 <= start < end <= len(text)):
                    # Leave an invalid span for decode to reject rather than repair.
                    span_text = ""
                else:
                    span_text = text[start:end]
                impl_index = int(outputs.impl_logits[batch_index, anc_index].argmax().item())
                if n_ops == 0:
                    owner_index = 0
                else:
                    masked_scores = outputs.own_logits[batch_index, anc_index].clone()
                    row_mask = outputs.own_mask[batch_index, anc_index]
                    masked_scores = masked_scores.masked_fill(~row_mask, float("-inf"))
                    if not bool(row_mask.any()):
                        owner_index = 0
                    else:
                        owner_index = int(masked_scores.argmax().item())
                anchors.append(
                    PredictedAnchor(
                        start=start,
                        end=end,
                        text=span_text,
                        owner_index=owner_index,
                        implicit_resolution=INDEX_TO_IMPLICIT[impl_index],
                        normalized_name=None,
                    )
                )

            dependency_pairs: set[tuple[int, int]] = set()
            if n_ops > 0:
                dep_logits = outputs.dep_logits[batch_index]
                dep_mask = outputs.dep_mask[batch_index]
                for source in range(n_ops):
                    for target in range(n_ops):
                        if not bool(dep_mask[source, target]):
                            continue
                        if float(dep_logits[source, target].item()) > 0.0:
                            dependency_pairs.add((source, target))

            items.append(
                PlannerPredictions(
                    operations=tuple(operations),
                    anchors=tuple(anchors),
                    dependency_pairs=frozenset(dependency_pairs),
                    aux_query_type=aux_query_type,
                )
            )

        return PlannerPredictionsBatch(
            items=tuple(items),
            token_views=tuple(tuple(view) for view in token_views),
            head_outputs=outputs,
        )


__all__ = [
    "PlannerHeadOutputs",
    "PlannerModel",
    "PlannerPredictionsBatch",
    "decode_bio_spans",
    "masked_mean_pool",
]
