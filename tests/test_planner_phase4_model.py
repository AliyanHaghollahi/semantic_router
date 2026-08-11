"""Unit tests for Phase-4 learned planner model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tiergraph.enums import OperatorType, QueryType, SlotType, TransferPolicy
from tiergraph.planner.align import BIO_B, BIO_I, BIO_IGNORE, BIO_O, TokenCharSpan
from tiergraph.planner.annotations import ImplicitResolution, PlannerExample
from tiergraph.planner.batching import (
    ANSWER_OPERATORS,
    GoldStructureBatch,
    OPERATOR_TO_INDEX,
    QUERY_TYPES,
    StructureBatch,
    collate_gold_structure_batch,
    dependency_eligibility_mask,
)
from tiergraph.planner.decode import GraphDecoder, PlannerPredictions, PredictedAnchor, PredictedOperation
from tiergraph.planner.encoder import EncoderBatch, MiniLMFeatureEncoder
from tiergraph.planner.loss import planner_loss
from tiergraph.planner.model import (
    PlannerModel,
    decode_bio_spans,
    masked_mean_pool,
)
from tiergraph.planner.targets import build_planner_targets


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "planner" / "where_is_my_gate.json"
)


class _CountingTokenizer:
    pad_token_id = 0
    all_special_ids = (101, 102)
    model_input_names = ("input_ids", "attention_mask")
    encode_calls = 0

    def __call__(self, texts, **kwargs):
        type(self).encode_calls += 1
        input_ids = []
        attention_mask = []
        offset_mapping = []
        for text in texts:
            ids = [101]
            offsets = [(0, 0)]
            cursor = 0
            for piece in text.split(" "):
                if cursor > 0:
                    cursor += 1  # space not covered
                start = cursor
                end = start + len(piece)
                ids.append(10 + len(piece))
                offsets.append((start, end))
                cursor = end
            ids.append(102)
            offsets.append((0, 0))
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
            offset_mapping.append(offsets)
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = offset_mapping
        return output


class _CountingTransformer(torch.nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask, **kwargs):
        self.forward_calls += 1
        hidden = self.config.hidden_size
        values = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        token_embeddings = values.repeat(1, 1, hidden)
        return SimpleNamespace(last_hidden_state=token_embeddings)


def _make_encoder(hidden_size: int = 4) -> tuple[MiniLMFeatureEncoder, _CountingTransformer]:
    tokenizer = _CountingTokenizer()
    model = _CountingTransformer(hidden_size=hidden_size)
    encoder = MiniLMFeatureEncoder(
        max_length=32,
        tokenizer_loader=lambda _name: tokenizer,
        model_loader=lambda _name: model,
    )
    return encoder, model


def _manual_token_views(batch: EncoderBatch) -> tuple[tuple[TokenCharSpan, ...], ...]:
    views = []
    for text, ids, mask in zip(
        batch.texts,
        batch.input_ids.tolist(),
        batch.attention_mask.tolist(),
        strict=True,
    ):
        tokens: list[TokenCharSpan] = []
        # Mirror counting tokenizer layout after padding.
        pieces = text.split(" ")
        cursor = 0
        content_offsets = []
        for piece in pieces:
            if cursor > 0:
                cursor += 1
            content_offsets.append((cursor, cursor + len(piece)))
            cursor += len(piece)
        content_iter = iter(content_offsets)
        for token_id, attended in zip(ids, mask, strict=True):
            if not bool(attended):
                tokens.append(TokenCharSpan(None, None, False, True))
                continue
            if token_id in (101, 102):
                tokens.append(TokenCharSpan(None, None, True, False))
                continue
            start, end = next(content_iter)
            tokens.append(TokenCharSpan(start, end, False, False))
        views.append(tuple(tokens))
    return tuple(views)


def test_exactly_one_encoder_forward_per_batch():
    _CountingTokenizer.encode_calls = 0
    encoder, transformer = _make_encoder()
    model = PlannerModel(encoder, hidden_size=4)
    features = model.encode(["Where is my gate?", "Read this sign"])
    assert transformer.forward_calls == 1
    assert _CountingTokenizer.encode_calls == 1
    views = _manual_token_views(features)
    empty = StructureBatch(
        op_span_mask=torch.zeros(2, 0, features.input_ids.shape[1], dtype=torch.bool),
        op_valid=torch.zeros(2, 0, dtype=torch.bool),
        anc_span_mask=torch.zeros(2, 0, features.input_ids.shape[1], dtype=torch.bool),
        anc_valid=torch.zeros(2, 0, dtype=torch.bool),
    )
    model.forward_heads(features, structure=empty)
    assert transformer.forward_calls == 1


def test_encoder_frozen_and_no_encoder_gradients():
    encoder, transformer = _make_encoder()
    model = PlannerModel(encoder, hidden_size=4)
    features = model.encode(["Where is my gate?"])
    assert all(not p.requires_grad for p in transformer.parameters())
    views = encoder.token_char_spans_for_batch(features)
    gold_targets = []
    # Minimal gold batch with no ops/anchors.
    from tiergraph.planner.targets import PlannerTargets
    from tiergraph.planner.align import AlignmentStats, BioEncoding

    # Build empty-ish gold via collate using fixture instead.
    example = PlannerExample.model_validate(
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    )
    targets = (build_planner_targets(example, views[0]),)
    # Pad batch size 1 features already.
    gold = collate_gold_structure_batch(
        features=features,
        targets=targets,
        token_views=views,
    )
    outputs = model.forward_train(features, gold)
    loss = outputs.query_type_logits.sum()
    loss.backward()
    for parameter in transformer.parameters():
        assert parameter.grad is None


def test_head_output_shapes():
    encoder, _ = _make_encoder(hidden_size=5)
    model = PlannerModel(encoder, hidden_size=5)
    features = model.encode(["abc def", "xy"])
    b, t, h = features.token_embeddings.shape
    assert h == 5
    o, a = 3, 2
    structure = StructureBatch(
        op_span_mask=torch.zeros(b, o, t, dtype=torch.bool),
        op_valid=torch.tensor([[True, True, False], [True, False, False]]),
        anc_span_mask=torch.zeros(b, a, t, dtype=torch.bool),
        anc_valid=torch.tensor([[True, False], [True, True]]),
        op_type_indices=torch.tensor([[2, 3, -1], [0, -1, -1]]),
    )
    # Mark some tokens in spans.
    structure.op_span_mask[0, 0, 1] = True
    structure.op_span_mask[0, 1, 2] = True
    structure.op_span_mask[1, 0, 1] = True
    structure.anc_span_mask[0, 0, 1] = True
    structure.anc_span_mask[1, 0, 1] = True
    structure.anc_span_mask[1, 1, 1] = True
    outputs = model.forward_heads(features, structure=structure)
    assert outputs.query_type_logits.shape == (b, 3)
    assert outputs.op_bio_logits.shape == (b, t, 3)
    assert outputs.anc_bio_logits.shape == (b, t, 3)
    assert outputs.op_type_logits.shape == (b, o, 6)
    assert outputs.impl_logits.shape == (b, a, 2)
    assert outputs.own_logits.shape == (b, a, o)
    assert outputs.dep_logits.shape == (b, o, o)
    assert len(ANSWER_OPERATORS) == 6


def test_operation_and_anchor_mean_pooling():
    token_embeddings = torch.tensor(
        [
            [[1.0, 0.0], [3.0, 3.0], [5.0, 5.0]],
        ]
    )
    span_mask = torch.tensor([[[False, True, True]]])
    pooled = masked_mean_pool(token_embeddings, span_mask)
    assert pooled.shape == (1, 1, 2)
    assert torch.allclose(pooled[0, 0], torch.tensor([4.0, 4.0]))


def test_h2_h4_overlap_independence():
    encoder, _ = _make_encoder()
    model = PlannerModel(encoder, hidden_size=4)
    features = model.encode(["Where is my gate?"])
    empty = StructureBatch(
        op_span_mask=torch.zeros(1, 0, features.input_ids.shape[1], dtype=torch.bool),
        op_valid=torch.zeros(1, 0, dtype=torch.bool),
        anc_span_mask=torch.zeros(1, 0, features.input_ids.shape[1], dtype=torch.bool),
        anc_valid=torch.zeros(1, 0, dtype=torch.bool),
    )
    outputs = model.forward_heads(features, structure=empty)
    # Independent modules: changing one head weight does not change the other logits.
    before_anc = outputs.anc_bio_logits.detach().clone()
    with torch.no_grad():
        model.op_bio_head.weight.mul_(2.0)
    after = model.forward_heads(features, structure=empty)
    assert torch.allclose(before_anc, after.anc_bio_logits)


def test_h6_masking_and_h7_directionality_and_eligibility():
    encoder, _ = _make_encoder()
    model = PlannerModel(encoder, hidden_size=4)
    features = model.encode(["abc def ghi"])
    # Replace isotropic fake embeddings with non-collinear token vectors so
    # asymmetric bilinear H7 scores are observable.
    b, t, h = features.token_embeddings.shape
    custom = torch.zeros(b, t, h)
    for token_index in range(t):
        custom[0, token_index] = torch.tensor(
            [
                float(token_index + 1),
                float((token_index + 1) ** 2),
                float(token_index % 2),
                float((token_index + 3) % 5),
            ]
        )
    features = EncoderBatch(
        input_ids=features.input_ids,
        attention_mask=features.attention_mask,
        token_embeddings=custom,
        pooled_embeddings=custom.mean(dim=1),
        truncated=features.truncated,
        texts=features.texts,
    )
    structure = StructureBatch(
        op_span_mask=torch.zeros(1, 2, t, dtype=torch.bool),
        op_valid=torch.tensor([[True, True]]),
        anc_span_mask=torch.zeros(1, 1, t, dtype=torch.bool),
        anc_valid=torch.tensor([[True]]),
        op_type_indices=torch.tensor(
            [[
                ANSWER_OPERATORS.index(OperatorType.IDENTIFY_ENVIRONMENTAL),
                ANSWER_OPERATORS.index(OperatorType.LOCATE_ENVIRONMENTAL),
            ]]
        ),
    )
    structure.op_span_mask[0, 0, 1] = True
    structure.op_span_mask[0, 1, 2] = True
    structure.anc_span_mask[0, 0, 1] = True
    outputs = model.forward_heads(features, structure=structure)
    assert outputs.own_mask.shape == (1, 1, 2)
    assert bool(outputs.own_mask[0, 0].all())
    assert outputs.dep_logits.shape == (1, 2, 2)
    # Self excluded / eligibility: identify->locate eligible; reverse depends.
    assert bool(outputs.dep_mask[0, 0, 1])
    assert not bool(outputs.dep_mask[0, 0, 0])
    assert not bool(outputs.dep_mask[0, 1, 1])
    # locate -> identify is ineligible under V1.
    assert not bool(outputs.dep_mask[0, 1, 0])
    # Distinct source/target projections + non-collinear ops => directed scores.
    with torch.no_grad():
        model.dep_source_proj.weight.copy_(torch.eye(4))
        target_weight = torch.zeros(4, 4)
        target_weight[0, 1] = 1.0
        target_weight[1, 0] = 2.0
        model.dep_target_proj.weight.copy_(target_weight)
    outputs = model.forward_heads(features, structure=structure)
    assert outputs.dep_logits[0, 0, 1] != outputs.dep_logits[0, 1, 0]
    assert not torch.allclose(outputs.dep_logits[0], outputs.dep_logits[0].T)


def test_zero_anchor_and_zero_dependency_safety():
    encoder, _ = _make_encoder()
    model = PlannerModel(encoder, hidden_size=4)
    features = model.encode(["hello world"])
    b, t, _ = features.token_embeddings.shape
    structure = StructureBatch(
        op_span_mask=torch.zeros(1, 1, t, dtype=torch.bool),
        op_valid=torch.tensor([[True]]),
        anc_span_mask=torch.zeros(1, 0, t, dtype=torch.bool),
        anc_valid=torch.zeros(1, 0, dtype=torch.bool),
        op_type_indices=torch.tensor(
            [[ANSWER_OPERATORS.index(OperatorType.DESCRIBE_ENVIRONMENT)]]
        ),
    )
    structure.op_span_mask[0, 0, 1] = True
    outputs = model.forward_heads(features, structure=structure)
    assert outputs.anc_repr.shape == (1, 0, 4)
    assert outputs.impl_logits.shape == (1, 0, 2)
    assert outputs.own_logits.shape == (1, 0, 1)
    assert outputs.dep_mask.shape == (1, 1, 1)
    assert not bool(outputs.dep_mask.any())


def test_bio_decode_invalid_i_starts_new_span():
    tokens = (
        TokenCharSpan(None, None, True, False),
        TokenCharSpan(0, 2, False, False),
        TokenCharSpan(2, 4, False, False),
        TokenCharSpan(None, None, True, False),
    )
    labels = [BIO_O, BIO_I, BIO_I, BIO_O]
    spans = decode_bio_spans(labels, tokens)
    assert spans == ((0, 4),)


def test_train_uses_gold_structures_not_predicted_bio():
    encoder, _ = _make_encoder()
    model = PlannerModel(encoder, hidden_size=4)
    features = model.encode(["Where is my gate?"])
    views = encoder.token_char_spans_for_batch(features)
    example = PlannerExample.model_validate(
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    )
    targets = (build_planner_targets(example, views[0]),)
    gold = collate_gold_structure_batch(
        features=features,
        targets=targets,
        token_views=views,
    )
    # Corrupt BIO head heavily; training forward still pools gold spans.
    with torch.no_grad():
        model.op_bio_head.weight.zero_()
        model.op_bio_head.bias.copy_(torch.tensor([10.0, -10.0, -10.0]))
    outputs = model.forward_train(features, gold)
    assert outputs.op_valid.shape[1] == gold.op_valid.shape[1]
    assert torch.equal(outputs.op_valid, gold.op_valid)
    # Gold supervised op span mask is used for pooling → op_repr nonzero where gold span is.
    assert outputs.op_repr.shape[1] == gold.op_span_mask.shape[1]


def test_inference_uses_predicted_structures(monkeypatch):
    encoder, _ = _make_encoder()
    model = PlannerModel(encoder, hidden_size=4)
    features = model.encode(["Where is my gate?"])
    views = encoder.token_char_spans_for_batch(features)

    def _fake_decode(labels, tokens):
        # Force one full-content span for ops and one for anchors.
        content = [token for token in tokens if token.is_content]
        if not content:
            return ()
        return ((content[0].char_start, content[-1].char_end),)

    monkeypatch.setattr(
        "tiergraph.planner.model.decode_bio_spans",
        _fake_decode,
    )
    predicted = model.predict_structures(features, token_views=views)
    assert len(predicted.items) == 1
    assert len(predicted.items[0].operations) == 1
    assert predicted.head_outputs.op_valid[0].sum() == 1


def test_h1_disagreement_does_not_override_graph_query_type():
    query = "Where is my gate?"
    predictions = PlannerPredictions(
        operations=(
            PredictedOperation(
                start=0,
                end=len(query),
                operator=OperatorType.LOCATE_ENVIRONMENTAL,
            ),
        ),
        anchors=(
            PredictedAnchor(
                start=query.index("gate"),
                end=query.index("gate") + 4,
                text="gate",
                owner_index=0,
                implicit_resolution=ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL,
                normalized_name="gate",
            ),
        ),
        dependency_pairs=frozenset(),
        aux_query_type=QueryType.ENVIRONMENTAL,
    )
    decoded = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="gate-aux",
    )
    assert predictions.aux_query_type is QueryType.ENVIRONMENTAL
    assert decoded.graph.query_type is QueryType.MIXED


def test_gate_predictions_reach_decoder_as_mixed():
    example = PlannerExample.model_validate(
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    )
    op = example.planner_labels.operation_spans[0]
    anchor = example.planner_labels.slot_anchors[0]
    predictions = PlannerPredictions(
        operations=(
            PredictedOperation(
                start=op.start,
                end=op.end,
                operator=op.operator,
            ),
        ),
        anchors=(
            PredictedAnchor(
                start=anchor.start,
                end=anchor.end,
                text=anchor.text,
                owner_index=0,
                implicit_resolution=anchor.implicit_resolution,
                normalized_name=anchor.normalized_name,
            ),
        ),
        aux_query_type=QueryType.PERSONAL,
    )
    decoded = GraphDecoder().decode(
        predictions,
        query=example.query,
        graph_id=example.example_id,
    )
    assert decoded.graph.query_type is QueryType.MIXED


def test_model_does_not_create_second_encoder():
    encoder, _ = _make_encoder()
    model = PlannerModel(encoder, hidden_size=4)
    assert model.encoder is encoder


def test_tokenizer_alignment_matches_encoded_tokens():
    encoder, _ = _make_encoder()
    model = PlannerModel(encoder, hidden_size=4)
    features = model.encode(["Where is my gate?"])
    views = encoder.token_char_spans_for_batch(features)
    assert len(views[0]) == features.input_ids.shape[1]
    attended = int(features.attention_mask[0].sum().item())
    content = [token for token in views[0] if token.is_content]
    specials = [token for token in views[0][:attended] if token.is_special]
    assert specials
    assert content
    # Non-attended tail is padding.
    for token, attended_flag in zip(
        views[0],
        features.attention_mask[0].tolist(),
        strict=True,
    ):
        if not bool(attended_flag):
            assert token.is_padding


def test_dependency_eligibility_helper_masks_self_pairs():
    op_valid = torch.tensor([[True, True]])
    op_types = torch.tensor(
        [[
            ANSWER_OPERATORS.index(OperatorType.IDENTIFY_ENVIRONMENTAL),
            ANSWER_OPERATORS.index(OperatorType.LOCATE_ENVIRONMENTAL),
        ]]
    )
    mask = dependency_eligibility_mask(op_valid=op_valid, op_type_indices=op_types)
    assert not bool(mask[0, 0, 0])
    assert not bool(mask[0, 1, 1])
    assert bool(mask[0, 0, 1])


def test_offset_path_does_not_reencode_transformer():
    _CountingTokenizer.encode_calls = 0
    encoder, transformer = _make_encoder()
    model = PlannerModel(encoder, hidden_size=4)
    features = model.encode(["Where is my gate?"])
    assert transformer.forward_calls == 1
    encode_tokenizer_calls = _CountingTokenizer.encode_calls

    views = encoder.token_char_spans_for_batch(features)
    assert transformer.forward_calls == 1
    assert _CountingTokenizer.encode_calls == encode_tokenizer_calls + 1

    attended = int(features.attention_mask[0].sum().item())
    retained_ids = features.input_ids[0, :attended].tolist()
    tokenizer = encoder._tokenizer
    assert tokenizer is not None
    reencoded = tokenizer(
        list(features.texts),
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )
    assert reencoded["input_ids"][0] == retained_ids
    assert len(views[0]) == features.input_ids.shape[1]
    assert sum(1 for token in views[0] if not token.is_padding) == attended


def test_positive_head_gradients_including_h7_projections():
    encoder, transformer = _make_encoder()
    model = PlannerModel(encoder, hidden_size=4)
    features = model.encode(["abc def ghi"])
    views = encoder.token_char_spans_for_batch(features)
    _batch, token_length, _hidden = features.token_embeddings.shape

    identify = OPERATOR_TO_INDEX[OperatorType.IDENTIFY_ENVIRONMENTAL]
    locate = OPERATOR_TO_INDEX[OperatorType.LOCATE_ENVIRONMENTAL]
    op_span_mask = torch.zeros(1, 2, token_length, dtype=torch.bool)
    op_span_mask[0, 0, 1] = True
    op_span_mask[0, 1, 2] = True
    token_loss_mask = torch.tensor([[token.is_content for token in views[0]]])
    op_bio_labels = torch.full((1, token_length), BIO_O, dtype=torch.long)
    for index, token in enumerate(views[0]):
        if token.is_content:
            op_bio_labels[0, index] = BIO_B
            break

    dep_mask = torch.zeros(1, 2, 2, dtype=torch.bool)
    dep_labels = torch.zeros(1, 2, 2)
    dep_mask[0, 0, 1] = True
    dep_labels[0, 0, 1] = 1.0
    assert bool(
        dependency_eligibility_mask(
            op_valid=torch.tensor([[True, True]]),
            op_type_indices=torch.tensor([[identify, locate]]),
        )[0, 0, 1]
    )

    gold = GoldStructureBatch(
        query_type_labels=torch.tensor([0]),
        op_bio_labels=op_bio_labels,
        anc_bio_labels=torch.full((1, token_length), BIO_IGNORE, dtype=torch.long),
        token_loss_mask=token_loss_mask,
        op_span_mask=op_span_mask,
        op_valid=torch.tensor([[True, True]]),
        op_type_labels=torch.tensor([[identify, locate]]),
        anc_span_mask=torch.zeros(1, 0, token_length, dtype=torch.bool),
        anc_valid=torch.zeros(1, 0, dtype=torch.bool),
        impl_labels=torch.zeros(1, 0, dtype=torch.long),
        own_labels=torch.zeros(1, 0, dtype=torch.long),
        own_mask=torch.zeros(1, 0, 2, dtype=torch.bool),
        dep_labels=dep_labels,
        dep_mask=dep_mask,
        example_ids=("h7-grad",),
    )
    outputs = model.forward_train(features, gold)
    assert bool(outputs.dep_mask[0, 0, 1])
    loss = planner_loss(outputs, gold).total
    loss.backward()

    assert model.op_bio_head.weight.grad is not None
    assert model.dep_source_proj.weight.grad is not None
    assert model.dep_target_proj.weight.grad is not None
    for parameter in transformer.parameters():
        assert parameter.grad is None


def test_model_predict_structures_to_decoder_gate_mixed():
    """Exercise predict_structures → GraphDecoder for the gate fixture shape."""
    encoder, transformer = _make_encoder()
    model = PlannerModel(encoder, hidden_size=4)
    query = "Where is my gate?"
    # Controlled token view: split "gate" from "?" so naming gets base "gate".
    token_views = (
        (
            TokenCharSpan(None, None, True, False),
            TokenCharSpan(0, 5, False, False),
            TokenCharSpan(6, 8, False, False),
            TokenCharSpan(9, 11, False, False),
            TokenCharSpan(12, 16, False, False),
            TokenCharSpan(16, 17, False, False),
            TokenCharSpan(None, None, True, False),
        ),
    )
    token_length = len(token_views[0])
    features = EncoderBatch(
        input_ids=torch.tensor([[101, 1, 2, 3, 4, 5, 102]]),
        attention_mask=torch.ones(1, token_length, dtype=torch.long),
        token_embeddings=torch.randn(1, token_length, 4),
        pooled_embeddings=torch.randn(1, 4),
        truncated=(False,),
        texts=(query,),
    )

    locate_index = ANSWER_OPERATORS.index(OperatorType.LOCATE_ENVIRONMENTAL)
    environmental_index = QUERY_TYPES.index(QueryType.ENVIRONMENTAL)
    implicit_index = 1  # IMPLICIT_RESOLVE_PERSONAL
    # BIO over [CLS, Where, is, my, gate, ?, SEP]
    op_bio_labels = [BIO_O, BIO_B, BIO_I, BIO_I, BIO_I, BIO_I, BIO_O]
    anc_bio_labels = [BIO_O, BIO_O, BIO_O, BIO_O, BIO_B, BIO_O, BIO_O]

    def _token_bio_logits(labels: list[int]):
        def _forward(x: torch.Tensor) -> torch.Tensor:
            batch_size, length, _ = x.shape
            logits = torch.full(
                (batch_size, length, 3),
                -50.0,
                dtype=x.dtype,
                device=x.device,
            )
            for token_index, label in enumerate(labels):
                logits[0, token_index, label] = 50.0
            return logits

        return _forward

    def _query_type_forward(x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((x.shape[0], 3), -50.0, dtype=x.dtype, device=x.device)
        logits[0, environmental_index] = 50.0
        return logits

    def _op_type_forward(x: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (x.shape[0], x.shape[1], len(ANSWER_OPERATORS)),
            -50.0,
            dtype=x.dtype,
            device=x.device,
        )
        logits[0, :, locate_index] = 50.0
        return logits

    def _impl_forward(x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((x.shape[0], x.shape[1], 2), -50.0, dtype=x.dtype, device=x.device)
        logits[0, :, implicit_index] = 50.0
        return logits

    model.op_bio_head.forward = _token_bio_logits(op_bio_labels)
    model.anc_bio_head.forward = _token_bio_logits(anc_bio_labels)
    model.query_type_head.forward = _query_type_forward
    model.op_type_head.forward = _op_type_forward
    model.impl_head.forward = _impl_forward

    predicted_batch = model.predict_structures(features, token_views=token_views)
    assert transformer.forward_calls == 0
    predictions = predicted_batch.items[0]
    assert predictions.aux_query_type is QueryType.ENVIRONMENTAL
    assert len(predictions.operations) == 1
    assert predictions.operations[0].operator is OperatorType.LOCATE_ENVIRONMENTAL
    assert predictions.operations[0].start == 0
    assert predictions.operations[0].end == 17
    assert len(predictions.anchors) == 1
    assert predictions.anchors[0].start == 12
    assert predictions.anchors[0].end == 16
    assert predictions.anchors[0].text == "gate"
    assert (
        predictions.anchors[0].implicit_resolution
        is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
    )
    assert predictions.anchors[0].owner_index == 0

    decoded = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="mixed-gate-model-wiring",
    )
    locate_nodes = [
        node
        for node in decoded.graph.nodes
        if node.operator is OperatorType.LOCATE_ENVIRONMENTAL
    ]
    resolve_nodes = [
        node
        for node in decoded.graph.nodes
        if node.operator is OperatorType.RESOLVE_PERSONAL
    ]
    assert len(locate_nodes) == 1
    assert len(resolve_nodes) == 1
    assert any(
        edge.transfer_policy is TransferPolicy.MINIMAL_REFERENCE
        and SlotType.RESOLVED_REFERENCE
        in (
            locate_nodes[0].required_inputs.get(edge.target_slot),
            resolve_nodes[0].produced_outputs.get(edge.source_slot),
        )
        for edge in decoded.graph.edges
    )
    assert any(
        edge.source_node_id == resolve_nodes[0].node_id
        and edge.target_node_id == locate_nodes[0].node_id
        and resolve_nodes[0].produced_outputs.get(edge.source_slot)
        is SlotType.RESOLVED_REFERENCE
        and locate_nodes[0].required_inputs.get(edge.target_slot)
        is SlotType.RESOLVED_REFERENCE
        for edge in decoded.graph.edges
    )
    assert decoded.graph.query_type is QueryType.MIXED
    assert predictions.aux_query_type is QueryType.ENVIRONMENTAL
