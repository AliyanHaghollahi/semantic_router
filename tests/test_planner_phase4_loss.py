"""Unit tests for Phase-4 planner multi-task loss."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from tiergraph.planner.align import BIO_IGNORE, BIO_O
from tiergraph.planner.annotations import PlannerExample
from tiergraph.planner.batching import GoldStructureBatch, collate_gold_structure_batch
from tiergraph.planner.encoder import MiniLMFeatureEncoder
from tiergraph.planner.loss import planner_loss
from tiergraph.planner.model import PlannerModel
from tiergraph.planner.targets import build_planner_targets


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "planner" / "where_is_my_gate.json"
)


class _Tok:
    pad_token_id = 0
    all_special_ids = (101, 102)
    model_input_names = ("input_ids", "attention_mask")

    def __call__(self, texts, **kwargs):
        input_ids = []
        attention_mask = []
        offset_mapping = []
        for text in texts:
            ids = [101]
            offsets = [(0, 0)]
            cursor = 0
            for piece in text.split(" "):
                if cursor > 0:
                    cursor += 1
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
        output = {"input_ids": input_ids, "attention_mask": attention_mask}
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = offset_mapping
        return output


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(hidden_size=4)

    def forward(self, input_ids, attention_mask, **kwargs):
        values = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=values.repeat(1, 1, 4))


def _encoder() -> MiniLMFeatureEncoder:
    return MiniLMFeatureEncoder(
        max_length=32,
        tokenizer_loader=lambda _n: _Tok(),
        model_loader=lambda _n: _Model(),
    )


def test_all_loss_components_finite_and_equal_weight_sum():
    encoder = _encoder()
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
    outputs = model.forward_train(features, gold)
    breakdown = planner_loss(outputs, gold)
    for value in (
        breakdown.total,
        breakdown.h1,
        breakdown.h2,
        breakdown.h3,
        breakdown.h4,
        breakdown.h5,
        breakdown.h6,
        breakdown.h7,
    ):
        assert torch.isfinite(value)
    expected = (
        breakdown.h1
        + breakdown.h2
        + breakdown.h3
        + breakdown.h4
        + breakdown.h5
        + breakdown.h6
        + breakdown.h7
    )
    assert torch.allclose(breakdown.total, expected)


def test_empty_head_losses_are_zero_not_nan():
    encoder = _encoder()
    model = PlannerModel(encoder, hidden_size=4)
    features = model.encode(["hello world"])
    views = encoder.token_char_spans_for_batch(features)
    _b, t = features.input_ids.shape

    gold = GoldStructureBatch(
        query_type_labels=torch.tensor([0]),
        op_bio_labels=torch.full((1, t), BIO_O, dtype=torch.long),
        anc_bio_labels=torch.full((1, t), BIO_IGNORE, dtype=torch.long),
        token_loss_mask=torch.tensor([[token.is_content for token in views[0]]]),
        op_span_mask=torch.zeros(1, 1, t, dtype=torch.bool),
        op_valid=torch.tensor([[True]]),
        op_type_labels=torch.tensor([[5]]),
        anc_span_mask=torch.zeros(1, 0, t, dtype=torch.bool),
        anc_valid=torch.zeros(1, 0, dtype=torch.bool),
        impl_labels=torch.zeros(1, 0, dtype=torch.long),
        own_labels=torch.zeros(1, 0, dtype=torch.long),
        own_mask=torch.zeros(1, 0, 1, dtype=torch.bool),
        dep_labels=torch.zeros(1, 1, 1),
        dep_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        example_ids=("synthetic",),
    )
    gold.op_span_mask[0, 0, 1] = True
    outputs = model.forward_train(features, gold)
    breakdown = planner_loss(outputs, gold)
    assert breakdown.h5.item() == 0.0
    assert breakdown.h6.item() == 0.0
    assert breakdown.h7.item() == 0.0
    assert not torch.isnan(breakdown.total)
