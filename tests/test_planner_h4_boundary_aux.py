"""H4.1 parameter-free start/end boundary auxiliaries."""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tiergraph.planner.align import BIO_B, BIO_I, BIO_O
from tiergraph.planner.batching import GoldStructureBatch
from tiergraph.planner.encoder import MiniLMFeatureEncoder
from tiergraph.planner.loss import (
    h4_boundary_targets,
    h4_end_log_probability,
    h4_end_log_terms,
    h4_end_loss,
    h4_start_logits,
    h4_start_loss,
    planner_loss,
)
from tiergraph.planner.model import PlannerHeadOutputs, PlannerModel
from tiergraph.planner.train import (
    TrainConfig,
    build_model,
    config_from_checkpoint,
    load_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / "artifacts" / "planner_train_v3_6epochs" / "best.pt"
HIGH = math.log(0.9)
LOW = math.log(0.1)


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


class _Enc(torch.nn.Module):
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
        tokenizer_loader=lambda _name: _Tok(),
        model_loader=lambda _name: _Enc(),
    )


def _peaked(label: int, magnitude: float = 20.0) -> torch.Tensor:
    logits = torch.full((3,), -magnitude)
    logits[label] = magnitude
    return logits


def _gold_and_outputs(
    labels: list[int],
    *,
    content: list[bool] | None = None,
    logits: torch.Tensor | None = None,
) -> tuple[PlannerHeadOutputs, GoldStructureBatch]:
    length = len(labels)
    if content is None:
        content = [True] * length
    label_tensor = torch.tensor([labels], dtype=torch.long)
    mask = torch.tensor([content])
    if logits is None:
        logits = torch.zeros(1, length, 3)
    zeros_ops = torch.zeros(1, 1, dtype=torch.bool)
    gold = GoldStructureBatch(
        query_type_labels=torch.tensor([0]),
        op_bio_labels=torch.zeros(1, length, dtype=torch.long),
        anc_bio_labels=label_tensor,
        token_loss_mask=mask,
        op_span_mask=torch.zeros(1, 1, length, dtype=torch.bool),
        op_valid=torch.tensor([[True]]),
        op_type_labels=torch.zeros(1, 1, dtype=torch.long),
        anc_span_mask=torch.zeros(1, 0, length, dtype=torch.bool),
        anc_valid=torch.zeros(1, 0, dtype=torch.bool),
        impl_labels=torch.zeros(1, 0, dtype=torch.long),
        own_labels=torch.zeros(1, 0, dtype=torch.long),
        own_mask=torch.zeros(1, 0, 1, dtype=torch.bool),
        dep_labels=torch.zeros(1, 1, 1),
        dep_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        example_ids=("synthetic",),
    )
    outputs = PlannerHeadOutputs(
        query_type_logits=torch.zeros(1, 3),
        op_bio_logits=torch.zeros(1, length, 3),
        anc_bio_logits=logits,
        op_type_logits=torch.zeros(1, 1, 6),
        impl_logits=torch.zeros(1, 0, 2),
        own_logits=torch.zeros(1, 0, 1),
        dep_logits=torch.zeros(1, 1, 1),
        op_valid=zeros_ops,
        anc_valid=torch.zeros(1, 0, dtype=torch.bool),
        own_mask=torch.zeros(1, 0, 1, dtype=torch.bool),
        dep_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        token_loss_mask=mask,
        op_repr=torch.zeros(1, 1, 4),
        anc_repr=torch.zeros(1, 0, 4),
    )
    return outputs, gold


def _end_at_first(first: int, second: int) -> float:
    logits = torch.stack([_peaked(first), _peaked(second)]).unsqueeze(0)
    mask = torch.tensor([[True, True]])
    return float(h4_end_log_probability(logits, mask)[0, 0])


def test_lambda_zero_matches_previous_h4_loss():
    labels = [BIO_O, BIO_B, BIO_I, BIO_O]
    outputs, gold = _gold_and_outputs(labels, logits=torch.randn(1, 4, 3))
    baseline = planner_loss(outputs, gold)
    explicit = planner_loss(
        outputs,
        gold,
        h4_boundary_lambda_start=0.0,
        h4_boundary_lambda_end=0.0,
    )
    assert torch.equal(baseline.h4, explicit.h4)
    assert torch.equal(baseline.total, explicit.total)


def test_start_logit_is_b_minus_logsumexp_outside():
    logits = torch.tensor([[[0.2, -0.4, 1.5], [3.0, 0.1, -2.0]]])
    outside = torch.logsumexp(logits[..., [0, 2]], dim=-1)
    expected = logits[..., 1] - outside
    assert torch.allclose(h4_start_logits(logits), expected)


def test_bio_transition_end_scores():
    assert _end_at_first(BIO_B, BIO_O) > HIGH
    assert _end_at_first(BIO_B, BIO_B) > HIGH
    assert _end_at_first(BIO_I, BIO_O) > HIGH
    assert _end_at_first(BIO_I, BIO_B) > HIGH
    assert _end_at_first(BIO_B, BIO_I) < LOW
    assert _end_at_first(BIO_I, BIO_I) < LOW
    assert _end_at_first(BIO_O, BIO_O) < LOW
    assert _end_at_first(BIO_O, BIO_B) < LOW
    assert _end_at_first(BIO_O, BIO_I) < LOW


def test_single_token_anchor_is_start_and_end():
    labels = torch.tensor([[BIO_B]])
    mask = torch.tensor([[True]])
    start, end = h4_boundary_targets(labels, mask)
    assert start.tolist() == [[True]]
    assert end.tolist() == [[True]]


def test_multitoken_span_ends_only_on_final_i():
    labels = torch.tensor([[BIO_B, BIO_I, BIO_I]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    start, end = h4_boundary_targets(labels, mask)
    assert start.tolist() == [[True, False, False]]
    assert end.tolist() == [[False, False, True]]


def test_adjacent_anchors():
    labels = torch.tensor([[BIO_B, BIO_I, BIO_B, BIO_I]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    start, end = h4_boundary_targets(labels, mask)
    assert start.tolist() == [[True, False, True, False]]
    assert end.tolist() == [[False, True, False, True]]


def test_anchorless_all_o():
    labels = torch.tensor([[BIO_O, BIO_O, BIO_O]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    start, end = h4_boundary_targets(labels, mask)
    assert not bool(start.any())
    assert not bool(end.any())
    logits = torch.randn(1, 3, 3)
    assert torch.isfinite(h4_end_loss(logits, labels, mask))
    assert torch.isfinite(h4_start_loss(logits, labels, mask))


def test_noncontent_tokens_are_skipped_as_next():
    # Content, pad/special, content. Next of the first content is the third index.
    labels = torch.tensor([[BIO_B, BIO_I, BIO_O]])
    mask = torch.tensor([[True, False, True]])
    start, end = h4_boundary_targets(labels, mask)
    assert start.tolist() == [[True, False, False]]
    # Next content is O, so the B token is an end. The pad I is ignored.
    assert end.tolist() == [[True, False, False]]
    logits = torch.stack(
        [
            _peaked(BIO_B),
            _peaked(BIO_I),  # would look like continuation if t+1 were used
            _peaked(BIO_O),
        ]
    ).unsqueeze(0)
    assert float(h4_end_log_probability(logits, mask)[0, 0]) > HIGH


def test_final_content_token_uses_inside_only():
    logits = torch.stack([_peaked(BIO_O), _peaked(BIO_B)]).unsqueeze(0)
    mask = torch.tensor([[True, True]])
    log_p = h4_end_log_probability(logits, mask)
    log_inside = torch.logsumexp(logits[0, 1, 1:], dim=-1) - torch.logsumexp(
        logits[0, 1],
        dim=-1,
    )
    assert torch.allclose(log_p[0, 1], log_inside)
    labels = torch.tensor([[BIO_O, BIO_B]])
    _start, end = h4_boundary_targets(labels, mask)
    assert end.tolist() == [[False, True]]


def test_auxiliary_backward_reaches_only_anc_bio_head():
    encoder = _encoder()
    model = PlannerModel(encoder, hidden_size=4)
    features = model.encode(["Where is my gate"])
    views = encoder.token_char_spans_for_batch(features)
    length = features.input_ids.shape[1]
    labels = torch.full((1, length), BIO_O, dtype=torch.long)
    mask = torch.tensor([[token.is_content for token in views[0]]])
    content = mask[0].nonzero(as_tuple=False).flatten()
    assert content.numel() >= 2
    labels[0, content[0]] = BIO_B
    labels[0, content[1]] = BIO_I
    op_span = torch.zeros(1, 1, length, dtype=torch.bool)
    anc_span = torch.zeros(1, 1, length, dtype=torch.bool)
    op_span[0, 0, content[0]] = True
    anc_span[0, 0, content[0]] = True
    gold = GoldStructureBatch(
        query_type_labels=torch.tensor([1]),
        op_bio_labels=torch.full((1, length), BIO_O, dtype=torch.long),
        anc_bio_labels=labels,
        token_loss_mask=mask,
        op_span_mask=op_span,
        op_valid=torch.tensor([[True]]),
        op_type_labels=torch.tensor([[0]]),
        anc_span_mask=anc_span,
        anc_valid=torch.tensor([[True]]),
        impl_labels=torch.tensor([[0]]),
        own_labels=torch.tensor([[0]]),
        own_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        dep_labels=torch.zeros(1, 1, 1),
        dep_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        example_ids=("aux",),
    )
    outputs = model.forward_train(features, gold)
    loss = h4_start_loss(
        outputs.anc_bio_logits, gold.anc_bio_labels, gold.token_loss_mask
    ) + h4_end_loss(outputs.anc_bio_logits, gold.anc_bio_labels, gold.token_loss_mask)
    loss.backward()
    assert model.anc_bio_head.weight.grad is not None
    assert float(model.anc_bio_head.weight.grad.abs().sum()) > 0.0
    encoder_scale = model.encoder._model.scale
    assert encoder_scale.requires_grad is False
    assert encoder_scale.grad is None
    for head in (
        model.query_type_head,
        model.op_bio_head,
        model.op_type_head,
        model.impl_head,
        model.anchor_proj,
        model.operation_proj,
        model.dep_source_proj,
        model.dep_target_proj,
    ):
        assert head.weight.grad is None


def test_disable_h4_drops_boundary_terms_from_total():
    labels = [BIO_B, BIO_I, BIO_O]
    outputs, gold = _gold_and_outputs(
        labels,
        logits=torch.randn(1, 3, 3, generator=torch.Generator().manual_seed(0)),
    )
    active = frozenset({"h1", "h2", "h3", "h5", "h6", "h7"})
    without = planner_loss(outputs, gold, active_heads=active)
    with_lambda = planner_loss(
        outputs,
        gold,
        active_heads=active,
        h4_boundary_lambda_start=1.0,
        h4_boundary_lambda_end=1.0,
    )
    assert torch.equal(without.total, with_lambda.total)
    assert torch.equal(without.h4, with_lambda.h4)
    enabled = planner_loss(
        outputs,
        gold,
        h4_boundary_lambda_start=1.0,
        h4_boundary_lambda_end=1.0,
    )
    assert enabled.h4 > without.h4


def _assert_complement(logits: torch.Tensor, mask: torch.Tensor) -> None:
    log_p_end, log_p_not_end = h4_end_log_terms(logits, mask)
    total = torch.exp(log_p_end) + torch.exp(log_p_not_end)
    assert torch.isfinite(log_p_end[mask]).all()
    assert torch.isfinite(log_p_not_end[mask]).all()
    assert torch.allclose(total[mask], torch.ones_like(total[mask]), atol=1e-5, rtol=1e-5)


def test_end_log_terms_are_complementary():
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 5, 3, generator=generator)
    mask = torch.tensor(
        [
            [False, True, False, True, True],
            [True, True, False, False, True],
        ]
    )
    _assert_complement(logits, mask)
    for first, second in (
        (BIO_B, BIO_O),
        (BIO_B, BIO_B),
        (BIO_I, BIO_O),
        (BIO_I, BIO_B),
        (BIO_B, BIO_I),
        (BIO_I, BIO_I),
        (BIO_O, BIO_B),
    ):
        peaked = torch.stack([_peaked(first), _peaked(second)]).unsqueeze(0)
        content = torch.tensor([[True, True]])
        _assert_complement(peaked, content)
        single = _peaked(first).unsqueeze(0).unsqueeze(0)
        _assert_complement(single, torch.tensor([[True]]))


def test_extreme_wrong_not_end_has_finite_nonzero_gradient():
    """Gold is a continuation, while logits are an extreme B->O end."""
    logits = torch.tensor(
        [[[-80.0, 80.0, -80.0], [80.0, -80.0, -80.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[BIO_B, BIO_I]])
    mask = torch.tensor([[True, True]])
    loss = h4_end_loss(logits, labels, mask)
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 1.0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[0, 0].abs().sum()) > 0.0

    encoder = _encoder()
    model = PlannerModel(encoder, hidden_size=4)
    features = model.encode(["a bb"])
    views = encoder.token_char_spans_for_batch(features)
    length = features.input_ids.shape[1]
    content = torch.tensor([token.is_content for token in views[0]]).nonzero(
        as_tuple=False
    ).flatten()
    assert content.numel() == 2
    id0 = float(features.input_ids[0, content[0]])
    id1 = float(features.input_ids[0, content[1]])
    wanted = (
        (-80.0, 80.0),
        (80.0, -80.0),
        (-80.0, -80.0),
    )
    weight = model.anc_bio_head.weight
    bias = model.anc_bio_head.bias
    assert weight is not None and bias is not None
    with torch.no_grad():
        weight.zero_()
        for row, (score0, score1) in enumerate(wanted):
            slope = (score0 - score1) / (id0 - id1)
            weight[row, 0] = slope
            bias[row] = score0 - slope * id0
    labels = torch.full((1, length), BIO_O, dtype=torch.long)
    labels[0, content[0]] = BIO_B
    labels[0, content[1]] = BIO_I
    mask = torch.tensor([[token.is_content for token in views[0]]])
    gold = GoldStructureBatch(
        query_type_labels=torch.tensor([0]),
        op_bio_labels=torch.zeros(1, length, dtype=torch.long),
        anc_bio_labels=labels,
        token_loss_mask=mask,
        op_span_mask=torch.zeros(1, 0, length, dtype=torch.bool),
        op_valid=torch.zeros(1, 0, dtype=torch.bool),
        op_type_labels=torch.zeros(1, 0, dtype=torch.long),
        anc_span_mask=torch.zeros(1, 0, length, dtype=torch.bool),
        anc_valid=torch.zeros(1, 0, dtype=torch.bool),
        impl_labels=torch.zeros(1, 0, dtype=torch.long),
        own_labels=torch.zeros(1, 0, dtype=torch.long),
        own_mask=torch.zeros(1, 0, 0, dtype=torch.bool),
        dep_labels=torch.zeros(1, 0, 0),
        dep_mask=torch.zeros(1, 0, 0, dtype=torch.bool),
        example_ids=("wrong-end",),
    )
    outputs = model.forward_train(features, gold)
    produced = outputs.anc_bio_logits[0, content]
    expected = torch.tensor([[-80.0, 80.0, -80.0], [80.0, -80.0, -80.0]])
    assert torch.allclose(produced, expected, atol=1e-3)
    head_loss = h4_end_loss(
        outputs.anc_bio_logits,
        gold.anc_bio_labels,
        gold.token_loss_mask,
    )
    assert torch.isfinite(head_loss)
    head_loss.backward()
    assert model.anc_bio_head.weight.grad is not None
    assert torch.isfinite(model.anc_bio_head.weight.grad).all()
    assert float(model.anc_bio_head.weight.grad.abs().sum()) > 0.0


def test_extreme_logits_stay_finite():
    huge = 80.0
    pairs = (
        (BIO_B, BIO_O),
        (BIO_B, BIO_B),
        (BIO_I, BIO_O),
        (BIO_I, BIO_B),
        (BIO_B, BIO_I),
        (BIO_I, BIO_I),
        (BIO_O, BIO_I),
    )
    for first, second in pairs:
        logits = torch.stack(
            [_peaked(first, huge), _peaked(second, huge)]
        ).unsqueeze(0)
        mask = torch.tensor([[True, True]])
        labels = torch.tensor([[first, second]])
        assert torch.isfinite(h4_end_log_probability(logits, mask)).all()
        assert torch.isfinite(h4_end_loss(logits, labels, mask))
        assert torch.isfinite(h4_start_loss(logits, labels, mask))
        assert torch.isfinite(h4_start_logits(logits)).all()


def test_negative_lambda_rejected():
    outputs, gold = _gold_and_outputs([BIO_O])
    with pytest.raises(ValueError, match="h4_boundary_lambda_start"):
        planner_loss(outputs, gold, h4_boundary_lambda_start=-0.1)
    with pytest.raises(ValueError, match="h4_boundary_lambda_end"):
        TrainConfig(h4_boundary_lambda_end=-1.0)


def test_cli_boundary_lambdas_default_to_zero():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import train_planner as cli

    args = cli.build_parser().parse_args([])
    assert args.h4_boundary_lambda_start == 0.0
    assert args.h4_boundary_lambda_end == 0.0
    _expected, config = cli._resolve_cli_defaults(args)
    assert config.h4_boundary_lambda_start == 0.0
    assert config.h4_boundary_lambda_end == 0.0
    with pytest.raises(SystemExit, match="h4_boundary_lambda_start"):
        cli._parse_boundary_lambda(-0.2, name="h4_boundary_lambda_start")


def test_champion_checkpoint_strict_loads():
    if not CHAMPION.is_file():
        pytest.skip("champion checkpoint is not present")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    payload = load_checkpoint(CHAMPION, map_location="cpu")
    config = config_from_checkpoint(payload, device="cpu")
    assert config.h4_boundary_lambda_start == 0.0
    assert config.h4_boundary_lambda_end == 0.0
    model = build_model(config)
    _ = model.encode(["warmup"])
    missing = model.load_state_dict(payload["model_head_state_dict"], strict=True)
    assert not missing.missing_keys
    assert not missing.unexpected_keys
