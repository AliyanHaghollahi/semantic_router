"""Tests for optional H2/H4 class-weighted BIO cross-entropy (Experiment B)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tiergraph.planner.align import BIO_B, BIO_I, BIO_IGNORE, BIO_O
from tiergraph.planner.annotations import PlannerExample
from tiergraph.planner.batching import collate_gold_structure_batch
from tiergraph.planner.encoder import MiniLMFeatureEncoder
from tiergraph.planner.loss import (
    BIO_CLASS_WEIGHT_LABELS,
    planner_loss,
    validate_bio_class_weights,
)
from tiergraph.planner.model import PlannerModel
from tiergraph.planner.targets import build_planner_targets
from tiergraph.planner.train import (
    BIO_CLASS_WEIGHT_FORMULA,
    TrainConfig,
    compute_bio_class_weights_from_examples,
    compute_h2_h4_bio_class_weights_from_examples,
    count_supervised_bio_labels,
    evaluate_examples,
    inverse_frequency_bio_weights,
    train_step,
)


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


def _fixture_example() -> PlannerExample:
    return PlannerExample.model_validate(
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    )


def _fixture_gold_and_outputs():
    encoder = _encoder()
    model = PlannerModel(encoder, hidden_size=4)
    features = model.encode(["Where is my gate?"])
    views = encoder.token_char_spans_for_batch(features)
    example = _fixture_example()
    targets = (build_planner_targets(example, views[0]),)
    gold = collate_gold_structure_batch(
        features=features,
        targets=targets,
        token_views=views,
    )
    outputs = model.forward_train(features, gold)
    return model, gold, outputs


def test_validate_bio_class_weights_none_and_order():
    assert validate_bio_class_weights(None) is None
    assert BIO_CLASS_WEIGHT_LABELS == ("O", "B", "I")
    assert validate_bio_class_weights((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    "bad",
    [
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
        (1.0, 0.0, 1.0),
        (-1.0, 1.0, 1.0),
        (1.0, float("nan"), 1.0),
    ],
)
def test_invalid_bio_class_weights_rejected(bad):
    with pytest.raises(ValueError):
        validate_bio_class_weights(bad, name="h2_bio_class_weights")


def test_none_weights_match_unweighted_ce():
    _model, gold, outputs = _fixture_gold_and_outputs()
    baseline = planner_loss(outputs, gold)
    explicit_none = planner_loss(
        outputs,
        gold,
        h2_bio_class_weights=None,
        h4_bio_class_weights=None,
    )
    assert torch.allclose(baseline.total, explicit_none.total)
    assert torch.allclose(baseline.h2, explicit_none.h2)
    assert torch.allclose(baseline.h4, explicit_none.h4)

    # Direct CE over the same masked labels must match when weight is omitted.
    flat_logits = outputs.op_bio_logits.reshape(-1, 3)
    safe = gold.op_bio_labels.reshape(-1).clone()
    safe = safe.masked_fill(~gold.token_loss_mask.reshape(-1), BIO_IGNORE)
    manual = F.cross_entropy(flat_logits, safe, ignore_index=BIO_IGNORE)
    assert torch.allclose(baseline.h2, manual)


def test_h2_only_weighting_leaves_h4_unweighted():
    _model, gold, outputs = _fixture_gold_and_outputs()
    baseline = planner_loss(outputs, gold)
    weighted = planner_loss(
        outputs,
        gold,
        h2_bio_class_weights=(2.0, 1.0, 0.5),
        h4_bio_class_weights=None,
    )
    assert not torch.allclose(baseline.h2, weighted.h2)
    assert torch.allclose(baseline.h4, weighted.h4)
    assert abs(weighted.total.item() - (
        weighted.h1 + weighted.h2 + weighted.h3 + weighted.h4
        + weighted.h5 + weighted.h6 + weighted.h7
    ).item()) < 1e-5


def test_h4_only_weighting_leaves_h2_unweighted():
    _model, gold, outputs = _fixture_gold_and_outputs()
    baseline = planner_loss(outputs, gold)
    weighted = planner_loss(
        outputs,
        gold,
        h2_bio_class_weights=None,
        h4_bio_class_weights=(0.5, 2.0, 1.5),
    )
    assert torch.allclose(baseline.h2, weighted.h2)
    assert not torch.allclose(baseline.h4, weighted.h4)


def test_weighted_ce_matches_torch_cross_entropy_weight_arg():
    _model, gold, outputs = _fixture_gold_and_outputs()
    weights = (2.0, 1.0, 0.5)
    breakdown = planner_loss(outputs, gold, h2_bio_class_weights=weights)
    weight_tensor = torch.tensor(weights, dtype=outputs.op_bio_logits.dtype)
    flat_logits = outputs.op_bio_logits.reshape(-1, 3)
    safe = gold.op_bio_labels.reshape(-1).clone()
    safe = safe.masked_fill(~gold.token_loss_mask.reshape(-1), BIO_IGNORE)
    expected = F.cross_entropy(
        flat_logits,
        safe,
        weight=weight_tensor,
        ignore_index=BIO_IGNORE,
    )
    assert torch.allclose(breakdown.h2, expected)


def test_train_config_defaults_unweighted():
    config = TrainConfig()
    assert config.h2_bio_class_weights is None
    assert config.h4_bio_class_weights is None
    assert "h2_bio_class_weights" in config.to_dict()
    assert config.to_dict()["h2_bio_class_weights"] is None


def test_train_config_rejects_invalid_weights():
    with pytest.raises(ValueError):
        TrainConfig(h2_bio_class_weights=(1.0, 0.0, 1.0))


def test_count_supervised_bio_labels_ignores_mask_and_ignore():
    labels = torch.tensor([[BIO_O, BIO_B, BIO_I, BIO_IGNORE, BIO_O]])
    mask = torch.tensor([[True, True, True, True, False]])
    counts = count_supervised_bio_labels(labels, mask)
    assert counts == {"O": 1, "B": 1, "I": 1, "N": 3}


def test_inverse_frequency_bio_weights_formula():
    counts = {"O": 2, "B": 1, "I": 1, "N": 4}
    weights = inverse_frequency_bio_weights(counts)
    assert weights == pytest.approx((4 / (3 * 2), 4 / (3 * 1), 4 / (3 * 1)))
    with pytest.raises(ValueError):
        inverse_frequency_bio_weights({"O": 0, "B": 1, "I": 1, "N": 2})


def test_compute_weights_from_examples_uses_only_supplied_examples(monkeypatch):
    model = PlannerModel(_encoder(), hidden_size=4)
    example = _fixture_example()

    labels = torch.tensor([[BIO_O, BIO_B, BIO_I, BIO_IGNORE]])
    mask = torch.tensor([[True, True, True, False]])
    gold = SimpleNamespace(
        op_bio_labels=labels,
        anc_bio_labels=torch.tensor([[BIO_O, BIO_O, BIO_B, BIO_IGNORE]]),
        token_loss_mask=mask,
    )

    def _fake_encode(_model, examples):
        assert list(examples) == [example]
        return None, None, gold

    monkeypatch.setattr(
        "tiergraph.planner.train.encode_gold_batch",
        _fake_encode,
    )
    result = compute_bio_class_weights_from_examples(
        [example],
        model=model,
        head="h2",
        batch_size=1,
    )
    assert result.head == "h2"
    assert result.formula == BIO_CLASS_WEIGHT_FORMULA
    assert result.counts == {"O": 1, "B": 1, "I": 1, "N": 3}
    assert result.weights == inverse_frequency_bio_weights(result.counts)

    # Helper source must not load DEV/TEST (or any split) itself.
    source = inspect.getsource(compute_bio_class_weights_from_examples)
    assert "load_and_split" not in source
    assert "split.test" not in source
    assert "split.dev" not in source
    dual_source = inspect.getsource(compute_h2_h4_bio_class_weights_from_examples)
    assert "load_and_split" not in dual_source


def test_compute_h2_h4_independent_results(monkeypatch):
    model = PlannerModel(_encoder(), hidden_size=4)
    example = _fixture_example()
    gold = SimpleNamespace(
        op_bio_labels=torch.tensor([[BIO_O, BIO_B, BIO_I, BIO_I]]),
        anc_bio_labels=torch.tensor([[BIO_O, BIO_O, BIO_B, BIO_I]]),
        token_loss_mask=torch.tensor([[True, True, True, True]]),
    )
    monkeypatch.setattr(
        "tiergraph.planner.train.encode_gold_batch",
        lambda _model, _examples: (None, None, gold),
    )
    h2, h4 = compute_h2_h4_bio_class_weights_from_examples(
        [example],
        model=model,
        batch_size=1,
    )
    assert h2.head == "h2"
    assert h4.head == "h4"
    assert h2.counts == {"O": 1, "B": 1, "I": 2, "N": 4}
    assert h4.counts == {"O": 2, "B": 1, "I": 1, "N": 4}
    assert h2.weights == inverse_frequency_bio_weights(h2.counts)
    assert h4.weights == inverse_frequency_bio_weights(h4.counts)
    assert h2.weights != h4.weights


def test_weighted_train_step_passes_weights_to_planner_loss(monkeypatch):
    from tiergraph.planner.train import build_optimizer

    model = PlannerModel(_encoder(), hidden_size=4)
    _ = model.encode(["Where is my gate?"])
    optimizer = build_optimizer(model, lr=1e-3)
    example = _fixture_example()
    captured: dict[str, object] = {}

    real_planner_loss = planner_loss

    def _spy(outputs, gold, **kwargs):
        captured["h2"] = kwargs.get("h2_bio_class_weights")
        captured["h4"] = kwargs.get("h4_bio_class_weights")
        return real_planner_loss(outputs, gold, **kwargs)

    monkeypatch.setattr("tiergraph.planner.train.planner_loss", _spy)
    train_step(
        model,
        optimizer,
        (example,),
        h2_bio_class_weights=(2.0, 1.0, 0.5),
        h4_bio_class_weights=(0.5, 2.0, 1.0),
    )
    assert captured["h2"] == (2.0, 1.0, 0.5)
    assert captured["h4"] == (0.5, 2.0, 1.0)


def test_dev_evaluate_examples_stays_unweighted_when_train_weights_set():
    """Selection-path evaluate_examples must omit weights even if API allows them."""
    model = PlannerModel(_encoder(), hidden_size=4)
    example = _fixture_example()
    unweighted = evaluate_examples(
        model,
        (example,),
        batch_size=1,
        seed=0,
    )
    # Explicit weights on evaluate_examples change mean_loss (API still allows it),
    # proving the selection path must omit them.
    weighted = evaluate_examples(
        model,
        (example,),
        batch_size=1,
        seed=0,
        h2_bio_class_weights=(3.0, 1.0, 0.25),
        h4_bio_class_weights=(0.25, 3.0, 1.0),
    )
    assert unweighted.mean_loss["total"] != weighted.mean_loss["total"]
    # Span F1 is prediction-based and must not depend on CE weights.
    assert unweighted.h2_span_f1 == weighted.h2_span_f1
    assert unweighted.h4_span_f1 == weighted.h4_span_f1


def test_run_training_dev_selection_omits_weights(monkeypatch):
    """best.pt selection calls evaluate_examples without class weights."""
    import tiergraph.planner.train as train_mod

    source = inspect.getsource(train_mod.run_training)
    # The DEV evaluate_examples call must not pass training weights.
    assert "h2_bio_class_weights=config.h2_bio_class_weights" in source
    # That assignment appears only for train_step, not evaluate_examples.
    train_step_block = source.split("dev_metrics = evaluate_examples")[0]
    eval_block = source.split("dev_metrics = evaluate_examples")[1].split(
        "epoch_record"
    )[0]
    assert "h2_bio_class_weights=config.h2_bio_class_weights" in train_step_block
    assert "h4_bio_class_weights=config.h4_bio_class_weights" in train_step_block
    assert "h2_bio_class_weights" not in eval_block
    assert "h4_bio_class_weights" not in eval_block
    assert "dev_metrics.mean_loss[\"total\"] < best_dev_loss" in source


def test_ensure_bio_class_weight_report_records_counts_and_final_weights(monkeypatch):
    from tiergraph.planner.train import _ensure_bio_class_weight_report

    model = PlannerModel(_encoder(), hidden_size=4)
    example = _fixture_example()
    config = TrainConfig(
        h2_bio_class_weights=(2.0, 1.0, 0.5),
        h4_bio_class_weights=(0.5, 2.0, 1.0),
        batch_size=1,
    )
    gold = SimpleNamespace(
        op_bio_labels=torch.tensor([[BIO_O, BIO_B, BIO_I]]),
        anc_bio_labels=torch.tensor([[BIO_O, BIO_B, BIO_I]]),
        token_loss_mask=torch.tensor([[True, True, True]]),
    )
    monkeypatch.setattr(
        "tiergraph.planner.train.encode_gold_batch",
        lambda _model, _examples: (None, None, gold),
    )
    report = _ensure_bio_class_weight_report(
        None,
        train_examples=[example],
        model=model,
        config=config,
    )
    assert report["h2_train_bio_counts"] == {"O": 1, "B": 1, "I": 1, "N": 3}
    assert report["h4_train_bio_counts"] == {"O": 1, "B": 1, "I": 1, "N": 3}
    assert report["h2_bio_class_weights"] == [2.0, 1.0, 0.5]
    assert report["h4_bio_class_weights"] == [0.5, 2.0, 1.0]
    # Explicit weights are preserved even when counts would imply different w_c.
    assert report["h2_bio_class_weights"] != list(
        inverse_frequency_bio_weights(report["h2_train_bio_counts"])
    )
