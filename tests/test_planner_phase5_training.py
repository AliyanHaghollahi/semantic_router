"""Focused tests for thin Stage-A planner training entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tiergraph.planner.annotation_step_a import (
    DEFAULT_STEP_A_ANNOTATIONS_PATH,
    fingerprint_file,
)
from tiergraph.planner.annotation_step_b import DEFAULT_STEP_B_ANNOTATIONS_PATH
from tiergraph.planner.annotations import PlannerExample
from tiergraph.planner.encoder import MiniLMFeatureEncoder
from tiergraph.planner.model import PlannerModel
from tiergraph.planner.stage_a_split import StageASplitResult
from tiergraph.planner.train import (
    TrainConfig,
    assert_encoder_frozen,
    build_optimizer,
    encode_gold_batch,
    encoder_parameters,
    evaluate_examples,
    head_parameters,
    iter_example_batches,
    run_training,
    set_seed,
    train_step,
)


ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "planner" / "where_is_my_gate.json"
)
STEP_A_PATH = ROOT / DEFAULT_STEP_A_ANNOTATIONS_PATH
STEP_B_PATH = ROOT / DEFAULT_STEP_B_ANNOTATIONS_PATH


class _CountingTokenizer:
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
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = offset_mapping
        return output


class _CountingTransformer(torch.nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_ids, attention_mask, **kwargs):
        hidden = self.config.hidden_size
        values = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        token_embeddings = values.repeat(1, 1, hidden) * self.scale
        return SimpleNamespace(last_hidden_state=token_embeddings)


def _make_encoder(hidden_size: int = 8) -> tuple[MiniLMFeatureEncoder, _CountingTransformer]:
    tokenizer = _CountingTokenizer()
    transformer = _CountingTransformer(hidden_size=hidden_size)
    encoder = MiniLMFeatureEncoder(
        max_length=32,
        tokenizer_loader=lambda _name: tokenizer,
        model_loader=lambda _name: transformer,
    )
    return encoder, transformer


def _load_fixture() -> PlannerExample:
    return PlannerExample.model_validate(
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    )


def _clone_example(example: PlannerExample, example_id: str) -> PlannerExample:
    payload = example.model_dump(mode="json")
    payload["example_id"] = example_id
    payload["graph"]["graph_id"] = example_id
    return PlannerExample.model_validate(payload)


def _tiny_split(seed: int = 7) -> StageASplitResult:
    base = _load_fixture()
    train = tuple(_clone_example(base, f"train-{index}") for index in range(4))
    dev = tuple(_clone_example(base, f"dev-{index}") for index in range(2))
    test = tuple(_clone_example(base, f"test-{index}") for index in range(2))
    assignment = {
        **{ex.example_id: "train" for ex in train},
        **{ex.example_id: "dev" for ex in dev},
        **{ex.example_id: "test" for ex in test},
    }
    return StageASplitResult(
        train=train,
        dev=dev,
        test=test,
        seed=seed,
        fingerprint="unit-test-fingerprint",
        report={
            "sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
            "example_to_split": assignment,
            "semantic_group_leakage": [],
        },
    )


def _make_model(hidden_size: int = 8) -> tuple[PlannerModel, _CountingTransformer]:
    encoder, transformer = _make_encoder(hidden_size)
    model = PlannerModel(encoder, hidden_size=hidden_size)
    _ = model.encode(["Where is my gate?"])
    assert_encoder_frozen(model)
    return model, transformer


def test_seed_reproducibility_for_batches_and_config():
    examples = _tiny_split().train
    set_seed(123)
    first = iter_example_batches(
        examples, batch_size=2, seed=123, epoch=0, shuffle=True
    )
    set_seed(123)
    second = iter_example_batches(
        examples, batch_size=2, seed=123, epoch=0, shuffle=True
    )
    assert [[ex.example_id for ex in batch] for batch in first] == [
        [ex.example_id for ex in batch] for batch in second
    ]
    set_seed(999)
    third = iter_example_batches(
        examples, batch_size=2, seed=999, epoch=0, shuffle=True
    )
    assert [[ex.example_id for ex in batch] for batch in third] != [
        [ex.example_id for ex in batch] for batch in first
    ]
    config = TrainConfig(seed=123, epochs=1, batch_size=2, smoke=True)
    assert config.to_dict()["seed"] == 123
    assert config.to_dict()["smoke"] is True


def test_optimizer_includes_heads_excludes_encoder():
    model, transformer = _make_model()
    optimizer = build_optimizer(model, lr=1e-3)
    opt_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    head_ids = {id(parameter) for parameter in head_parameters(model)}
    encoder_ids = {id(parameter) for parameter in encoder_parameters(model)}
    assert head_ids
    assert opt_ids == head_ids
    assert encoder_ids
    assert opt_ids.isdisjoint(encoder_ids)
    assert all(not parameter.requires_grad for parameter in transformer.parameters())


def test_one_train_step_finite_loss_and_head_updates():
    model, transformer = _make_model()
    optimizer = build_optimizer(model, lr=1e-2)
    before_heads = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    before_encoder = {
        name: parameter.detach().clone()
        for name, parameter in transformer.named_parameters()
    }
    example = _load_fixture()
    breakdown = train_step(model, optimizer, (example,))
    assert torch.isfinite(breakdown.total)
    assert all(
        torch.isfinite(getattr(breakdown, key))
        for key in ("h1", "h2", "h3", "h4", "h5", "h6", "h7")
    )
    changed = False
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not torch.allclose(parameter, before_heads[name]):
            changed = True
            break
    assert changed
    for name, parameter in transformer.named_parameters():
        assert torch.allclose(parameter, before_encoder[name])
        assert parameter.grad is None


def test_encoder_unchanged_and_no_gradients():
    model, transformer = _make_model()
    optimizer = build_optimizer(model, lr=1e-2)
    before = transformer.scale.detach().clone()
    train_step(model, optimizer, (_load_fixture(),))
    assert torch.equal(transformer.scale, before)
    assert transformer.scale.grad is None
    assert_encoder_frozen(model)


def test_evaluation_metrics_on_tiny_batch():
    model, _ = _make_model()
    examples = (_load_fixture(), _clone_example(_load_fixture(), "eval-2"))
    metrics = evaluate_examples(
        model,
        examples,
        batch_size=2,
        seed=0,
        max_batches=1,
    )
    assert metrics.n_examples == 2
    assert metrics.n_batches == 1
    assert torch.isfinite(torch.tensor(metrics.mean_loss["total"]))
    for key in ("h1", "h2", "h3", "h4", "h5", "h6", "h7"):
        assert key in metrics.mean_loss
        assert metrics.mean_loss[key] >= 0.0
    assert 0.0 <= metrics.h1_accuracy <= 1.0
    assert 0.0 <= metrics.h3_operator_accuracy <= 1.0
    assert 0.0 <= metrics.h5_accuracy <= 1.0
    assert 0.0 <= metrics.h6_ownership_accuracy <= 1.0
    assert 0.0 <= metrics.h7_f1 <= 1.0
    payload = metrics.to_dict()
    assert "not exact predicted-graph accuracy" in payload["note"]


def test_smoke_entrypoint_avoids_dataset_writes(tmp_path: Path):
    step_a_before = fingerprint_file(STEP_A_PATH)
    step_b_before = fingerprint_file(STEP_B_PATH)
    step_a_bytes = STEP_A_PATH.read_bytes()
    step_b_bytes = STEP_B_PATH.read_bytes()
    dataset_dir = ROOT / "dataset"
    before_dataset = {
        path: path.stat().st_mtime_ns
        for path in dataset_dir.rglob("*")
        if path.is_file()
    }

    model, _ = _make_model()
    split = _tiny_split(seed=11)
    output_dir = tmp_path / "smoke_out"
    config = TrainConfig(
        seed=11,
        epochs=1,
        batch_size=2,
        lr=1e-2,
        device="cpu",
        output_dir=str(output_dir),
        smoke=True,
        smoke_train_batches=1,
        smoke_eval_batches=1,
        step_a_path=str(STEP_A_PATH),
        step_b_path=str(STEP_B_PATH),
    )
    result = run_training(
        config,
        model=model,
        split=split,
        annotation_fingerprints=(step_a_before, step_b_before),
    )
    assert result.smoke_first_loss is not None
    assert result.smoke_last_loss is not None
    assert torch.isfinite(torch.tensor(result.smoke_first_loss))
    assert (output_dir / "smoke.pt").is_file()
    assert (output_dir / "train_config.json").is_file()
    assert not str(result.checkpoint_path).startswith(str(dataset_dir))

    after_dataset = {
        path: path.stat().st_mtime_ns
        for path in dataset_dir.rglob("*")
        if path.is_file()
    }
    assert after_dataset == before_dataset
    assert fingerprint_file(STEP_A_PATH) == step_a_before
    assert fingerprint_file(STEP_B_PATH) == step_b_before
    assert STEP_A_PATH.read_bytes() == step_a_bytes
    assert STEP_B_PATH.read_bytes() == step_b_bytes


def test_frozen_annotation_fingerprints_unchanged_with_real_split(tmp_path: Path):
    step_a_before = fingerprint_file(STEP_A_PATH)
    step_b_before = fingerprint_file(STEP_B_PATH)
    model, _ = _make_model()
    split = _tiny_split(seed=20260831)
    config = TrainConfig(
        seed=20260831,
        smoke=True,
        batch_size=2,
        output_dir=str(tmp_path / "out"),
        step_a_path=str(STEP_A_PATH),
        step_b_path=str(STEP_B_PATH),
    )
    run_training(
        config,
        model=model,
        split=split,
        annotation_fingerprints=(step_a_before, step_b_before),
    )
    assert fingerprint_file(STEP_A_PATH) == step_a_before
    assert fingerprint_file(STEP_B_PATH) == step_b_before


def test_encode_gold_batch_single_encoder_forward():
    model, transformer = _make_model()
    calls = {"n": 0}
    original = transformer.forward

    def counted_forward(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    transformer.forward = counted_forward  # type: ignore[method-assign]
    encode_gold_batch(model, (_load_fixture(),))
    assert calls["n"] == 1


def test_cli_help_smoke_flag_present():
    script = ROOT / "scripts" / "train_planner.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--smoke" in completed.stdout
    assert "--seed" in completed.stdout
    assert "--output-dir" in completed.stdout
