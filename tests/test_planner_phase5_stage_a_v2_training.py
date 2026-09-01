"""Focused tests for Stage-A v2 planner training wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tiergraph.planner.annotation_step_a import fingerprint_file
from tiergraph.planner.encoder import MiniLMFeatureEncoder
from tiergraph.planner.model import PlannerModel
from tiergraph.planner.stage_a_split import StageASplitResult
from tiergraph.planner.stage_a_v2_spec import (
    STAGE_A_V2_CORPUS_SIZE,
    STAGE_A_V2_DEV_SIZE,
    STAGE_A_V2_SPLIT_FINGERPRINT,
    STAGE_A_V2_STEP_A_PATH,
    STAGE_A_V2_STEP_B_PATH,
    STAGE_A_V2_TEST_SIZE,
    STAGE_A_V2_TRAIN_SIZE,
)
from tiergraph.planner.train import (
    TrainConfig,
    assert_encoder_frozen,
    build_optimizer,
    load_and_split_stage_a_v2,
    run_training,
    train_step,
)

from tests.test_planner_phase5_training import _make_model, _tiny_split


ROOT = Path(__file__).resolve().parent.parent
STEP_A_V2_PATH = ROOT / STAGE_A_V2_STEP_A_PATH
STEP_B_V2_PATH = ROOT / STAGE_A_V2_STEP_B_PATH


def test_load_and_split_stage_a_v2_frozen_assignment():
    step_a_before = fingerprint_file(STEP_A_V2_PATH)
    step_b_before = fingerprint_file(STEP_B_V2_PATH)
    config = TrainConfig(
        corpus_version="v2",
        step_a_path=str(STEP_A_V2_PATH),
        step_b_path=str(STEP_B_V2_PATH),
    )
    split, after_a, after_b = load_and_split_stage_a_v2(config)
    assert after_a == step_a_before
    assert after_b == step_b_before
    assert len(split.train) == STAGE_A_V2_TRAIN_SIZE
    assert len(split.dev) == STAGE_A_V2_DEV_SIZE
    assert len(split.test) == STAGE_A_V2_TEST_SIZE
    assert (
        len(split.train) + len(split.dev) + len(split.test) == STAGE_A_V2_CORPUS_SIZE
    )
    assert split.fingerprint == STAGE_A_V2_SPLIT_FINGERPRINT
    assert split.report["semantic_group_leakage"] == {}
    assert split.report["quarantined_in_test"] == []


def test_v2_smoke_training_preserves_annotations(tmp_path: Path):
    step_a_before = fingerprint_file(STEP_A_V2_PATH)
    step_b_before = fingerprint_file(STEP_B_V2_PATH)
    split, _, _ = load_and_split_stage_a_v2(
        TrainConfig(
            corpus_version="v2",
            step_a_path=str(STEP_A_V2_PATH),
            step_b_path=str(STEP_B_V2_PATH),
        )
    )
    tiny = StageASplitResult(
        train=split.train[:8],
        dev=split.dev[:4],
        test=split.test[:4],
        seed=split.seed,
        fingerprint=split.fingerprint,
        report=split.report,
    )
    model, _ = _make_model()
    config = TrainConfig(
        seed=split.seed,
        epochs=1,
        batch_size=4,
        lr=1e-2,
        device="cpu",
        output_dir=str(tmp_path / "v2_smoke"),
        smoke=True,
        smoke_train_batches=1,
        smoke_eval_batches=1,
        corpus_version="v2",
        step_a_path=str(STEP_A_V2_PATH),
        step_b_path=str(STEP_B_V2_PATH),
    )
    result = run_training(
        config,
        model=model,
        split=tiny,
        annotation_fingerprints=(step_a_before, step_b_before),
    )
    assert result.split_fingerprint == STAGE_A_V2_SPLIT_FINGERPRINT
    assert result.smoke_first_loss is not None
    assert torch.isfinite(torch.tensor(result.smoke_first_loss))
    assert fingerprint_file(STEP_A_V2_PATH) == step_a_before
    assert fingerprint_file(STEP_B_V2_PATH) == step_b_before


def test_v2_real_example_train_step_finite():
    split, _, _ = load_and_split_stage_a_v2(
        TrainConfig(
            corpus_version="v2",
            step_a_path=str(STEP_A_V2_PATH),
            step_b_path=str(STEP_B_V2_PATH),
        )
    )
    model, _ = _make_model(hidden_size=8)
    optimizer = build_optimizer(model, lr=1e-2)
    breakdown = train_step(model, optimizer, (split.train[0],))
    assert torch.isfinite(breakdown.total)
    assert_encoder_frozen(model)
