"""Sanity tests for Stage-A v2 experiment metric reporting."""

from __future__ import annotations

from pathlib import Path

from tiergraph.planner.stage_a_to_corpus import final_bucket_to_classification_label
from tiergraph.planner.stage_a_v2_spec import STAGE_A_V2_STEP_A_PATH, STAGE_A_V2_STEP_B_PATH
from tiergraph.planner.train import TrainConfig, load_and_split_stage_a_v2
from tiergraph.planner.v2_baselines import evaluate_bucket_oracle_h1


ROOT = Path(__file__).resolve().parent.parent


def test_bucket_oracle_h1_is_one_on_canonical_labels():
    split, _, _ = load_and_split_stage_a_v2(
        TrainConfig(
            corpus_version="v2",
            step_a_path=str(ROOT / STAGE_A_V2_STEP_A_PATH),
            step_b_path=str(ROOT / STAGE_A_V2_STEP_B_PATH),
        )
    )
    result = evaluate_bucket_oracle_h1(split.test)
    assert result.n_examples == 48
    assert result.h1_accuracy == 1.0
    assert result.gold_label == "canonical_h1_classification_label"
    assert result.baseline_role == "canonical_h1_sanity_oracle"
    assert result.comparable_metrics == ()


def test_classifier_baselines_use_tiergraph_h1_gold():
    split, _, _ = load_and_split_stage_a_v2(
        TrainConfig(
            corpus_version="v2",
            step_a_path=str(ROOT / STAGE_A_V2_STEP_A_PATH),
            step_b_path=str(ROOT / STAGE_A_V2_STEP_B_PATH),
        )
    )
    for example in split.test:
        assert (
            example.planner_labels.query_type.value
            == example.graph.query_type.value
        )


def test_test_set_has_decoded_graph_h1_mismatch_vs_bucket():
    """Six Mixed-bucket rows decode Environmental H1; oracle must not use that gold."""
    split, _, _ = load_and_split_stage_a_v2(
        TrainConfig(
            corpus_version="v2",
            step_a_path=str(ROOT / STAGE_A_V2_STEP_A_PATH),
            step_b_path=str(ROOT / STAGE_A_V2_STEP_B_PATH),
        )
    )
    mismatches = 0
    for example in split.test:
        canonical = final_bucket_to_classification_label(
            str(example.metadata["final_bucket"])
        )
        decoded = example.graph.query_type.value
        if canonical != decoded:
            mismatches += 1
    assert mismatches == 6
    assert round(1.0 - mismatches / len(split.test), 3) == 0.875
