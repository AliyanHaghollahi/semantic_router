"""Focused checks for the TRAIN-only H4.2 fold comparison driver.

These tests do not train a model and do not read DEV or TEST annotation bodies.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "run_planner_h42_cv",
    ROOT / "scripts" / "run_planner_h42_cv.py",
)
assert _SPEC is not None and _SPEC.loader is not None
h42 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(h42)

FOLD_FINGERPRINT = (
    "87d6bc26f2685a0abe10a5869405167f963245c0d5ccc3944b18efd55f4ff757"
)
FORBIDDEN_CALLS = {
    "run_training",
    "evaluate_examples",
    "evaluate_checkpoint",
    "load_and_split_stage_a_v3",
    "load_checkpoint",
}


def _jobs() -> tuple[dict, list[dict]]:
    split_ids = h42.h4cv.load_split_ids(ROOT / h42.h4cv.STAGE_A_V3_SPLIT_PATH)
    payload = h42.h4cv.load_fold_manifest(h42.DEFAULT_MANIFEST)
    mapping = h42.h4cv.validate_fold_manifest(payload, split_ids)
    return split_ids, h42.fold_jobs(mapping, split_ids)


def _call_names(node: ast.AST) -> list[str]:
    names = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
        if not isinstance(child, ast.FunctionDef):
            names.extend(_call_names(child))
    return names


def _function(name: str) -> ast.FunctionDef:
    source = Path(h42.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(found) == 1
    return found[0]


def test_protocol_matches_frozen_h41_constants():
    assert h42.CV_FOLD_FINGERPRINT == FOLD_FINGERPRINT
    assert h42.EPOCHS == 5
    assert h42.BATCH_SIZE == 8
    assert h42.LR == 0.001
    assert h42.DEVICE == "cpu"
    assert h42.DECODE == "viterbi"
    assert h42.FINAL_EPOCH_ONLY is True
    assert h42.BASE_SEED == 20260901
    assert (h42.LAMBDA_START, h42.LAMBDA_END) == (0.0, 0.25)
    assert [h42.fold_seed(fold) for fold in range(5)] == [
        20260901,
        20260902,
        20260903,
        20260904,
        20260905,
    ]


def test_five_folds_cover_train_once_without_overlap():
    split_ids, jobs = _jobs()
    assert len(jobs) == 5
    assert [job["n_holdout"] for job in jobs] == [77, 77, 77, 77, 76]
    covered: list[str] = []
    for job in jobs:
        train_ids = set(job["train_ids"])
        eval_ids = set(job["eval_ids"])
        assert train_ids.isdisjoint(eval_ids)
        assert train_ids <= split_ids["train"]
        assert eval_ids <= split_ids["train"]
        assert train_ids.isdisjoint(split_ids["dev"])
        assert eval_ids.isdisjoint(split_ids["test"])
        assert job["n_train"] == 384 - job["n_holdout"]
        assert job["train_ids_hash"] == h42.id_hash(job["train_ids"])
        covered.extend(job["eval_ids"])
    assert len(covered) == len(set(covered)) == 384


def test_contamination_is_rejected():
    split_ids, jobs = _jobs()
    mapping = {}
    for job in jobs:
        for sid in job["eval_ids"]:
            mapping[sid] = job["fold"]
    dev_id = next(iter(split_ids["dev"]))
    test_id = next(iter(split_ids["test"]))
    contaminated = dict(mapping)
    replaced = jobs[0]["eval_ids"][0]
    del contaminated[replaced]
    contaminated[dev_id] = 0
    with pytest.raises(RuntimeError, match="DEV was requested"):
        h42.fold_jobs(contaminated, split_ids)
    contaminated[test_id] = 0
    del contaminated[dev_id]
    with pytest.raises(RuntimeError, match="TEST was requested"):
        h42.fold_jobs(contaminated, split_ids)
    with pytest.raises(RuntimeError, match="overlaps"):
        h42.assert_train_only_split(
            ["a", "b"],
            ["b"],
            train_universe={"a", "b"},
            dev_ids=set(),
            test_ids=set(),
        )


def test_dry_run_prints_the_plan_and_does_not_train(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    called = {"train": 0}

    def refuse(*_args, **_kwargs):
        called["train"] += 1
        raise AssertionError("dry-run trained a fold")

    monkeypatch.setattr(h42, "execute_fold", refuse)
    monkeypatch.setattr(h42, "load_h41_baseline", lambda path=h42.H41_RESULTS: None)
    output = tmp_path / "planner_h42_cv"
    h42.run(["--dry-run", "--output-dir", str(output)])
    captured = capsys.readouterr().out
    assert "N_JOBS 5" in captured
    assert "FOLD_SIZES [77, 77, 77, 77, 76]" in captured
    assert "SEEDS 0:20260901 1:20260902 2:20260903 3:20260904 4:20260905" in captured
    assert "LAMBDA 0.00 0.25" in captured
    assert "EPOCHS 5" in captured
    assert "BATCH_SIZE 8" in captured
    assert "LR 0.001" in captured
    assert "DECODE viterbi" in captured
    assert "FINAL_EPOCH_ONLY true" in captured
    assert "DEV_USED false" in captured
    assert "TEST_USED false" in captured
    assert "H41_BASELINE missing" in captured
    assert captured.count("\njob ") == 5
    assert called["train"] == 0
    assert not output.exists()
    with pytest.raises(RuntimeError, match="does not fall back to DEV"):
        h42.run(["--output-dir", str(output)])
    assert called["train"] == 0
    assert not output.exists()


def test_training_step_uses_one_encoder_forward_and_fixed_lambda():
    step = _function("train_h42_step")
    text = ast.dump(step)
    assert text.count("encode_gold_batch") == 1
    source = Path(h42.__file__).read_text(encoding="utf-8")
    assert "model.encode(" not in source
    nested = [node for node in step.body if isinstance(node, ast.FunctionDef)]
    assert nested == []
    fold = _function("execute_fold")
    nested_names = [node.name for node in fold.body if isinstance(node, ast.FunctionDef)]
    assert nested_names == ["train_epoch"]
    train_epoch = next(node for node in fold.body if isinstance(node, ast.FunctionDef))
    assert "evaluate" not in ast.dump(train_epoch)
    ordered = _call_names(fold)
    assert ordered.index("run_fold_epochs") < ordered.index("evaluate_holdout")
    source = Path(h42.__file__).read_text(encoding="utf-8")
    assert "enumerate_word_spans(example.query" in source
    assert "build_planner_v31" not in source
    for name in FORBIDDEN_CALLS:
        assert name not in source
    parameters = inspect.signature(h42.free_h42_anchor_spans).parameters
    assert "gold" not in parameters
    assert "gold_anchors" not in parameters


def test_decode_keeps_every_candidate_above_none():
    class _Selector:
        def score(self, token_embeddings, operation_mask, candidate_mask):
            n_candidates = candidate_mask.shape[0]
            n_operations = operation_mask.shape[0]
            logits = torch.full((n_candidates, n_operations), -5.0)
            none = torch.zeros(n_operations)
            logits[0, 0] = 3.0
            logits[1, 0] = 1.0
            none[1] = 4.0
            return logits, none

    query = "my taxi"
    spans = h42.enumerate_word_spans(query)
    selected, chosen = h42.free_h42_anchor_spans(
        _Selector(),
        torch.zeros(1, 4),
        (),
        query,
        [(0, len(query)), (0, len(query))],
    )
    assert len(spans) >= 2
    assert [span.start for span, _owner in selected] == [spans[0].start, spans[1].start]
    assert [owner for _span, owner in selected] == [0, 0]
    assert chosen[1] == ()
    assert h42.exact_anchor_set_match({(0, 2), (3, 7)}, {(3, 7), (0, 2)})
    anchors = h42.anchors_from_selection(query, selected, [0, 1])
    assert len(anchors) == 2
    assert anchors[0].owner_index == 0
    assert anchors[1].implicit_resolution.value == "IMPLICIT_RESOLVE_PERSONAL"


def test_keep_rule_compares_h41_without_dev():
    kept = h42.compare_means(0.40, 0.30, 0.80, 0.80)
    assert kept["h42_worth_keeping"] is True
    assert kept["dev_used_for_decision"] is False
    small = h42.compare_means(0.304, 0.300, 0.80, 0.80)
    assert small["h42_worth_keeping"] is False
    degraded = h42.compare_means(0.40, 0.30, 0.70, 0.80)
    assert degraded["h42_worth_keeping"] is False
    rows = [
        {
            "fold": fold,
            "h4_precision": 0.5,
            "h4_recall": 0.5,
            "h4_f1": 0.5,
            "exact_anchor_set_accuracy": 0.2,
            "operation_joint_f1": 0.4,
            "valid_graph_rate": 0.9,
            "tier_routing_rate": 0.8,
            "semantic_match_rate": 0.1,
        }
        for fold in range(5)
    ]
    baseline = [
        {
            "fold": fold,
            "lambda_start": 0.0,
            "lambda_end": 0.25,
            "h4_f1": 0.40,
            "valid_graph_rate": 0.90,
            "cv_fold_fingerprint": FOLD_FINGERPRINT,
        }
        for fold in range(5)
    ]
    summary = h42.aggregate_against_h41(rows, baseline)
    assert summary["comparison"]["h42_worth_keeping"] is True
    assert summary["DEV_USED"] is False
    assert summary["TEST_USED"] is False
    assert h42.aggregate_against_h41(rows, None)["comparison"] is None


def test_h41_baseline_rejects_a_different_fingerprint(tmp_path: Path):
    path = tmp_path / "per_fold_results.jsonl"
    row = {
        "fold": 0,
        "lambda_start": 0.0,
        "lambda_end": 0.25,
        "h4_f1": 0.1,
        "cv_fold_fingerprint": "a" * 64,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint"):
        h42.load_h41_baseline(path)
    assert h42.load_h41_baseline(tmp_path / "missing.jsonl") is None
