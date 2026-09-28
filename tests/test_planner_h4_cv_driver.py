"""Focused checks for the TRAIN-only H4.1 lambda CV driver.

These tests do not train a model and do not read DEV or TEST annotation bodies.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "run_planner_h4_cv",
    ROOT / "scripts" / "run_planner_h4_cv.py",
)
assert _SPEC is not None and _SPEC.loader is not None
cv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cv)

FOLD_FINGERPRINT = (
    "87d6bc26f2685a0abe10a5869405167f963245c0d5ccc3944b18efd55f4ff757"
)
FORBIDDEN_CALLS = {
    "run_training",
    "evaluate_examples",
    "evaluate_checkpoint",
    "load_and_split_stage_a_v3",
}


def _manifest() -> dict:
    return cv.load_fold_manifest(cv.DEFAULT_MANIFEST)


def _validated():
    split_ids = cv.load_split_ids(ROOT / cv.STAGE_A_V3_SPLIT_PATH)
    mapping = cv.validate_fold_manifest(_manifest(), split_ids)
    return split_ids, mapping


def test_frozen_manifest_fingerprint_validation():
    split_ids, mapping = _validated()
    payload = _manifest()
    assert payload["cv_fold_fingerprint"] == FOLD_FINGERPRINT
    assert cv.CV_FOLD_FINGERPRINT == FOLD_FINGERPRINT
    assert len(mapping) == 384
    assert set(mapping) == split_ids["train"]
    assert sorted(set(mapping.values())) == [0, 1, 2, 3, 4]
    sizes = [sum(fold == index for fold in mapping.values()) for index in range(5)]
    assert sizes == [77, 77, 77, 77, 76]
    bad = dict(payload)
    bad["cv_fold_fingerprint"] = "0" * 64
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        cv.validate_fold_manifest(bad, split_ids)


def test_lambda_grid_is_the_preregistered_seven_pairs():
    assert cv.LAMBDA_GRID == (
        (0.00, 0.00),
        (0.25, 0.00),
        (0.00, 0.25),
        (0.25, 0.25),
        (0.50, 0.00),
        (0.00, 0.50),
        (0.50, 0.50),
    )
    assert cv.EPOCHS == 5
    assert cv.BATCH_SIZE == 8
    assert cv.LR == 0.001
    assert cv.BASE_SEED == 20260901
    assert cv.DEVICE == "cpu"
    assert cv.DECODE == "viterbi"
    assert cv.FINAL_EPOCH_ONLY is True


def test_fold_seed_is_independent_of_lambda():
    _split_ids, mapping = _validated()
    jobs = cv.planned_jobs(mapping)
    assert len(jobs) == 35
    by_fold: dict[int, set[int]] = {}
    for job in jobs:
        by_fold.setdefault(job["fold"], set()).add(job["seed"])
        assert job["seed"] == cv.fold_seed(job["fold"])
        assert job["seed"] == cv.BASE_SEED + job["fold"]
    assert by_fold == {fold: {cv.BASE_SEED + fold} for fold in range(5)}
    assert len({job["seed"] for job in jobs if job["fold"] == 2}) == 1
    assert all(
        left["seed"] == right["seed"]
        for left, right in zip(jobs[:5], jobs[5:10])
    )


def test_selection_rule_tie_and_strict_gap():
    clear = cv.select_lambda(
        [
            {"lambda_start": 0.0, "lambda_end": 0.0, "mean_h4_f1": 0.40},
            {"lambda_start": 0.5, "lambda_end": 0.5, "mean_h4_f1": 0.51},
        ]
    )
    assert (clear["lambda_start"], clear["lambda_end"]) == (0.5, 0.5)

    within = cv.select_lambda(
        [
            {"lambda_start": 0.5, "lambda_end": 0.5, "mean_h4_f1": 0.50},
            {"lambda_start": 0.0, "lambda_end": 0.0, "mean_h4_f1": 0.496},
            {"lambda_start": 0.25, "lambda_end": 0.0, "mean_h4_f1": 0.496},
        ]
    )
    assert (within["lambda_start"], within["lambda_end"]) == (0.0, 0.0)
    assert "0.005" in within["selection_reason"]

    same_sum = cv.select_lambda(
        [
            {"lambda_start": 0.0, "lambda_end": 0.25, "mean_h4_f1": 0.50},
            {"lambda_start": 0.25, "lambda_end": 0.0, "mean_h4_f1": 0.50},
        ]
    )
    assert (same_sum["lambda_start"], same_sum["lambda_end"]) == (0.25, 0.0)

    exact_gap = cv.select_lambda(
        [
            {"lambda_start": 0.5, "lambda_end": 0.5, "mean_h4_f1": 0.50},
            {"lambda_start": 0.0, "lambda_end": 0.0, "mean_h4_f1": 0.495},
        ]
    )
    assert (exact_gap["lambda_start"], exact_gap["lambda_end"]) == (0.5, 0.5)


def test_rejects_dev_and_test_contamination():
    split_ids, _mapping = _validated()
    payload = _manifest()
    contaminated = json.loads(json.dumps(payload))
    dev_id = sorted(split_ids["dev"])[0]
    test_id = sorted(split_ids["test"])[0]
    assert dev_id not in split_ids["train"]
    assert test_id not in split_ids["train"]
    contaminated["assignments"][0]["stage_a_id"] = dev_id
    with pytest.raises(RuntimeError, match="DEV or TEST"):
        cv.validate_fold_manifest(contaminated, split_ids)
    contaminated["assignments"][0]["stage_a_id"] = test_id
    with pytest.raises(RuntimeError, match="DEV or TEST"):
        cv.validate_fold_manifest(contaminated, split_ids)


def test_final_epoch_only_and_no_dev_checkpoint_path():
    source = (ROOT / "scripts" / "run_planner_h4_cv.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = set()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.add(func.id)
            elif isinstance(func, ast.Attribute):
                calls.add(func.attr)
    assert not (calls & FORBIDDEN_CALLS)
    assert not (imported & FORBIDDEN_CALLS)
    assert "evaluate_free_examples" in calls
    assert "run_fold_epochs" in calls

    calls_order = []

    def train_epoch(epoch: int) -> None:
        calls_order.append(("train", epoch))

    cv.run_fold_epochs(epochs=5, train_epoch=train_epoch)
    assert calls_order == [("train", epoch) for epoch in range(5)]
    with pytest.raises(ValueError, match="exactly 5"):
        cv.run_fold_epochs(epochs=4, train_epoch=train_epoch)

    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_job"
    )
    top_calls = [
        node.value.func.id
        for node in function.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    ]
    assert "run_fold_epochs" in top_calls
    assert "evaluate" not in top_calls
    nested = next(
        node
        for node in function.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_epoch"
    )
    nested_text = ast.dump(nested)
    assert "evaluate" not in nested_text
    assert "evaluate_examples" not in nested_text


def test_resume_rejects_metadata_mismatch_and_skips_matching_jobs(tmp_path: Path):
    metadata = cv.experiment_metadata()
    (tmp_path / cv.METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    row = {
        "lambda_start": 0.0,
        "lambda_end": 0.0,
        "fold": 0,
        "seed": cv.fold_seed(0),
        "cv_fold_fingerprint": FOLD_FINGERPRINT,
        "h4_f1": 0.1,
        "h4_precision": 0.1,
        "h4_recall": 0.1,
    }
    (tmp_path / cv.PER_FOLD_NAME).write_text(
        json.dumps(row, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completed = cv.load_completed(tmp_path, metadata)
    assert len(completed) == 1
    _split_ids, mapping = _validated()
    remaining = cv.pending_jobs(cv.planned_jobs(mapping), completed)
    assert len(remaining) == 34
    assert (0.0, 0.0, 0) not in {cv.job_key(job) for job in remaining}

    drifted = dict(metadata)
    drifted["cv_fold_fingerprint"] = "f" * 64
    (tmp_path / cv.METADATA_NAME).write_text(
        json.dumps(drifted) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="do not match"):
        cv.load_completed(tmp_path, metadata)

    (tmp_path / cv.METADATA_NAME).write_text(
        json.dumps(metadata) + "\n",
        encoding="utf-8",
    )
    bad_row = dict(row)
    bad_row["cv_fold_fingerprint"] = "a" * 64
    (tmp_path / cv.PER_FOLD_NAME).write_text(
        json.dumps(bad_row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="fingerprint"):
        cv.load_completed(tmp_path, metadata)

    (tmp_path / cv.METADATA_NAME).unlink()
    with pytest.raises(RuntimeError, match="refusing to resume"):
        cv.load_completed(tmp_path, metadata)


def test_dry_run_prints_jobs_and_does_not_train(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    output = tmp_path / "planner_h4_cv"
    cv.run(
        [
            "--dry-run",
            "--output-dir",
            str(output),
        ]
    )
    captured = capsys.readouterr().out
    assert "N_JOBS 35" in captured
    assert "FOLD_SIZES [77, 77, 77, 77, 76]" in captured
    assert "DEV_USED false" in captured
    assert "TEST_USED false" in captured
    assert "FINAL_EPOCH_ONLY true" in captured
    assert captured.count("\njob ") == 35
    assert "0:20260901 1:20260902 2:20260903 3:20260904 4:20260905" in captured
    assert not output.exists()
