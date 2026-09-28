"""Focused checks for the one-time H4.1 DEV confirmation driver.

These tests do not train a model and do not parse TEST annotation bodies.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "run_planner_h4_dev_confirmation",
    ROOT / "scripts" / "run_planner_h4_dev_confirmation.py",
)
assert _SPEC is not None and _SPEC.loader is not None
confirm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(confirm)

FORBIDDEN_CALLS = {
    "run_training",
    "load_and_split_for_config",
    "evaluate_checkpoint",
}


def _split_ids():
    return confirm.load_split_ids(ROOT / confirm.STAGE_A_V3_SPLIT_PATH)


def test_locked_constants_and_metadata_fields():
    assert confirm.LOCKED_LAMBDA_START == 0.0
    assert confirm.LOCKED_LAMBDA_END == 0.25
    assert confirm.EPOCHS == 5
    assert confirm.BATCH_SIZE == 8
    assert confirm.LR == 0.001
    assert confirm.SEED == 20260901
    assert confirm.DEVICE == "cpu"
    assert confirm.DECODE == "viterbi"
    assert confirm.DEV_EVALUATIONS == 1
    assert confirm.DEV_USED_FOR_SELECTION is False
    assert confirm.TEST_USED is False
    assert confirm.FINAL_EPOCH_ONLY is True
    assert confirm.CV_FOLD_FINGERPRINT == (
        "87d6bc26f2685a0abe10a5869405167f963245c0d5ccc3944b18efd55f4ff757"
    )
    assert confirm.CV_SELECTED_MEAN_H4_F1 == 0.4908415536413339
    metadata = confirm.experiment_metadata()
    for key in (
        "lambda_start",
        "lambda_end",
        "lambda_source",
        "cv_fold_fingerprint",
        "cv_selected_mean_h4_f1",
        "epochs",
        "final_epoch_only",
        "dev_evaluations",
        "dev_used_for_selection",
        "test_used",
        "seed",
        "batch_size",
        "lr",
        "git_commit",
        "annotation_fingerprint",
        "split_fingerprint",
    ):
        assert key in metadata
    assert metadata["lambda_start"] == 0.0
    assert metadata["lambda_end"] == 0.25
    assert metadata["lambda_source"] == "TRAIN-only 5-fold CV"
    assert metadata["dev_evaluations"] == 1
    assert metadata["dev_used_for_selection"] is False
    assert metadata["test_used"] is False
    assert metadata["cv_driver_commit"] == "7ed908d70e62908be02fdaa0d0137ba82255cf4e"
    config = confirm.locked_config(ROOT / "artifacts" / "unused")
    assert config.h2_bio_class_weights is None
    assert config.h4_bio_class_weights is None
    assert config.disabled_heads == ()
    assert config.h4_boundary_lambda_start == 0.0
    assert config.h4_boundary_lambda_end == 0.25


def test_train_dev_counts_and_test_ids_are_not_loaded():
    split_ids = _split_ids()
    train, dev, test = confirm.annotation_ids_for_confirmation(split_ids)
    assert len(train) == 384
    assert len(dev) == 48
    assert len(test) == 48
    assert train.isdisjoint(dev)
    assert train.isdisjoint(test)
    assert dev.isdisjoint(test)
    load_ids = train | dev
    assert load_ids.isdisjoint(test)
    confirm.assert_materialized_splits(
        train_ids=train,
        dev_ids=dev,
        test_ids=test,
        loaded_ids=load_ids,
        n_test_annotations_materialized=0,
    )
    leaked = set(load_ids)
    leaked.add(next(iter(test)))
    with pytest.raises(RuntimeError, match="TEST id was loaded"):
        confirm.assert_materialized_splits(
            train_ids=train,
            dev_ids=dev,
            test_ids=test,
            loaded_ids=leaked,
            n_test_annotations_materialized=0,
        )
    with pytest.raises(RuntimeError, match="TEST annotations materialized"):
        confirm.assert_materialized_splits(
            train_ids=train,
            dev_ids=dev,
            test_ids=test,
            loaded_ids=load_ids,
            n_test_annotations_materialized=1,
        )
    contaminated = {name: set(values) for name, values in split_ids.items()}
    contaminated["train"].add(next(iter(contaminated["test"])))
    with pytest.raises(RuntimeError, match="overlap"):
        confirm.annotation_ids_for_confirmation(contaminated)


def test_final_epoch_then_one_dev_confirmation_and_no_selection():
    events: list[tuple[str, int | None]] = []

    def train_epoch(epoch: int) -> None:
        events.append(("train", epoch))

    def save_final_checkpoint(epoch: int) -> None:
        events.append(("save", epoch))

    def confirm_dev() -> str:
        events.append(("dev", None))
        return "metrics-ignored-for-selection"

    result = confirm.run_final_epoch_confirmation(
        epochs=5,
        train_epoch=train_epoch,
        save_final_checkpoint=save_final_checkpoint,
        confirm_dev=confirm_dev,
    )
    assert events == [("train", epoch) for epoch in range(5)] + [
        ("save", 5),
        ("dev", None),
    ]
    assert result == "metrics-ignored-for-selection"
    assert events.count(("dev", None)) == 1
    assert events.index(("save", 5)) < events.index(("dev", None))
    with pytest.raises(ValueError, match="exactly 5"):
        confirm.run_final_epoch_confirmation(
            epochs=4,
            train_epoch=train_epoch,
            save_final_checkpoint=save_final_checkpoint,
            confirm_dev=confirm_dev,
        )


def test_no_dev_checkpoint_selection_in_source():
    source = (ROOT / "scripts" / "run_planner_h4_dev_confirmation.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    calls: set[str] = set()
    imported: set[str] = set()
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
    assert "run_final_epoch_confirmation" in calls
    assert "evaluate_free_examples" in calls
    assert "evaluate_examples" in calls
    protocol = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_final_epoch_confirmation"
    )
    call_names = [
        node.func.id
        for node in ast.walk(protocol)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert call_names.index("save_final_checkpoint") < call_names.index("confirm_dev")
    execute = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_confirmation"
    )
    save_calls = [
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "save_checkpoint")
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "save_checkpoint"
            )
        )
    ]
    assert len(save_calls) == 1
    keywords = {item.arg: item.value for item in save_calls[0].keywords}
    assert isinstance(keywords["best_dev_metrics"], ast.Constant)
    assert keywords["best_dev_metrics"].value is None
    assert isinstance(keywords["epoch"], ast.Name)
    assert keywords["epoch"].id == "epoch"


def test_dev_metrics_record_free_viterbi_as_primary():
    teacher = SimpleNamespace(
        n_examples=48,
        to_dict=lambda: {"h4_span_f1": 0.2, "mean_loss": {"total": 1.0}},
    )
    free = SimpleNamespace(
        n_examples=48,
        anchor_span_precision=0.4,
        anchor_span_recall=0.5,
        anchor_span_f1=0.44,
        operation_span_f1=0.3,
        operation_joint_f1=0.2,
        operator_type_accuracy_on_span_matches=0.6,
        query_type_accuracy=0.7,
        execution_mode_accuracy_all_examples=0.8,
        execution_mode_accuracy_valid_only=0.9,
        h5_accuracy_span_aligned=0.1,
        h6_ownership_accuracy_span_aligned=0.11,
        h7_f1_span_aligned=0.12,
        canonical_exact_graph_accuracy=0.13,
        valid_graph_rate=0.14,
    )
    payload = confirm.dev_metrics_payload(teacher, free)
    assert payload["primary_metric"] == "free_viterbi_h4_span_f1"
    assert payload["free_viterbi_h4_span_f1"] == 0.44
    view = payload["free_viterbi"]
    assert view["h4_precision"] == 0.4
    assert view["h4_recall"] == 0.5
    assert view["operation_span_f1"] == 0.3
    assert view["operation_joint_f1"] == 0.2
    assert view["exact_graph_rate"] == 0.13
    assert view["valid_graph_rate"] == 0.14
    assert payload["teacher_forced"]["h4_span_f1"] == 0.2
    assert payload["dev_used_for_selection"] is False
    assert payload["dev_evaluations"] == 1


def test_dry_run_prints_plan_and_does_not_train(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    output = tmp_path / "planner_h4_dev_confirmation"
    confirm.run(["--dry-run", "--output-dir", str(output)])
    captured = capsys.readouterr().out
    assert "N_TRAIN 384" in captured
    assert "N_DEV 48" in captured
    assert "N_TEST_ANNOTATIONS_LOADED 0" in captured
    assert "LOCKED_LAMBDAS 0.00 0.25" in captured
    assert "EPOCHS 5" in captured
    assert "DEV_EVALUATIONS_PLANNED 1" in captured
    assert "DEV_USED_FOR_SELECTION false" in captured
    assert "TEST_USED false" in captured
    assert "FINAL_EPOCH_ONLY true" in captured
    assert "save final.pt epoch 5 -> one DEV confirmation" in captured
    assert not output.exists()
