#!/usr/bin/env python3
"""One-time DEV confirmation for the locked H4.1 lambda pair.

Trains on all 384 frozen Stage-A v3 TRAIN examples for exactly five epochs.
The epoch-5 model is saved, then DEV is evaluated once. DEV metrics do not
choose the checkpoint, the lambdas, or any earlier epoch.

This script does not call ``run_training``. TEST annotation rows are not
parsed. TEST ids are read from split membership only so they can be rejected.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.free_eval import evaluate_free_examples
from tiergraph.planner.stage_a_v3_spec import (
    STAGE_A_V3_ANNOTATION_FINGERPRINT,
    STAGE_A_V3_DEV_SIZE,
    STAGE_A_V3_SPLIT_FINGERPRINT,
    STAGE_A_V3_SPLIT_PATH,
    STAGE_A_V3_STEP_A_PATH,
    STAGE_A_V3_STEP_B_PATH,
    STAGE_A_V3_TEST_SIZE,
    STAGE_A_V3_TRAIN_SIZE,
)
from tiergraph.planner.train import (
    TrainConfig,
    build_model,
    build_optimizer,
    evaluate_examples,
    iter_example_batches,
    load_and_split_stage_a_v3,
    save_checkpoint,
    set_seed,
    train_step,
)

LOCKED_LAMBDA_START = 0.0
LOCKED_LAMBDA_END = 0.25
LAMBDA_SOURCE = "TRAIN-only 5-fold CV"
CV_FOLD_FINGERPRINT = (
    "87d6bc26f2685a0abe10a5869405167f963245c0d5ccc3944b18efd55f4ff757"
)
CV_SELECTED_MEAN_H4_F1 = 0.4908415536413339
CV_BASELINE_MEAN_H4_F1 = 0.4801421132799428
CV_DRIVER_COMMIT = "7ed908d70e62908be02fdaa0d0137ba82255cf4e"
SPLIT_FINGERPRINT = (
    "ad221c67fb08290582f863bad682e88d1de7bfd1e127c790aee868030229cd10"
)
ANNOTATION_FINGERPRINT = "30ffae30f24c987c5a77a50fa2d9faa9054198e3d4273ccbecdf09d4a7e0c397"
SEED = 20260901
EPOCHS = 5
BATCH_SIZE = 8
LR = 0.001
DEVICE = "cpu"
DECODE = "viterbi"
N_TRAIN = 384
N_DEV = 48
N_TEST_MEMBERSHIP = 48
DEV_EVALUATIONS = 1
FINAL_EPOCH_ONLY = True
DEV_USED_FOR_SELECTION = False
TEST_USED = False
DEFAULT_OUTPUT = ROOT / "artifacts" / "planner_h4_dev_confirmation"
FINAL_NAME = "final.pt"
METRICS_NAME = "dev_metrics.json"
METADATA_NAME = "experiment_metadata.json"


def load_split_ids(split_path: Path) -> dict[str, set[str]]:
    """Split membership only. Annotation bodies are not read here."""
    ids: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    with Path(split_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            name = str(row["split"])
            if name not in ids:
                raise RuntimeError(f"unknown split label {name!r}")
            ids[name].add(str(row["stage_a_id"]))
    return ids


def annotation_ids_for_confirmation(
    split_ids: dict[str, set[str]],
) -> tuple[set[str], set[str], set[str]]:
    """TRAIN and DEV ids to materialize. TEST ids are rejected."""
    train = set(split_ids["train"])
    dev = set(split_ids["dev"])
    test = set(split_ids["test"])
    if train & dev or train & test or dev & test:
        raise RuntimeError("TRAIN, DEV, and TEST ids overlap")
    if len(train) != N_TRAIN or len(train) != STAGE_A_V3_TRAIN_SIZE:
        raise RuntimeError(f"expected {N_TRAIN} TRAIN ids, got {len(train)}")
    if len(dev) != N_DEV or len(dev) != STAGE_A_V3_DEV_SIZE:
        raise RuntimeError(f"expected {N_DEV} DEV ids, got {len(dev)}")
    if len(test) != N_TEST_MEMBERSHIP or len(test) != STAGE_A_V3_TEST_SIZE:
        raise RuntimeError(f"expected {N_TEST_MEMBERSHIP} TEST membership ids, got {len(test)}")
    load_ids = train | dev
    if load_ids & test:
        raise RuntimeError("TEST ids are in the annotation load set")
    return train, dev, test


def assert_materialized_splits(
    *,
    train_ids: set[str],
    dev_ids: set[str],
    test_ids: set[str],
    loaded_ids: set[str],
    n_test_annotations_materialized: int,
) -> None:
    """The loaded examples are exactly TRAIN+DEV."""
    if n_test_annotations_materialized != 0:
        raise RuntimeError(
            f"TEST annotations materialized: {n_test_annotations_materialized}"
        )
    if loaded_ids & test_ids:
        raise RuntimeError("a TEST id was loaded as an annotation")
    if loaded_ids != train_ids | dev_ids:
        raise RuntimeError("loaded annotations are not exactly TRAIN+DEV")
    if len(train_ids) != N_TRAIN or len(dev_ids) != N_DEV:
        raise RuntimeError(
            f"materialized sizes train={len(train_ids)} dev={len(dev_ids)}"
        )


def run_final_epoch_confirmation(
    *,
    epochs: int,
    train_epoch: Callable[[int], None],
    save_final_checkpoint: Callable[[int], None],
    confirm_dev: Callable[[], Any],
) -> Any:
    """Train every epoch, save that model, then confirm DEV once.

    The DEV callback result is not passed to the saver and cannot replace
    the checkpoint.
    """
    if epochs != EPOCHS:
        raise ValueError(f"DEV confirmation trains exactly {EPOCHS} epochs, got {epochs}")
    for epoch in range(epochs):
        train_epoch(epoch)
    save_final_checkpoint(epochs)
    return confirm_dev()


def locked_config(output_dir: Path) -> TrainConfig:
    return TrainConfig(
        seed=SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=DEVICE,
        output_dir=str(output_dir),
        corpus_version="v3",
        step_a_path=str(ROOT / STAGE_A_V3_STEP_A_PATH),
        step_b_path=str(ROOT / STAGE_A_V3_STEP_B_PATH),
        disabled_heads=(),
        bio_decode_mode=DECODE,
        h2_bio_class_weights=None,
        h4_bio_class_weights=None,
        h4_boundary_lambda_start=LOCKED_LAMBDA_START,
        h4_boundary_lambda_end=LOCKED_LAMBDA_END,
    )


def experiment_metadata() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    if STAGE_A_V3_SPLIT_FINGERPRINT != SPLIT_FINGERPRINT:
        raise RuntimeError("runtime split fingerprint constant drifted")
    if STAGE_A_V3_ANNOTATION_FINGERPRINT != ANNOTATION_FINGERPRINT:
        raise RuntimeError("runtime annotation fingerprint constant drifted")
    return {
        "lambda_start": LOCKED_LAMBDA_START,
        "lambda_end": LOCKED_LAMBDA_END,
        "lambda_source": LAMBDA_SOURCE,
        "cv_fold_fingerprint": CV_FOLD_FINGERPRINT,
        "cv_selected_mean_h4_f1": CV_SELECTED_MEAN_H4_F1,
        "cv_baseline_mean_h4_f1": CV_BASELINE_MEAN_H4_F1,
        "cv_driver_commit": CV_DRIVER_COMMIT,
        "epochs": EPOCHS,
        "final_epoch_only": FINAL_EPOCH_ONLY,
        "dev_evaluations": DEV_EVALUATIONS,
        "dev_used_for_selection": DEV_USED_FOR_SELECTION,
        "test_used": TEST_USED,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "device": DEVICE,
        "decode": DECODE,
        "git_commit": commit,
        "annotation_fingerprint": ANNOTATION_FINGERPRINT,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "n_train": N_TRAIN,
        "n_dev": N_DEV,
        "bio_class_weighting": False,
        "encoder": "frozen MiniLM",
        "heads": "all",
        "primary_metric": "free_viterbi_h4_span_f1",
    }


def dev_metrics_payload(teacher: Any, free: Any) -> dict[str, Any]:
    """Both views of the same final model. Free Viterbi H4 F1 is primary."""
    if int(teacher.n_examples) != N_DEV or int(free.n_examples) != N_DEV:
        raise RuntimeError(
            f"DEV eval coverage teacher={teacher.n_examples} free={free.n_examples}"
        )
    return {
        "primary_metric": "free_viterbi_h4_span_f1",
        "free_viterbi_h4_span_f1": free.anchor_span_f1,
        "free_viterbi": {
            "h4_precision": free.anchor_span_precision,
            "h4_recall": free.anchor_span_recall,
            "h4_span_f1": free.anchor_span_f1,
            "operation_span_f1": free.operation_span_f1,
            "operation_joint_f1": free.operation_joint_f1,
            "operator_type_accuracy_on_matched_operations": (
                free.operator_type_accuracy_on_span_matches
            ),
            "query_type_accuracy": free.query_type_accuracy,
            "execution_accuracy_all_examples": free.execution_mode_accuracy_all_examples,
            "execution_accuracy_valid_only": free.execution_mode_accuracy_valid_only,
            "h5_aligned_accuracy": free.h5_accuracy_span_aligned,
            "h6_aligned_accuracy": free.h6_ownership_accuracy_span_aligned,
            "h7_aligned_f1": free.h7_f1_span_aligned,
            "exact_graph_rate": free.canonical_exact_graph_accuracy,
            "valid_graph_rate": free.valid_graph_rate,
            "n_examples": free.n_examples,
            "decode": DECODE,
        },
        "teacher_forced": teacher.to_dict(),
        "n_dev": N_DEV,
        "dev_evaluations": DEV_EVALUATIONS,
        "dev_used_for_selection": False,
        "final_epoch_only": True,
    }


def _load_train_dev(config: TrainConfig, test_ids: set[str]):
    """Materialize TRAIN+DEV. ``load_and_split_stage_a_v3`` leaves TEST empty."""
    split, _before_a, _before_b = load_and_split_stage_a_v3(config)
    if split.test:
        raise RuntimeError("TEST examples were materialized")
    train_ids = {example.example_id for example in split.train}
    dev_ids = {example.example_id for example in split.dev}
    n_test = int(split.report.get("n_test_annotations_materialized", -1))
    assert_materialized_splits(
        train_ids=train_ids,
        dev_ids=dev_ids,
        test_ids=test_ids,
        loaded_ids=train_ids | dev_ids,
        n_test_annotations_materialized=n_test,
    )
    if len(split.train) != N_TRAIN or len(split.dev) != N_DEV:
        raise RuntimeError("TRAIN or DEV example count drifted after conversion")
    return split.train, split.dev


def execute_confirmation(output_dir: Path) -> dict[str, Any]:
    """Five TRAIN epochs, save epoch 5, then one DEV confirmation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / FINAL_NAME
    metrics_path = output_dir / METRICS_NAME
    if final_path.exists() or metrics_path.exists():
        raise RuntimeError(
            f"{output_dir} already has a confirmation artifact; refusing to overwrite"
        )
    metadata = experiment_metadata()
    (output_dir / METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    split_ids = load_split_ids(ROOT / STAGE_A_V3_SPLIT_PATH)
    _train_ids, _dev_ids, test_ids = annotation_ids_for_confirmation(split_ids)
    config = locked_config(output_dir)
    train_examples, dev_examples = _load_train_dev(config, test_ids)
    set_seed(SEED)
    model = build_model(config)
    optimizer = build_optimizer(model, lr=LR)
    if config.h2_bio_class_weights is not None or config.h4_bio_class_weights is not None:
        raise RuntimeError("BIO class weighting is disabled for this confirmation")
    state = {"train_started": time.perf_counter(), "training_seconds": None}

    def train_epoch(epoch: int) -> None:
        for batch in iter_example_batches(
            train_examples,
            batch_size=BATCH_SIZE,
            seed=SEED,
            epoch=epoch,
            shuffle=True,
        ):
            train_step(
                model,
                optimizer,
                batch,
                h4_boundary_lambda_start=LOCKED_LAMBDA_START,
                h4_boundary_lambda_end=LOCKED_LAMBDA_END,
            )

    def save_final_checkpoint(epoch: int) -> None:
        if epoch != EPOCHS:
            raise RuntimeError(f"refusing to save an epoch other than {EPOCHS}")
        state["training_seconds"] = time.perf_counter() - state["train_started"]
        save_checkpoint(
            final_path,
            model=model,
            config=config,
            split_fingerprint=SPLIT_FINGERPRINT,
            best_dev_loss=float("nan"),
            best_dev_metrics=None,
            epoch=epoch,
            extra={
                "checkpoint_role": "final_epoch",
                "dev_used_for_selection": False,
                "selected_by": None,
                "lambda_start": LOCKED_LAMBDA_START,
                "lambda_end": LOCKED_LAMBDA_END,
            },
        )

    def confirm_dev() -> dict[str, Any]:
        if state["training_seconds"] is None:
            raise RuntimeError("DEV confirmation ran before the epoch-5 checkpoint")
        eval_started = time.perf_counter()
        teacher = evaluate_examples(
            model,
            dev_examples,
            batch_size=BATCH_SIZE,
            seed=SEED,
            h4_boundary_lambda_start=LOCKED_LAMBDA_START,
            h4_boundary_lambda_end=LOCKED_LAMBDA_END,
        )
        free = evaluate_free_examples(
            model,
            dev_examples,
            batch_size=BATCH_SIZE,
            seed=SEED,
            bio_decode_mode=DECODE,
        )
        payload = dev_metrics_payload(teacher, free)
        payload["training_seconds"] = state["training_seconds"]
        payload["evaluation_seconds"] = time.perf_counter() - eval_started
        payload["checkpoint"] = FINAL_NAME
        payload["checkpoint_epoch"] = EPOCHS
        metrics_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return payload

    run_final_epoch_confirmation(
        epochs=EPOCHS,
        train_epoch=train_epoch,
        save_final_checkpoint=save_final_checkpoint,
        confirm_dev=confirm_dev,
    )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def print_dry_run(train_ids: set[str], dev_ids: set[str], test_ids: set[str]) -> None:
    print(f"N_TRAIN {len(train_ids)}")
    print(f"N_DEV {len(dev_ids)}")
    print(f"N_TEST_MEMBERSHIP {len(test_ids)}")
    print(f"N_TEST_ANNOTATIONS_LOADED 0")
    print(f"LOCKED_LAMBDAS {LOCKED_LAMBDA_START:.2f} {LOCKED_LAMBDA_END:.2f}")
    print(f"LAMBDA_SOURCE {LAMBDA_SOURCE}")
    print(f"EPOCHS {EPOCHS}")
    print(f"DEV_EVALUATIONS_PLANNED {DEV_EVALUATIONS}")
    print("DEV_USED_FOR_SELECTION false")
    print("TEST_USED false")
    print("FINAL_EPOCH_ONLY true")
    print(f"DECODE {DECODE}")
    print(f"SEED {SEED}")
    print(f"BATCH_SIZE {BATCH_SIZE}")
    print(f"LR {LR}")
    print(f"DEVICE {DEVICE}")
    print(f"CV_FOLD_FINGERPRINT {CV_FOLD_FINGERPRINT}")
    print(f"CV_SELECTED_MEAN_H4_F1 {CV_SELECTED_MEAN_H4_F1}")
    print("PLAN train_epoch x5 -> save final.pt epoch 5 -> one DEV confirmation")


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate TRAIN/DEV membership and print the locked plan without training",
    )
    args = parser.parse_args(argv)
    split_ids = load_split_ids(ROOT / STAGE_A_V3_SPLIT_PATH)
    train_ids, dev_ids, test_ids = annotation_ids_for_confirmation(split_ids)
    metadata = experiment_metadata()
    if metadata["dev_used_for_selection"] or metadata["test_used"]:
        raise RuntimeError("confirmation metadata allows DEV selection or TEST use")
    if metadata["lambda_start"] != 0.0 or metadata["lambda_end"] != 0.25:
        raise RuntimeError("locked lambdas drifted")
    if args.dry_run:
        print_dry_run(train_ids, dev_ids, test_ids)
        return
    execute_confirmation(args.output_dir)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
