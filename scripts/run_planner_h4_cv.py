#!/usr/bin/env python3
"""TRAIN-only 5-fold lambda search for the H4.1 boundary auxiliaries.

Trains on four frozen TRAIN folds for exactly five epochs and scores the
final model on the held-out TRAIN fold with free Viterbi decoding.

This driver does not call ``run_training`` and does not read DEV or TEST
annotations. DEV and TEST ids are taken from the split membership file only
so a contaminated fold map can be rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.annotation_step_a import StageAStepAAnnotation
from tiergraph.planner.annotation_step_b import StageAStepBAnnotation
from tiergraph.planner.free_eval import evaluate_free_examples
from tiergraph.planner.stage_a_to_corpus import step_ab_to_planner_example
from tiergraph.planner.stage_a_v3_h4_build import load_annotation_rows_for_ids
from tiergraph.planner.stage_a_v3_spec import (
    STAGE_A_V3_ANNOTATION_FINGERPRINT,
    STAGE_A_V3_SPLIT_FINGERPRINT,
    STAGE_A_V3_SPLIT_PATH,
    STAGE_A_V3_STEP_A_PATH,
    STAGE_A_V3_STEP_B_PATH,
)
from tiergraph.planner.train import (
    TrainConfig,
    build_model,
    build_optimizer,
    iter_example_batches,
    train_step,
)

CV_FOLD_FINGERPRINT = (
    "87d6bc26f2685a0abe10a5869405167f963245c0d5ccc3944b18efd55f4ff757"
)
SPLIT_FINGERPRINT = (
    "ad221c67fb08290582f863bad682e88d1de7bfd1e127c790aee868030229cd10"
)
ANNOTATION_FINGERPRINT = STAGE_A_V3_ANNOTATION_FINGERPRINT
BASE_SEED = 20260901
EPOCHS = 5
BATCH_SIZE = 8
LR = 0.001
DEVICE = "cpu"
DECODE = "viterbi"
FINAL_EPOCH_ONLY = True
EXPECTED_FOLD_SIZES = (77, 77, 77, 77, 76)
LAMBDA_GRID: tuple[tuple[float, float], ...] = (
    (0.00, 0.00),
    (0.25, 0.00),
    (0.00, 0.25),
    (0.25, 0.25),
    (0.50, 0.00),
    (0.00, 0.50),
    (0.50, 0.50),
)
SELECTION_RULE = (
    "Select the lambda pair with the highest mean held-out TRAIN H4 span F1. "
    "If a pair's mean F1 is within 0.005 below the maximum, treat it as tied. "
    "Among tied pairs prefer smaller lambda_start + lambda_end, then smaller "
    "lambda_end, then smaller lambda_start. Do not expand the grid."
)
TIE_F1_GAP = 0.005
DEFAULT_MANIFEST = ROOT / "dataset" / "planner" / "stage_a_v3_h4_cv_folds.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "planner_h4_cv"
METADATA_NAME = "experiment_metadata.json"
PER_FOLD_NAME = "per_fold_results.jsonl"
SUMMARY_JSON_NAME = "summary.json"
SUMMARY_CSV_NAME = "summary.csv"
COMPAT_FIELDS = (
    "cv_fold_fingerprint",
    "split_fingerprint",
    "annotation_fingerprint",
    "lambda_grid",
    "selection_rule",
    "epochs",
    "batch_size",
    "lr",
    "base_seed",
    "fold_seeds",
    "device",
    "decode",
    "final_epoch_only",
    "dev_used",
    "test_used",
    "git_commit",
)


def fold_seed(fold: int, base_seed: int = BASE_SEED) -> int:
    """Seed for one fold. Identical for every lambda pair."""
    if fold not in range(5):
        raise ValueError(f"fold must be 0..4, got {fold}")
    return int(base_seed) + int(fold)


def seeds_by_fold(base_seed: int = BASE_SEED) -> dict[str, int]:
    return {str(fold): fold_seed(fold, base_seed) for fold in range(5)}


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


def load_fold_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"fold manifest must be an object: {path}")
    return payload


def mapping_from_manifest(payload: dict[str, Any]) -> dict[str, int]:
    rows = payload.get("assignments")
    if not isinstance(rows, list):
        raise RuntimeError("fold manifest is missing assignments")
    mapping: dict[str, int] = {}
    for row in rows:
        sid = str(row["stage_a_id"])
        fold = int(row["fold"])
        if sid in mapping:
            raise RuntimeError(f"duplicate fold assignment {sid}")
        mapping[sid] = fold
    return mapping


def validate_fold_manifest(
    payload: dict[str, Any],
    split_ids: dict[str, set[str]],
) -> dict[str, int]:
    """Reject a fold map that is not the frozen TRAIN assignment."""
    fingerprint = str(payload.get("cv_fold_fingerprint") or "")
    if fingerprint != CV_FOLD_FINGERPRINT:
        raise RuntimeError(
            "CV fold fingerprint mismatch: "
            f"got {fingerprint}, expected {CV_FOLD_FINGERPRINT}"
        )
    if str(payload.get("split_fingerprint") or "") != SPLIT_FINGERPRINT:
        raise RuntimeError("manifest split fingerprint does not match the frozen split")
    if str(payload.get("annotation_fingerprint") or "") != ANNOTATION_FINGERPRINT:
        raise RuntimeError(
            "manifest annotation fingerprint does not match the frozen annotations"
        )
    if STAGE_A_V3_SPLIT_FINGERPRINT != SPLIT_FINGERPRINT:
        raise RuntimeError("runtime split fingerprint constant drifted")
    if ANNOTATION_FINGERPRINT != (
        "30ffae30f24c987c5a77a50fa2d9faa9054198e3d4273ccbecdf09d4a7e0c397"
    ):
        raise RuntimeError("runtime annotation fingerprint constant drifted")
    mapping = mapping_from_manifest(payload)
    if len(mapping) != 384:
        raise RuntimeError(f"expected 384 TRAIN ids, got {len(mapping)}")
    leaked = (set(mapping) & split_ids["dev"]) | (set(mapping) & split_ids["test"])
    if leaked:
        raise RuntimeError(
            f"DEV or TEST ids are in the fold map: {sorted(leaked)[:5]}"
        )
    if set(mapping) != split_ids["train"]:
        raise RuntimeError("fold ids are not exactly the frozen TRAIN set")
    if set(mapping.values()) != {0, 1, 2, 3, 4}:
        raise RuntimeError(f"folds must be exactly 0..4, got {sorted(set(mapping.values()))}")
    sizes = [sum(fold == index for fold in mapping.values()) for index in range(5)]
    if tuple(sizes) != EXPECTED_FOLD_SIZES:
        raise RuntimeError(f"fold sizes {sizes} != {list(EXPECTED_FOLD_SIZES)}")
    if list(payload.get("fold_sizes") or []) != list(EXPECTED_FOLD_SIZES):
        raise RuntimeError("manifest fold_sizes field does not match [77,77,77,77,76]")
    return mapping


def planned_jobs(mapping: dict[str, int]) -> list[dict[str, Any]]:
    """35 jobs. Fold seed does not depend on lambda."""
    jobs = []
    for lambda_start, lambda_end in LAMBDA_GRID:
        for fold in range(5):
            holdout = [sid for sid, assigned in mapping.items() if assigned == fold]
            jobs.append(
                {
                    "lambda_start": lambda_start,
                    "lambda_end": lambda_end,
                    "fold": fold,
                    "seed": fold_seed(fold),
                    "n_train": 384 - len(holdout),
                    "n_holdout": len(holdout),
                }
            )
    if len(jobs) != 35:
        raise RuntimeError(f"expected 35 jobs, got {len(jobs)}")
    return jobs


def select_lambda(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered rule. ``mean_h4_f1`` is required."""
    if not summaries:
        raise ValueError("no lambda summaries to select from")
    best_f1 = max(float(row["mean_h4_f1"]) for row in summaries)
    tied = [
        row
        for row in summaries
        if best_f1 - float(row["mean_h4_f1"]) < TIE_F1_GAP
    ]
    tied.sort(
        key=lambda row: (
            float(row["lambda_start"]) + float(row["lambda_end"]),
            float(row["lambda_end"]),
            float(row["lambda_start"]),
        )
    )
    winner = dict(tied[0])
    winner["selection_reason"] = (
        f"max mean H4 F1 is {best_f1:.6f}; "
        f"{len(tied)} pair(s) are within {TIE_F1_GAP}; "
        "winner has the smallest lambda sum, then lambda_end, then lambda_start"
    )
    return winner


def run_fold_epochs(
    *,
    epochs: int,
    train_epoch: Callable[[int], None],
) -> None:
    """Train every requested epoch. Evaluation is separate and happens once."""
    if epochs != EPOCHS:
        raise ValueError(f"H4 CV trains exactly {EPOCHS} epochs, got {epochs}")
    for epoch in range(epochs):
        train_epoch(epoch)


def experiment_metadata() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
        ).strip()
    )
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "cv_fold_fingerprint": CV_FOLD_FINGERPRINT,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "annotation_fingerprint": ANNOTATION_FINGERPRINT,
        "lambda_grid": [list(pair) for pair in LAMBDA_GRID],
        "selection_rule": SELECTION_RULE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "base_seed": BASE_SEED,
        "fold_seeds": seeds_by_fold(),
        "device": DEVICE,
        "decode": DECODE,
        "final_epoch_only": FINAL_EPOCH_ONLY,
        "dev_used": False,
        "test_used": False,
        "bio_class_weighting": False,
        "encoder": "frozen MiniLM",
        "heads": "all",
    }


def _compatible(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = []
    for key in COMPAT_FIELDS:
        if existing.get(key) != expected.get(key):
            mismatches.append(key)
    if mismatches:
        raise RuntimeError(
            "existing H4 CV artifacts do not match this experiment: "
            + ", ".join(mismatches)
        )


def load_completed(output_dir: Path, expected: dict[str, Any]) -> list[dict[str, Any]]:
    """Return finished fold rows, or fail if a previous run is incompatible."""
    meta_path = output_dir / METADATA_NAME
    results_path = output_dir / PER_FOLD_NAME
    has_meta = meta_path.is_file()
    has_results = results_path.is_file() and results_path.stat().st_size > 0
    if has_results and not has_meta:
        raise RuntimeError(
            f"{results_path} exists without {METADATA_NAME}; refusing to resume"
        )
    if not has_meta:
        return []
    existing = json.loads(meta_path.read_text(encoding="utf-8"))
    _compatible(existing, expected)
    if not has_results:
        return []
    rows = []
    for line_number, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("cv_fold_fingerprint")) != CV_FOLD_FINGERPRINT:
            raise RuntimeError(
                f"{results_path}:{line_number} fingerprint does not match the frozen folds"
            )
        rows.append(row)
    return rows


def job_key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (float(row["lambda_start"]), float(row["lambda_end"]), int(row["fold"]))


def pending_jobs(
    jobs: list[dict[str, Any]],
    completed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    done = {job_key(row) for row in completed}
    return [job for job in jobs if job_key(job) not in done]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and sample standard deviation for each complete lambda pair."""
    by_lambda: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        by_lambda.setdefault(job_key(row)[:2], []).append(row)
    summaries = []
    for lambda_start, lambda_end in LAMBDA_GRID:
        group = by_lambda.get((lambda_start, lambda_end), [])
        if len(group) != 5:
            continue
        group = sorted(group, key=lambda item: int(item["fold"]))
        f1 = [float(item["h4_f1"]) for item in group]
        precision = [float(item["h4_precision"]) for item in group]
        recall = [float(item["h4_recall"]) for item in group]
        summaries.append(
            {
                "lambda_start": lambda_start,
                "lambda_end": lambda_end,
                "mean_h4_f1": statistics.fmean(f1),
                "std_h4_f1": statistics.stdev(f1),
                "mean_h4_precision": statistics.fmean(precision),
                "mean_h4_recall": statistics.fmean(recall),
                "fold_h4_f1": f1,
            }
        )
    selected = select_lambda(summaries) if len(summaries) == len(LAMBDA_GRID) else None
    return {"lambda_summaries": summaries, "selected": selected}


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / SUMMARY_JSON_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected = summary.get("selected") or {}
    selected_key = (
        float(selected["lambda_start"]),
        float(selected["lambda_end"]),
    ) if selected else None
    with (output_dir / SUMMARY_CSV_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lambda_start",
                "lambda_end",
                "mean_h4_f1",
                "std_h4_f1",
                "mean_h4_precision",
                "mean_h4_recall",
                "fold0_h4_f1",
                "fold1_h4_f1",
                "fold2_h4_f1",
                "fold3_h4_f1",
                "fold4_h4_f1",
                "selected",
            ],
        )
        writer.writeheader()
        for row in summary["lambda_summaries"]:
            folds = row["fold_h4_f1"]
            key = (float(row["lambda_start"]), float(row["lambda_end"]))
            writer.writerow(
                {
                    "lambda_start": row["lambda_start"],
                    "lambda_end": row["lambda_end"],
                    "mean_h4_f1": row["mean_h4_f1"],
                    "std_h4_f1": row["std_h4_f1"],
                    "mean_h4_precision": row["mean_h4_precision"],
                    "mean_h4_recall": row["mean_h4_recall"],
                    "fold0_h4_f1": folds[0],
                    "fold1_h4_f1": folds[1],
                    "fold2_h4_f1": folds[2],
                    "fold3_h4_f1": folds[3],
                    "fold4_h4_f1": folds[4],
                    "selected": key == selected_key,
                }
            )


def _load_train_examples(ids: set[str]):
    step_a = load_annotation_rows_for_ids(
        ROOT / STAGE_A_V3_STEP_A_PATH,
        ids,
        model_cls=StageAStepAAnnotation,
    )
    step_b = load_annotation_rows_for_ids(
        ROOT / STAGE_A_V3_STEP_B_PATH,
        ids,
        model_cls=StageAStepBAnnotation,
    )
    examples = {}
    for sid in sorted(ids):
        example = step_ab_to_planner_example(
            step_a[sid],
            step_b[sid],
            use_semantic_h1=True,
        )
        if example.example_id != sid:
            raise RuntimeError(f"example id {example.example_id} != {sid}")
        examples[sid] = example
    return examples


def execute_job(job: dict[str, Any], examples_by_id: dict, mapping: dict[str, int]) -> dict[str, Any]:
    import torch

    fold = int(job["fold"])
    seed = int(job["seed"])
    if seed != fold_seed(fold):
        raise RuntimeError("refusing a lambda-dependent or mismatched fold seed")
    holdout_ids = sorted(sid for sid, assigned in mapping.items() if assigned == fold)
    train_ids = sorted(sid for sid in mapping if sid not in set(holdout_ids))
    holdout = [examples_by_id[sid] for sid in holdout_ids]
    train_examples = [examples_by_id[sid] for sid in train_ids]
    torch.manual_seed(seed)
    model = build_model(
        TrainConfig(
            seed=seed,
            device=DEVICE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
        )
    )
    optimizer = build_optimizer(model, lr=LR)
    lambda_start = float(job["lambda_start"])
    lambda_end = float(job["lambda_end"])

    def train_epoch(epoch: int) -> None:
        for batch in iter_example_batches(
            train_examples,
            batch_size=BATCH_SIZE,
            seed=seed,
            epoch=epoch,
            shuffle=True,
        ):
            train_step(
                model,
                optimizer,
                batch,
                h4_boundary_lambda_start=lambda_start,
                h4_boundary_lambda_end=lambda_end,
            )

    def evaluate():
        return evaluate_free_examples(
            model,
            holdout,
            batch_size=BATCH_SIZE,
            seed=seed,
            bio_decode_mode=DECODE,
        )

    train_started = time.perf_counter()
    run_fold_epochs(epochs=EPOCHS, train_epoch=train_epoch)
    train_seconds = time.perf_counter() - train_started
    eval_started = time.perf_counter()
    metrics = evaluate()
    eval_seconds = time.perf_counter() - eval_started
    if int(metrics.n_examples) != len(holdout):
        raise RuntimeError("held-out evaluation did not cover the fold")
    return {
        "lambda_start": lambda_start,
        "lambda_end": lambda_end,
        "fold": fold,
        "seed": seed,
        "n_train": len(train_examples),
        "n_holdout": len(holdout),
        "h4_precision": metrics.anchor_span_precision,
        "h4_recall": metrics.anchor_span_recall,
        "h4_f1": metrics.anchor_span_f1,
        "operation_span_f1": metrics.operation_span_f1,
        "query_type_accuracy": metrics.query_type_accuracy,
        "exact_graph_rate": metrics.canonical_exact_graph_accuracy,
        "valid_graph_rate": metrics.valid_graph_rate,
        "training_seconds": train_seconds,
        "evaluation_seconds": eval_seconds,
        "epochs": EPOCHS,
        "final_epoch_only": True,
        "decode": DECODE,
        "dev_used": False,
        "test_used": False,
        "cv_fold_fingerprint": CV_FOLD_FINGERPRINT,
    }


def _append_result(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def print_dry_run(jobs: list[dict[str, Any]]) -> None:
    print(f"N_JOBS {len(jobs)}")
    print(f"FOLD_SIZES {list(EXPECTED_FOLD_SIZES)}")
    print(
        "LAMBDA_GRID "
        + " ".join(f"({start:.2f},{end:.2f})" for start, end in LAMBDA_GRID)
    )
    print(
        "SEEDS_BY_FOLD "
        + " ".join(f"{fold}:{fold_seed(fold)}" for fold in range(5))
    )
    print("DEV_USED false")
    print("TEST_USED false")
    print("FINAL_EPOCH_ONLY true")
    for index, job in enumerate(jobs, start=1):
        print(
            f"job {index:02d} fold={job['fold']} "
            f"lambda_start={job['lambda_start']:.2f} "
            f"lambda_end={job['lambda_end']:.2f} "
            f"seed={job['seed']} "
            f"n_train={job['n_train']} n_holdout={job['n_holdout']}"
        )


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the frozen folds and print the 35 jobs without training",
    )
    args = parser.parse_args(argv)
    payload = load_fold_manifest(args.manifest)
    split_ids = load_split_ids(ROOT / STAGE_A_V3_SPLIT_PATH)
    mapping = validate_fold_manifest(payload, split_ids)
    jobs = planned_jobs(mapping)
    metadata = experiment_metadata()
    if args.dry_run:
        if args.output_dir.exists():
            completed = load_completed(args.output_dir, metadata)
            print(f"RESUME_COMPLETED {len(completed)}")
        print_dry_run(jobs)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / METADATA_NAME
    completed = load_completed(args.output_dir, metadata)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    remaining = pending_jobs(jobs, completed)
    examples = _load_train_examples(set(mapping))
    for job in remaining:
        row = execute_job(job, examples, mapping)
        _append_result(args.output_dir / PER_FOLD_NAME, row)
        completed.append(row)
        summary = aggregate(completed)
        _write_summary(args.output_dir, summary)
    if pending_jobs(jobs, completed):
        raise RuntimeError("CV finished with pending jobs")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
