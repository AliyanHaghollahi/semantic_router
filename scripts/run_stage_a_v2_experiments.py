#!/usr/bin/env python3
"""End-to-end Stage-A v2 experiment pipeline (multi-seed, baselines, ablations)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.stage_a_v2_spec import STAGE_A_V2_SPLIT_FINGERPRINT
from tiergraph.planner.train import (
    TrainConfig,
    evaluate_checkpoint,
    load_and_split_stage_a_v2,
    run_training,
)
from tiergraph.planner.v2_baselines import run_v2_baselines

DEFAULT_SEEDS = (20260901, 20260902, 20260903, 20260904)
DEFAULT_OUTPUT = ROOT / "artifacts" / "planner_v2_experiments"
REFERENCE_RUN = ROOT / "artifacts" / "planner_v2_run1"
V1_REFERENCE = ROOT / "artifacts" / "planner_run1"


def _json_ready(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _v2_train_config(seed: int, output_dir: Path, disabled_heads: tuple[str, ...] = ()) -> TrainConfig:
    return TrainConfig(
        seed=seed,
        epochs=40,
        batch_size=8,
        lr=5e-4,
        device="cpu",
        output_dir=str(output_dir),
        corpus_version="v2",
        step_a_path="dataset/planner/stage_a_v2_step_a_annotations.jsonl",
        step_b_path="dataset/planner/stage_a_v2_step_b_annotations.jsonl",
        disabled_heads=disabled_heads,
    )


def _best_epoch_from_history(history: list[dict[str, Any]]) -> int:
    return min(
        range(len(history)),
        key=lambda index: history[index]["dev"]["mean_loss"]["total"],
    )


def _eval_seed_record(
    seed: int,
    output_root: Path,
    *,
    reuse_reference: bool = True,
) -> dict[str, Any]:
    """Re-evaluate an existing checkpoint without training."""
    if seed == 20260901 and reuse_reference and (REFERENCE_RUN / "best.pt").is_file():
        run_dir = REFERENCE_RUN
        reused = True
    else:
        run_dir = output_root / "seeds" / str(seed)
        reused = False

    if not (run_dir / "best.pt").is_file():
        raise FileNotFoundError(f"Missing checkpoint for seed {seed}: {run_dir / 'best.pt'}")

    split, _, _ = load_and_split_stage_a_v2(_v2_train_config(seed, run_dir))
    teacher = evaluate_checkpoint(
        run_dir / "best.pt",
        split_name="test",
        split=split,
        expected_fingerprint=STAGE_A_V2_SPLIT_FINGERPRINT,
        eval_mode="teacher-forced",
    )
    free = evaluate_checkpoint(
        run_dir / "best.pt",
        split_name="test",
        split=split,
        expected_fingerprint=STAGE_A_V2_SPLIT_FINGERPRINT,
        eval_mode="free",
    )

    history_path = run_dir / "last.pt"
    best_epoch = teacher.checkpoint_epoch
    if history_path.is_file():
        import torch

        payload = torch.load(history_path, map_location="cpu", weights_only=False)
        history = payload.get("extra", {}).get("history")
        if isinstance(history, list) and history:
            best_epoch = _best_epoch_from_history(history)

    train_config_path = run_dir / "train_config.json"
    train_config = {}
    if train_config_path.is_file():
        train_config = json.loads(train_config_path.read_text(encoding="utf-8"))

    record = {
        "seed": seed,
        "reused_reference_run": reused,
        "checkpoint": str(run_dir / "best.pt"),
        "best_epoch": best_epoch,
        "best_dev_loss": teacher.checkpoint_best_dev_loss,
        "best_dev_metrics": train_config.get("best_dev_metrics"),
        "final_train_loss": None,
        "teacher_forced_test": teacher.metrics.to_dict(),
        "free_test": free.metrics.to_dict(),
    }
    if train_config_path.is_file():
        cfg = train_config
        history = None
        last_payload_path = run_dir / "last.pt"
        if last_payload_path.is_file():
            import torch

            payload = torch.load(last_payload_path, map_location="cpu", weights_only=False)
            history = payload.get("extra", {}).get("history")
        if isinstance(history, list) and history:
            record["final_train_loss"] = history[-1]["train_loss"]
        record["best_dev_metrics"] = cfg.get("best_dev_metrics")

    out_path = output_root / "seeds" / f"{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _train_and_eval_seed(
    seed: int,
    output_root: Path,
    *,
    reuse_reference: bool = True,
) -> dict[str, Any]:
    if seed == 20260901 and reuse_reference and (REFERENCE_RUN / "best.pt").is_file():
        run_dir = REFERENCE_RUN
        reused = True
    else:
        run_dir = output_root / "seeds" / str(seed)
        reused = False
        config = _v2_train_config(seed, run_dir)
        run_training(config)

    record = _eval_seed_record(seed, output_root, reuse_reference=reuse_reference)
    record["reused_reference_run"] = reused
    return record


def _eval_ablation_record(
    name: str,
    output_root: Path,
) -> dict[str, Any]:
    """Re-evaluate an existing ablation checkpoint without training."""
    run_dir = output_root / "ablations" / name
    if not (run_dir / "best.pt").is_file():
        raise FileNotFoundError(f"Missing ablation checkpoint: {run_dir / 'best.pt'}")

    train_config_path = run_dir / "train_config.json"
    disabled_heads: tuple[str, ...] = ()
    seed = 20260901
    if train_config_path.is_file():
        cfg = json.loads(train_config_path.read_text(encoding="utf-8"))
        nested = cfg.get("config", {})
        disabled_heads = tuple(nested.get("disabled_heads", cfg.get("disabled_heads", ())))
        seed = int(nested.get("seed", cfg.get("seed", seed)))

    config = _v2_train_config(seed, run_dir, disabled_heads=disabled_heads)
    split, _, _ = load_and_split_stage_a_v2(config)
    teacher = evaluate_checkpoint(
        run_dir / "best.pt",
        split_name="test",
        split=split,
        expected_fingerprint=STAGE_A_V2_SPLIT_FINGERPRINT,
        eval_mode="teacher-forced",
    )
    free = evaluate_checkpoint(
        run_dir / "best.pt",
        split_name="test",
        split=split,
        expected_fingerprint=STAGE_A_V2_SPLIT_FINGERPRINT,
        eval_mode="free",
    )
    record = {
        "name": name,
        "seed": seed,
        "disabled_heads": list(disabled_heads),
        "checkpoint": str(run_dir / "best.pt"),
        "best_epoch": teacher.checkpoint_epoch,
        "best_dev_loss": teacher.checkpoint_best_dev_loss,
        "teacher_forced_test": teacher.metrics.to_dict(),
        "free_test": free.metrics.to_dict(),
    }
    out_path = output_root / "ablations" / f"{name}.json"
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _run_ablation(
    name: str,
    disabled_heads: tuple[str, ...],
    output_root: Path,
    seed: int = 20260901,
) -> dict[str, Any]:
    run_dir = output_root / "ablations" / name
    config = _v2_train_config(seed, run_dir, disabled_heads=disabled_heads)
    run_training(config)
    return _eval_ablation_record(name, output_root)


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
    }


def _collect_metric(seed_records: list[dict[str, Any]], *keys: str) -> list[float]:
    out: list[float] = []
    for record in seed_records:
        node: Any = record
        for key in keys:
            node = node[key]
        out.append(float(node))
    return out


def _aggregate_multiseed(seed_records: list[dict[str, Any]]) -> dict[str, Any]:
    def agg(path: tuple[str, ...]) -> dict[str, float]:
        return _mean_std(_collect_metric(seed_records, *path))

    teacher_paths = {
        "dev_loss": ("best_dev_loss",),
        "h1_accuracy": ("teacher_forced_test", "h1_accuracy"),
        "h2_span_precision": ("teacher_forced_test", "h2_span_precision"),
        "h2_span_recall": ("teacher_forced_test", "h2_span_recall"),
        "h2_span_f1": ("teacher_forced_test", "h2_span_f1"),
        "h3_operator_accuracy": ("teacher_forced_test", "h3_operator_accuracy"),
        "h4_span_precision": ("teacher_forced_test", "h4_span_precision"),
        "h4_span_recall": ("teacher_forced_test", "h4_span_recall"),
        "h4_span_f1": ("teacher_forced_test", "h4_span_f1"),
        "h5_accuracy": ("teacher_forced_test", "h5_accuracy"),
        "h6_ownership_accuracy": ("teacher_forced_test", "h6_ownership_accuracy"),
        "h7_precision": ("teacher_forced_test", "h7_precision"),
        "h7_recall": ("teacher_forced_test", "h7_recall"),
        "h7_f1": ("teacher_forced_test", "h7_f1"),
    }
    free_paths = {
        "valid_graph_rate": ("free_test", "valid_graph_rate"),
        "canonical_exact_graph_accuracy": ("free_test", "canonical_exact_graph_accuracy"),
        "execution_mode_accuracy_valid_only": (
            "free_test",
            "execution_mode_accuracy_valid_only",
        ),
        "execution_mode_accuracy_all_examples": (
            "free_test",
            "execution_mode_accuracy_all_examples",
        ),
        "query_type_accuracy": ("free_test", "query_type_accuracy"),
        "h5_accuracy_span_aligned": ("free_test", "h5_accuracy_span_aligned"),
        "h6_ownership_accuracy_span_aligned": (
            "free_test",
            "h6_ownership_accuracy_span_aligned",
        ),
        "h7_precision_span_aligned": ("free_test", "h7_precision_span_aligned"),
        "h7_recall_span_aligned": ("free_test", "h7_recall_span_aligned"),
        "h7_f1_span_aligned": ("free_test", "h7_f1_span_aligned"),
    }

    teacher_agg = {key: agg(path) for key, path in teacher_paths.items()}
    free_agg = {key: agg(path) for key, path in free_paths.items()}

    by_metric = {
        "teacher_forced_test": teacher_agg,
        "free_test": free_agg,
    }
    best_seed = min(seed_records, key=lambda r: r["best_dev_loss"])["seed"]
    worst_seed = max(seed_records, key=lambda r: r["best_dev_loss"])["seed"]
    return {
        "n_seeds": len(seed_records),
        "seeds": [record["seed"] for record in seed_records],
        "by_metric": by_metric,
        "best_seed_by_dev_loss": best_seed,
        "worst_seed_by_dev_loss": worst_seed,
    }


def _load_v1_reference() -> dict[str, Any] | None:
    if not (V1_REFERENCE / "best.pt").is_file():
        return None
    teacher = evaluate_checkpoint(
        V1_REFERENCE / "best.pt",
        split_name="test",
        eval_mode="teacher-forced",
    )
    free = evaluate_checkpoint(
        V1_REFERENCE / "best.pt",
        split_name="test",
        eval_mode="free",
    )
    train_config = {}
    cfg_path = V1_REFERENCE / "train_config.json"
    if cfg_path.is_file():
        train_config = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {
        "checkpoint": str(V1_REFERENCE / "best.pt"),
        "corpus": "v1_feasibility_120",
        "test_size": 12,
        "best_dev_metrics": train_config.get("best_dev_metrics"),
        "teacher_forced_test": teacher.metrics.to_dict(),
        "free_test": free.metrics.to_dict(),
    }


def build_final_report(output_root: Path) -> dict[str, Any]:
    seed_records = []
    for seed in DEFAULT_SEEDS:
        path = output_root / "seeds" / f"{seed}.json"
        if path.is_file():
            seed_records.append(json.loads(path.read_text(encoding="utf-8")))
    seed_records.sort(key=lambda item: item["seed"])

    baselines_path = output_root / "baselines.json"
    baselines: dict[str, Any] = {}
    if baselines_path.is_file():
        baselines = json.loads(baselines_path.read_text(encoding="utf-8"))

    ablations = []
    ablations_dir = output_root / "ablations"
    if ablations_dir.is_dir():
        for path in sorted(ablations_dir.glob("*.json")):
            if path.name.endswith(".json") and path.stem not in {"summary"}:
                ablations.append(json.loads(path.read_text(encoding="utf-8")))

    reference_run = None
    ref_path = output_root / "seeds" / "20260901.json"
    if ref_path.is_file():
        reference_run = json.loads(ref_path.read_text(encoding="utf-8"))

    report = {
        "split_fingerprint": STAGE_A_V2_SPLIT_FINGERPRINT,
        "configuration": {
            "epochs": 40,
            "batch_size": 8,
            "lr": 5e-4,
            "encoder": "sentence-transformers/all-MiniLM-L6-v2 (frozen)",
            "checkpoint_selection": "dev_loss",
            "test_used_for_tuning": False,
        },
        "multiseed": {
            "seed_runs": seed_records,
            "aggregate": _aggregate_multiseed(seed_records) if seed_records else None,
        },
        "baselines": baselines,
        "ablations": ablations,
        "v1_feasibility_reference": _load_v1_reference(),
        "v2_run1_reference": reference_run,
        "limitations": [
            "H4 anchor extraction remains the weakest head in teacher-forced metrics.",
            "Canonical exact graph accuracy is low despite high valid-graph rate.",
            "Agent-assisted annotation provenance applies to v2 expansion rows.",
            "Authored sequential families add template diversity but also noise risk.",
            "Publication test (48 examples) is evaluated only after checkpoint freeze.",
        ],
        "headline_numbers": {},
    }

    if seed_records:
        agg = report["multiseed"]["aggregate"]
        report["headline_numbers"] = {
            "h1_test_mean": agg["by_metric"]["teacher_forced_test"]["h1_accuracy"]["mean"],
            "h7_f1_test_mean": agg["by_metric"]["teacher_forced_test"]["h7_f1"]["mean"],
            "valid_graph_rate_mean": agg["by_metric"]["free_test"]["valid_graph_rate"]["mean"],
            "canonical_exact_graph_mean": agg["by_metric"]["free_test"][
                "canonical_exact_graph_accuracy"
            ]["mean"],
            "execution_mode_accuracy_valid_only_mean": agg["by_metric"]["free_test"][
                "execution_mode_accuracy_valid_only"
            ]["mean"],
            "execution_mode_accuracy_all_examples_mean": agg["by_metric"]["free_test"][
                "execution_mode_accuracy_all_examples"
            ]["mean"],
        }

    out_path = output_root / "stage_a_v2_experiment_report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_pipeline(
    *,
    output_root: Path,
    phases: set[str],
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    if "multiseed" in phases:
        for seed in seeds:
            _train_and_eval_seed(seed, output_root)

    if "baselines" in phases:
        split, _, _ = load_and_split_stage_a_v2(_v2_train_config(20260901, output_root))
        baseline_results = run_v2_baselines(
            train_examples=split.train,
            eval_examples=split.test,
        )
        (output_root / "baselines.json").write_text(
            json.dumps(baseline_results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if "ablations" in phases:
        ablation_specs = {
            "no_h5": ("h5",),
            "no_h6": ("h6",),
            "no_h7": ("h7",),
            "no_h4": ("h4",),
        }
        for name, disabled in ablation_specs.items():
            _run_ablation(name, disabled, output_root)

    if "regenerate" in phases:
        split, _, _ = load_and_split_stage_a_v2(_v2_train_config(20260901, output_root))
        baseline_results = run_v2_baselines(
            train_examples=split.train,
            eval_examples=split.test,
        )
        (output_root / "baselines.json").write_text(
            json.dumps(baseline_results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for seed in seeds:
            _eval_seed_record(seed, output_root)
        for name in ("no_h4", "no_h5", "no_h6", "no_h7"):
            ablation_dir = output_root / "ablations" / name
            if (ablation_dir / "best.pt").is_file():
                _eval_ablation_record(name, output_root)

    if "report" in phases:
        return build_final_report(output_root)
    return build_final_report(output_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="multiseed,baselines,ablations,report",
        help="comma-separated: multiseed,baselines,ablations,regenerate,report",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    args = parser.parse_args(argv)
    phases = {part.strip() for part in args.phases.split(",") if part.strip()}
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    report = run_pipeline(output_root=args.output_dir, phases=phases, seeds=seeds)
    print(json.dumps(report.get("headline_numbers", {}), indent=2, sort_keys=True))
    print(f"report: {args.output_dir / 'stage_a_v2_experiment_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
