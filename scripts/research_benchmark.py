"""
scripts/research_benchmark.py - Research classifier benchmark
=============================================================

Compares the three research-relevant classifier backends on the same
stratified split:

  - minilm_lr
  - setfit
  - fastfit

The benchmark reports metrics aligned with the routing paper/project goals:

  - Personal recall
  - Mixed-query F1
  - Classifier latency on edge
  - Saved model size and optional process RSS memory
  - Personal-to-fog error rate, including Mixed queries sent fully to fog

Usage:
    python scripts/research_benchmark.py
    python scripts/research_benchmark.py --test-size 0.25 --seed 0
    python scripts/research_benchmark.py --backends minilm_lr setfit fastfit
    python scripts/research_benchmark.py --json-output results.json

Notes:
    SetFit and FastFit are optional dependencies. Missing backends are skipped.
    Runtime RSS is reported only when psutil is installed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from scripts.evaluate import load_data


TRAIN_PATH = "dataset/training_data.json"
LABELS = ["Environmental", "Mixed", "Personal"]
RESEARCH_BACKENDS = ["minilm_lr", "setfit", "fastfit"]

# Safety-heavy by design: leaking personal queries to fog is more expensive
# than a small latency or size difference.
BALANCE_WEIGHTS = {
    "personal_recall": 0.30,
    "personal_to_fog_safety": 0.25,
    "mixed_f1": 0.25,
    "latency": 0.10,
    "model_size": 0.10,
}


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(ordered[int(k)])
    return float(ordered[lo] * (hi - k) + ordered[hi] * (k - lo))


def _read_split(path: str) -> tuple[list[str], list[str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [d["query"] for d in data], [d["label"] for d in data]


def _write_split(path: Path, queries: list[str], labels: list[str]) -> None:
    payload = [{"query": q, "label": l} for q, l in zip(queries, labels)]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@contextmanager
def _preserve_model_dir(backend: str, tmp_path: Path):
    """Restore model artifacts after benchmarking unless explicitly kept."""
    model_dir = Path("models") / "classifier" / backend
    backup_dir = tmp_path / f"{backend}_model_backup"
    had_original = model_dir.exists()

    if had_original:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(model_dir, backup_dir)

    try:
        yield
    finally:
        if model_dir.exists():
            shutil.rmtree(model_dir)
        if had_original:
            shutil.copytree(backup_dir, model_dir)


def _run_worker(
    backend: str,
    train_file: str,
    test_file: str,
    warmup: int,
) -> dict[str, Any]:
    t_train = time.perf_counter()
    rss_start = _rss_mb()

    from router.classifier import QueryClassifier

    train_q, train_l = _read_split(train_file)
    test_q, test_l = _read_split(test_file)

    clf = QueryClassifier(backend=backend)
    clf.train(train_q, train_l)
    train_sec = time.perf_counter() - t_train
    rss_after_train = _rss_mb()

    # Warm up the encoder/head so latency reflects steady-state edge inference.
    for q in train_q[: max(0, warmup)]:
        clf.predict(q)

    preds: list[str] = []
    confs: list[float] = []
    latencies: list[float] = []

    for q in test_q:
        t0 = time.perf_counter()
        result = clf.predict(q)
        latencies.append((time.perf_counter() - t0) * 1000)
        preds.append(result.label)
        confs.append(float(result.confidence))

    rss_after_eval = _rss_mb()

    precision, recall, f1, support = precision_recall_fscore_support(
        test_l, preds, labels=LABELS, average=None, zero_division=0
    )
    per_label = {
        label: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for label, p, r, f, s in zip(LABELS, precision, recall, f1, support)
    }

    n_privacy_bearing = sum(1 for label in test_l if label in ("Personal", "Mixed"))
    personal_to_fog = [
        i
        for i, (truth, pred) in enumerate(zip(test_l, preds))
        if (truth == "Personal" and pred != "Personal")
        or (truth == "Mixed" and pred == "Environmental")
    ]
    direct_personal_to_fog = [
        i
        for i, (truth, pred) in enumerate(zip(test_l, preds))
        if truth in ("Personal", "Mixed") and pred == "Environmental"
    ]
    errors = [
        {
            "query": test_q[i],
            "actual": test_l[i],
            "predicted": preds[i],
            "confidence": confs[i],
        }
        for i in range(len(test_q))
        if preds[i] != test_l[i]
    ]

    return {
        "backend": backend,
        "ok": True,
        "n_train": len(train_q),
        "n_test": len(test_q),
        "train_sec": float(train_sec),
        "avg_latency_ms": float(statistics.fmean(latencies)) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 95),
        "model_size_mb": float(clf._backend.model_size_mb),
        "rss_start_mb": rss_start,
        "rss_after_train_mb": rss_after_train,
        "rss_after_eval_mb": rss_after_eval,
        "rss_delta_mb": (
            float(rss_after_eval - rss_start)
            if rss_after_eval is not None and rss_start is not None
            else None
        ),
        "personal_recall": per_label["Personal"]["recall"],
        "mixed_f1": per_label["Mixed"]["f1"],
        "personal_to_fog_count": len(personal_to_fog),
        "personal_to_fog_rate": (
            len(personal_to_fog) / n_privacy_bearing if n_privacy_bearing else 0.0
        ),
        "direct_personal_to_fog_count": len(direct_personal_to_fog),
        "direct_personal_to_fog_rate": (
            len(direct_personal_to_fog) / n_privacy_bearing if n_privacy_bearing else 0.0
        ),
        "per_label": per_label,
        "n_errors": len(errors),
        "errors": errors[:20],
        "privacy_errors": [
            errors_i
            for errors_i in errors
            if (errors_i["actual"] == "Personal" and errors_i["predicted"] != "Personal")
            or (errors_i["actual"] == "Mixed" and errors_i["predicted"] == "Environmental")
        ][:20],
        "test_distribution": dict(Counter(test_l)),
    }


def _worker_main(args: argparse.Namespace) -> int:
    try:
        manager = (
            nullcontext()
            if args.keep_trained_models
            else _preserve_model_dir(args.backend, Path(args.output_file).parent)
        )
        with manager:
            result = _run_worker(
                backend=args.backend,
                train_file=args.train_file,
                test_file=args.test_file,
                warmup=args.warmup,
            )
    except ImportError as e:
        result = {"backend": args.backend, "ok": False, "error": str(e)}
    except Exception as e:
        result = {"backend": args.backend, "ok": False, "error": repr(e)}

    Path(args.output_file).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


def _inverse_scores(results: list[dict[str, Any]], key: str) -> dict[str, float]:
    ok = [r for r in results if r.get("ok")]
    values = [float(r[key]) for r in ok]
    if not values:
        return {}
    best = min(values)
    worst = max(values)
    if math.isclose(best, worst):
        return {r["backend"]: 1.0 for r in ok}
    return {
        r["backend"]: max(0.0, 1.0 - (float(r[key]) - best) / (worst - best))
        for r in ok
    }


def _add_balance_scores(results: list[dict[str, Any]]) -> None:
    latency_scores = _inverse_scores(results, "avg_latency_ms")
    size_scores = _inverse_scores(results, "model_size_mb")

    for r in results:
        if not r.get("ok"):
            continue
        score = (
            BALANCE_WEIGHTS["personal_recall"] * r["personal_recall"]
            + BALANCE_WEIGHTS["personal_to_fog_safety"] * (1.0 - r["personal_to_fog_rate"])
            + BALANCE_WEIGHTS["mixed_f1"] * r["mixed_f1"]
            + BALANCE_WEIGHTS["latency"] * latency_scores.get(r["backend"], 0.0)
            + BALANCE_WEIGHTS["model_size"] * size_scores.get(r["backend"], 0.0)
        )
        r["balance_score"] = float(score)


def _print_table(results: list[dict[str, Any]], test_l: list[str]) -> None:
    _add_balance_scores(results)
    print()
    print("=" * 120)
    print("  RESEARCH CLASSIFIER BENCHMARK")
    print("=" * 120)
    print(
        f"  {'Backend':<12} {'PersRec':>8} {'MixedF1':>8} {'P->Fog':>8} "
        f"{'DirectP->Fog':>12} {'LatAvg':>9} {'LatP95':>9} {'SizeMB':>8} "
        f"{'RSSDelta':>9} {'Score':>8} {'Errors':>7}"
    )
    print("  " + "-" * 116)

    for r in results:
        if not r.get("ok"):
            print(f"  {r['backend']:<12}  SKIPPED - {r.get('error', '')[:90]}")
            continue
        rss_delta = r["rss_delta_mb"]
        rss_text = f"{rss_delta:>9.1f}" if rss_delta is not None else f"{'n/a':>9}"
        print(
            f"  {r['backend']:<12} "
            f"{r['personal_recall']:>8.3f} "
            f"{r['mixed_f1']:>8.3f} "
            f"{r['personal_to_fog_rate']:>8.3f} "
            f"{r['direct_personal_to_fog_rate']:>12.3f} "
            f"{r['avg_latency_ms']:>9.1f} "
            f"{r['p95_latency_ms']:>9.1f} "
            f"{r['model_size_mb']:>8.1f} "
            f"{rss_text} "
            f"{r['balance_score']:>8.3f} "
            f"{r['n_errors']:>7}"
        )

    print("  " + "-" * 116)
    print(f"  Test set: {len(test_l)} queries | Distribution: {dict(Counter(test_l))}")
    print(
        "  Balance weights: "
        + ", ".join(f"{k}={v:.2f}" for k, v in BALANCE_WEIGHTS.items())
    )
    print("  P->Fog counts true Personal predicted as Mixed/Environmental and true Mixed predicted as Environmental.")
    print("  DirectP->Fog counts true Personal or Mixed predicted as Environmental.")

    ok = [r for r in results if r.get("ok")]
    if ok:
        best = max(ok, key=lambda r: r["balance_score"])
        safest = min(ok, key=lambda r: (r["personal_to_fog_rate"], -r["personal_recall"]))
        mixed = max(ok, key=lambda r: r["mixed_f1"])
        fastest = min(ok, key=lambda r: r["avg_latency_ms"])
        smallest = min(ok, key=lambda r: r["model_size_mb"])
        print()
        print(f"  Best balance : {best['backend']} ({best['balance_score']:.3f})")
        print(
            f"  Safest       : {safest['backend']} "
            f"(P->Fog={safest['personal_to_fog_rate']:.3f}, "
            f"Personal recall={safest['personal_recall']:.3f})"
        )
        print(f"  Best Mixed F1: {mixed['backend']} ({mixed['mixed_f1']:.3f})")
        print(f"  Fastest      : {fastest['backend']} ({fastest['avg_latency_ms']:.1f} ms)")
        print(f"  Smallest     : {smallest['backend']} ({smallest['model_size_mb']:.1f} MB)")

    print("=" * 120)

    for r in ok:
        if not r.get("privacy_errors"):
            continue
        print(f"\n  [{r['backend']}] Personal-to-fog risk examples")
        for item in r["privacy_errors"][:5]:
            print(
                f"    Pred={item['predicted']:<13} conf={item['confidence']:.2f} "
                f"'{item['query'][:80]}'"
            )


def _run_parent(args: argparse.Namespace) -> list[dict[str, Any]]:
    all_q, all_l = load_data(TRAIN_PATH)
    train_q, test_q, train_l, test_l = train_test_split(
        all_q,
        all_l,
        test_size=args.test_size,
        stratify=all_l,
        random_state=args.seed,
    )

    print(f"Stratified split: {len(train_q)} train / {len(test_q)} test")
    print(f"Train: {dict(Counter(train_l))} | Test: {dict(Counter(test_l))}")
    print(f"Backends: {args.backends}")

    with tempfile.TemporaryDirectory(prefix="semantic_router_bench_") as tmp:
        tmp_path = Path(tmp)
        train_file = tmp_path / "train.json"
        test_file = tmp_path / "test.json"
        _write_split(train_file, train_q, train_l)
        _write_split(test_file, test_q, test_l)

        results: list[dict[str, Any]] = []
        script_path = Path(__file__).resolve()
        for backend in args.backends:
            print(f"\n[{backend}] running isolated benchmark...", flush=True)
            output_file = tmp_path / f"{backend}.json"
            cmd = [
                sys.executable,
                str(script_path),
                "--worker",
                "--backend",
                backend,
                "--train-file",
                str(train_file),
                "--test-file",
                str(test_file),
                "--output-file",
                str(output_file),
                "--warmup",
                str(args.warmup),
            ]
            if args.keep_trained_models:
                cmd.append("--keep-trained-models")
            completed = subprocess.run(cmd, text=True)
            if completed.returncode != 0:
                results.append(
                    {
                        "backend": backend,
                        "ok": False,
                        "error": f"worker exited with code {completed.returncode}",
                    }
                )
                continue
            results.append(json.loads(output_file.read_text(encoding="utf-8")))

    _print_table(results, test_l)

    if args.json_output:
        payload = {
            "train_size": len(train_q),
            "test_size": len(test_q),
            "test_distribution": dict(Counter(test_l)),
            "balance_weights": BALANCE_WEIGHTS,
            "results": results,
        }
        Path(args.json_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON results to {args.json_output}")

    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research benchmark for classifier backends")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=RESEARCH_BACKENDS,
        choices=RESEARCH_BACKENDS,
        help="Backends to compare (default: minilm_lr setfit fastfit)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument(
        "--keep-trained-models",
        action="store_true",
        help="Keep trained benchmark models in models/classifier instead of restoring prior artifacts",
    )

    # Internal worker mode. Called by the parent process so each backend gets
    # isolated runtime memory measurements.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=RESEARCH_BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--train-file", help=argparse.SUPPRESS)
    parser.add_argument("--test-file", help=argparse.SUPPRESS)
    parser.add_argument("--output-file", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        missing = [
            name
            for name in ("backend", "train_file", "test_file", "output_file")
            if getattr(args, name) is None
        ]
        if missing:
            raise SystemExit(f"Missing worker argument(s): {', '.join(missing)}")
        return _worker_main(args)

    _run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
