"""
scripts/compare_classifiers.py — Side-by-Side Classifier Comparison
====================================================================
Trains and evaluates all four backends on the same stratified split,
then prints a unified comparison table.

Usage:
    python scripts/compare_classifiers.py
    python scripts/compare_classifiers.py --backends cosine minilm_lr
    python scripts/compare_classifiers.py --test-size 0.25 --seed 0

Output example:
    Backend        Macro F1   Env F1  Mix F1  Per F1  Latency(ms)  Size(MB)  Errors
    ────────────────────────────────────────────────────────────────────────────────
    cosine           0.712    0.680   0.648   0.808       2.1         0.0       28
    minilm_lr        0.912    0.940   0.908   0.887       7.3         0.1       5
    setfit           0.931    0.952   0.926   0.915      18.4       412.0       4
    fastfit          0.938    0.960   0.934   0.921      12.1       380.0       3
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support
from collections import Counter

from scripts.evaluate import load_data, run_evaluation

TRAIN_PATH  = "dataset/training_data.json"
ALL_BACKENDS = ["cosine", "minilm_lr", "setfit", "fastfit"]
LABELS       = ["Environmental", "Mixed", "Personal"]


def run_backend(backend: str, train_q, train_l, test_q, test_l) -> dict:
    """Train and evaluate one backend. Returns metrics dict or error info."""
    print(f"\n  [{backend}] training...", flush=True)
    t_train = time.perf_counter()

    try:
        from router.classifier import QueryClassifier
        clf = QueryClassifier(backend=backend)
        clf.train(train_q, train_l)
        train_sec = time.perf_counter() - t_train
        print(f"  [{backend}] trained in {train_sec:.1f}s. Evaluating...", flush=True)

        metrics = run_evaluation(test_q, test_l, clf)
        metrics["train_sec"]   = train_sec
        metrics["model_size"]  = clf._backend.model_size_mb
        metrics["backend"]     = backend
        metrics["ok"]          = True

        # Per-class F1
        preds = metrics["predictions"]
        _, _, f1s, _ = precision_recall_fscore_support(
            test_l, preds, labels=LABELS, average=None, zero_division=0
        )
        metrics["per_class_f1"] = {l: float(f) for l, f in zip(LABELS, f1s)}

        print(f"  [{backend}] Macro F1={metrics['macro_f1']:.3f}  "
              f"latency={metrics['avg_latency_ms']:.1f}ms")
        return metrics

    except ImportError as e:
        print(f"  [{backend}] SKIPPED — {e}")
        return {"backend": backend, "ok": False, "error": str(e)}
    except Exception as e:
        print(f"  [{backend}] FAILED — {e}")
        return {"backend": backend, "ok": False, "error": str(e)}


def print_comparison(results: list, test_l: list):
    print("\n")
    print("=" * 84)
    print("  CLASSIFIER COMPARISON")
    print("=" * 84)

    # Header
    print(
        f"  {'Backend':<14} {'MacroF1':>8} {'Env F1':>8} {'Mix F1':>8} "
        f"{'Per F1':>8} {'Lat(ms)':>9} {'Train(s)':>9} {'Size(MB)':>9} {'Errors':>7}"
    )
    print("  " + "─" * 80)

    for r in results:
        if not r["ok"]:
            print(f"  {r['backend']:<14}  {'SKIPPED — ' + r.get('error','')[:55]}")
            continue

        cf1 = r["per_class_f1"]
        print(
            f"  {r['backend']:<14} "
            f"{r['macro_f1']:>8.3f} "
            f"{cf1.get('Environmental', 0):>8.3f} "
            f"{cf1.get('Mixed', 0):>8.3f} "
            f"{cf1.get('Personal', 0):>8.3f} "
            f"{r['avg_latency_ms']:>9.1f} "
            f"{r['train_sec']:>9.1f} "
            f"{r['model_size']:>9.1f} "
            f"{r['n_errors']:>7}"
        )

    print("  " + "─" * 80)
    print(f"\n  Test set: {len(test_l)} queries  |  "
          f"Distribution: {dict(Counter(test_l))}")

    # Best backend
    ok = [r for r in results if r["ok"]]
    if ok:
        best = max(ok, key=lambda r: r["macro_f1"])
        print(f"\n  Best Macro F1: {best['backend']}  ({best['macro_f1']:.3f})")

        fastest = min(ok, key=lambda r: r["avg_latency_ms"])
        print(f"  Fastest      : {fastest['backend']}  ({fastest['avg_latency_ms']:.1f} ms/query)")

    print("=" * 84)

    # Misclassification deep-dive per backend
    print("\n  PER-BACKEND MISCLASSIFICATIONS (first 5 each)")
    print("  " + "─" * 80)
    for r in results:
        if not r["ok"] or not r.get("errors"):
            continue
        print(f"\n  [{r['backend']}]  {r['n_errors']} errors")
        for q, actual, pred, conf in r["errors"][:5]:
            print(f"    Actual={actual:15} Pred={pred:15} conf={conf:.2f}  '{q[:48]}'")


def main():
    parser = argparse.ArgumentParser(description="Compare all classifier backends")
    parser.add_argument(
        "--backends", nargs="+", default=ALL_BACKENDS,
        choices=ALL_BACKENDS,
        help="Backends to compare (default: all four)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data held out for testing (default: 0.2)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_q, all_l = load_data(TRAIN_PATH)

    train_q, test_q, train_l, test_l = train_test_split(
        all_q, all_l,
        test_size=args.test_size,
        stratify=all_l,
        random_state=args.seed,
    )

    print(f"Stratified split: {len(train_q)} train / {len(test_q)} test")
    print(f"Train: {Counter(train_l)}  |  Test: {Counter(test_l)}")
    print(f"\nRunning {len(args.backends)} backends: {args.backends}")

    results = []
    for backend in args.backends:
        r = run_backend(backend, train_q, train_l, test_q, test_l)
        results.append(r)

    print_comparison(results, test_l)


if __name__ == "__main__":
    main()
