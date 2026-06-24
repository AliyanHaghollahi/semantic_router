"""
scripts/evaluate.py — Comprehensive Evaluation
===============================================
Evaluates classifier backends + privacy, answer quality, and resource efficiency.

Metrics (per backend):
  - Per-class Precision, Recall, F1
  - Macro F1 (main metric)
  - Confusion matrix
  - Per-class accuracy
  - Avg inference latency (ms)
  - Saved model size (MB)
  - Confidence distribution (correct vs incorrect)
  - Top misclassified examples

Full evaluation (--full):
  - Privacy preservation: Personal queries must never route to fog
  - Answer quality: keyword recall + route correctness on simulation responses
  - Resource efficiency: CPU, RAM, P95 latency, throughput

Usage:
    python scripts/evaluate.py                            # default: minilm_lr
    python scripts/evaluate.py --backend cosine
    python scripts/evaluate.py --backend all              # all four backends
    python scripts/evaluate.py --backend minilm_lr --test path/to/test.json
    python scripts/evaluate.py --backend minilm_lr --retrain
    python scripts/evaluate.py --full                     # privacy + quality + resources
"""

import sys
import os
import json
import time
import asyncio
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# ── Privacy test queries — must NEVER be routed to fog ────────────
PRIVACY_TEST_QUERIES = [
    "What is my blood pressure medication?",
    "Show me my passport details",
    "What is my insurance plan?",
    "When is my next doctor appointment?",
    "What are my allergies?",
    "What is my bank account number?",
    "Read my latest calendar entry",
    "What medication am I currently taking?",
    "What is my flight booking reference?",
    "Show me my contact list",
    "What is my home address?",
    "What is my phone number?",
    "Tell me my prescription dosage",
    "What is my diagnosis?",
    "Show me my health records",
]

# ── Answer quality test cases (simulation mode) ───────────────────
QUALITY_TEST_CASES = [
    {
        "query": "where is my gate",
        "expected_route_prefix": "mixed",
        "keywords": ["gate", "terminal", "walk"],
    },
    {
        "query": "is there a pharmacy nearby",
        "expected_route_prefix": "fog",
        "keywords": ["pharmacy", "meters", "open"],
    },
    {
        "query": "what is my blood pressure medication",
        "expected_route_prefix": "edge",
        "keywords": ["medication", "mg"],
    },
    {
        "query": "what medication am I holding and is there a pharmacy nearby",
        "expected_route_prefix": "mixed",
        "keywords": ["pharmacy", "medication"],
    },
    {
        "query": "where am I",
        "expected_route_prefix": "fog",
        "keywords": ["terminal", "hall", "area", "departures"],
    },
    {
        "query": "when is my next appointment and is there a café nearby",
        "expected_route_prefix": "mixed",
        "keywords": ["appointment", "café"],
    },
]

LABELS       = ["Environmental", "Mixed", "Personal"]
ALL_BACKENDS = ["cosine", "minilm_lr", "setfit", "fastfit"]
TRAIN_PATH   = "dataset/training_data.json"


def load_data(path: str):
    with open(path) as f:
        data = json.load(f)
    return [d["query"] for d in data], [d["label"] for d in data]


def build_classifier(backend: str, retrain: bool, train_queries, train_labels):
    from router.classifier import QueryClassifier
    clf = QueryClassifier(backend=backend)
    if retrain:
        print(f"  Training {backend} on {len(train_queries)} examples...")
        clf.train(train_queries, train_labels)
    else:
        clf.load_or_train(TRAIN_PATH)
    return clf


def run_evaluation(test_queries: list, test_labels: list, clf) -> dict:
    predictions, confidences, latencies = [], [], []

    for q in test_queries:
        t0 = time.perf_counter()
        result = clf.predict(q)
        latencies.append((time.perf_counter() - t0) * 1000)
        predictions.append(result.label)
        confidences.append(result.confidence)

    macro_f1 = f1_score(test_labels, predictions, average="macro", zero_division=0)
    avg_lat  = float(np.mean(latencies))
    avg_conf = float(np.mean(confidences))

    correct_conf   = [c for c, p, t in zip(confidences, predictions, test_labels) if p == t]
    incorrect_conf = [c for c, p, t in zip(confidences, predictions, test_labels) if p != t]

    errors = [
        (test_queries[i], test_labels[i], predictions[i], confidences[i])
        for i in range(len(test_queries))
        if predictions[i] != test_labels[i]
    ]

    per_class = {}
    for label in LABELS:
        idx = [i for i, t in enumerate(test_labels) if t == label]
        correct = sum(1 for i in idx if predictions[i] == label)
        per_class[label] = (correct, len(idx))

    _, _, f1s, _ = precision_recall_fscore_support(
        test_labels, predictions, labels=LABELS, average=None, zero_division=0
    )
    per_class_f1 = {l: float(f) for l, f in zip(LABELS, f1s)}

    return {
        "macro_f1":       macro_f1,
        "avg_latency_ms": avg_lat,
        "avg_confidence": avg_conf,
        "n_test":         len(test_queries),
        "n_errors":       len(errors),
        "predictions":    predictions,
        "confidences":    confidences,
        "correct_conf":   correct_conf,
        "incorrect_conf": incorrect_conf,
        "per_class":      per_class,
        "per_class_f1":   per_class_f1,
        "errors":         errors,
    }


def print_report(backend: str, test_labels, metrics: dict, model_size_mb: float):
    preds = metrics["predictions"]
    print("\n" + "=" * 65)
    print(f"  EVALUATION REPORT — {backend.upper()}")
    print("=" * 65)
    print(f"  Test set      : {metrics['n_test']} queries")
    print(f"  Avg latency   : {metrics['avg_latency_ms']:.1f} ms")
    print(f"  Avg confidence: {metrics['avg_confidence']:.3f}")
    print(f"  Model size    : {model_size_mb:.1f} MB  (saved files, excl. encoder)")
    print()

    print(classification_report(test_labels, preds, labels=LABELS, digits=3))
    print(f"  Macro F1 : {metrics['macro_f1']:.3f}")

    # Confusion matrix
    cm = confusion_matrix(test_labels, preds, labels=LABELS)
    print()
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':>16} {'Env':>6} {'Mixed':>6} {'Personal':>9}")
    for i, label in enumerate(LABELS):
        row = "  ".join(f"{v:6d}" for v in cm[i])
        print(f"  {label:16} {row}")

    # Confidence split
    print()
    if metrics["correct_conf"]:
        print(f"  Avg confidence (correct)  : {np.mean(metrics['correct_conf']):.3f}")
    if metrics["incorrect_conf"]:
        print(f"  Avg confidence (incorrect): {np.mean(metrics['incorrect_conf']):.3f}")
    else:
        print("  No misclassifications!")

    # Per-class accuracy
    print()
    print("  Per-class accuracy:")
    for label in LABELS:
        correct, total = metrics["per_class"][label]
        pct = correct / total * 100 if total else 0
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {label:15}: {correct}/{total}  ({pct:5.1f}%)  {bar}")

    # Misclassified examples
    if metrics["errors"]:
        print(f"\n  Misclassified examples ({len(metrics['errors'])} total):")
        for q, actual, pred, conf in metrics["errors"][:10]:
            print(f"    Actual={actual:15} Pred={pred:15} conf={conf:.2f}  '{q[:52]}'")
    else:
        print("\n  No misclassifications!")

    print("=" * 65)


def print_comparison_table(results: list, test_labels: list):
    print("\n\n" + "=" * 88)
    print("  COMPARISON SUMMARY")
    print("=" * 88)
    print(
        f"  {'Backend':<14} {'MacroF1':>8} {'Env F1':>8} {'Mix F1':>8} "
        f"{'Per F1':>8} {'Lat(ms)':>9} {'Size(MB)':>9} {'Errors':>7}"
    )
    print("  " + "─" * 84)

    for r in results:
        if not r["ok"]:
            print(f"  {r['backend']:<14}  SKIPPED — {r.get('error', '')[:55]}")
            continue
        cf1 = r["per_class_f1"]
        print(
            f"  {r['backend']:<14} "
            f"{r['macro_f1']:>8.3f} "
            f"{cf1.get('Environmental', 0):>8.3f} "
            f"{cf1.get('Mixed', 0):>8.3f} "
            f"{cf1.get('Personal', 0):>8.3f} "
            f"{r['avg_latency_ms']:>9.1f} "
            f"{r['model_size_mb']:>9.1f} "
            f"{r['n_errors']:>7}"
        )

    print("  " + "─" * 84)
    print(f"  Test set: {len(test_labels)} queries  |  Distribution: {dict(Counter(test_labels))}")

    ok = [r for r in results if r["ok"]]
    if ok:
        best    = max(ok, key=lambda r: r["macro_f1"])
        fastest = min(ok, key=lambda r: r["avg_latency_ms"])
        print(f"\n  Best Macro F1 : {best['backend']}  ({best['macro_f1']:.3f})")
        print(f"  Fastest       : {fastest['backend']}  ({fastest['avg_latency_ms']:.1f} ms/query)")

    print("=" * 88)


# ── Privacy Preservation ──────────────────────────────────────────

def evaluate_privacy(clf) -> dict:
    """Personal queries classified as Environmental = privacy leak (sent to fog)."""
    leaks = []
    for query in PRIVACY_TEST_QUERIES:
        result = clf.predict(query)
        if result.label == "Environmental":
            leaks.append((query, result.confidence))
    n = len(PRIVACY_TEST_QUERIES)
    return {
        "n_tested": n,
        "n_leaks": len(leaks),
        "privacy_score": 1.0 - len(leaks) / n,
        "leaks": leaks,
    }


def print_privacy_report(metrics: dict):
    print("\n" + "=" * 65)
    print("  PRIVACY PRESERVATION")
    print("=" * 65)
    score = metrics["privacy_score"]
    bar_len = int(score * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    print(f"  Privacy score : {score:.3f}  {bar}  ({metrics['n_tested'] - metrics['n_leaks']}/{metrics['n_tested']} safe)")
    if metrics["leaks"]:
        print(f"\n  LEAKS — Personal queries routed to Fog ({len(metrics['leaks'])}):")
        for query, conf in metrics["leaks"]:
            print(f"    conf={conf:.2f}  '{query[:60]}'")
    else:
        print("\n  No privacy leaks detected.")
    print("=" * 65)


# ── Resource Efficiency ───────────────────────────────────────────

def evaluate_resources(clf, queries: list) -> dict:
    """Measure CPU, RAM, P95 latency, and throughput during inference."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        has_psutil = True
    except ImportError:
        has_psutil = False

    mem_before = process.memory_info().rss / 1024 / 1024 if has_psutil else 0
    latencies = []

    for q in queries:
        t0 = time.perf_counter()
        clf.predict(q)
        latencies.append((time.perf_counter() - t0) * 1000)

    mem_after = process.memory_info().rss / 1024 / 1024 if has_psutil else 0
    total_sec = sum(latencies) / 1000

    result = {
        "n_queries":        len(queries),
        "avg_latency_ms":   float(np.mean(latencies)),
        "p95_latency_ms":   float(np.percentile(latencies, 95)),
        "min_latency_ms":   float(np.min(latencies)),
        "max_latency_ms":   float(np.max(latencies)),
        "throughput_qps":   len(queries) / total_sec if total_sec > 0 else 0,
        "peak_memory_mb":   mem_after,
        "memory_delta_mb":  mem_after - mem_before,
        "psutil_available": has_psutil,
    }
    return result


def print_resource_report(metrics: dict):
    print("\n" + "=" * 65)
    print("  RESOURCE EFFICIENCY")
    print("=" * 65)
    print(f"  Queries tested  : {metrics['n_queries']}")
    print(f"  Avg latency     : {metrics['avg_latency_ms']:.1f} ms")
    print(f"  P95 latency     : {metrics['p95_latency_ms']:.1f} ms")
    print(f"  Min / Max       : {metrics['min_latency_ms']:.1f} / {metrics['max_latency_ms']:.1f} ms")
    print(f"  Throughput      : {metrics['throughput_qps']:.1f} queries/sec")
    if metrics["psutil_available"]:
        print(f"  Peak RAM        : {metrics['peak_memory_mb']:.1f} MB")
        print(f"  RAM delta       : {metrics['memory_delta_mb']:+.1f} MB")
    else:
        print("  RAM             : install psutil for memory metrics  (pip install psutil)")
    print("=" * 65)


# ── Answer Quality ────────────────────────────────────────────────

def evaluate_answer_quality() -> dict:
    """Run test queries through the pipeline in simulation mode, check keyword recall."""
    from edge.pipeline import RoutingPipeline
    pipeline = RoutingPipeline.from_config(config_override={"simulation_mode": True})

    cases = []
    for tc in QUALITY_TEST_CASES:
        t0 = time.perf_counter()
        try:
            result = asyncio.get_event_loop().run_until_complete(pipeline.process(tc["query"]))
            latency_ms = (time.perf_counter() - t0) * 1000
            response = result.final_response.lower()
            route_ok = result.route.startswith(tc["expected_route_prefix"])
            found = [k for k in tc["keywords"] if k in response]
            keyword_recall = len(found) / len(tc["keywords"])
        except Exception as e:
            latency_ms = 0.0
            route_ok = False
            found = []
            keyword_recall = 0.0
            result = None

        cases.append({
            "query":            tc["query"],
            "expected_route":   tc["expected_route_prefix"],
            "actual_route":     result.route if result else "error",
            "route_correct":    route_ok,
            "keywords_expected": tc["keywords"],
            "keywords_found":   found,
            "keyword_recall":   keyword_recall,
            "latency_ms":       latency_ms,
        })

    return {
        "cases":               cases,
        "avg_keyword_recall":  float(np.mean([c["keyword_recall"] for c in cases])),
        "route_accuracy":      float(np.mean([c["route_correct"] for c in cases])),
    }


def print_quality_report(metrics: dict):
    print("\n" + "=" * 65)
    print("  ANSWER QUALITY  (simulation mode)")
    print("=" * 65)
    print(f"  Route accuracy    : {metrics['route_accuracy']:.3f}")
    print(f"  Avg keyword recall: {metrics['avg_keyword_recall']:.3f}")
    print()
    for c in metrics["cases"]:
        route_sym = "✓" if c["route_correct"] else "✗"
        recall_bar = "█" * int(c["keyword_recall"] * 10) + "░" * (10 - int(c["keyword_recall"] * 10))
        print(f"  {route_sym} [{c['actual_route']:18}] recall={c['keyword_recall']:.2f} {recall_bar}  '{c['query'][:40]}'")
        if c["keyword_recall"] < 1.0:
            missing = [k for k in c["keywords_expected"] if k not in c["keywords_found"]]
            print(f"      missing keywords: {missing}")
    print("=" * 65)


def evaluate_backend(backend: str, train_q, train_l, test_q, test_l, retrain: bool) -> dict:
    try:
        clf = build_classifier(backend, retrain=retrain, train_queries=train_q, train_labels=train_l)
        metrics = run_evaluation(test_q, test_l, clf)
        model_size = clf._backend.model_size_mb
        print_report(backend, test_l, metrics, model_size)
        return {**metrics, "backend": backend, "model_size_mb": model_size, "ok": True}
    except ImportError as e:
        print(f"\n  [{backend}] SKIPPED — {e}")
        return {"backend": backend, "ok": False, "error": str(e)}
    except Exception as e:
        print(f"\n  [{backend}] FAILED — {e}")
        return {"backend": backend, "ok": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier backend(s)")
    parser.add_argument(
        "--backend", nargs="+", default=["minilm_lr"],
        metavar="BACKEND",
        help=(
            "Backend(s) to evaluate. Choices: cosine minilm_lr setfit fastfit all. "
            "Examples: --backend setfit  |  --backend all  |  --backend cosine minilm_lr"
        ),
    )
    parser.add_argument(
        "--test", type=str, default=None,
        help="Path to a separate test JSON file. If not given, splits training data.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of training data to hold out as test set (default: 0.2)",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retraining even if a saved model exists",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full", action="store_true",
        help="Run full evaluation: privacy preservation, answer quality, resource efficiency",
    )
    args = parser.parse_args()

    # Resolve backend list
    backends = ALL_BACKENDS if "all" in args.backend else args.backend
    invalid = [b for b in backends if b not in ALL_BACKENDS]
    if invalid:
        parser.error(f"Unknown backend(s): {invalid}. Choose from: {ALL_BACKENDS + ['all']}")

    all_queries, all_labels = load_data(TRAIN_PATH)

    # Build train/test split
    if args.test:
        train_q, train_l = all_queries, all_labels
        test_q, test_l   = load_data(args.test)
        print(f"Using external test file: {args.test}  ({len(test_q)} examples)")
    else:
        train_q, test_q, train_l, test_l = train_test_split(
            all_queries, all_labels,
            test_size=args.test_size,
            stratify=all_labels,
            random_state=args.seed,
        )
        print(f"Stratified split: {len(train_q)} train / {len(test_q)} test")
        print(f"Train: {Counter(train_l)}  |  Test: {Counter(test_l)}")

    print(f"\nEvaluating: {backends}\n")

    results = []
    for backend in backends:
        print(f"\n{'─' * 65}")
        print(f"  Backend: {backend}")
        print(f"{'─' * 65}")
        r = evaluate_backend(
            backend, train_q, train_l, test_q, test_l,
            retrain=(args.retrain or args.test is None),
        )
        results.append(r)

    # Print comparison table only when evaluating more than one backend
    if len(backends) > 1:
        print_comparison_table(results, test_l)

    # Full evaluation: privacy, quality, resources
    if args.full:
        ok_results = [r for r in results if r.get("ok")]
        if ok_results:
            clf_for_full = build_classifier(
                backends[0], retrain=False,
                train_queries=train_q, train_labels=train_l,
            )
            print("\n\nRunning full evaluation (privacy + quality + resources)...")

            privacy_metrics = evaluate_privacy(clf_for_full)
            print_privacy_report(privacy_metrics)

            resource_metrics = evaluate_resources(clf_for_full, test_q[:50])
            print_resource_report(resource_metrics)

            print("\nRunning answer quality evaluation (simulation mode)...")
            quality_metrics = evaluate_answer_quality()
            print_quality_report(quality_metrics)

    return results


if __name__ == "__main__":
    main()
