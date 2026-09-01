"""Fair v2-split baselines for Stage-A planner experiments."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tiergraph.planner.annotations import PlannerExample
from tiergraph.planner.stage_a_to_corpus import final_bucket_to_classification_label

# TierGraph teacher-forced H1 gold (must match training/eval code path):
# PlannerExample.planner_labels.query_type == graph.query_type (validated).
TIERGRAPH_H1_GOLD = "decoded_graph_query_type"
CANONICAL_H1_GOLD = "canonical_h1_classification_label"

TIERGRAPH_H1_CODE_PATH: tuple[str, ...] = (
    "stage_ab_to_planner_example()",
    "semantic_annotation_to_planner_example() sets "
    "planner_labels.query_type = decoded.graph.query_type",
    "build_planner_targets() -> PlannerTargets.query_type = labels.query_type",
    "collate_gold_structure_batch() -> query_type_labels",
    "planner_loss() H1 cross_entropy(outputs.query_type_logits, gold.query_type_labels)",
    "accumulate_head_metrics() h1_accuracy vs gold.query_type_labels",
)


def _tiergraph_h1_gold(example: PlannerExample) -> str:
    """Same gold as TierGraph H1 training and teacher-forced evaluation."""
    return example.planner_labels.query_type.value


def _canonical_h1(example: PlannerExample) -> str:
    return final_bucket_to_classification_label(str(example.metadata["final_bucket"]))


@dataclass(frozen=True, slots=True)
class H1BaselineResult:
    name: str
    predicts: str
    gold_label: str
    comparable_metrics: tuple[str, ...]
    h1_accuracy: float
    n_examples: int
    mean_latency_ms: float | None = None
    baseline_role: str = "tiergraph_comparable"
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "predicts": self.predicts,
            "gold_label": self.gold_label,
            "baseline_role": self.baseline_role,
            "comparable_metrics": list(self.comparable_metrics),
            "h1_accuracy": self.h1_accuracy,
            "n_examples": self.n_examples,
            "mean_latency_ms": self.mean_latency_ms,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


def evaluate_bucket_oracle_h1(examples: Sequence[PlannerExample]) -> H1BaselineResult:
    """Sanity check: canonical bucket label (not TierGraph H1 gold)."""
    correct = 0
    for example in examples:
        pred = _canonical_h1(example)
        if pred == _canonical_h1(example):
            correct += 1
    return H1BaselineResult(
        name="bucket_oracle_h1",
        predicts="canonical H1 from final_bucket_to_classification_label",
        gold_label=CANONICAL_H1_GOLD,
        comparable_metrics=(),
        baseline_role="canonical_h1_sanity_oracle",
        h1_accuracy=correct / len(examples) if examples else 0.0,
        n_examples=len(examples),
        mean_latency_ms=0.0,
        extra={
            "note": (
                "Non-deployable sanity oracle. Not comparable to TierGraph H1. "
                "Accuracy is 1.0 by construction."
            ),
        },
    )


def _evaluate_classifier_backend(
    *,
    name: str,
    predicts: str,
    backend_factory,
    train_examples: Sequence[PlannerExample],
    eval_examples: Sequence[PlannerExample],
) -> H1BaselineResult:
    train_q = [ex.query for ex in train_examples]
    train_l = [_tiergraph_h1_gold(ex) for ex in train_examples]
    eval_q = [ex.query for ex in eval_examples]
    eval_l = [_tiergraph_h1_gold(ex) for ex in eval_examples]

    backend = backend_factory()
    t0 = time.perf_counter()
    backend.train(train_q, train_l)
    train_sec = time.perf_counter() - t0

    correct = 0
    latencies: list[float] = []
    for query, gold in zip(eval_q, eval_l, strict=True):
        t1 = time.perf_counter()
        proba = backend.predict_proba(query)
        label = max(proba, key=proba.get)
        latencies.append((time.perf_counter() - t1) * 1000.0)
        if label == gold:
            correct += 1

    return H1BaselineResult(
        name=name,
        predicts=predicts,
        gold_label=TIERGRAPH_H1_GOLD,
        comparable_metrics=("h1_accuracy",),
        baseline_role="tiergraph_comparable",
        h1_accuracy=correct / len(eval_examples) if eval_examples else 0.0,
        n_examples=len(eval_examples),
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else None,
        extra={"train_sec": train_sec},
    )


def evaluate_cosine_classifier_h1(
    train_examples: Sequence[PlannerExample],
    eval_examples: Sequence[PlannerExample],
) -> H1BaselineResult:
    from router.classifiers.cosine_clf import CosineClassifier

    return _evaluate_classifier_backend(
        name="cosine_nearest_centroid",
        predicts="H1 query type via cosine nearest-centroid (legacy router backend)",
        backend_factory=CosineClassifier,
        train_examples=train_examples,
        eval_examples=eval_examples,
    )


def evaluate_minilm_lr_classifier_h1(
    train_examples: Sequence[PlannerExample],
    eval_examples: Sequence[PlannerExample],
) -> H1BaselineResult:
    from router.classifiers.minilm_lr import MiniLMLRClassifier

    return _evaluate_classifier_backend(
        name="minilm_lr_classifier",
        predicts="H1 query type via frozen MiniLM + logistic regression",
        backend_factory=MiniLMLRClassifier,
        train_examples=train_examples,
        eval_examples=eval_examples,
    )


def evaluate_full_query_classifier_h1(
    train_examples: Sequence[PlannerExample],
    eval_examples: Sequence[PlannerExample],
) -> H1BaselineResult:
    from router.classifier import QueryClassifier

    train_q = [ex.query for ex in train_examples]
    train_l = [_tiergraph_h1_gold(ex) for ex in train_examples]
    eval_q = [ex.query for ex in eval_examples]
    eval_l = [_tiergraph_h1_gold(ex) for ex in eval_examples]

    clf = QueryClassifier(backend="minilm_lr")
    t0 = time.perf_counter()
    clf.train(train_q, train_l)
    train_sec = time.perf_counter() - t0

    correct = 0
    latencies: list[float] = []
    routes: dict[str, int] = {}
    for query, gold in zip(eval_q, eval_l, strict=True):
        result = clf.predict(query)
        latencies.append(result.latency_ms)
        routes[result.triggered_by] = routes.get(result.triggered_by, 0) + 1
        if result.label == gold:
            correct += 1

    return H1BaselineResult(
        name="full_query_classifier_5rule",
        predicts="H1 via legacy 5-rule QueryClassifier + MiniLM+LR backend",
        gold_label=TIERGRAPH_H1_GOLD,
        comparable_metrics=("h1_accuracy",),
        baseline_role="tiergraph_comparable",
        h1_accuracy=correct / len(eval_examples) if eval_examples else 0.0,
        n_examples=len(eval_examples),
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else None,
        extra={"train_sec": train_sec, "route_counts": routes},
    )


def run_v2_baselines(
    *,
    train_examples: Sequence[PlannerExample],
    eval_examples: Sequence[PlannerExample],
) -> dict[str, Any]:
    tiergraph_comparable = [
        evaluate_cosine_classifier_h1(train_examples, eval_examples),
        evaluate_minilm_lr_classifier_h1(train_examples, eval_examples),
        evaluate_full_query_classifier_h1(train_examples, eval_examples),
    ]
    canonical_sanity = evaluate_bucket_oracle_h1(eval_examples)
    return {
        "tiergraph_h1_gold": {
            "definition": TIERGRAPH_H1_GOLD,
            "source_field": "planner_labels.query_type",
            "equivalent_field": "graph.query_type",
            "code_path": list(TIERGRAPH_H1_CODE_PATH),
            "canonical_classification_label_not_used_for_h1": True,
        },
        "decoded_graph_query_type_baselines": [
            item.to_dict() for item in tiergraph_comparable
        ],
        "canonical_classification_h1_sanity": canonical_sanity.to_dict(),
    }


__all__ = [
    "CANONICAL_H1_GOLD",
    "H1BaselineResult",
    "TIERGRAPH_H1_CODE_PATH",
    "TIERGRAPH_H1_GOLD",
    "evaluate_bucket_oracle_h1",
    "evaluate_cosine_classifier_h1",
    "evaluate_full_query_classifier_h1",
    "evaluate_minilm_lr_classifier_h1",
    "run_v2_baselines",
]
