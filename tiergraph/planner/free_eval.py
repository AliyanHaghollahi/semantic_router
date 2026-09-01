"""Free (non-teacher-forced) predicted-graph evaluation for the H1-H7 planner.

Inference path:
``query → MiniLMFeatureEncoder → PlannerModel.predict_structures → GraphDecoder``

Uses the existing decoder only. Metrics are free-structure / decoded-graph
metrics, not gold-structure teacher-forced head losses.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from tiergraph.enums import OperatorType
from tiergraph.graph import ExecutionGraph
from tiergraph.planner.annotations import PlannerExample
from tiergraph.planner.canonicalize import graphs_exactly_match
from tiergraph.planner.decode import (
    GraphDecoder,
    PlannerDecodeError,
    PlannerPredictions,
)
from tiergraph.planner.model import PlannerModel


def _iter_batches(
    examples: Sequence[PlannerExample],
    *,
    batch_size: int,
    max_batches: int | None = None,
) -> list[tuple[PlannerExample, ...]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches: list[tuple[PlannerExample, ...]] = []
    for start in range(0, len(examples), batch_size):
        batches.append(tuple(examples[start : start + batch_size]))
        if max_batches is not None and len(batches) >= max_batches:
            break
    return batches


def classify_decode_error(message: str) -> str:
    """Map a ``PlannerDecodeError`` message to a coarse failure bucket."""
    text = message.lower()
    if "at least one explicit operation" in text:
        return "no_operations"
    if (
        "span must be nonempty" in text
        or "span outside query" in text
        or "must not overlap" in text
        or "anchor text does not match" in text
    ):
        return "invalid_span_structure"
    if "owner_index" in text or "ownership" in text:
        return "ownership_failure"
    if (
        "dependency" in text
        or "self-loop" in text
        or "not eligible" in text
        or "h7" in text
    ):
        return "illegal_dependency"
    if "cycle" in text:
        return "cycle"
    if (
        "slot" in text
        or "naming" in text
        or "normalized" in text
        or "base name" in text
        or "principal" in text
    ):
        return "slot_naming_failure"
    return "other"


def execution_mode_from_graph(graph: ExecutionGraph) -> str:
    """Coarse execution shape: single / parallel_fusion / sequential.

    ``sequential`` covers both explicit op→op H7 edges and mandatory
    implicit→owner edges that create dependent structure. ``parallel_fusion``
    covers multi-answer / FUSE graphs without those dependencies.
    """
    non_fuse_ids = {
        node.node_id
        for node in graph.nodes
        if node.operator is not OperatorType.FUSE
    }
    has_dependency = any(
        edge.source_node_id in non_fuse_ids and edge.target_node_id in non_fuse_ids
        for edge in graph.edges
    )
    has_fuse = any(node.operator is OperatorType.FUSE for node in graph.nodes)
    explicit_answer_ops = {
        node.node_id
        for node in graph.nodes
        if node.operator is not OperatorType.FUSE
        and not node.node_id.startswith("impl_")
    }
    if has_dependency:
        return "sequential"
    if has_fuse or len(explicit_answer_ops) > 1:
        return "parallel_fusion"
    return "single"


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        return 0.0, 0.0, 0.0
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def _set_prf(pred: set, gold: set) -> tuple[int, int, int]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return tp, fp, fn


def gold_h7_index_pairs(example: PlannerExample) -> set[tuple[int, int]]:
    """Gold explicit op→op dependency pairs in ``operation_spans`` index space."""
    id_to_index = {
        span.node_id: index
        for index, span in enumerate(example.planner_labels.operation_spans)
    }
    pairs: set[tuple[int, int]] = set()
    for edge in example.graph.edges:
        source = id_to_index.get(edge.source_node_id)
        target = id_to_index.get(edge.target_node_id)
        if source is None or target is None:
            continue
        pairs.add((source, target))
    return pairs


def _aligned_h7_counts(
    example: PlannerExample,
    predictions: PlannerPredictions,
) -> tuple[int, int, int]:
    """H7 P/R/F1 after aligning ops by exact char spans."""
    gold_spans = {
        (span.start, span.end): index
        for index, span in enumerate(example.planner_labels.operation_spans)
    }
    pred_to_gold: dict[int, int] = {}
    used_gold: set[int] = set()
    for pred_index, operation in enumerate(predictions.operations):
        key = (operation.start, operation.end)
        gold_index = gold_spans.get(key)
        if gold_index is None or gold_index in used_gold:
            continue
        pred_to_gold[pred_index] = gold_index
        used_gold.add(gold_index)
    gold_pairs = gold_h7_index_pairs(example)
    mapped_pred: set[tuple[int, int]] = set()
    for source, target in predictions.dependency_pairs:
        if source in pred_to_gold and target in pred_to_gold:
            mapped_pred.add((pred_to_gold[source], pred_to_gold[target]))
    return _set_prf(mapped_pred, gold_pairs)


def _aligned_h5_h6_counts(
    example: PlannerExample,
    predictions: PlannerPredictions,
) -> tuple[int, int, int, int]:
    """Return (h5_correct, h5_total, h6_correct, h6_total) on span-aligned anchors."""
    gold_ops = list(example.planner_labels.operation_spans)
    gold_by_id = {span.node_id: index for index, span in enumerate(gold_ops)}
    gold_anchors = {
        (anchor.start, anchor.end): anchor
        for anchor in example.planner_labels.slot_anchors
    }
    h5_correct = h5_total = h6_correct = h6_total = 0
    for pred_anchor in predictions.anchors:
        key = (pred_anchor.start, pred_anchor.end)
        gold_anchor = gold_anchors.get(key)
        if gold_anchor is None:
            continue
        h5_total += 1
        if pred_anchor.implicit_resolution is gold_anchor.implicit_resolution:
            h5_correct += 1
        owner_gold_index = gold_by_id.get(gold_anchor.owner_node_id)
        if owner_gold_index is None:
            continue
        if not (0 <= pred_anchor.owner_index < len(predictions.operations)):
            continue
        h6_total += 1
        pred_owner = predictions.operations[pred_anchor.owner_index]
        gold_owner = gold_ops[owner_gold_index]
        if (
            pred_owner.start == gold_owner.start
            and pred_owner.end == gold_owner.end
            and pred_owner.operator is gold_owner.operator
        ):
            h6_correct += 1
    return h5_correct, h5_total, h6_correct, h6_total


@dataclass
class FreeEvalCounters:
    n_examples: int = 0
    n_valid_graphs: int = 0
    n_canonical_exact: int = 0
    h1_correct: int = 0
    h1_decoded_correct: int = 0
    h1_decoded_total: int = 0
    op_span_tp: int = 0
    op_span_fp: int = 0
    op_span_fn: int = 0
    op_joint_tp: int = 0
    op_joint_fp: int = 0
    op_joint_fn: int = 0
    op_type_correct: int = 0
    op_type_total: int = 0
    anc_span_tp: int = 0
    anc_span_fp: int = 0
    anc_span_fn: int = 0
    h5_correct: int = 0
    h5_total: int = 0
    h6_correct: int = 0
    h6_total: int = 0
    h7_tp: int = 0
    h7_fp: int = 0
    h7_fn: int = 0
    mode_correct: int = 0
    mode_total: int = 0
    mode_correct_all: int = 0
    mode_total_all: int = 0
    decode_failures: Counter = field(default_factory=Counter)
    failure_examples: list = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FreeEvalMetrics:
    """Aggregated free predicted-graph metrics."""

    n_examples: int
    valid_graph_rate: float
    canonical_exact_graph_accuracy: float
    query_type_accuracy: float
    decoded_query_type_accuracy: float
    operation_span_precision: float
    operation_span_recall: float
    operation_span_f1: float
    operation_joint_precision: float
    operation_joint_recall: float
    operation_joint_f1: float
    operator_type_accuracy_on_span_matches: float
    anchor_span_precision: float
    anchor_span_recall: float
    anchor_span_f1: float
    h5_accuracy_span_aligned: float
    h6_ownership_accuracy_span_aligned: float
    h7_precision_span_aligned: float
    h7_recall_span_aligned: float
    h7_f1_span_aligned: float
    execution_mode_accuracy_valid_only: float
    execution_mode_accuracy_all_examples: float
    decode_failure_counts: dict[str, int]
    decode_failure_examples: tuple[dict[str, str], ...]
    n_valid_graphs: int
    n_canonical_exact: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "n_valid_graphs": self.n_valid_graphs,
            "n_canonical_exact": self.n_canonical_exact,
            "valid_graph_rate": self.valid_graph_rate,
            "canonical_exact_graph_accuracy": self.canonical_exact_graph_accuracy,
            "query_type_accuracy": self.query_type_accuracy,
            "decoded_query_type_accuracy": self.decoded_query_type_accuracy,
            "operation_span_precision": self.operation_span_precision,
            "operation_span_recall": self.operation_span_recall,
            "operation_span_f1": self.operation_span_f1,
            "operation_joint_precision": self.operation_joint_precision,
            "operation_joint_recall": self.operation_joint_recall,
            "operation_joint_f1": self.operation_joint_f1,
            "operator_type_accuracy_on_span_matches": (
                self.operator_type_accuracy_on_span_matches
            ),
            "anchor_span_precision": self.anchor_span_precision,
            "anchor_span_recall": self.anchor_span_recall,
            "anchor_span_f1": self.anchor_span_f1,
            "h5_accuracy_span_aligned": self.h5_accuracy_span_aligned,
            "h6_ownership_accuracy_span_aligned": (
                self.h6_ownership_accuracy_span_aligned
            ),
            "h7_precision_span_aligned": self.h7_precision_span_aligned,
            "h7_recall_span_aligned": self.h7_recall_span_aligned,
            "h7_f1_span_aligned": self.h7_f1_span_aligned,
            "execution_mode_accuracy_valid_only": (
                self.execution_mode_accuracy_valid_only
            ),
            "execution_mode_accuracy_all_examples": (
                self.execution_mode_accuracy_all_examples
            ),
            "execution_mode_accuracy": self.execution_mode_accuracy_valid_only,
            "decode_failure_counts": dict(self.decode_failure_counts),
            "decode_failure_examples": list(self.decode_failure_examples),
            "note": (
                "Free prediction metrics: predict_structures -> GraphDecoder. "
                "Not teacher-forced. H5/H6/H7 use span-aligned predicted "
                "structures where measurable."
            ),
        }


def predict_batch(
    model: PlannerModel,
    examples: Sequence[PlannerExample],
) -> tuple[Any, tuple[PlannerPredictions, ...]]:
    """One encoder forward + free structure prediction (no gold structure)."""
    texts = [example.query for example in examples]
    features = model.encode(texts)
    predicted = model.predict_structures(features)
    return features, predicted.items


def evaluate_free_examples(
    model: PlannerModel,
    examples: Sequence[PlannerExample],
    *,
    batch_size: int,
    seed: int,
    max_batches: int | None = None,
    decoder: GraphDecoder | None = None,
) -> FreeEvalMetrics:
    """Run free predicted-graph evaluation over examples."""
    _ = seed  # kept for API symmetry with teacher-forced eval
    decoder = decoder or GraphDecoder()
    counters = FreeEvalCounters()
    model.eval()
    batches = _iter_batches(
        examples,
        batch_size=batch_size,
        max_batches=max_batches,
    )
    with torch.no_grad():
        for batch in batches:
            _features, items = predict_batch(model, batch)
            for example, predictions in zip(batch, items, strict=True):
                _accumulate_example(counters, example, predictions, decoder)
    return _finalize(counters)


def evaluate_free_predictions(
    pairs: Sequence[tuple[PlannerExample, PlannerPredictions]],
    *,
    decoder: GraphDecoder | None = None,
) -> FreeEvalMetrics:
    """Evaluate already-produced free predictions (for unit tests)."""
    decoder = decoder or GraphDecoder()
    counters = FreeEvalCounters()
    for example, predictions in pairs:
        _accumulate_example(counters, example, predictions, decoder)
    return _finalize(counters)


def _accumulate_example(
    counters: FreeEvalCounters,
    example: PlannerExample,
    predictions: PlannerPredictions,
    decoder: GraphDecoder,
) -> None:
    counters.n_examples += 1

    gold_op_spans = {
        (span.start, span.end) for span in example.planner_labels.operation_spans
    }
    pred_op_spans = {(op.start, op.end) for op in predictions.operations}
    tp, fp, fn = _set_prf(pred_op_spans, gold_op_spans)
    counters.op_span_tp += tp
    counters.op_span_fp += fp
    counters.op_span_fn += fn

    gold_op_joint = {
        (span.start, span.end, span.operator)
        for span in example.planner_labels.operation_spans
    }
    pred_op_joint = {
        (op.start, op.end, op.operator) for op in predictions.operations
    }
    jtp, jfp, jfn = _set_prf(pred_op_joint, gold_op_joint)
    counters.op_joint_tp += jtp
    counters.op_joint_fp += jfp
    counters.op_joint_fn += jfn

    gold_span_to_op = {
        (span.start, span.end): span.operator
        for span in example.planner_labels.operation_spans
    }
    for operation in predictions.operations:
        key = (operation.start, operation.end)
        if key in gold_span_to_op:
            counters.op_type_total += 1
            if operation.operator is gold_span_to_op[key]:
                counters.op_type_correct += 1

    gold_anc = {
        (anchor.start, anchor.end) for anchor in example.planner_labels.slot_anchors
    }
    pred_anc = {(anchor.start, anchor.end) for anchor in predictions.anchors}
    atp, afp, afn = _set_prf(pred_anc, gold_anc)
    counters.anc_span_tp += atp
    counters.anc_span_fp += afp
    counters.anc_span_fn += afn

    h5_c, h5_t, h6_c, h6_t = _aligned_h5_h6_counts(example, predictions)
    counters.h5_correct += h5_c
    counters.h5_total += h5_t
    counters.h6_correct += h6_c
    counters.h6_total += h6_t

    h7_tp, h7_fp, h7_fn = _aligned_h7_counts(example, predictions)
    counters.h7_tp += h7_tp
    counters.h7_fp += h7_fp
    counters.h7_fn += h7_fn

    if predictions.aux_query_type is example.graph.query_type:
        counters.h1_correct += 1

    gold_mode = execution_mode_from_graph(example.graph)
    counters.mode_total_all += 1

    try:
        decoded = decoder.decode(
            predictions,
            query=example.query,
            graph_id=f"pred::{example.example_id}",
        )
    except PlannerDecodeError as exc:
        reason = classify_decode_error(str(exc))
        counters.decode_failures[reason] += 1
        if len(counters.failure_examples) < 32:
            counters.failure_examples.append(
                {
                    "example_id": example.example_id,
                    "reason": reason,
                    "message": str(exc),
                }
            )
        return

    counters.n_valid_graphs += 1
    counters.h1_decoded_total += 1
    if decoded.graph.query_type is example.graph.query_type:
        counters.h1_decoded_correct += 1
    if graphs_exactly_match(decoded.graph, example.graph):
        counters.n_canonical_exact += 1

    pred_mode = execution_mode_from_graph(decoded.graph)
    if pred_mode == gold_mode:
        counters.mode_correct_all += 1

    counters.mode_total += 1
    if pred_mode == gold_mode:
        counters.mode_correct += 1


def _finalize(counters: FreeEvalCounters) -> FreeEvalMetrics:
    n = max(counters.n_examples, 1)
    op_p, op_r, op_f = _prf(
        counters.op_span_tp, counters.op_span_fp, counters.op_span_fn
    )
    oj_p, oj_r, oj_f = _prf(
        counters.op_joint_tp, counters.op_joint_fp, counters.op_joint_fn
    )
    an_p, an_r, an_f = _prf(
        counters.anc_span_tp, counters.anc_span_fp, counters.anc_span_fn
    )
    h7_p, h7_r, h7_f = _prf(counters.h7_tp, counters.h7_fp, counters.h7_fn)
    return FreeEvalMetrics(
        n_examples=counters.n_examples,
        valid_graph_rate=counters.n_valid_graphs / n,
        canonical_exact_graph_accuracy=counters.n_canonical_exact / n,
        query_type_accuracy=counters.h1_correct / n,
        decoded_query_type_accuracy=(
            counters.h1_decoded_correct / counters.h1_decoded_total
            if counters.h1_decoded_total
            else 0.0
        ),
        operation_span_precision=op_p,
        operation_span_recall=op_r,
        operation_span_f1=op_f,
        operation_joint_precision=oj_p,
        operation_joint_recall=oj_r,
        operation_joint_f1=oj_f,
        operator_type_accuracy_on_span_matches=(
            counters.op_type_correct / counters.op_type_total
            if counters.op_type_total
            else 0.0
        ),
        anchor_span_precision=an_p,
        anchor_span_recall=an_r,
        anchor_span_f1=an_f,
        h5_accuracy_span_aligned=(
            counters.h5_correct / counters.h5_total if counters.h5_total else 0.0
        ),
        h6_ownership_accuracy_span_aligned=(
            counters.h6_correct / counters.h6_total if counters.h6_total else 0.0
        ),
        h7_precision_span_aligned=h7_p,
        h7_recall_span_aligned=h7_r,
        h7_f1_span_aligned=h7_f,
        execution_mode_accuracy_valid_only=(
            counters.mode_correct / counters.mode_total
            if counters.mode_total
            else 0.0
        ),
        execution_mode_accuracy_all_examples=(
            counters.mode_correct_all / counters.mode_total_all
            if counters.mode_total_all
            else 0.0
        ),
        decode_failure_counts=dict(counters.decode_failures),
        decode_failure_examples=tuple(counters.failure_examples),
        n_valid_graphs=counters.n_valid_graphs,
        n_canonical_exact=counters.n_canonical_exact,
    )


__all__ = [
    "FreeEvalMetrics",
    "classify_decode_error",
    "evaluate_free_examples",
    "evaluate_free_predictions",
    "execution_mode_from_graph",
    "gold_h7_index_pairs",
    "predict_batch",
]
