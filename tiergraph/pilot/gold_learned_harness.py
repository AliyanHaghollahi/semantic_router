"""Gold versus learned execution comparison on one deterministic backend.

The stub result depends on the operator, tier, local anchor, and any resolved
predecessor value. It never calls a model. The primary response metric is an
order-independent semantic signature of those structured results.
``user_response`` string equality is retained only as a secondary metric.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tiergraph.enums import NodeSemanticType, OperatorType, SlotType, Tier
from tiergraph.executor import GraphExecutionError, GraphExecutor
from tiergraph.graph import ExecutionGraph, SemanticNode
from tiergraph.planner.canonicalize import graphs_exactly_match

METRIC_NAMES: tuple[str, ...] = (
    "graph_valid_rate",
    "exact_graph_rate",
    "tier_routing_rate",
    "gold_execution_success_rate",
    "learned_execution_success_rate",
    "response_semantic_match_rate",
    "response_exact_string_match_rate",
    "gold_execute_latency_ms_median",
    "learned_predict_execute_latency_ms_median",
)

_SLOT_SUFFIX_BY_TYPE: dict[SlotType, str] = {
    SlotType.RESOLVED_REFERENCE: "identifier",
    SlotType.PERSONAL_FACT: "fact",
    SlotType.PERSONAL_RECORD: "record",
    SlotType.ENVIRONMENTAL_FACT: "fact",
    SlotType.LOCATION: "location",
    SlotType.NAVIGATION_INSTRUCTION: "navigation",
    SlotType.SCENE_DESCRIPTION: "scene",
    SlotType.FINAL_RESPONSE: "response",
}


def normalize_text(value: Any) -> str:
    """Casefold and collapse whitespace. No other rewriting."""
    return " ".join(str(value).casefold().split())


def operator_stub_runner(
    node: SemanticNode,
    bound_inputs: Mapping[str, Any],
    transfer: Any,
) -> dict[str, str]:
    """Return one deterministic slot value for this node.

    The value changes when the operator, tier, local anchor, or a resolved
    predecessor value changes. ``FUSE`` stays inside ``GraphExecutor``.
    """
    if node.operator is OperatorType.FUSE:
        raise GraphExecutionError("operator stub does not execute FUSE nodes")
    slot_name = next(iter(node.produced_outputs))
    return {slot_name: canonical_backend_value(node, bound_inputs, transfer)}


def canonical_backend_value(
    node: SemanticNode,
    bound_inputs: Mapping[str, Any],
    transfer: Any,
) -> str:
    """Stable backend result for one answer node."""
    anchor = local_anchor(node)
    predecessors = predecessor_values(bound_inputs, transfer)
    return (
        f"{node.operator.value}|{node.tier.value}|anchor={anchor}|pred={predecessors}"
    )


def local_anchor(node: SemanticNode) -> str:
    """Normalized anchor carried by the node task and principal slot name."""
    slot_name = next(iter(node.produced_outputs))
    slot_type = node.produced_outputs[slot_name]
    return normalize_text(f"{slot_base(slot_name, slot_type)}|{node.task}")


def slot_base(slot_name: str, slot_type: SlotType) -> str:
    suffix = "_" + _SLOT_SUFFIX_BY_TYPE.get(slot_type, "")
    if suffix != "_" and slot_name.endswith(suffix):
        return slot_name[: -len(suffix)]
    return slot_name


def predecessor_values(bound_inputs: Mapping[str, Any], transfer: Any) -> str:
    """Sorted normalized predecessor texts. Source node ids are ignored."""
    values: list[str] = []
    for slot_name in sorted(bound_inputs):
        bound = bound_inputs[slot_name]
        raw = bound.value if hasattr(bound, "value") else bound
        values.append(normalize_text(raw))
    transferred = getattr(transfer, "transferred_slots", None)
    if isinstance(transferred, Mapping):
        already = set(values)
        for slot_name in sorted(transferred):
            text = normalize_text(transferred[slot_name])
            if text not in already:
                values.append(text)
    return ",".join(values)


def make_stub_executor() -> GraphExecutor:
    """One executor shared by the gold and learned conditions."""
    return GraphExecutor(
        edge_client=object(),
        fog_client=object(),
        node_runner=operator_stub_runner,
    )


@dataclass(frozen=True)
class GoldExampleView:
    """Gold graph plus the slice labels needed for the frozen metrics."""

    example_id: str
    query: str
    graph: ExecutionGraph
    final_bucket: str


@dataclass(frozen=True)
class ComparisonRecord:
    example_id: str
    final_bucket: str
    op_group: str
    graph_valid: bool
    exact_graph: bool
    tier_routing: bool
    gold_execution_success: bool
    learned_execution_success: bool
    response_semantic_match: bool
    response_exact_string_match: bool
    gold_execute_ms: float | None
    learned_predict_execute_ms: float | None
    gold_user_response: str | None
    learned_user_response: str | None


def op_group_for_graph(graph: ExecutionGraph) -> str:
    """Single-op means one non-FUSE answer node on the gold graph."""
    count = sum(1 for node in graph.nodes if node.operator is not OperatorType.FUSE)
    return "single_op" if count <= 1 else "multi_op"


def tier_signature(graph: ExecutionGraph) -> Counter[tuple[OperatorType, Tier]]:
    return Counter(
        (node.operator, node.tier)
        for node in graph.nodes
        if node.operator is not OperatorType.FUSE
        and node.semantic_type is not NodeSemanticType.CONTROL
    )


def semantic_response_signature(graph: ExecutionGraph, result: Any) -> tuple[Any, ...]:
    """Canonical execution signature.

    Parallel nodes inside one dependency wave are sorted, so node order alone
    does not change the signature. Wave order preserves sequential dependencies.
    Output values, predecessor values, and the fusion comparison are included.
    Node ids are not.
    """
    by_id = {node.node_id: node for node in _answer_nodes(graph)}
    waves: list[tuple[Any, ...]] = []
    for wave in _answer_waves(graph):
        steps = []
        for node_id in wave:
            node = by_id[node_id]
            slot_name = next(iter(node.produced_outputs))
            slot_type = node.produced_outputs[slot_name]
            output = normalize_text(result.results[node_id].outputs[slot_name])
            steps.append(
                (
                    node.operator.value,
                    node.tier.value,
                    slot_type.value,
                    output,
                    _predecessor_step(graph, node, result),
                )
            )
        waves.append(tuple(sorted(steps)))
    return (tuple(waves), _fusion_outcome(graph, result))


def compare_pair(
    example: GoldExampleView,
    predicted_graph: ExecutionGraph | None,
    *,
    predict_ms: float,
    executor: GraphExecutor | None = None,
) -> ComparisonRecord:
    """Execute gold and, when present, the learned graph on the same stub."""
    runner = executor or make_stub_executor()
    gold_result, gold_ms, gold_error = _execute(runner, example.graph)
    learned_result = None
    learned_ms: float | None = None
    if predicted_graph is not None:
        learned_result, learned_execute_ms, _learned_error = _execute(
            runner,
            predicted_graph,
        )
        learned_ms = float(predict_ms) + learned_execute_ms
    else:
        learned_ms = float(predict_ms)

    gold_text = None if gold_result is None else gold_result.user_response
    learned_text = None if learned_result is None else learned_result.user_response
    gold_signature = (
        None
        if gold_result is None
        else semantic_response_signature(example.graph, gold_result)
    )
    learned_signature = (
        None
        if learned_result is None or predicted_graph is None
        else semantic_response_signature(predicted_graph, learned_result)
    )
    graph_valid = predicted_graph is not None
    return ComparisonRecord(
        example_id=example.example_id,
        final_bucket=example.final_bucket,
        op_group=op_group_for_graph(example.graph),
        graph_valid=graph_valid,
        exact_graph=(
            graph_valid and graphs_exactly_match(predicted_graph, example.graph)
        ),
        tier_routing=(
            graph_valid
            and tier_signature(predicted_graph) == tier_signature(example.graph)
        ),
        gold_execution_success=gold_error is None and gold_result is not None,
        learned_execution_success=learned_result is not None,
        response_semantic_match=(
            gold_signature is not None and gold_signature == learned_signature
        ),
        response_exact_string_match=(
            gold_text is not None and learned_text is not None and gold_text == learned_text
        ),
        gold_execute_ms=gold_ms,
        learned_predict_execute_ms=learned_ms,
        gold_user_response=gold_text,
        learned_user_response=learned_text,
    )


def summarize(records: Sequence[ComparisonRecord]) -> dict[str, Any]:
    """Overall rates plus slices by gold bucket and single-op versus multi-op."""
    return {
        "n": len(records),
        "overall": _rates(records),
        "by_final_bucket": {
            bucket: _rates(group)
            for bucket, group in _groups(records, "final_bucket").items()
        },
        "by_op_group": {
            name: _rates(group)
            for name, group in _groups(records, "op_group").items()
        },
        "latency_note": "stub harness wall time, not Pi or Ollama time",
        "response_criterion": "response_semantic_match",
        "secondary_response_criterion": "response_exact_string_match",
        "fusion_plan": None,
    }


def _answer_nodes(graph: ExecutionGraph) -> tuple[SemanticNode, ...]:
    return tuple(
        node
        for node in graph.nodes
        if node.operator is not OperatorType.FUSE
        and node.semantic_type is not NodeSemanticType.CONTROL
    )


def _answer_waves(graph: ExecutionGraph) -> tuple[tuple[str, ...], ...]:
    """Dependency waves over answer nodes. Node ids stay internal."""
    answers = _answer_nodes(graph)
    answer_ids = {node.node_id for node in answers}
    successors: dict[str, list[str]] = {node_id: [] for node_id in answer_ids}
    indegree = {node_id: 0 for node_id in answer_ids}
    for edge in graph.edges:
        if edge.source_node_id in answer_ids and edge.target_node_id in answer_ids:
            successors[edge.source_node_id].append(edge.target_node_id)
            indegree[edge.target_node_id] += 1
    ready = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    waves: list[tuple[str, ...]] = []
    while ready:
        wave = tuple(ready)
        waves.append(wave)
        ready = deque()
        for node_id in wave:
            for successor_id in successors[node_id]:
                indegree[successor_id] -= 1
                if indegree[successor_id] == 0:
                    ready.append(successor_id)
    return tuple(waves)


def _predecessor_step(
    graph: ExecutionGraph,
    node: SemanticNode,
    result: Any,
) -> tuple[tuple[str, str, str, str], ...]:
    steps = []
    for edge in graph.edges:
        if edge.target_node_id != node.node_id:
            continue
        source = graph.node_by_id(edge.source_node_id)
        if source.operator is OperatorType.FUSE:
            continue
        slot_name = edge.source_slot
        value = result.results[source.node_id].outputs[slot_name]
        steps.append(
            (
                source.operator.value,
                source.tier.value,
                source.produced_outputs[slot_name].value,
                normalize_text(value),
            )
        )
    return tuple(sorted(steps))


def _fusion_outcome(graph: ExecutionGraph, result: Any) -> tuple[Any, ...]:
    edge_values = sorted(
        _principal_text(node, result)
        for node in _answer_nodes(graph)
        if node.tier is Tier.EDGE
    )
    fog_values = sorted(
        _principal_text(node, result)
        for node in _answer_nodes(graph)
        if node.tier is Tier.FOG
    )
    if not edge_values or not fog_values:
        comparison = "not_applicable"
    elif edge_values == fog_values:
        comparison = "match"
    else:
        comparison = "mismatch"
    consumed = tuple(
        sorted(
            (
                node.operator.value,
                node.tier.value,
                next(iter(node.produced_outputs.values())).value,
                _principal_text(node, result),
            )
            for node in _answer_nodes(graph)
            if any(
                edge.source_node_id == node.node_id
                and graph.node_by_id(edge.target_node_id).operator is OperatorType.FUSE
                for edge in graph.edges
            )
        )
    )
    return (result.response_fusion_mode, comparison, consumed)


def _principal_text(node: SemanticNode, result: Any) -> str:
    slot_name = next(iter(node.produced_outputs))
    return normalize_text(result.results[node.node_id].outputs[slot_name])


def _groups(
    records: Sequence[ComparisonRecord],
    field: str,
) -> dict[str, list[ComparisonRecord]]:
    grouped: dict[str, list[ComparisonRecord]] = {}
    for record in records:
        grouped.setdefault(str(getattr(record, field)), []).append(record)
    return dict(sorted(grouped.items()))


def _rates(records: Sequence[ComparisonRecord]) -> dict[str, Any]:
    return {
        "n": len(records),
        "graph_valid_rate": _rate(records, "graph_valid"),
        "exact_graph_rate": _rate(records, "exact_graph"),
        "tier_routing_rate": _rate(records, "tier_routing"),
        "gold_execution_success_rate": _rate(records, "gold_execution_success"),
        "learned_execution_success_rate": _rate(records, "learned_execution_success"),
        "response_semantic_match_rate": _rate(records, "response_semantic_match"),
        "response_exact_string_match_rate": _rate(records, "response_exact_string_match"),
        "gold_execute_latency_ms_median": _median(
            record.gold_execute_ms for record in records
        ),
        "learned_predict_execute_latency_ms_median": _median(
            record.learned_predict_execute_ms for record in records
        ),
    }


def _rate(records: Sequence[ComparisonRecord], field: str) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if getattr(record, field)) / len(records)


def _median(values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return float(statistics.median(numbers))


def _execute(
    executor: GraphExecutor,
    graph: ExecutionGraph,
) -> tuple[Any, float, str | None]:
    started = time.perf_counter()
    try:
        result = asyncio.run(executor.execute(graph, fusion_plan=None))
    except GraphExecutionError as exc:
        return None, (time.perf_counter() - started) * 1000, str(exc)
    return result, (time.perf_counter() - started) * 1000, None
