"""Read-only attribution of TRAIN semantic failures to planner components.

Categories may overlap. One primary cause uses a fixed precedence and never
selects query type over a concrete graph error.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from tiergraph.enums import (
    NodeSemanticType,
    OperatorType,
    QueryType,
    SlotType,
    Tier,
    TransferPolicy,
)
from tiergraph.executor import GraphExecutionError
from tiergraph.graph import DependencyEdge, ExecutionGraph, SemanticNode
from tiergraph.pilot.gold_learned_harness import (
    GoldExampleView,
    compare_pair,
    make_stub_executor,
    normalize_text,
    op_group_for_graph,
    semantic_response_signature,
    slot_base,
)
from tiergraph.planner.naming import principal_slot_name
from tiergraph.planner.tasks import render_answer_task

H1_QUERY_TYPE = "H1_QUERY_TYPE"
H2_OPERATION_STRUCTURE = "H2_OPERATION_STRUCTURE"
H3_OPERATOR_TYPE = "H3_OPERATOR_TYPE"
H4_H5_ANCHOR_TARGET = "H4_H5_ANCHOR_TARGET"
H6_TIER_OWNERSHIP = "H6_TIER_OWNERSHIP"
H7_DEPENDENCY = "H7_DEPENDENCY"
INVALID_GRAPH = "INVALID_GRAPH"
OTHER = "OTHER"

GRAPH_CATEGORIES: tuple[str, ...] = (
    H2_OPERATION_STRUCTURE,
    H3_OPERATOR_TYPE,
    H4_H5_ANCHOR_TARGET,
    H6_TIER_OWNERSHIP,
    H7_DEPENDENCY,
    INVALID_GRAPH,
    OTHER,
)

PRIMARY_ORDER: tuple[str, ...] = (
    INVALID_GRAPH,
    H2_OPERATION_STRUCTURE,
    H3_OPERATOR_TYPE,
    H4_H5_ANCHOR_TARGET,
    H6_TIER_OWNERSHIP,
    H7_DEPENDENCY,
    OTHER,
)

REPRESENTATIVE_LIMIT = 5


def answer_nodes(graph: ExecutionGraph) -> tuple[SemanticNode, ...]:
    return tuple(
        node
        for node in graph.nodes
        if node.operator is not OperatorType.FUSE
        and node.semantic_type.value != "control"
    )


def node_base(node: SemanticNode) -> str:
    slot_name, slot_type = next(iter(node.produced_outputs.items()))
    return slot_base(slot_name, slot_type)


def is_implicit(node: SemanticNode) -> bool:
    return node.node_id.startswith("impl_")


def attribute_failure(
    gold: ExecutionGraph,
    learned: ExecutionGraph | None,
    *,
    learned_decoded: bool,
) -> dict[str, Any]:
    """Non-exclusive categories plus one primary cause for a semantic failure."""
    categories: set[str] = set()
    if not learned_decoded or learned is None:
        categories.add(INVALID_GRAPH)
        learned_pairs: tuple[tuple[SemanticNode, SemanticNode], ...] | None = None
    else:
        if gold.query_type is not learned.query_type:
            categories.add(H1_QUERY_TYPE)
        gold_nodes = answer_nodes(gold)
        learned_nodes = answer_nodes(learned)
        if len(gold_nodes) != len(learned_nodes):
            categories.add(H2_OPERATION_STRUCTURE)
        pairs = safe_alignment(gold_nodes, learned_nodes)
        learned_pairs = pairs
        if pairs is not None:
            _flags_from_alignment(gold, learned, pairs, categories)
        else:
            _flags_from_multisets(gold, learned, gold_nodes, learned_nodes, categories)
    concrete = categories & set(GRAPH_CATEGORIES)
    if not concrete:
        categories.add(OTHER)
    return {
        "categories": _ordered(categories),
        "primary": _primary(categories),
        "alignment_safe": learned_pairs is not None,
        "pairs": learned_pairs,
    }


def safe_alignment(
    gold_nodes: Sequence[SemanticNode],
    learned_nodes: Sequence[SemanticNode],
) -> tuple[tuple[SemanticNode, SemanticNode], ...] | None:
    """Unique node bijection by target base, by operator, or by both agreeing.

    Duplicate keys and disagreeing pairings are ambiguous and are not aligned.
    """
    if len(gold_nodes) != len(learned_nodes):
        return None
    by_base = _unique_bijection(gold_nodes, learned_nodes, node_base)
    by_operator = _unique_bijection(
        gold_nodes,
        learned_nodes,
        lambda node: node.operator.value,
    )
    if by_base is not None and by_operator is not None:
        if _pair_ids(by_base) != _pair_ids(by_operator):
            return None
        return by_base
    return by_base if by_base is not None else by_operator


def oracle_repairs(
    gold: ExecutionGraph,
    learned: ExecutionGraph,
    pairs: tuple[tuple[SemanticNode, SemanticNode], ...],
    executor: Any,
) -> dict[str, bool]:
    """Whether correcting one dimension restores the semantic task outcome."""
    return {
        "operator_type": _repaired_matches(
            gold,
            _repair_operator(gold, learned, pairs),
            executor,
        ),
        "anchor_target": _repaired_matches(
            gold,
            _repair_anchor(gold, learned, pairs),
            executor,
        ),
        "tier_ownership": _repaired_matches(
            gold,
            _repair_tier(gold, learned, pairs),
            executor,
        ),
        "dependency": _repaired_matches(
            gold,
            _repair_dependency(gold, learned, pairs),
            executor,
        ),
    }


def graph_summary(graph: ExecutionGraph) -> dict[str, Any]:
    nodes = [
        {
            "operator": node.operator.value,
            "tier": node.tier.value,
            "target": node_base(node),
            "implicit": is_implicit(node),
        }
        for node in answer_nodes(graph)
    ]
    answer_ids = {node.node_id for node in answer_nodes(graph)}
    edges = [
        {
            "source": node_base(graph.node_by_id(edge.source_node_id)),
            "target": node_base(graph.node_by_id(edge.target_node_id)),
        }
        for edge in graph.edges
        if edge.source_node_id in answer_ids and edge.target_node_id in answer_ids
    ]
    return {
        "query_type": graph.query_type.value,
        "nodes": nodes,
        "answer_edges": edges,
    }


def summarize_attributions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate success counts, prevalence, primary causes, and slices."""
    failures = [row for row in rows if not row["semantic_success"]]
    successes = len(rows) - len(failures)
    prevalence = Counter()
    primary = Counter()
    cardinality = Counter()
    by_bucket: dict[str, Counter[str]] = {}
    by_group: dict[str, Counter[str]] = {}
    oracle_applicable = 0
    oracle_hits = Counter()
    representatives: dict[str, list[dict[str, Any]]] = {name: [] for name in PRIMARY_ORDER}
    for row in rows:
        bucket = str(row["final_bucket"])
        group = str(row["op_group"])
        by_bucket.setdefault(bucket, Counter())
        by_group.setdefault(group, Counter())
        if row["semantic_success"]:
            by_bucket[bucket]["semantic_success"] += 1
            by_group[group]["semantic_success"] += 1
            continue
        by_bucket[bucket]["semantic_failure"] += 1
        by_group[group]["semantic_failure"] += 1
        categories = list(row["categories"])
        for category in categories:
            if category != H1_QUERY_TYPE:
                prevalence[category] += 1
        h1 = H1_QUERY_TYPE in categories
        concrete = [category for category in categories if category != H1_QUERY_TYPE]
        cardinality[_cardinality_key(len(concrete))] += 1
        cause = str(row["primary"])
        primary[cause] += 1
        by_bucket[bucket][cause] += 1
        by_group[group][cause] += 1
        if h1:
            prevalence[H1_QUERY_TYPE] += 1
        if row.get("oracle_applicable"):
            oracle_applicable += 1
            for name, hit in row["oracle"].items():
                if hit:
                    oracle_hits[name] += 1
        samples = representatives[cause]
        if len(samples) < REPRESENTATIVE_LIMIT:
            samples.append(_representative(row))
    n_fail = len(failures)
    if primary:
        top = max(PRIMARY_ORDER, key=lambda name: (primary[name], -PRIMARY_ORDER.index(name)))
    else:
        top = None
    return {
        "n_total": len(rows),
        "n_semantic_success": successes,
        "n_semantic_failure": n_fail,
        "error_prevalence": _counts(prevalence, (*GRAPH_CATEGORIES, H1_QUERY_TYPE)),
        "primary_causes": _counts(primary, PRIMARY_ORDER),
        "error_cardinality": {
            "exactly_1": cardinality["exactly_1"],
            "exactly_2": cardinality["exactly_2"],
            "three_or_more": cardinality["three_or_more"],
            "excludes": H1_QUERY_TYPE,
        },
        "by_final_bucket": {
            bucket: _bucket_block(counts)
            for bucket, counts in sorted(by_bucket.items())
        },
        "by_op_group": {
            group: _bucket_block(counts)
            for group, counts in sorted(by_group.items())
        },
        "oracle_repair_results": {
            "n_failures": n_fail,
            "n_alignment_safe": oracle_applicable,
            "coverage": (oracle_applicable / n_fail) if n_fail else 0.0,
            "among_aligned": {
                name: oracle_hits[name]
                for name in (
                    "operator_type",
                    "anchor_target",
                    "tier_ownership",
                    "dependency",
                )
            },
        },
        "top_bottleneck": top,
        "representative_failures": {
            cause: samples
            for cause, samples in representatives.items()
            if samples
        },
        "h1_not_used_as_primary": True,
        "dev_used": False,
        "test_used": False,
    }


def _flags_from_alignment(
    gold: ExecutionGraph,
    learned: ExecutionGraph,
    pairs: tuple[tuple[SemanticNode, SemanticNode], ...],
    categories: set[str],
) -> None:
    for gold_node, learned_node in pairs:
        if gold_node.operator is not learned_node.operator:
            categories.add(H3_OPERATOR_TYPE)
        if node_base(gold_node) != node_base(learned_node):
            categories.add(H4_H5_ANCHOR_TARGET)
        elif gold_node.operator is learned_node.operator and normalize_text(
            gold_node.task
        ) != normalize_text(learned_node.task):
            categories.add(H4_H5_ANCHOR_TARGET)
        if gold_node.tier is not learned_node.tier:
            categories.add(H6_TIER_OWNERSHIP)
        if is_implicit(gold_node) != is_implicit(learned_node):
            categories.add(H4_H5_ANCHOR_TARGET)
    if _implicit_bases(gold) != _implicit_bases(learned):
        categories.add(H4_H5_ANCHOR_TARGET)
    gold_keys = _dependency_keys(gold, {node.node_id: node.node_id for node in answer_nodes(gold)})
    learned_keys = _dependency_keys(
        learned,
        {learned_node.node_id: gold_node.node_id for gold_node, learned_node in pairs},
    )
    if gold_keys != learned_keys:
        categories.add(H7_DEPENDENCY)


def _flags_from_multisets(
    gold: ExecutionGraph,
    learned: ExecutionGraph,
    gold_nodes: Sequence[SemanticNode],
    learned_nodes: Sequence[SemanticNode],
    categories: set[str],
) -> None:
    if Counter(node.operator for node in gold_nodes) != Counter(
        node.operator for node in learned_nodes
    ):
        categories.add(H3_OPERATOR_TYPE)
    if Counter(node_base(node) for node in gold_nodes) != Counter(
        node_base(node) for node in learned_nodes
    ):
        categories.add(H4_H5_ANCHOR_TARGET)
    if _implicit_bases(gold) != _implicit_bases(learned):
        categories.add(H4_H5_ANCHOR_TARGET)
    if Counter(node.tier for node in gold_nodes) != Counter(
        node.tier for node in learned_nodes
    ):
        categories.add(H6_TIER_OWNERSHIP)
    if Counter(_edge_base_pairs(gold)) != Counter(_edge_base_pairs(learned)):
        categories.add(H7_DEPENDENCY)


def _dependency_keys(
    graph: ExecutionGraph,
    canonical_id: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    answer_ids = {node.node_id for node in answer_nodes(graph)}
    keys = []
    for edge in graph.edges:
        if edge.source_node_id not in answer_ids or edge.target_node_id not in answer_ids:
            continue
        source = canonical_id.get(edge.source_node_id)
        target = canonical_id.get(edge.target_node_id)
        if source is None or target is None:
            continue
        keys.append((source, target))
    return tuple(sorted(keys))


def _edge_base_pairs(graph: ExecutionGraph) -> tuple[tuple[str, str], ...]:
    answer_ids = {node.node_id for node in answer_nodes(graph)}
    pairs = []
    for edge in graph.edges:
        if edge.source_node_id not in answer_ids or edge.target_node_id not in answer_ids:
            continue
        source = graph.node_by_id(edge.source_node_id)
        target = graph.node_by_id(edge.target_node_id)
        pairs.append((node_base(source), node_base(target)))
    return tuple(sorted(pairs))


def _implicit_bases(graph: ExecutionGraph) -> tuple[str, ...]:
    return tuple(sorted(node_base(node) for node in answer_nodes(graph) if is_implicit(node)))


def _unique_bijection(left, right, key_fn):
    left_map: dict[Any, SemanticNode] = {}
    right_map: dict[Any, SemanticNode] = {}
    for node in left:
        key = key_fn(node)
        if key in left_map:
            return None
        left_map[key] = node
    for node in right:
        key = key_fn(node)
        if key in right_map:
            return None
        right_map[key] = node
    if set(left_map) != set(right_map):
        return None
    return tuple((left_map[key], right_map[key]) for key in sorted(left_map, key=str))


def _pair_ids(pairs: tuple[tuple[SemanticNode, SemanticNode], ...]) -> frozenset[tuple[str, str]]:
    return frozenset((gold.node_id, learned.node_id) for gold, learned in pairs)


def _primary(categories: set[str]) -> str:
    for name in PRIMARY_ORDER:
        if name in categories:
            return name
    return OTHER


def _ordered(categories: set[str]) -> list[str]:
    order = (H1_QUERY_TYPE, *PRIMARY_ORDER)
    return [name for name in order if name in categories]


def _cardinality_key(count: int) -> str:
    if count <= 1:
        return "exactly_1"
    if count == 2:
        return "exactly_2"
    return "three_or_more"


def _counts(counter: Counter, names: Sequence[str]) -> dict[str, int]:
    return {name: int(counter[name]) for name in names}


def _bucket_block(counts: Counter) -> dict[str, int]:
    success = int(counts["semantic_success"])
    failure = int(counts["semantic_failure"])
    return {
        "semantic_success": success,
        "semantic_failure": failure,
        "primary_causes": {
            name: int(counts[name]) for name in PRIMARY_ORDER if counts[name]
        },
    }


def _representative(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage_a_id": row["stage_a_id"],
        "query": row["query"],
        "final_bucket": row["final_bucket"],
        "gold_graph_summary": row["gold_graph_summary"],
        "learned_graph_summary": row["learned_graph_summary"],
        "mismatch_categories": list(row["categories"]),
        "primary": row["primary"],
    }


def _repaired_matches(
    gold: ExecutionGraph,
    repaired: ExecutionGraph | None,
    executor: Any,
) -> bool:
    if repaired is None:
        return False
    try:
        gold_result = _execute(executor, gold)
        learned_result = _execute(executor, repaired)
    except GraphExecutionError:
        return False
    if gold_result is None or learned_result is None:
        return False
    return semantic_response_signature(gold, gold_result) == semantic_response_signature(
        repaired,
        learned_result,
    )


def _execute(executor: Any, graph: ExecutionGraph) -> Any:
    import asyncio

    return asyncio.run(executor.execute(graph, fusion_plan=None))


def _repair_operator(
    gold: ExecutionGraph,
    learned: ExecutionGraph,
    pairs: tuple[tuple[SemanticNode, SemanticNode], ...],
) -> ExecutionGraph | None:
    return _rebuild(
        learned,
        pairs,
        lambda gold_node, learned_node: _retarget_node(
            learned_node,
            operator=gold_node.operator,
            tier=learned_node.tier,
            base=node_base(learned_node),
            task=_task_for(gold_node.operator, node_base(learned_node), gold_node),
        ),
    )


def _repair_anchor(
    gold: ExecutionGraph,
    learned: ExecutionGraph,
    pairs: tuple[tuple[SemanticNode, SemanticNode], ...],
) -> ExecutionGraph | None:
    del gold
    return _rebuild(
        learned,
        pairs,
        lambda gold_node, learned_node: _retarget_node(
            learned_node,
            operator=learned_node.operator,
            tier=learned_node.tier,
            base=node_base(gold_node),
            task=_task_for(learned_node.operator, node_base(gold_node), gold_node),
        ),
    )


def _repair_tier(
    gold: ExecutionGraph,
    learned: ExecutionGraph,
    pairs: tuple[tuple[SemanticNode, SemanticNode], ...],
) -> ExecutionGraph | None:
    del gold
    return _rebuild(
        learned,
        pairs,
        lambda gold_node, learned_node: learned_node.model_copy(
            update={"tier": gold_node.tier}
        ),
    )


def _repair_dependency(
    gold: ExecutionGraph,
    learned: ExecutionGraph,
    pairs: tuple[tuple[SemanticNode, SemanticNode], ...],
) -> ExecutionGraph | None:
    by_gold = {gold_node.node_id: learned_node for gold_node, learned_node in pairs}
    required: dict[str, dict[str, SlotType]] = {
        node.node_id: {} for node in learned.nodes if node.operator is not OperatorType.FUSE
    }
    edges: list[DependencyEdge] = []
    answer_ids = {node.node_id for node in answer_nodes(gold)}
    for edge in gold.edges:
        if edge.source_node_id not in answer_ids or edge.target_node_id not in answer_ids:
            continue
        source = by_gold.get(edge.source_node_id)
        target = by_gold.get(edge.target_node_id)
        if source is None or target is None:
            return None
        slot_name, slot_type = next(iter(source.produced_outputs.items()))
        if slot_name in target.produced_outputs:
            return None
        required[target.node_id][slot_name] = slot_type
        edges.append(
            DependencyEdge(
                source_node_id=source.node_id,
                source_slot=slot_name,
                target_node_id=target.node_id,
                target_slot=slot_name,
                transfer_policy=_policy(source.tier, target.tier),
            )
        )
    nodes = []
    for node in learned.nodes:
        if node.operator is OperatorType.FUSE:
            continue
        nodes.append(node.model_copy(update={"required_inputs": required[node.node_id]}))
    return _assemble(learned, nodes, edges)


def _rebuild(learned, pairs, edit) -> ExecutionGraph | None:
    updates = {
        learned_node.node_id: edit(gold_node, learned_node)
        for gold_node, learned_node in pairs
    }
    nodes = []
    for node in learned.nodes:
        if node.operator is OperatorType.FUSE:
            continue
        nodes.append(updates.get(node.node_id, node))
    wired, edges = _wire_edges(learned, nodes)
    return _assemble(learned, wired, edges)


def _wire_edges(
    learned: ExecutionGraph,
    nodes: Sequence[SemanticNode],
) -> tuple[list[SemanticNode], list[DependencyEdge]]:
    by_id = {node.node_id: node for node in nodes}
    required: dict[str, dict[str, SlotType]] = {node.node_id: {} for node in nodes}
    edges: list[DependencyEdge] = []
    for edge in learned.edges:
        source = by_id.get(edge.source_node_id)
        target = by_id.get(edge.target_node_id)
        if source is None or target is None:
            continue
        slot_name, slot_type = next(iter(source.produced_outputs.items()))
        if slot_name in target.produced_outputs:
            continue
        required[target.node_id][slot_name] = slot_type
        edges.append(
            DependencyEdge(
                source_node_id=source.node_id,
                source_slot=slot_name,
                target_node_id=target.node_id,
                target_slot=slot_name,
                transfer_policy=_policy(source.tier, target.tier),
            )
        )
    wired = [
        node.model_copy(update={"required_inputs": required[node.node_id]})
        for node in nodes
    ]
    return wired, edges


def _assemble(
    source: ExecutionGraph,
    nodes: Sequence[SemanticNode],
    edges: Sequence[DependencyEdge],
) -> ExecutionGraph | None:
    answer = [node for node in nodes if node.operator is not OperatorType.FUSE]
    produced = {node.node_id: node for node in answer}
    incoming = {
        edge.target_node_id
        for edge in edges
        if edge.source_node_id in produced and edge.target_node_id in produced
    }
    sinks = [node for node in answer if node.node_id not in incoming]
    built_nodes = list(answer)
    built_edges = [
        edge
        for edge in edges
        if edge.source_node_id in produced and edge.target_node_id in produced
    ]
    if len(sinks) > 1:
        fuse_inputs: dict[str, SlotType] = {}
        fuse_edges: list[DependencyEdge] = []
        for sink in sinks:
            slot_name, slot_type = next(iter(sink.produced_outputs.items()))
            fuse_slot = f"{sink.node_id}__{slot_name}"
            fuse_inputs[fuse_slot] = slot_type
            fuse_edges.append(
                DependencyEdge(
                    source_node_id=sink.node_id,
                    source_slot=slot_name,
                    target_node_id="fuse",
                    target_slot=fuse_slot,
                    transfer_policy=TransferPolicy.DIRECT,
                )
            )
        built_nodes.append(
            SemanticNode(
                node_id="fuse",
                semantic_type=NodeSemanticType.CONTROL,
                operator=OperatorType.FUSE,
                tier=Tier.EDGE,
                task="Fuse the terminal answers",
                required_inputs=fuse_inputs,
                produced_outputs={"response": SlotType.FINAL_RESPONSE},
            )
        )
        built_edges.extend(fuse_edges)
    try:
        return ExecutionGraph(
            graph_id=source.graph_id + "::repair",
            original_query=source.original_query,
            query_type=source.query_type,
            nodes=tuple(built_nodes),
            edges=tuple(built_edges),
        )
    except (ValidationError, ValueError):
        return None


def _retarget_node(
    node: SemanticNode,
    *,
    operator: OperatorType,
    tier: Tier,
    base: str,
    task: str,
) -> SemanticNode:
    from tiergraph.graph import _ALLOWED_OUTPUT_TYPES, _OPERATOR_SEMANTICS

    slot_type = next(iter(node.produced_outputs.values()))
    allowed = _ALLOWED_OUTPUT_TYPES[operator]
    if slot_type not in allowed:
        slot_type = sorted(allowed, key=lambda item: item.value)[0]
    slot_name = principal_slot_name(base_name=base, slot_type=slot_type)
    return node.model_copy(
        update={
            "operator": operator,
            "semantic_type": _OPERATOR_SEMANTICS[operator],
            "tier": tier,
            "task": task,
            "produced_outputs": {slot_name: slot_type},
            "required_inputs": {},
        }
    )


def _task_for(operator: OperatorType, base: str, gold_node: SemanticNode) -> str:
    if gold_node.operator is operator and node_base(gold_node) == base:
        return gold_node.task
    try:
        return render_answer_task(operator=operator, base_name=base)
    except Exception:
        return gold_node.task


def _policy(source: Tier, target: Tier) -> TransferPolicy:
    if source is Tier.EDGE and target is Tier.FOG:
        return TransferPolicy.MINIMAL_REFERENCE
    return TransferPolicy.DIRECT


def semantic_success(
    example: GoldExampleView,
    learned: ExecutionGraph | None,
    *,
    executor: Any,
    predict_ms: float = 0.0,
) -> bool:
    record = compare_pair(example, learned, predict_ms=predict_ms, executor=executor)
    return bool(record.response_semantic_match)


def make_executor() -> Any:
    return make_stub_executor()


def op_group(graph: ExecutionGraph) -> str:
    return op_group_for_graph(graph)


def query_type_name(query_type: QueryType) -> str:
    return query_type.value
