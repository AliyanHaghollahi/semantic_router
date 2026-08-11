"""ID- and task-text-invariant semantic comparison of ExecutionGraphs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations

from tiergraph.enums import (
    NodeSemanticType,
    OperatorType,
    SlotType,
    Tier,
    TransferPolicy,
)
from tiergraph.graph import DependencyEdge, ExecutionGraph, SemanticNode


def _type_multiset(
    slots: Mapping[str, SlotType],
) -> frozenset[tuple[SlotType, int]]:
    counts = Counter(slots.values())
    return frozenset(counts.items())


@dataclass(frozen=True, slots=True)
class CanonicalNode:
    """Semantic node fingerprint excluding generated IDs and task text.

    Non-FUSE nodes retain named typed slots. FUSE nodes compare on typed slot
    multisets only so ``{node_id}__{slot}`` fuse input names cannot leak IDs
    into exact-match.
    """

    operator: OperatorType
    semantic_type: NodeSemanticType
    tier: Tier
    produced_named: frozenset[tuple[str, SlotType]] | None
    required_named: frozenset[tuple[str, SlotType]] | None
    produced_types: frozenset[tuple[SlotType, int]]
    required_types: frozenset[tuple[SlotType, int]]


@dataclass(frozen=True, slots=True)
class CanonicalEdge:
    """Typed dependency fingerprint under a node-ID remapping."""

    source_key: CanonicalNode
    source_slot: str
    target_key: CanonicalNode
    target_slot: str
    transfer_policy: TransferPolicy


@dataclass(frozen=True, slots=True)
class CanonicalGraph:
    """Permutation-ready semantic graph view."""

    nodes: tuple[CanonicalNode, ...]
    edges: tuple[tuple[int, str, int, str, TransferPolicy], ...]
    has_fuse: bool
    sink_count: int


def canonicalize_node(node: SemanticNode) -> CanonicalNode:
    """Project a SemanticNode to an ID/task-invariant fingerprint."""
    produced_types = _type_multiset(node.produced_outputs)
    required_types = _type_multiset(node.required_inputs)
    if node.operator is OperatorType.FUSE:
        return CanonicalNode(
            operator=node.operator,
            semantic_type=node.semantic_type,
            tier=node.tier,
            produced_named=None,
            required_named=None,
            produced_types=produced_types,
            required_types=required_types,
        )
    return CanonicalNode(
        operator=node.operator,
        semantic_type=node.semantic_type,
        tier=node.tier,
        produced_named=frozenset(node.produced_outputs.items()),
        required_named=frozenset(node.required_inputs.items()),
        produced_types=produced_types,
        required_types=required_types,
    )


def canonicalize_graph(graph: ExecutionGraph) -> CanonicalGraph:
    """Build a canonical structural view of an ExecutionGraph."""
    nodes = tuple(graph.nodes)
    keys = tuple(canonicalize_node(node) for node in nodes)
    index = {node.node_id: position for position, node in enumerate(nodes)}
    edges: list[tuple[int, str, int, str, TransferPolicy]] = []
    for edge in graph.edges:
        source_index = index[edge.source_node_id]
        target_index = index[edge.target_node_id]
        source_slot = edge.source_slot
        target_slot = edge.target_slot
        # Normalize FUSE endpoint slot names to the non-FUSE principal slot so
        # generated ``{node_id}__{slot}`` labels do not affect exact-match.
        if nodes[target_index].operator is OperatorType.FUSE:
            target_slot = source_slot
        if nodes[source_index].operator is OperatorType.FUSE:
            source_slot = target_slot
        edges.append(
            (
                source_index,
                source_slot,
                target_index,
                target_slot,
                edge.transfer_policy,
            )
        )
    answer_ids = {
        node.node_id
        for node in nodes
        if node.semantic_type is not NodeSemanticType.CONTROL
    }
    answer_sources = {
        edge.source_node_id
        for edge in graph.edges
        if edge.source_node_id in answer_ids and edge.target_node_id in answer_ids
    }
    sinks = sum(1 for node_id in answer_ids if node_id not in answer_sources)
    has_fuse = any(node.operator is OperatorType.FUSE for node in nodes)
    return CanonicalGraph(
        nodes=keys,
        edges=tuple(edges),
        has_fuse=has_fuse,
        sink_count=sinks,
    )


def graphs_exactly_match(left: ExecutionGraph, right: ExecutionGraph) -> bool:
    """Return whether two graphs share identical semantic structure.

    Matching is invariant to generated node IDs, node list permutation, and
    task-string differences. Slot types/operators/tiers, directed typed edges,
    transfer policies, and FUSE/sink topology must agree.
    """
    return canonical_graphs_equal(canonicalize_graph(left), canonicalize_graph(right))


def canonical_graphs_equal(left: CanonicalGraph, right: CanonicalGraph) -> bool:
    """Compare two canonical graphs with permutation-invariant node matching."""
    if left.has_fuse != right.has_fuse:
        return False
    if left.sink_count != right.sink_count:
        return False
    if len(left.nodes) != len(right.nodes):
        return False
    if Counter(left.nodes) != Counter(right.nodes):
        return False

    left_groups = _group_indices(left.nodes)
    right_groups = _group_indices(right.nodes)
    if left_groups.keys() != right_groups.keys():
        return False

    group_keys = tuple(sorted(left_groups.keys(), key=repr))
    left_edge_multiset = Counter(left.edges)

    for mapping in _iter_group_bijections(left_groups, right_groups, group_keys):
        # mapping: left_index -> right_index; invert to place right edges in
        # left index space.
        inverse = {
            right_index: left_index for left_index, right_index in mapping.items()
        }
        remapped_right = Counter(
            (
                inverse[source],
                source_slot,
                inverse[target],
                target_slot,
                policy,
            )
            for source, source_slot, target, target_slot, policy in right.edges
        )
        if remapped_right == left_edge_multiset:
            return True
    return False


def _group_indices(
    nodes: Sequence[CanonicalNode],
) -> dict[CanonicalNode, tuple[int, ...]]:
    groups: dict[CanonicalNode, list[int]] = {}
    for index, node in enumerate(nodes):
        groups.setdefault(node, []).append(index)
    return {key: tuple(values) for key, values in groups.items()}


def _iter_group_bijections(
    left_groups: Mapping[CanonicalNode, Sequence[int]],
    right_groups: Mapping[CanonicalNode, Sequence[int]],
    group_keys: Sequence[CanonicalNode],
) -> Iterator[dict[int, int]]:
    """Yield left_index -> right_index bijections within fingerprint groups."""

    def _extend(
        remaining: Sequence[CanonicalNode],
        partial: dict[int, int],
    ) -> Iterator[dict[int, int]]:
        if not remaining:
            yield dict(partial)
            return
        key, *rest = remaining
        left_indices = left_groups[key]
        right_indices = right_groups[key]
        if len(left_indices) != len(right_indices):
            return
        for permutation in permutations(right_indices):
            updated = dict(partial)
            updated.update(zip(left_indices, permutation, strict=True))
            yield from _extend(rest, updated)

    yield from _extend(group_keys, {})


def semantic_edge_fingerprint(
    edge: DependencyEdge,
    id_to_key: Mapping[str, CanonicalNode],
) -> CanonicalEdge:
    """Map a concrete edge into canonical node fingerprints."""
    return CanonicalEdge(
        source_key=id_to_key[edge.source_node_id],
        source_slot=edge.source_slot,
        target_key=id_to_key[edge.target_node_id],
        target_slot=edge.target_slot,
        transfer_policy=edge.transfer_policy,
    )


def node_key_map(graph: ExecutionGraph) -> dict[str, CanonicalNode]:
    """Map node IDs to canonical fingerprints."""
    return {node.node_id: canonicalize_node(node) for node in graph.nodes}


__all__ = [
    "CanonicalEdge",
    "CanonicalGraph",
    "CanonicalNode",
    "canonical_graphs_equal",
    "canonicalize_graph",
    "canonicalize_node",
    "graphs_exactly_match",
    "node_key_map",
    "semantic_edge_fingerprint",
]
