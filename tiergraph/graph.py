"""Typed execution DAG schemas and validation for TierGraph."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, Mapping, Self

from pydantic import Field, JsonValue, field_validator, model_validator

from tiergraph.enums import (
    ExecutionStatus,
    NodeSemanticType,
    OperatorType,
    QueryType,
    SlotType,
    Tier,
    TransferPolicy,
)
from tiergraph.models import TierGraphSchema


_OPERATOR_SEMANTICS = {
    OperatorType.RESOLVE_PERSONAL: NodeSemanticType.PERSONAL,
    OperatorType.RETRIEVE_PERSONAL: NodeSemanticType.PERSONAL,
    OperatorType.IDENTIFY_ENVIRONMENTAL: NodeSemanticType.ENVIRONMENTAL,
    OperatorType.LOCATE_ENVIRONMENTAL: NodeSemanticType.ENVIRONMENTAL,
    OperatorType.NAVIGATE_TO: NodeSemanticType.ENVIRONMENTAL,
    OperatorType.DESCRIBE_ENVIRONMENT: NodeSemanticType.ENVIRONMENTAL,
    OperatorType.FUSE: NodeSemanticType.CONTROL,
}

_ALLOWED_OUTPUT_TYPES = {
    OperatorType.RESOLVE_PERSONAL: frozenset({SlotType.RESOLVED_REFERENCE}),
    OperatorType.RETRIEVE_PERSONAL: frozenset(
        {
            SlotType.RESOLVED_REFERENCE,
            SlotType.PERSONAL_FACT,
            SlotType.PERSONAL_RECORD,
        }
    ),
    OperatorType.IDENTIFY_ENVIRONMENTAL: frozenset(
        {
            SlotType.ENVIRONMENTAL_FACT,
            SlotType.LOCATION,
            SlotType.SCENE_DESCRIPTION,
        }
    ),
    OperatorType.LOCATE_ENVIRONMENTAL: frozenset({SlotType.LOCATION}),
    OperatorType.NAVIGATE_TO: frozenset({SlotType.NAVIGATION_INSTRUCTION}),
    OperatorType.DESCRIBE_ENVIRONMENT: frozenset({SlotType.SCENE_DESCRIPTION}),
    OperatorType.FUSE: frozenset({SlotType.FINAL_RESPONSE}),
}

_PERSONAL_SLOT_TYPES = frozenset(
    {SlotType.PERSONAL_FACT, SlotType.PERSONAL_RECORD}
)


class SemanticNode(TierGraphSchema):
    """One semantic answer-producing or control operation in a graph."""

    node_id: str
    semantic_type: NodeSemanticType
    operator: OperatorType
    tier: Tier
    task: str
    required_inputs: dict[str, SlotType] = Field(default_factory=dict)
    produced_outputs: dict[str, SlotType]
    status: ExecutionStatus = ExecutionStatus.PENDING
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("node_id", "task")
    @classmethod
    def _validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("required_inputs", "produced_outputs")
    @classmethod
    def _validate_slot_names(
        cls, slots: dict[str, SlotType]
    ) -> dict[str, SlotType]:
        if any(not name.strip() for name in slots):
            raise ValueError("slot names must not be blank")
        return slots

    @model_validator(mode="after")
    def _validate_operator_contract(self) -> "SemanticNode":
        if not self.produced_outputs:
            raise ValueError("a node must produce at least one output")

        overlapping_slots = set(self.required_inputs) & set(self.produced_outputs)
        if overlapping_slots:
            raise ValueError(
                "required and produced slot names must not overlap: "
                f"{sorted(overlapping_slots)}"
            )

        expected_semantic = _OPERATOR_SEMANTICS[self.operator]
        if self.semantic_type is not expected_semantic:
            raise ValueError(
                f"{self.operator.value} requires {expected_semantic.value} semantics"
            )

        if self.semantic_type in {
            NodeSemanticType.PERSONAL,
            NodeSemanticType.CONTROL,
        } and self.tier is not Tier.EDGE:
            raise ValueError(
                f"{self.semantic_type.value} nodes must execute on Edge"
            )

        if self.tier is Tier.FOG and self.semantic_type is not NodeSemanticType.ENVIRONMENTAL:
            raise ValueError("Fog nodes must be environmental")

        allowed_outputs = _ALLOWED_OUTPUT_TYPES[self.operator]
        invalid_outputs = {
            slot_type
            for slot_type in self.produced_outputs.values()
            if slot_type not in allowed_outputs
        }
        if invalid_outputs:
            invalid_names = sorted(slot_type.value for slot_type in invalid_outputs)
            raise ValueError(
                f"{self.operator.value} cannot produce slot types {invalid_names}"
            )

        if self.operator is OperatorType.FUSE:
            if len(self.produced_outputs) != 1:
                raise ValueError("FUSE must produce exactly one FINAL_RESPONSE")
            if not self.required_inputs:
                raise ValueError("FUSE must require at least one input")

        if self.tier is Tier.FOG:
            invalid_inputs = {
                slot_type
                for slot_type in self.required_inputs.values()
                if slot_type in _PERSONAL_SLOT_TYPES
            }
            if invalid_inputs:
                invalid_names = sorted(slot_type.value for slot_type in invalid_inputs)
                raise ValueError(
                    f"Fog inputs cannot accept personal slot types {invalid_names}"
                )

        return self


class DependencyEdge(TierGraphSchema):
    """Typed data dependency between two named node slots."""

    source_node_id: str
    source_slot: str
    target_node_id: str
    target_slot: str
    transfer_policy: TransferPolicy = TransferPolicy.DIRECT

    @field_validator(
        "source_node_id",
        "source_slot",
        "target_node_id",
        "target_slot",
    )
    @classmethod
    def _validate_nonblank_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


@dataclass(frozen=True)
class _GraphTopology:
    """Reusable node-level topology derived from serialized graph order."""

    node_lookup: dict[str, SemanticNode]
    node_index: dict[str, int]
    indegree: dict[str, int]
    adjacency: dict[str, tuple[str, ...]]
    reverse_adjacency: dict[str, tuple[str, ...]]


def _build_topology(
    nodes: tuple[SemanticNode, ...],
    edges: tuple[DependencyEdge, ...],
) -> _GraphTopology:
    node_lookup = {node.node_id: node for node in nodes}
    node_index = {node.node_id: index for index, node in enumerate(nodes)}
    indegree = {node.node_id: 0 for node in nodes}
    adjacency_lists: dict[str, list[str]] = {node.node_id: [] for node in nodes}
    reverse_lists: dict[str, list[str]] = {node.node_id: [] for node in nodes}
    seen_node_dependencies: set[tuple[str, str]] = set()

    for edge in edges:
        dependency = (edge.source_node_id, edge.target_node_id)
        if dependency in seen_node_dependencies:
            continue
        seen_node_dependencies.add(dependency)
        adjacency_lists[edge.source_node_id].append(edge.target_node_id)
        reverse_lists[edge.target_node_id].append(edge.source_node_id)
        indegree[edge.target_node_id] += 1

    return _GraphTopology(
        node_lookup=node_lookup,
        node_index=node_index,
        indegree=indegree,
        adjacency={node_id: tuple(values) for node_id, values in adjacency_lists.items()},
        reverse_adjacency={
            node_id: tuple(values) for node_id, values in reverse_lists.items()
        },
    )


def _kahn_order(topology: _GraphTopology) -> tuple[str, ...]:
    """Return a stable O(V+E) topological order or its acyclic prefix."""
    indegree = topology.indegree.copy()
    ready = deque(
        node_id for node_id in topology.node_lookup if indegree[node_id] == 0
    )
    ordered: list[str] = []

    while ready:
        node_id = ready.popleft()
        ordered.append(node_id)
        for successor_id in topology.adjacency[node_id]:
            indegree[successor_id] -= 1
            if indegree[successor_id] == 0:
                ready.append(successor_id)

    return tuple(ordered)


class ExecutionGraph(TierGraphSchema):
    """A validated, typed directed acyclic execution graph."""

    schema_version: Literal["1.0"] = "1.0"
    graph_id: str
    original_query: str
    query_type: QueryType
    nodes: tuple[SemanticNode, ...]
    edges: tuple[DependencyEdge, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @cached_property
    def _topology(self) -> _GraphTopology:
        return _build_topology(self.nodes, self.edges)

    @field_validator("graph_id", "original_query")
    @classmethod
    def _validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("nodes", "edges", mode="before")
    @classmethod
    def _normalize_json_arrays(cls, value: object) -> object:
        if type(value) is list:
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_graph(self) -> "ExecutionGraph":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node IDs must be unique")

        node_map = {node.node_id: node for node in self.nodes}
        incoming_slots: set[tuple[str, str]] = set()
        edge_keys: set[tuple[str, str, str, str]] = set()

        for edge in self.edges:
            edge_key = (
                edge.source_node_id,
                edge.source_slot,
                edge.target_node_id,
                edge.target_slot,
            )
            if edge_key in edge_keys:
                raise ValueError(f"duplicate dependency edge: {edge_key}")
            edge_keys.add(edge_key)

            if edge.source_node_id not in node_map:
                raise ValueError(
                    f"edge source node does not exist: {edge.source_node_id}"
                )
            if edge.target_node_id not in node_map:
                raise ValueError(
                    f"edge target node does not exist: {edge.target_node_id}"
                )
            if edge.source_node_id == edge.target_node_id:
                raise ValueError("self-edges are not allowed")

            source = node_map[edge.source_node_id]
            target = node_map[edge.target_node_id]

            if edge.source_slot not in source.produced_outputs:
                raise ValueError(
                    f"source slot does not exist: "
                    f"{edge.source_node_id}.{edge.source_slot}"
                )
            if edge.target_slot not in target.required_inputs:
                raise ValueError(
                    f"target slot does not exist: "
                    f"{edge.target_node_id}.{edge.target_slot}"
                )

            source_type = source.produced_outputs[edge.source_slot]
            target_type = target.required_inputs[edge.target_slot]
            if source_type is not target_type:
                raise ValueError(
                    "incompatible slot types: "
                    f"{source_type.value} -> {target_type.value}"
                )

            target_binding = (edge.target_node_id, edge.target_slot)
            if target_binding in incoming_slots:
                raise ValueError(
                    f"multiple producers for input: "
                    f"{edge.target_node_id}.{edge.target_slot}"
                )
            incoming_slots.add(target_binding)

            edge_to_fog = source.tier is Tier.EDGE and target.tier is Tier.FOG
            if edge_to_fog:
                if edge.transfer_policy is not TransferPolicy.MINIMAL_REFERENCE:
                    raise ValueError(
                        "Edge-to-Fog dependencies require minimal_reference"
                    )
                if source_type is not SlotType.RESOLVED_REFERENCE:
                    raise ValueError(
                        "Edge-to-Fog dependencies may transfer only "
                        "RESOLVED_REFERENCE"
                    )
            elif edge.transfer_policy is not TransferPolicy.DIRECT:
                raise ValueError(
                    "minimal_reference is valid only for Edge-to-Fog dependencies"
                )

            if source.operator is OperatorType.FUSE:
                raise ValueError("FUSE nodes must be terminal")

        for node in self.nodes:
            for slot_name in node.required_inputs:
                if (node.node_id, slot_name) not in incoming_slots:
                    raise ValueError(
                        f"unresolved required input: {node.node_id}.{slot_name}"
                    )

        topological_ids = _kahn_order(self._topology)
        if len(topological_ids) != len(self.nodes):
            ordered_ids = set(topological_ids)
            blocked_ids = [
                node_id for node_id in node_ids if node_id not in ordered_ids
            ]
            raise ValueError(
                f"nodes blocked by a cycle: {blocked_ids}"
            )

        personal_count = sum(
            node.semantic_type is NodeSemanticType.PERSONAL for node in self.nodes
        )
        environmental_count = sum(
            node.semantic_type is NodeSemanticType.ENVIRONMENTAL
            for node in self.nodes
        )
        if self.query_type is QueryType.PERSONAL:
            if personal_count == 0 or environmental_count != 0:
                raise ValueError(
                    "Personal graphs require personal nodes and no environmental nodes"
                )
        elif self.query_type is QueryType.ENVIRONMENTAL:
            if environmental_count == 0 or personal_count != 0:
                raise ValueError(
                    "Environmental graphs require environmental nodes and no personal nodes"
                )
        elif personal_count == 0 or environmental_count == 0:
            raise ValueError(
                "Mixed graphs require both personal and environmental nodes"
            )

        return self

    def node_by_id(self, node_id: str) -> SemanticNode:
        """Return a node by ID, raising KeyError when it is absent."""
        try:
            return self._topology.node_lookup[node_id]
        except KeyError:
            raise KeyError(node_id) from None

    def predecessors(self, node_id: str) -> tuple[SemanticNode, ...]:
        """Return unique predecessor nodes in serialized edge order."""
        self.node_by_id(node_id)
        return tuple(
            self._topology.node_lookup[predecessor_id]
            for predecessor_id in self._topology.reverse_adjacency[node_id]
        )

    def successors(self, node_id: str) -> tuple[SemanticNode, ...]:
        """Return unique successor nodes in serialized edge order."""
        self.node_by_id(node_id)
        return tuple(
            self._topology.node_lookup[successor_id]
            for successor_id in self._topology.adjacency[node_id]
        )

    def topological_order(self) -> tuple[str, ...]:
        """Return node IDs in deterministic topological order."""
        return _kahn_order(self._topology)

    def execution_waves(self) -> tuple[tuple[str, ...], ...]:
        """Return deterministic waves whose nodes can execute concurrently."""
        topology = self._topology
        indegree = topology.indegree.copy()
        ready = deque(
            node_id for node_id in topology.node_lookup if indegree[node_id] == 0
        )
        waves: list[tuple[str, ...]] = []
        while ready:
            wave = tuple(ready.popleft() for _ in range(len(ready)))
            waves.append(wave)
            for node_id in wave:
                for successor_id in topology.adjacency[node_id]:
                    indegree[successor_id] -= 1
                    if indegree[successor_id] == 0:
                        ready.append(successor_id)
        return tuple(waves)

    def execution_mode(self) -> Literal["parallel", "sequential", "hybrid"]:
        """Derive scheduling mode in O(V+E), ignoring control sinks."""
        answer_ids = [
            node.node_id
            for node in self.nodes
            if node.semantic_type is not NodeSemanticType.CONTROL
        ]
        if len(answer_ids) <= 1:
            return "sequential"

        topology = self._topology
        answer_set = set(answer_ids)
        answer_adjacency: dict[str, tuple[str, ...]] = {}
        answer_indegree = {node_id: 0 for node_id in answer_ids}
        dependency_count = 0
        for node_id in answer_ids:
            successors = tuple(
                successor_id
                for successor_id in topology.adjacency[node_id]
                if successor_id in answer_set
            )
            answer_adjacency[node_id] = successors
            dependency_count += len(successors)
            for successor_id in successors:
                answer_indegree[successor_id] += 1

        if dependency_count == 0:
            return "parallel"

        ready = deque(
            node_id for node_id in answer_ids if answer_indegree[node_id] == 0
        )
        unique_order = True
        visited = 0
        while ready:
            if len(ready) != 1:
                unique_order = False
            node_id = ready.popleft()
            visited += 1
            for successor_id in answer_adjacency[node_id]:
                answer_indegree[successor_id] -= 1
                if answer_indegree[successor_id] == 0:
                    ready.append(successor_id)

        if visited != len(answer_ids):
            raise RuntimeError("validated execution graph unexpectedly contains a cycle")
        return "sequential" if unique_order else "hybrid"

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Copy the graph, fully validating updates and rebuilding topology."""
        if update is None:
            return super().model_copy(deep=deep)

        model_data = self.model_dump(mode="python", round_trip=True)
        update_data = dict(update)
        if deep:
            model_data = deepcopy(model_data)
            update_data = deepcopy(update_data)
        model_data.update(update_data)
        return type(self).model_validate(model_data)
