"""Strict annotation schemas for supervised TierGraph planner examples.

Character offsets are zero-based Python string offsets over the exact original
query and use the half-open convention ``[start, end)``. Queries must not be
normalized before offsets and annotated text are checked.

Annotation models prevent attribute reassignment, but their nested containers
are only shallowly frozen. Callers must treat metadata dictionaries, graph and
fusion-plan nested mappings, node ``required_inputs`` and ``produced_outputs``,
and planner-label collections as immutable after validation.

An anchor labeled ``IMPLICIT_RESOLVE_PERSONAL`` deterministically creates a
``RESOLVED_REFERENCE`` dependency from its implicit node to ``owner_node_id``.
That mandatory pair is annotation structure, not a learned dependency: later
training and decoding must mask it from ordinary dependency-head loss and
ordinary learned dependency decoding.
"""

from copy import deepcopy
from enum import Enum
from typing import Any, Mapping, Self

from pydantic import Field, JsonValue, field_validator, model_validator

from tiergraph.enums import (
    FusionStrategy,
    NodeSemanticType,
    OperatorType,
    QueryType,
    SlotType,
    Tier,
    TransferPolicy,
    _CanonicalWireEnum,
)
from tiergraph.fusion import FusionPlan
from tiergraph.graph import ExecutionGraph, SemanticNode
from tiergraph.models import TierGraphSchema


_OPERATOR_SEMANTICS = {
    OperatorType.RESOLVE_PERSONAL: NodeSemanticType.PERSONAL,
    OperatorType.RETRIEVE_PERSONAL: NodeSemanticType.PERSONAL,
    OperatorType.IDENTIFY_ENVIRONMENTAL: NodeSemanticType.ENVIRONMENTAL,
    OperatorType.LOCATE_ENVIRONMENTAL: NodeSemanticType.ENVIRONMENTAL,
    OperatorType.NAVIGATE_TO: NodeSemanticType.ENVIRONMENTAL,
    OperatorType.DESCRIBE_ENVIRONMENT: NodeSemanticType.ENVIRONMENTAL,
}

_PRINCIPAL_OUTPUT_TYPES = {
    OperatorType.RESOLVE_PERSONAL: SlotType.RESOLVED_REFERENCE,
    OperatorType.RETRIEVE_PERSONAL: SlotType.PERSONAL_FACT,
    OperatorType.IDENTIFY_ENVIRONMENTAL: SlotType.ENVIRONMENTAL_FACT,
    OperatorType.LOCATE_ENVIRONMENTAL: SlotType.LOCATION,
    OperatorType.NAVIGATE_TO: SlotType.NAVIGATION_INSTRUCTION,
    OperatorType.DESCRIBE_ENVIRONMENT: SlotType.SCENE_DESCRIPTION,
}


class ImplicitResolution(_CanonicalWireEnum, str, Enum):
    """Whether a slot anchor creates an implicit personal resolver node."""

    NONE = "NONE"
    IMPLICIT_RESOLVE_PERSONAL = "IMPLICIT_RESOLVE_PERSONAL"


class _ValidatedAnnotationSchema(TierGraphSchema):
    """Revalidate annotation updates instead of copying unchecked state."""

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Copy this model, fully validating any supplied field updates."""
        if update is None:
            return super().model_copy(deep=deep)

        model_data = self.model_dump(mode="python", round_trip=True)
        update_data = dict(update)
        if deep:
            model_data = deepcopy(model_data)
            update_data = deepcopy(update_data)
        model_data.update(update_data)
        return type(self).model_validate(model_data)


class OperationSpanLabel(_ValidatedAnnotationSchema):
    """One explicit answer operation aligned to a query character span."""

    node_id: str
    semantic_type: NodeSemanticType
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    operator: OperatorType

    @field_validator("node_id")
    @classmethod
    def _validate_node_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("node_id must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_explicit_operation(self) -> "OperationSpanLabel":
        if self.end <= self.start:
            raise ValueError("operation span must be nonempty")
        if self.semantic_type is NodeSemanticType.CONTROL:
            raise ValueError("operation spans cannot represent control nodes")
        if self.operator is OperatorType.FUSE:
            raise ValueError("operation spans cannot represent FUSE")
        expected_semantic = _OPERATOR_SEMANTICS[self.operator]
        if self.semantic_type is not expected_semantic:
            raise ValueError(
                f"{self.operator.value} requires {expected_semantic.value} semantics"
            )
        return self


class SlotAnchorLabel(_ValidatedAnnotationSchema):
    """One owned slot anchor aligned to exact original-query text."""

    anchor_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    normalized_name: str
    owner_node_id: str
    implicit_resolution: ImplicitResolution
    implicit_node_id: str | None

    @field_validator(
        "anchor_id",
        "text",
        "normalized_name",
        "owner_node_id",
    )
    @classmethod
    def _validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_implicit_node_reference(self) -> "SlotAnchorLabel":
        if self.end <= self.start:
            raise ValueError("slot anchor span must be nonempty")
        if self.implicit_resolution is ImplicitResolution.NONE:
            if self.implicit_node_id is not None:
                raise ValueError(
                    "implicit_node_id must be null when implicit_resolution is NONE"
                )
        elif self.implicit_node_id is None or not self.implicit_node_id.strip():
            raise ValueError(
                "implicit_node_id must be nonblank for "
                "IMPLICIT_RESOLVE_PERSONAL"
            )
        return self


class PlannerLabels(_ValidatedAnnotationSchema):
    """Planner supervision associated with one validated execution graph."""

    query_type: QueryType
    operation_spans: tuple[OperationSpanLabel, ...]
    slot_anchors: tuple[SlotAnchorLabel, ...]

    @field_validator("operation_spans", "slot_anchors", mode="before")
    @classmethod
    def _normalize_json_arrays(cls, value: object) -> object:
        if type(value) is list:
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_label_identifiers(self) -> "PlannerLabels":
        if not self.operation_spans:
            raise ValueError("operation_spans must not be empty")

        operation_ids = [span.node_id for span in self.operation_spans]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("duplicate operation node_id")

        anchor_ids = [anchor.anchor_id for anchor in self.slot_anchors]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("duplicate anchor_id")
        return self


class PlannerExample(_ValidatedAnnotationSchema):
    """A complete, graph-validated planner annotation record."""

    example_id: str
    query: str
    graph: ExecutionGraph
    fusion_plan: FusionPlan | None
    planner_labels: PlannerLabels
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("example_id", "query")
    @classmethod
    def _validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_annotation_contract(self) -> "PlannerExample":
        graph = ExecutionGraph.model_validate(
            self.graph.model_dump(mode="python", round_trip=True)
        )
        labels = PlannerLabels.model_validate(
            self.planner_labels.model_dump(mode="python", round_trip=True)
        )

        if graph.original_query != self.query:
            raise ValueError("graph.original_query must equal query exactly")
        if labels.query_type is not graph.query_type:
            raise ValueError(
                "planner_labels.query_type must equal graph.query_type"
            )

        answer_nodes = tuple(
            node
            for node in graph.nodes
            if node.semantic_type is not NodeSemanticType.CONTROL
        )
        answer_map = {node.node_id: node for node in answer_nodes}
        derived_query_type = _derive_query_type(answer_nodes)
        if graph.query_type is not derived_query_type:
            raise ValueError(
                "graph.query_type does not match answer-node semantics: "
                f"expected {derived_query_type.value}"
            )

        _validate_principal_outputs(answer_nodes)
        operation_map = _validate_operation_spans(
            query=self.query,
            operation_spans=labels.operation_spans,
            answer_map=answer_map,
        )
        implicit_node_ids = _validate_anchors(
            query=self.query,
            anchors=labels.slot_anchors,
            operation_map=operation_map,
            answer_map=answer_map,
            graph=graph,
        )

        explicit_answer_ids = set(answer_map) - implicit_node_ids
        if set(operation_map) != explicit_answer_ids:
            missing = sorted(explicit_answer_ids - set(operation_map))
            unexpected = sorted(set(operation_map) - explicit_answer_ids)
            raise ValueError(
                "every explicit answer node must have exactly one operation span; "
                f"missing={missing}, unexpected={unexpected}"
            )

        _validate_fusion_contract(graph, self.fusion_plan, answer_nodes)
        return self


def _derive_query_type(answer_nodes: tuple[SemanticNode, ...]) -> QueryType:
    has_personal = any(
        node.semantic_type is NodeSemanticType.PERSONAL for node in answer_nodes
    )
    has_environmental = any(
        node.semantic_type is NodeSemanticType.ENVIRONMENTAL
        for node in answer_nodes
    )
    if has_personal and has_environmental:
        return QueryType.MIXED
    if has_personal:
        return QueryType.PERSONAL
    if has_environmental:
        return QueryType.ENVIRONMENTAL
    raise ValueError("an annotation graph must contain at least one answer node")


def _validate_principal_outputs(answer_nodes: tuple[SemanticNode, ...]) -> None:
    for node in answer_nodes:
        expected_type = _PRINCIPAL_OUTPUT_TYPES.get(node.operator)
        if expected_type is None:
            raise ValueError(
                f"unsupported answer operator in planner annotation: "
                f"{node.operator.value}"
            )
        output_types = tuple(node.produced_outputs.values())
        if output_types != (expected_type,):
            raise ValueError(
                f"{node.node_id} must produce exactly one principal output "
                f"of type {expected_type.value}"
            )


def _validate_operation_spans(
    *,
    query: str,
    operation_spans: tuple[OperationSpanLabel, ...],
    answer_map: dict[str, SemanticNode],
) -> dict[str, OperationSpanLabel]:
    operation_map: dict[str, OperationSpanLabel] = {}
    ordered_spans = sorted(
        operation_spans,
        key=lambda span: (span.start, span.end, span.node_id),
    )
    previous: OperationSpanLabel | None = None

    for span in ordered_spans:
        if span.end > len(query):
            raise ValueError(
                f"operation span for {span.node_id} is outside the query"
            )
        if previous is not None and span.start < previous.end:
            raise ValueError(
                "operation spans must not overlap or nest: "
                f"{previous.node_id}, {span.node_id}"
            )
        previous = span

        if span.node_id in operation_map:
            raise ValueError(f"duplicate operation node_id: {span.node_id}")
        node = answer_map.get(span.node_id)
        if node is None:
            raise ValueError(
                f"operation span must reference an answer node: {span.node_id}"
            )
        if span.semantic_type is not node.semantic_type:
            raise ValueError(
                f"operation semantic_type does not match graph node: {span.node_id}"
            )
        if span.operator is not node.operator:
            raise ValueError(
                f"operation operator does not match graph node: {span.node_id}"
            )
        operation_map[span.node_id] = span

    return operation_map


def _validate_anchors(
    *,
    query: str,
    anchors: tuple[SlotAnchorLabel, ...],
    operation_map: dict[str, OperationSpanLabel],
    answer_map: dict[str, SemanticNode],
    graph: ExecutionGraph,
) -> set[str]:
    anchor_ids: set[str] = set()
    implicit_node_ids: set[str] = set()

    for anchor in anchors:
        if anchor.anchor_id in anchor_ids:
            raise ValueError(f"duplicate anchor_id: {anchor.anchor_id}")
        anchor_ids.add(anchor.anchor_id)

        if anchor.end > len(query):
            raise ValueError(
                f"slot anchor {anchor.anchor_id} is outside the query"
            )
        if query[anchor.start : anchor.end] != anchor.text:
            raise ValueError(
                f"slot anchor {anchor.anchor_id} text does not match "
                "query[start:end]"
            )
        if anchor.owner_node_id not in operation_map:
            raise ValueError(
                f"anchor owner must reference an explicit operation node: "
                f"{anchor.owner_node_id}"
            )

        if anchor.implicit_resolution is ImplicitResolution.NONE:
            continue

        implicit_node_id = anchor.implicit_node_id
        if implicit_node_id is None:
            raise ValueError("positive implicit resolution requires a node ID")
        if implicit_node_id in implicit_node_ids:
            raise ValueError(
                "each positive anchor must reference a distinct implicit node: "
                f"{implicit_node_id}"
            )
        if implicit_node_id in operation_map:
            raise ValueError(
                f"implicit node must not have an operation span: {implicit_node_id}"
            )
        implicit_node = answer_map.get(implicit_node_id)
        if implicit_node is None:
            raise ValueError(
                f"implicit_node_id does not reference an answer node: "
                f"{implicit_node_id}"
            )
        _validate_implicit_node(implicit_node)
        _validate_implicit_owner_edge(
            graph=graph,
            implicit_node=implicit_node,
            owner_node=answer_map[anchor.owner_node_id],
        )
        implicit_node_ids.add(implicit_node_id)

    return implicit_node_ids


def _validate_implicit_node(node: SemanticNode) -> None:
    if node.semantic_type is not NodeSemanticType.PERSONAL:
        raise ValueError(f"implicit node must be personal: {node.node_id}")
    if node.operator is not OperatorType.RESOLVE_PERSONAL:
        raise ValueError(
            f"implicit node must use RESOLVE_PERSONAL: {node.node_id}"
        )
    if node.tier is not Tier.EDGE:
        raise ValueError(f"implicit node must execute on Edge: {node.node_id}")
    if node.required_inputs:
        raise ValueError(
            f"implicit RESOLVE_PERSONAL node must be a root: {node.node_id}"
        )
    if tuple(node.produced_outputs.values()) != (
        SlotType.RESOLVED_REFERENCE,
    ):
        raise ValueError(
            "implicit RESOLVE_PERSONAL node must produce exactly one "
            f"RESOLVED_REFERENCE: {node.node_id}"
        )


def _validate_implicit_owner_edge(
    *,
    graph: ExecutionGraph,
    implicit_node: SemanticNode,
    owner_node: SemanticNode,
) -> None:
    source_slot = next(iter(implicit_node.produced_outputs))
    pair_edges = [
        edge
        for edge in graph.edges
        if edge.source_node_id == implicit_node.node_id
        and edge.target_node_id == owner_node.node_id
    ]
    if len(pair_edges) != 1:
        raise ValueError(
            "IMPLICIT_RESOLVE_PERSONAL must create exactly one "
            f"RESOLVED_REFERENCE edge from {implicit_node.node_id} to "
            f"{owner_node.node_id}"
        )

    edge = pair_edges[0]
    if edge.source_slot != source_slot:
        raise ValueError(
            "implicit-owner edge must use the implicit node's sole "
            f"RESOLVED_REFERENCE output slot: {source_slot}"
        )
    if edge.target_slot != source_slot:
        raise ValueError(
            "implicit-owner edge target slot must match the resolved "
            f"reference output slot: {source_slot}"
        )
    source_type = implicit_node.produced_outputs.get(edge.source_slot)
    target_type = owner_node.required_inputs.get(edge.target_slot)
    if (
        source_type is not SlotType.RESOLVED_REFERENCE
        or target_type is not SlotType.RESOLVED_REFERENCE
        or source_type is not target_type
    ):
        raise ValueError(
            "implicit-owner edge must transfer matching RESOLVED_REFERENCE "
            "slot types"
        )

    expected_policy = (
        TransferPolicy.MINIMAL_REFERENCE
        if implicit_node.tier is Tier.EDGE and owner_node.tier is Tier.FOG
        else TransferPolicy.DIRECT
    )
    if edge.transfer_policy is not expected_policy:
        raise ValueError(
            "implicit-owner edge has incorrect transfer policy: "
            f"expected {expected_policy.value}"
        )


def _validate_fusion_contract(
    graph: ExecutionGraph,
    fusion_plan: FusionPlan | None,
    answer_nodes: tuple[SemanticNode, ...],
) -> None:
    answer_ids = {node.node_id for node in answer_nodes}
    answer_sources = {
        edge.source_node_id
        for edge in graph.edges
        if edge.source_node_id in answer_ids and edge.target_node_id in answer_ids
    }
    answer_sinks = tuple(
        node for node in answer_nodes if node.node_id not in answer_sources
    )
    fuse_nodes = tuple(
        node for node in graph.nodes if node.operator is OperatorType.FUSE
    )

    if len(answer_sinks) == 1:
        if fuse_nodes:
            raise ValueError("FUSE must be absent when there is one answer sink")
        if fusion_plan is not None:
            raise ValueError("fusion_plan must be null when FUSE is absent")
        return

    if len(fuse_nodes) != 1:
        raise ValueError(
            "multiple answer sinks require exactly one terminal Edge FUSE"
        )
    fuse_node = fuse_nodes[0]
    if (
        fuse_node.semantic_type is not NodeSemanticType.CONTROL
        or fuse_node.tier is not Tier.EDGE
    ):
        raise ValueError("FUSE must be an Edge control node")

    incoming_fuse_edges = tuple(
        edge for edge in graph.edges if edge.target_node_id == fuse_node.node_id
    )
    incoming_sources = [edge.source_node_id for edge in incoming_fuse_edges]
    sink_ids = {node.node_id for node in answer_sinks}
    if len(incoming_sources) != len(sink_ids) or set(incoming_sources) != sink_ids:
        raise ValueError(
            "FUSE must receive exactly one dependency from every answer sink"
        )
    if fusion_plan is None:
        raise ValueError("multiple answer sinks require a FusionPlan")

    validated_plan = FusionPlan.model_validate(
        fusion_plan.model_dump(mode="python", round_trip=True)
    )
    if validated_plan.strategy is not FusionStrategy.VALIDATED_SLM:
        raise ValueError("deterministic fusion strategy must be validated_slm")
    validated_plan.validate_against_graph(graph)
