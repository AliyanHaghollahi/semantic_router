"""Deterministic GraphDecoder from planner prediction structures to ExecutionGraph.

No second graph representation, no duplicate DAG validator, no C1-C4 / spaCy /
keyword / LLM repair. Invalid structures raise ``PlannerDecodeError``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tiergraph.enums import (
    FusionStrategy,
    NodeSemanticType,
    OperatorType,
    QueryType,
    SlotType,
    Tier,
    TransferPolicy,
)
from tiergraph.fusion import FusionPlan
from tiergraph.graph import DependencyEdge, ExecutionGraph, SemanticNode
from tiergraph.planner.annotations import (
    ImplicitResolution,
    _OPERATOR_SEMANTICS,
    _derive_query_type,
)
from tiergraph.planner.naming import (
    SlotNamingError,
    default_base_for_operator,
    fuse_input_slot_name,
    fuse_output_slot_name,
    normalize_base_name,
    principal_slot_name,
)
from tiergraph.planner.operator_io import (
    h7_dependency_slot_type,
    is_h7_pair_eligible,
    principal_output_type,
)
from tiergraph.planner.tasks import TaskTemplateError, render_answer_task, render_fuse_task


class PlannerDecodeError(ValueError):
    """Invalid or unrecoverable planner prediction structure."""


@dataclass(frozen=True, slots=True)
class PredictedOperation:
    """One predicted explicit answer operation span."""

    start: int
    end: int
    operator: OperatorType


@dataclass(frozen=True, slots=True)
class PredictedAnchor:
    """One predicted slot anchor with H6 ownership pointer."""

    start: int
    end: int
    text: str
    owner_index: int
    implicit_resolution: ImplicitResolution
    normalized_name: str | None = None


@dataclass(frozen=True, slots=True)
class PlannerPredictions:
    """Decoded head outputs consumed by ``GraphDecoder``.

    ``aux_query_type`` may carry H1 for metrics only; it never overrides graph
    semantics.
    """

    operations: tuple[PredictedOperation, ...]
    anchors: tuple[PredictedAnchor, ...]
    dependency_pairs: frozenset[tuple[int, int]] = frozenset()
    aux_query_type: QueryType | None = None


@dataclass(frozen=True, slots=True)
class DecodedPlan:
    """Validated execution graph plus optional multi-sink fusion plan.

    Public success type for :meth:`GraphDecoder.decode`. Failures raise
    :class:`PlannerDecodeError` instead of returning an error union.
    ``fusion_plan`` is ``None`` for single-sink graphs and an
    annotation-valid ``FusionPlan(VALIDATED_SLM)`` when FUSE is inserted.
    Phase-3 does not execute ``VALIDATED_SLM``; structural prediction only.
    """

    graph: ExecutionGraph
    fusion_plan: FusionPlan | None


def default_tier_for_operator(operator: OperatorType) -> Tier:
    """V1 deterministic tier assignment for answer operators."""
    if operator is OperatorType.FUSE:
        return Tier.EDGE
    semantic = _OPERATOR_SEMANTICS[operator]
    if semantic is NodeSemanticType.PERSONAL:
        return Tier.EDGE
    if semantic is NodeSemanticType.ENVIRONMENTAL:
        return Tier.FOG
    raise PlannerDecodeError(f"unsupported operator for tier assignment: {operator}")


def transfer_policy_for_tiers(source_tier: Tier, target_tier: Tier) -> TransferPolicy:
    """Deterministic transfer policy from endpoint tiers."""
    if source_tier is Tier.EDGE and target_tier is Tier.FOG:
        return TransferPolicy.MINIMAL_REFERENCE
    return TransferPolicy.DIRECT


class GraphDecoder:
    """Assemble an ``ExecutionGraph`` from predicted planner structures."""

    def decode(
        self,
        predictions: PlannerPredictions,
        *,
        query: str,
        graph_id: str,
    ) -> DecodedPlan:
        """Decode predictions into a validated ``DecodedPlan``.

        Returns a typed success object with ``graph`` and optional
        ``fusion_plan``. Raises ``PlannerDecodeError`` on invalid or
        unrecoverable structures. ``predictions.aux_query_type`` is ignored
        for graph semantics.
        """
        try:
            return self._decode(predictions, query=query, graph_id=graph_id)
        except PlannerDecodeError:
            raise
        except (SlotNamingError, TaskTemplateError, ValueError, TypeError) as exc:
            raise PlannerDecodeError(str(exc)) from exc

    def _decode(
        self,
        predictions: PlannerPredictions,
        *,
        query: str,
        graph_id: str,
    ) -> DecodedPlan:
        if type(query) is not str or not query.strip():
            raise PlannerDecodeError("query must be a nonblank string")
        if type(graph_id) is not str or not graph_id.strip():
            raise PlannerDecodeError("graph_id must be a nonblank string")
        if not predictions.operations:
            raise PlannerDecodeError("at least one explicit operation is required")

        operations = predictions.operations
        _validate_operation_spans(query=query, operations=operations)
        _validate_anchors(query=query, anchors=predictions.anchors, n_ops=len(operations))

        base_names = _base_names_for_operations(
            operations=operations,
            anchors=predictions.anchors,
        )

        explicit_nodes: list[SemanticNode] = []
        explicit_ids: list[str] = []
        for index, operation in enumerate(operations):
            node_id = f"op_{index + 1}"
            explicit_ids.append(node_id)
            base_name = base_names[index]
            principal_type = principal_output_type(operation.operator)
            produced_name = principal_slot_name(
                base_name=base_name,
                slot_type=principal_type,
            )
            explicit_nodes.append(
                SemanticNode.model_validate(
                    {
                        "node_id": node_id,
                        "semantic_type": _OPERATOR_SEMANTICS[operation.operator],
                        "operator": operation.operator,
                        "tier": default_tier_for_operator(operation.operator),
                        "task": render_answer_task(
                            operator=operation.operator,
                            base_name=base_name,
                        ),
                        "required_inputs": {},
                        "produced_outputs": {produced_name: principal_type},
                        "status": "pending",
                        "metadata": {},
                    }
                )
            )

        # Mutable slot maps while wiring dependencies.
        required_inputs: list[dict[str, SlotType]] = [
            dict(node.required_inputs) for node in explicit_nodes
        ]
        produced_outputs: list[dict[str, SlotType]] = [
            dict(node.produced_outputs) for node in explicit_nodes
        ]

        implicit_nodes: list[SemanticNode] = []
        edges: list[DependencyEdge] = []
        implicit_counter = 0

        for anchor in predictions.anchors:
            owner_index = anchor.owner_index
            if anchor.implicit_resolution is ImplicitResolution.NONE:
                continue
            implicit_counter += 1
            implicit_id = f"impl_{implicit_counter}"
            base_name = _anchor_base_name(anchor)
            produced_name = principal_slot_name(
                base_name=base_name,
                slot_type=SlotType.RESOLVED_REFERENCE,
            )
            implicit_node = SemanticNode.model_validate(
                {
                    "node_id": implicit_id,
                    "semantic_type": NodeSemanticType.PERSONAL,
                    "operator": OperatorType.RESOLVE_PERSONAL,
                    "tier": Tier.EDGE,
                    "task": render_answer_task(
                        operator=OperatorType.RESOLVE_PERSONAL,
                        base_name=base_name,
                    ),
                    "required_inputs": {},
                    "produced_outputs": {
                        produced_name: SlotType.RESOLVED_REFERENCE
                    },
                    "status": "pending",
                    "metadata": {},
                }
            )
            implicit_nodes.append(implicit_node)

            owner_node = explicit_nodes[owner_index]
            if produced_name in required_inputs[owner_index]:
                raise PlannerDecodeError(
                    f"duplicate required input {produced_name!r} on {owner_node.node_id}"
                )
            if produced_name in produced_outputs[owner_index]:
                raise PlannerDecodeError(
                    f"required input {produced_name!r} collides with produced "
                    f"output on {owner_node.node_id}"
                )
            required_inputs[owner_index][produced_name] = SlotType.RESOLVED_REFERENCE
            edges.append(
                DependencyEdge.model_validate(
                    {
                        "source_node_id": implicit_id,
                        "source_slot": produced_name,
                        "target_node_id": owner_node.node_id,
                        "target_slot": produced_name,
                        "transfer_policy": transfer_policy_for_tiers(
                            Tier.EDGE,
                            owner_node.tier,
                        ).value,
                    }
                )
            )

        # H7 explicit dependencies: presence only; slots from V1 contract.
        for source_index, target_index in sorted(predictions.dependency_pairs):
            if not (
                0 <= source_index < len(operations)
                and 0 <= target_index < len(operations)
            ):
                raise PlannerDecodeError(
                    f"dependency pair out of range: {(source_index, target_index)}"
                )
            if source_index == target_index:
                raise PlannerDecodeError("dependency self-loop is invalid")
            source_op = operations[source_index].operator
            target_op = operations[target_index].operator
            if not is_h7_pair_eligible(source_op, target_op):
                raise PlannerDecodeError(
                    "incompatible explicit dependency under "
                    f"OPERATOR_IO_CONTRACT_V1: {source_op.value} -> {target_op.value}"
                )
            slot_type = h7_dependency_slot_type(source_op, target_op)
            source_node = explicit_nodes[source_index]
            source_slot = next(iter(produced_outputs[source_index]))
            source_principal = produced_outputs[source_index][source_slot]
            if source_principal is not slot_type:
                raise PlannerDecodeError(
                    "source principal SlotType mismatch for H7 edge"
                )
            target_node = explicit_nodes[target_index]
            if source_slot in required_inputs[target_index]:
                raise PlannerDecodeError(
                    f"duplicate required input {source_slot!r} on {target_node.node_id}"
                )
            if source_slot in produced_outputs[target_index]:
                raise PlannerDecodeError(
                    f"required input {source_slot!r} collides with produced "
                    f"output on {target_node.node_id}"
                )
            required_inputs[target_index][source_slot] = slot_type
            edges.append(
                DependencyEdge.model_validate(
                    {
                        "source_node_id": source_node.node_id,
                        "source_slot": source_slot,
                        "target_node_id": target_node.node_id,
                        "target_slot": source_slot,
                        "transfer_policy": transfer_policy_for_tiers(
                            source_node.tier,
                            target_node.tier,
                        ).value,
                    }
                )
            )

        # Rebuild explicit nodes with wired required_inputs.
        wired_explicit = [
            SemanticNode.model_validate(
                {
                    **node.model_dump(mode="python", round_trip=True),
                    "required_inputs": required_inputs[index],
                    "produced_outputs": produced_outputs[index],
                }
            )
            for index, node in enumerate(explicit_nodes)
        ]

        answer_nodes = tuple([*implicit_nodes, *wired_explicit])
        query_type = _derive_query_type(answer_nodes)

        fuse_node: SemanticNode | None = None
        fusion_plan: FusionPlan | None = None
        answer_ids = {node.node_id for node in answer_nodes}
        answer_sources = {
            edge.source_node_id
            for edge in edges
            if edge.source_node_id in answer_ids and edge.target_node_id in answer_ids
        }
        sinks = tuple(
            node for node in answer_nodes if node.node_id not in answer_sources
        )

        if len(sinks) > 1:
            fuse_id = "fuse"
            fuse_required: dict[str, SlotType] = {}
            ordered_slots: list[str] = []
            for sink in sinks:
                source_slot = next(iter(sink.produced_outputs))
                slot_type = sink.produced_outputs[source_slot]
                fuse_slot = fuse_input_slot_name(
                    source_node_id=sink.node_id,
                    source_slot=source_slot,
                )
                fuse_required[fuse_slot] = slot_type
                ordered_slots.append(fuse_slot)
                edges.append(
                    DependencyEdge.model_validate(
                        {
                            "source_node_id": sink.node_id,
                            "source_slot": source_slot,
                            "target_node_id": fuse_id,
                            "target_slot": fuse_slot,
                            "transfer_policy": transfer_policy_for_tiers(
                                sink.tier,
                                Tier.EDGE,
                            ).value,
                        }
                    )
                )
            fuse_node = SemanticNode.model_validate(
                {
                    "node_id": fuse_id,
                    "semantic_type": NodeSemanticType.CONTROL,
                    "operator": OperatorType.FUSE,
                    "tier": Tier.EDGE,
                    "task": render_fuse_task(),
                    "required_inputs": fuse_required,
                    "produced_outputs": {
                        fuse_output_slot_name(): SlotType.FINAL_RESPONSE
                    },
                    "status": "pending",
                    "metadata": {},
                }
            )
            fusion_plan = FusionPlan.model_validate(
                {
                    "schema_version": "1.0",
                    "plan_id": f"{graph_id}__fuse",
                    "graph_id": graph_id,
                    "fusion_node_id": fuse_id,
                    "strategy": FusionStrategy.VALIDATED_SLM,
                    "required_slots": fuse_required,
                    "ordered_slots": ordered_slots,
                    "max_sentences": 2,
                    "spoken_style": True,
                    "instructions": (
                        "Fuse the typed answers into a concise spoken response."
                    ),
                    "metadata": {
                        "phase4_note": (
                            "FusionStrategy.VALIDATED_SLM is predicted structurally; "
                            "Phase-3 executor does not implement VALIDATED_SLM."
                        )
                    },
                }
            )
        elif len(sinks) == 0:
            raise PlannerDecodeError("decoded graph has no answer sinks")

        nodes: list[SemanticNode] = [*implicit_nodes, *wired_explicit]
        if fuse_node is not None:
            nodes.append(fuse_node)

        graph = ExecutionGraph.model_validate(
            {
                "schema_version": "1.0",
                "graph_id": graph_id,
                "original_query": query,
                "query_type": query_type,
                "nodes": [
                    node.model_dump(mode="python", round_trip=True) for node in nodes
                ],
                "edges": [
                    edge.model_dump(mode="python", round_trip=True) for edge in edges
                ],
                "metadata": {},
            }
        )
        if fusion_plan is not None:
            fusion_plan.validate_against_graph(graph)
        return DecodedPlan(graph=graph, fusion_plan=fusion_plan)


def _validate_operation_spans(
    *,
    query: str,
    operations: Sequence[PredictedOperation],
) -> None:
    ordered = sorted(
        enumerate(operations),
        key=lambda item: (item[1].start, item[1].end, item[0]),
    )
    previous: PredictedOperation | None = None
    previous_index: int | None = None
    for index, operation in ordered:
        if operation.end <= operation.start:
            raise PlannerDecodeError(
                f"operation span must be nonempty at index {index}"
            )
        if operation.end > len(query):
            raise PlannerDecodeError(
                f"operation span outside query at index {index}"
            )
        if operation.operator is OperatorType.FUSE:
            raise PlannerDecodeError("predicted operations cannot be FUSE")
        if operation.operator not in _OPERATOR_SEMANTICS:
            raise PlannerDecodeError(
                f"unsupported operation operator: {operation.operator.value}"
            )
        if previous is not None and operation.start < previous.end:
            raise PlannerDecodeError(
                "operation spans must not overlap or nest: "
                f"indices {previous_index} and {index}"
            )
        previous = operation
        previous_index = index


def _validate_anchors(
    *,
    query: str,
    anchors: Sequence[PredictedAnchor],
    n_ops: int,
) -> None:
    for index, anchor in enumerate(anchors):
        if anchor.end <= anchor.start:
            raise PlannerDecodeError(f"anchor span must be nonempty at index {index}")
        if anchor.end > len(query):
            raise PlannerDecodeError(f"anchor span outside query at index {index}")
        if query[anchor.start : anchor.end] != anchor.text:
            raise PlannerDecodeError(
                f"anchor text does not match query[start:end] at index {index}"
            )
        if not (0 <= anchor.owner_index < n_ops):
            raise PlannerDecodeError(
                f"anchor owner_index out of range at index {index}"
            )
        if anchor.implicit_resolution not in ImplicitResolution:
            raise PlannerDecodeError(
                f"invalid implicit_resolution at anchor index {index}"
            )


def _anchor_base_name(anchor: PredictedAnchor) -> str:
    if anchor.normalized_name is not None:
        return normalize_base_name(anchor.normalized_name)
    return normalize_base_name(anchor.text)


def _base_names_for_operations(
    *,
    operations: Sequence[PredictedOperation],
    anchors: Sequence[PredictedAnchor],
) -> list[str]:
    """Resolve principal slot bases: owned anchor if present, else operator default."""
    names: list[str | None] = [None] * len(operations)
    for anchor in anchors:
        base = _anchor_base_name(anchor)
        current = names[anchor.owner_index]
        if current is None:
            names[anchor.owner_index] = base
        elif current != base:
            raise PlannerDecodeError(
                "SLOT_NAMING_V1 cannot represent multiple distinct base names "
                f"on one operation: {current!r} vs {base!r}"
            )
    resolved: list[str] = []
    for index, operation in enumerate(operations):
        owned = names[index]
        if owned is not None:
            resolved.append(owned)
            continue
        resolved.append(default_base_for_operator(operation.operator))
    return resolved


__all__ = [
    "DecodedPlan",
    "GraphDecoder",
    "PlannerDecodeError",
    "PlannerPredictions",
    "PredictedAnchor",
    "PredictedOperation",
    "default_tier_for_operator",
    "transfer_policy_for_tiers",
]
