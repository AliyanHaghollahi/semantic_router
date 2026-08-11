"""Oracle TierGraph DAG executor (Phase 3).

Executes an already-validated :class:`~tiergraph.graph.ExecutionGraph` using
``execution_waves()`` for scheduling. Independent nodes in a wave run
concurrently via ``asyncio.gather``.

This module does **not** predict graphs. Planner prediction remains separate.

Fusion behavior in this phase is an explicit temporary oracle placeholder
(``phase3_temporary_concatenate``). It is **not**
``FusionStrategy.VALIDATED_SLM`` and must not be treated as typed/learned
fusion.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import JsonValue

from tiergraph.enums import (
    ExecutionStatus,
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
from tiergraph.models import EvidenceItem, TierResult

PHASE3_TEMPORARY_FUSION_METHOD = "phase3_temporary_concatenate"
_PHASE3_SUPPORTED_FUSION_STRATEGIES = frozenset({FusionStrategy.CONCATENATE})

NodeRunner = Callable[
    [SemanticNode, Mapping[str, "BoundInput"], "FogTransferRecord | None"],
    Mapping[str, JsonValue] | Awaitable[Mapping[str, JsonValue]],
]


class GraphExecutionError(ValueError):
    """Fail-fast contract or execution violation during graph execution."""


@dataclass(frozen=True)
class BoundInput:
    """One typed dependency value bound into a node (any legal edge)."""

    slot_name: str
    slot_type: SlotType
    value: JsonValue
    source_node_id: str
    source_slot: str
    transfer_policy: TransferPolicy
    source_tier: Tier
    target_tier: Tier


@dataclass(frozen=True)
class FogTransferRecord:
    """Values that crossed an Edge→Fog boundary for one Fog node.

    Not a universal dependency representation. Fog→Fog and other legal
    same-tier / Fog→Edge bindings propagate via :class:`BoundInput` only.
    """

    target_node_id: str
    task: str
    transferred_slots: dict[str, JsonValue]


@dataclass(frozen=True)
class GraphExecutionResult:
    """Successful oracle execution of one validated execution graph."""

    graph_id: str
    original_query: str
    query_type: QueryType
    results: dict[str, TierResult]
    waves: tuple[tuple[str, ...], ...]
    execution_mode: Literal["parallel", "sequential", "hybrid"]
    fog_transfers: tuple[FogTransferRecord, ...]
    total_latency_ms: float
    final_response: str
    fusion_method: str | None


class GraphExecutor:
    """Wave-scheduled executor for validated TierGraph DAGs."""

    def __init__(
        self,
        edge_client: Any,
        fog_client: Any,
        *,
        edge_context_fn: Callable[[str], str] | None = None,
        fog_context_fn: Callable[[str], str] | None = None,
        node_runner: NodeRunner | None = None,
    ) -> None:
        self._edge_client = edge_client
        self._fog_client = fog_client
        self._edge_context_fn = edge_context_fn
        self._fog_context_fn = fog_context_fn
        self._node_runner = node_runner

    async def execute(
        self,
        graph: ExecutionGraph,
        *,
        image_b64: str | None = None,
        fusion_plan: FusionPlan | None = None,
    ) -> GraphExecutionResult:
        """Execute ``graph`` fail-fast and return a successful result object."""
        t0 = time.perf_counter()
        _require_fuse_when_multiple_answer_sinks(graph)
        if fusion_plan is not None:
            fusion_plan.validate_against_graph(graph)
            _require_phase3_supported_fusion_strategy(fusion_plan)

        waves = graph.execution_waves()
        execution_mode = graph.execution_mode()
        results: dict[str, TierResult] = {}
        fog_transfers: list[FogTransferRecord] = []
        fusion_method: str | None = None

        for wave in waves:
            wave_results = await asyncio.gather(
                *(
                    self._execute_node(
                        graph,
                        graph.node_by_id(node_id),
                        results,
                        image_b64=image_b64,
                        fusion_plan=fusion_plan,
                    )
                    for node_id in wave
                )
            )
            # Publish transfers in deterministic wave/node order (not task completion order).
            for node_id, (tier_result, used_fusion_method, transfer) in zip(
                wave,
                wave_results,
                strict=True,
            ):
                results[node_id] = tier_result
                if transfer is not None:
                    fog_transfers.append(transfer)
                if used_fusion_method is not None:
                    fusion_method = used_fusion_method

        final_response = _derive_final_response(graph, results)
        return GraphExecutionResult(
            graph_id=graph.graph_id,
            original_query=graph.original_query,
            query_type=graph.query_type,
            results=results,
            waves=waves,
            execution_mode=execution_mode,
            fog_transfers=tuple(fog_transfers),
            total_latency_ms=(time.perf_counter() - t0) * 1000,
            final_response=final_response,
            fusion_method=fusion_method,
        )

    async def _execute_node(
        self,
        graph: ExecutionGraph,
        node: SemanticNode,
        results: Mapping[str, TierResult],
        *,
        image_b64: str | None,
        fusion_plan: FusionPlan | None,
    ) -> tuple[TierResult, str | None, FogTransferRecord | None]:
        bound_inputs = _bind_inputs(graph, node, results)
        t0 = time.perf_counter()

        if node.operator is OperatorType.FUSE:
            outputs = _phase3_temporary_concatenate(node, bound_inputs, fusion_plan)
            evidence = _evidence_for_outputs(graph, node, outputs)
            result = TierResult(
                result_id=_new_id("result"),
                graph_id=graph.graph_id,
                node_id=node.node_id,
                tier=node.tier,
                status=ExecutionStatus.SUCCEEDED,
                outputs=outputs,
                evidence=evidence,
                latency_ms=(time.perf_counter() - t0) * 1000,
                metadata={"fusion_method": PHASE3_TEMPORARY_FUSION_METHOD},
            )
            return result, PHASE3_TEMPORARY_FUSION_METHOD, None

        transfer: FogTransferRecord | None = None
        if node.tier is Tier.FOG:
            transfer = _build_edge_to_fog_transfer(graph, node, bound_inputs)

        try:
            outputs = await self._run_answer_node(
                node,
                bound_inputs,
                transfer,
                image_b64=image_b64,
            )
        except GraphExecutionError:
            raise
        except (AttributeError, TypeError, AssertionError):
            raise
        except Exception as exc:
            raise GraphExecutionError(
                f"node {node.node_id} failed during execution: {exc}"
            ) from exc

        _validate_produced_outputs(node, outputs)
        evidence = _evidence_for_outputs(graph, node, outputs)
        result = TierResult(
            result_id=_new_id("result"),
            graph_id=graph.graph_id,
            node_id=node.node_id,
            tier=node.tier,
            status=ExecutionStatus.SUCCEEDED,
            outputs=dict(outputs),
            evidence=evidence,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        return result, None, transfer

    async def _run_answer_node(
        self,
        node: SemanticNode,
        bound_inputs: Mapping[str, BoundInput],
        transfer: FogTransferRecord | None,
        *,
        image_b64: str | None,
    ) -> Mapping[str, JsonValue]:
        if self._node_runner is not None:
            produced = self._node_runner(node, bound_inputs, transfer)
            if inspect.isawaitable(produced):
                produced = await produced
            return produced

        return await self._default_client_runner(
            node,
            bound_inputs,
            transfer,
            image_b64=image_b64,
        )

    async def _default_client_runner(
        self,
        node: SemanticNode,
        bound_inputs: Mapping[str, BoundInput],
        transfer: FogTransferRecord | None,
        *,
        image_b64: str | None,
    ) -> Mapping[str, JsonValue]:
        prompt = _build_slot_json_prompt(node, bound_inputs, transfer)
        if node.tier is Tier.EDGE:
            context = self._edge_context_fn(node.task) if self._edge_context_fn else ""
            raw = await _call_generate(
                self._edge_client,
                prompt,
                context=context,
                image_b64=image_b64,
            )
        elif node.tier is Tier.FOG:
            env_context = self._fog_context_fn(node.task) if self._fog_context_fn else ""
            fog_context = _compose_fog_context(env_context, transfer, bound_inputs)
            # Edge→Fog privacy is structural: only transfer.transferred_slots
            # may carry Edge-origin values into the Fog request.
            raw = await _call_generate(
                self._fog_client,
                prompt,
                context=fog_context,
                image_b64=image_b64,
            )
        else:
            raise GraphExecutionError(f"unsupported tier for node {node.node_id}")

        return _parse_slot_json(raw, node)


def _require_fuse_when_multiple_answer_sinks(graph: ExecutionGraph) -> None:
    answer_nodes = [
        node
        for node in graph.nodes
        if node.semantic_type is not NodeSemanticType.CONTROL
    ]
    answer_ids = {node.node_id for node in answer_nodes}
    sources = {
        edge.source_node_id
        for edge in graph.edges
        if edge.source_node_id in answer_ids and edge.target_node_id in answer_ids
    }
    sinks = [node for node in answer_nodes if node.node_id not in sources]
    fuse_nodes = [node for node in graph.nodes if node.operator is OperatorType.FUSE]
    if len(sinks) > 1 and not fuse_nodes:
        sink_ids = sorted(node.node_id for node in sinks)
        raise GraphExecutionError(
            "multiple answer sinks require an explicit FUSE node; "
            f"found sinks={sink_ids}"
        )


def _require_phase3_supported_fusion_strategy(fusion_plan: FusionPlan) -> None:
    """Reject strategies that Phase-3 temporary fusion must not silently execute."""
    if fusion_plan.strategy in _PHASE3_SUPPORTED_FUSION_STRATEGIES:
        return
    if fusion_plan.strategy is FusionStrategy.VALIDATED_SLM:
        raise GraphExecutionError(
            "FusionStrategy.VALIDATED_SLM is not implemented in Phase 3; "
            "Phase-3 temporary fusion supports only concatenate "
            f"({PHASE3_TEMPORARY_FUSION_METHOD})"
        )
    raise GraphExecutionError(
        "unsupported FusionPlan.strategy for Phase-3 temporary fusion: "
        f"{fusion_plan.strategy.value}; Phase 3 supports only concatenate "
        f"({PHASE3_TEMPORARY_FUSION_METHOD})"
    )


def _bind_inputs(
    graph: ExecutionGraph,
    node: SemanticNode,
    results: Mapping[str, TierResult],
) -> dict[str, BoundInput]:
    incoming = [edge for edge in graph.edges if edge.target_node_id == node.node_id]
    bound: dict[str, BoundInput] = {}
    for edge in incoming:
        source = graph.node_by_id(edge.source_node_id)
        if edge.source_node_id not in results:
            raise GraphExecutionError(
                f"missing dependency result for {edge.source_node_id} "
                f"required by {node.node_id}.{edge.target_slot}"
            )
        source_result = results[edge.source_node_id]
        if source_result.status is not ExecutionStatus.SUCCEEDED:
            raise GraphExecutionError(
                f"dependency {edge.source_node_id} did not succeed "
                f"(status={source_result.status.value})"
            )
        if edge.source_slot not in source_result.outputs:
            raise GraphExecutionError(
                f"undeclared dependency output "
                f"{edge.source_node_id}.{edge.source_slot}"
            )
        if edge.source_slot not in source.produced_outputs:
            raise GraphExecutionError(
                f"source slot is not declared on node "
                f"{edge.source_node_id}.{edge.source_slot}"
            )
        source_type = source.produced_outputs[edge.source_slot]
        target_type = node.required_inputs.get(edge.target_slot)
        if target_type is None:
            raise GraphExecutionError(
                f"target slot is not declared on node "
                f"{node.node_id}.{edge.target_slot}"
            )
        if source_type is not target_type:
            raise GraphExecutionError(
                "incompatible dependency slot types: "
                f"{source_type.value} -> {target_type.value}"
            )

        # Structural Edge→Fog transfer constraints are checked when building
        # FogTransferRecord; binding still carries the typed value.
        if source.tier is Tier.EDGE and node.tier is Tier.FOG:
            _validate_edge_to_fog_contract(edge, source_type)

        bound[edge.target_slot] = BoundInput(
            slot_name=edge.target_slot,
            slot_type=target_type,
            value=source_result.outputs[edge.source_slot],
            source_node_id=edge.source_node_id,
            source_slot=edge.source_slot,
            transfer_policy=edge.transfer_policy,
            source_tier=source.tier,
            target_tier=node.tier,
        )

    missing = set(node.required_inputs) - set(bound)
    if missing:
        raise GraphExecutionError(
            f"unresolved required inputs for {node.node_id}: {sorted(missing)}"
        )
    return bound


def _validate_edge_to_fog_contract(edge: DependencyEdge, source_type: SlotType) -> None:
    if edge.transfer_policy is not TransferPolicy.MINIMAL_REFERENCE:
        raise GraphExecutionError(
            "Edge→Fog dependencies require TransferPolicy.MINIMAL_REFERENCE "
            f"(edge {edge.source_node_id}->{edge.target_node_id})"
        )
    if source_type is not SlotType.RESOLVED_REFERENCE:
        raise GraphExecutionError(
            "Edge→Fog dependencies may transfer only RESOLVED_REFERENCE; "
            f"got {source_type.value} for "
            f"{edge.source_node_id}.{edge.source_slot}"
        )


def _build_edge_to_fog_transfer(
    graph: ExecutionGraph,
    node: SemanticNode,
    bound_inputs: Mapping[str, BoundInput],
) -> FogTransferRecord | None:
    transferred: dict[str, JsonValue] = {}
    for edge in graph.edges:
        if edge.target_node_id != node.node_id:
            continue
        source = graph.node_by_id(edge.source_node_id)
        if source.tier is not Tier.EDGE or node.tier is not Tier.FOG:
            continue

        bound = bound_inputs.get(edge.target_slot)
        if bound is None:
            raise GraphExecutionError(
                f"missing bound input for Edge→Fog slot {edge.target_slot}"
            )
        _validate_edge_to_fog_contract(edge, bound.slot_type)
        if bound.slot_type is not SlotType.RESOLVED_REFERENCE:
            raise GraphExecutionError(
                "Edge→Fog transfer rejected non-RESOLVED_REFERENCE slot "
                f"{bound.slot_name}"
            )
        # Serialize only the permitted reference value — never raw model text.
        transferred[edge.target_slot] = bound.value

    if not transferred:
        return None
    return FogTransferRecord(
        target_node_id=node.node_id,
        task=node.task,
        transferred_slots=transferred,
    )


def _phase3_temporary_concatenate(
    node: SemanticNode,
    bound_inputs: Mapping[str, BoundInput],
    fusion_plan: FusionPlan | None,
) -> dict[str, JsonValue]:
    """Oracle-only temporary fusion. Not FusionStrategy.VALIDATED_SLM."""
    if fusion_plan is not None:
        order: Sequence[str] = fusion_plan.ordered_slots
    else:
        order = tuple(node.required_inputs)

    parts: list[str] = []
    for slot_name in order:
        if slot_name not in bound_inputs:
            raise GraphExecutionError(
                f"FUSE node {node.node_id} missing bound input {slot_name}"
            )
        value = bound_inputs[slot_name].value
        parts.append(_jsonable_to_text(value))

    response_slot = next(iter(node.produced_outputs))
    return {response_slot: " ".join(part for part in parts if part).strip()}


def _derive_final_response(
    graph: ExecutionGraph,
    results: Mapping[str, TierResult],
) -> str:
    fuse_nodes = [node for node in graph.nodes if node.operator is OperatorType.FUSE]
    if fuse_nodes:
        fuse = fuse_nodes[0]
        fuse_result = results[fuse.node_id]
        response_slot = next(iter(fuse.produced_outputs))
        return _jsonable_to_text(fuse_result.outputs[response_slot])

    answer_nodes = [
        node
        for node in graph.nodes
        if node.semantic_type is not NodeSemanticType.CONTROL
    ]
    answer_ids = {node.node_id for node in answer_nodes}
    sources = {
        edge.source_node_id
        for edge in graph.edges
        if edge.source_node_id in answer_ids and edge.target_node_id in answer_ids
    }
    sinks = [node for node in answer_nodes if node.node_id not in sources]
    if len(sinks) != 1:
        raise GraphExecutionError(
            "cannot derive final response without FUSE or a single answer sink"
        )
    sink = sinks[0]
    sink_result = results[sink.node_id]
    output_slot = next(iter(sink.produced_outputs))
    return _jsonable_to_text(sink_result.outputs[output_slot])


def _validate_produced_outputs(
    node: SemanticNode,
    outputs: Mapping[str, JsonValue],
) -> None:
    expected = set(node.produced_outputs)
    actual = set(outputs)
    if actual != expected:
        raise GraphExecutionError(
            f"node {node.node_id} outputs mismatch: "
            f"expected={sorted(expected)}, got={sorted(actual)}"
        )
    for name, value in outputs.items():
        if name.strip() == "":
            raise GraphExecutionError("blank output slot name")
        if _looks_like_raw_response_wrapper(value):
            raise GraphExecutionError(
                f"node {node.node_id} output {name} looks like raw model "
                "response text rather than a typed slot value"
            )


def _looks_like_raw_response_wrapper(value: JsonValue) -> bool:
    # Structural guard only: disallow smuggling an undeclared envelope that
    # packages raw model prose under reserved keys.
    if isinstance(value, Mapping):
        forbidden = {"raw_response", "raw_model_text", "full_edge_response"}
        return any(key in value for key in forbidden)
    return False


def _evidence_for_outputs(
    graph: ExecutionGraph,
    node: SemanticNode,
    outputs: Mapping[str, JsonValue],
) -> tuple[EvidenceItem, ...]:
    items: list[EvidenceItem] = []
    for slot_name, value in outputs.items():
        items.append(
            EvidenceItem(
                evidence_id=_new_id("evidence"),
                graph_id=graph.graph_id,
                node_id=node.node_id,
                slot_name=slot_name,
                slot_type=node.produced_outputs[slot_name],
                tier=node.tier,
                value=value,
                source=node.operator.value,
            )
        )
    return tuple(items)


def _build_slot_json_prompt(
    node: SemanticNode,
    bound_inputs: Mapping[str, BoundInput],
    transfer: FogTransferRecord | None,
) -> str:
    schema = {
        slot_name: slot_type.value
        for slot_name, slot_type in node.produced_outputs.items()
    }
    payload: dict[str, Any] = {
        "task": node.task,
        "operator": node.operator.value,
        "required_outputs": schema,
    }
    if node.tier is Tier.FOG and transfer is not None:
        payload["resolved_references"] = transfer.transferred_slots
        # Non-Edge→Fog deps (e.g. Fog→Fog) remain available as typed inputs.
        other = {
            name: bound.value
            for name, bound in bound_inputs.items()
            if not (
                bound.source_tier is Tier.EDGE and bound.target_tier is Tier.FOG
            )
        }
        if other:
            payload["typed_inputs"] = other
    elif bound_inputs:
        payload["typed_inputs"] = {
            name: bound.value for name, bound in bound_inputs.items()
        }

    return (
        "Return a JSON object whose keys are exactly the required output "
        "slot names and whose values are the typed slot values.\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )


def _compose_fog_context(
    env_context: str,
    transfer: FogTransferRecord | None,
    bound_inputs: Mapping[str, BoundInput],
) -> str:
    sections: list[str] = []
    if env_context.strip():
        sections.append(env_context.strip())
    if transfer is not None and transfer.transferred_slots:
        sections.append(
            "resolved_references="
            + json.dumps(transfer.transferred_slots, ensure_ascii=True)
        )
    other = {
        name: bound.value
        for name, bound in bound_inputs.items()
        if not (bound.source_tier is Tier.EDGE and bound.target_tier is Tier.FOG)
    }
    if other:
        sections.append("typed_inputs=" + json.dumps(other, ensure_ascii=True))
    return "\n".join(sections)


def _parse_slot_json(raw: str, node: SemanticNode) -> dict[str, JsonValue]:
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GraphExecutionError(
            f"node {node.node_id} did not return valid JSON slot outputs"
        ) from exc
    if not isinstance(data, dict):
        raise GraphExecutionError(
            f"node {node.node_id} JSON output must be an object"
        )
    return data


async def _call_generate(
    client: Any,
    query: str,
    *,
    context: str,
    image_b64: str | None,
) -> str:
    if hasattr(client, "generate_async"):
        result = client.generate_async(query, context=context, image_b64=image_b64)
        if inspect.isawaitable(result):
            return await result
        return result
    if hasattr(client, "generate"):
        result = client.generate(query, context=context, image_b64=image_b64)
        if inspect.isawaitable(result):
            return await result
        return await asyncio.to_thread(
            lambda: client.generate(query, context=context, image_b64=image_b64)
        )
    raise GraphExecutionError("client must provide generate_async or generate")


def _jsonable_to_text(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


__all__ = [
    "PHASE3_TEMPORARY_FUSION_METHOD",
    "BoundInput",
    "FogTransferRecord",
    "GraphExecutionError",
    "GraphExecutionResult",
    "GraphExecutor",
    "NodeRunner",
]
