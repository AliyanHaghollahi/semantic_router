"""Shared physical Edge–Fog trial execution for gold and learned-graph pilots."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from edge.fusion import ResponseFuser
from tiergraph import GraphExecutor
from tiergraph.enums import Tier
from tiergraph.executor import GraphExecutionError
from tiergraph.graph import ExecutionGraph
from tiergraph.pilot.physical_harness import (
    EDGE_PHYSICAL,
    ORCHESTRATOR_DEVICE,
    PI_EDGE_MODEL,
    PilotQuerySpec,
    augment_slot_prompt,
    build_deployment_locations,
    coerce_slot_json,
    extract_json_payload,
    fog_contacted_from_calls,
    fusion_route_for_bucket,
    node_device_map,
    personal_fog_isolation_ok,
    tier_text_from_results,
)

WARMUP_PROMPT = "Reply with exactly one word: ready."


@dataclass
class CallRecord:
    tier: str
    latency_ms: float
    prompt: str
    raw_response: str


@dataclass
class InstrumentedClient:
    """Wrap a model client and record every generate_async call."""

    inner: Any
    tier: str
    original_query: str = ""
    calls: list[CallRecord] = field(default_factory=list)

    async def generate_async(
        self,
        query: str,
        context: str = "",
        image_b64: str | None = None,
    ) -> str:
        outbound = augment_slot_prompt(query, self.original_query)
        t0 = time.perf_counter()
        raw = await self.inner.generate_async(
            outbound,
            context=context,
            image_b64=image_b64,
        )
        normalized = coerce_slot_json(
            extract_json_payload(raw) or raw.strip(),
            outbound,
        )
        self.calls.append(
            CallRecord(
                tier=self.tier,
                latency_ms=(time.perf_counter() - t0) * 1000,
                prompt=outbound,
                raw_response=raw,
            )
        )
        return normalized


def sum_tier_latency(graph: ExecutionGraph, result: Any, tier: Tier) -> float | None:
    total = 0.0
    found = False
    for node in graph.nodes:
        if node.tier is not tier:
            continue
        tier_result = result.results.get(node.node_id)
        if tier_result is None:
            continue
        total += float(tier_result.latency_ms)
        found = True
    return total if found else None


def wave_concurrency_evidence(graph: ExecutionGraph, result: Any) -> dict[str, Any]:
    edge_nodes = {node.node_id for node in graph.nodes if node.tier is Tier.EDGE}
    fog_nodes = {node.node_id for node in graph.nodes if node.tier is Tier.FOG}
    parallel_waves: list[dict[str, Any]] = []
    for wave in result.waves:
        edge_in_wave = [node_id for node_id in wave if node_id in edge_nodes]
        fog_in_wave = [node_id for node_id in wave if node_id in fog_nodes]
        if edge_in_wave and fog_in_wave:
            parallel_waves.append(
                {
                    "wave": list(wave),
                    "edge_nodes": edge_in_wave,
                    "fog_nodes": fog_in_wave,
                }
            )
    return {
        "execution_mode": result.execution_mode,
        "waves": [list(wave) for wave in result.waves],
        "parallel_edge_fog_waves": parallel_waves,
        "ordering_note": (
            "Sequential graphs should not place dependent Edge/Fog answer nodes "
            "in the same wave; parallel graphs should."
        ),
    }


async def warmup_client(client: InstrumentedClient, label: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        response = await client.generate_async(WARMUP_PROMPT)
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "label": label,
            "success": True,
            "latency_ms": latency_ms,
            "response_preview": response[:120],
            "excluded_from_steady_state_summary": True,
        }
    except Exception as exc:
        return {
            "label": label,
            "success": False,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "failure_reason": str(exc),
            "excluded_from_steady_state_summary": True,
        }


async def execute_physical_trial(
    *,
    spec: PilotQuerySpec,
    graph: ExecutionGraph,
    fusion_plan: Any | None,
    edge_wrapped: InstrumentedClient,
    fog_wrapped: InstrumentedClient,
    template_fuser: ResponseFuser,
    edge_slm_fuser: ResponseFuser,
    graph_source: str,
    planner_runs: bool,
    planner_latency_ms: float | None,
    deployment_locations: dict[str, str] | None = None,
    evaluation_labels: dict[str, Any] | None = None,
    predicted_graph_valid: bool | None = None,
    predicted_decode_failure_reason: str | None = None,
    predicted_decode_failure_bucket: str | None = None,
    predicted_execution_mode: str | None = None,
) -> dict[str, Any]:
    """Execute one physical trial against the provided graph (gold or predicted)."""
    edge_wrapped.calls.clear()
    fog_wrapped.calls.clear()
    edge_wrapped.original_query = spec.query
    fog_wrapped.original_query = spec.query

    deployment = deployment_locations or build_deployment_locations()
    record: dict[str, Any] = {
        "query_id": spec.query_id,
        "bucket": spec.bucket,
        "query": spec.query,
        "example_id": spec.example_id,
        "source_split": spec.source_split,
        "provenance": spec.provenance,
        "execution_graph_id": graph.graph_id,
        "execution_mode_expected": spec.execution_mode_expected,
        "node_devices": node_device_map(graph),
        "is_mixed": spec.bucket.startswith("MIXED_"),
        "deployment_locations": deployment,
        "graph_source": graph_source,
        "planner_runs": planner_runs,
        "planner_latency_ms": planner_latency_ms,
        "predicted_graph_valid": predicted_graph_valid,
        "predicted_decode_failure_reason": predicted_decode_failure_reason,
        "predicted_decode_failure_bucket": predicted_decode_failure_bucket,
        "predicted_execution_mode": predicted_execution_mode,
        "evaluation_labels": evaluation_labels,
        "communication_overhead_ms": None,
        "communication_overhead_note": "Not separately measured in this pilot.",
    }

    executor = GraphExecutor(
        edge_client=edge_wrapped,
        fog_client=fog_wrapped,
        edge_context_fn=None,
        fog_context_fn=None,
    )

    trial_t0 = time.perf_counter()
    graph_t0 = time.perf_counter()
    try:
        result = await executor.execute(graph, fusion_plan=fusion_plan)
        graph_ms = (time.perf_counter() - graph_t0) * 1000
        record["success"] = True
        record["failure_reason"] = None
    except (GraphExecutionError, RuntimeError, Exception) as exc:
        record["success"] = False
        record["failure_reason"] = str(exc)
        record["latencies"] = {
            "planner_latency_ms": planner_latency_ms,
            "graph_executor_total_ms": (time.perf_counter() - graph_t0) * 1000,
            "end_to_end_latency_ms": (time.perf_counter() - trial_t0) * 1000,
            "end_to_end_ms": (time.perf_counter() - trial_t0) * 1000,
        }
        record["edge_calls"] = [call.__dict__ for call in edge_wrapped.calls]
        record["fog_calls"] = [call.__dict__ for call in fog_wrapped.calls]
        record["fog_contacted"] = fog_contacted_from_calls(len(fog_wrapped.calls))
        record["personal_fog_isolation_ok"] = personal_fog_isolation_ok(
            spec.bucket,
            len(fog_wrapped.calls),
        )
        return record

    edge_text = tier_text_from_results(graph, result.results, Tier.EDGE)
    fog_text = tier_text_from_results(graph, result.results, Tier.FOG)
    fog_contacted = fog_contacted_from_calls(len(fog_wrapped.calls))
    personal_ok = personal_fog_isolation_ok(spec.bucket, len(fog_wrapped.calls))

    record.update(
        {
            "execution_mode": result.execution_mode,
            "wave_concurrency": wave_concurrency_evidence(graph, result),
            "edge_response": edge_text,
            "fog_response": fog_text,
            "graph_executor_fusion_method": result.fusion_method,
            "final_response": result.final_response,
            "edge_calls": [call.__dict__ for call in edge_wrapped.calls],
            "fog_calls": [call.__dict__ for call in fog_wrapped.calls],
            "fog_contacted": fog_contacted,
            "personal_fog_isolation_ok": personal_ok,
            "latencies": {
                "planner_latency_ms": planner_latency_ms,
                "edge_inference_latency_ms": sum_tier_latency(graph, result, Tier.EDGE),
                "fog_inference_latency_ms": sum_tier_latency(graph, result, Tier.FOG),
                "fusion_latency_ms": None,
                "graph_executor_total_ms": result.total_latency_ms,
                "measured_graph_executor_wall_ms": graph_ms,
                "end_to_end_latency_ms": None,
                "edge_inference_ms": sum_tier_latency(graph, result, Tier.EDGE),
                "fog_inference_ms": sum_tier_latency(graph, result, Tier.FOG),
            },
        }
    )

    fusion: dict[str, Any] = {}
    if record["is_mixed"]:
        route = fusion_route_for_bucket(spec.bucket, result.execution_mode)
        fusion["concatenate"] = {
            "method": result.fusion_method or "phase3_temporary_concatenate",
            "label": "temporary_phase3_baseline_not_final_tiergraph_fusion",
            "text": result.final_response,
            "latency_ms": 0.0,
            "device": ORCHESTRATOR_DEVICE,
        }
        template_result = await template_fuser.fuse(
            edge_response=edge_text,
            fog_response=fog_text,
            route=route,
            original_query=spec.query,
        )
        fusion["template"] = {
            "method": template_result.method,
            "text": template_result.text,
            "latency_ms": template_result.latency_ms,
            "device": ORCHESTRATOR_DEVICE,
        }
        edge_slm_result = await edge_slm_fuser.fuse(
            edge_response=edge_text,
            fog_response=fog_text,
            route=route,
            original_query=spec.query,
        )
        fusion["edge_slm"] = {
            "method": edge_slm_result.method,
            "text": edge_slm_result.text,
            "latency_ms": edge_slm_result.latency_ms,
            "device": ORCHESTRATOR_DEVICE,
            "edge_model": f"{EDGE_PHYSICAL} {PI_EDGE_MODEL}",
        }
    record["fusion"] = fusion
    if record["is_mixed"]:
        record["latencies"]["fusion_latency_ms"] = sum(
            float(section.get("latency_ms", 0.0))
            for section in fusion.values()
            if isinstance(section, dict)
        )
    record["latencies"]["end_to_end_latency_ms"] = (
        time.perf_counter() - trial_t0
    ) * 1000
    record["latencies"]["end_to_end_ms"] = record["latencies"]["end_to_end_latency_ms"]
    if record["bucket"] == "Personal" and record.get("fog_contacted"):
        record["success"] = False
        record["failure_reason"] = (
            "Personal query contacted Fog; isolation violation"
        )
    return record
