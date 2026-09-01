#!/usr/bin/env python3
"""Physical Edge–Fog pilot evaluation harness (real Ollama, no simulation)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from edge.fusion import ResponseFuser
from edge.model import EdgeModelClient, FogModelClient
from tiergraph import GraphExecutor
from tiergraph.enums import Tier
from tiergraph.executor import GraphExecutionError
from tiergraph.fusion import FusionStrategy
from tiergraph.planner.stage_a_v2_split import regenerate_stage_a_v2_split_report
from tiergraph.pilot.physical_harness import (
    EDGE_PHYSICAL,
    FOG_PHYSICAL,
    ORCHESTRATOR_DEVICE,
    PI_EDGE_MODEL,
    PI_EDGE_URL,
    PI_FOG_MODEL,
    PI_FOG_URL,
    PILOT_BUCKETS,
    SMOKE_BUCKETS,
    PilotQuerySpec,
    augment_slot_prompt,
    build_deployment_locations,
    build_markdown_report,
    build_pi_topology,
    build_pilot_query_set,
    coerce_slot_json,
    compute_latency_summary,
    describe_graph_source,
    describe_planner_checkpoint,
    extract_json_payload,
    fog_contacted_from_calls,
    fusion_route_for_bucket,
    node_device_map,
    personal_fog_isolation_ok,
    run_preflight_checks,
    tier_text_from_results,
    verify_ollama_endpoint,
    write_manual_review_csv,
    write_pilot_queries,
)

DEFAULT_OUTPUT = ROOT / "artifacts" / "edge_fog_pilot"
EDGE_URL = PI_EDGE_URL
FOG_URL = PI_FOG_URL
EDGE_MODEL = PI_EDGE_MODEL
FOG_MODEL = PI_FOG_MODEL
WARMUP_PROMPT = "Reply with exactly one word: ready."

COLD_START_OBSERVATIONS = {
    "edge_note": (
        "Manual cold-start observation only (not a steady-state measurement): "
        "Edge gemma3:4b ~26.9 s during initial system bring-up."
    ),
    "fog_note": (
        "Manual cold-start observation only (not a steady-state measurement): "
        "Fog gemma3:12b ~72.1 s during initial system bring-up."
    ),
    "label": "cold_start_system_bring_up_only",
}


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


def _examples_by_id() -> dict[str, Any]:
    split = regenerate_stage_a_v2_split_report()
    return {example.example_id: example for example in split.dev}


def _fusion_plan_for_execution(example: Any) -> Any | None:
    plan = example.fusion_plan
    if plan is None:
        return None
    if plan.strategy is FusionStrategy.CONCATENATE:
        return plan
    return None


def _sum_tier_latency(graph: Any, result: Any, tier: Tier) -> float | None:
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


def _wave_concurrency_evidence(graph: Any, result: Any) -> dict[str, Any]:
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


async def _warmup_client(client: Any, label: str) -> dict[str, Any]:
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


async def execute_trial(
    *,
    spec: PilotQuerySpec,
    example: Any,
    edge_wrapped: InstrumentedClient,
    fog_wrapped: InstrumentedClient,
    template_fuser: ResponseFuser,
    edge_slm_fuser: ResponseFuser,
) -> dict[str, Any]:
    edge_wrapped.calls.clear()
    fog_wrapped.calls.clear()
    edge_wrapped.original_query = spec.query
    fog_wrapped.original_query = spec.query

    graph = example.graph
    deployment = build_deployment_locations()
    graph_source = describe_graph_source()
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
        "graph_source": graph_source["graph_source"],
        "planner_latency_ms": None,
        "planner_runs": graph_source["uses_learned_planner"],
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
    try:
        graph_t0 = time.perf_counter()
        result = await executor.execute(
            graph,
            fusion_plan=_fusion_plan_for_execution(example),
        )
        graph_ms = (time.perf_counter() - graph_t0) * 1000
        record["success"] = True
        record["failure_reason"] = None
    except (GraphExecutionError, RuntimeError, Exception) as exc:
        record["success"] = False
        record["failure_reason"] = str(exc)
        record["latencies"] = {
            "graph_executor_total_ms": (time.perf_counter() - graph_t0) * 1000,
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
            "wave_concurrency": _wave_concurrency_evidence(graph, result),
            "edge_response": edge_text,
            "fog_response": fog_text,
            "graph_executor_fusion_method": result.fusion_method,
            "final_response": result.final_response,
            "edge_calls": [call.__dict__ for call in edge_wrapped.calls],
            "fog_calls": [call.__dict__ for call in fog_wrapped.calls],
            "fog_contacted": fog_contacted,
            "personal_fog_isolation_ok": personal_ok,
            "latencies": {
                "planner_latency_ms": None,
                "edge_inference_latency_ms": _sum_tier_latency(graph, result, Tier.EDGE),
                "fog_inference_latency_ms": _sum_tier_latency(graph, result, Tier.FOG),
                "fusion_latency_ms": None,
                "graph_executor_total_ms": result.total_latency_ms,
                "measured_graph_executor_wall_ms": graph_ms,
                "end_to_end_latency_ms": None,
                # Back-compat aliases
                "edge_inference_ms": _sum_tier_latency(graph, result, Tier.EDGE),
                "fog_inference_ms": _sum_tier_latency(graph, result, Tier.FOG),
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
            "edge_model": f"{EDGE_PHYSICAL} {EDGE_MODEL}",
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


async def run_pilot(
    *,
    output_dir: Path,
    smoke: bool,
    skip_endpoint_check: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    topology = build_pi_topology()
    deployment_locations = build_deployment_locations()
    graph_source = describe_graph_source()
    planner_checkpoint = describe_planner_checkpoint(ROOT)

    endpoint_checks: dict[str, Any] = {}
    if not skip_endpoint_check:
        endpoint_checks["edge"] = await verify_ollama_endpoint(
            EDGE_URL,
            expected_model=EDGE_MODEL,
        )
        endpoint_checks["fog"] = await verify_ollama_endpoint(
            FOG_URL,
            expected_model=FOG_MODEL,
        )
        for label, check in endpoint_checks.items():
            if not check.get("expected_model_present"):
                raise RuntimeError(
                    f"{label} endpoint reachable but model {check['expected_model']!r} "
                    f"not found in /api/tags: {check.get('models')}"
                )

    edge_base = EdgeModelClient(
        base_url=EDGE_URL,
        model=EDGE_MODEL,
        timeout=180.0,
        simulation_mode=False,
    )
    fog_base = FogModelClient(
        base_url=FOG_URL,
        model=FOG_MODEL,
        timeout=180.0,
        simulation_mode=False,
    )
    edge_wrapped = InstrumentedClient(inner=edge_base, tier="edge")
    fog_wrapped = InstrumentedClient(inner=fog_base, tier="fog")

    warmup = {
        "edge": await _warmup_client(edge_wrapped, "edge"),
        "fog": await _warmup_client(fog_wrapped, "fog"),
    }

    split = regenerate_stage_a_v2_split_report()
    if smoke:
        smoke_specs: list[PilotQuerySpec] = []
        for bucket in SMOKE_BUCKETS:
            smoke_specs.extend(
                build_pilot_query_set(
                    split.dev,
                    per_bucket=1,
                    buckets=(bucket,),
                )
            )
        pilot_specs = smoke_specs
        run_label = "smoke"
    else:
        pilot_specs = build_pilot_query_set(split.dev, per_bucket=3)
        run_label = "full"

    write_pilot_queries(output_dir / "pilot_queries.json", pilot_specs)
    examples_by_id = {example.example_id: example for example in split.dev}

    template_fuser = ResponseFuser(edge_client=edge_base, use_llm_fusion=False)
    edge_slm_fuser = ResponseFuser(edge_client=edge_base, use_llm_fusion=True)

    trial_records: list[dict[str, Any]] = []
    for spec in pilot_specs:
        example = examples_by_id[spec.example_id]
        record = await execute_trial(
            spec=spec,
            example=example,
            edge_wrapped=edge_wrapped,
            fog_wrapped=fog_wrapped,
            template_fuser=template_fuser,
            edge_slm_fuser=edge_slm_fuser,
        )
        trial_records.append(record)
        (output_dir / "trials" / f"{spec.query_id}.json").parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        (output_dir / "trials" / f"{spec.query_id}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    latency_summary = compute_latency_summary(trial_records)
    failures = [
        {
            "query_id": record["query_id"],
            "failure_reason": record.get("failure_reason"),
        }
        for record in trial_records
        if not record.get("success")
    ]
    representative = [record for record in trial_records if record.get("success")][:3]

    report_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_label": run_label,
        "research_integrity": {
            "orchestrator_device": ORCHESTRATOR_DEVICE,
            "edge_physical_execution_on_pi": True,
            "fog_physical_execution_on_laptop": True,
            "reverse_ssh_fog_transport": True,
            "planner_not_executed_uses_gold_graphs": True,
            "not_publication_scale": True,
            "pilot_query_source": "v2_dev_split_not_test",
            "tiergraph_concatenate_is_phase3_placeholder": True,
            "deprecated_laptop_orchestrated_smoke_note": (
                "Earlier laptop-orchestrated smoke latencies are not Pi deployment results."
            ),
        },
        "deployment_locations": deployment_locations,
        "graph_source": graph_source,
        "planner_checkpoint": planner_checkpoint,
        "topology": topology,
        "endpoint_checks": endpoint_checks,
        "cold_start_observations": COLD_START_OBSERVATIONS,
        "warmup": warmup,
        "pilot_design": {
            "source_split": "dev",
            "n_queries": len(pilot_specs),
            "buckets": list(PILOT_BUCKETS),
            "smoke": smoke,
        },
        "latency_summary": latency_summary,
        "trials": trial_records,
        "failures": failures,
        "representative_examples": representative,
    }

    report_json = output_dir / "physical_edge_fog_pilot_report.json"
    report_json.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_md = build_markdown_report(
        topology=topology,
        warmup=warmup,
        cold_start_observations=COLD_START_OBSERVATIONS,
        pilot_design=report_payload["pilot_design"],
        latency_summary=latency_summary,
        representative_examples=representative,
        failures=failures,
    )
    (output_dir / "physical_edge_fog_pilot_report.md").write_text(
        report_md + "\n",
        encoding="utf-8",
    )
    write_manual_review_csv(output_dir / "fusion_manual_review.csv", trial_records)
    return report_payload


async def run_preflight(
    *,
    output_dir: Path,
    skip_endpoint_check: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = await run_preflight_checks(
        repo_root=ROOT,
        skip_endpoint_check=skip_endpoint_check,
    )
    split = regenerate_stage_a_v2_split_report()
    smoke_specs: list[PilotQuerySpec] = []
    for bucket in SMOKE_BUCKETS:
        smoke_specs.extend(
            build_pilot_query_set(split.dev, per_bucket=1, buckets=(bucket,))
        )
    full_specs = build_pilot_query_set(split.dev, per_bucket=3)
    result["pilot_query_counts"] = {
        "smoke": len(smoke_specs),
        "full": len(full_specs),
    }
    result["smoke_query_ids"] = [spec.query_id for spec in smoke_specs]
    out_path = output_dir / "physical_edge_fog_preflight.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["preflight_report"] = str(out_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one Personal, one Environmental, one Mixed query only.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full 15-query pilot (3 per bucket).",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate Pi topology/endpoints and static config only (no inference).",
    )
    parser.add_argument(
        "--skip-endpoint-check",
        action="store_true",
        help="Skip /api/tags verification (unit tests / offline only).",
    )
    args = parser.parse_args(argv)
    if args.preflight:
        result = asyncio.run(
            run_preflight(
                output_dir=args.output_dir,
                skip_endpoint_check=args.skip_endpoint_check,
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ready_for_inference") else 1
    if args.full and args.smoke:
        parser.error("Use either --smoke or --full, not both.")
    smoke = not args.full
    report = asyncio.run(
        run_pilot(
            output_dir=args.output_dir,
            smoke=smoke,
            skip_endpoint_check=args.skip_endpoint_check,
        )
    )
    print(
        json.dumps(
            {
                "run_label": report["run_label"],
                "n_queries": report["pilot_design"]["n_queries"],
                "successes": sum(1 for t in report["trials"] if t.get("success")),
                "failures": len(report["failures"]),
            },
            indent=2,
        )
    )
    print(f"report: {args.output_dir / 'physical_edge_fog_pilot_report.json'}")
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
