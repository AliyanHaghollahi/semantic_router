#!/usr/bin/env python3
"""End-to-end learned TierGraph pilot: planner -> predicted graph -> physical execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from edge.fusion import ResponseFuser
from edge.model import EdgeModelClient, FogModelClient
from tiergraph.planner.stage_a_v2_split import regenerate_stage_a_v2_split_report
from tiergraph.pilot.learned_planner_pilot import (
    DEFAULT_CHECKPOINT_PATH,
    build_decode_failure_record,
    build_learned_deployment_locations,
    describe_learned_graph_source,
    load_planner_for_physical_pilot,
    predict_execution_graph,
)
from tiergraph.pilot.physical_execution import (
    InstrumentedClient,
    execute_physical_trial,
    warmup_client,
)
from tiergraph.pilot.physical_harness import (
    PILOT_BUCKETS,
    PI_EDGE_MODEL,
    PI_EDGE_URL,
    PI_FOG_MODEL,
    PI_FOG_URL,
    SMOKE_BUCKETS,
    PilotQuerySpec,
    build_markdown_report,
    build_pi_topology,
    build_pilot_query_set,
    compute_latency_summary,
    describe_planner_checkpoint,
    run_preflight_checks,
    verify_ollama_endpoint,
    write_manual_review_csv,
    write_pilot_queries,
)

DEFAULT_OUTPUT = ROOT / "artifacts" / "end_to_end_tiergraph_pilot"
EDGE_URL = PI_EDGE_URL
FOG_URL = PI_FOG_URL
EDGE_MODEL = PI_EDGE_MODEL
FOG_MODEL = PI_FOG_MODEL


async def execute_learned_trial(
    *,
    spec: PilotQuerySpec,
    model: Any,
    edge_wrapped: InstrumentedClient,
    fog_wrapped: InstrumentedClient,
    template_fuser: ResponseFuser,
    edge_slm_fuser: ResponseFuser,
    checkpoint_path: Path,
) -> dict[str, Any]:
    deployment = build_learned_deployment_locations(checkpoint_path=str(checkpoint_path))
    outcome = predict_execution_graph(
        model,
        query=spec.query,
        graph_id=f"pred::{spec.query_id}",
    )
    evaluation_labels = {
        "gold_example_id": spec.example_id,
        "gold_execution_mode_expected": spec.execution_mode_expected,
    }
    if not outcome.predicted_graph_valid or outcome.graph is None:
        record = build_decode_failure_record(
            spec=spec,
            outcome=outcome,
            deployment_locations=deployment,
        )
        record["evaluation_labels"] = evaluation_labels
        return record

    return await execute_physical_trial(
        spec=spec,
        graph=outcome.graph,
        fusion_plan=None,
        edge_wrapped=edge_wrapped,
        fog_wrapped=fog_wrapped,
        template_fuser=template_fuser,
        edge_slm_fuser=edge_slm_fuser,
        graph_source=describe_learned_graph_source()["graph_source"],
        planner_runs=True,
        planner_latency_ms=outcome.planner_latency_ms,
        deployment_locations=deployment,
        evaluation_labels=evaluation_labels,
        predicted_graph_valid=True,
        predicted_decode_failure_reason=None,
        predicted_decode_failure_bucket=None,
        predicted_execution_mode=outcome.predicted_execution_mode,
    )


async def run_learned_pilot(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    smoke: bool,
    skip_endpoint_check: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    topology = build_pi_topology()
    graph_source = describe_learned_graph_source()
    planner_checkpoint = describe_planner_checkpoint(ROOT)
    deployment_locations = build_learned_deployment_locations(
        checkpoint_path=str(checkpoint_path),
    )

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

    model = None
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"planner checkpoint not found: {checkpoint_path}")
    model = load_planner_for_physical_pilot(checkpoint_path, device="cpu")

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
        "edge": await warmup_client(edge_wrapped, "edge"),
        "fog": await warmup_client(fog_wrapped, "fog"),
    }

    split = regenerate_stage_a_v2_split_report()
    if smoke:
        pilot_specs: list[PilotQuerySpec] = []
        for bucket in SMOKE_BUCKETS:
            pilot_specs.extend(
                build_pilot_query_set(split.dev, per_bucket=1, buckets=(bucket,))
            )
        run_label = "smoke"
    else:
        pilot_specs = build_pilot_query_set(split.dev, per_bucket=3)
        run_label = "full"

    write_pilot_queries(output_dir / "pilot_queries.json", pilot_specs)

    template_fuser = ResponseFuser(edge_client=edge_base, use_llm_fusion=False)
    edge_slm_fuser = ResponseFuser(edge_client=edge_base, use_llm_fusion=True)

    trial_records: list[dict[str, Any]] = []
    for spec in pilot_specs:
        record = await execute_learned_trial(
            spec=spec,
            model=model,
            edge_wrapped=edge_wrapped,
            fog_wrapped=fog_wrapped,
            template_fuser=template_fuser,
            edge_slm_fuser=edge_slm_fuser,
            checkpoint_path=checkpoint_path,
        )
        trial_records.append(record)
        trial_path = output_dir / "trials" / f"{spec.query_id}.json"
        trial_path.parent.mkdir(parents=True, exist_ok=True)
        trial_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    latency_summary = compute_latency_summary(trial_records) if trial_records else {}
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
        "pilot_mode": "learned_h1_h7_predicted_graph",
        "research_integrity": {
            "planner_executed_on_pi_cpu": True,
            "predicted_graph_not_gold": True,
            "gold_fusion_plan_not_used": True,
            "edge_physical_execution_on_pi": True,
            "fog_physical_execution_on_laptop": True,
            "reverse_ssh_fog_transport": True,
            "not_publication_scale": True,
            "pilot_query_source": "v2_dev_split_labels_only",
            "tiergraph_concatenate_is_phase3_placeholder": True,
        },
        "deployment_locations": deployment_locations,
        "graph_source": graph_source,
        "planner_checkpoint": {
            **planner_checkpoint,
            "checkpoint_used": str(checkpoint_path),
            "encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "max_length": 128,
            "load_strict": True,
            "device": "cpu",
        },
        "topology": topology,
        "endpoint_checks": endpoint_checks,
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

    report_json = output_dir / "end_to_end_tiergraph_pilot_report.json"
    report_json.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_md = build_markdown_report(
        topology=topology,
        warmup=warmup,
        cold_start_observations={
            "edge_note": "n/a for learned-pilot pre-run",
            "fog_note": "n/a for learned-pilot pre-run",
        },
        pilot_design=report_payload["pilot_design"],
        latency_summary=latency_summary,
        representative_examples=representative,
        failures=failures,
    )
    (output_dir / "end_to_end_tiergraph_pilot_report.md").write_text(
        report_md + "\n",
        encoding="utf-8",
    )
    write_manual_review_csv(output_dir / "fusion_manual_review.csv", trial_records)
    return report_payload


async def run_preflight(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    skip_endpoint_check: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = await run_preflight_checks(
        repo_root=ROOT,
        skip_endpoint_check=skip_endpoint_check,
    )
    result["pilot_mode"] = "learned_h1_h7_predicted_graph"
    result["graph_source"] = describe_learned_graph_source()
    result["deployment_locations"] = build_learned_deployment_locations(
        checkpoint_path=str(checkpoint_path),
    )
    result["planner_checkpoint"]["checkpoint_used"] = str(checkpoint_path)
    result["planner_checkpoint"]["required_for_this_pilot"] = True
    result["planner_checkpoint"]["exists_on_this_device"] = checkpoint_path.is_file()
    out_path = output_dir / "end_to_end_tiergraph_preflight.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["preflight_report"] = str(out_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Learned planner checkpoint (model_head_state_dict, strict load).",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--skip-endpoint-check", action="store_true")
    args = parser.parse_args(argv)

    if args.preflight:
        result = asyncio.run(
            run_preflight(
                output_dir=args.output_dir,
                checkpoint_path=args.checkpoint,
                skip_endpoint_check=args.skip_endpoint_check,
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ready_for_inference") else 1

    if args.full and args.smoke:
        parser.error("Use either --smoke or --full, not both.")
    smoke = not args.full
    report = asyncio.run(
        run_learned_pilot(
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            smoke=smoke,
            skip_endpoint_check=args.skip_endpoint_check,
        )
    )
    print(
        json.dumps(
            {
                "run_label": report["run_label"],
                "pilot_mode": report["pilot_mode"],
                "n_queries": report["pilot_design"]["n_queries"],
                "successes": sum(1 for t in report["trials"] if t.get("success")),
                "failures": len(report["failures"]),
            },
            indent=2,
        )
    )
    print(f"report: {args.output_dir / 'end_to_end_tiergraph_pilot_report.json'}")
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
