"""Physical Edge–Fog pilot harness (query selection, stats, reporting helpers)."""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from tiergraph.enums import Tier
from tiergraph.executor import GraphExecutionResult
from tiergraph.graph import ExecutionGraph
from tiergraph.models import TierResult
from tiergraph.planner.annotations import PlannerExample

EDGE_PHYSICAL = "Raspberry Pi 5"
FOG_PHYSICAL = "Windows laptop"
ORCHESTRATOR_DEVICE = EDGE_PHYSICAL

# Pi-orchestrated reverse-SSH topology (script must run ON the Pi).
PI_EDGE_URL = "http://127.0.0.1:11434"
PI_FOG_URL = "http://127.0.0.1:11435"
PI_EDGE_MODEL = "gemma3:4b"
PI_FOG_MODEL = "gemma3:12b"
PI_FOG_REVERSE_SSH = (
    "laptop initiated: ssh -N -R 11435:127.0.0.1:11434 aliyan@<pi-host>"
)

# Legacy laptop-orchestrated smoke (deprecated for deployment metrics).
LEGACY_LAPTOP_EDGE_URL = "http://127.0.0.1:11436"
LEGACY_LAPTOP_FOG_URL = "http://127.0.0.1:11434"

GRAPH_SOURCE_GOLD_DEV = "gold_preconstructed_dev_split"
PLANNER_CHECKPOINT_LAPTOP_PATH = Path("artifacts/planner_v2_run1/best.pt")

PILOT_BUCKETS: tuple[str, ...] = (
    "Personal",
    "Environmental",
    "MIXED_IMPLICIT",
    "MIXED_PARALLEL",
    "MIXED_SEQUENTIAL",
)

SMOKE_BUCKETS: tuple[str, ...] = (
    "Personal",
    "Environmental",
    "MIXED_SEQUENTIAL",
)

MANUAL_REVIEW_FIELDS: tuple[str, ...] = (
    "preserves_edge_information",
    "preserves_fog_information",
    "grounded_in_inputs",
    "dependency_consistent",
    "concise_natural_response",
)


@dataclass(frozen=True, slots=True)
class PilotQuerySpec:
    query_id: str
    bucket: str
    query: str
    example_id: str
    source_split: str
    provenance: str
    execution_mode_expected: str | None


def build_pi_topology() -> dict[str, Any]:
    """Runtime topology when the pilot script executes on the Raspberry Pi."""
    return {
        "orchestrator_device": ORCHESTRATOR_DEVICE,
        "edge": {
            "physical_device": EDGE_PHYSICAL,
            "base_url": PI_EDGE_URL,
            "model": PI_EDGE_MODEL,
            "local_ollama": True,
        },
        "fog": {
            "physical_device": FOG_PHYSICAL,
            "base_url": PI_FOG_URL,
            "model": PI_FOG_MODEL,
            "reverse_ssh_tunnel": PI_FOG_REVERSE_SSH,
            "reachable_from_pi_via": "127.0.0.1:11435",
            "physical_ollama_on_laptop_port": 11434,
        },
        "simulation_mode": False,
        "deprecated_laptop_orchestrated_smoke": {
            "note": (
                "Earlier 3-query smoke used laptop orchestration "
                f"(edge={LEGACY_LAPTOP_EDGE_URL}, fog={LEGACY_LAPTOP_FOG_URL}). "
                "Do not treat those latencies as Pi deployment results."
            ),
        },
    }


def describe_graph_source() -> dict[str, Any]:
    """Current pilot graph provenance (no silent gold substitution)."""
    return {
        "graph_source": GRAPH_SOURCE_GOLD_DEV,
        "uses_learned_planner": False,
        "description": (
            "Pilot loads gold ExecutionGraph objects from frozen v2 DEV "
            "PlannerExample records (example.graph). The learned H1-H7 "
            "planner is NOT invoked during graph selection."
        ),
    }


def describe_planner_checkpoint(repo_root: Path | None = None) -> dict[str, Any]:
    """Checkpoint needed only if switching pilot to predicted graphs."""
    root = repo_root or Path.cwd()
    checkpoint = root / PLANNER_CHECKPOINT_LAPTOP_PATH
    payload: dict[str, Any] = {
        "required_for_predicted_graphs": True,
        "used_in_current_pilot": False,
        "laptop_relative_path": str(PLANNER_CHECKPOINT_LAPTOP_PATH),
        "exists_on_this_device": checkpoint.is_file(),
    }
    if checkpoint.is_file():
        payload["size_bytes"] = checkpoint.stat().st_size
        payload["absolute_path"] = str(checkpoint.resolve())
    payload["transfer_command_example"] = (
        "scp artifacts/planner_v2_run1/best.pt "
        "aliyan@172.20.10.9:~/semantic_router_current/artifacts/planner_v2_run1/best.pt"
    )
    payload["note"] = (
        "Copy checkpoint to Pi only if tomorrow's run should use predicted "
        "graphs instead of gold DEV graphs. Current pilot uses gold graphs."
    )
    return payload


def build_deployment_locations() -> dict[str, str]:
    """Explicit component placement for meeting reports."""
    return {
        "orchestrator_device": ORCHESTRATOR_DEVICE,
        "planner_location": (
            "not executed (gold graphs from frozen DEV PlannerExample.graph)"
        ),
        "graph_executor_location": ORCHESTRATOR_DEVICE,
        "edge_model_location": f"{EDGE_PHYSICAL} local Ollama ({PI_EDGE_URL}, {PI_EDGE_MODEL})",
        "fog_model_location": (
            f"{FOG_PHYSICAL} Ollama via reverse SSH "
            f"({PI_FOG_URL} -> laptop :11434, {PI_FOG_MODEL})"
        ),
        "fusion_location": ORCHESTRATOR_DEVICE,
        "physical_edge_model": f"{EDGE_PHYSICAL} {PI_EDGE_MODEL}",
        "physical_fog_model": f"{FOG_PHYSICAL} {PI_FOG_MODEL}",
        "graph_source": GRAPH_SOURCE_GOLD_DEV,
    }


def fog_contacted_from_calls(fog_call_count: int) -> bool:
    return fog_call_count > 0


def personal_fog_isolation_ok(bucket: str, fog_call_count: int) -> bool:
    if bucket != "Personal":
        return True
    return fog_call_count == 0


def validate_pilot_deployment_config(topology: Mapping[str, Any]) -> list[str]:
    """Static preflight validation errors (empty list == OK)."""
    errors: list[str] = []
    edge = topology.get("edge", {})
    fog = topology.get("fog", {})
    if edge.get("base_url") != PI_EDGE_URL:
        errors.append(f"edge base_url must be {PI_EDGE_URL} for Pi orchestration")
    if fog.get("base_url") != PI_FOG_URL:
        errors.append(f"fog base_url must be {PI_FOG_URL} for reverse-SSH topology")
    if edge.get("model") != PI_EDGE_MODEL:
        errors.append(f"edge model must be {PI_EDGE_MODEL}")
    if fog.get("model") != PI_FOG_MODEL:
        errors.append(f"fog model must be {PI_FOG_MODEL}")
    if topology.get("orchestrator_device") != ORCHESTRATOR_DEVICE:
        errors.append(f"orchestrator must be {ORCHESTRATOR_DEVICE}")
    return errors


async def run_preflight_checks(
    *,
    repo_root: Path | None = None,
    skip_endpoint_check: bool = False,
) -> dict[str, Any]:
    """Endpoint + static validation without model inference."""
    topology = build_pi_topology()
    config_errors = validate_pilot_deployment_config(topology)
    endpoint_checks: dict[str, Any] = {}
    if not skip_endpoint_check:
        endpoint_checks["edge"] = await verify_ollama_endpoint(
            PI_EDGE_URL,
            expected_model=PI_EDGE_MODEL,
        )
        endpoint_checks["fog"] = await verify_ollama_endpoint(
            PI_FOG_URL,
            expected_model=PI_FOG_MODEL,
        )
        for label, check in endpoint_checks.items():
            if not check.get("expected_model_present"):
                config_errors.append(
                    f"{label} endpoint reachable but model "
                    f"{check.get('expected_model')!r} not in /api/tags"
                )
    return {
        "preflight_only": True,
        "topology": topology,
        "deployment_locations": build_deployment_locations(),
        "graph_source": describe_graph_source(),
        "planner_checkpoint": describe_planner_checkpoint(repo_root),
        "endpoint_checks": endpoint_checks,
        "config_errors": config_errors,
        "ready_for_inference": not config_errors,
    }


def build_pilot_query_set(
    dev_examples: Sequence[PlannerExample],
    *,
    per_bucket: int = 3,
    buckets: Sequence[str] = PILOT_BUCKETS,
    source_split: str = "dev",
) -> list[PilotQuerySpec]:
    """Deterministically select pilot queries from DEV (never publication TEST)."""
    by_bucket: dict[str, list[PlannerExample]] = {bucket: [] for bucket in buckets}
    for example in dev_examples:
        bucket = str(example.metadata["final_bucket"])
        if bucket in by_bucket:
            by_bucket[bucket].append(example)

    selected: list[PilotQuerySpec] = []
    for bucket in buckets:
        items = sorted(by_bucket[bucket], key=lambda ex: ex.example_id)
        if len(items) < per_bucket:
            raise ValueError(
                f"DEV split has only {len(items)} examples for bucket {bucket!r}; "
                f"need {per_bucket}"
            )
        for index, example in enumerate(items[:per_bucket]):
            selected.append(
                PilotQuerySpec(
                    query_id=f"pilot_{bucket.lower()}_{index + 1:02d}",
                    bucket=bucket,
                    query=example.query,
                    example_id=example.example_id,
                    source_split=source_split,
                    provenance=(
                        f"frozen v2 {source_split} split; deterministic pick "
                        f"#{index + 1} by example_id within {bucket}"
                    ),
                    execution_mode_expected=example.graph.execution_mode(),
                )
            )
    return selected


def extract_json_payload(raw: str) -> str | None:
    """Best-effort JSON extraction for physical SLM responses (pilot harness only)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            return None
    return None


def coerce_slot_json(response_text: str, prompt: str) -> str:
    """Pilot-only: lift nested required_outputs values to top-level slot keys."""
    payload = extract_json_payload(response_text) or response_text.strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return response_text
    if not isinstance(data, dict):
        return response_text

    prompt_json_start = prompt.find('{"task"')
    expected_slots: list[str] = []
    if prompt_json_start >= 0:
        prompt_end = prompt.find("}\n\nRespond with ONLY", prompt_json_start)
        if prompt_end < 0:
            prompt_end = prompt.rfind("}")
        try:
            prompt_obj = json.loads(prompt[prompt_json_start : prompt_end + 1])
            required = prompt_obj.get("required_outputs", {})
            if isinstance(required, dict):
                expected_slots = list(required.keys())
        except json.JSONDecodeError:
            expected_slots = []

    if expected_slots and set(data.keys()) == set(expected_slots):
        return json.dumps(data, ensure_ascii=True)

    nested = data.get("required_outputs")
    if isinstance(nested, dict) and expected_slots:
        coerced = {
            slot: nested[slot]
            for slot in expected_slots
            if slot in nested
        }
        if coerced:
            return json.dumps(coerced, ensure_ascii=True)
    return json.dumps(data, ensure_ascii=True)


def augment_slot_prompt(prompt: str, original_query: str) -> str:
    """Pilot-only prompt augmentation; does not modify GraphExecutor."""
    if "required_outputs" not in prompt:
        return prompt
    return (
        f"User query: {original_query}\n\n"
        f"{prompt}\n\n"
        "Respond with ONLY a JSON object. Top-level keys must be exactly the "
        "required output slot names from required_outputs. Values must be real "
        "answers grounded in the user query, not the schema itself."
    )


def fusion_route_for_bucket(bucket: str, execution_mode: str) -> str:
    if bucket == "Personal":
        return "edge_only"
    if bucket == "Environmental":
        return "fog_only"
    if bucket == "MIXED_SEQUENTIAL" or execution_mode == "sequential":
        return "mixed_sequential"
    if bucket == "MIXED_PARALLEL" or execution_mode == "parallel":
        return "mixed_parallel"
    if bucket == "MIXED_IMPLICIT":
        return "mixed_sequential"
    return f"mixed_{execution_mode}"


def tier_text_from_results(
    graph: ExecutionGraph,
    results: Mapping[str, TierResult],
    tier: Tier,
) -> str | None:
    parts: list[str] = []
    for node in graph.nodes:
        if node.tier is not tier:
            continue
        if node.node_id not in results:
            continue
        tier_result = results[node.node_id]
        for value in tier_result.outputs.values():
            if isinstance(value, str):
                parts.append(value.strip())
            else:
                parts.append(json.dumps(value, ensure_ascii=True))
    if not parts:
        return None
    return " ".join(part for part in parts if part).strip()


def node_device_map(graph: ExecutionGraph) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for node in graph.nodes:
        if node.tier is Tier.EDGE:
            mapping[node.node_id] = EDGE_PHYSICAL
        elif node.tier is Tier.FOG:
            mapping[node.node_id] = FOG_PHYSICAL
    return mapping


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _latency_stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "success_rate": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "n": len(values),
        "success_rate": 1.0,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
    }


def compute_latency_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate steady-state latency stats from completed pilot trial records."""
    successful = [r for r in records if r.get("success")]
    by_bucket: dict[str, list[Mapping[str, Any]]] = {}
    for record in successful:
        by_bucket.setdefault(str(record["bucket"]), []).append(record)

    def collect(path: tuple[str, ...]) -> list[float]:
        out: list[float] = []
        for record in successful:
            node: Any = record
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            if node is not None:
                out.append(float(node))
        return out

    overall_success_rate = len(successful) / len(records) if records else 0.0

    def bucket_stats(bucket: str, field_paths: dict[str, tuple[str, ...]]) -> dict[str, Any]:
        bucket_records = by_bucket.get(bucket, [])
        bucket_success_rate = (
            len(bucket_records) / len([r for r in records if r["bucket"] == bucket])
            if any(r["bucket"] == bucket for r in records)
            else 0.0
        )
        return {
            "n": len(bucket_records),
            "success_rate": bucket_success_rate,
            **{
                name: _latency_stats(
                    [
                        float(_dig(record, path))
                        for record in bucket_records
                        if _dig(record, path) is not None
                    ]
                )
                for name, path in field_paths.items()
            },
        }

    field_paths = {
        "edge_inference_ms": ("latencies", "edge_inference_ms"),
        "fog_inference_ms": ("latencies", "fog_inference_ms"),
        "graph_executor_total_ms": ("latencies", "graph_executor_total_ms"),
        "end_to_end_ms": ("latencies", "end_to_end_ms"),
        "fusion_concatenate_ms": ("fusion", "concatenate", "latency_ms"),
        "fusion_template_ms": ("fusion", "template", "latency_ms"),
        "fusion_edge_slm_ms": ("fusion", "edge_slm", "latency_ms"),
    }

    summary: dict[str, Any] = {
        "overall": {
            "n_trials": len(records),
            "success_rate": overall_success_rate,
            **{
                name: _latency_stats(collect(path))
                for name, path in field_paths.items()
            },
        },
        "by_bucket": {
            bucket: bucket_stats(bucket, field_paths) for bucket in PILOT_BUCKETS
        },
        "mixed_parallel_vs_sequential": {
            "MIXED_PARALLEL": bucket_stats("MIXED_PARALLEL", field_paths),
            "MIXED_SEQUENTIAL": bucket_stats("MIXED_SEQUENTIAL", field_paths),
        },
    }
    return summary


def _dig(record: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = record
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


async def verify_ollama_endpoint(
    base_url: str,
    *,
    expected_model: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Verify Ollama /api/tags responds and optionally contains expected model."""
    url = base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(f"{url}/api/tags")
        response.raise_for_status()
        payload = response.json()
    models = [item.get("name", "") for item in payload.get("models", [])]
    model_present = (
        any(expected_model in name for name in models)
        if expected_model
        else None
    )
    return {
        "base_url": url,
        "reachable": True,
        "models": models,
        "expected_model": expected_model,
        "expected_model_present": model_present,
    }


def write_pilot_queries(path: Path, queries: Sequence[PilotQuerySpec]) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_split": "dev",
        "note": (
            "Physical-system pilot queries derived from frozen v2 DEV split. "
            "Not publication TEST. Gold planner annotations unchanged."
        ),
        "queries": [
            {
                "query_id": item.query_id,
                "bucket": item.bucket,
                "query": item.query,
                "example_id": item.example_id,
                "source_split": item.source_split,
                "provenance": item.provenance,
                "execution_mode_expected": item.execution_mode_expected,
            }
            for item in queries
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_manual_review_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "bucket",
        "query",
        "edge_response",
        "fog_response",
        "concatenate_output",
        "template_output",
        "edge_slm_output",
        *MANUAL_REVIEW_FIELDS,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if not record.get("is_mixed"):
                continue
            fusion = record.get("fusion", {})
            row = {
                "query_id": record.get("query_id"),
                "bucket": record.get("bucket"),
                "query": record.get("query"),
                "edge_response": record.get("edge_response"),
                "fog_response": record.get("fog_response"),
                "concatenate_output": _dig(fusion, ("concatenate", "text")),
                "template_output": _dig(fusion, ("template", "text")),
                "edge_slm_output": _dig(fusion, ("edge_slm", "text")),
            }
            for field in MANUAL_REVIEW_FIELDS:
                row[field] = ""
            writer.writerow(row)


def build_markdown_report(
    *,
    topology: Mapping[str, Any],
    warmup: Mapping[str, Any],
    cold_start_observations: Mapping[str, Any],
    pilot_design: Mapping[str, Any],
    latency_summary: Mapping[str, Any],
    representative_examples: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Physical Edge–Fog Pilot Report",
        "",
        "## Research integrity",
        "",
        "- Raspberry Pi orchestrates GraphExecutor and fusion locally.",
        "- Edge inference runs on Pi local Ollama (gemma3:4b).",
        "- Fog inference runs on the Windows laptop; Pi reaches it via reverse SSH.",
        "- SSH is transport only; Fog model weights execute on the laptop.",
        "- Current TierGraph `phase3_temporary_concatenate` fusion is a Phase-3 placeholder.",
        "- `ResponseFuser` template and Edge-SLM paths are pilot fusion alternatives.",
        "- This is a **preliminary physical-system pilot**, not publication-scale evaluation.",
        "- The TierGraph planner is **not executed** in the current pilot "
        "(gold DEV graphs are used).",
        "",
        "## 1. Physical topology",
        "",
        f"- Orchestrator: {topology.get('orchestrator_device', ORCHESTRATOR_DEVICE)}",
        f"- Edge endpoint: `{topology['edge']['base_url']}` → {EDGE_PHYSICAL} local Ollama",
        (
            f"- Fog endpoint: `{topology['fog']['base_url']}` → {FOG_PHYSICAL} "
            f"(reverse SSH tunnel from Pi port 11435 to laptop Ollama :11434)"
        ),
        "",
        "## 2. Hardware / models",
        "",
        f"- Edge: {EDGE_PHYSICAL}, model `{topology['edge']['model']}`",
        f"- Fog: {FOG_PHYSICAL}, model `{topology['fog']['model']}`",
        "",
        "## 3. Pilot query design",
        "",
        f"- Source: {pilot_design.get('source_split', 'dev')} split (not publication TEST)",
        f"- Queries in this run: {pilot_design.get('n_queries', 0)}",
        f"- Buckets: {', '.join(PILOT_BUCKETS)}",
        "",
        "## 4. Cold-start observations (manual, not steady-state)",
        "",
        f"- Edge cold-start note: {cold_start_observations.get('edge_note', 'n/a')}",
        f"- Fog cold-start note: {cold_start_observations.get('fog_note', 'n/a')}",
        "",
        "## 5. Warm-up (excluded from steady-state summary)",
        "",
        f"- Edge warm-up latency: {warmup.get('edge', {}).get('latency_ms', 'n/a')} ms",
        f"- Fog warm-up latency: {warmup.get('fog', {}).get('latency_ms', 'n/a')} ms",
        "",
        "## 6. Steady-state Edge / Fog latency",
        "",
        "```json",
        json.dumps(latency_summary.get("overall", {}), indent=2),
        "```",
        "",
        "## 7. Parallel vs sequential (mixed buckets)",
        "",
        "```json",
        json.dumps(latency_summary.get("mixed_parallel_vs_sequential", {}), indent=2),
        "```",
        "",
        "## 8. Fusion comparison",
        "",
        "Methods: (1) `phase3_temporary_concatenate` via GraphExecutor; "
        "(2) ResponseFuser template; (3) ResponseFuser Edge-SLM on Pi.",
        "",
        "## 9. Representative complete examples",
        "",
    ]
    for index, example in enumerate(representative_examples, start=1):
        lines.extend(
            [
                f"### Example {index}: `{example.get('query_id')}`",
                "",
                f"- Query: {example.get('query')}",
                f"- Bucket: {example.get('bucket')}",
                f"- Success: {example.get('success')}",
                f"- Final response: {example.get('final_response', '')[:500]}",
                "",
            ]
        )
    lines.extend(
        [
            "## 10. Failures",
            "",
        ]
    )
    if failures:
        for failure in failures:
            lines.append(
                f"- `{failure.get('query_id')}`: {failure.get('failure_reason')}"
            )
    else:
        lines.append("- None recorded in this run.")
    lines.extend(
        [
            "",
            "## 11. Limitations",
            "",
            "- 15-query pilot (or smoke subset) is not publication-scale.",
            "- GraphExecutor expects typed JSON slot outputs; real SLMs may fail parsing.",
            "- Communication/tunnel overhead was not separately measured unless noted per trial.",
            "- No automated LLM judge; fusion quality requires manual review CSV.",
            "",
            "## 12. Experiments to scale (next two weeks)",
            "",
            "- Expand beyond DEV-derived pilot to held-out queries without tuning on TEST.",
            "- Measure steady-state distributions with more repetitions per query.",
            "- Compare fusion methods with blinded human ratings.",
            "- Profile SSH tunnel overhead with controlled ping vs inference timing.",
            "- Deploy planner inference path separately from execution if on-device planning is required.",
            "",
        ]
    )
    return "\n".join(lines)
