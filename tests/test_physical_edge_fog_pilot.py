"""Unit tests for physical Edge–Fog pilot harness helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tiergraph.pilot.physical_harness import (
    PI_EDGE_URL,
    PI_FOG_URL,
    PILOT_BUCKETS,
    build_deployment_locations,
    build_pi_topology,
    build_pilot_query_set,
    coerce_slot_json,
    compute_latency_summary,
    describe_graph_source,
    extract_json_payload,
    fog_contacted_from_calls,
    fusion_route_for_bucket,
    personal_fog_isolation_ok,
    validate_pilot_deployment_config,
    verify_ollama_endpoint,
    write_pilot_queries,
)


def _example(example_id: str, bucket: str, query: str):
    graph = SimpleNamespace(execution_mode=lambda: "parallel")
    return SimpleNamespace(
        example_id=example_id,
        query=query,
        metadata={"final_bucket": bucket},
        graph=graph,
    )


def test_build_pilot_query_set_deterministic():
    dev = []
    for bucket in PILOT_BUCKETS:
        for index in range(5):
            dev.append(_example(f"sa_{bucket[:2]}_{index:03d}", bucket, f"q {bucket} {index}"))
    selected = build_pilot_query_set(dev, per_bucket=3)
    assert len(selected) == 15
    personal = [item for item in selected if item.bucket == "Personal"]
    assert [item.example_id for item in personal] == ["sa_Pe_000", "sa_Pe_001", "sa_Pe_002"]


def test_fusion_route_for_bucket():
    assert fusion_route_for_bucket("MIXED_PARALLEL", "parallel") == "mixed_parallel"
    assert fusion_route_for_bucket("MIXED_SEQUENTIAL", "sequential") == "mixed_sequential"
    assert fusion_route_for_bucket("Personal", "parallel") == "edge_only"


def test_compute_latency_summary():
    records = [
        {
            "bucket": "MIXED_PARALLEL",
            "success": True,
            "latencies": {
                "edge_inference_ms": 100.0,
                "fog_inference_ms": 200.0,
                "graph_executor_total_ms": 250.0,
                "end_to_end_ms": 300.0,
            },
            "fusion": {
                "concatenate": {"latency_ms": 0.0},
                "template": {"latency_ms": 1.0},
                "edge_slm": {"latency_ms": 50.0},
            },
        },
        {
            "bucket": "Personal",
            "success": False,
            "latencies": {"end_to_end_ms": 10.0},
        },
    ]
    summary = compute_latency_summary(records)
    assert summary["overall"]["n_trials"] == 2
    assert summary["overall"]["success_rate"] == 0.5
    assert summary["overall"]["edge_inference_ms"]["n"] == 1


def test_verify_ollama_endpoint():
    async def _run() -> None:
        payload = {"models": [{"name": "gemma3:4b"}]}
        mock_response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: payload,
        )
        with patch("tiergraph.pilot.physical_harness.httpx.AsyncClient") as client_cls:
            client = AsyncMock()
            client.__aenter__.return_value = client
            client.get = AsyncMock(return_value=mock_response)
            client_cls.return_value = client
            result = await verify_ollama_endpoint(
                "http://127.0.0.1:11436",
                expected_model="gemma3:4b",
            )
        assert result["reachable"] is True
        assert result["expected_model_present"] is True

    import asyncio

    asyncio.run(_run())


def test_write_pilot_queries(tmp_path: Path):
    dev = [_example("sa_001", "Personal", "What is my gate?")]
    selected = build_pilot_query_set(dev, per_bucket=1, buckets=("Personal",))
    path = tmp_path / "pilot_queries.json"
    write_pilot_queries(path, selected)
    assert path.is_file()
    assert "dev" in path.read_text(encoding="utf-8")


def test_extract_json_payload_from_markdown_fence():
    raw = '```json\n{"this_store_scene": "quiet"}\n```'
    assert extract_json_payload(raw) == '{"this_store_scene": "quiet"}'


def test_coerce_slot_json_nested_required_outputs():
    prompt = (
        'User query: test\n{"task": "t", "operator": "RETRIEVE_PERSONAL", '
        '"required_outputs": {"personal_fact": "PERSONAL_FACT"}}\n\n'
        "Respond with ONLY a JSON object."
    )
    raw = (
        '{"task": "t", "operator": "RETRIEVE_PERSONAL", '
        '"required_outputs": {"personal_fact": "yes, covered"}}'
    )
    coerced = json.loads(coerce_slot_json(raw, prompt))
    assert coerced == {"personal_fact": "yes, covered"}


def test_pi_topology_endpoints():
    topology = build_pi_topology()
    assert topology["edge"]["base_url"] == PI_EDGE_URL
    assert topology["fog"]["base_url"] == PI_FOG_URL
    assert validate_pilot_deployment_config(topology) == []


def test_deployment_locations_and_graph_source():
    locations = build_deployment_locations()
    assert locations["orchestrator_device"] == "Raspberry Pi 5"
    assert locations["graph_source"] == "gold_preconstructed_dev_split"
    graph_source = describe_graph_source()
    assert graph_source["uses_learned_planner"] is False


def test_personal_fog_isolation_helpers():
    assert fog_contacted_from_calls(0) is False
    assert fog_contacted_from_calls(1) is True
    assert personal_fog_isolation_ok("Personal", 0) is True
    assert personal_fog_isolation_ok("Personal", 1) is False
    assert personal_fog_isolation_ok("Environmental", 1) is True
