"""Tests for learned end-to-end TierGraph physical pilot helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from tiergraph.enums import QueryType
from tiergraph.planner.annotations import PlannerExample
from tiergraph.planner.decode import PlannerPredictions, PredictedAnchor, PredictedOperation
from tiergraph.pilot.learned_planner_pilot import (
    GRAPH_SOURCE_LEARNED,
    PlannerPredictionOutcome,
    build_decode_failure_record,
    build_learned_deployment_locations,
    describe_learned_graph_source,
    load_planner_for_physical_pilot,
    predict_execution_graph,
)
from tiergraph.pilot.physical_harness import (
    GRAPH_SOURCE_GOLD_DEV,
    PilotQuerySpec,
    describe_graph_source,
)
from tiergraph.pilot.physical_execution import InstrumentedClient, execute_physical_trial
from tests.test_planner_phase5_training import _make_model

ROOT = Path(__file__).resolve().parent.parent
GATE_FIXTURE = ROOT / "tests" / "fixtures" / "planner" / "where_is_my_gate.json"


def _spec() -> PilotQuerySpec:
    return PilotQuerySpec(
        query_id="pilot_mixed_sequential_01",
        bucket="MIXED_SEQUENTIAL",
        query="Where is my gate?",
        example_id="mixed-gate-001",
        source_split="dev",
        provenance="test",
        execution_mode_expected="sequential",
    )


def _gate_example() -> PlannerExample:
    return PlannerExample.model_validate(
        json.loads(GATE_FIXTURE.read_text(encoding="utf-8"))
    )


def _gold_predictions(example: PlannerExample) -> PlannerPredictions:
    operations = tuple(
        PredictedOperation(start=span.start, end=span.end, operator=span.operator)
        for span in example.planner_labels.operation_spans
    )
    id_to_index = {
        span.node_id: index
        for index, span in enumerate(example.planner_labels.operation_spans)
    }
    anchors = tuple(
        PredictedAnchor(
            start=anchor.start,
            end=anchor.end,
            text=anchor.text,
            owner_index=id_to_index[anchor.owner_node_id],
            implicit_resolution=anchor.implicit_resolution,
            normalized_name=anchor.normalized_name,
        )
        for anchor in example.planner_labels.slot_anchors
    )
    deps = set()
    for edge in example.graph.edges:
        if edge.source_node_id in id_to_index and edge.target_node_id in id_to_index:
            deps.add(
                (id_to_index[edge.source_node_id], id_to_index[edge.target_node_id])
            )
    return PlannerPredictions(
        operations=operations,
        anchors=anchors,
        dependency_pairs=frozenset(deps),
        aux_query_type=example.graph.query_type,
    )


class _StubPlannerModel:
    def __init__(self, predictions: PlannerPredictions) -> None:
        self._predictions = predictions
        self._inner = _make_model(hidden_size=8)[0]

    def encode(self, texts):
        return self._inner.encode(texts)

    def predict_structures(self, features, token_views=None):
        _ = features, token_views
        return SimpleNamespace(items=(self._predictions,))


def test_graph_source_paths_are_separate():
    gold = describe_graph_source()
    learned = describe_learned_graph_source()
    assert gold["graph_source"] == GRAPH_SOURCE_GOLD_DEV
    assert learned["graph_source"] == GRAPH_SOURCE_LEARNED
    assert gold["uses_learned_planner"] is False
    assert learned["uses_learned_planner"] is True


def test_predict_execution_graph_decodes_predicted_graph():
    example = _gate_example()
    model = _StubPlannerModel(_gold_predictions(example))
    outcome = predict_execution_graph(
        model,
        query=example.query,
        graph_id="pred::smoke",
    )
    assert outcome.planner_latency_ms >= 0.0
    assert outcome.predicted_graph_valid is True
    assert outcome.graph is not None
    assert outcome.graph.graph_id == "pred::smoke"
    assert outcome.predicted_execution_mode == outcome.graph.execution_mode()


def test_predict_execution_graph_reports_decode_failure_for_empty_ops():
    model = _StubPlannerModel(
        PlannerPredictions(
            operations=(),
            anchors=(),
            aux_query_type=QueryType.MIXED,
        )
    )
    outcome = predict_execution_graph(
        model,
        query="broken",
        graph_id="pred::broken",
    )
    assert outcome.predicted_graph_valid is False
    assert outcome.decode_failure_bucket == "no_operations"
    assert outcome.graph is None


def test_build_decode_failure_record():
    spec = _spec()
    outcome = PlannerPredictionOutcome(
        planner_latency_ms=12.5,
        graph=None,
        predicted_graph_valid=False,
        decode_failure_reason="at least one explicit operation is required",
        decode_failure_bucket="no_operations",
        predicted_execution_mode=None,
    )
    record = build_decode_failure_record(
        spec=spec,
        outcome=outcome,
        deployment_locations=build_learned_deployment_locations(
            checkpoint_path="artifacts/planner_v2_run1/best.pt"
        ),
    )
    assert record["planner_runs"] is True
    assert record["graph_source"] == GRAPH_SOURCE_LEARNED
    assert record["fog_contacted"] is False
    assert record["success"] is False


def test_learned_deployment_locations_include_planner_on_pi():
    locations = build_learned_deployment_locations(
        checkpoint_path="artifacts/planner_v2_run1/best.pt"
    )
    assert locations["graph_source"] == GRAPH_SOURCE_LEARNED
    assert "Raspberry Pi 5" in locations["planner_location"]


def test_load_planner_for_physical_pilot_strict(tmp_path: Path, monkeypatch):
    from tiergraph.planner.train import TrainConfig, save_checkpoint

    model, _ = _make_model(hidden_size=8)
    config = TrainConfig(seed=1, device="cpu", epochs=1, output_dir=str(tmp_path))
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        config=config,
        split_fingerprint="test-fingerprint",
        best_dev_loss=1.0,
        best_dev_metrics=None,
        epoch=1,
    )

    def _fake_build_model(config):
        built, _ = _make_model(hidden_size=8)
        _ = built.encode(["planner warmup"])
        return built

    monkeypatch.setattr(
        "tiergraph.pilot.learned_planner_pilot.build_model",
        _fake_build_model,
    )
    loaded = load_planner_for_physical_pilot(checkpoint, device="cpu")
    assert loaded.hidden_size == 8


def test_learned_planner_checkpoint_metadata(tmp_path: Path):
    from scripts.run_end_to_end_tiergraph_pilot import (
        PI_CHECKPOINT_DEST,
        describe_learned_planner_checkpoint,
    )

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake-checkpoint")
    metadata = describe_learned_planner_checkpoint(
        ROOT,
        checkpoint_path=checkpoint,
    )
    assert metadata["used_in_current_pilot"] is True
    assert metadata["required_for_this_pilot"] is True
    assert metadata["checkpoint_used"] == str(checkpoint)
    assert metadata["exists_on_this_device"] is True
    assert "Learned H1-H7" in metadata["note"]
    assert "gold DEV example.graph" in metadata["note"]
    assert PI_CHECKPOINT_DEST in metadata["transfer_command_example"]
    assert "semantic_router_tiergraph" in metadata["transfer_command_example"]
    assert "semantic_router_current" not in metadata["transfer_command_example"]


def test_preflight_learned_script_offline(tmp_path: Path):
    async def _run() -> None:
        from scripts.run_end_to_end_tiergraph_pilot import run_preflight

        result = await run_preflight(
            output_dir=tmp_path / "preflight",
            checkpoint_path=tmp_path / "missing.pt",
            skip_endpoint_check=True,
        )
        assert result["pilot_mode"] == "learned_h1_h7_predicted_graph"
        assert result["graph_source"]["graph_source"] == GRAPH_SOURCE_LEARNED
        checkpoint_meta = result["planner_checkpoint"]
        assert checkpoint_meta["used_in_current_pilot"] is True
        assert checkpoint_meta["required_for_this_pilot"] is True
        assert "Learned H1-H7" in checkpoint_meta["note"]
        assert "semantic_router_tiergraph" in checkpoint_meta["transfer_command_example"]
        assert "gold graphs" not in checkpoint_meta["note"].lower()

    asyncio.run(_run())


def test_execute_physical_trial_accepts_predicted_graph_metadata():
    async def _run() -> None:
        example = _gate_example()
        graph = example.graph.model_copy(update={"graph_id": "pred::test"})

        class _Edge:
            async def generate_async(self, query, context="", image_b64=None):
                _ = context, image_b64
                return json.dumps({"gate_identifier": "D34"})

        class _Fog:
            async def generate_async(self, query, context="", image_b64=None):
                _ = context, image_b64
                return json.dumps({"gate_location": "Terminal D"})

        from edge.fusion import ResponseFuser

        edge = InstrumentedClient(inner=_Edge(), tier="edge")
        fog = InstrumentedClient(inner=_Fog(), tier="fog")
        template = ResponseFuser(edge_client=_Edge(), use_llm_fusion=False)
        edge_slm = ResponseFuser(edge_client=_Edge(), use_llm_fusion=True)
        record = await execute_physical_trial(
            spec=_spec(),
            graph=graph,
            fusion_plan=None,
            edge_wrapped=edge,
            fog_wrapped=fog,
            template_fuser=template,
            edge_slm_fuser=edge_slm,
            graph_source=GRAPH_SOURCE_LEARNED,
            planner_runs=True,
            planner_latency_ms=9.5,
            deployment_locations=build_learned_deployment_locations(
                checkpoint_path="artifacts/planner_v2_run1/best.pt"
            ),
            predicted_graph_valid=True,
            predicted_execution_mode="sequential",
        )
        assert record["graph_source"] == GRAPH_SOURCE_LEARNED
        assert record["planner_runs"] is True
        assert record["planner_latency_ms"] == 9.5
        assert record["success"] is True

    asyncio.run(_run())
