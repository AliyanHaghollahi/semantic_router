"""Tests for the Phase-3 TierGraph oracle DAG executor."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from edge.pipeline import RoutingPipeline, TierGraphPipelineResult
from edge.session import SessionManager
from tiergraph import (
    PHASE3_TEMPORARY_FUSION_METHOD,
    DependencyEdge,
    ExecutionGraph,
    ExecutionStatus,
    FusionPlan,
    FusionStrategy,
    GraphExecutionError,
    GraphExecutionResult,
    GraphExecutor,
    NodeSemanticType,
    OperatorType,
    QueryType,
    SemanticNode,
    SlotType,
    Tier,
    TierResult,
    TransferPolicy,
)
from tiergraph.executor import (
    BoundInput,
    FogTransferRecord,
    _bind_inputs,
    _build_edge_to_fog_transfer,
    _validate_edge_to_fog_contract,
)


FIXTURE_GATE = (
    Path(__file__).parent / "fixtures" / "planner" / "where_is_my_gate.json"
)


def _node(**overrides) -> SemanticNode:
    values = {
        "node_id": "personal",
        "semantic_type": NodeSemanticType.PERSONAL,
        "operator": OperatorType.RETRIEVE_PERSONAL,
        "tier": Tier.EDGE,
        "task": "Retrieve the requested personal fact",
        "required_inputs": {},
        "produced_outputs": {"fact": SlotType.PERSONAL_FACT},
    }
    values.update(overrides)
    return SemanticNode(**values)


def _graph(**overrides) -> ExecutionGraph:
    values = {
        "graph_id": "graph-personal",
        "original_query": "What medication do I take?",
        "query_type": QueryType.PERSONAL,
        "nodes": (_node(),),
        "edges": (),
    }
    values.update(overrides)
    return ExecutionGraph(**values)


def _gate_graph() -> ExecutionGraph:
    q1 = _node(
        node_id="q1",
        semantic_type=NodeSemanticType.PERSONAL,
        operator=OperatorType.RESOLVE_PERSONAL,
        tier=Tier.EDGE,
        task="Resolve the user's gate identifier",
        produced_outputs={"gate_identifier": SlotType.RESOLVED_REFERENCE},
    )
    q2 = _node(
        node_id="q2",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.LOCATE_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Locate the resolved gate",
        required_inputs={"gate_identifier": SlotType.RESOLVED_REFERENCE},
        produced_outputs={"gate_location": SlotType.LOCATION},
    )
    return _graph(
        graph_id="graph-gate",
        original_query="Where is my gate?",
        query_type=QueryType.MIXED,
        nodes=(q1, q2),
        edges=(
            DependencyEdge(
                source_node_id="q1",
                source_slot="gate_identifier",
                target_node_id="q2",
                target_slot="gate_identifier",
                transfer_policy=TransferPolicy.MINIMAL_REFERENCE,
            ),
        ),
    )


def _parallel_with_fuse() -> tuple[ExecutionGraph, FusionPlan]:
    personal = _node(
        node_id="personal",
        produced_outputs={"medication": SlotType.PERSONAL_FACT},
    )
    environmental = _node(
        node_id="environmental",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Identify the nearby pharmacy",
        produced_outputs={"pharmacy": SlotType.ENVIRONMENTAL_FACT},
    )
    fusion = _node(
        node_id="fusion",
        semantic_type=NodeSemanticType.CONTROL,
        operator=OperatorType.FUSE,
        tier=Tier.EDGE,
        task="Fuse the personal and environmental answers",
        required_inputs={
            "personal": SlotType.PERSONAL_FACT,
            "environmental": SlotType.ENVIRONMENTAL_FACT,
        },
        produced_outputs={"response": SlotType.FINAL_RESPONSE},
    )
    graph = _graph(
        graph_id="graph-parallel",
        original_query="What medication do I take and is there a pharmacy nearby?",
        query_type=QueryType.MIXED,
        nodes=(personal, environmental, fusion),
        edges=(
            DependencyEdge(
                source_node_id="personal",
                source_slot="medication",
                target_node_id="fusion",
                target_slot="personal",
            ),
            DependencyEdge(
                source_node_id="environmental",
                source_slot="pharmacy",
                target_node_id="fusion",
                target_slot="environmental",
            ),
        ),
    )
    plan = FusionPlan(
        plan_id="plan-parallel",
        graph_id="graph-parallel",
        fusion_node_id="fusion",
        strategy=FusionStrategy.CONCATENATE,
        required_slots={
            "personal": SlotType.PERSONAL_FACT,
            "environmental": SlotType.ENVIRONMENTAL_FACT,
        },
        ordered_slots=("personal", "environmental"),
        max_sentences=2,
        spoken_style=True,
        instructions="Oracle temporary concatenate only.",
    )
    return graph, plan


def _hybrid_with_fuse() -> ExecutionGraph:
    gate = _gate_graph()
    scene = _node(
        node_id="q3",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.DESCRIBE_ENVIRONMENT,
        tier=Tier.FOG,
        task="Describe the current scene",
        produced_outputs={"scene": SlotType.SCENE_DESCRIPTION},
    )
    fusion = _node(
        node_id="fusion",
        semantic_type=NodeSemanticType.CONTROL,
        operator=OperatorType.FUSE,
        tier=Tier.EDGE,
        task="Temporary oracle fuse",
        required_inputs={
            "gate_location": SlotType.LOCATION,
            "scene": SlotType.SCENE_DESCRIPTION,
        },
        produced_outputs={"response": SlotType.FINAL_RESPONSE},
    )
    return _graph(
        graph_id="graph-hybrid",
        original_query="Where is my gate and what is around me?",
        query_type=QueryType.MIXED,
        nodes=gate.nodes + (scene, fusion),
        edges=gate.edges
        + (
            DependencyEdge(
                source_node_id="q2",
                source_slot="gate_location",
                target_node_id="fusion",
                target_slot="gate_location",
            ),
            DependencyEdge(
                source_node_id="q3",
                source_slot="scene",
                target_node_id="fusion",
                target_slot="scene",
            ),
        ),
    )


def _fog_to_fog_graph() -> ExecutionGraph:
    upstream = _node(
        node_id="up",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Identify landmark",
        produced_outputs={"landmark": SlotType.ENVIRONMENTAL_FACT},
    )
    downstream = _node(
        node_id="down",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.LOCATE_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Locate relative to landmark",
        required_inputs={"landmark": SlotType.ENVIRONMENTAL_FACT},
        produced_outputs={"place": SlotType.LOCATION},
    )
    return _graph(
        graph_id="graph-fog-fog",
        original_query="Where is the exit near that landmark?",
        query_type=QueryType.ENVIRONMENTAL,
        nodes=(upstream, downstream),
        edges=(
            DependencyEdge(
                source_node_id="up",
                source_slot="landmark",
                target_node_id="down",
                target_slot="landmark",
                transfer_policy=TransferPolicy.DIRECT,
            ),
        ),
    )


class RecordingRunner:
    """Async fake runner that records transfers and supports barriers."""

    def __init__(self, outputs_by_node: dict[str, dict]):
        self.outputs_by_node = outputs_by_node
        self.calls: list[tuple[str, dict[str, BoundInput], FogTransferRecord | None]] = []
        self.started: dict[str, asyncio.Event] = {}
        self.release: dict[str, asyncio.Event] = {}
        self.active_count = 0
        self.max_active = 0
        self._lock = asyncio.Lock()

    def arm_barrier(self, node_ids: list[str]) -> None:
        for node_id in node_ids:
            self.started[node_id] = asyncio.Event()
            self.release[node_id] = asyncio.Event()

    async def __call__(self, node, bound_inputs, transfer):
        async with self._lock:
            self.active_count += 1
            self.max_active = max(self.max_active, self.active_count)

        self.calls.append((node.node_id, dict(bound_inputs), transfer))
        if node.node_id in self.started:
            self.started[node.node_id].set()
            await self.release[node.node_id].wait()

        try:
            return self.outputs_by_node[node.node_id]
        finally:
            async with self._lock:
                self.active_count -= 1


def _executor(runner) -> GraphExecutor:
    return GraphExecutor(edge_client=object(), fog_client=object(), node_runner=runner)


def test_single_edge_node():
    asyncio.run(_test_single_edge_node())


async def _test_single_edge_node():
    graph = _graph()
    runner = RecordingRunner({"personal": {"fact": "Lisinopril 10mg"}})
    result = await _executor(runner).execute(graph)

    assert isinstance(result, GraphExecutionResult)
    assert result.waves == (("personal",),)
    assert result.fog_transfers == ()
    assert result.results["personal"].status is ExecutionStatus.SUCCEEDED
    assert result.results["personal"].outputs["fact"] == "Lisinopril 10mg"
    assert result.final_response == "Lisinopril 10mg"
    assert result.fusion_method is None


def test_single_fog_node():
    asyncio.run(_test_single_fog_node())


async def _test_single_fog_node():
    graph = _graph(
        graph_id="graph-env",
        original_query="What does this sign say?",
        query_type=QueryType.ENVIRONMENTAL,
        nodes=(
            _node(
                node_id="env",
                semantic_type=NodeSemanticType.ENVIRONMENTAL,
                operator=OperatorType.DESCRIBE_ENVIRONMENT,
                tier=Tier.FOG,
                task="Read the sign",
                produced_outputs={"scene": SlotType.SCENE_DESCRIPTION},
            ),
        ),
    )
    runner = RecordingRunner({"env": {"scene": "Emergency Exit"}})
    result = await _executor(runner).execute(graph)

    assert result.results["env"].tier is Tier.FOG
    assert result.fog_transfers == ()
    assert result.final_response == "Emergency Exit"
    node_id, bound, transfer = runner.calls[0]
    assert node_id == "env"
    assert bound == {}
    assert transfer is None


def test_parallel_wave_runs_concurrently():
    asyncio.run(_test_parallel_wave_runs_concurrently())


async def _test_parallel_wave_runs_concurrently():
    graph, plan = _parallel_with_fuse()
    runner = RecordingRunner(
        {
            "personal": {"medication": "Lisinopril"},
            "environmental": {"pharmacy": "CVS nearby"},
            # FUSE is handled inside executor, not the runner.
        }
    )
    runner.arm_barrier(["personal", "environmental"])

    async def unlock_when_both_started():
        await runner.started["personal"].wait()
        await runner.started["environmental"].wait()
        assert runner.max_active >= 2
        runner.release["personal"].set()
        runner.release["environmental"].set()

    unlock = asyncio.create_task(unlock_when_both_started())
    result = await _executor(runner).execute(graph, fusion_plan=plan)
    await unlock

    assert result.waves[0] == ("personal", "environmental")
    assert result.execution_mode == "parallel"
    assert runner.max_active >= 2
    assert result.fusion_method == PHASE3_TEMPORARY_FUSION_METHOD
    assert "Lisinopril" in result.final_response
    assert "CVS nearby" in result.final_response


def test_sequential_edge_to_fog_gate():
    asyncio.run(_test_sequential_edge_to_fog_gate())


async def _test_sequential_edge_to_fog_gate():
    graph = _gate_graph()
    runner = RecordingRunner(
        {
            "q1": {"gate_identifier": "D34"},
            "q2": {"gate_location": "Gate D34 is ahead"},
        }
    )
    result = await _executor(runner).execute(graph)

    assert result.waves == (("q1",), ("q2",))
    assert result.execution_mode == "sequential"
    assert runner.calls[0][0] == "q1"
    assert runner.calls[1][0] == "q2"
    assert "gate_identifier" in runner.calls[1][1]
    assert runner.calls[1][1]["gate_identifier"].value == "D34"


def test_resolved_reference_minimal_transfer_payload():
    asyncio.run(_test_resolved_reference_minimal_transfer_payload())


async def _test_resolved_reference_minimal_transfer_payload():
    graph = _gate_graph()
    runner = RecordingRunner(
        {
            "q1": {"gate_identifier": "D34"},
            "q2": {"gate_location": "Terminal D"},
        }
    )
    result = await _executor(runner).execute(graph)

    assert len(result.fog_transfers) == 1
    transfer = result.fog_transfers[0]
    assert transfer.target_node_id == "q2"
    assert transfer.transferred_slots == {"gate_identifier": "D34"}
    # Outbound Fog transfer arg observed by the runner.
    assert runner.calls[1][2] == transfer


def test_hybrid_multi_wave_with_fuse():
    asyncio.run(_test_hybrid_multi_wave_with_fuse())


async def _test_hybrid_multi_wave_with_fuse():
    graph = _hybrid_with_fuse()
    runner = RecordingRunner(
        {
            "q1": {"gate_identifier": "D34"},
            "q2": {"gate_location": "Gate D34 left"},
            "q3": {"scene": "Atrium ahead"},
        }
    )
    result = await _executor(runner).execute(graph)

    assert result.execution_mode == "hybrid"
    assert result.waves[0] == ("q1", "q3") or set(result.waves[0]) == {"q1", "q3"}
    assert "q2" in result.waves[1]
    assert result.fusion_method == PHASE3_TEMPORARY_FUSION_METHOD
    assert "Gate D34 left" in result.final_response
    assert "Atrium ahead" in result.final_response


def test_missing_dependency_result_fails():
    asyncio.run(_test_missing_dependency_result_fails())


async def _test_missing_dependency_result_fails():
    graph = _gate_graph()
    with pytest.raises(GraphExecutionError, match="missing dependency result"):
        _bind_inputs(graph, graph.node_by_id("q2"), results={})


def test_outbound_fog_payload_is_only_resolved_reference():
    asyncio.run(_test_outbound_fog_payload_is_only_resolved_reference())


async def _test_outbound_fog_payload_is_only_resolved_reference():
    graph = _gate_graph()
    full_edge_prose = (
        "Your boarding pass shows Gate D34 for flight UA447 to Dallas at 3:45 PM."
    )

    async def runner(node, bound_inputs, transfer):
        if node.node_id == "q1":
            # Typed slot value only — not the prose narrative.
            return {"gate_identifier": "D34"}
        assert transfer is not None
        assert transfer.transferred_slots == {"gate_identifier": "D34"}
        assert full_edge_prose not in json.dumps(transfer.transferred_slots)
        assert full_edge_prose not in transfer.task
        return {"gate_location": "Walk to D34"}

    result = await _executor(runner).execute(graph)
    assert result.fog_transfers[0].transferred_slots == {"gate_identifier": "D34"}


def test_edge_to_fog_personal_fact_rejected_structurally():
    edge = DependencyEdge(
        source_node_id="q1",
        source_slot="fact",
        target_node_id="q2",
        target_slot="fact",
        transfer_policy=TransferPolicy.MINIMAL_REFERENCE,
    )
    with pytest.raises(GraphExecutionError, match="RESOLVED_REFERENCE"):
        _validate_edge_to_fog_contract(edge, SlotType.PERSONAL_FACT)
    with pytest.raises(GraphExecutionError, match="RESOLVED_REFERENCE"):
        _validate_edge_to_fog_contract(edge, SlotType.PERSONAL_RECORD)


def test_edge_to_fog_rejects_non_minimal_policy():
    edge = DependencyEdge(
        source_node_id="q1",
        source_slot="gate_identifier",
        target_node_id="q2",
        target_slot="gate_identifier",
        transfer_policy=TransferPolicy.DIRECT,
    )
    with pytest.raises(GraphExecutionError, match="MINIMAL_REFERENCE"):
        _validate_edge_to_fog_contract(edge, SlotType.RESOLVED_REFERENCE)


def test_raw_response_envelope_in_outputs_rejected():
    asyncio.run(_test_raw_response_envelope_in_outputs_rejected())


async def _test_raw_response_envelope_in_outputs_rejected():
    graph = _graph()

    async def runner(node, bound_inputs, transfer):
        return {"fact": {"raw_response": "secret personal dump"}}

    with pytest.raises(GraphExecutionError, match="raw model"):
        await _executor(runner).execute(graph)


def test_fog_to_fog_propagates_via_bound_inputs_without_fog_transfer():
    asyncio.run(_test_fog_to_fog_propagates_via_bound_inputs_without_fog_transfer())


async def _test_fog_to_fog_propagates_via_bound_inputs_without_fog_transfer():
    graph = _fog_to_fog_graph()
    runner = RecordingRunner(
        {
            "up": {"landmark": "fountain"},
            "down": {"place": "exit by fountain"},
        }
    )
    result = await _executor(runner).execute(graph)

    assert result.fog_transfers == ()
    down_bound = runner.calls[1][1]
    assert down_bound["landmark"].value == "fountain"
    assert down_bound["landmark"].source_tier is Tier.FOG
    assert runner.calls[1][2] is None


def test_multiple_answer_sinks_without_fuse_fail():
    asyncio.run(_test_multiple_answer_sinks_without_fuse_fail())


async def _test_multiple_answer_sinks_without_fuse_fail():
    personal = _node(node_id="personal")
    environmental = _node(
        node_id="environmental",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Find pharmacy",
        produced_outputs={"pharmacy": SlotType.ENVIRONMENTAL_FACT},
    )
    graph = _graph(
        graph_id="graph-parallel",
        original_query="meds and pharmacy",
        query_type=QueryType.MIXED,
        nodes=(personal, environmental),
        edges=(),
    )
    runner = RecordingRunner(
        {
            "personal": {"fact": "Lisinopril"},
            "environmental": {"pharmacy": "CVS"},
        }
    )
    with pytest.raises(GraphExecutionError, match="explicit FUSE"):
        await _executor(runner).execute(graph)


def test_phase3_fusion_is_temporary_concatenate_not_validated_slm():
    asyncio.run(_test_phase3_fusion_is_temporary_concatenate_not_validated_slm())


async def _test_phase3_fusion_is_temporary_concatenate_not_validated_slm():
    graph, plan = _parallel_with_fuse()
    assert plan.strategy is FusionStrategy.CONCATENATE
    runner = RecordingRunner(
        {
            "personal": {"medication": "Lisinopril"},
            "environmental": {"pharmacy": "CVS"},
        }
    )
    result = await _executor(runner).execute(graph, fusion_plan=plan)
    assert result.fusion_method == PHASE3_TEMPORARY_FUSION_METHOD
    assert result.fusion_method != FusionStrategy.VALIDATED_SLM.value
    assert result.results["fusion"].metadata["fusion_method"] == (
        PHASE3_TEMPORARY_FUSION_METHOD
    )
    assert FusionStrategy.VALIDATED_SLM.value not in result.final_response
    assert result.results["fusion"].metadata.get("fusion_method") != (
        FusionStrategy.VALIDATED_SLM.value
    )


def test_validated_slm_fusion_plan_is_rejected():
    asyncio.run(_test_validated_slm_fusion_plan_is_rejected())


async def _test_validated_slm_fusion_plan_is_rejected():
    graph, plan = _parallel_with_fuse()
    invalid = plan.model_copy(update={"strategy": FusionStrategy.VALIDATED_SLM})
    runner = RecordingRunner(
        {
            "personal": {"medication": "Lisinopril"},
            "environmental": {"pharmacy": "CVS"},
        }
    )
    with pytest.raises(GraphExecutionError, match="VALIDATED_SLM is not implemented"):
        await _executor(runner).execute(graph, fusion_plan=invalid)


def test_unsupported_fusion_strategy_is_rejected():
    asyncio.run(_test_unsupported_fusion_strategy_is_rejected())


async def _test_unsupported_fusion_strategy_is_rejected():
    graph, plan = _parallel_with_fuse()
    invalid = plan.model_copy(update={"strategy": FusionStrategy.TEMPLATE})
    runner = RecordingRunner(
        {
            "personal": {"medication": "Lisinopril"},
            "environmental": {"pharmacy": "CVS"},
        }
    )
    with pytest.raises(GraphExecutionError, match="unsupported FusionPlan.strategy"):
        await _executor(runner).execute(graph, fusion_plan=invalid)


def test_legacy_pipeline_unchanged_when_tiergraph_disabled():
    calls = {"executor": 0}

    class BoomExecutor:
        async def execute(self, *args, **kwargs):
            calls["executor"] += 1
            raise AssertionError("executor must not run")

    class FakeClassifier:
        def predict(self, query):
            return type(
                "C",
                (),
                {
                    "label": "Personal",
                    "confidence": 1.0,
                    "triggered_by": "test",
                },
            )()

    class FakeDispatcher:
        async def dispatch_personal(self, query, context="", image_b64=None):
            return type(
                "D",
                (),
                {
                    "edge_response": "Lisinopril 10mg",
                    "fog_response": None,
                    "route": "edge_only",
                },
            )()

        async def dispatch_environmental(self, *args, **kwargs):
            raise AssertionError("unexpected environmental dispatch")

        async def dispatch_mixed_parallel(self, *args, **kwargs):
            raise AssertionError("unexpected mixed dispatch")

        async def dispatch_mixed_sequential(self, *args, **kwargs):
            raise AssertionError("unexpected mixed dispatch")

    class FakeFuser:
        async def fuse(self, edge_response, fog_response, route, original_query=""):
            return type(
                "F",
                (),
                {
                    "text": edge_response or fog_response or "",
                    "method": "passthrough",
                    "latency_ms": 0.0,
                },
            )()

    class FakeStore:
        def retrieve_context(self, query):
            return ""

        def retrieve(self, query):
            return ""

    class BoomDecomposer:
        def decompose(self, query):
            raise AssertionError("decomposer must not run for Personal")

    class BoomDependency:
        def detect(self, *args, **kwargs):
            raise AssertionError("dependency detector must not run for Personal")

    pipeline = RoutingPipeline(
        classifier=FakeClassifier(),
        decomposer=BoomDecomposer(),
        dependency_detector=BoomDependency(),
        dispatcher=FakeDispatcher(),
        fuser=FakeFuser(),
        edge_store=FakeStore(),
        fog_store=FakeStore(),
        session_manager=SessionManager(buffer_size=3),
        tiergraph_enabled=False,
        graph_executor=BoomExecutor(),
    )

    result = pipeline.process_sync("What is my blood pressure medication?")
    assert calls["executor"] == 0
    assert result.route == "edge_only"
    assert result.final_response == "Lisinopril 10mg"
    assert result.decomposition is None


def test_process_execution_graph_preserves_graph_result():
    asyncio.run(_test_process_execution_graph_preserves_graph_result())


async def _test_process_execution_graph_preserves_graph_result():
    graph = _gate_graph()
    runner = RecordingRunner(
        {
            "q1": {"gate_identifier": "D34"},
            "q2": {"gate_location": "Gate D34 ahead"},
        }
    )
    executor = _executor(runner)
    pipeline = RoutingPipeline(
        classifier=object(),
        decomposer=object(),
        dependency_detector=object(),
        dispatcher=object(),
        fuser=object(),
        edge_store=object(),
        fog_store=object(),
        session_manager=SessionManager(buffer_size=3),
        tiergraph_enabled=True,
        graph_executor=executor,
    )

    wrapped = await pipeline.process_execution_graph(graph)
    assert isinstance(wrapped, TierGraphPipelineResult)
    assert isinstance(wrapped.graph_result, GraphExecutionResult)
    assert wrapped.graph_result.final_response == "Gate D34 ahead"
    assert wrapped.pipeline_result.final_response == "Gate D34 ahead"
    assert wrapped.pipeline_result.decomposition is None
    assert wrapped.pipeline_result.dependency is None
    assert wrapped.graph_result.fog_transfers[0].transferred_slots == {
        "gate_identifier": "D34"
    }


def test_oracle_gate_fixture_executes():
    asyncio.run(_test_oracle_gate_fixture_executes())


async def _test_oracle_gate_fixture_executes():
    example = json.loads(FIXTURE_GATE.read_text(encoding="utf-8"))
    graph = ExecutionGraph.model_validate(example["graph"])
    runner = RecordingRunner(
        {
            "q1": {"gate_identifier": "D34"},
            "q2": {"gate_location": "Terminal D, Gate D34"},
        }
    )
    result = await _executor(runner).execute(graph)
    assert result.graph_id == "mixed-gate-001"
    assert result.fog_transfers[0].transferred_slots == {"gate_identifier": "D34"}
    assert result.final_response == "Terminal D, Gate D34"


def test_build_edge_to_fog_transfer_serializes_only_reference_slots():
    graph = _gate_graph()
    q2 = graph.node_by_id("q2")
    bound = {
        "gate_identifier": BoundInput(
            slot_name="gate_identifier",
            slot_type=SlotType.RESOLVED_REFERENCE,
            value="D34",
            source_node_id="q1",
            source_slot="gate_identifier",
            transfer_policy=TransferPolicy.MINIMAL_REFERENCE,
            source_tier=Tier.EDGE,
            target_tier=Tier.FOG,
        )
    }
    transfer = _build_edge_to_fog_transfer(graph, q2, bound)
    assert transfer is not None
    assert transfer.transferred_slots == {"gate_identifier": "D34"}


def test_failed_predecessor_status_blocks_dependent_binding():
    asyncio.run(_test_failed_predecessor_status_blocks_dependent_binding())


async def _test_failed_predecessor_status_blocks_dependent_binding():
    graph = _gate_graph()
    failed = TierResult(
        result_id="r1",
        graph_id=graph.graph_id,
        node_id="q1",
        tier=Tier.EDGE,
        status=ExecutionStatus.FAILED,
        outputs={},
        latency_ms=1.0,
        error="boom",
    )
    with pytest.raises(GraphExecutionError, match="did not succeed"):
        _bind_inputs(graph, graph.node_by_id("q2"), results={"q1": failed})


class FakeEdgeClient:
    """Production-shaped Edge client returning JSON slot objects."""

    def __init__(self, outputs_by_task_substring: dict[str, dict]):
        self.outputs_by_task_substring = outputs_by_task_substring
        self.calls: list[dict] = []

    async def generate_async(self, query: str, context: str = "", image_b64=None) -> str:
        self.calls.append(
            {"query": query, "context": context, "image_b64": image_b64}
        )
        for needle, outputs in self.outputs_by_task_substring.items():
            if needle.lower() in query.lower():
                return json.dumps(outputs)
        raise AssertionError(f"no Edge stub matched query: {query!r}")


class FakeFogClient:
    """Production-shaped Fog client that records outbound payloads."""

    def __init__(self, outputs_by_task_substring: dict[str, dict]):
        self.outputs_by_task_substring = outputs_by_task_substring
        self.calls: list[dict] = []

    async def generate_async(self, query: str, context: str = "", image_b64=None) -> str:
        self.calls.append(
            {"query": query, "context": context, "image_b64": image_b64}
        )
        for needle, outputs in self.outputs_by_task_substring.items():
            if needle.lower() in query.lower():
                return json.dumps(outputs)
        raise AssertionError(f"no Fog stub matched query: {query!r}")


def test_default_fog_client_outbound_contains_only_resolved_reference():
    asyncio.run(_test_default_fog_client_outbound_contains_only_resolved_reference())


async def _test_default_fog_client_outbound_contains_only_resolved_reference():
    graph = _gate_graph()
    edge = FakeEdgeClient(
        {
            "gate identifier": {
                "gate_identifier": "D34",
                # Undeclared keys must never reach Fog via structural transfer.
            }
        }
    )
    fog = FakeFogClient({"locate the resolved gate": {"gate_location": "Terminal D"}})

    executor = GraphExecutor(edge_client=edge, fog_client=fog, node_runner=None)
    result = await executor.execute(graph)

    assert len(fog.calls) == 1
    outbound = fog.calls[0]
    outbound_blob = json.dumps(outbound, ensure_ascii=True)

    assert result.fog_transfers[0].transferred_slots == {"gate_identifier": "D34"}
    assert '"gate_identifier": "D34"' in outbound["query"] or (
        '"gate_identifier": "D34"' in outbound["context"]
    )
    assert "D34" in outbound_blob

    # Structural absences: no TierResult dump, no undeclared Edge fields, no envelopes.
    assert "result_id" not in outbound_blob
    assert "TierResult" not in outbound_blob
    assert "evidence" not in outbound_blob
    assert "UA447" not in outbound_blob
    assert "boarding pass" not in outbound_blob
    assert "raw_response" not in outbound_blob
    assert "full_edge_response" not in outbound_blob
    assert "PERSONAL_FACT" not in outbound_blob
    assert "PERSONAL_RECORD" not in outbound_blob
    assert "medication" not in outbound_blob.lower()


def test_default_client_fog_to_fog_dependency_without_edge_transfer():
    asyncio.run(_test_default_client_fog_to_fog_dependency_without_edge_transfer())


async def _test_default_client_fog_to_fog_dependency_without_edge_transfer():
    graph = _fog_to_fog_graph()
    edge = FakeEdgeClient({})
    fog = FakeFogClient(
        {
            "identify landmark": {"landmark": "fountain"},
            "locate relative": {"place": "exit by fountain"},
        }
    )
    result = await GraphExecutor(edge, fog).execute(graph)

    assert result.fog_transfers == ()
    assert len(fog.calls) == 2
    down = fog.calls[1]
    assert '"landmark": "fountain"' in down["query"] or (
        '"landmark": "fountain"' in down["context"]
    )
    assert "PERSONAL_FACT" not in json.dumps(down)
    assert result.final_response == "exit by fountain"


def test_fog_transfers_follow_wave_node_order():
    asyncio.run(_test_fog_transfers_follow_wave_node_order())


async def _test_fog_transfers_follow_wave_node_order():
    """Two independent Fog roots in wave 0; transfers stay empty but order stable.

    Uses two Edge->Fog chains that finish with independent Fog nodes in one wave
    after their Edge parents: actually both Edges in wave0, both Fogs in wave1.
    """
    left_edge = _node(
        node_id="e1",
        operator=OperatorType.RESOLVE_PERSONAL,
        produced_outputs={"ref": SlotType.RESOLVED_REFERENCE},
        task="Resolve left",
    )
    right_edge = _node(
        node_id="e2",
        operator=OperatorType.RESOLVE_PERSONAL,
        produced_outputs={"ref": SlotType.RESOLVED_REFERENCE},
        task="Resolve right",
    )
    left_fog = _node(
        node_id="f1",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.LOCATE_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Locate left",
        required_inputs={"ref": SlotType.RESOLVED_REFERENCE},
        produced_outputs={"loc": SlotType.LOCATION},
    )
    right_fog = _node(
        node_id="f2",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.LOCATE_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Locate right",
        required_inputs={"ref": SlotType.RESOLVED_REFERENCE},
        produced_outputs={"loc": SlotType.LOCATION},
    )
    fusion = _node(
        node_id="fusion",
        semantic_type=NodeSemanticType.CONTROL,
        operator=OperatorType.FUSE,
        tier=Tier.EDGE,
        task="Fuse",
        required_inputs={
            "left": SlotType.LOCATION,
            "right": SlotType.LOCATION,
        },
        produced_outputs={"response": SlotType.FINAL_RESPONSE},
    )
    graph = _graph(
        graph_id="graph-two-transfers",
        original_query="Where are my gates?",
        query_type=QueryType.MIXED,
        nodes=(left_edge, right_edge, left_fog, right_fog, fusion),
        edges=(
            DependencyEdge(
                source_node_id="e1",
                source_slot="ref",
                target_node_id="f1",
                target_slot="ref",
                transfer_policy=TransferPolicy.MINIMAL_REFERENCE,
            ),
            DependencyEdge(
                source_node_id="e2",
                source_slot="ref",
                target_node_id="f2",
                target_slot="ref",
                transfer_policy=TransferPolicy.MINIMAL_REFERENCE,
            ),
            DependencyEdge(
                source_node_id="f1",
                source_slot="loc",
                target_node_id="fusion",
                target_slot="left",
            ),
            DependencyEdge(
                source_node_id="f2",
                source_slot="loc",
                target_node_id="fusion",
                target_slot="right",
            ),
        ),
    )
    runner = RecordingRunner(
        {
            "e1": {"ref": "A1"},
            "e2": {"ref": "B2"},
            "f1": {"loc": "left-loc"},
            "f2": {"loc": "right-loc"},
        }
    )
    # Delay f1 longer so completion order would invert if we published by finish time.
    runner.arm_barrier(["f1", "f2"])

    async def unlock():
        await runner.started["f1"].wait()
        await runner.started["f2"].wait()
        runner.release["f2"].set()
        await asyncio.sleep(0.01)
        runner.release["f1"].set()

    unlock_task = asyncio.create_task(unlock())
    result = await _executor(runner).execute(graph)
    await unlock_task

    wave1 = result.waves[1]
    assert set(wave1) == {"f1", "f2"}
    assert [t.target_node_id for t in result.fog_transfers] == list(wave1)
