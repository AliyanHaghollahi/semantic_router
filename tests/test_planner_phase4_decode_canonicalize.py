"""Phase-4 step-1 tests: GraphDecoder and canonical graph comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiergraph.enums import (
    NodeSemanticType,
    OperatorType,
    QueryType,
    SlotType,
    Tier,
    TransferPolicy,
)
from tiergraph.graph import ExecutionGraph
from tiergraph.planner.annotations import ImplicitResolution, PlannerExample
from tiergraph.planner.canonicalize import graphs_exactly_match
from tiergraph.planner.decode import (
    DecodedPlan,
    GraphDecoder,
    PlannerDecodeError,
    PlannerPredictions,
    PredictedAnchor,
    PredictedOperation,
)
from tiergraph.planner.naming import principal_slot_name
from tiergraph.planner.tasks import render_answer_task


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "planner" / "where_is_my_gate.json"
)


def _gate_example() -> PlannerExample:
    return PlannerExample.model_validate(
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    )


def _gate_predictions() -> tuple[str, PlannerPredictions]:
    example = _gate_example()
    query = example.query
    op = example.planner_labels.operation_spans[0]
    anchor = example.planner_labels.slot_anchors[0]
    predictions = PlannerPredictions(
        operations=(
            PredictedOperation(
                start=op.start,
                end=op.end,
                operator=op.operator,
            ),
        ),
        anchors=(
            PredictedAnchor(
                start=anchor.start,
                end=anchor.end,
                text=anchor.text,
                owner_index=0,
                implicit_resolution=anchor.implicit_resolution,
                normalized_name=anchor.normalized_name,
            ),
        ),
        dependency_pairs=frozenset(),
        aux_query_type=QueryType.ENVIRONMENTAL,
    )
    return query, predictions


def test_gate_decode_creates_environmental_and_resolve_nodes():
    query, predictions = _gate_predictions()
    decoded = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="mixed-gate-001",
    )
    graph = decoded.graph
    locate = [
        node
        for node in graph.nodes
        if node.operator is OperatorType.LOCATE_ENVIRONMENTAL
    ]
    resolve = [
        node
        for node in graph.nodes
        if node.operator is OperatorType.RESOLVE_PERSONAL
    ]
    assert len(locate) == 1
    assert locate[0].semantic_type is NodeSemanticType.ENVIRONMENTAL
    assert locate[0].tier is Tier.FOG
    assert len(resolve) == 1
    assert resolve[0].tier is Tier.EDGE
    assert tuple(resolve[0].produced_outputs.values()) == (
        SlotType.RESOLVED_REFERENCE,
    )


def test_gate_decode_creates_mandatory_resolved_reference_edge():
    query, predictions = _gate_predictions()
    graph = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="mixed-gate-001",
    ).graph
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.transfer_policy is TransferPolicy.MINIMAL_REFERENCE
    source = graph.node_by_id(edge.source_node_id)
    target = graph.node_by_id(edge.target_node_id)
    assert source.operator is OperatorType.RESOLVE_PERSONAL
    assert target.operator is OperatorType.LOCATE_ENVIRONMENTAL
    assert source.produced_outputs[edge.source_slot] is SlotType.RESOLVED_REFERENCE
    assert target.required_inputs[edge.target_slot] is SlotType.RESOLVED_REFERENCE


def test_gate_decode_query_type_is_mixed():
    query, predictions = _gate_predictions()
    graph = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="mixed-gate-001",
    ).graph
    assert graph.query_type is QueryType.MIXED


def test_aux_query_type_cannot_override_graph_semantics():
    query, predictions = _gate_predictions()
    assert predictions.aux_query_type is QueryType.ENVIRONMENTAL
    graph = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="mixed-gate-001",
    ).graph
    assert graph.query_type is QueryType.MIXED


def test_legal_principal_only_explicit_dependency_decodes():
    query = "Identify this sign then locate this gate"
    split = query.index(" then ")
    right = split + len(" then ")
    sign_start = query.index("sign")
    gate_start = query.index("gate")
    predictions = PlannerPredictions(
        operations=(
            PredictedOperation(
                start=0,
                end=split,
                operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
            ),
            PredictedOperation(
                start=right,
                end=len(query),
                operator=OperatorType.LOCATE_ENVIRONMENTAL,
            ),
        ),
        anchors=(
            PredictedAnchor(
                start=sign_start,
                end=sign_start + 4,
                text="sign",
                owner_index=0,
                implicit_resolution=ImplicitResolution.NONE,
                normalized_name="sign",
            ),
            PredictedAnchor(
                start=gate_start,
                end=gate_start + 4,
                text="gate",
                owner_index=1,
                implicit_resolution=ImplicitResolution.NONE,
                normalized_name="gate",
            ),
        ),
        dependency_pairs=frozenset({(0, 1)}),
    )
    graph = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="identify-locate-001",
    ).graph
    answer_edges = [
        edge
        for edge in graph.edges
        if graph.node_by_id(edge.target_node_id).operator
        is not OperatorType.FUSE
    ]
    assert len(answer_edges) == 1
    edge = answer_edges[0]
    source = graph.node_by_id(edge.source_node_id)
    target = graph.node_by_id(edge.target_node_id)
    assert source.operator is OperatorType.IDENTIFY_ENVIRONMENTAL
    assert target.operator is OperatorType.LOCATE_ENVIRONMENTAL
    assert source.produced_outputs[edge.source_slot] is SlotType.ENVIRONMENTAL_FACT
    assert target.required_inputs[edge.target_slot] is SlotType.ENVIRONMENTAL_FACT


def test_incompatible_explicit_dependency_raises():
    query = "Retrieve my code then locate this gate"
    split = query.index(" then ")
    right = split + len(" then ")
    code_start = query.index("code")
    gate_start = query.index("gate")
    predictions = PlannerPredictions(
        operations=(
            PredictedOperation(
                start=0,
                end=split,
                operator=OperatorType.RETRIEVE_PERSONAL,
            ),
            PredictedOperation(
                start=right,
                end=len(query),
                operator=OperatorType.LOCATE_ENVIRONMENTAL,
            ),
        ),
        anchors=(
            PredictedAnchor(
                start=code_start,
                end=code_start + 4,
                text="code",
                owner_index=0,
                implicit_resolution=ImplicitResolution.NONE,
                normalized_name="code",
            ),
            PredictedAnchor(
                start=gate_start,
                end=gate_start + 4,
                text="gate",
                owner_index=1,
                implicit_resolution=ImplicitResolution.NONE,
                normalized_name="gate",
            ),
        ),
        dependency_pairs=frozenset({(0, 1)}),
    )
    with pytest.raises(PlannerDecodeError, match="incompatible explicit dependency"):
        GraphDecoder().decode(
            predictions,
            query=query,
            graph_id="bad-dep-001",
        )


def test_no_heuristic_graph_repair_on_op_overlap():
    query = "Where is my gate?"
    predictions = PlannerPredictions(
        operations=(
            PredictedOperation(
                start=0,
                end=10,
                operator=OperatorType.LOCATE_ENVIRONMENTAL,
            ),
            PredictedOperation(
                start=5,
                end=17,
                operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
            ),
        ),
        anchors=(
            PredictedAnchor(
                start=12,
                end=16,
                text="gate",
                owner_index=0,
                implicit_resolution=ImplicitResolution.NONE,
                normalized_name="gate",
            ),
        ),
        dependency_pairs=frozenset(),
    )
    with pytest.raises(PlannerDecodeError, match="must not overlap"):
        GraphDecoder().decode(
            predictions,
            query=query,
            graph_id="overlap-001",
        )


def test_multiple_sinks_create_deterministic_fuse():
    query = "Read this sign then describe this room"
    split = query.index(" then ")
    right = split + len(" then ")
    sign_start = query.index("sign")
    room_start = query.index("room")
    predictions = PlannerPredictions(
        operations=(
            PredictedOperation(
                start=0,
                end=split,
                operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
            ),
            PredictedOperation(
                start=right,
                end=len(query),
                operator=OperatorType.DESCRIBE_ENVIRONMENT,
            ),
        ),
        anchors=(
            PredictedAnchor(
                start=sign_start,
                end=sign_start + 4,
                text="sign",
                owner_index=0,
                implicit_resolution=ImplicitResolution.NONE,
                normalized_name="sign",
            ),
            PredictedAnchor(
                start=room_start,
                end=room_start + 4,
                text="room",
                owner_index=1,
                implicit_resolution=ImplicitResolution.NONE,
                normalized_name="room",
            ),
        ),
        dependency_pairs=frozenset(),
    )
    decoded = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="multi-sink-001",
    )
    fuse_nodes = [
        node for node in decoded.graph.nodes if node.operator is OperatorType.FUSE
    ]
    assert len(fuse_nodes) == 1
    assert decoded.fusion_plan is not None
    assert decoded.fusion_plan.strategy.value == "validated_slm"
    assert set(decoded.fusion_plan.required_slots) == set(
        fuse_nodes[0].required_inputs
    )


def test_single_sink_does_not_create_fuse():
    query, predictions = _gate_predictions()
    decoded = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="mixed-gate-001",
    )
    assert all(
        node.operator is not OperatorType.FUSE for node in decoded.graph.nodes
    )
    assert decoded.fusion_plan is None


def test_operation_anchor_overlap_decodes():
    query, predictions = _gate_predictions()
    op = predictions.operations[0]
    anchor = predictions.anchors[0]
    assert op.start <= anchor.start < anchor.end <= op.end
    graph = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="mixed-gate-001",
    ).graph
    assert graph.query_type is QueryType.MIXED


def test_canonical_match_ignores_node_ids():
    query, predictions = _gate_predictions()
    left = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="a",
    ).graph
    right = _relabel_graph(left, {"op_1": "q2", "impl_1": "q1"})
    assert graphs_exactly_match(left, right)


def test_canonical_match_ignores_task_strings():
    query, predictions = _gate_predictions()
    left = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="a",
    ).graph
    mutated = left.model_dump(mode="python", round_trip=True)
    for node in mutated["nodes"]:
        node["task"] = f"DIFFERENT::{node['task']}"
    right = ExecutionGraph.model_validate(mutated)
    assert left.nodes[0].task != right.nodes[0].task
    assert graphs_exactly_match(left, right)


def test_canonical_match_breaks_on_operator_or_edge_change():
    query, predictions = _gate_predictions()
    base = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="a",
    ).graph
    changed_op = base.model_dump(mode="python", round_trip=True)
    for node in changed_op["nodes"]:
        if node["operator"] == "LOCATE_ENVIRONMENTAL":
            node["operator"] = "IDENTIFY_ENVIRONMENTAL"
            node["semantic_type"] = "environmental"
            node["produced_outputs"] = {
                principal_slot_name(
                    base_name="gate",
                    slot_type=SlotType.ENVIRONMENTAL_FACT,
                ): "ENVIRONMENTAL_FACT"
            }
            node["task"] = render_answer_task(
                operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
                base_name="gate",
            )
    altered = ExecutionGraph.model_validate(changed_op)
    assert not graphs_exactly_match(base, altered)

    dropped_edge = base.model_dump(mode="python", round_trip=True)
    dropped_edge["edges"] = []
    for node in dropped_edge["nodes"]:
        if node["operator"] == "LOCATE_ENVIRONMENTAL":
            node["required_inputs"] = {}
    no_edge = ExecutionGraph.model_validate(dropped_edge)
    assert not graphs_exactly_match(base, no_edge)


def test_canonical_match_invariant_to_node_permutation():
    query, predictions = _gate_predictions()
    left = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="a",
    ).graph
    permuted = left.model_dump(mode="python", round_trip=True)
    permuted["nodes"] = list(reversed(permuted["nodes"]))
    right = ExecutionGraph.model_validate(permuted)
    assert [node.node_id for node in left.nodes] != [
        node.node_id for node in right.nodes
    ]
    assert graphs_exactly_match(left, right)


def test_naming_and_task_strings_separate_from_semantic_match():
    query, predictions = _gate_predictions()
    decoded = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="mixed-gate-001",
    ).graph
    locate = next(
        node
        for node in decoded.nodes
        if node.operator is OperatorType.LOCATE_ENVIRONMENTAL
    )
    assert "gate_location" in locate.produced_outputs
    assert locate.task == "Locate the resolved gate"
    # Gold fixture uses different IDs/tasks but same semantics.
    gold = _gate_example().graph
    assert graphs_exactly_match(decoded, gold)


def test_anchorless_describe_and_identify_use_operator_defaults():
    cases = (
        (
            "Describe what I see.",
            OperatorType.DESCRIBE_ENVIRONMENT,
            "scene_scene",
            QueryType.ENVIRONMENTAL,
        ),
        (
            "What is in front of me?",
            OperatorType.IDENTIFY_ENVIRONMENTAL,
            "environment_fact",
            QueryType.ENVIRONMENTAL,
        ),
    )
    for query, operator, expected_slot, query_type in cases:
        predictions = PlannerPredictions(
            operations=(
                PredictedOperation(start=0, end=len(query), operator=operator),
            ),
            anchors=(),
            dependency_pairs=frozenset(),
        )
        decoded = GraphDecoder().decode(
            predictions,
            query=query,
            graph_id=f"anchorless-{operator.value}",
        )
        assert isinstance(decoded, DecodedPlan)
        assert decoded.fusion_plan is None
        assert len(decoded.graph.nodes) == 1
        node = decoded.graph.nodes[0]
        assert node.operator is operator
        assert expected_slot in node.produced_outputs
        assert decoded.graph.query_type is query_type


def test_decode_returns_decoded_plan_not_bare_graph():
    query, predictions = _gate_predictions()
    decoded = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="mixed-gate-001",
    )
    assert isinstance(decoded, DecodedPlan)
    assert isinstance(decoded.graph, ExecutionGraph)
    assert decoded.fusion_plan is None


def test_canonical_match_distinguishes_duplicate_signatures_with_crossed_edges():
    """Duplicate local signatures: permutation matches; missing/crossed wiring does not."""

    def _parallel_identify_fuse(
        *,
        node_ids: tuple[str, str, str],
        edge_sources: tuple[str, str],
        graph_id: str,
    ) -> ExecutionGraph:
        left_id, right_id, fuse_id = node_ids
        left_slot = f"{edge_sources[0]}__environment_fact"
        right_slot = f"{edge_sources[1]}__environment_fact"
        return ExecutionGraph.model_validate(
            {
                "schema_version": "1.0",
                "graph_id": graph_id,
                "original_query": "x",
                "query_type": "Environmental",
                "nodes": [
                    {
                        "node_id": left_id,
                        "semantic_type": "environmental",
                        "operator": "IDENTIFY_ENVIRONMENTAL",
                        "tier": "fog",
                        "task": "Identify the environment",
                        "required_inputs": {},
                        "produced_outputs": {
                            "environment_fact": "ENVIRONMENTAL_FACT"
                        },
                        "status": "pending",
                        "metadata": {},
                    },
                    {
                        "node_id": right_id,
                        "semantic_type": "environmental",
                        "operator": "IDENTIFY_ENVIRONMENTAL",
                        "tier": "fog",
                        "task": "Identify the environment again",
                        "required_inputs": {},
                        "produced_outputs": {
                            "environment_fact": "ENVIRONMENTAL_FACT"
                        },
                        "status": "pending",
                        "metadata": {},
                    },
                    {
                        "node_id": fuse_id,
                        "semantic_type": "control",
                        "operator": "FUSE",
                        "tier": "edge",
                        "task": "Fuse the terminal answers",
                        "required_inputs": {
                            left_slot: "ENVIRONMENTAL_FACT",
                            right_slot: "ENVIRONMENTAL_FACT",
                        },
                        "produced_outputs": {"response": "FINAL_RESPONSE"},
                        "status": "pending",
                        "metadata": {},
                    },
                ],
                "edges": [
                    {
                        "source_node_id": edge_sources[0],
                        "source_slot": "environment_fact",
                        "target_node_id": fuse_id,
                        "target_slot": left_slot,
                        "transfer_policy": "direct",
                    },
                    {
                        "source_node_id": edge_sources[1],
                        "source_slot": "environment_fact",
                        "target_node_id": fuse_id,
                        "target_slot": right_slot,
                        "transfer_policy": "direct",
                    },
                ],
                "metadata": {},
            }
        )

    baseline = _parallel_identify_fuse(
        node_ids=("a", "b", "fuse"),
        edge_sources=("a", "b"),
        graph_id="dup-1",
    )
    # Duplicate IDENTIFY fingerprints; node-ID permutation with both edges kept.
    permuted = _parallel_identify_fuse(
        node_ids=("x", "y", "fuse"),
        edge_sources=("x", "y"),
        graph_id="dup-2",
    )
    assert graphs_exactly_match(baseline, permuted)

    # Crossed/degenerate wiring: both fuse inputs fed by the same identify node.
    # Invalid for ExecutionGraph (multiple producers... actually same source twice
    # to different slots is allowed). Use a chain instead for non-match.
    chain = ExecutionGraph.model_validate(
        {
            "schema_version": "1.0",
            "graph_id": "dup-3",
            "original_query": "x",
            "query_type": "Environmental",
            "nodes": [
                {
                    "node_id": "a",
                    "semantic_type": "environmental",
                    "operator": "IDENTIFY_ENVIRONMENTAL",
                    "tier": "fog",
                    "task": "Identify the environment",
                    "required_inputs": {},
                    "produced_outputs": {"environment_fact": "ENVIRONMENTAL_FACT"},
                    "status": "pending",
                    "metadata": {},
                },
                {
                    "node_id": "b",
                    "semantic_type": "environmental",
                    "operator": "LOCATE_ENVIRONMENTAL",
                    "tier": "fog",
                    "task": "Locate the resolved location",
                    "required_inputs": {
                        "environment_fact": "ENVIRONMENTAL_FACT"
                    },
                    "produced_outputs": {"location_location": "LOCATION"},
                    "status": "pending",
                    "metadata": {},
                },
            ],
            "edges": [
                {
                    "source_node_id": "a",
                    "source_slot": "environment_fact",
                    "target_node_id": "b",
                    "target_slot": "environment_fact",
                    "transfer_policy": "direct",
                }
            ],
            "metadata": {},
        }
    )
    assert not graphs_exactly_match(baseline, chain)

    # Reversed dependency on the chain: locate cannot feed identify under contracts,
    # so build identify←locate by swapping roles with compatible PERSONAL chain.
    forward_retrieve = ExecutionGraph.model_validate(
        {
            "schema_version": "1.0",
            "graph_id": "ret-1",
            "original_query": "x",
            "query_type": "Personal",
            "nodes": [
                {
                    "node_id": "src",
                    "semantic_type": "personal",
                    "operator": "RETRIEVE_PERSONAL",
                    "tier": "edge",
                    "task": "A",
                    "required_inputs": {},
                    "produced_outputs": {"personal_fact": "PERSONAL_FACT"},
                    "status": "pending",
                    "metadata": {},
                },
                {
                    "node_id": "mid",
                    "semantic_type": "personal",
                    "operator": "RETRIEVE_PERSONAL",
                    "tier": "edge",
                    "task": "B",
                    "required_inputs": {"personal_fact": "PERSONAL_FACT"},
                    "produced_outputs": {"mid_fact": "PERSONAL_FACT"},
                    "status": "pending",
                    "metadata": {},
                },
                {
                    "node_id": "dst",
                    "semantic_type": "personal",
                    "operator": "RETRIEVE_PERSONAL",
                    "tier": "edge",
                    "task": "C",
                    "required_inputs": {"mid_fact": "PERSONAL_FACT"},
                    "produced_outputs": {"answer_fact": "PERSONAL_FACT"},
                    "status": "pending",
                    "metadata": {},
                },
            ],
            "edges": [
                {
                    "source_node_id": "src",
                    "source_slot": "personal_fact",
                    "target_node_id": "mid",
                    "target_slot": "personal_fact",
                    "transfer_policy": "direct",
                },
                {
                    "source_node_id": "mid",
                    "source_slot": "mid_fact",
                    "target_node_id": "dst",
                    "target_slot": "mid_fact",
                    "transfer_policy": "direct",
                },
            ],
            "metadata": {},
        }
    )
    # Skip-connection / crossed structure: src→dst only (different connectivity).
    crossed = ExecutionGraph.model_validate(
        {
            "schema_version": "1.0",
            "graph_id": "ret-2",
            "original_query": "x",
            "query_type": "Personal",
            "nodes": [
                {
                    "node_id": "src",
                    "semantic_type": "personal",
                    "operator": "RETRIEVE_PERSONAL",
                    "tier": "edge",
                    "task": "A",
                    "required_inputs": {},
                    "produced_outputs": {"personal_fact": "PERSONAL_FACT"},
                    "status": "pending",
                    "metadata": {},
                },
                {
                    "node_id": "mid",
                    "semantic_type": "personal",
                    "operator": "RETRIEVE_PERSONAL",
                    "tier": "edge",
                    "task": "B",
                    "required_inputs": {},
                    "produced_outputs": {"mid_fact": "PERSONAL_FACT"},
                    "status": "pending",
                    "metadata": {},
                },
                {
                    "node_id": "dst",
                    "semantic_type": "personal",
                    "operator": "RETRIEVE_PERSONAL",
                    "tier": "edge",
                    "task": "C",
                    "required_inputs": {"personal_fact": "PERSONAL_FACT"},
                    "produced_outputs": {"answer_fact": "PERSONAL_FACT"},
                    "status": "pending",
                    "metadata": {},
                },
                {
                    "node_id": "fuse",
                    "semantic_type": "control",
                    "operator": "FUSE",
                    "tier": "edge",
                    "task": "Fuse the terminal answers",
                    "required_inputs": {
                        "mid__mid_fact": "PERSONAL_FACT",
                        "dst__answer_fact": "PERSONAL_FACT",
                    },
                    "produced_outputs": {"response": "FINAL_RESPONSE"},
                    "status": "pending",
                    "metadata": {},
                },
            ],
            "edges": [
                {
                    "source_node_id": "src",
                    "source_slot": "personal_fact",
                    "target_node_id": "dst",
                    "target_slot": "personal_fact",
                    "transfer_policy": "direct",
                },
                {
                    "source_node_id": "mid",
                    "source_slot": "mid_fact",
                    "target_node_id": "fuse",
                    "target_slot": "mid__mid_fact",
                    "transfer_policy": "direct",
                },
                {
                    "source_node_id": "dst",
                    "source_slot": "answer_fact",
                    "target_node_id": "fuse",
                    "target_slot": "dst__answer_fact",
                    "transfer_policy": "direct",
                },
            ],
            "metadata": {},
        }
    )
    assert not graphs_exactly_match(forward_retrieve, crossed)


def _relabel_graph(
    graph: ExecutionGraph,
    mapping: dict[str, str],
) -> ExecutionGraph:
    payload = graph.model_dump(mode="python", round_trip=True)
    for node in payload["nodes"]:
        node["node_id"] = mapping.get(node["node_id"], node["node_id"])
    for edge in payload["edges"]:
        edge["source_node_id"] = mapping.get(
            edge["source_node_id"],
            edge["source_node_id"],
        )
        edge["target_node_id"] = mapping.get(
            edge["target_node_id"],
            edge["target_node_id"],
        )
    return ExecutionGraph.model_validate(payload)
