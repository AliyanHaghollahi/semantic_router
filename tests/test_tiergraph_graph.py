"""Unit tests for typed TierGraph DAG construction and validation."""

import pytest
from pydantic import ValidationError

from tiergraph import (
    DependencyEdge,
    ExecutionGraph,
    NodeSemanticType,
    OperatorType,
    QueryType,
    SemanticNode,
    SlotType,
    Tier,
    TransferPolicy,
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


def _edge(**overrides) -> DependencyEdge:
    values = {
        "source_node_id": "q1",
        "source_slot": "gate_identifier",
        "target_node_id": "q2",
        "target_slot": "gate_identifier",
        "transfer_policy": TransferPolicy.MINIMAL_REFERENCE,
    }
    values.update(overrides)
    return DependencyEdge(**values)


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
        edges=(_edge(),),
    )


def _parallel_graph(include_fusion: bool = False) -> ExecutionGraph:
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
    nodes = [personal, environmental]
    edges = []
    if include_fusion:
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
        nodes.append(fusion)
        edges.extend(
            [
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
            ]
        )
    return _graph(
        graph_id="graph-parallel",
        original_query="What medication do I take and is there a pharmacy nearby?",
        query_type=QueryType.MIXED,
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


def _hybrid_graph() -> ExecutionGraph:
    gate_graph = _gate_graph()
    scene = _node(
        node_id="q3",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.DESCRIBE_ENVIRONMENT,
        tier=Tier.FOG,
        task="Describe the current scene",
        produced_outputs={"scene": SlotType.SCENE_DESCRIPTION},
    )
    return _graph(
        graph_id="graph-hybrid",
        original_query="Where is my gate and what is around me?",
        query_type=QueryType.MIXED,
        nodes=gate_graph.nodes + (scene,),
        edges=gate_graph.edges,
    )


def _fork_graph(reverse_edges: bool = False) -> ExecutionGraph:
    source = _node(
        node_id="source",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Identify two environmental facts",
        produced_outputs={
            "left_fact": SlotType.ENVIRONMENTAL_FACT,
            "right_fact": SlotType.ENVIRONMENTAL_FACT,
        },
    )
    left = _node(
        node_id="left",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Interpret the left fact",
        required_inputs={"fact": SlotType.ENVIRONMENTAL_FACT},
        produced_outputs={"left_result": SlotType.ENVIRONMENTAL_FACT},
    )
    right = _node(
        node_id="right",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Interpret the right fact",
        required_inputs={"fact": SlotType.ENVIRONMENTAL_FACT},
        produced_outputs={"right_result": SlotType.ENVIRONMENTAL_FACT},
    )
    edges = [
        DependencyEdge(
            source_node_id="source",
            source_slot="left_fact",
            target_node_id="left",
            target_slot="fact",
        ),
        DependencyEdge(
            source_node_id="source",
            source_slot="right_fact",
            target_node_id="right",
            target_slot="fact",
        ),
    ]
    if reverse_edges:
        edges.reverse()
    return _graph(
        graph_id="graph-fork",
        original_query="Identify and interpret both environmental facts",
        query_type=QueryType.ENVIRONMENTAL,
        nodes=(source, left, right),
        edges=tuple(edges),
    )


def _diamond_graph() -> ExecutionGraph:
    source = _node(
        node_id="source",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Produce two environmental facts",
        produced_outputs={
            "left_fact": SlotType.ENVIRONMENTAL_FACT,
            "right_fact": SlotType.ENVIRONMENTAL_FACT,
        },
    )
    left = _node(
        node_id="left",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Process the left fact",
        required_inputs={"fact": SlotType.ENVIRONMENTAL_FACT},
        produced_outputs={"result": SlotType.ENVIRONMENTAL_FACT},
    )
    right = _node(
        node_id="right",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Process the right fact",
        required_inputs={"fact": SlotType.ENVIRONMENTAL_FACT},
        produced_outputs={"result": SlotType.ENVIRONMENTAL_FACT},
    )
    sink = _node(
        node_id="sink",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Combine both processed facts",
        required_inputs={
            "left": SlotType.ENVIRONMENTAL_FACT,
            "right": SlotType.ENVIRONMENTAL_FACT,
        },
        produced_outputs={"combined": SlotType.ENVIRONMENTAL_FACT},
    )
    edges = (
        DependencyEdge(
            source_node_id="source",
            source_slot="left_fact",
            target_node_id="left",
            target_slot="fact",
        ),
        DependencyEdge(
            source_node_id="source",
            source_slot="right_fact",
            target_node_id="right",
            target_slot="fact",
        ),
        DependencyEdge(
            source_node_id="left",
            source_slot="result",
            target_node_id="sink",
            target_slot="left",
        ),
        DependencyEdge(
            source_node_id="right",
            source_slot="result",
            target_node_id="sink",
            target_slot="right",
        ),
    )
    return _graph(
        graph_id="graph-diamond",
        original_query="Combine two environmental observations",
        query_type=QueryType.ENVIRONMENTAL,
        nodes=(source, left, right, sink),
        edges=edges,
    )


def test_valid_personal_graph():
    graph = _graph()

    assert graph.query_type is QueryType.PERSONAL
    assert graph.nodes[0].required_inputs == {}
    assert graph.execution_mode() == "sequential"


def test_valid_environmental_fog_graph():
    node = _node(
        node_id="scene",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.DESCRIBE_ENVIRONMENT,
        tier=Tier.FOG,
        task="Describe the scene",
        produced_outputs={"scene": SlotType.SCENE_DESCRIPTION},
    )
    graph = _graph(
        graph_id="graph-scene",
        original_query="What is in front of me?",
        query_type=QueryType.ENVIRONMENTAL,
        nodes=(node,),
    )

    assert graph.node_by_id("scene") is node
    assert graph.execution_waves() == (("scene",),)


def test_where_is_my_gate_is_two_node_minimal_reference_graph():
    graph = _gate_graph()

    assert len(graph.nodes) == 2
    assert [node.operator for node in graph.nodes] == [
        OperatorType.RESOLVE_PERSONAL,
        OperatorType.LOCATE_ENVIRONMENTAL,
    ]
    assert graph.edges == (
        DependencyEdge(
            source_node_id="q1",
            source_slot="gate_identifier",
            target_node_id="q2",
            target_slot="gate_identifier",
            transfer_policy=TransferPolicy.MINIMAL_REFERENCE,
        ),
    )
    assert graph.execution_mode() == "sequential"
    assert graph.topological_order() == ("q1", "q2")
    assert graph.execution_waves() == (("q1",), ("q2",))


def test_parallel_graph_and_relationship_helpers():
    graph = _parallel_graph()

    assert graph.execution_mode() == "parallel"
    assert graph.execution_waves() == (("personal", "environmental"),)
    assert graph.predecessors("personal") == ()
    assert graph.successors("environmental") == ()


def test_hybrid_graph():
    graph = _hybrid_graph()

    assert graph.execution_mode() == "hybrid"
    assert graph.execution_waves() == (("q1", "q3"), ("q2",))
    assert graph.predecessors("q2") == (graph.node_by_id("q1"),)
    assert graph.successors("q1") == (graph.node_by_id("q2"),)


def test_fusion_convergence_does_not_change_parallel_mode():
    graph = _parallel_graph(include_fusion=True)

    assert graph.execution_mode() == "parallel"
    assert graph.execution_waves() == (
        ("personal", "environmental"),
        ("fusion",),
    )


def test_environmental_operator_may_run_on_edge():
    node = _node(
        node_id="scene",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.DESCRIBE_ENVIRONMENT,
        tier=Tier.EDGE,
        task="Describe the scene locally",
        produced_outputs={"scene": SlotType.SCENE_DESCRIPTION},
    )

    graph = _graph(
        query_type=QueryType.ENVIRONMENTAL,
        nodes=(node,),
    )
    assert graph.node_by_id("scene").tier is Tier.EDGE


def test_graph_dictionary_and_json_round_trips_accept_canonical_wire_values():
    original = _gate_graph()

    json_mapping = original.model_dump(mode="json")
    restored_from_mapping = ExecutionGraph.model_validate(json_mapping)
    restored_from_json = ExecutionGraph.model_validate_json(
        original.model_dump_json()
    )

    assert restored_from_mapping == original
    assert restored_from_json == original
    assert "execution_mode" not in json_mapping


def test_model_copy_with_edge_update_revalidates_and_rebuilds_topology():
    original = _diamond_graph()
    original_waves = original.execution_waves()
    original_order = original.topological_order()
    original_predecessors = original.predecessors("sink")
    original_successors = original.successors("source")
    updated_edges = tuple(reversed(original.edges))

    copied = original.model_copy(update={"edges": updated_edges})
    fresh_data = original.model_dump(mode="python", round_trip=True)
    fresh_data["edges"] = updated_edges
    reconstructed = ExecutionGraph.model_validate(fresh_data)

    assert copied == reconstructed
    assert copied.execution_waves() == reconstructed.execution_waves()
    assert copied.topological_order() == reconstructed.topological_order()
    assert copied.predecessors("sink") == reconstructed.predecessors("sink")
    assert copied.successors("source") == reconstructed.successors("source")
    assert copied.execution_waves() != original_waves
    assert copied.topological_order() != original_order
    assert copied.predecessors("sink") != original_predecessors
    assert copied.successors("source") != original_successors
    assert "_topology" not in copied.model_dump(mode="json")


def test_model_copy_without_update_preserves_consistent_graph():
    original = _diamond_graph()
    original.execution_waves()

    copied = original.model_copy()

    assert copied == original
    assert copied is not original
    assert copied.execution_waves() == original.execution_waves()
    assert copied.topological_order() == original.topological_order()


def test_model_copy_rejects_invalid_graph_update():
    original = _gate_graph()
    original.execution_waves()

    with pytest.raises(ValidationError, match="unresolved required input"):
        original.model_copy(update={"edges": ()})


@pytest.mark.parametrize("field", ["node_id", "task"])
def test_node_rejects_blank_id_and_task(field):
    with pytest.raises(ValidationError, match="must not be blank"):
        _node(**{field: "  "})


@pytest.mark.parametrize(
    "field",
    ["source_node_id", "source_slot", "target_node_id", "target_slot"],
)
def test_edge_rejects_blank_ids_and_slot_names(field):
    with pytest.raises(ValidationError, match="must not be blank"):
        _edge(**{field: "\t"})


@pytest.mark.parametrize("field", ["graph_id", "original_query"])
def test_graph_rejects_blank_id_and_original_query(field):
    with pytest.raises(ValidationError, match="must not be blank"):
        _graph(**{field: "\n"})


def test_graph_rejects_empty_nodes():
    with pytest.raises(ValidationError, match="require personal nodes"):
        _graph(nodes=())


def test_graph_rejects_unsupported_schema_version():
    with pytest.raises(ValidationError):
        _graph(schema_version="2.0")


def test_graph_models_reject_unknown_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _node(unexpected=True)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _edge(unexpected=True)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _graph(unexpected=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("semantic_type", "unknown"),
        ("operator", "UNKNOWN_OPERATOR"),
        ("tier", "cloud"),
        ("status", "complete"),
    ],
)
def test_node_rejects_unknown_enum_values(field, value):
    with pytest.raises(ValidationError):
        _node(**{field: value})


def test_graph_and_edge_reject_unknown_enum_values():
    with pytest.raises(ValidationError):
        _node(produced_outputs={"fact": "UNKNOWN_SLOT"})

    with pytest.raises(ValidationError):
        _edge(transfer_policy="unknown")

    with pytest.raises(ValidationError):
        _graph(query_type="Unknown")


def test_node_by_id_rejects_unknown_id():
    with pytest.raises(KeyError):
        _graph().node_by_id("missing")


def test_execution_mode_and_wave_membership_are_stable_when_edges_reordered():
    original = _fork_graph()
    reordered = _fork_graph(reverse_edges=True)

    assert original.execution_mode() == reordered.execution_mode() == "hybrid"
    assert tuple(map(frozenset, original.execution_waves())) == tuple(
        map(frozenset, reordered.execution_waves())
    )


@pytest.mark.parametrize(
    "operator,semantic_type,tier,outputs",
    [
        (
            OperatorType.RESOLVE_PERSONAL,
            NodeSemanticType.ENVIRONMENTAL,
            Tier.EDGE,
            {"reference": SlotType.RESOLVED_REFERENCE},
        ),
        (
            OperatorType.RETRIEVE_PERSONAL,
            NodeSemanticType.PERSONAL,
            Tier.FOG,
            {"fact": SlotType.PERSONAL_FACT},
        ),
        (
            OperatorType.DESCRIBE_ENVIRONMENT,
            NodeSemanticType.CONTROL,
            Tier.EDGE,
            {"scene": SlotType.SCENE_DESCRIPTION},
        ),
        (
            OperatorType.FUSE,
            NodeSemanticType.CONTROL,
            Tier.FOG,
            {"response": SlotType.FINAL_RESPONSE},
        ),
    ],
)
def test_node_rejects_invalid_operator_semantic_or_tier(
    operator, semantic_type, tier, outputs
):
    with pytest.raises(ValidationError):
        _node(
            operator=operator,
            semantic_type=semantic_type,
            tier=tier,
            required_inputs={"input": SlotType.ENVIRONMENTAL_FACT}
            if operator is OperatorType.FUSE
            else {},
            produced_outputs=outputs,
        )


@pytest.mark.parametrize(
    "operator,semantic_type,outputs",
    [
        (
            OperatorType.RESOLVE_PERSONAL,
            NodeSemanticType.PERSONAL,
            {"fact": SlotType.PERSONAL_FACT},
        ),
        (
            OperatorType.LOCATE_ENVIRONMENTAL,
            NodeSemanticType.ENVIRONMENTAL,
            {"scene": SlotType.SCENE_DESCRIPTION},
        ),
        (
            OperatorType.NAVIGATE_TO,
            NodeSemanticType.ENVIRONMENTAL,
            {"location": SlotType.LOCATION},
        ),
    ],
)
def test_node_rejects_operator_output_type_mismatch(
    operator, semantic_type, outputs
):
    with pytest.raises(ValidationError, match="cannot produce"):
        _node(
            operator=operator,
            semantic_type=semantic_type,
            produced_outputs=outputs,
        )


def test_node_rejects_empty_outputs_blank_slots_and_overlap():
    with pytest.raises(ValidationError, match="at least one output"):
        _node(produced_outputs={})

    with pytest.raises(ValidationError, match="slot names must not be blank"):
        _node(produced_outputs={" ": SlotType.PERSONAL_FACT})

    with pytest.raises(ValidationError, match="must not overlap"):
        _node(
            required_inputs={"fact": SlotType.PERSONAL_FACT},
            produced_outputs={"fact": SlotType.PERSONAL_FACT},
        )


@pytest.mark.parametrize("slot_type", [SlotType.PERSONAL_FACT, SlotType.PERSONAL_RECORD])
def test_fog_node_rejects_personal_inputs(slot_type):
    with pytest.raises(ValidationError, match="Fog inputs cannot accept"):
        _node(
            semantic_type=NodeSemanticType.ENVIRONMENTAL,
            operator=OperatorType.LOCATE_ENVIRONMENTAL,
            tier=Tier.FOG,
            required_inputs={"personal": slot_type},
            produced_outputs={"location": SlotType.LOCATION},
        )


def test_graph_rejects_duplicate_node_ids():
    with pytest.raises(ValidationError, match="node IDs must be unique"):
        _graph(nodes=(_node(), _node()))


@pytest.mark.parametrize(
    "edge,error",
    [
        (_edge(source_node_id="missing"), "source node does not exist"),
        (_edge(target_node_id="missing"), "target node does not exist"),
        (
            _edge(source_node_id="q1", target_node_id="q1"),
            "self-edges are not allowed",
        ),
        (_edge(source_slot="missing"), "source slot does not exist"),
        (_edge(target_slot="missing"), "target slot does not exist"),
    ],
)
def test_graph_rejects_invalid_edge_endpoints_and_slots(edge, error):
    gate_graph = _gate_graph()
    with pytest.raises(ValidationError, match=error):
        _graph(
            query_type=QueryType.MIXED,
            nodes=gate_graph.nodes,
            edges=(edge,),
        )


def test_graph_rejects_incompatible_slot_types():
    gate_graph = _gate_graph()
    q2 = gate_graph.node_by_id("q2").model_copy(
        update={"required_inputs": {"gate_identifier": SlotType.ENVIRONMENTAL_FACT}}
    )
    with pytest.raises(ValidationError, match="incompatible slot types"):
        _graph(
            query_type=QueryType.MIXED,
            nodes=(gate_graph.node_by_id("q1"), q2),
            edges=gate_graph.edges,
        )


def test_graph_rejects_unresolved_required_input():
    gate_graph = _gate_graph()
    with pytest.raises(ValidationError, match="unresolved required input"):
        _graph(
            query_type=QueryType.MIXED,
            nodes=gate_graph.nodes,
            edges=(),
        )


def test_graph_rejects_multiple_producers_for_one_input():
    gate_graph = _gate_graph()
    q3 = gate_graph.node_by_id("q1").model_copy(update={"node_id": "q3"})
    duplicate_binding = _edge(source_node_id="q3")
    with pytest.raises(ValidationError, match="multiple producers"):
        _graph(
            query_type=QueryType.MIXED,
            nodes=gate_graph.nodes + (q3,),
            edges=gate_graph.edges + (duplicate_binding,),
        )


def test_graph_rejects_duplicate_edges():
    gate_graph = _gate_graph()
    with pytest.raises(ValidationError, match="duplicate dependency edge"):
        _graph(
            query_type=QueryType.MIXED,
            nodes=gate_graph.nodes,
            edges=gate_graph.edges * 2,
        )


def test_edge_to_fog_requires_minimal_reference_policy():
    gate_graph = _gate_graph()
    direct_edge = gate_graph.edges[0].model_copy(
        update={"transfer_policy": TransferPolicy.DIRECT}
    )
    with pytest.raises(ValidationError, match="require minimal_reference"):
        _graph(
            query_type=QueryType.MIXED,
            nodes=gate_graph.nodes,
            edges=(direct_edge,),
        )


def test_edge_to_fog_may_transfer_only_resolved_reference():
    source = _node(
        node_id="q1",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.EDGE,
        task="Identify a local environmental fact",
        produced_outputs={"fact": SlotType.ENVIRONMENTAL_FACT},
    )
    target = _node(
        node_id="q2",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Use a fact",
        required_inputs={"fact": SlotType.ENVIRONMENTAL_FACT},
        produced_outputs={"result": SlotType.ENVIRONMENTAL_FACT},
    )
    with pytest.raises(ValidationError, match="only RESOLVED_REFERENCE"):
        _graph(
            query_type=QueryType.ENVIRONMENTAL,
            nodes=(source, target),
            edges=(
                DependencyEdge(
                    source_node_id="q1",
                    source_slot="fact",
                    target_node_id="q2",
                    target_slot="fact",
                    transfer_policy=TransferPolicy.MINIMAL_REFERENCE,
                ),
            ),
        )


def test_minimal_reference_rejected_for_non_edge_to_fog_dependency():
    source = _node(
        node_id="q1",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Identify a landmark",
        produced_outputs={"landmark": SlotType.ENVIRONMENTAL_FACT},
    )
    target = _node(
        node_id="q2",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Identify the next landmark",
        required_inputs={"landmark": SlotType.ENVIRONMENTAL_FACT},
        produced_outputs={"next": SlotType.ENVIRONMENTAL_FACT},
    )
    edge = DependencyEdge(
        source_node_id="q1",
        source_slot="landmark",
        target_node_id="q2",
        target_slot="landmark",
        transfer_policy=TransferPolicy.MINIMAL_REFERENCE,
    )
    with pytest.raises(ValidationError, match="valid only for Edge-to-Fog"):
        _graph(
            query_type=QueryType.ENVIRONMENTAL,
            nodes=(source, target),
            edges=(edge,),
        )


def test_valid_same_tier_direct_dependency_with_multiple_slots_is_deduplicated():
    q1 = _node(
        node_id="q1",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.EDGE,
        task="Produce two environmental facts",
        produced_outputs={
            "first": SlotType.ENVIRONMENTAL_FACT,
            "second": SlotType.ENVIRONMENTAL_FACT,
        },
    )
    q2 = _node(
        node_id="q2",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.EDGE,
        task="Consume both environmental facts",
        required_inputs={
            "first_input": SlotType.ENVIRONMENTAL_FACT,
            "second_input": SlotType.ENVIRONMENTAL_FACT,
        },
        produced_outputs={"combined": SlotType.ENVIRONMENTAL_FACT},
    )
    edges = (
        DependencyEdge(
            source_node_id="q1",
            source_slot="first",
            target_node_id="q2",
            target_slot="first_input",
        ),
        DependencyEdge(
            source_node_id="q1",
            source_slot="second",
            target_node_id="q2",
            target_slot="second_input",
        ),
    )
    graph = _graph(
        query_type=QueryType.ENVIRONMENTAL,
        nodes=(q1, q2),
        edges=edges,
    )

    assert graph.topological_order() == ("q1", "q2")
    assert graph.execution_waves() == (("q1",), ("q2",))
    assert graph.predecessors("q2") == (q1,)
    assert graph.successors("q1") == (q2,)


def test_valid_fog_to_edge_direct_dependency():
    fog = _node(
        node_id="fog",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.FOG,
        task="Identify a remote environmental fact",
        produced_outputs={"fact": SlotType.ENVIRONMENTAL_FACT},
    )
    edge = _node(
        node_id="edge",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.EDGE,
        task="Use the environmental fact on Edge",
        required_inputs={"fact": SlotType.ENVIRONMENTAL_FACT},
        produced_outputs={"result": SlotType.ENVIRONMENTAL_FACT},
    )
    dependency = DependencyEdge(
        source_node_id="fog",
        source_slot="fact",
        target_node_id="edge",
        target_slot="fact",
        transfer_policy=TransferPolicy.DIRECT,
    )

    graph = _graph(
        query_type=QueryType.ENVIRONMENTAL,
        nodes=(fog, edge),
        edges=(dependency,),
    )
    assert graph.execution_mode() == "sequential"


def _three_node_cycle(include_downstream: bool = False):
    nodes = []
    edges = []
    for index in range(1, 4):
        previous = 3 if index == 1 else index - 1
        nodes.append(
            _node(
                node_id=f"q{index}",
                semantic_type=NodeSemanticType.ENVIRONMENTAL,
                operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
                tier=Tier.FOG,
                task=f"Cycle node {index}",
                required_inputs={
                    f"from_q{previous}": SlotType.ENVIRONMENTAL_FACT
                },
                produced_outputs={
                    f"from_q{index}": SlotType.ENVIRONMENTAL_FACT
                },
            )
        )
        target = 1 if index == 3 else index + 1
        edges.append(
            DependencyEdge(
                source_node_id=f"q{index}",
                source_slot=f"from_q{index}",
                target_node_id=f"q{target}",
                target_slot=f"from_q{index}",
            )
        )

    if include_downstream:
        nodes.append(
            _node(
                node_id="downstream",
                semantic_type=NodeSemanticType.ENVIRONMENTAL,
                operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
                tier=Tier.FOG,
                task="Node downstream from the cycle",
                required_inputs={"from_q3": SlotType.ENVIRONMENTAL_FACT},
                produced_outputs={"result": SlotType.ENVIRONMENTAL_FACT},
            )
        )
        edges.append(
            DependencyEdge(
                source_node_id="q3",
                source_slot="from_q3",
                target_node_id="downstream",
                target_slot="from_q3",
            )
        )
    return tuple(nodes), tuple(edges)


@pytest.mark.parametrize("include_downstream", [False, True])
def test_graph_rejects_three_node_cycle_and_reports_blocked_nodes(
    include_downstream,
):
    nodes, edges = _three_node_cycle(include_downstream)
    with pytest.raises(ValidationError, match="nodes blocked by a cycle"):
        _graph(
            query_type=QueryType.ENVIRONMENTAL,
            nodes=nodes,
            edges=edges,
        )


@pytest.mark.parametrize(
    "query_type,nodes",
    [
        (QueryType.PERSONAL, (_parallel_graph().node_by_id("environmental"),)),
        (QueryType.ENVIRONMENTAL, (_node(),)),
        (QueryType.MIXED, (_node(),)),
    ],
)
def test_graph_rejects_query_type_node_semantic_mismatch(query_type, nodes):
    with pytest.raises(ValidationError):
        _graph(query_type=query_type, nodes=nodes)


def test_fuse_node_must_be_terminal():
    parallel_graph = _parallel_graph(include_fusion=True)
    downstream = _node(
        node_id="downstream",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
        tier=Tier.EDGE,
        task="Consume an invalid fused response",
        required_inputs={"response": SlotType.FINAL_RESPONSE},
        produced_outputs={"fact": SlotType.ENVIRONMENTAL_FACT},
    )
    outgoing = DependencyEdge(
        source_node_id="fusion",
        source_slot="response",
        target_node_id="downstream",
        target_slot="response",
    )
    with pytest.raises(ValidationError, match="FUSE nodes must be terminal"):
        _graph(
            query_type=QueryType.MIXED,
            nodes=parallel_graph.nodes + (downstream,),
            edges=parallel_graph.edges + (outgoing,),
        )
