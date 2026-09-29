"""Official user responses for four finished execution graphs.

These tests use injected node results. They do not call an Edge or Fog model.
"""

from __future__ import annotations

import asyncio

import pytest

from edge.fusion import ResponseFuser
from tiergraph import (
    DependencyEdge,
    ExecutionGraph,
    GraphExecutor,
    NodeSemanticType,
    OperatorType,
    QueryType,
    SemanticNode,
    SlotType,
    Tier,
    TransferPolicy,
)
from tiergraph.executor import PHASE3_TEMPORARY_FUSION_METHOD
from tiergraph.response import compose_user_response


def _node(**overrides) -> SemanticNode:
    values = {
        "node_id": "personal",
        "semantic_type": NodeSemanticType.PERSONAL,
        "operator": OperatorType.RESOLVE_PERSONAL,
        "tier": Tier.EDGE,
        "task": "Resolve the requested personal reference",
        "required_inputs": {},
        "produced_outputs": {"gate_identifier": SlotType.RESOLVED_REFERENCE},
    }
    values.update(overrides)
    return SemanticNode(**values)


def _graph(**overrides) -> ExecutionGraph:
    values = {
        "graph_id": "graph-response",
        "original_query": "What is my gate number?",
        "query_type": QueryType.PERSONAL,
        "nodes": (_node(),),
        "edges": (),
    }
    values.update(overrides)
    return ExecutionGraph(**values)


class _Runner:
    def __init__(self, outputs: dict[str, dict[str, str]]) -> None:
        self._outputs = outputs

    def __call__(self, node, _bound, _transfer):
        return self._outputs[node.node_id]


def _execute(graph: ExecutionGraph, outputs: dict[str, dict[str, str]]):
    executor = GraphExecutor(
        edge_client=object(),
        fog_client=object(),
        node_runner=_Runner(outputs),
    )
    return asyncio.run(executor.execute(graph))


def _personal_graph() -> ExecutionGraph:
    return _graph()


def _environmental_graph() -> ExecutionGraph:
    return _graph(
        graph_id="graph-where",
        original_query="Where is Gate 54?",
        query_type=QueryType.ENVIRONMENTAL,
        nodes=(
            _node(
                node_id="where",
                semantic_type=NodeSemanticType.ENVIRONMENTAL,
                operator=OperatorType.LOCATE_ENVIRONMENTAL,
                tier=Tier.FOG,
                task="Locate Gate 54",
                produced_outputs={"location": SlotType.LOCATION},
            ),
        ),
    )


def _sequential_graph() -> ExecutionGraph:
    resolve = _node(
        node_id="resolve",
        task="Resolve my gate",
    )
    navigate = _node(
        node_id="navigate",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.NAVIGATE_TO,
        tier=Tier.FOG,
        task="Navigate to the resolved gate",
        required_inputs={"gate_identifier": SlotType.RESOLVED_REFERENCE},
        produced_outputs={"directions": SlotType.NAVIGATION_INSTRUCTION},
    )
    return _graph(
        graph_id="graph-how",
        original_query="How do I get to my gate?",
        query_type=QueryType.MIXED,
        nodes=(resolve, navigate),
        edges=(
            DependencyEdge(
                source_node_id="resolve",
                source_slot="gate_identifier",
                target_node_id="navigate",
                target_slot="gate_identifier",
                transfer_policy=TransferPolicy.MINIMAL_REFERENCE,
            ),
        ),
    )


def _parallel_graph() -> ExecutionGraph:
    order = _node(
        node_id="order",
        operator=OperatorType.RETRIEVE_PERSONAL,
        task="Retrieve what I ordered",
        produced_outputs={"order": SlotType.PERSONAL_FACT},
    )
    receipt = _node(
        node_id="receipt",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.DESCRIBE_ENVIRONMENT,
        tier=Tier.FOG,
        task="Read the receipt",
        produced_outputs={"receipt": SlotType.SCENE_DESCRIPTION},
    )
    fuse = _node(
        node_id="fuse",
        semantic_type=NodeSemanticType.CONTROL,
        operator=OperatorType.FUSE,
        tier=Tier.EDGE,
        task="Combine the receipt and the order",
        required_inputs={
            "order": SlotType.PERSONAL_FACT,
            "receipt": SlotType.SCENE_DESCRIPTION,
        },
        produced_outputs={"response": SlotType.FINAL_RESPONSE},
    )
    return _graph(
        graph_id="graph-receipt",
        original_query=(
            "What is written on this receipt and does it match what I ordered?"
        ),
        query_type=QueryType.MIXED,
        nodes=(order, receipt, fuse),
        edges=(
            DependencyEdge(
                source_node_id="order",
                source_slot="order",
                target_node_id="fuse",
                target_slot="order",
            ),
            DependencyEdge(
                source_node_id="receipt",
                source_slot="receipt",
                target_node_id="fuse",
                target_slot="receipt",
            ),
        ),
    )


def test_personal_edge_only_response():
    graph = _personal_graph()
    result = _execute(graph, {"personal": {"gate_identifier": "Gate 54"}})
    assert result.final_response == "Gate 54"
    assert result.response_fusion_mode == "none"
    assert result.user_response == "Your gate is Gate 54."
    assert result.fog_transfers == ()


def test_environmental_fog_only_response():
    graph = _environmental_graph()
    result = _execute(
        graph,
        {"where": {"location": "the east concourse"}},
    )
    assert result.final_response == "the east concourse"
    assert result.response_fusion_mode == "none"
    assert result.user_response == "The location is the east concourse."
    assert result.fog_transfers == ()


def test_mixed_sequential_response_uses_dependency():
    graph = _sequential_graph()
    outputs = {
        "resolve": {"gate_identifier": "Gate 54"},
        "navigate": {"directions": "walk straight and turn left at the cafe"},
    }
    result = _execute(graph, outputs)
    assert result.waves == (("resolve",), ("navigate",))
    assert result.fog_transfers[0].transferred_slots == {"gate_identifier": "Gate 54"}
    assert result.response_fusion_mode == "sequential"
    assert result.user_response == (
        "Your gate is Gate 54. To get there, walk straight and turn left at the cafe."
    )
    assert result.final_response == "walk straight and turn left at the cafe"
    assert result.user_response != result.final_response


def test_mixed_parallel_compares_receipt_and_order():
    graph = _parallel_graph()
    match = _execute(
        graph,
        {
            "order": {"order": "Large coffee"},
            "receipt": {"receipt": "large  coffee"},
        },
    )
    mismatch = _execute(
        graph,
        {
            "order": {"order": "Large coffee"},
            "receipt": {"receipt": "Green tea"},
        },
    )
    assert match.response_fusion_mode == "parallel"
    assert match.fusion_method == PHASE3_TEMPORARY_FUSION_METHOD
    assert match.user_response == (
        "Your order is Large coffee. The receipt is large coffee. They match."
    )
    assert mismatch.user_response == (
        "Your order is Large coffee. The receipt is Green tea. They do not match."
    )
    concatenated = "Large coffee large coffee"
    assert match.final_response.replace("  ", " ") != match.user_response
    assert match.user_response != concatenated
    assert " ".join(match.user_response.split()) != " ".join(
        "Large coffee large coffee".split()
    )
    assert mismatch.user_response != "Large coffee Green tea"


def test_user_response_is_deterministic_and_skips_llm_fuser(monkeypatch):
    def _forbid_llm(*_args, **_kwargs):
        raise AssertionError("official response path called the LLM fuser")

    monkeypatch.setattr(ResponseFuser, "fuse", _forbid_llm)
    monkeypatch.setattr(ResponseFuser, "_llm_fuse", _forbid_llm)
    graph = _personal_graph()
    outputs = {"personal": {"gate_identifier": "Gate 54"}}
    first = _execute(graph, outputs)
    second = _execute(graph, outputs)
    again = compose_user_response(graph, first.results)
    assert first.user_response == second.user_response == again.text
    assert first.response_fusion_mode == "none"


def test_parallel_response_is_not_whitespace_concatenation():
    graph = _parallel_graph()
    result = _execute(
        graph,
        {
            "order": {"order": "bagel"},
            "receipt": {"receipt": "bagel"},
        },
    )
    raw_values = ("bagel", "bagel")
    assert result.user_response == (
        "Your order is bagel. The receipt is bagel. They match."
    )
    assert result.user_response not in {
        " ".join(raw_values),
        "\n".join(raw_values),
        "\n\n".join(raw_values),
        "".join(raw_values),
    }
    assert result.final_response == "bagel bagel"
    assert result.user_response != result.final_response
