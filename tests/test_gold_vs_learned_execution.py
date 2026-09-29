"""Fixture checks for gold versus learned stub execution.

These tests do not load a checkpoint, DEV annotations, or TEST.
"""

from __future__ import annotations

import importlib.util
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
from tiergraph.graph import DependencyEdge, ExecutionGraph, SemanticNode
from tiergraph.pilot.gold_learned_harness import (
    METRIC_NAMES,
    GoldExampleView,
    compare_pair,
    make_stub_executor,
    operator_stub_runner,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_gold_vs_learned_execution",
    ROOT / "scripts" / "run_gold_vs_learned_execution.py",
)
assert _SPEC is not None and _SPEC.loader is not None
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def _node(**overrides) -> SemanticNode:
    values = {
        "node_id": "personal",
        "semantic_type": NodeSemanticType.PERSONAL,
        "operator": OperatorType.RESOLVE_PERSONAL,
        "tier": Tier.EDGE,
        "task": "Resolve the personal reference",
        "required_inputs": {},
        "produced_outputs": {"gate_identifier": SlotType.RESOLVED_REFERENCE},
    }
    values.update(overrides)
    return SemanticNode(**values)


def _graph(nodes, **overrides) -> ExecutionGraph:
    values = {
        "graph_id": "gold",
        "original_query": "What is my gate number?",
        "query_type": QueryType.PERSONAL,
        "nodes": nodes,
        "edges": (),
    }
    values.update(overrides)
    return ExecutionGraph(**values)


def _personal() -> GoldExampleView:
    graph = _graph((_node(),))
    return GoldExampleView(
        example_id="sa_personal",
        query=graph.original_query,
        graph=graph,
        final_bucket="Personal",
    )


def _sequential() -> ExecutionGraph:
    resolve = _node(node_id="resolve")
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
        (resolve, navigate),
        graph_id="gold-seq",
        original_query="How do I get to my gate?",
        query_type=QueryType.MIXED,
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


def _parallel(order: tuple[str, str] = ("order", "receipt")) -> ExecutionGraph:
    nodes = {
        "order": _node(
            node_id="order",
            operator=OperatorType.RETRIEVE_PERSONAL,
            task="Retrieve the order",
            produced_outputs={"order": SlotType.PERSONAL_FACT},
        ),
        "receipt": _node(
            node_id="receipt",
            semantic_type=NodeSemanticType.ENVIRONMENTAL,
            operator=OperatorType.DESCRIBE_ENVIRONMENT,
            tier=Tier.FOG,
            task="Read the receipt",
            produced_outputs={"receipt": SlotType.SCENE_DESCRIPTION},
        ),
        "fuse": _node(
            node_id="fuse",
            semantic_type=NodeSemanticType.CONTROL,
            operator=OperatorType.FUSE,
            tier=Tier.EDGE,
            task="Combine the branches",
            required_inputs={
                "order": SlotType.PERSONAL_FACT,
                "receipt": SlotType.SCENE_DESCRIPTION,
            },
            produced_outputs={"response": SlotType.FINAL_RESPONSE},
        ),
    }
    return _graph(
        tuple(nodes[name] for name in (*order, "fuse")),
        graph_id="gold-par",
        original_query="Does this receipt match what I ordered?",
        query_type=QueryType.MIXED,
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


def test_identical_graphs_match_on_the_shared_stub():
    example = _personal()
    learned = example.graph.model_copy(update={"graph_id": "pred::sa_personal"})
    executor = make_stub_executor()
    record = compare_pair(example, learned, predict_ms=1.5, executor=executor)
    again = compare_pair(example, learned, predict_ms=1.5, executor=executor)
    assert record.graph_valid is True
    assert record.exact_graph is True
    assert record.tier_routing is True
    assert record.gold_execution_success is True
    assert record.learned_execution_success is True
    assert record.response_semantic_match is True
    assert record.response_exact_string_match is True
    assert record.gold_user_response == record.learned_user_response == again.gold_user_response
    assert record.op_group == "single_op"
    assert record.learned_predict_execute_ms is not None
    assert record.learned_predict_execute_ms >= 1.5


def test_sequential_dependency_reaches_one_user_response():
    graph = _sequential()
    example = GoldExampleView(
        example_id="sa_seq",
        query=graph.original_query,
        graph=graph,
        final_bucket="MIXED_SEQUENTIAL",
    )
    learned = graph.model_copy(
        update={
            "graph_id": "pred::sa_seq",
            "nodes": tuple(
                node.model_copy(update={"node_id": f"p_{node.node_id}"})
                for node in graph.nodes
            ),
            "edges": (
                DependencyEdge(
                    source_node_id="p_resolve",
                    source_slot="gate_identifier",
                    target_node_id="p_navigate",
                    target_slot="gate_identifier",
                    transfer_policy=TransferPolicy.MINIMAL_REFERENCE,
                ),
            ),
        }
    )
    record = compare_pair(example, learned, predict_ms=0.0)
    assert record.exact_graph is True
    assert record.op_group == "multi_op"
    assert record.response_semantic_match is True
    assert record.response_exact_string_match is True
    assert record.gold_user_response is not None
    assert "To get there" in record.gold_user_response


def test_node_order_can_change_the_response_without_breaking_exact_graph():
    gold = _parallel(("order", "receipt"))
    learned = _parallel(("receipt", "order"))
    example = GoldExampleView(
        example_id="sa_par",
        query=gold.original_query,
        graph=gold,
        final_bucket="MIXED_PARALLEL",
    )
    record = compare_pair(example, learned, predict_ms=0.0)
    assert record.exact_graph is True
    assert record.tier_routing is True
    assert record.response_semantic_match is True
    assert record.response_exact_string_match is False
    assert record.gold_user_response != record.learned_user_response


def test_missing_learned_graph_does_not_match_a_response():
    example = _personal()
    record = compare_pair(example, None, predict_ms=4.0)
    assert record.graph_valid is False
    assert record.exact_graph is False
    assert record.tier_routing is False
    assert record.gold_execution_success is True
    assert record.learned_execution_success is False
    assert record.response_semantic_match is False
    assert record.response_exact_string_match is False
    assert record.learned_user_response is None
    assert record.learned_predict_execute_ms == 4.0


def test_unfused_parallel_prediction_fails_execution():
    example = _personal()
    learned = _graph(
        (
            _node(node_id="order", operator=OperatorType.RETRIEVE_PERSONAL, produced_outputs={"order": SlotType.PERSONAL_FACT}),
            _node(
                node_id="where",
                semantic_type=NodeSemanticType.ENVIRONMENTAL,
                operator=OperatorType.LOCATE_ENVIRONMENTAL,
                tier=Tier.FOG,
                task="Locate",
                produced_outputs={"location": SlotType.LOCATION},
            ),
        ),
        graph_id="pred-bad",
        original_query="mixed without fusion",
        query_type=QueryType.MIXED,
    )
    record = compare_pair(example, learned, predict_ms=0.2)
    assert record.graph_valid is True
    assert record.learned_execution_success is False
    assert record.gold_execution_success is True
    assert record.response_semantic_match is False
    assert record.response_exact_string_match is False
    assert record.exact_graph is False
    assert record.tier_routing is False


def test_summary_slices_by_bucket_and_op_group():
    personal = compare_pair(_personal(), _personal().graph, predict_ms=2.0)
    sequential = compare_pair(
        GoldExampleView("sa_seq", _sequential().original_query, _sequential(), "MIXED_SEQUENTIAL"),
        None,
        predict_ms=3.0,
    )
    summary = summarize([personal, sequential])
    assert set(summary["overall"]) >= set(METRIC_NAMES)
    assert summary["overall"]["n"] == 2
    assert summary["overall"]["graph_valid_rate"] == 0.5
    assert summary["overall"]["response_semantic_match_rate"] == 0.5
    assert summary["by_final_bucket"]["Personal"]["response_semantic_match_rate"] == 1.0
    assert summary["by_op_group"]["single_op"]["n"] == 1
    assert summary["by_op_group"]["multi_op"]["graph_valid_rate"] == 0.0
    assert summary["fusion_plan"] is None
    assert summary["response_criterion"] == "response_semantic_match"
    assert summary["secondary_response_criterion"] == "response_exact_string_match"


def test_stub_changes_when_the_anchor_or_predecessor_changes():
    gate_54 = _node(
        node_id="nav54",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.NAVIGATE_TO,
        tier=Tier.FOG,
        task="Navigate to the gate_54",
        produced_outputs={"gate_54_navigation": SlotType.NAVIGATION_INSTRUCTION},
    )
    gate_21 = gate_54.model_copy(
        update={
            "node_id": "nav21",
            "task": "Navigate to the gate_21",
            "produced_outputs": {"gate_21_navigation": SlotType.NAVIGATION_INSTRUCTION},
        }
    )
    same_anchor_other_id = gate_54.model_copy(update={"node_id": "nav54b"})
    assert operator_stub_runner(gate_54, {}, None) != operator_stub_runner(gate_21, {}, None)
    assert operator_stub_runner(gate_54, {}, None) == operator_stub_runner(
        same_anchor_other_id, {}, None
    )

    from tiergraph.executor import BoundInput

    bound_54 = {
        "gate_identifier": BoundInput(
            slot_name="gate_identifier",
            slot_type=SlotType.RESOLVED_REFERENCE,
            value="Gate 54",
            source_node_id="left",
            source_slot="gate_identifier",
            transfer_policy=TransferPolicy.MINIMAL_REFERENCE,
            source_tier=Tier.EDGE,
            target_tier=Tier.FOG,
        )
    }
    bound_21 = {
        "gate_identifier": BoundInput(
            slot_name="gate_identifier",
            slot_type=SlotType.RESOLVED_REFERENCE,
            value="Gate 21",
            source_node_id="right",
            source_slot="gate_identifier",
            transfer_policy=TransferPolicy.MINIMAL_REFERENCE,
            source_tier=Tier.EDGE,
            target_tier=Tier.FOG,
        )
    }
    shared = _node(
        node_id="navigate",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.NAVIGATE_TO,
        tier=Tier.FOG,
        task="Navigate to the resolved gate",
        required_inputs={"gate_identifier": SlotType.RESOLVED_REFERENCE},
        produced_outputs={"directions": SlotType.NAVIGATION_INSTRUCTION},
    )
    assert operator_stub_runner(shared, bound_54, None) != operator_stub_runner(
        shared, bound_21, None
    )


def test_wrong_anchor_fails_semantic_match():
    gold_node = _node(task="Resolve the user's gate_54 identifier")
    learned_node = _node(
        node_id="other",
        task="Resolve the user's gate_21 identifier",
    )
    gold = _graph((gold_node,))
    learned = _graph((learned_node,), graph_id="pred-anchor")
    example = GoldExampleView("sa_gate", gold.original_query, gold, "Personal")
    record = compare_pair(example, learned, predict_ms=0.0)
    assert record.gold_execution_success is True
    assert record.learned_execution_success is True
    assert record.response_semantic_match is False


def test_sequential_predecessor_value_error_fails_semantic_match():
    gold = _sequential()
    wrong = _sequential()
    wrong = wrong.model_copy(
        update={
            "nodes": tuple(
                node.model_copy(update={"task": "Resolve the user's gate_21 identifier"})
                if node.node_id == "resolve"
                else node
                for node in wrong.nodes
            )
        }
    )
    example = GoldExampleView("sa_seq", gold.original_query, gold, "MIXED_SEQUENTIAL")
    record = compare_pair(example, wrong, predict_ms=0.0)
    assert record.gold_execution_success is True
    assert record.learned_execution_success is True
    assert record.response_semantic_match is False
    assert record.gold_user_response != record.learned_user_response


def test_harness_does_not_call_a_model_or_retain_test_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = (ROOT / "tiergraph" / "pilot" / "gold_learned_harness.py").read_text(
        encoding="utf-8"
    )
    script = (ROOT / "scripts" / "run_gold_vs_learned_execution.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("httpx", "ollama", "ResponseFuser", "use_llm_fusion", "openai"):
        assert forbidden not in source
        assert forbidden not in script
    split = tmp_path / "split.jsonl"
    split.write_text(
        "\n".join(
            [
                '{"split": "train", "stage_a_id": "sa_train"}',
                '{"split": "dev", "stage_a_id": "sa_dev"}',
                '{"split": "test", "stage_a_id": "sa_secret"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(driver, "STAGE_A_V3_TRAIN_SIZE", 1)
    assert driver.train_ids(split) == ["sa_train"]
    record = compare_pair(_personal(), _personal().graph, predict_ms=0.0)
    assert record.response_semantic_match is True


def test_dry_run_prints_protocol_without_model_or_execution(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run loaded examples, a model, or executed graphs")

    monkeypatch.setattr(driver, "load_train_gold_examples", _forbidden)
    monkeypatch.setattr(driver, "run_comparison", _forbidden)
    driver.run(["--dry-run"])
    captured = capsys.readouterr().out
    assert "CHECKPOINT artifacts/planner_h4_dev_confirmation/final.pt" in captured
    assert "N_TRAIN 384" in captured
    assert "METRICS " + " ".join(METRIC_NAMES) in captured
    assert "FUSION_PLAN none" in captured
    assert "DEV_USED false" in captured
    assert "TEST_USED false" in captured
    assert not (ROOT / "artifacts" / "gold_vs_learned_execution").exists()
