"""Classification checks for TRAIN semantic-failure attribution.

These tests build graphs in memory. They do not load a checkpoint, DEV, or TEST.
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
from tiergraph.pilot.train_error_attribution import (
    H1_QUERY_TYPE,
    H2_OPERATION_STRUCTURE,
    H3_OPERATOR_TYPE,
    H4_H5_ANCHOR_TARGET,
    H6_TIER_OWNERSHIP,
    H7_DEPENDENCY,
    PRIMARY_ORDER,
    attribute_failure,
    make_executor,
    oracle_repairs,
    summarize_attributions,
)
from tiergraph.planner.tasks import render_answer_task

ROOT = Path(__file__).resolve().parents[1]


def _node(**overrides) -> SemanticNode:
    values = {
        "node_id": "personal",
        "semantic_type": NodeSemanticType.PERSONAL,
        "operator": OperatorType.RESOLVE_PERSONAL,
        "tier": Tier.EDGE,
        "task": render_answer_task(operator=OperatorType.RESOLVE_PERSONAL, base_name="gate"),
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


def test_wrong_operator_is_h3_and_only_an_operator_repair_fixes_it():
    gold = _graph((_node(),))
    learned = _graph(
        (
            _node(
                node_id="pred",
                operator=OperatorType.RETRIEVE_PERSONAL,
                task=render_answer_task(
                    operator=OperatorType.RETRIEVE_PERSONAL,
                    base_name="gate",
                ),
                produced_outputs={"gate_fact": SlotType.PERSONAL_FACT},
            ),
        ),
        graph_id="pred",
    )
    attributed = attribute_failure(gold, learned, learned_decoded=True)
    assert H3_OPERATOR_TYPE in attributed["categories"]
    assert H2_OPERATION_STRUCTURE not in attributed["categories"]
    assert H4_H5_ANCHOR_TARGET not in attributed["categories"]
    assert attributed["primary"] == H3_OPERATOR_TYPE
    executor = make_executor()
    repairs = oracle_repairs(gold, learned, attributed["pairs"], executor)
    assert repairs["operator_type"] is True
    assert repairs["anchor_target"] is False
    assert repairs["tier_ownership"] is False
    assert repairs["dependency"] is False


def test_compound_mismatch_counts_in_each_prevalence_category():
    gold = _graph(
        (
            _node(node_id="order", task=render_answer_task(operator=OperatorType.RETRIEVE_PERSONAL, base_name="order"), operator=OperatorType.RETRIEVE_PERSONAL, produced_outputs={"order_fact": SlotType.PERSONAL_FACT}),
            _node(
                node_id="receipt",
                semantic_type=NodeSemanticType.ENVIRONMENTAL,
                operator=OperatorType.DESCRIBE_ENVIRONMENT,
                tier=Tier.FOG,
                task=render_answer_task(
                    operator=OperatorType.DESCRIBE_ENVIRONMENT,
                    base_name="receipt",
                ),
                produced_outputs={"receipt_scene": SlotType.SCENE_DESCRIPTION},
            ),
            _node(
                node_id="fuse",
                semantic_type=NodeSemanticType.CONTROL,
                operator=OperatorType.FUSE,
                tier=Tier.EDGE,
                task="Fuse the terminal answers",
                required_inputs={
                    "order_fact": SlotType.PERSONAL_FACT,
                    "receipt_scene": SlotType.SCENE_DESCRIPTION,
                },
                produced_outputs={"response": SlotType.FINAL_RESPONSE},
            ),
        ),
        query_type=QueryType.MIXED,
        original_query="Does my receipt match?",
        edges=(
            DependencyEdge(
                source_node_id="order",
                source_slot="order_fact",
                target_node_id="fuse",
                target_slot="order_fact",
            ),
            DependencyEdge(
                source_node_id="receipt",
                source_slot="receipt_scene",
                target_node_id="fuse",
                target_slot="receipt_scene",
            ),
        ),
    )
    learned = _graph(
        (
            _node(
                node_id="order",
                operator=OperatorType.RESOLVE_PERSONAL,
                task=render_answer_task(operator=OperatorType.RESOLVE_PERSONAL, base_name="order"),
                produced_outputs={"order_identifier": SlotType.RESOLVED_REFERENCE},
            ),
        ),
        graph_id="pred",
        query_type=QueryType.PERSONAL,
        original_query="Does my receipt match?",
    )
    attributed = attribute_failure(gold, learned, learned_decoded=True)
    assert H2_OPERATION_STRUCTURE in attributed["categories"]
    assert H3_OPERATOR_TYPE in attributed["categories"]
    assert H4_H5_ANCHOR_TARGET in attributed["categories"]
    assert H1_QUERY_TYPE in attributed["categories"]
    assert attributed["primary"] == H2_OPERATION_STRUCTURE
    summary = summarize_attributions(
        [
            {
                "semantic_success": False,
                "final_bucket": "MIXED_PARALLEL",
                "op_group": "multi_op",
                "categories": attributed["categories"],
                "primary": attributed["primary"],
                "oracle_applicable": False,
                "oracle": None,
                "stage_a_id": "sa_mix",
                "query": gold.original_query,
                "gold_graph_summary": {"nodes": []},
                "learned_graph_summary": {"nodes": []},
            }
        ]
    )
    prevalence = summary["error_prevalence"]
    assert prevalence[H2_OPERATION_STRUCTURE] == 1
    assert prevalence[H3_OPERATOR_TYPE] == 1
    assert prevalence[H4_H5_ANCHOR_TARGET] == 1
    assert prevalence[H1_QUERY_TYPE] == 1
    assert summary["primary_causes"][H2_OPERATION_STRUCTURE] == 1
    assert summary["error_cardinality"]["three_or_more"] == 1


def test_missing_dependency_is_h7_and_primary_causes_sum_to_failures():
    resolve = _node(node_id="resolve")
    navigate = _node(
        node_id="navigate",
        semantic_type=NodeSemanticType.ENVIRONMENTAL,
        operator=OperatorType.NAVIGATE_TO,
        tier=Tier.FOG,
        task=render_answer_task(operator=OperatorType.NAVIGATE_TO, base_name="gate"),
        required_inputs={"gate_identifier": SlotType.RESOLVED_REFERENCE},
        produced_outputs={"gate_navigation": SlotType.NAVIGATION_INSTRUCTION},
    )
    edge = DependencyEdge(
        source_node_id="resolve",
        source_slot="gate_identifier",
        target_node_id="navigate",
        target_slot="gate_identifier",
        transfer_policy=TransferPolicy.MINIMAL_REFERENCE,
    )
    gold = _graph(
        (resolve, navigate),
        query_type=QueryType.MIXED,
        original_query="How do I get to my gate?",
        edges=(edge,),
    )
    learned = _graph(
        (
            resolve.model_copy(update={"node_id": "p_resolve", "required_inputs": {}}),
            navigate.model_copy(update={"node_id": "p_navigate", "required_inputs": {}}),
        ),
        graph_id="pred",
        query_type=QueryType.MIXED,
        original_query="How do I get to my gate?",
        edges=(),
    )
    missing_edge = attribute_failure(gold, learned, learned_decoded=True)
    assert missing_edge["categories"] == [H7_DEPENDENCY]
    assert missing_edge["primary"] == H7_DEPENDENCY
    wrong_tier = attribute_failure(
        gold,
        _graph(
            (
                resolve.model_copy(update={"node_id": "p_resolve"}),
                navigate.model_copy(
                    update={
                        "node_id": "p_navigate",
                        "tier": Tier.EDGE,
                        "required_inputs": {"gate_identifier": SlotType.RESOLVED_REFERENCE},
                    }
                ),
            ),
            graph_id="tier",
            query_type=QueryType.MIXED,
            original_query=gold.original_query,
            edges=(
                DependencyEdge(
                    source_node_id="p_resolve",
                    source_slot="gate_identifier",
                    target_node_id="p_navigate",
                    target_slot="gate_identifier",
                    transfer_policy=TransferPolicy.DIRECT,
                ),
            ),
        ),
        learned_decoded=True,
    )
    assert H6_TIER_OWNERSHIP in wrong_tier["categories"]
    rows = [
        _failure_row("sa_h7", missing_edge, "MIXED_SEQUENTIAL", "multi_op"),
        _failure_row("sa_h3", attribute_failure(
            _graph((_node(),)),
            _graph(
                (
                    _node(
                        node_id="pred",
                        operator=OperatorType.RETRIEVE_PERSONAL,
                        task=render_answer_task(
                            operator=OperatorType.RETRIEVE_PERSONAL,
                            base_name="gate",
                        ),
                        produced_outputs={"gate_fact": SlotType.PERSONAL_FACT},
                    ),
                ),
                graph_id="pred",
            ),
            learned_decoded=True,
        ), "Personal", "single_op"),
        {
            "semantic_success": True,
            "final_bucket": "Personal",
            "op_group": "single_op",
            "categories": [],
            "primary": None,
            "oracle_applicable": False,
            "oracle": None,
            "stage_a_id": "sa_ok",
            "query": "ok",
            "gold_graph_summary": {},
            "learned_graph_summary": {},
        },
    ]
    summary = summarize_attributions(rows)
    assert summary["n_total"] == 3
    assert summary["n_semantic_success"] == 1
    assert summary["n_semantic_failure"] == 2
    assert sum(summary["primary_causes"].values()) == summary["n_semantic_failure"]
    assert set(summary["primary_causes"]) == set(PRIMARY_ORDER)
    assert summary["by_final_bucket"]["Personal"]["semantic_success"] == 1


def test_invalid_graph_precedes_other_categories_and_blocks_alignment():
    gold = _graph((_node(),))
    attributed = attribute_failure(gold, None, learned_decoded=False)
    assert attributed["primary"] == "INVALID_GRAPH"
    assert attributed["alignment_safe"] is False
    assert H1_QUERY_TYPE not in attributed["categories"]


def test_analysis_script_uses_train_membership_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script_path = ROOT / "scripts" / "analyze_planner_train_failures.py"
    source = script_path.read_text(encoding="utf-8")
    module_source = (
        ROOT / "tiergraph" / "pilot" / "train_error_attribution.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("load_and_split_stage_a_v3", "n_test", "TEST annotations"):
        assert forbidden not in source
        assert forbidden not in module_source
    spec = importlib.util.spec_from_file_location(
        "analyze_planner_train_failures",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    driver = analysis._gold_driver()
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


def _failure_row(example_id, attributed, bucket, group):
    return {
        "semantic_success": False,
        "final_bucket": bucket,
        "op_group": group,
        "categories": attributed["categories"],
        "primary": attributed["primary"],
        "oracle_applicable": False,
        "oracle": None,
        "stage_a_id": example_id,
        "query": example_id,
        "gold_graph_summary": {"nodes": []},
        "learned_graph_summary": {"nodes": []},
    }
