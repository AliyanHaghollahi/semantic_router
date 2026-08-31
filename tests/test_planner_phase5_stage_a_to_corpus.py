"""Tests for Stage-A Step-A+B → PlannerSemanticAnnotation → PlannerExample."""

from __future__ import annotations

from pathlib import Path

import pytest

from tiergraph.enums import OperatorType, QueryType
from tiergraph.planner.annotation_step_a import (
    DEFAULT_STEP_A_ANNOTATIONS_PATH,
    EXPECTED_STAGE_A_COUNT,
    fingerprint_file,
    load_step_a_annotations,
)
from tiergraph.planner.annotation_step_b import (
    DEFAULT_STEP_B_ANNOTATIONS_PATH,
    load_step_b_annotations,
)
from tiergraph.planner.annotations import ImplicitResolution
from tiergraph.planner.corpus import PlannerBucket
from tiergraph.planner.stage_a_to_corpus import (
    count_explicit_h7_edges,
    derive_anchor_normalized_name,
    load_stage_a_planner_examples,
    step_ab_to_planner_example,
    step_ab_to_semantic_annotation,
    validate_step_ab_linkage,
)


ROOT = Path(__file__).resolve().parent.parent
STEP_A_PATH = ROOT / DEFAULT_STEP_A_ANNOTATIONS_PATH
STEP_B_PATH = ROOT / DEFAULT_STEP_B_ANNOTATIONS_PATH


@pytest.fixture(scope="module")
def step_a_by_id():
    records = load_step_a_annotations(STEP_A_PATH)
    return {item.stage_a_id: item for item in records}


@pytest.fixture(scope="module")
def step_b_by_id():
    records = load_step_b_annotations(STEP_B_PATH)
    return {item.stage_a_id: item for item in records}


def test_linkage_mismatch_rejected(step_a_by_id, step_b_by_id):
    step_a = step_a_by_id["sa_0001"]
    step_b = step_b_by_id["sa_0002"]
    with pytest.raises(ValueError, match="stage_a_id mismatch"):
        validate_step_ab_linkage(step_a, step_b)
    with pytest.raises(ValueError, match="stage_a_id mismatch"):
        step_ab_to_semantic_annotation(step_a, step_b)


def test_personal_retrieve_maps_correctly(step_a_by_id, step_b_by_id):
    step_a = step_a_by_id["sa_0001"]
    step_b = step_b_by_id["sa_0001"]
    assert step_a.final_bucket == "Personal"
    example = step_ab_to_planner_example(step_a, step_b)
    assert example.example_id == "sa_0001"
    assert example.graph.query_type is QueryType.PERSONAL
    assert example.planner_labels.query_type is QueryType.PERSONAL
    ops = example.planner_labels.operation_spans
    assert len(ops) == 1
    assert ops[0].operator is OperatorType.RETRIEVE_PERSONAL
    assert ops[0].start == step_a.operations[0].char_start
    assert ops[0].end == step_a.operations[0].char_end
    assert example.metadata["stage_a_id"] == "sa_0001"
    assert example.metadata["final_bucket"] == "Personal"
    assert example.metadata["semantic_group"] == step_a.semantic_group


def test_mixed_implicit_synthesizes_resolver_via_decoder(step_a_by_id, step_b_by_id):
    step_a = step_a_by_id["sa_0049"]
    step_b = step_b_by_id["sa_0049"]
    annotation = step_ab_to_semantic_annotation(step_a, step_b)
    assert annotation.planner_bucket is PlannerBucket.MIXED_IMPLICIT
    assert all(
        op.operator_type is not OperatorType.RESOLVE_PERSONAL
        for op in annotation.operations
    )
    assert any(
        anchor.implicit_resolution is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
        for anchor in annotation.anchors
    )
    example = step_ab_to_planner_example(step_a, step_b)
    assert example.graph.query_type is QueryType.MIXED
    operators = {node.operator for node in example.graph.nodes}
    assert OperatorType.IDENTIFY_ENVIRONMENTAL in operators
    assert OperatorType.RESOLVE_PERSONAL in operators
    assert all(
        span.operator is not OperatorType.RESOLVE_PERSONAL
        for span in example.planner_labels.operation_spans
    )
    impl_anchors = [
        anchor
        for anchor in example.planner_labels.slot_anchors
        if anchor.implicit_resolution is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
    ]
    assert len(impl_anchors) == 1
    assert impl_anchors[0].implicit_node_id is not None
    assert impl_anchors[0].implicit_node_id.startswith("impl_")


def test_mixed_parallel_no_h7(step_a_by_id, step_b_by_id):
    step_a = step_a_by_id["sa_0074"]
    step_b = step_b_by_id["sa_0074"]
    assert step_b.dependencies == ()
    example = step_ab_to_planner_example(step_a, step_b)
    assert example.graph.query_type is QueryType.MIXED
    assert count_explicit_h7_edges(example) == 0
    operators = {node.operator for node in example.graph.nodes}
    assert OperatorType.DESCRIBE_ENVIRONMENT in operators
    assert OperatorType.RETRIEVE_PERSONAL in operators


def test_mixed_sequential_h7_survives(step_a_by_id, step_b_by_id):
    step_a = step_a_by_id["sa_0101"]
    step_b = step_b_by_id["sa_0101"]
    assert len(step_b.dependencies) == 1
    assert step_b.dependencies[0].source_operation_index == 0
    assert step_b.dependencies[0].target_operation_index == 1
    example = step_ab_to_planner_example(step_a, step_b)
    assert example.graph.query_type is QueryType.MIXED
    assert count_explicit_h7_edges(example) == 1
    explicit_ids = {
        node.node_id
        for node in example.graph.nodes
        if node.operator
        in {
            OperatorType.IDENTIFY_ENVIRONMENTAL,
            OperatorType.LOCATE_ENVIRONMENTAL,
        }
    }
    h7 = [
        edge
        for edge in example.graph.edges
        if edge.source_node_id in explicit_ids and edge.target_node_id in explicit_ids
    ]
    assert len(h7) == 1
    source = next(
        node
        for node in example.graph.nodes
        if node.node_id == h7[0].source_node_id
    )
    target = next(
        node
        for node in example.graph.nodes
        if node.node_id == h7[0].target_node_id
    )
    assert source.operator is OperatorType.IDENTIFY_ENVIRONMENTAL
    assert target.operator is OperatorType.LOCATE_ENVIRONMENTAL


def test_same_owner_distinct_normalized_names(step_a_by_id, step_b_by_id):
    step_a = step_a_by_id["sa_0049"]
    step_b = step_b_by_id["sa_0049"]
    annotation = step_ab_to_semantic_annotation(step_a, step_b)
    names = [anchor.normalized_name for anchor in annotation.anchors]
    assert names == [
        derive_anchor_normalized_name(anchor.text) for anchor in step_a.anchors
    ]
    assert names[0] != names[1]
    # Literal surfaces normalize without stripping possessives/determiners.
    assert names == ["this_snack", "my_allergies"]
    example = step_ab_to_planner_example(step_a, step_b)
    identify = next(
        node
        for node in example.graph.nodes
        if node.operator is OperatorType.IDENTIFY_ENVIRONMENTAL
    )
    resolve = next(
        node
        for node in example.graph.nodes
        if node.operator is OperatorType.RESOLVE_PERSONAL
    )
    # Principal from NONE anchor; IMPLICIT keeps its own base.
    assert "this_snack_fact" in identify.produced_outputs
    assert "my_allergies_identifier" in resolve.produced_outputs
    assert "my_allergies_identifier" in identify.required_inputs


def test_full_checked_in_corpus_conversion():
    step_a_before = fingerprint_file(STEP_A_PATH)
    step_b_before = fingerprint_file(STEP_B_PATH)
    step_a_bytes = STEP_A_PATH.read_bytes()
    step_b_bytes = STEP_B_PATH.read_bytes()

    examples = load_stage_a_planner_examples(STEP_A_PATH, STEP_B_PATH)
    assert len(examples) == EXPECTED_STAGE_A_COUNT
    assert [example.example_id for example in examples] == [
        f"sa_{index:04d}" for index in range(1, EXPECTED_STAGE_A_COUNT + 1)
    ]

    h1_counts = {
        QueryType.PERSONAL: 0,
        QueryType.ENVIRONMENTAL: 0,
        QueryType.MIXED: 0,
    }
    for example in examples:
        h1_counts[example.graph.query_type] += 1
        # PlannerExample construction already graph-validates.
        assert example.planner_labels.query_type is example.graph.query_type
    assert h1_counts[QueryType.PERSONAL] == 24
    assert h1_counts[QueryType.ENVIRONMENTAL] == 24
    assert h1_counts[QueryType.MIXED] == 72

    step_b_records = load_step_b_annotations(STEP_B_PATH)
    gold_h7 = sum(len(item.dependencies) for item in step_b_records)
    assert gold_h7 == 15
    decoded_h7 = sum(count_explicit_h7_edges(example) for example in examples)
    assert decoded_h7 == 15

    assert fingerprint_file(STEP_A_PATH) == step_a_before
    assert fingerprint_file(STEP_B_PATH) == step_b_before
    assert STEP_A_PATH.read_bytes() == step_a_bytes
    assert STEP_B_PATH.read_bytes() == step_b_bytes
