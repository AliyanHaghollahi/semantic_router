"""Phase-5 Step-1 tests: candidate selection and semantic annotation scaffold."""

from __future__ import annotations

from pathlib import Path

import pytest

from tiergraph.enums import OperatorType, QueryType
from tiergraph.planner.annotations import ImplicitResolution
from tiergraph.planner.corpus import (
    DEFAULT_CANDIDATE_SEED,
    PlannerBucket,
    PlannerSemanticAnnotation,
    SemanticAnchorSpan,
    SemanticOperationSpan,
    build_unique_query_pool,
    load_candidates_jsonl,
    load_classification_rows,
    normalize_query_key,
    select_stage_a_candidates,
    semantic_annotation_to_planner_example,
    write_candidates_jsonl,
)
from tiergraph.planner.decode import PlannerDecodeError


ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "dataset" / "training_data.json"
CANDIDATES_PATH = ROOT / "dataset" / "planner" / "stage_a_candidates.jsonl"


def test_deduplicate_training_data_to_700_unique():
    rows = load_classification_rows(TRAIN_PATH)
    assert len(rows) == 1293
    pool = build_unique_query_pool(rows)
    assert len(pool) == 700
    by_label = {}
    for item in pool:
        by_label.setdefault(item.source_classification_label, 0)
        by_label[item.source_classification_label] += 1
    assert by_label == {
        "Personal": 148,
        "Environmental": 425,
        "Mixed": 127,
    }


def test_no_label_conflicts_on_normalized_duplicates():
    rows = load_classification_rows(TRAIN_PATH)
    build_unique_query_pool(rows)

    conflicting = [
        {"query": "Hello World", "label": "Personal"},
        {"query": "  hello   world ", "label": "Mixed"},
    ]
    with pytest.raises(ValueError, match="conflicting classification labels"):
        build_unique_query_pool(conflicting)


def test_original_query_text_preserved():
    rows = [
        {"query": "  Keep Exact   Spacing? ", "label": "Personal"},
        {"query": "keep exact spacing?", "label": "Personal"},
    ]
    pool = build_unique_query_pool(rows)
    assert len(pool) == 1
    assert pool[0].query == "  Keep Exact   Spacing? "
    assert normalize_query_key(pool[0].query) == "keep exact spacing?"


def test_deterministic_candidate_selection_and_expected_pool_counts(tmp_path):
    rows = load_classification_rows(TRAIN_PATH)
    pool = build_unique_query_pool(rows)
    first = select_stage_a_candidates(pool, seed=DEFAULT_CANDIDATE_SEED)
    second = select_stage_a_candidates(pool, seed=DEFAULT_CANDIDATE_SEED)
    assert [item.source_query_id for item in first] == [
        item.source_query_id for item in second
    ]
    assert len(first) == 187
    counts = {
        "Personal": 0,
        "Environmental": 0,
        "Mixed": 0,
    }
    for item in first:
        counts[item.source_classification_label] += 1
        assert item.annotation_status.value == "unreviewed"
        assert item.planner_bucket is None
        assert item.split is None
        assert item.semantic_group_id == item.source_query_id
    assert counts == {"Personal": 30, "Environmental": 30, "Mixed": 127}

    out = tmp_path / "stage_a_candidates.jsonl"
    write_candidates_jsonl(out, first)
    reloaded = load_candidates_jsonl(out)
    assert len(reloaded) == 187
    assert reloaded[0].model_dump(mode="json") == first[0].model_dump(mode="json")


def test_checked_in_candidates_match_deterministic_selection():
    assert CANDIDATES_PATH.is_file(), "run scripts/select_planner_candidates.py"
    rows = load_classification_rows(TRAIN_PATH)
    expected = select_stage_a_candidates(build_unique_query_pool(rows))
    actual = load_candidates_jsonl(CANDIDATES_PATH)
    assert len(actual) == 187
    assert [item.source_query_id for item in actual] == [
        item.source_query_id for item in expected
    ]
    assert [item.query for item in actual] == [item.query for item in expected]


def test_invalid_span_rejected():
    with pytest.raises(ValueError, match="out of bounds|operation span"):
        PlannerSemanticAnnotation.model_validate(
            {
                "source_query_id": "src_test",
                "semantic_group_id": "src_test",
                "query": "short",
                "source_classification_label": "Personal",
                "planner_bucket": "personal",
                "operations": [
                    {
                        "char_start": 0,
                        "char_end": 99,
                        "operator_type": "RETRIEVE_PERSONAL",
                    }
                ],
            }
        )


def test_invalid_owner_rejected():
    with pytest.raises(ValueError, match="owner_operation_index"):
        PlannerSemanticAnnotation.model_validate(
            {
                "source_query_id": "src_test",
                "semantic_group_id": "src_test",
                "query": "Where is my gate?",
                "source_classification_label": "Mixed",
                "planner_bucket": "mixed_implicit",
                "operations": [
                    {
                        "char_start": 0,
                        "char_end": 17,
                        "operator_type": "LOCATE_ENVIRONMENTAL",
                    }
                ],
                "anchors": [
                    {
                        "char_start": 12,
                        "char_end": 16,
                        "normalized_name": "gate",
                        "owner_operation_index": 3,
                        "implicit_resolution": "IMPLICIT_RESOLVE_PERSONAL",
                    }
                ],
            }
        )


def test_structurally_impossible_h7_pair_rejected():
    with pytest.raises(ValueError, match="structurally impossible H7 pair"):
        PlannerSemanticAnnotation.model_validate(
            {
                "source_query_id": "src_test",
                "semantic_group_id": "src_test",
                "query": "Locate this then retrieve that personal fact please.",
                "source_classification_label": "Mixed",
                "planner_bucket": "mixed_sequential",
                "operations": [
                    {
                        "char_start": 0,
                        "char_end": 11,
                        "operator_type": "LOCATE_ENVIRONMENTAL",
                    },
                    {
                        "char_start": 12,
                        "char_end": 40,
                        "operator_type": "RETRIEVE_PERSONAL",
                    },
                ],
                "dependencies": [
                    {
                        "source_operation_index": 0,
                        "target_operation_index": 1,
                    }
                ],
            }
        )


def test_valid_semantic_annotation_to_planner_example():
    annotation = PlannerSemanticAnnotation(
        source_query_id="src_env_001",
        semantic_group_id="src_env_001",
        query="What does this sign say?",
        source_classification_label="Environmental",
        planner_bucket=PlannerBucket.ENVIRONMENTAL,
        operations=(
            SemanticOperationSpan(
                char_start=0,
                char_end=24,
                operator_type=OperatorType.IDENTIFY_ENVIRONMENTAL,
            ),
        ),
    )
    example = semantic_annotation_to_planner_example(annotation)
    assert example.query == annotation.query
    assert example.graph.query_type is QueryType.ENVIRONMENTAL
    assert len(example.planner_labels.operation_spans) == 1
    assert example.planner_labels.operation_spans[0].operator is (
        OperatorType.IDENTIFY_ENVIRONMENTAL
    )
    assert example.metadata["source_query_id"] == "src_env_001"


def test_gate_style_annotation_to_mixed_with_deterministic_implicit_edge():
    query = "Where is my gate?"
    annotation = PlannerSemanticAnnotation(
        source_query_id="src_gate",
        semantic_group_id="src_gate",
        query=query,
        source_classification_label="Mixed",
        planner_bucket=PlannerBucket.MIXED_IMPLICIT,
        operations=(
            SemanticOperationSpan(
                char_start=0,
                char_end=17,
                operator_type=OperatorType.LOCATE_ENVIRONMENTAL,
            ),
        ),
        anchors=(
            SemanticAnchorSpan(
                char_start=12,
                char_end=16,
                normalized_name="gate",
                owner_operation_index=0,
                implicit_resolution=ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL,
            ),
        ),
        dependencies=(),
    )
    example = semantic_annotation_to_planner_example(annotation)
    assert example.graph.query_type is QueryType.MIXED
    operators = {node.operator for node in example.graph.nodes}
    assert OperatorType.LOCATE_ENVIRONMENTAL in operators
    assert OperatorType.RESOLVE_PERSONAL in operators
    assert len(example.graph.edges) == 1
    edge = example.graph.edges[0]
    assert edge.source_node_id.startswith("impl_")
    assert edge.target_node_id.startswith("op_")
    assert example.planner_labels.slot_anchors[0].implicit_node_id == (
        edge.source_node_id
    )


def test_decode_failure_reports_source_query_id():
    annotation = PlannerSemanticAnnotation(
        source_query_id="src_bad",
        semantic_group_id="src_bad",
        query="abc",
        source_classification_label="Personal",
        planner_bucket=PlannerBucket.PERSONAL,
        operations=(
            SemanticOperationSpan(
                char_start=0,
                char_end=3,
                operator_type=OperatorType.RETRIEVE_PERSONAL,
            ),
        ),
        anchors=(
            SemanticAnchorSpan(
                char_start=0,
                char_end=3,
                normalized_name="bad-name",
                owner_operation_index=0,
                implicit_resolution=ImplicitResolution.NONE,
            ),
        ),
    )
    with pytest.raises(PlannerDecodeError, match="src_bad"):
        semantic_annotation_to_planner_example(annotation)
