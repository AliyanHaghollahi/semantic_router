"""Focused tests for free predicted-graph evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from tiergraph.enums import QueryType
from tiergraph.planner.annotation_step_a import (
    DEFAULT_STEP_A_ANNOTATIONS_PATH,
    fingerprint_file,
)
from tiergraph.planner.annotation_step_b import DEFAULT_STEP_B_ANNOTATIONS_PATH
from tiergraph.planner.annotations import PlannerExample
from tiergraph.planner.canonicalize import graphs_exactly_match
from tiergraph.planner.decode import (
    GraphDecoder,
    PlannerPredictions,
    PredictedAnchor,
    PredictedOperation,
)
from tiergraph.planner.encoder import MiniLMFeatureEncoder
from tiergraph.planner.free_eval import (
    classify_decode_error,
    evaluate_free_predictions,
    execution_mode_from_graph,
    predict_batch,
)
from tiergraph.planner.model import PlannerModel
from tiergraph.planner.train import encode_gold_batch


ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "planner" / "where_is_my_gate.json"
)
STEP_A_PATH = ROOT / DEFAULT_STEP_A_ANNOTATIONS_PATH
STEP_B_PATH = ROOT / DEFAULT_STEP_B_ANNOTATIONS_PATH


class _CountingTokenizer:
    pad_token_id = 0
    all_special_ids = (101, 102)
    model_input_names = ("input_ids", "attention_mask")

    def __call__(self, texts, **kwargs):
        input_ids = []
        attention_mask = []
        offset_mapping = []
        for text in texts:
            ids = [101]
            offsets = [(0, 0)]
            cursor = 0
            for piece in text.split(" "):
                if cursor > 0:
                    cursor += 1
                start = cursor
                end = start + len(piece)
                ids.append(10 + len(piece))
                offsets.append((start, end))
                cursor = end
            ids.append(102)
            offsets.append((0, 0))
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
            offset_mapping.append(offsets)
        output = {"input_ids": input_ids, "attention_mask": attention_mask}
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = offset_mapping
        return output


class _CountingTransformer(torch.nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_ids, attention_mask, **kwargs):
        hidden = self.config.hidden_size
        values = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        return SimpleNamespace(
            last_hidden_state=values.repeat(1, 1, hidden) * self.scale
        )


def _make_model(hidden_size: int = 8) -> PlannerModel:
    encoder = MiniLMFeatureEncoder(
        max_length=32,
        tokenizer_loader=lambda _name: _CountingTokenizer(),
        model_loader=lambda _name: _CountingTransformer(hidden_size),
    )
    model = PlannerModel(encoder, hidden_size=hidden_size)
    _ = model.encode(["Where is my gate?"])
    return model


def _load_fixture() -> PlannerExample:
    return PlannerExample.model_validate(
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
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


def test_free_inference_uses_predicted_structures_not_gold_batch():
    model = _make_model()
    example = _load_fixture()
    features, items = predict_batch(model, (example,))
    assert len(items) == 1
    assert isinstance(items[0], PlannerPredictions)
    _features, _views, gold = encode_gold_batch(model, (example,))
    assert gold.op_valid.shape[1] >= 1
    assert features.batch_size == 1
    assert items[0].aux_query_type in {
        QueryType.PERSONAL,
        QueryType.ENVIRONMENTAL,
        QueryType.MIXED,
    }


def test_valid_prediction_decodes_through_existing_graph_decoder():
    example = _load_fixture()
    predictions = _gold_predictions(example)
    decoded = GraphDecoder().decode(
        predictions,
        query=example.query,
        graph_id="free-gold",
    )
    assert graphs_exactly_match(decoded.graph, example.graph)
    metrics = evaluate_free_predictions([(example, predictions)])
    assert metrics.n_examples == 1
    assert metrics.valid_graph_rate == 1.0
    assert metrics.canonical_exact_graph_accuracy == 1.0
    assert metrics.query_type_accuracy == 1.0


def test_malformed_prediction_counts_decoder_failure():
    example = _load_fixture()
    bad = PlannerPredictions(
        operations=(),
        anchors=(),
        dependency_pairs=frozenset(),
        aux_query_type=QueryType.MIXED,
    )
    metrics = evaluate_free_predictions([(example, bad)])
    assert metrics.valid_graph_rate == 0.0
    assert metrics.decode_failure_counts.get("no_operations") == 1
    assert metrics.decode_failure_examples[0]["example_id"] == example.example_id
    assert classify_decode_error("at least one explicit operation is required") == (
        "no_operations"
    )


def test_canonical_exact_match_id_invariant():
    example = _load_fixture()
    predictions = _gold_predictions(example)
    left = GraphDecoder().decode(
        predictions, query=example.query, graph_id="left-id"
    )
    right = GraphDecoder().decode(
        predictions, query=example.query, graph_id="right-id"
    )
    assert left.graph.graph_id != right.graph.graph_id
    assert graphs_exactly_match(left.graph, right.graph)
    metrics = evaluate_free_predictions([(example, predictions)])
    assert metrics.canonical_exact_graph_accuracy == 1.0
    assert execution_mode_from_graph(left.graph) == execution_mode_from_graph(
        example.graph
    )


def test_free_eval_does_not_mutate_frozen_annotations():
    before_a = fingerprint_file(STEP_A_PATH)
    before_b = fingerprint_file(STEP_B_PATH)
    bytes_a = STEP_A_PATH.read_bytes()
    bytes_b = STEP_B_PATH.read_bytes()
    example = _load_fixture()
    evaluate_free_predictions([(example, _gold_predictions(example))])
    evaluate_free_predictions(
        [
            (
                example,
                PlannerPredictions(
                    operations=(),
                    anchors=(),
                    aux_query_type=QueryType.PERSONAL,
                ),
            )
        ]
    )
    assert fingerprint_file(STEP_A_PATH) == before_a
    assert fingerprint_file(STEP_B_PATH) == before_b
    assert STEP_A_PATH.read_bytes() == bytes_a
    assert STEP_B_PATH.read_bytes() == bytes_b


def test_teacher_forced_eval_still_imports():
    from tiergraph.planner.train import evaluate_checkpoint, evaluate_examples

    assert callable(evaluate_examples)
    assert callable(evaluate_checkpoint)
