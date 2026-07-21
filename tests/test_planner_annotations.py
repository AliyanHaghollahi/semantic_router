"""Tests for the isolated TierGraph planner annotation contract."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from tiergraph import (
    ExecutionGraph,
    FusionStrategy,
    NodeSemanticType,
    OperatorType,
    QueryType,
    SlotType,
    Tier,
    TransferPolicy,
)
from tiergraph.planner import (
    ImplicitResolution,
    PlannerExample,
    PlannerRecordValidationError,
    load_planner_jsonl,
    load_planner_record,
    validate_planner_records,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "planner"
    / "where_is_my_gate.json"
)


def _gate_data() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _single_environmental_data(
    query: str = "Read this sign",
    anchor_text: str = "sign",
) -> dict:
    anchor_start = query.index(anchor_text)
    return {
        "example_id": "environmental-sign-001",
        "query": query,
        "graph": {
            "schema_version": "1.0",
            "graph_id": "environmental-sign-001",
            "original_query": query,
            "query_type": "Environmental",
            "nodes": [
                {
                    "node_id": "q1",
                    "semantic_type": "environmental",
                    "operator": "IDENTIFY_ENVIRONMENTAL",
                    "tier": "fog",
                    "task": "Identify the sign",
                    "required_inputs": {},
                    "produced_outputs": {
                        "sign_fact": "ENVIRONMENTAL_FACT"
                    },
                    "status": "pending",
                    "metadata": {},
                }
            ],
            "edges": [],
            "metadata": {},
        },
        "fusion_plan": None,
        "planner_labels": {
            "query_type": "Environmental",
            "operation_spans": [
                {
                    "node_id": "q1",
                    "semantic_type": "environmental",
                    "start": 0,
                    "end": len(query),
                    "operator": "IDENTIFY_ENVIRONMENTAL",
                }
            ],
            "slot_anchors": [
                {
                    "anchor_id": "a1",
                    "start": anchor_start,
                    "end": anchor_start + len(anchor_text),
                    "text": anchor_text,
                    "normalized_name": anchor_text.casefold(),
                    "owner_node_id": "q1",
                    "implicit_resolution": "NONE",
                    "implicit_node_id": None,
                }
            ],
        },
        "metadata": {},
    }


def _multi_sink_data(include_fuse: bool = True) -> dict:
    query = "Identify this sign and describe this room"
    split = query.index(" and ")
    right_start = split + len(" and ")
    nodes = [
        {
            "node_id": "q1",
            "semantic_type": "environmental",
            "operator": "IDENTIFY_ENVIRONMENTAL",
            "tier": "fog",
            "task": "Identify this sign",
            "required_inputs": {},
            "produced_outputs": {"sign_fact": "ENVIRONMENTAL_FACT"},
            "status": "pending",
            "metadata": {},
        },
        {
            "node_id": "q2",
            "semantic_type": "environmental",
            "operator": "DESCRIBE_ENVIRONMENT",
            "tier": "fog",
            "task": "Describe this room",
            "required_inputs": {},
            "produced_outputs": {"room_scene": "SCENE_DESCRIPTION"},
            "status": "pending",
            "metadata": {},
        },
    ]
    edges = []
    fusion_plan = None
    if include_fuse:
        required_slots = {
            "q1__sign_fact": "ENVIRONMENTAL_FACT",
            "q2__room_scene": "SCENE_DESCRIPTION",
        }
        nodes.append(
            {
                "node_id": "fusion",
                "semantic_type": "control",
                "operator": "FUSE",
                "tier": "edge",
                "task": "Fuse the terminal environmental answers",
                "required_inputs": required_slots,
                "produced_outputs": {"response": "FINAL_RESPONSE"},
                "status": "pending",
                "metadata": {},
            }
        )
        edges = [
            {
                "source_node_id": "q1",
                "source_slot": "sign_fact",
                "target_node_id": "fusion",
                "target_slot": "q1__sign_fact",
                "transfer_policy": "direct",
            },
            {
                "source_node_id": "q2",
                "source_slot": "room_scene",
                "target_node_id": "fusion",
                "target_slot": "q2__room_scene",
                "transfer_policy": "direct",
            },
        ]
        fusion_plan = {
            "schema_version": "1.0",
            "plan_id": "plan-environmental-multi",
            "graph_id": "environmental-multi-001",
            "fusion_node_id": "fusion",
            "strategy": "validated_slm",
            "required_slots": required_slots,
            "ordered_slots": ["q1__sign_fact", "q2__room_scene"],
            "max_sentences": 2,
            "spoken_style": True,
            "instructions": "Fuse the typed answers into a concise response.",
            "metadata": {},
        }

    sign_start = query.index("sign")
    room_start = query.index("room")
    return {
        "example_id": "environmental-multi-001",
        "query": query,
        "graph": {
            "schema_version": "1.0",
            "graph_id": "environmental-multi-001",
            "original_query": query,
            "query_type": "Environmental",
            "nodes": nodes,
            "edges": edges,
            "metadata": {},
        },
        "fusion_plan": fusion_plan,
        "planner_labels": {
            "query_type": "Environmental",
            "operation_spans": [
                {
                    "node_id": "q1",
                    "semantic_type": "environmental",
                    "start": 0,
                    "end": split,
                    "operator": "IDENTIFY_ENVIRONMENTAL",
                },
                {
                    "node_id": "q2",
                    "semantic_type": "environmental",
                    "start": right_start,
                    "end": len(query),
                    "operator": "DESCRIBE_ENVIRONMENT",
                },
            ],
            "slot_anchors": [
                {
                    "anchor_id": "a1",
                    "start": sign_start,
                    "end": sign_start + len("sign"),
                    "text": "sign",
                    "normalized_name": "sign",
                    "owner_node_id": "q1",
                    "implicit_resolution": "NONE",
                    "implicit_node_id": None,
                },
                {
                    "anchor_id": "a2",
                    "start": room_start,
                    "end": room_start + len("room"),
                    "text": "room",
                    "normalized_name": "room",
                    "owner_node_id": "q2",
                    "implicit_resolution": "NONE",
                    "implicit_node_id": None,
                },
            ],
        },
        "metadata": {},
    }


def _one_sink_with_fuse_data() -> dict:
    data = _gate_data()
    data["graph"]["nodes"].append(
        {
            "node_id": "fusion",
            "semantic_type": "control",
            "operator": "FUSE",
            "tier": "edge",
            "task": "Fuse the gate answer",
            "required_inputs": {"gate_location": "LOCATION"},
            "produced_outputs": {"response": "FINAL_RESPONSE"},
            "status": "pending",
            "metadata": {},
        }
    )
    data["graph"]["edges"].append(
        {
            "source_node_id": "q2",
            "source_slot": "gate_location",
            "target_node_id": "fusion",
            "target_slot": "gate_location",
            "transfer_policy": "direct",
        }
    )
    data["fusion_plan"] = {
        "schema_version": "1.0",
        "plan_id": "plan-gate",
        "graph_id": "mixed-gate-001",
        "fusion_node_id": "fusion",
        "strategy": "validated_slm",
        "required_slots": {"gate_location": "LOCATION"},
        "ordered_slots": ["gate_location"],
        "max_sentences": 2,
        "spoken_style": True,
        "instructions": "Fuse the typed gate answer.",
        "metadata": {},
    }
    return data


def _malformed_implicit_node_data() -> dict:
    query = "Use my medication"
    medication_start = query.index("medication")
    return {
        "example_id": "malformed-implicit-001",
        "query": query,
        "graph": {
            "schema_version": "1.0",
            "graph_id": "malformed-implicit-001",
            "original_query": query,
            "query_type": "Personal",
            "nodes": [
                {
                    "node_id": "q1",
                    "semantic_type": "personal",
                    "operator": "RETRIEVE_PERSONAL",
                    "tier": "edge",
                    "task": "Retrieve a medication fact",
                    "required_inputs": {},
                    "produced_outputs": {"medication_fact": "PERSONAL_FACT"},
                    "status": "pending",
                    "metadata": {},
                },
                {
                    "node_id": "q2",
                    "semantic_type": "personal",
                    "operator": "RETRIEVE_PERSONAL",
                    "tier": "edge",
                    "task": "Use the medication fact",
                    "required_inputs": {"medication_fact": "PERSONAL_FACT"},
                    "produced_outputs": {"answer_fact": "PERSONAL_FACT"},
                    "status": "pending",
                    "metadata": {},
                },
            ],
            "edges": [
                {
                    "source_node_id": "q1",
                    "source_slot": "medication_fact",
                    "target_node_id": "q2",
                    "target_slot": "medication_fact",
                    "transfer_policy": "direct",
                }
            ],
            "metadata": {},
        },
        "fusion_plan": None,
        "planner_labels": {
            "query_type": "Personal",
            "operation_spans": [
                {
                    "node_id": "q2",
                    "semantic_type": "personal",
                    "start": 0,
                    "end": len(query),
                    "operator": "RETRIEVE_PERSONAL",
                }
            ],
            "slot_anchors": [
                {
                    "anchor_id": "a1",
                    "start": medication_start,
                    "end": medication_start + len("medication"),
                    "text": "medication",
                    "normalized_name": "medication",
                    "owner_node_id": "q2",
                    "implicit_resolution": "IMPLICIT_RESOLVE_PERSONAL",
                    "implicit_node_id": "q1",
                }
            ],
        },
        "metadata": {},
    }


def _multiple_implicit_data() -> dict:
    query = "Where are my gate and my hotel?"
    gate_start = query.index("gate")
    hotel_start = query.index("hotel")
    split = query.index(" and ")
    right_start = split + len(" and ")
    required_slots = {
        "q2__gate_location": "LOCATION",
        "q4__hotel_location": "LOCATION",
    }
    nodes = []
    edges = []
    for resolver_id, locator_id, name in (
        ("q1", "q2", "gate"),
        ("q3", "q4", "hotel"),
    ):
        nodes.extend(
            [
                {
                    "node_id": resolver_id,
                    "semantic_type": "personal",
                    "operator": "RESOLVE_PERSONAL",
                    "tier": "edge",
                    "task": f"Resolve the user's {name} identifier",
                    "required_inputs": {},
                    "produced_outputs": {
                        f"{name}_identifier": "RESOLVED_REFERENCE"
                    },
                    "status": "pending",
                    "metadata": {},
                },
                {
                    "node_id": locator_id,
                    "semantic_type": "environmental",
                    "operator": "LOCATE_ENVIRONMENTAL",
                    "tier": "fog",
                    "task": f"Locate the resolved {name}",
                    "required_inputs": {
                        f"{name}_identifier": "RESOLVED_REFERENCE"
                    },
                    "produced_outputs": {f"{name}_location": "LOCATION"},
                    "status": "pending",
                    "metadata": {},
                },
            ]
        )
        edges.append(
            {
                "source_node_id": resolver_id,
                "source_slot": f"{name}_identifier",
                "target_node_id": locator_id,
                "target_slot": f"{name}_identifier",
                "transfer_policy": "minimal_reference",
            }
        )
    nodes.append(
        {
            "node_id": "fusion",
            "semantic_type": "control",
            "operator": "FUSE",
            "tier": "edge",
            "task": "Fuse the gate and hotel locations",
            "required_inputs": required_slots,
            "produced_outputs": {"response": "FINAL_RESPONSE"},
            "status": "pending",
            "metadata": {},
        }
    )
    edges.extend(
        [
            {
                "source_node_id": "q2",
                "source_slot": "gate_location",
                "target_node_id": "fusion",
                "target_slot": "q2__gate_location",
                "transfer_policy": "direct",
            },
            {
                "source_node_id": "q4",
                "source_slot": "hotel_location",
                "target_node_id": "fusion",
                "target_slot": "q4__hotel_location",
                "transfer_policy": "direct",
            },
        ]
    )
    return {
        "example_id": "mixed-multiple-implicit-001",
        "query": query,
        "graph": {
            "schema_version": "1.0",
            "graph_id": "mixed-multiple-implicit-001",
            "original_query": query,
            "query_type": "Mixed",
            "nodes": nodes,
            "edges": edges,
            "metadata": {},
        },
        "fusion_plan": {
            "schema_version": "1.0",
            "plan_id": "plan-multiple-implicit",
            "graph_id": "mixed-multiple-implicit-001",
            "fusion_node_id": "fusion",
            "strategy": "validated_slm",
            "required_slots": required_slots,
            "ordered_slots": [
                "q2__gate_location",
                "q4__hotel_location",
            ],
            "max_sentences": 2,
            "spoken_style": True,
            "instructions": "Fuse the typed locations into a concise response.",
            "metadata": {},
        },
        "planner_labels": {
            "query_type": "Mixed",
            "operation_spans": [
                {
                    "node_id": "q2",
                    "semantic_type": "environmental",
                    "start": 0,
                    "end": split,
                    "operator": "LOCATE_ENVIRONMENTAL",
                },
                {
                    "node_id": "q4",
                    "semantic_type": "environmental",
                    "start": right_start,
                    "end": len(query),
                    "operator": "LOCATE_ENVIRONMENTAL",
                },
            ],
            "slot_anchors": [
                {
                    "anchor_id": "a1",
                    "start": gate_start,
                    "end": gate_start + len("gate"),
                    "text": "gate",
                    "normalized_name": "gate",
                    "owner_node_id": "q2",
                    "implicit_resolution": "IMPLICIT_RESOLVE_PERSONAL",
                    "implicit_node_id": "q1",
                },
                {
                    "anchor_id": "a2",
                    "start": hotel_start,
                    "end": hotel_start + len("hotel"),
                    "text": "hotel",
                    "normalized_name": "hotel",
                    "owner_node_id": "q4",
                    "implicit_resolution": "IMPLICIT_RESOLVE_PERSONAL",
                    "implicit_node_id": "q3",
                },
            ],
        },
        "metadata": {},
    }


def _edge_to_edge_implicit_data() -> dict:
    query = "Tell me about my gate"
    gate_start = query.index("gate")
    return {
        "example_id": "personal-gate-001",
        "query": query,
        "graph": {
            "schema_version": "1.0",
            "graph_id": "personal-gate-001",
            "original_query": query,
            "query_type": "Personal",
            "nodes": [
                {
                    "node_id": "q1",
                    "semantic_type": "personal",
                    "operator": "RESOLVE_PERSONAL",
                    "tier": "edge",
                    "task": "Resolve the user's gate identifier",
                    "required_inputs": {},
                    "produced_outputs": {
                        "gate_identifier": "RESOLVED_REFERENCE"
                    },
                    "status": "pending",
                    "metadata": {},
                },
                {
                    "node_id": "q2",
                    "semantic_type": "personal",
                    "operator": "RETRIEVE_PERSONAL",
                    "tier": "edge",
                    "task": "Retrieve the personal gate fact",
                    "required_inputs": {
                        "gate_identifier": "RESOLVED_REFERENCE"
                    },
                    "produced_outputs": {"gate_fact": "PERSONAL_FACT"},
                    "status": "pending",
                    "metadata": {},
                },
            ],
            "edges": [
                {
                    "source_node_id": "q1",
                    "source_slot": "gate_identifier",
                    "target_node_id": "q2",
                    "target_slot": "gate_identifier",
                    "transfer_policy": "direct",
                }
            ],
            "metadata": {},
        },
        "fusion_plan": None,
        "planner_labels": {
            "query_type": "Personal",
            "operation_spans": [
                {
                    "node_id": "q2",
                    "semantic_type": "personal",
                    "start": 0,
                    "end": len(query),
                    "operator": "RETRIEVE_PERSONAL",
                }
            ],
            "slot_anchors": [
                {
                    "anchor_id": "a1",
                    "start": gate_start,
                    "end": gate_start + len("gate"),
                    "text": "gate",
                    "normalized_name": "gate",
                    "owner_node_id": "q2",
                    "implicit_resolution": "IMPLICIT_RESOLVE_PERSONAL",
                    "implicit_node_id": "q1",
                }
            ],
        },
        "metadata": {},
    }


def _multi_sink_with_nonsink_fuse_input_data() -> dict:
    data = _multi_sink_data()
    old_query = data["query"]
    query = old_query + " and locate this exit"
    q3_start = len(old_query) + len(" and ")
    data["query"] = query
    data["graph"]["original_query"] = query

    data["graph"]["nodes"][1]["required_inputs"] = {
        "sign_fact": "ENVIRONMENTAL_FACT"
    }
    data["graph"]["nodes"].insert(
        2,
        {
            "node_id": "q3",
            "semantic_type": "environmental",
            "operator": "LOCATE_ENVIRONMENTAL",
            "tier": "fog",
            "task": "Locate this exit",
            "required_inputs": {},
            "produced_outputs": {"exit_location": "LOCATION"},
            "status": "pending",
            "metadata": {},
        },
    )
    fuse = data["graph"]["nodes"][3]
    fuse["required_inputs"]["q3__exit_location"] = "LOCATION"

    data["graph"]["edges"].insert(
        0,
        {
            "source_node_id": "q1",
            "source_slot": "sign_fact",
            "target_node_id": "q2",
            "target_slot": "sign_fact",
            "transfer_policy": "direct",
        },
    )
    data["graph"]["edges"].append(
        {
            "source_node_id": "q3",
            "source_slot": "exit_location",
            "target_node_id": "fusion",
            "target_slot": "q3__exit_location",
            "transfer_policy": "direct",
        }
    )
    data["fusion_plan"]["required_slots"]["q3__exit_location"] = "LOCATION"
    data["fusion_plan"]["ordered_slots"].append("q3__exit_location")
    data["planner_labels"]["operation_spans"].append(
        {
            "node_id": "q3",
            "semantic_type": "environmental",
            "start": q3_start,
            "end": len(query),
            "operator": "LOCATE_ENVIRONMENTAL",
        }
    )
    exit_start = query.index("exit")
    data["planner_labels"]["slot_anchors"].append(
        {
            "anchor_id": "a3",
            "start": exit_start,
            "end": exit_start + len("exit"),
            "text": "exit",
            "normalized_name": "exit",
            "owner_node_id": "q3",
            "implicit_resolution": "NONE",
            "implicit_node_id": None,
        }
    )
    return data


def test_gate_fixture_dictionary_and_json_round_trips():
    original = PlannerExample.model_validate(_gate_data())

    from_dictionary = PlannerExample.model_validate(
        original.model_dump(mode="json")
    )
    from_json = PlannerExample.model_validate_json(original.model_dump_json())

    assert from_dictionary == original
    assert from_json == original
    assert from_json.planner_labels.slot_anchors[0].implicit_resolution is (
        ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
    )


@pytest.mark.parametrize(
    "update",
    [
        {"start": -1},
        {"start": 4, "end": 4},
        {"unknown": True},
    ],
)
def test_operation_span_model_copy_revalidates_updates(update):
    span = PlannerExample.model_validate(
        _gate_data()
    ).planner_labels.operation_spans[0]

    with pytest.raises(ValidationError):
        span.model_copy(update=update)


@pytest.mark.parametrize(
    "factory,update,error",
    [
        (
            _gate_data,
            {"implicit_node_id": None},
            "must be nonblank",
        ),
        (
            _single_environmental_data,
            {"implicit_node_id": "q9"},
            "must be null",
        ),
        (
            _gate_data,
            {"unknown": True},
            "Extra inputs are not permitted",
        ),
    ],
)
def test_slot_anchor_model_copy_revalidates_updates(factory, update, error):
    anchor = PlannerExample.model_validate(
        factory()
    ).planner_labels.slot_anchors[0]

    with pytest.raises(ValidationError, match=error):
        anchor.model_copy(update=update)


def test_planner_labels_model_copy_rejects_empty_or_duplicate_operations():
    labels = PlannerExample.model_validate(_multi_sink_data()).planner_labels

    with pytest.raises(ValidationError, match="must not be empty"):
        labels.model_copy(update={"operation_spans": ()})
    with pytest.raises(ValidationError, match="duplicate operation node_id"):
        labels.model_copy(
            update={"operation_spans": (labels.operation_spans[0],) * 2}
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        labels.model_copy(update={"unknown": True})


def test_planner_example_model_copy_reruns_all_consistency_validation():
    gate = PlannerExample.model_validate(_gate_data())
    wrong_query_graph = gate.graph.model_copy(
        update={"original_query": "A different query"}
    )
    wrong_labels = gate.planner_labels.model_copy(
        update={"query_type": "Personal"}
    )

    with pytest.raises(ValidationError, match="graph.original_query"):
        gate.model_copy(update={"query": "A different query"})
    with pytest.raises(ValidationError, match="graph.original_query"):
        gate.model_copy(update={"graph": wrong_query_graph})
    with pytest.raises(ValidationError, match="planner_labels.query_type"):
        gate.model_copy(update={"planner_labels": wrong_labels})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        gate.model_copy(update={"unknown": True})

    multi = PlannerExample.model_validate(_multi_sink_data())
    assert multi.fusion_plan is not None
    wrong_plan = multi.fusion_plan.model_copy(update={"graph_id": "wrong-graph"})
    with pytest.raises(ValidationError, match="graph_id does not match"):
        multi.model_copy(update={"fusion_plan": wrong_plan})


def test_annotation_model_copy_honors_deep_true_and_valid_updates():
    original = PlannerExample.model_validate(_gate_data())
    update_metadata = {"nested": {"values": [1]}}

    copied = original.model_copy(
        update={"metadata": update_metadata},
        deep=True,
    )
    update_metadata["nested"]["values"].append(2)

    assert copied.metadata == {"nested": {"values": [1]}}
    assert copied == PlannerExample.model_validate(
        copied.model_dump(mode="python", round_trip=True)
    )


def test_load_one_record_and_validate_record_collection():
    serialized = FIXTURE_PATH.read_text(encoding="utf-8")

    loaded = load_planner_record(serialized)
    collection = validate_planner_records([serialized, loaded.model_dump(mode="json")])

    assert loaded.example_id == "mixed-gate-001"
    assert collection == (loaded, loaded)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda data: data.update({"unknown": True}),
        lambda data: data["planner_labels"]["operation_spans"][0].update(
            {"unknown": True}
        ),
        lambda data: data["planner_labels"]["operation_spans"][0].update(
            {"start": "0"}
        ),
        lambda data: data["planner_labels"]["slot_anchors"][0].update(
            {"implicit_resolution": "UNKNOWN"}
        ),
    ],
)
def test_annotation_schemas_reject_unknown_fields_and_noncanonical_values(mutator):
    data = _gate_data()
    mutator(data)

    with pytest.raises(ValidationError):
        PlannerExample.model_validate(data)


def test_offsets_use_zero_based_half_open_original_query_slices():
    example = PlannerExample.model_validate(_gate_data())
    anchor = example.planner_labels.slot_anchors[0]

    assert anchor.start == 12
    assert anchor.end == 16
    assert example.query[anchor.start : anchor.end] == "gate"

    invalid = _gate_data()
    invalid["planner_labels"]["slot_anchors"][0]["end"] = 17
    with pytest.raises(ValidationError, match="text does not match"):
        PlannerExample.model_validate(invalid)


def test_unicode_offsets_use_python_characters_not_encoded_bytes():
    data = _single_environmental_data(
        query="Décris ce café ☕",
        anchor_text="café",
    )

    example = PlannerExample.model_validate(data)
    anchor = example.planner_labels.slot_anchors[0]

    assert example.query[anchor.start : anchor.end] == "café"
    assert anchor.start == 10
    assert anchor.end == 14


def test_anchor_offset_text_mismatch_is_rejected_without_normalization():
    data = _single_environmental_data(query="Read Café", anchor_text="Café")
    data["planner_labels"]["slot_anchors"][0]["text"] = "café"

    with pytest.raises(ValidationError, match="text does not match"):
        PlannerExample.model_validate(data)


def test_adjacent_operation_spans_are_accepted():
    data = _multi_sink_data()
    second_start = data["planner_labels"]["operation_spans"][1]["start"]
    data["planner_labels"]["operation_spans"][0]["end"] = second_start

    example = PlannerExample.model_validate(data)

    first, second = example.planner_labels.operation_spans
    assert first.end == second.start


@pytest.mark.parametrize(
    "start,end",
    [
        (10, 30),
        (2, 10),
    ],
)
def test_overlapping_and_nested_operation_spans_are_rejected(start, end):
    data = _multi_sink_data()
    data["planner_labels"]["operation_spans"][1].update(
        {"start": start, "end": end}
    )

    with pytest.raises(ValidationError, match="must not overlap or nest"):
        PlannerExample.model_validate(data)


def test_operation_span_outside_query_is_rejected():
    data = _single_environmental_data()
    data["planner_labels"]["operation_spans"][0]["end"] = len(data["query"]) + 1

    with pytest.raises(ValidationError, match="outside the query"):
        PlannerExample.model_validate(data)


def test_duplicate_operation_node_ids_are_rejected():
    data = _multi_sink_data()
    duplicate = data["planner_labels"]["operation_spans"][1]
    duplicate.update(
        {
            "node_id": "q1",
            "semantic_type": "environmental",
            "operator": "IDENTIFY_ENVIRONMENTAL",
        }
    )

    with pytest.raises(ValidationError, match="duplicate operation node_id"):
        PlannerExample.model_validate(data)


def test_duplicate_anchor_ids_are_rejected():
    data = _multi_sink_data()
    data["planner_labels"]["slot_anchors"][1]["anchor_id"] = "a1"

    with pytest.raises(ValidationError, match="duplicate anchor_id"):
        PlannerExample.model_validate(data)


def test_invalid_anchor_owner_is_rejected():
    data = _gate_data()
    data["planner_labels"]["slot_anchors"][0]["owner_node_id"] = "missing"

    with pytest.raises(ValidationError, match="explicit operation node"):
        PlannerExample.model_validate(data)


def test_none_resolution_requires_serialized_null_implicit_node_id():
    data = _single_environmental_data()
    data["planner_labels"]["slot_anchors"][0]["implicit_node_id"] = "q2"

    with pytest.raises(ValidationError, match="must be null"):
        PlannerExample.model_validate(data)

    valid = PlannerExample.model_validate(_single_environmental_data())
    assert "implicit_node_id" in valid.model_dump(mode="json")["planner_labels"][
        "slot_anchors"
    ][0]
    assert valid.planner_labels.slot_anchors[0].implicit_node_id is None


def test_positive_resolution_requires_nonblank_implicit_node_id():
    data = _gate_data()
    data["planner_labels"]["slot_anchors"][0]["implicit_node_id"] = None

    with pytest.raises(ValidationError, match="must be nonblank"):
        PlannerExample.model_validate(data)


def test_positive_anchors_must_reference_distinct_implicit_nodes():
    data = _gate_data()
    duplicate_reference = deepcopy(
        data["planner_labels"]["slot_anchors"][0]
    )
    duplicate_reference["anchor_id"] = "a2"
    data["planner_labels"]["slot_anchors"].append(duplicate_reference)

    with pytest.raises(ValidationError, match="distinct implicit node"):
        PlannerExample.model_validate(data)


def test_malformed_implicit_personal_node_is_rejected():
    with pytest.raises(ValidationError, match="must use RESOLVE_PERSONAL"):
        PlannerExample.model_validate(_malformed_implicit_node_data())


def test_missing_implicit_owner_edge_is_rejected():
    data = _gate_data()
    data["graph"]["edges"] = []
    data["graph"]["nodes"][1]["required_inputs"] = {}

    with pytest.raises(ValidationError, match="must create exactly one"):
        PlannerExample.model_validate(data)


def test_incorrect_implicit_owner_target_slot_is_rejected():
    data = _gate_data()
    data["graph"]["nodes"][1]["required_inputs"] = {
        "resolved_gate": "RESOLVED_REFERENCE"
    }
    data["graph"]["edges"][0]["target_slot"] = "resolved_gate"

    with pytest.raises(ValidationError, match="target slot must match"):
        PlannerExample.model_validate(data)


def test_additional_edge_between_implicit_node_and_owner_is_rejected():
    data = _gate_data()
    data["graph"]["nodes"][1]["required_inputs"]["extra_reference"] = (
        "RESOLVED_REFERENCE"
    )
    data["graph"]["edges"].append(
        {
            "source_node_id": "q1",
            "source_slot": "gate_identifier",
            "target_node_id": "q2",
            "target_slot": "extra_reference",
            "transfer_policy": "minimal_reference",
        }
    )

    with pytest.raises(ValidationError, match="exactly one"):
        PlannerExample.model_validate(data)


def test_duplicate_implicit_owner_edge_is_rejected():
    data = _gate_data()
    data["graph"]["edges"].append(deepcopy(data["graph"]["edges"][0]))

    with pytest.raises(ValidationError, match="duplicate dependency edge"):
        PlannerExample.model_validate(data)


def test_wrong_implicit_owner_source_slot_is_rejected():
    data = _gate_data()
    data["graph"]["edges"][0]["source_slot"] = "wrong_reference"

    with pytest.raises(ValidationError, match="source slot does not exist"):
        PlannerExample.model_validate(data)


def test_implicit_edge_to_wrong_owner_node_is_rejected():
    data = _gate_data()
    data["graph"]["nodes"][1]["required_inputs"] = {}
    data["graph"]["nodes"].append(
        {
            "node_id": "q3",
            "semantic_type": "environmental",
            "operator": "LOCATE_ENVIRONMENTAL",
            "tier": "fog",
            "task": "Locate a different reference",
            "required_inputs": {
                "gate_identifier": "RESOLVED_REFERENCE"
            },
            "produced_outputs": {"other_location": "LOCATION"},
            "status": "pending",
            "metadata": {},
        }
    )
    data["graph"]["edges"][0]["target_node_id"] = "q3"

    with pytest.raises(ValidationError, match="exactly one"):
        PlannerExample.model_validate(data)


def test_incorrect_implicit_owner_transfer_policy_is_rejected():
    data = _gate_data()
    data["graph"]["edges"][0]["transfer_policy"] = "direct"

    with pytest.raises(ValidationError, match="minimal_reference"):
        PlannerExample.model_validate(data)


def test_edge_to_edge_implicit_owner_dependency_uses_direct_policy():
    example = PlannerExample.model_validate(_edge_to_edge_implicit_data())

    assert example.graph.edges[0].transfer_policy is TransferPolicy.DIRECT


def test_edge_to_edge_implicit_owner_dependency_rejects_minimal_reference():
    data = _edge_to_edge_implicit_data()
    data["graph"]["edges"][0]["transfer_policy"] = "minimal_reference"

    with pytest.raises(ValidationError, match="only for Edge-to-Fog"):
        PlannerExample.model_validate(data)


@pytest.mark.parametrize(
    "updates,error",
    [
        (
            {"semantic_type": "personal", "operator": "RESOLVE_PERSONAL"},
            "semantic_type does not match",
        ),
        (
            {"semantic_type": "environmental", "operator": "DESCRIBE_ENVIRONMENT"},
            "operator does not match",
        ),
    ],
)
def test_operation_semantic_and_operator_must_match_graph_node(updates, error):
    data = _gate_data()
    data["planner_labels"]["operation_spans"][0].update(updates)

    with pytest.raises(ValidationError, match=error):
        PlannerExample.model_validate(data)


def test_deterministic_principal_output_mismatch_is_rejected():
    data = {
        **_single_environmental_data(query="Retrieve my code", anchor_text="code"),
    }
    data["example_id"] = "personal-output-mismatch"
    data["graph"]["graph_id"] = "personal-output-mismatch"
    data["graph"]["query_type"] = "Personal"
    node = data["graph"]["nodes"][0]
    node.update(
        {
            "semantic_type": "personal",
            "operator": "RETRIEVE_PERSONAL",
            "tier": "edge",
            "produced_outputs": {"code": "RESOLVED_REFERENCE"},
        }
    )
    operation = data["planner_labels"]["operation_spans"][0]
    operation.update(
        {
            "semantic_type": "personal",
            "operator": "RETRIEVE_PERSONAL",
        }
    )
    data["planner_labels"]["query_type"] = "Personal"

    with pytest.raises(ValidationError, match="PERSONAL_FACT"):
        PlannerExample.model_validate(data)


def test_graph_query_type_must_match_answer_node_semantics():
    data = _single_environmental_data()
    data["graph"]["query_type"] = "Personal"
    data["planner_labels"]["query_type"] = "Personal"

    with pytest.raises(ValidationError, match="Personal graphs require"):
        PlannerExample.model_validate(data)


def test_planner_label_query_type_must_match_graph():
    data = _gate_data()
    data["planner_labels"]["query_type"] = "Personal"

    with pytest.raises(ValidationError, match="planner_labels.query_type"):
        PlannerExample.model_validate(data)


def test_one_answer_sink_rejects_fuse():
    with pytest.raises(ValidationError, match="one answer sink"):
        PlannerExample.model_validate(_one_sink_with_fuse_data())


def test_multiple_answer_sinks_require_fuse():
    with pytest.raises(ValidationError, match="exactly one terminal Edge FUSE"):
        PlannerExample.model_validate(_multi_sink_data(include_fuse=False))


def test_fuse_with_successor_is_rejected():
    data = _multi_sink_data()
    data["graph"]["nodes"].append(
        {
            "node_id": "post_fusion",
            "semantic_type": "control",
            "operator": "FUSE",
            "tier": "edge",
            "task": "Invalid second fusion stage",
            "required_inputs": {"response": "FINAL_RESPONSE"},
            "produced_outputs": {"final_response": "FINAL_RESPONSE"},
            "status": "pending",
            "metadata": {},
        }
    )
    data["graph"]["edges"].append(
        {
            "source_node_id": "fusion",
            "source_slot": "response",
            "target_node_id": "post_fusion",
            "target_slot": "response",
            "transfer_policy": "direct",
        }
    )

    with pytest.raises(ValidationError, match="FUSE nodes must be terminal"):
        PlannerExample.model_validate(data)


def test_fuse_input_from_non_sink_answer_node_is_rejected():
    with pytest.raises(
        ValidationError,
        match="exactly one dependency from every answer sink",
    ):
        PlannerExample.model_validate(
            _multi_sink_with_nonsink_fuse_input_data()
        )


@pytest.mark.parametrize(
    "produced_outputs",
    [
        {"response": "LOCATION"},
        {
            "response": "FINAL_RESPONSE",
            "other_response": "FINAL_RESPONSE",
        },
    ],
)
def test_fuse_must_produce_exactly_one_final_response(produced_outputs):
    data = _multi_sink_data()
    data["graph"]["nodes"][2]["produced_outputs"] = produced_outputs

    with pytest.raises(ValidationError, match="FUSE"):
        PlannerExample.model_validate(data)


def test_fusion_plan_required_slots_must_match_fuse_inputs():
    data = _multi_sink_data()
    data["fusion_plan"]["required_slots"] = dict(
        data["fusion_plan"]["required_slots"]
    )
    data["fusion_plan"]["required_slots"]["q1__sign_fact"] = "LOCATION"

    with pytest.raises(ValidationError, match="required_slots must exactly match"):
        PlannerExample.model_validate(data)


def test_annotation_fusion_strategy_must_be_validated_slm():
    data = _multi_sink_data()
    data["fusion_plan"]["strategy"] = "template"

    with pytest.raises(ValidationError, match="must be validated_slm"):
        PlannerExample.model_validate(data)


def test_fuse_requires_a_fusion_plan():
    data = _multi_sink_data()
    data["fusion_plan"] = None

    with pytest.raises(ValidationError, match="require a FusionPlan"):
        PlannerExample.model_validate(data)


def test_fusion_plan_is_rejected_when_fuse_is_absent():
    data = _single_environmental_data()
    data["fusion_plan"] = deepcopy(_multi_sink_data()["fusion_plan"])

    with pytest.raises(ValidationError, match="fusion_plan must be null"):
        PlannerExample.model_validate(data)


def test_multiple_sink_fusion_plan_validates_against_graph():
    example = PlannerExample.model_validate(_multi_sink_data())

    assert example.fusion_plan is not None
    assert example.fusion_plan.strategy is FusionStrategy.VALIDATED_SLM
    assert example.fusion_plan.validate_against_graph(example.graph)


def test_invalid_fusion_plan_graph_reference_is_rejected():
    data = _multi_sink_data()
    data["fusion_plan"]["graph_id"] = "other-graph"

    with pytest.raises(ValidationError, match="graph_id does not match"):
        PlannerExample.model_validate(data)


def test_complete_gate_fixture_preserves_permanent_invariant():
    example = load_planner_record(FIXTURE_PATH.read_text(encoding="utf-8"))
    graph = example.graph
    q1 = graph.node_by_id("q1")
    q2 = graph.node_by_id("q2")
    anchor = example.planner_labels.slot_anchors[0]

    assert example.query == "Where is my gate?"
    assert len([node for node in graph.nodes if node.operator is not OperatorType.FUSE]) == 2
    assert q1.operator is OperatorType.RESOLVE_PERSONAL
    assert q1.semantic_type is NodeSemanticType.PERSONAL
    assert q1.tier is Tier.EDGE
    assert q2.operator is OperatorType.LOCATE_ENVIRONMENTAL
    assert q2.semantic_type is NodeSemanticType.ENVIRONMENTAL
    assert q2.tier is Tier.FOG
    assert anchor.owner_node_id == "q2"
    assert anchor.implicit_node_id == "q1"
    assert graph.edges[0].source_slot == "gate_identifier"
    assert graph.edges[0].target_slot == "gate_identifier"
    assert graph.edges[0].transfer_policy is TransferPolicy.MINIMAL_REFERENCE
    assert graph.successors("q1") == (q2,)
    assert graph.successors("q2") == ()
    assert all(node.operator is not OperatorType.FUSE for node in graph.nodes)
    assert all("sanit" not in node.operator.value.lower() for node in graph.nodes)
    assert example.fusion_plan is None


def test_multiple_independent_implicit_anchors_are_valid():
    example = PlannerExample.model_validate(_multiple_implicit_data())
    anchors = example.planner_labels.slot_anchors

    assert {anchor.owner_node_id for anchor in anchors} == {"q2", "q4"}
    assert {anchor.implicit_node_id for anchor in anchors} == {"q1", "q3"}
    assert example.graph.execution_mode() == "hybrid"
    assert example.fusion_plan is not None


def test_jsonl_reports_physical_line_number_and_example_id(tmp_path):
    valid = json.dumps(_gate_data())
    invalid_data = _gate_data()
    invalid_data["example_id"] = "broken-gate"
    invalid_data["planner_labels"]["slot_anchors"][0]["owner_node_id"] = "missing"
    path = tmp_path / "annotations.jsonl"
    path.write_text(
        valid + "\n\n" + json.dumps(invalid_data) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PlannerRecordValidationError) as caught:
        load_planner_jsonl(path)

    assert caught.value.record_number == 3
    assert caught.value.example_id == "broken-gate"
    assert "planner record 3" in str(caught.value)
    assert "broken-gate" in str(caught.value)


def test_jsonl_malformed_json_reports_physical_line_and_unknown_id(tmp_path):
    path = tmp_path / "malformed.jsonl"
    path.write_text(
        json.dumps(_gate_data()) + "\n\n\n" + '{"example_id": ',
        encoding="utf-8",
    )

    with pytest.raises(PlannerRecordValidationError) as caught:
        load_planner_jsonl(path)

    assert caught.value.record_number == 4
    assert caught.value.example_id == "<unknown>"
    assert "planner record 4" in str(caught.value)


def test_jsonl_schema_error_without_readable_id_uses_unknown_placeholder(
    tmp_path,
):
    path = tmp_path / "missing-id.jsonl"
    path.write_text("\n{}\n", encoding="utf-8")

    with pytest.raises(PlannerRecordValidationError) as caught:
        load_planner_jsonl(path)

    assert caught.value.record_number == 2
    assert caught.value.example_id == "<unknown>"


def test_jsonl_whitespace_only_id_uses_unknown_placeholder(tmp_path):
    path = tmp_path / "blank-id.jsonl"
    path.write_text('{"example_id": "   "}\n', encoding="utf-8")

    with pytest.raises(PlannerRecordValidationError) as caught:
        load_planner_jsonl(path)

    assert caught.value.record_number == 1
    assert caught.value.example_id == "<unknown>"


def test_planner_annotation_import_is_isolated():
    script = """
import json
import sys
import tiergraph
top_level_loaded_planner = "tiergraph.planner" in sys.modules
import tiergraph.planner
forbidden = (
    "sentence_transformers",
    "transformers",
    "torch",
    "router",
    "edge",
    "fog",
    "context_store",
    "config",
    "sqlite3",
    "faiss",
    "httpx",
    "aiohttp",
)
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
print(json.dumps({"top_level_loaded_planner": top_level_loaded_planner, "loaded": loaded}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)

    assert report == {"top_level_loaded_planner": False, "loaded": []}
