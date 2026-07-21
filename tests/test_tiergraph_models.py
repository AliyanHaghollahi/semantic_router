"""Unit tests for TierGraph enums and execution result schemas."""

import math

import pytest
from pydantic import ValidationError

from tiergraph import (
    EvidenceItem,
    ExecutionStatus,
    NodeSemanticType,
    OperatorType,
    QueryType,
    SlotType,
    Tier,
    TierResult,
    TransferPolicy,
)


def _evidence(**overrides) -> EvidenceItem:
    values = {
        "evidence_id": "evidence-gate",
        "graph_id": "graph-gate",
        "node_id": "resolve-gate",
        "slot_name": "gate_identifier",
        "slot_type": SlotType.RESOLVED_REFERENCE,
        "tier": Tier.EDGE,
        "value": "D34",
        "source": "personal_booking_store",
        "confidence": 0.99,
        "metadata": {"record_count": 1},
    }
    values.update(overrides)
    return EvidenceItem(**values)


def _tier_result(**overrides) -> TierResult:
    values = {
        "result_id": "result-gate",
        "graph_id": "graph-gate",
        "node_id": "resolve-gate",
        "tier": Tier.EDGE,
        "status": ExecutionStatus.SUCCEEDED,
        "outputs": {"gate_identifier": "D34"},
        "evidence": (_evidence(),),
        "latency_ms": 12.5,
        "metadata": {"attempt": 1},
    }
    values.update(overrides)
    return TierResult(**values)


def test_enum_wire_values_are_stable():
    assert [item.value for item in QueryType] == [
        "Personal",
        "Environmental",
        "Mixed",
    ]
    assert [item.value for item in NodeSemanticType] == [
        "personal",
        "environmental",
        "control",
    ]
    assert [item.value for item in SlotType] == [
        "RESOLVED_REFERENCE",
        "PERSONAL_FACT",
        "PERSONAL_RECORD",
        "ENVIRONMENTAL_FACT",
        "LOCATION",
        "NAVIGATION_INSTRUCTION",
        "SCENE_DESCRIPTION",
        "FINAL_RESPONSE",
    ]
    assert [item.value for item in Tier] == ["edge", "fog"]
    assert [item.value for item in OperatorType] == [
        "RESOLVE_PERSONAL",
        "RETRIEVE_PERSONAL",
        "IDENTIFY_ENVIRONMENTAL",
        "LOCATE_ENVIRONMENTAL",
        "NAVIGATE_TO",
        "DESCRIBE_ENVIRONMENT",
        "FUSE",
    ]
    assert [item.value for item in ExecutionStatus] == [
        "pending",
        "ready",
        "running",
        "succeeded",
        "failed",
        "skipped",
        "blocked",
        "cancelled",
    ]
    assert [item.value for item in TransferPolicy] == [
        "direct",
        "minimal_reference",
    ]


@pytest.mark.parametrize(
    "enum_type",
    [
        QueryType,
        NodeSemanticType,
        SlotType,
        Tier,
        OperatorType,
        ExecutionStatus,
        TransferPolicy,
    ],
)
def test_enums_inherit_from_str_and_enum(enum_type):
    from enum import Enum

    assert issubclass(enum_type, str)
    assert issubclass(enum_type, Enum)


def test_evidence_item_accepts_json_values_and_optional_fields():
    evidence = _evidence(
        value={"gate": "D34", "alternatives": ["D32", "D36"]},
        source=None,
        confidence=None,
    )

    assert evidence.slot_type is SlotType.RESOLVED_REFERENCE
    assert evidence.tier is Tier.EDGE
    assert evidence.value["gate"] == "D34"
    assert evidence.source is None
    assert evidence.confidence is None


def test_evidence_item_accepts_canonical_enum_wire_strings():
    evidence = EvidenceItem.model_validate(
        {
            "evidence_id": "evidence-gate",
            "graph_id": "graph-gate",
            "node_id": "resolve-gate",
            "slot_name": "gate_identifier",
            "slot_type": "RESOLVED_REFERENCE",
            "tier": "edge",
            "value": "D34",
        }
    )

    assert evidence.slot_type is SlotType.RESOLVED_REFERENCE
    assert evidence.tier is Tier.EDGE


@pytest.mark.parametrize(
    "field,value",
    [
        ("slot_type", "UNKNOWN_SLOT_TYPE"),
        ("tier", "cloud"),
    ],
)
def test_evidence_item_rejects_unknown_enum_wire_strings(field, value):
    with pytest.raises(ValidationError):
        _evidence(**{field: value})


@pytest.mark.parametrize("field", ["evidence_id", "graph_id", "node_id", "slot_name"])
def test_evidence_item_rejects_blank_identifiers(field):
    with pytest.raises(ValidationError, match="must not be blank"):
        _evidence(**{field: "   "})


@pytest.mark.parametrize("confidence", [-0.01, 1.01, math.inf, -math.inf, math.nan])
def test_evidence_item_rejects_invalid_confidence(confidence):
    with pytest.raises(ValidationError):
        _evidence(confidence=confidence)


@pytest.mark.parametrize("confidence", ["0.99", True])
def test_evidence_item_rejects_confidence_coercion(confidence):
    with pytest.raises(ValidationError):
        _evidence(confidence=confidence)


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_evidence_item_rejects_non_finite_nested_json(non_finite):
    with pytest.raises(ValidationError):
        _evidence(value={"score": non_finite})

    with pytest.raises(ValidationError):
        _evidence(metadata={"score": non_finite})


def test_evidence_item_rejects_non_json_value_and_metadata():
    with pytest.raises(ValidationError):
        _evidence(value=object())

    with pytest.raises(ValidationError):
        _evidence(metadata={"unsupported": object()})


def test_evidence_item_rejects_blank_source():
    with pytest.raises(ValidationError, match="source must not be blank"):
        _evidence(source="  ")


def test_shared_schema_is_strict_frozen_and_forbids_extra_fields():
    evidence = _evidence()

    with pytest.raises(ValidationError, match="frozen"):
        evidence.node_id = "different-node"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _evidence(unexpected="value")

    with pytest.raises(ValidationError):
        _evidence(confidence="0.99")


def test_tier_result_accepts_successful_edge_result():
    result = _tier_result()

    assert result.schema_version == "1.0"
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.outputs == {"gate_identifier": "D34"}
    assert result.evidence[0].evidence_id == "evidence-gate"
    assert result.latency_ms == 12.5
    assert result.error is None


def test_tier_result_dictionary_round_trip():
    original = _tier_result()

    restored = TierResult.model_validate(original.model_dump(mode="json"))

    assert restored == original


def test_tier_result_json_round_trip():
    original = _tier_result()

    restored = TierResult.model_validate_json(original.model_dump_json())

    assert restored == original


def test_tier_result_accepts_canonical_enum_wire_strings():
    values = _tier_result().model_dump(mode="json")
    values["tier"] = "edge"
    values["status"] = "succeeded"

    result = TierResult.model_validate(values)

    assert result.tier is Tier.EDGE
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.evidence[0].slot_type is SlotType.RESOLVED_REFERENCE


@pytest.mark.parametrize(
    "field,value",
    [
        ("tier", "cloud"),
        ("status", "complete"),
    ],
)
def test_tier_result_rejects_unknown_enum_wire_strings(field, value):
    values = _tier_result().model_dump(mode="json")
    values[field] = value

    with pytest.raises(ValidationError):
        TierResult.model_validate(values)


def test_tier_result_accepts_failed_result_with_error():
    result = _tier_result(
        status=ExecutionStatus.FAILED,
        outputs={},
        evidence=(),
        error="Personal store unavailable",
    )

    assert result.status is ExecutionStatus.FAILED
    assert result.error == "Personal store unavailable"


@pytest.mark.parametrize("error", [None, "", "   "])
def test_failed_tier_result_requires_nonblank_error(error):
    with pytest.raises(ValidationError, match="requires a nonblank error"):
        _tier_result(status=ExecutionStatus.FAILED, error=error)


def test_succeeded_tier_result_rejects_error():
    with pytest.raises(ValidationError, match="must not contain an error"):
        _tier_result(error="unexpected")


@pytest.mark.parametrize("latency_ms", [-0.01, math.inf, -math.inf, math.nan])
def test_tier_result_rejects_invalid_latency(latency_ms):
    with pytest.raises(ValidationError):
        _tier_result(latency_ms=latency_ms)


@pytest.mark.parametrize("latency_ms", ["12.5", True])
def test_tier_result_rejects_latency_coercion(latency_ms):
    with pytest.raises(ValidationError):
        _tier_result(latency_ms=latency_ms)


@pytest.mark.parametrize("field", ["result_id", "graph_id", "node_id"])
def test_tier_result_rejects_blank_identifiers(field):
    with pytest.raises(ValidationError, match="must not be blank"):
        _tier_result(**{field: "\t"})


def test_tier_result_rejects_blank_output_slot_name():
    with pytest.raises(ValidationError, match="output slot names must not be blank"):
        _tier_result(outputs={" ": "D34"})


def test_tier_result_rejects_unsupported_schema_version():
    with pytest.raises(ValidationError):
        _tier_result(schema_version="2.0")


def test_tier_result_rejects_non_json_output_and_metadata():
    with pytest.raises(ValidationError):
        _tier_result(outputs={"gate_identifier": object()})

    with pytest.raises(ValidationError):
        _tier_result(metadata={"unsupported": object()})


def test_tier_result_forbids_extra_fields_and_is_frozen():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _tier_result(unexpected="value")

    result = _tier_result()
    with pytest.raises(ValidationError, match="frozen"):
        result.status = ExecutionStatus.FAILED
