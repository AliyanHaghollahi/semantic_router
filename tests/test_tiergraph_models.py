"""Unit tests for TierGraph enums and execution result schemas."""

import math

import pytest
from pydantic import ValidationError

from tiergraph import (
    EvidenceItem,
    ExecutionStatus,
    FusionOutput,
    FusionPlan,
    FusionStrategy,
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


def _fusion_plan(**overrides) -> FusionPlan:
    values = {
        "plan_id": "plan-parallel",
        "graph_id": "graph-parallel",
        "fusion_node_id": "fusion",
        "strategy": FusionStrategy.CONCATENATE,
        "required_slots": {
            "personal": SlotType.PERSONAL_FACT,
            "environmental": SlotType.ENVIRONMENTAL_FACT,
        },
        "ordered_slots": ("personal", "environmental"),
        "max_sentences": 2,
        "spoken_style": True,
        "instructions": "Combine the supplied facts into a concise response.",
        "metadata": {"locale": "en"},
    }
    values.update(overrides)
    return FusionPlan(**values)


def _fusion_output(**overrides) -> FusionOutput:
    values = {
        "output_id": "output-parallel",
        "plan_id": "plan-parallel",
        "graph_id": "graph-parallel",
        "fusion_node_id": "fusion",
        "strategy": FusionStrategy.CONCATENATE,
        "status": ExecutionStatus.SUCCEEDED,
        "text": "Take your medication; a pharmacy is nearby.",
        "method": "concatenate_v1",
        "evidence_ids": ("evidence-personal", "evidence-environmental"),
        "latency_ms": 3.5,
        "metadata": {"sentence_count": 1},
    }
    values.update(overrides)
    return FusionOutput(**values)


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
    assert [item.value for item in FusionStrategy] == [
        "concatenate",
        "template",
        "slm",
        "validated_slm",
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
        FusionStrategy,
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


@pytest.mark.parametrize("strategy", list(FusionStrategy))
def test_fusion_plan_accepts_each_strategy(strategy):
    plan = _fusion_plan(strategy=strategy)

    assert plan.strategy is strategy


def test_fusion_plan_accepts_canonical_dictionary_wire_values():
    values = _fusion_plan().model_dump(mode="json")

    restored = FusionPlan.model_validate(values)

    assert restored.strategy is FusionStrategy.CONCATENATE
    assert restored.required_slots["personal"] is SlotType.PERSONAL_FACT
    assert restored.ordered_slots == ("personal", "environmental")


def test_fusion_plan_dictionary_and_json_round_trips():
    original = _fusion_plan()

    from_dictionary = FusionPlan.model_validate(original.model_dump(mode="json"))
    from_json = FusionPlan.model_validate_json(original.model_dump_json())

    assert from_dictionary == original
    assert from_json == original


def test_fusion_plan_rejects_unknown_strategy_wire_value():
    with pytest.raises(ValidationError):
        _fusion_plan(strategy="unknown")


def test_fusion_plan_rejects_unknown_required_slot_type():
    with pytest.raises(ValidationError):
        _fusion_plan(
            required_slots={"personal": "UNKNOWN_SLOT"},
            ordered_slots=("personal",),
        )


@pytest.mark.parametrize(
    "ordered_slots,error",
    [
        (("personal", "personal"), "duplicates"),
        (("personal",), "exact permutation"),
        (("personal", "environmental", "extra"), "exact permutation"),
    ],
)
def test_fusion_plan_rejects_invalid_ordered_slots(ordered_slots, error):
    with pytest.raises(ValidationError, match=error):
        _fusion_plan(ordered_slots=ordered_slots)


def test_fusion_plan_rejects_empty_required_slots():
    with pytest.raises(ValidationError, match="must not be empty"):
        _fusion_plan(required_slots={}, ordered_slots=())


@pytest.mark.parametrize(
    "required_slots,ordered_slots",
    [
        ({" ": SlotType.PERSONAL_FACT}, (" ",)),
        ({"personal": SlotType.PERSONAL_FACT}, (" ",)),
    ],
)
def test_fusion_plan_rejects_blank_slot_names(required_slots, ordered_slots):
    with pytest.raises(ValidationError, match="slot names must not be blank"):
        _fusion_plan(
            required_slots=required_slots,
            ordered_slots=ordered_slots,
        )


@pytest.mark.parametrize("max_sentences", [0, -1, "2", True, 2.0, math.nan, math.inf])
def test_fusion_plan_rejects_invalid_max_sentences(max_sentences):
    with pytest.raises(ValidationError):
        _fusion_plan(max_sentences=max_sentences)


@pytest.mark.parametrize("spoken_style", [0, 1, "true"])
def test_fusion_plan_rejects_spoken_style_coercion(spoken_style):
    with pytest.raises(ValidationError):
        _fusion_plan(spoken_style=spoken_style)


@pytest.mark.parametrize(
    "field",
    ["plan_id", "graph_id", "fusion_node_id", "instructions"],
)
def test_fusion_plan_rejects_blank_text_fields(field):
    with pytest.raises(ValidationError, match="must not be blank"):
        _fusion_plan(**{field: "  "})


def test_fusion_plan_has_exact_fields_and_edge_execution_tier():
    plan = _fusion_plan()
    serialized = plan.model_dump(mode="json")

    assert set(serialized) == {
        "schema_version",
        "plan_id",
        "graph_id",
        "fusion_node_id",
        "strategy",
        "required_slots",
        "ordered_slots",
        "max_sentences",
        "spoken_style",
        "instructions",
        "metadata",
    }
    assert plan.execution_tier is Tier.EDGE
    assert "execution_tier" not in serialized
    assert "original_query" not in serialized


def test_fusion_plan_rejects_unsupported_version_and_extra_fields():
    with pytest.raises(ValidationError):
        _fusion_plan(schema_version="2.0")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _fusion_plan(unexpected="value")


@pytest.mark.parametrize(
    "update,error",
    [
        ({"max_sentences": 0}, "max_sentences"),
        ({"ordered_slots": ("personal",)}, "exact permutation"),
        (
            {"ordered_slots": ("personal", "personal")},
            "duplicates",
        ),
        ({"required_slots": {}}, "must not be empty"),
        ({"strategy": "unknown"}, "strategy"),
    ],
)
def test_fusion_plan_model_copy_rejects_invalid_updates(update, error):
    with pytest.raises(ValidationError, match=error):
        _fusion_plan().model_copy(update=update)


def test_fusion_plan_model_copy_validates_updates_and_preserves_serialization():
    original = _fusion_plan()

    copied = original.model_copy(update={"strategy": "template"})
    ordinary_copy = original.model_copy()

    assert copied.strategy is FusionStrategy.TEMPLATE
    assert ordinary_copy == original
    assert "execution_tier" not in copied.model_dump(mode="json")


def test_fusion_output_accepts_success_and_failure():
    succeeded = _fusion_output()
    failed = _fusion_output(
        status=ExecutionStatus.FAILED,
        text="",
        error="Fusion model unavailable",
    )

    assert succeeded.error is None
    assert failed.error == "Fusion model unavailable"


def test_fusion_output_accepts_integer_latency_as_float():
    output = _fusion_output(latency_ms=1)

    assert output.latency_ms == 1.0
    assert type(output.latency_ms) is float


def test_fusion_output_accepts_canonical_dictionary_wire_values():
    values = _fusion_output().model_dump(mode="json")

    restored = FusionOutput.model_validate(values)

    assert restored.strategy is FusionStrategy.CONCATENATE
    assert restored.status is ExecutionStatus.SUCCEEDED
    assert restored.evidence_ids == (
        "evidence-personal",
        "evidence-environmental",
    )


def test_fusion_output_dictionary_and_json_round_trips():
    original = _fusion_output()

    from_dictionary = FusionOutput.model_validate(original.model_dump(mode="json"))
    from_json = FusionOutput.model_validate_json(original.model_dump_json())

    assert from_dictionary == original
    assert from_json == original


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"text": "  "}, "requires nonblank text"),
        ({"error": "unexpected"}, "must not contain an error"),
        (
            {"status": ExecutionStatus.FAILED, "text": "", "error": None},
            "requires a nonblank error",
        ),
        (
            {"status": ExecutionStatus.FAILED, "text": "", "error": "  "},
            "requires a nonblank error",
        ),
    ],
)
def test_fusion_output_enforces_status_contract(overrides, error):
    with pytest.raises(ValidationError, match=error):
        _fusion_output(**overrides)


@pytest.mark.parametrize(
    "latency_ms",
    [-0.01, math.nan, math.inf, -math.inf, "3.5", True],
)
def test_fusion_output_rejects_invalid_latency(latency_ms):
    with pytest.raises(ValidationError):
        _fusion_output(latency_ms=latency_ms)


@pytest.mark.parametrize(
    "evidence_ids,error",
    [
        (("evidence", "evidence"), "duplicates"),
        (("evidence", " "), "must not be blank"),
    ],
)
def test_fusion_output_rejects_invalid_evidence_ids(evidence_ids, error):
    with pytest.raises(ValidationError, match=error):
        _fusion_output(evidence_ids=evidence_ids)


@pytest.mark.parametrize(
    "field",
    ["output_id", "plan_id", "graph_id", "fusion_node_id", "method"],
)
def test_fusion_output_rejects_blank_identifiers_and_method(field):
    with pytest.raises(ValidationError, match="must not be blank"):
        _fusion_output(**{field: "\t"})


@pytest.mark.parametrize(
    "field,value",
    [("strategy", "unknown"), ("status", "complete")],
)
def test_fusion_output_rejects_unknown_enum_wire_values(field, value):
    with pytest.raises(ValidationError):
        _fusion_output(**{field: value})


@pytest.mark.parametrize(
    "status",
    [
        ExecutionStatus.PENDING,
        ExecutionStatus.READY,
        ExecutionStatus.RUNNING,
        ExecutionStatus.SKIPPED,
        ExecutionStatus.BLOCKED,
        ExecutionStatus.CANCELLED,
    ],
)
def test_fusion_output_rejects_nonterminal_statuses(status):
    with pytest.raises(ValidationError, match="must be succeeded or failed"):
        _fusion_output(status=status)


def test_fusion_output_has_exact_fields_and_edge_execution_tier():
    output = _fusion_output()
    serialized = output.model_dump(mode="json")

    assert set(serialized) == {
        "schema_version",
        "output_id",
        "plan_id",
        "graph_id",
        "fusion_node_id",
        "strategy",
        "status",
        "text",
        "method",
        "evidence_ids",
        "latency_ms",
        "error",
        "metadata",
    }
    assert output.execution_tier is Tier.EDGE
    assert "execution_tier" not in serialized


def test_fusion_output_rejects_unsupported_version_and_extra_fields():
    with pytest.raises(ValidationError):
        _fusion_output(schema_version="2.0")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _fusion_output(unexpected="value")


@pytest.mark.parametrize(
    "update,error",
    [
        ({"latency_ms": -1.0}, "latency_ms"),
        ({"text": ""}, "requires nonblank text"),
        ({"error": "unexpected"}, "must not contain an error"),
        (
            {"status": ExecutionStatus.FAILED, "text": "", "error": None},
            "requires a nonblank error",
        ),
        (
            {"evidence_ids": ("evidence", "evidence")},
            "duplicates",
        ),
        ({"status": "complete"}, "status"),
    ],
)
def test_fusion_output_model_copy_rejects_invalid_updates(update, error):
    with pytest.raises(ValidationError, match=error):
        _fusion_output().model_copy(update=update)


def test_fusion_output_model_copy_validates_updates_and_preserves_serialization():
    original = _fusion_output()

    copied = original.model_copy(update={"latency_ms": 1})
    ordinary_copy = original.model_copy()

    assert copied.latency_ms == 1.0
    assert ordinary_copy == original
    assert "execution_tier" not in copied.model_dump(mode="json")
