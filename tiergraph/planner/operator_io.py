"""Versioned operator I/O contract for learned explicit-operation dependencies.

``OPERATOR_IO_CONTRACT_V1`` is a modeling assumption for Phase-4 H7:

* each answer operation exposes exactly one principal output
* learned ``op_i -> op_j`` dependencies consume that principal output only
* target required inputs are taken only from the declared allowed input types

Future annotation needs that cannot be expressed here require a documented V2
contract rather than silent decoder heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass

from tiergraph.enums import OperatorType, SlotType
from tiergraph.planner.annotations import _PRINCIPAL_OUTPUT_TYPES


CONTRACT_VERSION = "OPERATOR_IO_CONTRACT_V1"


@dataclass(frozen=True, slots=True)
class OperatorIOSpec:
    """Principal output and allowed learned-input types for one operator."""

    principal_output: SlotType
    allowed_learned_input_types: frozenset[SlotType]


def _spec(
    operator: OperatorType,
    allowed_learned_input_types: frozenset[SlotType],
) -> OperatorIOSpec:
    return OperatorIOSpec(
        principal_output=_PRINCIPAL_OUTPUT_TYPES[operator],
        allowed_learned_input_types=allowed_learned_input_types,
    )


OPERATOR_IO_CONTRACT_V1: dict[OperatorType, OperatorIOSpec] = {
    OperatorType.RESOLVE_PERSONAL: _spec(
        OperatorType.RESOLVE_PERSONAL,
        frozenset(),
    ),
    OperatorType.RETRIEVE_PERSONAL: _spec(
        OperatorType.RETRIEVE_PERSONAL,
        frozenset({SlotType.RESOLVED_REFERENCE}),
    ),
    OperatorType.IDENTIFY_ENVIRONMENTAL: _spec(
        OperatorType.IDENTIFY_ENVIRONMENTAL,
        frozenset({SlotType.RESOLVED_REFERENCE}),
    ),
    OperatorType.LOCATE_ENVIRONMENTAL: _spec(
        OperatorType.LOCATE_ENVIRONMENTAL,
        frozenset(
            {
                SlotType.RESOLVED_REFERENCE,
                SlotType.ENVIRONMENTAL_FACT,
            }
        ),
    ),
    OperatorType.NAVIGATE_TO: _spec(
        OperatorType.NAVIGATE_TO,
        frozenset(
            {
                SlotType.LOCATION,
                SlotType.RESOLVED_REFERENCE,
            }
        ),
    ),
    OperatorType.DESCRIBE_ENVIRONMENT: _spec(
        OperatorType.DESCRIBE_ENVIRONMENT,
        frozenset(
            {
                SlotType.RESOLVED_REFERENCE,
                SlotType.LOCATION,
                SlotType.ENVIRONMENTAL_FACT,
            }
        ),
    ),
}


_ANSWER_OPERATORS = frozenset(OPERATOR_IO_CONTRACT_V1)


def principal_output_type(operator: OperatorType) -> SlotType:
    """Return the V1 principal output SlotType for an answer operator."""
    if operator is OperatorType.FUSE:
        raise ValueError("FUSE is not an answer operator in OPERATOR_IO_CONTRACT_V1")
    try:
        return _PRINCIPAL_OUTPUT_TYPES[operator]
    except KeyError as exc:
        raise ValueError(
            f"unsupported answer operator for principal output: {operator.value}"
        ) from exc


def operator_io_spec(operator: OperatorType) -> OperatorIOSpec:
    """Return the V1 I/O spec for an answer operator."""
    try:
        return OPERATOR_IO_CONTRACT_V1[operator]
    except KeyError as exc:
        raise ValueError(
            f"operator is outside OPERATOR_IO_CONTRACT_V1: {operator.value}"
        ) from exc


def is_h7_pair_eligible(
    source_operator: OperatorType,
    target_operator: OperatorType,
) -> bool:
    """Whether an ordered explicit-operation pair may carry a learned H7 edge.

    Eligibility is purely structural at the operator-type level: the source
    principal ``SlotType`` must be an allowed learned input of the target.
    Non-answer operators (including ``FUSE``) are never eligible.

    Same-index self-loops are filtered by callers (``targets`` / ``decode``);
    this helper does not receive node indices and therefore does not reject
    distinct nodes that share an operator type. No slot-name heuristics are
    applied.
    """
    if source_operator is OperatorType.FUSE or target_operator is OperatorType.FUSE:
        return False
    if source_operator not in _ANSWER_OPERATORS or target_operator not in _ANSWER_OPERATORS:
        return False
    source_principal = principal_output_type(source_operator)
    allowed = OPERATOR_IO_CONTRACT_V1[target_operator].allowed_learned_input_types
    return source_principal in allowed


def h7_dependency_slot_type(
    source_operator: OperatorType,
    target_operator: OperatorType,
) -> SlotType:
    """Return the unique SlotType carried by an eligible H7 dependency.

    Raises ``ValueError`` when the pair is structurally ineligible. Callers that
    need decode-time failures should catch and wrap as ``PlannerDecodeError``.
    """
    if not is_h7_pair_eligible(source_operator, target_operator):
        raise ValueError(
            "H7 pair is structurally ineligible under OPERATOR_IO_CONTRACT_V1: "
            f"{source_operator.value} -> {target_operator.value}"
        )
    return principal_output_type(source_operator)


__all__ = [
    "CONTRACT_VERSION",
    "OPERATOR_IO_CONTRACT_V1",
    "OperatorIOSpec",
    "h7_dependency_slot_type",
    "is_h7_pair_eligible",
    "operator_io_spec",
    "principal_output_type",
]
