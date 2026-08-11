"""Versioned deterministic slot naming for planner-generated ExecutionGraphs.

``SLOT_NAMING_V1`` is not learned. Structures that cannot be named under this
convention raise ``SlotNamingError`` rather than falling back to heuristics.

Anchor-derived bases must already be V1-compatible (``[a-z0-9_]+`` after
casefold/whitespace collapse). Apostrophes, hyphens, and non-ASCII characters
are rejected rather than silently stripped or transliterated.

Anchorless answer operations use ``DEFAULT_BASE_BY_OPERATOR_V1`` — a fixed
operator→base table. Names are never inferred from arbitrary query text.
"""

from __future__ import annotations

import re

from tiergraph.enums import OperatorType, SlotType


NAMING_VERSION = "SLOT_NAMING_V1"

_NAME_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")

_SLOT_TYPE_SUFFIX: dict[SlotType, str] = {
    SlotType.RESOLVED_REFERENCE: "identifier",
    SlotType.PERSONAL_FACT: "fact",
    SlotType.PERSONAL_RECORD: "record",
    SlotType.ENVIRONMENTAL_FACT: "fact",
    SlotType.LOCATION: "location",
    SlotType.NAVIGATION_INSTRUCTION: "navigation",
    SlotType.SCENE_DESCRIPTION: "scene",
    SlotType.FINAL_RESPONSE: "response",
}

# Deterministic bases for explicit answer ops that own no slot anchor.
# Keep the mapping minimal and stable; do not derive names from query text.
DEFAULT_BASE_BY_OPERATOR_V1: dict[OperatorType, str] = {
    OperatorType.RESOLVE_PERSONAL: "reference",
    OperatorType.RETRIEVE_PERSONAL: "personal",
    OperatorType.IDENTIFY_ENVIRONMENTAL: "environment",
    OperatorType.LOCATE_ENVIRONMENTAL: "location",
    OperatorType.NAVIGATE_TO: "destination",
    OperatorType.DESCRIBE_ENVIRONMENT: "scene",
}


class SlotNamingError(ValueError):
    """Raised when SLOT_NAMING_V1 cannot represent a requested structure."""


def normalize_base_name(text: str) -> str:
    """Deterministically normalize a surface string into a slot base name.

    Only casefolding and whitespace→underscore collapsing are applied. Any
    remaining non-``[a-z0-9_]`` character (apostrophe, hyphen, Unicode, …)
    raises ``SlotNamingError``.
    """
    if type(text) is not str:
        raise TypeError("base name text must be a string")
    collapsed = "_".join(text.casefold().split())
    if not collapsed:
        raise SlotNamingError("slot base name must not be blank")
    if not _NAME_RE.fullmatch(collapsed):
        raise SlotNamingError(
            f"SLOT_NAMING_V1 cannot represent base name {text!r}; "
            "expected casefolded alphanumeric tokens separated by underscores"
        )
    return collapsed


def default_base_for_operator(operator: OperatorType) -> str:
    """Return the V1 default slot base for an anchorless answer operator."""
    if operator is OperatorType.FUSE:
        raise SlotNamingError(
            "FUSE has no answer-operation default base under SLOT_NAMING_V1"
        )
    try:
        return DEFAULT_BASE_BY_OPERATOR_V1[operator]
    except KeyError as exc:
        raise SlotNamingError(
            f"SLOT_NAMING_V1 has no default base for operator {operator.value}"
        ) from exc


def principal_slot_name(*, base_name: str, slot_type: SlotType) -> str:
    """Name the sole principal produced slot for an answer operation."""
    normalized = normalize_base_name(base_name)
    try:
        suffix = _SLOT_TYPE_SUFFIX[slot_type]
    except KeyError as exc:
        raise SlotNamingError(
            f"SLOT_NAMING_V1 has no suffix for slot type {slot_type!r}"
        ) from exc
    if slot_type is SlotType.FINAL_RESPONSE:
        # Answer principals never use FINAL_RESPONSE; keep explicit failure.
        raise SlotNamingError(
            "FINAL_RESPONSE is reserved for FUSE under SLOT_NAMING_V1"
        )
    return f"{normalized}_{suffix}"


def fuse_output_slot_name() -> str:
    """Deterministic sole FUSE produced-slot name."""
    return "response"


def fuse_input_slot_name(*, source_node_id: str, source_slot: str) -> str:
    """Name a FUSE required input that re-exports a sink principal slot."""
    if type(source_node_id) is not str or not source_node_id.strip():
        raise SlotNamingError("FUSE source_node_id must be nonblank")
    if type(source_slot) is not str or not source_slot.strip():
        raise SlotNamingError("FUSE source_slot must be nonblank")
    if "__" in source_node_id:
        raise SlotNamingError(
            "SLOT_NAMING_V1 cannot represent source_node_id containing '__'"
        )
    return f"{source_node_id}__{source_slot}"


__all__ = [
    "DEFAULT_BASE_BY_OPERATOR_V1",
    "NAMING_VERSION",
    "SlotNamingError",
    "default_base_for_operator",
    "fuse_input_slot_name",
    "fuse_output_slot_name",
    "normalize_base_name",
    "principal_slot_name",
]
