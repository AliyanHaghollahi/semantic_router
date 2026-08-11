"""Opt-in annotation contracts for the TierGraph planner.

This package contains data validation only. It does not load an encoder,
initialize a model, or import the baseline routing pipeline.

Learned planner modules (``model``, ``loss``, ``batching``) and
``GraphDecoder`` are imported from their submodules directly.
"""

from tiergraph.planner.annotations import (
    ImplicitResolution,
    OperationSpanLabel,
    PlannerExample,
    PlannerLabels,
    SlotAnchorLabel,
)
from tiergraph.planner.jsonl import (
    PlannerRecordValidationError,
    load_planner_jsonl,
    load_planner_record,
    validate_planner_records,
)

__all__ = [
    "ImplicitResolution",
    "OperationSpanLabel",
    "PlannerExample",
    "PlannerLabels",
    "PlannerRecordValidationError",
    "SlotAnchorLabel",
    "load_planner_jsonl",
    "load_planner_record",
    "validate_planner_records",
]
