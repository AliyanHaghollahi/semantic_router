"""Minimal JSON and JSONL loading utilities for planner annotations.

Collection and JSONL validation is fail-fast: loading stops at the first
malformed or invalid nonblank record.
"""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tiergraph.planner.annotations import PlannerExample


PlannerRecord = str | bytes | bytearray | Mapping[str, Any] | PlannerExample


class PlannerRecordValidationError(ValueError):
    """Validation failure annotated with its record number and example ID."""

    def __init__(
        self,
        record_number: int,
        example_id: str,
        cause: Exception,
    ) -> None:
        self.record_number = record_number
        self.example_id = example_id
        self.cause = cause
        super().__init__(
            f"planner record {record_number} "
            f"(example_id={example_id!r}) failed validation: {cause}"
        )


def load_planner_record(record: PlannerRecord) -> PlannerExample:
    """Load and validate one mapping or serialized JSON record."""
    if isinstance(record, PlannerExample):
        data = record.model_dump(mode="python", round_trip=True)
    elif isinstance(record, Mapping):
        data = dict(record)
    else:
        data = json.loads(record)
    return PlannerExample.model_validate(data)


def validate_planner_records(
    records: Iterable[PlannerRecord],
) -> tuple[PlannerExample, ...]:
    """Validate records, stopping at the first invalid one with its position."""
    validated: list[PlannerExample] = []
    for record_number, record in enumerate(records, start=1):
        validated.append(_validate_with_context(record, record_number))
    return tuple(validated)


def load_planner_jsonl(path: str | Path) -> tuple[PlannerExample, ...]:
    """Load UTF-8 JSONL, failing on the first invalid nonblank physical line."""
    validated: list[PlannerExample] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                validated.append(_validate_with_context(line, line_number))
    return tuple(validated)


def _validate_with_context(
    record: PlannerRecord,
    record_number: int,
) -> PlannerExample:
    example_id = _extract_example_id(record)
    try:
        return load_planner_record(record)
    except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as exc:
        raise PlannerRecordValidationError(
            record_number,
            example_id,
            exc,
        ) from exc


def _extract_example_id(record: PlannerRecord) -> str:
    if isinstance(record, PlannerExample):
        return record.example_id
    try:
        if isinstance(record, Mapping):
            data: object = record
        else:
            data = json.loads(record)
    except (json.JSONDecodeError, TypeError):
        return "<unknown>"
    if isinstance(data, dict):
        example_id = data.get("example_id")
        if isinstance(example_id, str) and example_id.strip():
            return example_id
    return "<unknown>"
