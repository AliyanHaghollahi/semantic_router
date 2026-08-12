"""Step-A Stage-A planner annotation: operations, operator types, anchors.

Step A covers H2/H3/H4 span supervision only. H5/H6/H7 belong to Step B.
Humans select exact substrings; the tool computes character offsets.

For implicit personal references (e.g. "my gate"), mark an H4 anchor in Step A;
do not annotate RESOLVE_PERSONAL as an H2/H3 operation. H5/decoder synthesize
the implicit resolver in Step B. Prefer the smallest complete referring phrase
for H4 (preserve possessives/determiners: "my gate", "this medication").
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from tiergraph.enums import OperatorType, QueryType, _CanonicalWireEnum
from tiergraph.models import TierGraphSchema
from tiergraph.planner.align import (
    TokenCharSpan,
    TruncationKind,
    align_char_span,
    encode_bio_labels,
)
from tiergraph.planner.corpus import normalize_query_key
from tiergraph.planner.operator_io import OPERATOR_IO_CONTRACT_V1
from tiergraph.planner.stage_a_selection import (
    DEFAULT_SELECTION_PATH,
    load_jsonl as load_selection_jsonl,
)


DEFAULT_STEP_A_ANNOTATIONS_PATH = Path(
    "dataset/planner/stage_a_step_a_annotations.jsonl"
)
DEFAULT_FROZEN_SELECTION_PATH = DEFAULT_SELECTION_PATH
EXPECTED_STAGE_A_COUNT = 120

ANSWER_OPERATORS: tuple[OperatorType, ...] = tuple(OPERATOR_IO_CONTRACT_V1.keys())

RESOLVE_PERSONAL_IMPLICIT_WARNING = (
    "For implicit personal references such as 'my gate', 'my appointment', "
    "or 'my reserved seat', do NOT annotate RESOLVE_PERSONAL as an H2 operation. "
    "Mark the explicit reference as an H4 anchor in Step A. "
    "H5/decoder will synthesize the implicit resolver in Step B. "
    "Choose RESOLVE_PERSONAL as H3 only when the query itself contains a genuinely "
    "explicit personal-resolution answer operation supported by the annotation "
    "contract."
)

H4_ANCHOR_GUIDANCE = (
    "H4 anchors: prefer the smallest complete referring phrase, preserving "
    "meaningful determiners/possessives "
    '(e.g. "my gate" not just "gate"; "this medication" not just "medication"; '
    '"my appointment" not just "appointment") when those modifiers are '
    "semantically part of the reference."
)

OPERATOR_HELP: tuple[tuple[OperatorType, str], ...] = (
    (
        OperatorType.RESOLVE_PERSONAL,
        "Resolve a personal reference to a concrete entity "
        "(explicit personal-resolution answer operation only). "
        + RESOLVE_PERSONAL_IMPLICIT_WARNING,
    ),
    (
        OperatorType.RETRIEVE_PERSONAL,
        "Retrieve a personal fact/record "
        '(e.g. allergies, appointment time, reservation).',
    ),
    (
        OperatorType.IDENTIFY_ENVIRONMENTAL,
        "Identify/classify something in the current scene "
        '(e.g. "what medication is this").',
    ),
    (
        OperatorType.LOCATE_ENVIRONMENTAL,
        "Locate an entity/place in the environment "
        '(e.g. "where is the pharmacy counter").',
    ),
    (
        OperatorType.NAVIGATE_TO,
        "Produce navigation instructions to a destination "
        '(e.g. "how do I get there").',
    ),
    (
        OperatorType.DESCRIBE_ENVIRONMENT,
        "Read/describe scene text or appearance "
        '(e.g. "what does this sign say").',
    ),
)

BUCKET_TO_QUERY_TYPE: dict[str, QueryType] = {
    "Personal": QueryType.PERSONAL,
    "Environmental": QueryType.ENVIRONMENTAL,
    "MIXED_IMPLICIT": QueryType.MIXED,
    "MIXED_PARALLEL": QueryType.MIXED,
    "MIXED_SEQUENTIAL": QueryType.MIXED,
}


class StepAStatus(_CanonicalWireEnum, str, Enum):
    UNREVIEWED = "UNREVIEWED"
    COMPLETE = "COMPLETE"


class _StepASchema(TierGraphSchema):
    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update is None:
            return super().model_copy(deep=deep)
        data = self.model_dump(mode="python", round_trip=True)
        update_data = dict(update)
        if deep:
            data = deepcopy(data)
            update_data = deepcopy(update_data)
        data.update(update_data)
        return type(self).model_validate(data)


class StepAOperation(_StepASchema):
    """One Step-A answer operation with exact character offsets."""

    operation_index: int = Field(ge=0)
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    operator_type: OperatorType

    @field_validator("text")
    @classmethod
    def _nonblank_text(cls, value: str) -> str:
        if type(value) is not str or not value:
            raise ValueError("operation text must be a nonempty string")
        return value

    @model_validator(mode="after")
    def _validate_operation(self) -> "StepAOperation":
        if self.char_end <= self.char_start:
            raise ValueError("operation span must be nonempty")
        if self.operator_type is OperatorType.FUSE:
            raise ValueError("operation spans cannot use FUSE")
        if self.operator_type not in OPERATOR_IO_CONTRACT_V1:
            raise ValueError(
                f"operator_type outside V1 answer set: {self.operator_type.value}"
            )
        return self


class StepAAnchor(_StepASchema):
    """One Step-A anchor/reference mention (ownership is Step B)."""

    anchor_index: int = Field(ge=0)
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @field_validator("text")
    @classmethod
    def _nonblank_text(cls, value: str) -> str:
        if type(value) is not str or not value:
            raise ValueError("anchor text must be a nonempty string")
        return value

    @model_validator(mode="after")
    def _validate_anchor(self) -> "StepAAnchor":
        if self.char_end <= self.char_start:
            raise ValueError("anchor span must be nonempty")
        return self


class StageAStepAAnnotation(_StepASchema):
    """One frozen Stage-A example with optional Step-A span annotations."""

    stage_a_id: str
    source_id: str | None = None
    candidate_id: str | None = None
    query: str
    final_bucket: str
    source_kind: str
    semantic_group: str
    template_group: str
    provenance: dict[str, Any]
    derived_query_type: QueryType
    operations: tuple[StepAOperation, ...] = ()
    anchors: tuple[StepAAnchor, ...] = ()
    step_a_status: StepAStatus = StepAStatus.UNREVIEWED

    @field_validator(
        "stage_a_id",
        "query",
        "final_bucket",
        "source_kind",
        "semantic_group",
        "template_group",
    )
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if type(value) is not str or not value.strip():
            raise ValueError("must be a nonblank string")
        return value

    @field_validator("operations", "anchors", mode="before")
    @classmethod
    def _tupleize(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_record(self) -> "StageAStepAAnnotation":
        if self.source_id is None and self.candidate_id is None:
            raise ValueError("source_id or candidate_id is required")
        expected_qt = derive_query_type(self.final_bucket)
        if self.derived_query_type is not expected_qt:
            raise ValueError(
                f"derived_query_type {self.derived_query_type.value!r} does not "
                f"match final_bucket {self.final_bucket!r} "
                f"(expected {expected_qt.value!r})"
            )
        _validate_annotation_spans(self, require_complete=False)
        if self.step_a_status is StepAStatus.COMPLETE:
            _validate_annotation_spans(self, require_complete=True)
        return self


def derive_query_type(final_bucket: str) -> QueryType:
    try:
        return BUCKET_TO_QUERY_TYPE[final_bucket]
    except KeyError as exc:
        raise ValueError(f"unknown final_bucket for H1 derivation: {final_bucket!r}") from exc


def find_substring_occurrences(query: str, substring: str) -> list[tuple[int, int]]:
    """Return all exact ``[start, end)`` occurrences of substring in query."""
    if type(query) is not str or type(substring) is not str:
        raise TypeError("query and substring must be strings")
    if not substring:
        raise ValueError("substring must be nonempty")
    occurrences: list[tuple[int, int]] = []
    start = 0
    while True:
        index = query.find(substring, start)
        if index < 0:
            break
        occurrences.append((index, index + len(substring)))
        start = index + 1
    return occurrences


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Half-open interval overlap."""
    return a_start < b_end and b_start < a_end


def _validate_annotation_spans(
    record: StageAStepAAnnotation,
    *,
    require_complete: bool,
) -> None:
    query = record.query
    query_len = len(query)
    ops = list(record.operations)
    anchors = list(record.anchors)

    if require_complete and not ops:
        raise ValueError(
            f"{record.stage_a_id}: COMPLETE requires at least one operation"
        )

    for index, op in enumerate(ops):
        if op.operation_index != index:
            raise ValueError(
                f"{record.stage_a_id}: operation_index must be contiguous "
                f"(expected {index}, got {op.operation_index})"
            )
        if op.char_end > query_len:
            raise ValueError(
                f"{record.stage_a_id}: operation bounds exceed query length"
            )
        if query[op.char_start:op.char_end] != op.text:
            raise ValueError(
                f"{record.stage_a_id}: operation text does not match "
                f"query[{op.char_start}:{op.char_end}]"
            )

    for index, anchor in enumerate(anchors):
        if anchor.anchor_index != index:
            raise ValueError(
                f"{record.stage_a_id}: anchor_index must be contiguous "
                f"(expected {index}, got {anchor.anchor_index})"
            )
        if anchor.char_end > query_len:
            raise ValueError(
                f"{record.stage_a_id}: anchor bounds exceed query length"
            )
        if query[anchor.char_start:anchor.char_end] != anchor.text:
            raise ValueError(
                f"{record.stage_a_id}: anchor text does not match "
                f"query[{anchor.char_start}:{anchor.char_end}]"
            )

    for i, left in enumerate(ops):
        for right in ops[i + 1 :]:
            if spans_overlap(
                left.char_start, left.char_end, right.char_start, right.char_end
            ):
                raise ValueError(
                    f"{record.stage_a_id}: operation spans must not overlap "
                    f"(op {left.operation_index} vs {right.operation_index})"
                )

    for i, left in enumerate(anchors):
        for right in anchors[i + 1 :]:
            if spans_overlap(
                left.char_start, left.char_end, right.char_start, right.char_end
            ):
                raise ValueError(
                    f"{record.stage_a_id}: anchor spans must not overlap "
                    f"(anchor {left.anchor_index} vs {right.anchor_index})"
                )

    # Stable ordering by text position.
    if ops != sorted(ops, key=lambda item: (item.char_start, item.char_end)):
        raise ValueError(
            f"{record.stage_a_id}: operations must be ordered by text position"
        )
    if anchors != sorted(
        anchors, key=lambda item: (item.char_start, item.char_end)
    ):
        raise ValueError(
            f"{record.stage_a_id}: anchors must be ordered by text position"
        )


def reindex_operations(
    operations: Sequence[StepAOperation],
) -> tuple[StepAOperation, ...]:
    ordered = sorted(operations, key=lambda item: (item.char_start, item.char_end))
    return tuple(
        op.model_copy(update={"operation_index": index})
        for index, op in enumerate(ordered)
    )


def reindex_anchors(anchors: Sequence[StepAAnchor]) -> tuple[StepAAnchor, ...]:
    ordered = sorted(anchors, key=lambda item: (item.char_start, item.char_end))
    return tuple(
        anchor.model_copy(update={"anchor_index": index})
        for index, anchor in enumerate(ordered)
    )


def format_operator_help() -> str:
    lines = ["OperatorType help (OPERATOR_IO_CONTRACT_V1 answer ops):", ""]
    for index, (operator, description) in enumerate(OPERATOR_HELP, start=1):
        lines.append(f"{index}. {operator.value}")
        if operator is OperatorType.RESOLVE_PERSONAL:
            short = (
                "Resolve a personal reference to a concrete entity "
                "(explicit personal-resolution answer operation only)."
            )
            lines.append(f"   {short}")
            lines.append(f"   WARNING: {RESOLVE_PERSONAL_IMPLICIT_WARNING}")
        else:
            lines.append(f"   {description}")
        lines.append("")
    lines.append("FUSE is not a learned Step-A operation annotation.")
    lines.append("")
    lines.append(H4_ANCHOR_GUIDANCE)
    return "\n".join(lines)


def format_query_with_word_indexes(query: str) -> str:
    """Show simple whitespace word indexes for human readability."""
    parts: list[str] = []
    for index, match in enumerate(re.finditer(r"\S+", query)):
        parts.append(f"[{index}:{match.group()}]")
    return " ".join(parts)


def create_operation_from_substring(
    query: str,
    substring: str,
    operator_type: OperatorType | str,
    *,
    occurrence: int = 0,
    existing_operations: Sequence[StepAOperation] = (),
) -> StepAOperation:
    """Build an operation from an exact substring and occurrence index."""
    if isinstance(operator_type, str):
        try:
            operator_type = OperatorType(operator_type)
        except ValueError as exc:
            raise ValueError(f"invalid OperatorType: {operator_type!r}") from exc
    if operator_type is OperatorType.FUSE:
        raise ValueError("invalid OperatorType: FUSE")
    if operator_type not in OPERATOR_IO_CONTRACT_V1:
        raise ValueError(f"invalid OperatorType: {operator_type.value}")

    occurrences = find_substring_occurrences(query, substring)
    if not occurrences:
        raise ValueError(f"substring not found in query: {substring!r}")
    if occurrence < 0 or occurrence >= len(occurrences):
        raise ValueError(
            f"occurrence {occurrence} out of range for {len(occurrences)} matches"
        )
    char_start, char_end = occurrences[occurrence]
    candidate = StepAOperation(
        operation_index=len(existing_operations),
        text=query[char_start:char_end],
        char_start=char_start,
        char_end=char_end,
        operator_type=operator_type,
    )
    trial = reindex_operations(tuple(existing_operations) + (candidate,))
    # Validate via temporary record-like checks.
    for i, left in enumerate(trial):
        for right in trial[i + 1 :]:
            if spans_overlap(
                left.char_start, left.char_end, right.char_start, right.char_end
            ):
                raise ValueError("operation spans must not overlap")
    # Return with provisional index; caller reindexes full set.
    return candidate


def create_anchor_from_substring(
    query: str,
    substring: str,
    *,
    occurrence: int = 0,
    existing_anchors: Sequence[StepAAnchor] = (),
) -> StepAAnchor:
    occurrences = find_substring_occurrences(query, substring)
    if not occurrences:
        raise ValueError(f"substring not found in query: {substring!r}")
    if occurrence < 0 or occurrence >= len(occurrences):
        raise ValueError(
            f"occurrence {occurrence} out of range for {len(occurrences)} matches"
        )
    char_start, char_end = occurrences[occurrence]
    candidate = StepAAnchor(
        anchor_index=len(existing_anchors),
        text=query[char_start:char_end],
        char_start=char_start,
        char_end=char_end,
    )
    trial = reindex_anchors(tuple(existing_anchors) + (candidate,))
    for i, left in enumerate(trial):
        for right in trial[i + 1 :]:
            if spans_overlap(
                left.char_start, left.char_end, right.char_start, right.char_end
            ):
                raise ValueError("anchor spans must not overlap")
    return candidate


def check_span_alignment(
    query: str,
    spans: Sequence[tuple[int, int, str]],
    tokens: Sequence[TokenCharSpan],
) -> list[dict[str, Any]]:
    """Return alignment issues for (start, end, label) spans. Empty if all OK."""
    issues: list[dict[str, Any]] = []
    for start, end, label in spans:
        alignment = align_char_span(start, end, tokens)
        if not alignment.representable:
            issues.append(
                {
                    "label": label,
                    "text": query[start:end],
                    "char_start": start,
                    "char_end": end,
                    "truncation_kind": alignment.truncation_kind.value,
                    "representable": False,
                }
            )
    # Also ensure within-head BIO conflicts are impossible for ops/anchors
    # when both sets are checked separately by callers.
    return issues


def check_bio_conflicts(
    spans: Sequence[tuple[int, int]],
    tokens: Sequence[TokenCharSpan],
) -> None:
    """Raise if aligned spans conflict under BIO encoding."""
    alignments = [align_char_span(start, end, tokens) for start, end in spans]
    representable = [item for item in alignments if item.representable]
    if representable:
        encode_bio_labels(representable, tokens, allow_token_conflicts=False)


def initialize_step_a_annotations_from_selection(
    selection_rows: Sequence[Mapping[str, Any]],
) -> tuple[StageAStepAAnnotation, ...]:
    if len(selection_rows) != EXPECTED_STAGE_A_COUNT:
        raise ValueError(
            f"frozen selection must contain {EXPECTED_STAGE_A_COUNT} rows, "
            f"got {len(selection_rows)}"
        )
    records: list[StageAStepAAnnotation] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for row in selection_rows:
        stage_a_id = str(row["stage_a_id"])
        query = str(row["query"])
        if stage_a_id in seen_ids:
            raise ValueError(f"duplicate stage_a_id in selection: {stage_a_id}")
        key = normalize_query_key(query)
        if key in seen_queries:
            raise ValueError(f"duplicate query in selection: {stage_a_id}")
        seen_ids.add(stage_a_id)
        seen_queries.add(key)
        if row.get("selected") is not True:
            raise ValueError(f"{stage_a_id} is not selected=true in frozen manifest")
        records.append(
            StageAStepAAnnotation(
                stage_a_id=stage_a_id,
                source_id=row.get("source_id"),
                candidate_id=row.get("candidate_id"),
                query=query,
                final_bucket=str(row["final_bucket"]),
                source_kind=str(row["source_kind"]),
                semantic_group=str(row["semantic_group"]),
                template_group=str(row["template_group"]),
                provenance=dict(row.get("provenance") or {}),
                derived_query_type=derive_query_type(str(row["final_bucket"])),
                operations=(),
                anchors=(),
                step_a_status=StepAStatus.UNREVIEWED,
            )
        )
    records.sort(key=lambda item: item.stage_a_id)
    return tuple(records)


def load_step_a_annotations(
    path: str | Path,
) -> tuple[StageAStepAAnnotation, ...]:
    path = Path(path)
    if not path.is_file():
        return ()
    records: list[StageAStepAAnnotation] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(StageAStepAAnnotation.model_validate(json.loads(line)))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"invalid Step-A annotation at {path}:{line_number}: {exc}"
            ) from exc
    records.sort(key=lambda item: item.stage_a_id)
    return tuple(records)


def write_step_a_annotations(
    path: str | Path,
    records: Sequence[StageAStepAAnnotation],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda item: item.stage_a_id)
    lines = [
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for item in ordered
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def ensure_step_a_annotations_initialized(
    *,
    selection_path: str | Path = DEFAULT_FROZEN_SELECTION_PATH,
    annotations_path: str | Path = DEFAULT_STEP_A_ANNOTATIONS_PATH,
) -> tuple[StageAStepAAnnotation, ...]:
    """Load annotations or initialize 120 UNREVIEWED rows from frozen selection."""
    selection_path = Path(selection_path)
    annotations_path = Path(annotations_path)
    selection_rows = load_selection_jsonl(selection_path)
    frozen_ids = [str(row["stage_a_id"]) for row in selection_rows]
    frozen_queries = {str(row["stage_a_id"]): str(row["query"]) for row in selection_rows}

    existing = load_step_a_annotations(annotations_path)
    if not existing:
        initialized = initialize_step_a_annotations_from_selection(selection_rows)
        write_step_a_annotations(annotations_path, initialized)
        return initialized

    by_id = {item.stage_a_id: item for item in existing}
    if sorted(by_id) != sorted(frozen_ids):
        raise ValueError(
            "Step-A annotation stage_a_ids do not match frozen selection"
        )
    for stage_a_id, query in frozen_queries.items():
        if by_id[stage_a_id].query != query:
            raise ValueError(
                f"Step-A annotation query drift for {stage_a_id}"
            )
    return tuple(by_id[stage_a_id] for stage_a_id in sorted(frozen_ids))


def summarize_step_a_progress(
    records: Sequence[StageAStepAAnnotation],
) -> dict[str, Any]:
    status_counts = Counter(item.step_a_status.value for item in records)
    by_bucket: dict[str, dict[str, int]] = {}
    for item in records:
        bucket = by_bucket.setdefault(
            item.final_bucket, {"COMPLETE": 0, "UNREVIEWED": 0, "total": 0}
        )
        bucket["total"] += 1
        bucket[item.step_a_status.value] += 1
    op_type_counts = Counter(
        op.operator_type.value for item in records for op in item.operations
    )
    return {
        "total": len(records),
        "COMPLETE": status_counts.get(StepAStatus.COMPLETE.value, 0),
        "UNREVIEWED": status_counts.get(StepAStatus.UNREVIEWED.value, 0),
        "by_final_bucket": by_bucket,
        "total_operations": sum(len(item.operations) for item in records),
        "total_anchors": sum(len(item.anchors) for item in records),
        "operator_type_counts": dict(sorted(op_type_counts.items())),
    }


def format_step_a_progress(summary: Mapping[str, Any]) -> str:
    lines = [
        f"Total: {summary['total']}",
        f"COMPLETE: {summary['COMPLETE']}",
        f"UNREVIEWED: {summary['UNREVIEWED']}",
        "By final_bucket:",
    ]
    for bucket in (
        "Personal",
        "Environmental",
        "MIXED_IMPLICIT",
        "MIXED_PARALLEL",
        "MIXED_SEQUENTIAL",
    ):
        stats = summary["by_final_bucket"].get(bucket, {})
        lines.append(
            f"  {bucket}: COMPLETE={stats.get('COMPLETE', 0)} "
            f"UNREVIEWED={stats.get('UNREVIEWED', 0)} "
            f"total={stats.get('total', 0)}"
        )
    lines.append(f"total_operations: {summary['total_operations']}")
    lines.append(f"total_anchors: {summary['total_anchors']}")
    if summary["operator_type_counts"]:
        lines.append("operator_type_counts:")
        for name, count in summary["operator_type_counts"].items():
            lines.append(f"  {name}: {count}")
    return "\n".join(lines)


def format_annotation_preview(record: StageAStepAAnnotation) -> str:
    lines = [
        f"stage_a_id: {record.stage_a_id}",
        f"final_bucket: {record.final_bucket}",
        f"derived_query_type: {record.derived_query_type.value}",
        f"status: {record.step_a_status.value}",
        f"query: {record.query}",
        "words: " + format_query_with_word_indexes(record.query),
        "operations:",
    ]
    if not record.operations:
        lines.append("  (none)")
    for op in record.operations:
        lines.append(
            f"  [{op.operation_index}] {op.operator_type.value} "
            f"{op.char_start}:{op.char_end} {op.text!r}"
        )
    lines.append("anchors:")
    if not record.anchors:
        lines.append("  (none)")
    for anchor in record.anchors:
        lines.append(
            f"  [{anchor.anchor_index}] {anchor.char_start}:{anchor.char_end} "
            f"{anchor.text!r}"
        )
    return "\n".join(lines)


TokenViewFactory = Callable[[str], tuple[TokenCharSpan, ...]]


def validate_step_a_corpus(
    records: Sequence[StageAStepAAnnotation],
    *,
    selection_path: str | Path = DEFAULT_FROZEN_SELECTION_PATH,
    token_view_factory: TokenViewFactory | None = None,
    require_all_complete: bool = False,
) -> list[str]:
    """Return validation errors (empty if OK)."""
    errors: list[str] = []
    selection_rows = load_selection_jsonl(selection_path)
    if len(records) != EXPECTED_STAGE_A_COUNT:
        errors.append(f"annotation count {len(records)} != {EXPECTED_STAGE_A_COUNT}")
    if len(selection_rows) != EXPECTED_STAGE_A_COUNT:
        errors.append(
            f"frozen selection count {len(selection_rows)} != {EXPECTED_STAGE_A_COUNT}"
        )

    frozen_by_id = {str(row["stage_a_id"]): row for row in selection_rows}
    ann_by_id = {item.stage_a_id: item for item in records}
    if sorted(ann_by_id) != sorted(frozen_by_id):
        errors.append("stage_a_id set does not match frozen selection")

    query_keys = [normalize_query_key(item.query) for item in records]
    if len(query_keys) != len(set(query_keys)):
        errors.append("duplicate normalized queries in annotations")
    if len(ann_by_id) != len(records):
        errors.append("duplicate stage_a_id in annotations")

    for stage_a_id, frozen in frozen_by_id.items():
        item = ann_by_id.get(stage_a_id)
        if item is None:
            errors.append(f"missing annotation for {stage_a_id}")
            continue
        if item.query != frozen["query"]:
            errors.append(f"{stage_a_id}: query differs from frozen selection")
        if item.final_bucket != frozen["final_bucket"]:
            errors.append(f"{stage_a_id}: final_bucket differs from frozen selection")
        if item.source_kind != frozen["source_kind"]:
            errors.append(f"{stage_a_id}: source_kind differs from frozen selection")
        if item.semantic_group != frozen["semantic_group"]:
            errors.append(f"{stage_a_id}: semantic_group differs from frozen selection")
        if item.template_group != frozen["template_group"]:
            errors.append(f"{stage_a_id}: template_group differs from frozen selection")
        if item.source_id != frozen.get("source_id"):
            errors.append(f"{stage_a_id}: source_id differs from frozen selection")
        if item.candidate_id != frozen.get("candidate_id"):
            errors.append(f"{stage_a_id}: candidate_id differs from frozen selection")
        if item.derived_query_type is not derive_query_type(item.final_bucket):
            errors.append(f"{stage_a_id}: bad derived_query_type")
        try:
            _validate_annotation_spans(
                item,
                require_complete=(item.step_a_status is StepAStatus.COMPLETE),
            )
        except ValueError as exc:
            errors.append(str(exc))
        if require_all_complete and item.step_a_status is not StepAStatus.COMPLETE:
            errors.append(f"{stage_a_id}: not COMPLETE")

        if token_view_factory is not None and item.operations:
            tokens = token_view_factory(item.query)
            span_specs = [
                (op.char_start, op.char_end, f"operation[{op.operation_index}]")
                for op in item.operations
            ] + [
                (
                    anchor.char_start,
                    anchor.char_end,
                    f"anchor[{anchor.anchor_index}]",
                )
                for anchor in item.anchors
            ]
            for issue in check_span_alignment(item.query, span_specs, tokens):
                errors.append(
                    f"{stage_a_id}: unrepresentable span "
                    f"{issue['label']} {issue['text']!r} "
                    f"[{issue['char_start']}:{issue['char_end']}] "
                    f"truncation={issue['truncation_kind']}"
                )
            try:
                check_bio_conflicts(
                    [(op.char_start, op.char_end) for op in item.operations],
                    tokens,
                )
                check_bio_conflicts(
                    [(a.char_start, a.char_end) for a in item.anchors],
                    tokens,
                )
            except ValueError as exc:
                errors.append(f"{stage_a_id}: BIO conflict: {exc}")

    return errors


def fingerprint_file(path: str | Path) -> tuple[int, str]:
    import hashlib

    payload = Path(path).read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


class StepAAnnotationSession:
    """Resumable Step-A annotation over frozen Stage-A examples."""

    def __init__(
        self,
        records: Sequence[StageAStepAAnnotation],
        *,
        annotations_path: str | Path,
        selection_path: str | Path = DEFAULT_FROZEN_SELECTION_PATH,
    ) -> None:
        selection_rows = load_selection_jsonl(selection_path)
        if len(records) != EXPECTED_STAGE_A_COUNT:
            raise ValueError(
                f"expected {EXPECTED_STAGE_A_COUNT} annotations, got {len(records)}"
            )
        if len(selection_rows) != EXPECTED_STAGE_A_COUNT:
            raise ValueError(
                f"frozen selection must contain {EXPECTED_STAGE_A_COUNT} rows"
            )
        frozen_by_id = {str(row["stage_a_id"]): row for row in selection_rows}
        record_ids = sorted(item.stage_a_id for item in records)
        if record_ids != sorted(frozen_by_id):
            raise ValueError("annotation IDs must match frozen selection")
        for item in records:
            frozen = frozen_by_id[item.stage_a_id]
            if item.query != frozen["query"]:
                raise ValueError(
                    f"query drift vs frozen selection: {item.stage_a_id}"
                )

        ordered = tuple(sorted(records, key=lambda item: item.stage_a_id))
        self._records = {item.stage_a_id: item for item in ordered}
        self._order = [item.stage_a_id for item in ordered]
        self.annotations_path = Path(annotations_path)
        self.selection_path = Path(selection_path)
        self._frozen_fingerprint = fingerprint_file(self.selection_path)
        self._history: list[str] = []
        self._cursor = self._first_unreviewed_index()

    @property
    def records(self) -> tuple[StageAStepAAnnotation, ...]:
        return tuple(self._records[stage_a_id] for stage_a_id in self._order)

    def _first_unreviewed_index(self) -> int:
        for index, stage_a_id in enumerate(self._order):
            if self._records[stage_a_id].step_a_status is StepAStatus.UNREVIEWED:
                return index
        return len(self._order)

    def current(self) -> StageAStepAAnnotation | None:
        if self._cursor >= len(self._order):
            return None
        return self._records[self._order[self._cursor]]

    def current_position(self) -> tuple[int, int]:
        if self._cursor >= len(self._order):
            return len(self._order), len(self._order)
        return self._cursor + 1, len(self._order)

    def summary(self) -> dict[str, Any]:
        return summarize_step_a_progress(self.records)

    def save(self) -> None:
        if fingerprint_file(self.selection_path) != self._frozen_fingerprint:
            raise ValueError(
                "frozen Stage-A selection file was modified; refusing to save"
            )
        write_step_a_annotations(self.annotations_path, self.records)

    def _update_current(
        self,
        updater: Callable[[StageAStepAAnnotation], StageAStepAAnnotation],
        *,
        persist: bool = True,
        track_history: bool = True,
    ) -> StageAStepAAnnotation:
        current = self.current()
        if current is None:
            raise ValueError("no current example")
        updated = updater(current)
        self._records[updated.stage_a_id] = updated
        if track_history and (
            not self._history or self._history[-1] != updated.stage_a_id
        ):
            self._history.append(updated.stage_a_id)
        if persist:
            self.save()
        return updated

    def add_operation(
        self,
        substring: str,
        operator_type: OperatorType | str,
        *,
        occurrence: int = 0,
        persist: bool = True,
    ) -> StageAStepAAnnotation:
        def updater(record: StageAStepAAnnotation) -> StageAStepAAnnotation:
            op = create_operation_from_substring(
                record.query,
                substring,
                operator_type,
                occurrence=occurrence,
                existing_operations=record.operations,
            )
            operations = reindex_operations(record.operations + (op,))
            return record.model_copy(
                update={
                    "operations": operations,
                    "step_a_status": StepAStatus.UNREVIEWED,
                }
            )

        return self._update_current(updater, persist=persist)

    def remove_operation(
        self, operation_index: int, *, persist: bool = True
    ) -> StageAStepAAnnotation:
        def updater(record: StageAStepAAnnotation) -> StageAStepAAnnotation:
            kept = [
                op
                for op in record.operations
                if op.operation_index != operation_index
            ]
            if len(kept) == len(record.operations):
                raise ValueError(f"unknown operation_index: {operation_index}")
            return record.model_copy(
                update={
                    "operations": reindex_operations(kept),
                    "step_a_status": StepAStatus.UNREVIEWED,
                }
            )

        return self._update_current(updater, persist=persist)

    def add_anchor(
        self,
        substring: str,
        *,
        occurrence: int = 0,
        persist: bool = True,
    ) -> StageAStepAAnnotation:
        def updater(record: StageAStepAAnnotation) -> StageAStepAAnnotation:
            anchor = create_anchor_from_substring(
                record.query,
                substring,
                occurrence=occurrence,
                existing_anchors=record.anchors,
            )
            anchors = reindex_anchors(record.anchors + (anchor,))
            return record.model_copy(
                update={
                    "anchors": anchors,
                    "step_a_status": StepAStatus.UNREVIEWED,
                }
            )

        return self._update_current(updater, persist=persist)

    def remove_anchor(
        self, anchor_index: int, *, persist: bool = True
    ) -> StageAStepAAnnotation:
        def updater(record: StageAStepAAnnotation) -> StageAStepAAnnotation:
            kept = [
                anchor
                for anchor in record.anchors
                if anchor.anchor_index != anchor_index
            ]
            if len(kept) == len(record.anchors):
                raise ValueError(f"unknown anchor_index: {anchor_index}")
            return record.model_copy(
                update={
                    "anchors": reindex_anchors(kept),
                    "step_a_status": StepAStatus.UNREVIEWED,
                }
            )

        return self._update_current(updater, persist=persist)

    def mark_complete(self, *, persist: bool = True) -> StageAStepAAnnotation:
        def updater(record: StageAStepAAnnotation) -> StageAStepAAnnotation:
            updated = record.model_copy(
                update={"step_a_status": StepAStatus.COMPLETE}
            )
            # model validator enforces COMPLETE constraints
            return updated

        updated = self._update_current(updater, persist=persist)
        self._cursor = self._first_unreviewed_index()
        return updated

    def reopen_for_edit(self, *, persist: bool = True) -> StageAStepAAnnotation:
        def updater(record: StageAStepAAnnotation) -> StageAStepAAnnotation:
            return record.model_copy(
                update={"step_a_status": StepAStatus.UNREVIEWED}
            )

        return self._update_current(updater, persist=persist)

    def skip(self) -> StageAStepAAnnotation | None:
        if self._cursor >= len(self._order):
            return None
        self._cursor += 1
        while self._cursor < len(self._order):
            if (
                self._records[self._order[self._cursor]].step_a_status
                is StepAStatus.UNREVIEWED
            ):
                break
            self._cursor += 1
        return self.current()

    def back(self, *, persist: bool = True) -> StageAStepAAnnotation | None:
        if not self._history:
            if self._cursor <= 0:
                return self.current()
            self._cursor = max(0, self._cursor - 1)
            return self.current()
        stage_a_id = self._history.pop()
        # Move cursor to that example and reopen if complete.
        for index, item_id in enumerate(self._order):
            if item_id == stage_a_id:
                self._cursor = index
                break
        record = self._records[stage_a_id]
        if record.step_a_status is StepAStatus.COMPLETE:
            self._records[stage_a_id] = record.model_copy(
                update={"step_a_status": StepAStatus.UNREVIEWED}
            )
            if persist:
                self.save()
        return self.current()


def parse_step_a_command(raw: str) -> tuple[str, str | None]:
    token = raw.strip()
    if not token:
        raise ValueError("empty command")
    key = token.lower()
    mapping = {
        "a": "add_operation",
        "r": "remove_operation",
        "n": "to_anchors",
        "x": "add_anchor",
        "d": "remove_anchor",
        "v": "preview",
        "h": "help",
        "s": "skip",
        "b": "back",
        "p": "progress",
        "q": "quit",
        "c": "complete",
        "e": "reopen",
    }
    if key in mapping:
        return mapping[key], None
    raise ValueError(f"unknown command: {raw!r}")


def demo_step_a_interaction() -> str:
    """In-memory dry-run demo that does not touch Stage-A annotation files."""
    query = "Where is my gate and how do I get there?"
    record = StageAStepAAnnotation(
        stage_a_id="demo_sa_0000",
        source_id="demo_src",
        candidate_id=None,
        query=query,
        final_bucket="MIXED_SEQUENTIAL",
        source_kind="demo",
        semantic_group="demo_group",
        template_group="demo_template",
        provenance={"kind": "demo"},
        derived_query_type=QueryType.MIXED,
    )
    lines = [
        "=== DRY-RUN DEMO (temporary in-memory example; no files written) ===",
        f"QUERY: {query}",
        "words: " + format_query_with_word_indexes(query),
        "",
        "Contract note:",
        '  "my gate" is an H4 anchor;',
        "  RESOLVE_PERSONAL is intentionally absent from Step-A H2/H3 because it will",
        "  be synthesized from H5 during Step B.",
        f"  {H4_ANCHOR_GUIDANCE}",
        "",
        "Human: a  (add operation)",
        "Human substring: Where is my gate",
        "Human operator: LOCATE_ENVIRONMENTAL",
    ]
    op1 = create_operation_from_substring(
        query, "Where is my gate", OperatorType.LOCATE_ENVIRONMENTAL
    )
    record = record.model_copy(update={"operations": reindex_operations((op1,))})
    lines.append(f"Tool stored op0: {op1.char_start}:{op1.char_end} {op1.text!r}")
    lines.append("")
    lines.append("Human: a")
    lines.append("Human substring: how do I get there")
    lines.append("Human operator: NAVIGATE_TO")
    op2 = create_operation_from_substring(
        query,
        "how do I get there",
        OperatorType.NAVIGATE_TO,
        existing_operations=record.operations,
    )
    record = record.model_copy(
        update={"operations": reindex_operations(record.operations + (op2,))}
    )
    lines.append(f"Tool stored op1: {op2.char_start}:{op2.char_end} {op2.text!r}")
    lines.append("")
    lines.append("Human: n  (finish operations / move to anchors)")
    lines.append("Human: x  (add anchor)")
    lines.append("Human substring: my gate")
    lines.append(
        'Note: "my gate" is an H4 anchor; RESOLVE_PERSONAL is intentionally '
        "absent from Step-A H2/H3 because it will be synthesized from H5 during Step B."
    )
    anchor = create_anchor_from_substring(query, "my gate")
    record = record.model_copy(
        update={"anchors": reindex_anchors((anchor,))}
    )
    lines.append(
        f"Tool stored anchor0: {anchor.char_start}:{anchor.char_end} {anchor.text!r}"
    )
    assert all(
        op.operator_type is not OperatorType.RESOLVE_PERSONAL
        for op in record.operations
    )
    assert record.anchors[0].text == "my gate"
    lines.append("")
    lines.append("Human: v  (preview)")
    lines.append(format_annotation_preview(record))
    lines.append("")
    lines.append(
        "Human would press c to COMPLETE on a real example; "
        "demo leaves status UNREVIEWED and writes nothing."
    )
    lines.append(f"demo status remains: {record.step_a_status.value}")
    return "\n".join(lines)


__all__ = [
    "ANSWER_OPERATORS",
    "BUCKET_TO_QUERY_TYPE",
    "DEFAULT_FROZEN_SELECTION_PATH",
    "DEFAULT_STEP_A_ANNOTATIONS_PATH",
    "EXPECTED_STAGE_A_COUNT",
    "H4_ANCHOR_GUIDANCE",
    "OPERATOR_HELP",
    "RESOLVE_PERSONAL_IMPLICIT_WARNING",
    "StageAStepAAnnotation",
    "StepAAnchor",
    "StepAAnnotationSession",
    "StepAOperation",
    "StepAStatus",
    "check_bio_conflicts",
    "check_span_alignment",
    "create_anchor_from_substring",
    "create_operation_from_substring",
    "demo_step_a_interaction",
    "derive_query_type",
    "ensure_step_a_annotations_initialized",
    "find_substring_occurrences",
    "fingerprint_file",
    "format_annotation_preview",
    "format_operator_help",
    "format_query_with_word_indexes",
    "format_step_a_progress",
    "initialize_step_a_annotations_from_selection",
    "load_step_a_annotations",
    "parse_step_a_command",
    "reindex_anchors",
    "reindex_operations",
    "spans_overlap",
    "summarize_step_a_progress",
    "validate_step_a_corpus",
    "write_step_a_annotations",
]
