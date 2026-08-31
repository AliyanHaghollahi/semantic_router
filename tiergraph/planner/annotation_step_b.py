"""Step-B Stage-A planner annotation: H5/H6/H7 over frozen Step-A gold.

Step A remains the sole source of H2/H3/H4 spans. Step B stores only:
- H5 implicit resolution per Step-A anchor_index
- H6 owner_operation_index per anchor
- H7 directed explicit operation dependencies

Uses existing ``ImplicitResolution`` and ``is_h7_pair_eligible``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from tiergraph.enums import OperatorType, _CanonicalWireEnum
from tiergraph.models import TierGraphSchema
from tiergraph.planner.annotation_step_a import (
    DEFAULT_STEP_A_ANNOTATIONS_PATH,
    EXPECTED_STAGE_A_COUNT,
    StageAStepAAnnotation,
    StepAStatus,
    fingerprint_file,
    load_step_a_annotations,
)
from tiergraph.planner.annotations import ImplicitResolution
from tiergraph.planner.operator_io import is_h7_pair_eligible


DEFAULT_STEP_B_ANNOTATIONS_PATH = Path(
    "dataset/planner/stage_a_step_b_annotations.jsonl"
)

H5_HELP = (
    "H5 implicit resolution (existing ImplicitResolution enum):\n"
    "  NONE — anchor does not synthesize RESOLVE_PERSONAL\n"
    "  IMPLICIT_RESOLVE_PERSONAL — decoder synthesizes RESOLVE_PERSONAL for this "
    "personal reference (e.g. 'my gate', 'my appointment')\n"
    "Do not mark every possessive automatically; use semantics.\n"
    "Do not create H2 RESOLVE_PERSONAL spans; that belongs to Step A / decoder."
)

H6_HELP = (
    "H6 ownership: assign each H4 anchor to the explicit H2 operation that "
    "semantically uses it (operation_index from Step A).\n"
    "One owner per anchor. Ownership is not an H7 edge."
)

H7_HELP = (
    "H7: directed explicit operation -> operation dependencies only.\n"
    "Eligible under OPERATOR_IO_CONTRACT_V1 (is_h7_pair_eligible).\n"
    "Do NOT encode FUSE, fusion-only compare, or implicit RESOLVE_PERSONAL edges.\n"
    "With 0 or 1 explicit operations, H7 must be empty."
)


class StepBStatus(_CanonicalWireEnum, str, Enum):
    UNREVIEWED = "UNREVIEWED"
    COMPLETE = "COMPLETE"


class _StepBSchema(TierGraphSchema):
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


class StepBAnchorDecision(_StepBSchema):
    """H5/H6 decision for one frozen Step-A anchor_index."""

    anchor_index: int = Field(ge=0)
    text: str | None = None  # audit copy from Step A
    implicit_resolution: ImplicitResolution = ImplicitResolution.NONE
    owner_operation_index: int | None = None


class StepBDependency(_StepBSchema):
    """One explicit H7 edge between Step-A operation indices."""

    source_operation_index: int = Field(ge=0)
    target_operation_index: int = Field(ge=0)

    @model_validator(mode="after")
    def _no_self_loop(self) -> "StepBDependency":
        if self.source_operation_index == self.target_operation_index:
            raise ValueError("H7 self-loop is invalid")
        return self


class StageAStepBAnnotation(_StepBSchema):
    """Step-B gold linked to one frozen Step-A example."""

    stage_a_id: str
    source_id: str | None = None
    candidate_id: str | None = None
    query: str
    final_bucket: str
    n_operations: int = Field(ge=0)
    n_anchors: int = Field(ge=0)
    # Audit snapshots of Step-A operator types in index order (not H2 spans).
    operation_types: tuple[str, ...] = ()
    anchor_decisions: tuple[StepBAnchorDecision, ...] = ()
    dependencies: tuple[StepBDependency, ...] = ()
    step_b_status: StepBStatus = StepBStatus.UNREVIEWED

    @field_validator("stage_a_id", "query", "final_bucket")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if type(value) is not str or not value.strip():
            raise ValueError("must be a nonblank string")
        return value

    @field_validator(
        "operation_types", "anchor_decisions", "dependencies", mode="before"
    )
    @classmethod
    def _tupleize(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_record(self) -> "StageAStepBAnnotation":
        if self.source_id is None and self.candidate_id is None:
            raise ValueError("source_id or candidate_id is required")
        if len(self.operation_types) != self.n_operations:
            raise ValueError("operation_types length must equal n_operations")
        if len(self.anchor_decisions) != self.n_anchors:
            raise ValueError("anchor_decisions length must equal n_anchors")
        for index, decision in enumerate(self.anchor_decisions):
            if decision.anchor_index != index:
                raise ValueError(
                    f"{self.stage_a_id}: anchor_index must be contiguous "
                    f"(expected {index}, got {decision.anchor_index})"
                )
        if self.step_b_status is StepBStatus.COMPLETE:
            validate_step_b_record_complete(self)
        else:
            validate_step_b_record_structure(self, require_complete=False)
        return self


def _operator_type_at(
    record: StageAStepBAnnotation, operation_index: int
) -> OperatorType:
    try:
        return OperatorType(record.operation_types[operation_index])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"{record.stage_a_id}: invalid operation_types[{operation_index}]"
        ) from exc


def h7_forms_cycle(
    n_operations: int, edges: Sequence[tuple[int, int]]
) -> bool:
    """Return True if directed edges contain a cycle among operation indices."""
    adjacency: dict[int, list[int]] = defaultdict(list)
    indegree = [0] * n_operations
    for source, target in edges:
        adjacency[source].append(target)
        indegree[target] += 1
    queue = deque(index for index in range(n_operations) if indegree[index] == 0)
    seen = 0
    while queue:
        node = queue.popleft()
        seen += 1
        for nxt in adjacency[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return seen != n_operations


def validate_step_b_record_structure(
    record: StageAStepBAnnotation,
    *,
    require_complete: bool,
) -> None:
    """Validate H5/H6/H7 structure; raise ValueError on failure."""
    n_ops = record.n_operations
    n_anc = record.n_anchors

    for decision in record.anchor_decisions:
        if decision.implicit_resolution not in ImplicitResolution:
            raise ValueError(
                f"{record.stage_a_id}: invalid H5 {decision.implicit_resolution!r}"
            )
        if decision.owner_operation_index is not None:
            if not (0 <= decision.owner_operation_index < n_ops):
                raise ValueError(
                    f"{record.stage_a_id}: H6 owner_operation_index "
                    f"{decision.owner_operation_index} out of range"
                )
        elif require_complete and n_anc > 0:
            raise ValueError(
                f"{record.stage_a_id}: COMPLETE requires H6 owner for "
                f"anchor[{decision.anchor_index}]"
            )

    if n_ops <= 1 and record.dependencies:
        raise ValueError(
            f"{record.stage_a_id}: H7 must be empty when n_operations <= 1"
        )

    seen: set[tuple[int, int]] = set()
    edge_list: list[tuple[int, int]] = []
    for dependency in record.dependencies:
        pair = (
            dependency.source_operation_index,
            dependency.target_operation_index,
        )
        if pair in seen:
            raise ValueError(f"{record.stage_a_id}: duplicate H7 edge {pair}")
        seen.add(pair)
        if not (0 <= dependency.source_operation_index < n_ops):
            raise ValueError(
                f"{record.stage_a_id}: H7 source out of range {pair}"
            )
        if not (0 <= dependency.target_operation_index < n_ops):
            raise ValueError(
                f"{record.stage_a_id}: H7 target out of range {pair}"
            )
        source_op = _operator_type_at(record, dependency.source_operation_index)
        target_op = _operator_type_at(record, dependency.target_operation_index)
        if source_op is OperatorType.RESOLVE_PERSONAL:
            raise ValueError(
                f"{record.stage_a_id}: H7 must not use RESOLVE_PERSONAL as "
                "an explicit source (implicit edges are decoder-synthesized)"
            )
        if target_op is OperatorType.RESOLVE_PERSONAL:
            raise ValueError(
                f"{record.stage_a_id}: H7 must not target RESOLVE_PERSONAL"
            )
        if not is_h7_pair_eligible(source_op, target_op):
            raise ValueError(
                f"{record.stage_a_id}: illegal typed H7 edge "
                f"{source_op.value} -> {target_op.value}"
            )
        edge_list.append(pair)

    if n_ops > 0 and h7_forms_cycle(n_ops, edge_list):
        raise ValueError(f"{record.stage_a_id}: H7 graph contains a cycle")


def validate_step_b_record_complete(record: StageAStepBAnnotation) -> None:
    if record.n_operations < 1:
        raise ValueError(
            f"{record.stage_a_id}: COMPLETE requires n_operations >= 1 "
            "(Step A must be COMPLETE)"
        )
    validate_step_b_record_structure(record, require_complete=True)


def initialize_step_b_from_step_a(
    step_a: StageAStepAAnnotation,
) -> StageAStepBAnnotation:
    """Create an UNREVIEWED Step-B shell for one COMPLETE Step-A example."""
    if step_a.step_a_status is not StepAStatus.COMPLETE:
        raise ValueError(
            f"{step_a.stage_a_id}: Step A must be COMPLETE before Step B init"
        )
    decisions = tuple(
        StepBAnchorDecision(
            anchor_index=anchor.anchor_index,
            text=anchor.text,
            implicit_resolution=ImplicitResolution.NONE,
            owner_operation_index=None,
        )
        for anchor in step_a.anchors
    )
    return StageAStepBAnnotation(
        stage_a_id=step_a.stage_a_id,
        source_id=step_a.source_id,
        candidate_id=step_a.candidate_id,
        query=step_a.query,
        final_bucket=step_a.final_bucket,
        n_operations=len(step_a.operations),
        n_anchors=len(step_a.anchors),
        operation_types=tuple(op.operator_type.value for op in step_a.operations),
        anchor_decisions=decisions,
        dependencies=(),
        step_b_status=StepBStatus.UNREVIEWED,
    )


def initialize_step_b_annotations_from_step_a(
    step_a_records: Sequence[StageAStepAAnnotation],
) -> list[StageAStepBAnnotation]:
    if len(step_a_records) != EXPECTED_STAGE_A_COUNT:
        raise ValueError(
            f"expected {EXPECTED_STAGE_A_COUNT} Step-A records, "
            f"got {len(step_a_records)}"
        )
    incomplete = [
        item.stage_a_id
        for item in step_a_records
        if item.step_a_status is not StepAStatus.COMPLETE
    ]
    if incomplete:
        preview = incomplete[:5]
        suffix = "..." if len(incomplete) > 5 else ""
        raise ValueError(
            "Step A is not fully COMPLETE; refusing Step B init. "
            f"incomplete={preview}{suffix}"
        )
    return [
        initialize_step_b_from_step_a(item)
        for item in sorted(step_a_records, key=lambda row: row.stage_a_id)
    ]


def load_step_b_annotations(path: str | Path) -> list[StageAStepBAnnotation]:
    path = Path(path)
    rows: list[StageAStepBAnnotation] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            rows.append(StageAStepBAnnotation.model_validate(payload))
    return rows


def write_step_b_annotations(
    path: str | Path,
    records: Sequence[StageAStepBAnnotation],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda item: item.stage_a_id)
    lines = [
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for item in ordered
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def ensure_step_b_annotations_initialized(
    *,
    step_a_path: str | Path = DEFAULT_STEP_A_ANNOTATIONS_PATH,
    step_b_path: str | Path = DEFAULT_STEP_B_ANNOTATIONS_PATH,
) -> list[StageAStepBAnnotation]:
    """Load Step B, or initialize from frozen COMPLETE Step A if missing."""
    step_a_path = Path(step_a_path)
    step_b_path = Path(step_b_path)
    step_a_before = fingerprint_file(step_a_path)
    step_a_records = load_step_a_annotations(step_a_path)
    if fingerprint_file(step_a_path) != step_a_before:
        raise ValueError("Step-A annotation file mutated while loading")

    if step_b_path.is_file():
        records = load_step_b_annotations(step_b_path)
        errors = validate_step_b_corpus(
            records,
            step_a_records=step_a_records,
            step_a_path=step_a_path,
            require_all_complete=False,
        )
        if errors:
            raise ValueError(
                "existing Step-B annotations failed validation:\n"
                + "\n".join(errors[:20])
            )
        if fingerprint_file(step_a_path) != step_a_before:
            raise ValueError("Step-A annotation file mutated during Step-B load")
        return records

    records = initialize_step_b_annotations_from_step_a(step_a_records)
    write_step_b_annotations(step_b_path, records)
    if fingerprint_file(step_a_path) != step_a_before:
        raise ValueError("Step-A annotation file mutated during Step-B init")
    return records


def summarize_step_b_progress(
    records: Sequence[StageAStepBAnnotation],
) -> dict[str, Any]:
    status_counts = Counter(item.step_b_status.value for item in records)
    bucket_status: dict[str, Counter[str]] = defaultdict(Counter)
    h5_counts: Counter[str] = Counter()
    for item in records:
        bucket_status[item.final_bucket][item.step_b_status.value] += 1
        for decision in item.anchor_decisions:
            h5_counts[decision.implicit_resolution.value] += 1
    return {
        "total": len(records),
        "COMPLETE": status_counts.get(StepBStatus.COMPLETE.value, 0),
        "UNREVIEWED": status_counts.get(StepBStatus.UNREVIEWED.value, 0),
        "by_final_bucket": {
            bucket: {
                "COMPLETE": counts.get(StepBStatus.COMPLETE.value, 0),
                "UNREVIEWED": counts.get(StepBStatus.UNREVIEWED.value, 0),
                "total": sum(counts.values()),
            }
            for bucket, counts in sorted(bucket_status.items())
        },
        "total_h7_edges": sum(len(item.dependencies) for item in records),
        "h5_counts": dict(sorted(h5_counts.items())),
        "anchors_with_owner": sum(
            1
            for item in records
            for decision in item.anchor_decisions
            if decision.owner_operation_index is not None
        ),
    }


def format_step_b_progress(summary: Mapping[str, Any]) -> str:
    lines = [
        f"Total: {summary['total']}",
        f"COMPLETE: {summary['COMPLETE']}",
        f"UNREVIEWED: {summary['UNREVIEWED']}",
        "By final_bucket:",
    ]
    for bucket, counts in summary["by_final_bucket"].items():
        lines.append(
            f"  {bucket}: COMPLETE={counts['COMPLETE']} "
            f"UNREVIEWED={counts['UNREVIEWED']} total={counts['total']}"
        )
    lines.append(f"total_h7_edges: {summary['total_h7_edges']}")
    lines.append(f"anchors_with_owner: {summary['anchors_with_owner']}")
    if summary.get("h5_counts"):
        lines.append("h5_counts:")
        for key, count in summary["h5_counts"].items():
            lines.append(f"  {key}: {count}")
    return "\n".join(lines)


def format_step_a_context(step_a: StageAStepAAnnotation) -> str:
    lines = [
        f"stage_a_id: {step_a.stage_a_id}",
        f"final_bucket: {step_a.final_bucket}",
        f"query: {step_a.query}",
        "operations (Step A / H2+H3):",
    ]
    for op in step_a.operations:
        lines.append(
            f"  [{op.operation_index}] {op.operator_type.value} "
            f"{op.char_start}:{op.char_end} {op.text!r}"
        )
    lines.append("anchors (Step A / H4):")
    if not step_a.anchors:
        lines.append("  (none)")
    for anchor in step_a.anchors:
        lines.append(
            f"  [{anchor.anchor_index}] {anchor.char_start}:{anchor.char_end} "
            f"{anchor.text!r}"
        )
    return "\n".join(lines)


def format_step_b_preview(
    record: StageAStepBAnnotation,
    *,
    step_a: StageAStepAAnnotation | None = None,
) -> str:
    lines: list[str] = []
    if step_a is not None:
        lines.append(format_step_a_context(step_a))
        lines.append("")
    lines.extend(
        [
            f"step_b_status: {record.step_b_status.value}",
            "H5/H6 anchor decisions:",
        ]
    )
    if not record.anchor_decisions:
        lines.append("  (no anchors)")
    for decision in record.anchor_decisions:
        owner = (
            "UNSET"
            if decision.owner_operation_index is None
            else str(decision.owner_operation_index)
        )
        lines.append(
            f"  [{decision.anchor_index}] text={decision.text!r} "
            f"H5={decision.implicit_resolution.value} H6_owner={owner}"
        )
    lines.append("H7 dependencies:")
    if not record.dependencies:
        lines.append("  (none)")
    for dependency in record.dependencies:
        src = dependency.source_operation_index
        tgt = dependency.target_operation_index
        src_op = (
            record.operation_types[src]
            if src < len(record.operation_types)
            else "?"
        )
        tgt_op = (
            record.operation_types[tgt]
            if tgt < len(record.operation_types)
            else "?"
        )
        lines.append(f"  {src}({src_op}) -> {tgt}({tgt_op})")
    return "\n".join(lines)


def format_step_b_help() -> str:
    return "\n\n".join([H5_HELP, H6_HELP, H7_HELP])


def set_anchor_h5(
    record: StageAStepBAnnotation,
    anchor_index: int,
    implicit_resolution: ImplicitResolution | str,
) -> StageAStepBAnnotation:
    if isinstance(implicit_resolution, str):
        try:
            implicit_resolution = ImplicitResolution(implicit_resolution)
        except ValueError as exc:
            raise ValueError(f"invalid H5 value: {implicit_resolution!r}") from exc
    if not (0 <= anchor_index < record.n_anchors):
        raise ValueError(f"anchor_index out of range: {anchor_index}")
    decisions = list(record.anchor_decisions)
    decisions[anchor_index] = decisions[anchor_index].model_copy(
        update={"implicit_resolution": implicit_resolution}
    )
    return record.model_copy(
        update={
            "anchor_decisions": tuple(decisions),
            "step_b_status": StepBStatus.UNREVIEWED,
        }
    )


def set_anchor_h6(
    record: StageAStepBAnnotation,
    anchor_index: int,
    owner_operation_index: int,
) -> StageAStepBAnnotation:
    if not (0 <= anchor_index < record.n_anchors):
        raise ValueError(f"anchor_index out of range: {anchor_index}")
    if not (0 <= owner_operation_index < record.n_operations):
        raise ValueError(
            f"owner_operation_index out of range: {owner_operation_index}"
        )
    decisions = list(record.anchor_decisions)
    decisions[anchor_index] = decisions[anchor_index].model_copy(
        update={"owner_operation_index": owner_operation_index}
    )
    return record.model_copy(
        update={
            "anchor_decisions": tuple(decisions),
            "step_b_status": StepBStatus.UNREVIEWED,
        }
    )


def add_h7_dependency(
    record: StageAStepBAnnotation,
    source_operation_index: int,
    target_operation_index: int,
) -> StageAStepBAnnotation:
    if record.n_operations <= 1:
        raise ValueError("H7 must be empty when n_operations <= 1")
    trial = record.model_copy(
        update={
            "dependencies": tuple(
                list(record.dependencies)
                + [
                    StepBDependency(
                        source_operation_index=source_operation_index,
                        target_operation_index=target_operation_index,
                    )
                ]
            ),
            "step_b_status": StepBStatus.UNREVIEWED,
        }
    )
    validate_step_b_record_structure(trial, require_complete=False)
    return trial


def remove_h7_dependency(
    record: StageAStepBAnnotation,
    source_operation_index: int,
    target_operation_index: int,
) -> StageAStepBAnnotation:
    kept = [
        dependency
        for dependency in record.dependencies
        if not (
            dependency.source_operation_index == source_operation_index
            and dependency.target_operation_index == target_operation_index
        )
    ]
    if len(kept) == len(record.dependencies):
        raise ValueError(
            f"H7 edge not found: {source_operation_index} -> {target_operation_index}"
        )
    return record.model_copy(
        update={
            "dependencies": tuple(kept),
            "step_b_status": StepBStatus.UNREVIEWED,
        }
    )


def mark_step_b_complete(record: StageAStepBAnnotation) -> StageAStepBAnnotation:
    trial = record.model_copy(update={"step_b_status": StepBStatus.COMPLETE})
    validate_step_b_record_complete(trial)
    return trial


def validate_step_b_against_step_a(
    step_b: StageAStepBAnnotation,
    step_a: StageAStepAAnnotation,
) -> list[str]:
    errors: list[str] = []
    if step_a.step_a_status is not StepAStatus.COMPLETE:
        errors.append(f"{step_b.stage_a_id}: linked Step A is not COMPLETE")
    if step_b.query != step_a.query:
        errors.append(f"{step_b.stage_a_id}: query differs from Step A")
    if step_b.final_bucket != step_a.final_bucket:
        errors.append(f"{step_b.stage_a_id}: final_bucket differs from Step A")
    if step_b.source_id != step_a.source_id:
        errors.append(f"{step_b.stage_a_id}: source_id differs from Step A")
    if step_b.candidate_id != step_a.candidate_id:
        errors.append(f"{step_b.stage_a_id}: candidate_id differs from Step A")
    if step_b.n_operations != len(step_a.operations):
        errors.append(f"{step_b.stage_a_id}: n_operations differs from Step A")
    if step_b.n_anchors != len(step_a.anchors):
        errors.append(f"{step_b.stage_a_id}: n_anchors differs from Step A")
    expected_types = tuple(op.operator_type.value for op in step_a.operations)
    if step_b.operation_types != expected_types:
        errors.append(
            f"{step_b.stage_a_id}: operation_types drifted from Step A H3"
        )
    if len(step_b.anchor_decisions) != len(step_a.anchors):
        errors.append(
            f"{step_b.stage_a_id}: anchor_decisions count differs from Step A"
        )
    else:
        for decision, anchor in zip(
            step_b.anchor_decisions, step_a.anchors, strict=True
        ):
            if decision.anchor_index != anchor.anchor_index:
                errors.append(
                    f"{step_b.stage_a_id}: anchor_index mismatch vs Step A"
                )
            if decision.text is not None and decision.text != anchor.text:
                errors.append(
                    f"{step_b.stage_a_id}: anchor[{decision.anchor_index}] text "
                    "differs from Step A"
                )
    return errors


def validate_step_b_corpus(
    records: Sequence[StageAStepBAnnotation],
    *,
    step_a_records: Sequence[StageAStepAAnnotation],
    step_a_path: str | Path = DEFAULT_STEP_A_ANNOTATIONS_PATH,
    require_all_complete: bool = False,
    step_a_fingerprint: tuple[int, str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if len(records) != EXPECTED_STAGE_A_COUNT:
        errors.append(
            f"Step-B count {len(records)} != {EXPECTED_STAGE_A_COUNT}"
        )
    if len(step_a_records) != EXPECTED_STAGE_A_COUNT:
        errors.append(
            f"Step-A count {len(step_a_records)} != {EXPECTED_STAGE_A_COUNT}"
        )

    step_a_path = Path(step_a_path)
    if step_a_fingerprint is not None:
        current = fingerprint_file(step_a_path)
        if current != step_a_fingerprint:
            errors.append("frozen Step-A annotation file was mutated")

    by_b = {item.stage_a_id: item for item in records}
    by_a = {item.stage_a_id: item for item in step_a_records}
    if len(by_b) != len(records):
        errors.append("duplicate stage_a_id in Step B")
    if sorted(by_b) != sorted(by_a):
        errors.append("stage_a_id set does not match Step A")

    query_keys = [item.query.strip().casefold() for item in records]
    if len(query_keys) != len(set(query_keys)):
        errors.append("duplicate normalized queries in Step B")

    for stage_a_id, step_a in by_a.items():
        step_b = by_b.get(stage_a_id)
        if step_b is None:
            errors.append(f"missing Step-B record for {stage_a_id}")
            continue
        errors.extend(validate_step_b_against_step_a(step_b, step_a))
        try:
            if step_b.step_b_status is StepBStatus.COMPLETE:
                validate_step_b_record_complete(step_b)
            else:
                validate_step_b_record_structure(step_b, require_complete=False)
        except ValueError as exc:
            errors.append(str(exc))
        if (
            require_all_complete
            and step_b.step_b_status is not StepBStatus.COMPLETE
        ):
            errors.append(f"{stage_a_id}: not COMPLETE")
    return errors


class StepBAnnotationSession:
    """Resumable Step-B annotation; never writes Step A."""

    def __init__(
        self,
        records: Sequence[StageAStepBAnnotation],
        *,
        step_a_records: Sequence[StageAStepAAnnotation],
        step_b_path: str | Path,
        step_a_path: str | Path = DEFAULT_STEP_A_ANNOTATIONS_PATH,
    ) -> None:
        if len(records) != EXPECTED_STAGE_A_COUNT:
            raise ValueError(
                f"expected {EXPECTED_STAGE_A_COUNT} Step-B rows, got {len(records)}"
            )
        step_a_by_id = {item.stage_a_id: item for item in step_a_records}
        if sorted(step_a_by_id) != sorted(item.stage_a_id for item in records):
            raise ValueError("Step-B IDs must match Step A")
        for item in records:
            errors = validate_step_b_against_step_a(
                item, step_a_by_id[item.stage_a_id]
            )
            if errors:
                raise ValueError("; ".join(errors))

        ordered = tuple(sorted(records, key=lambda item: item.stage_a_id))
        self._records = {item.stage_a_id: item for item in ordered}
        self._order = [item.stage_a_id for item in ordered]
        self._step_a = step_a_by_id
        self.step_b_path = Path(step_b_path)
        self.step_a_path = Path(step_a_path)
        self._step_a_fingerprint = fingerprint_file(self.step_a_path)
        self._history: list[str] = []
        self._cursor = self._first_unreviewed_index()

    @property
    def records(self) -> tuple[StageAStepBAnnotation, ...]:
        return tuple(self._records[stage_a_id] for stage_a_id in self._order)

    def _first_unreviewed_index(self) -> int:
        for index, stage_a_id in enumerate(self._order):
            if self._records[stage_a_id].step_b_status is StepBStatus.UNREVIEWED:
                return index
        return len(self._order)

    def current(self) -> StageAStepBAnnotation | None:
        if self._cursor >= len(self._order):
            return None
        return self._records[self._order[self._cursor]]

    def current_step_a(self) -> StageAStepAAnnotation | None:
        current = self.current()
        if current is None:
            return None
        return self._step_a[current.stage_a_id]

    def current_position(self) -> tuple[int, int]:
        if self._cursor >= len(self._order):
            return len(self._order), len(self._order)
        return self._cursor + 1, len(self._order)

    def summary(self) -> dict[str, Any]:
        return summarize_step_b_progress(self.records)

    def save(self) -> None:
        if fingerprint_file(self.step_a_path) != self._step_a_fingerprint:
            raise ValueError(
                "frozen Step-A annotations were modified; refusing to save Step B"
            )
        write_step_b_annotations(self.step_b_path, self.records)

    def _update_current(
        self,
        updater: Callable[[StageAStepBAnnotation], StageAStepBAnnotation],
        *,
        persist: bool = True,
    ) -> StageAStepBAnnotation:
        current = self.current()
        if current is None:
            raise ValueError("no current example")
        updated = updater(current)
        self._records[updated.stage_a_id] = updated
        if not self._history or self._history[-1] != updated.stage_a_id:
            self._history.append(updated.stage_a_id)
        if persist:
            self.save()
        return updated

    def set_h5(
        self,
        anchor_index: int,
        implicit_resolution: ImplicitResolution | str,
        *,
        persist: bool = True,
    ) -> StageAStepBAnnotation:
        return self._update_current(
            lambda record: set_anchor_h5(
                record, anchor_index, implicit_resolution
            ),
            persist=persist,
        )

    def set_h6(
        self,
        anchor_index: int,
        owner_operation_index: int,
        *,
        persist: bool = True,
    ) -> StageAStepBAnnotation:
        return self._update_current(
            lambda record: set_anchor_h6(
                record, anchor_index, owner_operation_index
            ),
            persist=persist,
        )

    def add_dependency(
        self,
        source_operation_index: int,
        target_operation_index: int,
        *,
        persist: bool = True,
    ) -> StageAStepBAnnotation:
        return self._update_current(
            lambda record: add_h7_dependency(
                record, source_operation_index, target_operation_index
            ),
            persist=persist,
        )

    def remove_dependency(
        self,
        source_operation_index: int,
        target_operation_index: int,
        *,
        persist: bool = True,
    ) -> StageAStepBAnnotation:
        return self._update_current(
            lambda record: remove_h7_dependency(
                record, source_operation_index, target_operation_index
            ),
            persist=persist,
        )

    def mark_complete(self, *, persist: bool = True) -> StageAStepBAnnotation:
        return self._update_current(mark_step_b_complete, persist=persist)

    def skip(self, *, persist: bool = True) -> StageAStepBAnnotation | None:
        current = self.current()
        if current is None:
            return None
        if persist:
            self.save()
        self._cursor += 1
        while self._cursor < len(self._order):
            if (
                self._records[self._order[self._cursor]].step_b_status
                is StepBStatus.UNREVIEWED
            ):
                break
            self._cursor += 1
        return self.current()

    def back(self, *, persist: bool = True) -> StageAStepBAnnotation | None:
        if not self._history:
            if self._cursor > 0:
                self._cursor -= 1
            current = self.current()
            if current is not None and current.step_b_status is StepBStatus.COMPLETE:
                reopened = current.model_copy(
                    update={"step_b_status": StepBStatus.UNREVIEWED}
                )
                self._records[reopened.stage_a_id] = reopened
                if persist:
                    self.save()
                return reopened
            return current
        stage_a_id = self._history.pop()
        self._cursor = self._order.index(stage_a_id)
        current = self._records[stage_a_id]
        if current.step_b_status is StepBStatus.COMPLETE:
            reopened = current.model_copy(
                update={"step_b_status": StepBStatus.UNREVIEWED}
            )
            self._records[stage_a_id] = reopened
            if persist:
                self.save()
            return reopened
        return current

    def reopen(
        self, stage_a_id: str, *, persist: bool = True
    ) -> StageAStepBAnnotation:
        if stage_a_id not in self._records:
            raise ValueError(f"unknown stage_a_id: {stage_a_id}")
        record = self._records[stage_a_id]
        updated = record.model_copy(
            update={"step_b_status": StepBStatus.UNREVIEWED}
        )
        self._records[stage_a_id] = updated
        self._cursor = self._order.index(stage_a_id)
        if persist:
            self.save()
        return updated


def parse_step_b_command(raw: str) -> tuple[str, str | None]:
    key = raw.strip().casefold()
    mapping = {
        "i": "set_h5",
        "o": "set_h6",
        "d": "add_dependency",
        "x": "remove_dependency",
        "v": "preview",
        "h": "help",
        "s": "skip",
        "b": "back",
        "e": "reopen",
        "p": "progress",
        "c": "complete",
        "q": "quit",
    }
    if key in mapping:
        return mapping[key], None
    raise ValueError(f"unknown command: {raw!r}")


def demo_step_b_interaction() -> str:
    """In-memory dry-run; does not write Step A/B files."""
    from tiergraph.enums import QueryType
    from tiergraph.planner.annotation_step_a import (
        StageAStepAAnnotation,
        StepAAnchor,
        StepAOperation,
        StepAStatus,
    )

    query = "Where is my gate and how do I get there?"
    step_a = StageAStepAAnnotation(
        stage_a_id="demo_sa_0000",
        source_id="demo_src",
        candidate_id=None,
        query=query,
        final_bucket="MIXED_SEQUENTIAL",
        source_kind="demo",
        semantic_group="demo",
        template_group="demo",
        provenance={"kind": "demo"},
        derived_query_type=QueryType.MIXED,
        operations=(
            StepAOperation(
                operation_index=0,
                text="Where is my gate",
                char_start=0,
                char_end=16,
                operator_type=OperatorType.LOCATE_ENVIRONMENTAL,
            ),
            StepAOperation(
                operation_index=1,
                text="how do I get there",
                char_start=21,
                char_end=39,
                operator_type=OperatorType.NAVIGATE_TO,
            ),
        ),
        anchors=(
            StepAAnchor(
                anchor_index=0, text="my gate", char_start=9, char_end=16
            ),
            StepAAnchor(
                anchor_index=1, text="there", char_start=34, char_end=39
            ),
        ),
        step_a_status=StepAStatus.COMPLETE,
    )
    record = initialize_step_b_from_step_a(step_a)
    lines = [
        "=== DRY-RUN STEP-B DEMO (in-memory; no files written) ===",
        format_step_a_context(step_a),
        "",
        "Human: i  (set H5 for anchor 0)",
        "Human: IMPLICIT_RESOLVE_PERSONAL",
    ]
    record = set_anchor_h5(
        record, 0, ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
    )
    lines.append("Human: i  (set H5 for anchor 1)")
    lines.append("Human: NONE")
    record = set_anchor_h5(record, 1, ImplicitResolution.NONE)
    lines.append("Human: o  (set H6 owner for anchor 0 -> op 0)")
    record = set_anchor_h6(record, 0, 0)
    lines.append("Human: o  (set H6 owner for anchor 1 -> op 1)")
    record = set_anchor_h6(record, 1, 1)
    lines.append("Human: d  (add H7 0 -> 1)")
    record = add_h7_dependency(record, 0, 1)
    lines.append("")
    lines.append("Human: v")
    lines.append(format_step_b_preview(record, step_a=step_a))
    lines.append("")
    lines.append(
        "Demo leaves status UNREVIEWED; real workflow would press c to COMPLETE."
    )
    lines.append(f"demo status remains: {record.step_b_status.value}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_STEP_A_ANNOTATIONS_PATH",
    "DEFAULT_STEP_B_ANNOTATIONS_PATH",
    "EXPECTED_STAGE_A_COUNT",
    "H5_HELP",
    "H6_HELP",
    "H7_HELP",
    "StageAStepBAnnotation",
    "StepBAnchorDecision",
    "StepBAnnotationSession",
    "StepBDependency",
    "StepBStatus",
    "add_h7_dependency",
    "demo_step_b_interaction",
    "ensure_step_b_annotations_initialized",
    "format_step_a_context",
    "format_step_b_help",
    "format_step_b_preview",
    "format_step_b_progress",
    "h7_forms_cycle",
    "initialize_step_b_annotations_from_step_a",
    "initialize_step_b_from_step_a",
    "load_step_b_annotations",
    "mark_step_b_complete",
    "parse_step_b_command",
    "remove_h7_dependency",
    "set_anchor_h5",
    "set_anchor_h6",
    "summarize_step_b_progress",
    "validate_step_b_against_step_a",
    "validate_step_b_corpus",
    "validate_step_b_record_complete",
    "validate_step_b_record_structure",
    "write_step_b_annotations",
]
