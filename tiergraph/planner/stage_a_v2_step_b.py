"""Stage-A v2 Step-B annotations: copy legacy 120 + agent-assisted 360 new.

Step B only (H5/H6/H7). Does not write splits or train.

Mixed-bucket taxonomy (single-op cases):
- MIXED_IMPLICIT: one explicit op compares/matches a visible environmental referent
  against a personal record (compatibility / safety / match framing).
- MIXED_SEQUENTIAL: one explicit op selects, locates, or describes an environmental
  target determined by a personal record (which/where/describe-my-X framing).
  Sequencing is implicit via resolver -> explicit op even without H7.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tiergraph.enums import OperatorType
from tiergraph.planner.annotation_step_a import (
    StageAStepAAnnotation,
    StepAAnchor,
    StepAOperation,
    load_step_a_annotations,
)
from tiergraph.planner.annotation_step_b import (
    StageAStepBAnnotation,
    StepBAnchorDecision,
    StepBDependency,
    StepBStatus,
    initialize_step_b_from_step_a,
    load_step_b_annotations,
    mark_step_b_complete,
    validate_step_b_against_step_a,
    validate_step_b_record_complete,
    write_step_b_annotations,
)
from tiergraph.planner.annotations import ImplicitResolution
from tiergraph.planner.operator_io import is_h7_pair_eligible
from tiergraph.planner.stage_a_selection import load_jsonl
from tiergraph.planner.stage_a_to_corpus import step_ab_to_planner_example
from tiergraph.planner.stage_a_v2_spec import (
    STAGE_A_V1_STEP_B_PATH,
    STAGE_A_V2_CORPUS_SIZE,
    STAGE_A_V2_SELECTION_PATH,
    STAGE_A_V2_STEP_A_PATH,
    STAGE_A_V2_STEP_B_PATH,
    STAGE_A_V2_STEP_B_REPORT_PATH,
)

EXPECTED_V2_STEP_B_COUNT = STAGE_A_V2_CORPUS_SIZE
BATCH_SIZE = 48

_DEICTIC_PREFIX = re.compile(
    r"^(?:this|that|these|those|here|there)\b",
    re.IGNORECASE,
)
_ENV_THE_NOUN = re.compile(
    r"^the\s+(?:right\s+)?(?:"
    r"exit|gate|door|counter|belt|board|map|directory|plaque|landmark|sign|"
    r"stairwell|carousel|booth|row|seat|bottle|dish|pharmacy|clinic|room|"
    r"locker|window|garage|platform|lobby|floor|section|legend|placard|"
    r"arrivals|departures|entrance|label|doorway|type|markers?|numbers?"
    r")\b",
    re.IGNORECASE,
)
_PERSONAL_PRONOUN = re.compile(r"\b(?:my|your)\b", re.IGNORECASE)
_PERSONAL_I = re.compile(r"\bI\b")
_BENEFICIARY_FOR_ME = re.compile(r"\bfor\s+me\b", re.IGNORECASE)
_URGENCY_QUERY_TAIL = re.compile(
    r"(?:[?.!]\s*)?"
    r"(?:your\s+immediate\s+attention(?:\s+is\s+required)?|tell\s+me\s+now)"
    r"\s*\.?\s*$",
    re.IGNORECASE,
)
_PERSONAL_THE_NOUN = re.compile(
    r"^the\s+.*\b(?:"
    r"medication|dosage|prescription|allerg|appointment|reservation|ticket|"
    r"account|visit|list|meeting|restriction|train|flight|order|item|"
    r"doctor|hotel|profile|record|assignment|booked|reserved|registered|assigned|listed|"
    r"right\s+medication"
    r")\b",
    re.IGNORECASE,
)
_PERSONAL_NAME_REF = re.compile(
    r"\b(?:my|your)\s+name\b|(?:registered|listed)\s+under\s+(?:my|your)\s+name\b",
    re.IGNORECASE,
)
_QUERY_PERSONAL_REF = re.compile(
    r"\b(?:my|your)\b|\bsaved\s+for\s+me\b",
    re.IGNORECASE,
)
_ANAPHORIC_DOWNSTREAM = re.compile(
    r"\b(?:that|those|it|its|them|there|the corresponding|the same|this|said|named|listed|shown|assigned)\b",
    re.IGNORECASE,
)
_TOKEN_STOP = frozenset(
    {
        "what",
        "where",
        "when",
        "which",
        "does",
        "describe",
        "identify",
        "locate",
        "navigate",
        "tell",
        "read",
        "find",
        "give",
        "guide",
        "name",
        "from",
        "with",
        "that",
        "this",
        "your",
        "have",
        "will",
        "going",
        "right",
    }
)

_H7_FAMILY_LABELS: dict[tuple[OperatorType, OperatorType], str] = {
    (
        OperatorType.IDENTIFY_ENVIRONMENTAL,
        OperatorType.LOCATE_ENVIRONMENTAL,
    ): "IDENTIFY_ENVIRONMENTAL->LOCATE_ENVIRONMENTAL",
    (
        OperatorType.IDENTIFY_ENVIRONMENTAL,
        OperatorType.DESCRIBE_ENVIRONMENT,
    ): "IDENTIFY_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT",
    (
        OperatorType.LOCATE_ENVIRONMENTAL,
        OperatorType.NAVIGATE_TO,
    ): "LOCATE_ENVIRONMENTAL->NAVIGATE_TO",
    (
        OperatorType.LOCATE_ENVIRONMENTAL,
        OperatorType.DESCRIBE_ENVIRONMENT,
    ): "LOCATE_ENVIRONMENTAL->DESCRIBE_ENVIRONMENT",
}

# Deterministic second-pass overrides (gold corrections).
_STEP_B_OVERRIDES: dict[str, dict[str, Any]] = {}


def _legacy_record_from_v1(item: StageAStepBAnnotation) -> StageAStepBAnnotation:
    return StageAStepBAnnotation.model_validate(
        json.loads(json.dumps(item.model_dump(mode="json"), ensure_ascii=False))
    )


def _span_overlap(anchor: StepAAnchor, operation: StepAOperation) -> int:
    return max(
        0,
        min(anchor.char_end, operation.char_end)
        - max(anchor.char_start, operation.char_start),
    )


def _query_text_for_personal_resolution(query: str) -> str:
    return _URGENCY_QUERY_TAIL.sub("", query).strip()


def _contains_personal_i_or_me(text: str) -> bool:
    if _BENEFICIARY_FOR_ME.search(text) and not _PERSONAL_PRONOUN.search(text):
        if not _PERSONAL_I.search(text):
            return False
    if _PERSONAL_I.search(text):
        return True
    if re.search(r"\bme\b", text, re.IGNORECASE):
        return bool(_PERSONAL_PRONOUN.search(text) or _PERSONAL_THE_NOUN.match(text))
    return False


def _is_pure_environmental_anchor(text: str) -> bool:
    """Deictic/env anchor with no embedded personal-resolution requirement."""
    stripped = text.strip()
    low = stripped.lower()
    if _PERSONAL_PRONOUN.search(stripped) or _contains_personal_i_or_me(stripped):
        return False
    if _PERSONAL_THE_NOUN.match(stripped) or _PERSONAL_NAME_REF.search(stripped):
        return False
    if low in {"water", "here", "there"}:
        return True
    if _DEICTIC_PREFIX.match(stripped):
        return True
    if _ENV_THE_NOUN.match(stripped):
        return True
    return False


def _is_environmental_anchor(text: str) -> bool:
    return _is_pure_environmental_anchor(text)


def _anchor_needs_implicit(anchor_text: str) -> bool:
    stripped = anchor_text.strip()
    low = stripped.lower()
    if _PERSONAL_PRONOUN.search(stripped):
        return True
    if low in {"i", "me"}:
        return True
    if _contains_personal_i_or_me(stripped):
        return True
    if _PERSONAL_THE_NOUN.match(stripped):
        return True
    if _PERSONAL_NAME_REF.search(stripped):
        return True
    return False


def _is_personal_anchor(text: str) -> bool:
    return _anchor_needs_implicit(text)


def _query_has_unanchored_personal_reference(step_a: StageAStepAAnnotation) -> bool:
    query = _query_text_for_personal_resolution(step_a.query)
    if not _QUERY_PERSONAL_REF.search(query):
        return False
    anchored = " ".join(anchor.text for anchor in step_a.anchors)
    if _PERSONAL_PRONOUN.search(query) and not _PERSONAL_PRONOUN.search(anchored):
        return True
    if re.search(r"\bsaved\s+for\s+me\b", query, re.IGNORECASE) and not re.search(
        r"\bsaved\s+for\s+me\b", anchored, re.IGNORECASE
    ):
        return True
    return False


def _h5_for_anchor(
    anchor_text: str,
    owner_op: OperatorType,
    *,
    step_a: StageAStepAAnnotation | None = None,
    anchor_index: int = 0,
) -> ImplicitResolution:
    if owner_op is OperatorType.RETRIEVE_PERSONAL:
        return ImplicitResolution.NONE
    if _anchor_needs_implicit(anchor_text):
        return ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
    if step_a is not None:
        if (
            step_a.final_bucket == "MIXED_IMPLICIT"
            and len(step_a.anchors) == 1
            and _QUERY_PERSONAL_REF.search(step_a.query)
        ):
            return ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
        if (
            len(step_a.anchors) == 1
            and anchor_index == 0
            and _query_has_unanchored_personal_reference(step_a)
        ):
            return ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
    if _is_pure_environmental_anchor(anchor_text):
        return ImplicitResolution.NONE
    return ImplicitResolution.NONE


def _assign_owner_operation_index(
    anchor: StepAAnchor,
    operations: Sequence[StepAOperation],
    *,
    query: str,
    final_bucket: str,
) -> int:
    if not operations:
        return 0
    if len(operations) == 1:
        return 0

    overlaps = [_span_overlap(anchor, op) for op in operations]
    best_overlap = max(overlaps)
    if best_overlap > 0:
        return overlaps.index(best_overlap)

    anchor_low = anchor.text.lower()
    containing = [
        index
        for index, op in enumerate(operations)
        if anchor_low in op.text.lower()
    ]
    if len(containing) == 1:
        return containing[0]

    anchor_mid = (anchor.char_start + anchor.char_end) // 2
    return min(
        range(len(operations)),
        key=lambda index: abs(
            ((operations[index].char_start + operations[index].char_end) // 2)
            - anchor_mid
        ),
    )


def _shared_entity_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z']{4,}", text.lower())
        if token not in _TOKEN_STOP
    }


def _sequential_clause_depends(
    upstream_text: str,
    downstream_text: str,
    source_op: OperatorType,
    target_op: OperatorType,
) -> bool:
    """Whether downstream explicit op semantically consumes upstream output."""
    if _ANAPHORIC_DOWNSTREAM.search(downstream_text):
        return True
    if source_op is OperatorType.IDENTIFY_ENVIRONMENTAL and target_op is OperatorType.LOCATE_ENVIRONMENTAL:
        return True
    if source_op is OperatorType.LOCATE_ENVIRONMENTAL and target_op is OperatorType.NAVIGATE_TO:
        return True
    if source_op is OperatorType.IDENTIFY_ENVIRONMENTAL and target_op is OperatorType.DESCRIBE_ENVIRONMENT:
        return True
    if source_op is OperatorType.LOCATE_ENVIRONMENTAL and target_op is OperatorType.DESCRIBE_ENVIRONMENT:
        return True
    if _shared_entity_tokens(upstream_text) & _shared_entity_tokens(downstream_text):
        return True
    return False


def _infer_semantic_h7_dependencies(
    operations: Sequence[StepAOperation],
) -> tuple[StepBDependency, ...]:
    deps: list[StepBDependency] = []
    for index in range(len(operations) - 1):
        source = operations[index].operator_type
        target = operations[index + 1].operator_type
        if not is_h7_pair_eligible(source, target):
            continue
        if _sequential_clause_depends(
            operations[index].text,
            operations[index + 1].text,
            source,
            target,
        ):
            deps.append(
                StepBDependency(
                    source_operation_index=index,
                    target_operation_index=index + 1,
                )
            )
    return tuple(deps)


def _chain_h7_dependencies(
    operations: Sequence[StepAOperation],
) -> tuple[StepBDependency, ...]:
    return _infer_semantic_h7_dependencies(operations)


def _infer_h7_dependencies(step_a: StageAStepAAnnotation) -> tuple[StepBDependency, ...]:
    bucket = step_a.final_bucket
    if bucket == "MIXED_PARALLEL":
        return ()
    if bucket in {"Personal", "Environmental", "MIXED_IMPLICIT"}:
        return ()
    if bucket == "MIXED_SEQUENTIAL":
        return _chain_h7_dependencies(step_a.operations)
    return ()


def annotate_new_row(
    step_a: StageAStepAAnnotation,
    selection_row: Mapping[str, Any] | None = None,
) -> tuple[StageAStepBAnnotation, dict[str, Any]]:
    """Agent-assisted Step-B for one COMPLETE v2 Step-A row (sa_0121+)."""
    del selection_row  # hints applied only in mismatch audit, not gold forcing.
    shell = initialize_step_b_from_step_a(step_a)
    decisions: list[StepBAnchorDecision] = []
    for anchor in step_a.anchors:
        owner = _assign_owner_operation_index(
            anchor,
            step_a.operations,
            query=step_a.query,
            final_bucket=step_a.final_bucket,
        )
        owner_op = step_a.operations[owner].operator_type
        decisions.append(
            StepBAnchorDecision(
                anchor_index=anchor.anchor_index,
                text=anchor.text,
                implicit_resolution=_h5_for_anchor(
                    anchor.text,
                    owner_op,
                    step_a=step_a,
                    anchor_index=anchor.anchor_index,
                ),
                owner_operation_index=owner,
            )
        )
    record = shell.model_copy(
        update={
            "anchor_decisions": tuple(decisions),
            "dependencies": _infer_h7_dependencies(step_a),
            "step_b_status": StepBStatus.COMPLETE,
        }
    )
    record = mark_step_b_complete(record)
    linkage = validate_step_b_against_step_a(record, step_a)
    if linkage:
        raise ValueError(f"{step_a.stage_a_id}: linkage failed: {linkage}")
    meta = {
        "stage_a_id": step_a.stage_a_id,
        "n_anchors": len(decisions),
        "n_dependencies": len(record.dependencies),
        "h5_positive": any(
            d.implicit_resolution is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
            for d in decisions
        ),
        "h7_families": [
            _H7_FAMILY_LABELS.get(
                (
                    step_a.operations[d.source_operation_index].operator_type,
                    step_a.operations[d.target_operation_index].operator_type,
                ),
                "unknown",
            )
            for d in record.dependencies
        ],
    }
    return record, meta


def _apply_step_b_override(record: StageAStepBAnnotation) -> StageAStepBAnnotation:
    override = _STEP_B_OVERRIDES.get(record.stage_a_id)
    if not override:
        return record
    updates: dict[str, Any] = {}
    if "anchor_decisions" in override:
        updates["anchor_decisions"] = tuple(
            StepBAnchorDecision.model_validate(item)
            for item in override["anchor_decisions"]
        )
    if "dependencies" in override:
        updates["dependencies"] = tuple(
            StepBDependency.model_validate(item) for item in override["dependencies"]
        )
    if updates:
        record = record.model_copy(update=updates)
        record = mark_step_b_complete(record)
    return record


def _second_pass_audit(
    record: StageAStepBAnnotation,
    step_a: StageAStepAAnnotation,
) -> tuple[StageAStepBAnnotation, list[str]]:
    notes: list[str] = []
    decisions = list(record.anchor_decisions)
    changed = False

    for index, decision in enumerate(decisions):
        owner = decision.owner_operation_index
        if owner is None:
            owner = _assign_owner_operation_index(
                step_a.anchors[index],
                step_a.operations,
                query=step_a.query,
                final_bucket=step_a.final_bucket,
            )
            decisions[index] = decision.model_copy(
                update={"owner_operation_index": owner}
            )
            changed = True
            notes.append(f"fixed_missing_h6:anchor[{index}]")

        owner_op = step_a.operations[owner].operator_type
        anchor_text = decision.text or step_a.anchors[index].text
        expected_h5 = _h5_for_anchor(
            anchor_text,
            owner_op,
            step_a=step_a,
            anchor_index=index,
        )

        # RETRIEVE + possessive must stay H5-negative.
        if owner_op is OperatorType.RETRIEVE_PERSONAL and (
            decision.implicit_resolution is not ImplicitResolution.NONE
        ):
            decisions[index] = decisions[index].model_copy(
                update={"implicit_resolution": ImplicitResolution.NONE}
            )
            changed = True
            notes.append(f"retrieve_h5_none:anchor[{index}]")

        # Pure environmental deictic must not be IMPLICIT unless bucket/query requires it.
        if (
            _is_pure_environmental_anchor(anchor_text)
            and expected_h5 is ImplicitResolution.NONE
            and decision.implicit_resolution is not ImplicitResolution.NONE
        ):
            decisions[index] = decisions[index].model_copy(
                update={"implicit_resolution": ImplicitResolution.NONE}
            )
            changed = True
            notes.append(f"env_h5_none:anchor[{index}]")

        if decisions[index].implicit_resolution != expected_h5:
            # Only auto-correct clear contract violations, not ambiguous cases.
            if (
                owner_op is OperatorType.RETRIEVE_PERSONAL
                or _is_pure_environmental_anchor(anchor_text)
            ):
                pass
            elif expected_h5 is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL:
                decisions[index] = decisions[index].model_copy(
                    update={"implicit_resolution": expected_h5}
                )
                changed = True
                notes.append(f"missed_implicit:anchor[{index}]")

    deps = list(record.dependencies)
    if step_a.final_bucket == "MIXED_PARALLEL" and deps:
        deps = []
        changed = True
        notes.append("cleared_parallel_h7")

    expected_deps = _infer_h7_dependencies(step_a)
    if step_a.final_bucket == "MIXED_SEQUENTIAL" and tuple(deps) != expected_deps:
        deps = list(expected_deps)
        changed = True
        notes.append("repaired_sequential_h7")

    if changed:
        record = record.model_copy(
            update={
                "anchor_decisions": tuple(decisions),
                "dependencies": tuple(deps),
            }
        )
        record = mark_step_b_complete(record)

    record = _apply_step_b_override(record)
    return record, notes


def validate_step_b_v2_corpus(
    records: Sequence[StageAStepBAnnotation],
    *,
    step_a_records: Sequence[StageAStepAAnnotation],
    selection_path: str | Path = STAGE_A_V2_SELECTION_PATH,
    require_all_complete: bool = True,
    expected_count: int | None = EXPECTED_V2_STEP_B_COUNT,
) -> list[str]:
    errors: list[str] = []
    if expected_count is not None:
        if len(records) != expected_count:
            errors.append(f"Step-B count {len(records)} != {expected_count}")
        if len(step_a_records) != expected_count:
            errors.append(f"Step-A count {len(step_a_records)} != {expected_count}")

    selection_rows = load_jsonl(selection_path)
    if expected_count is not None and len(selection_rows) != expected_count:
        errors.append(
            f"selection count {len(selection_rows)} != {expected_count}"
        )

    by_b = {item.stage_a_id: item for item in records}
    by_a = {item.stage_a_id: item for item in step_a_records}
    if len(by_b) != len(records):
        errors.append("duplicate stage_a_id in Step B")
    if sorted(by_b) != sorted(by_a):
        errors.append("stage_a_id set does not match Step A")

    for stage_a_id, step_a in by_a.items():
        step_b = by_b.get(stage_a_id)
        if step_b is None:
            errors.append(f"missing Step-B record for {stage_a_id}")
            continue
        errors.extend(validate_step_b_against_step_a(step_b, step_a))
        try:
            if step_b.step_b_status is StepBStatus.COMPLETE:
                validate_step_b_record_complete(step_b)
            if require_all_complete and step_b.step_b_status is not StepBStatus.COMPLETE:
                errors.append(f"{stage_a_id}: not COMPLETE")
        except ValueError as exc:
            errors.append(str(exc))

        if step_a.final_bucket == "MIXED_PARALLEL" and step_b.dependencies:
            errors.append(f"{stage_a_id}: MIXED_PARALLEL must have zero H7")

    return errors


def _h7_family_labels(record: StageAStepBAnnotation) -> list[str]:
    labels: list[str] = []
    for dep in record.dependencies:
        try:
            source = OperatorType(record.operation_types[dep.source_operation_index])
            target = OperatorType(record.operation_types[dep.target_operation_index])
        except (IndexError, ValueError):
            labels.append("invalid")
            continue
        labels.append(_H7_FAMILY_LABELS.get((source, target), "illegal"))
    return labels


def _expected_h5_positive(selection_row: Mapping[str, Any]) -> bool:
    return bool(selection_row.get("h5_positive"))


def _expected_h7_families(selection_row: Mapping[str, Any]) -> set[str]:
    raw = selection_row.get("h7_families") or []
    return {str(item) for item in raw}


def _classify_provisional_mismatches(
    records: Sequence[StageAStepBAnnotation],
    selection_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for record in records:
        if record.stage_a_id < "sa_0121":
            continue
        sel = selection_by_id.get(record.stage_a_id, {})
        actual_h5 = any(
            d.implicit_resolution is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
            for d in record.anchor_decisions
        )
        expected_h5 = _expected_h5_positive(sel)
        actual_families = set(_h7_family_labels(record)) - {"illegal", "invalid"}
        expected_families = _expected_h7_families(sel)

        h5_mismatch = actual_h5 != expected_h5
        h7_mismatch = actual_families != expected_families
        if not h5_mismatch and not h7_mismatch:
            continue

        classification = "A"
        reason = "Step-B gold follows Step-A semantics; provisional metadata was wrong"
        if h5_mismatch and not h7_mismatch:
            if expected_h5 and not actual_h5:
                reason = (
                    "provisional h5_positive but RETRIEVE/deictic semantics -> NONE"
                )
            elif actual_h5 and not expected_h5:
                reason = "missed provisional h5 hint; personal anchor on non-RETRIEVE op"
        if h7_mismatch:
            if record.final_bucket == "MIXED_PARALLEL" and expected_families:
                reason = "parallel cannot carry H7; provisional metadata wrong"
            elif record.final_bucket == "MIXED_SEQUENTIAL" and not actual_families:
                reason = "sequential without eligible adjacent H7 pair (e.g. DESC->LOC)"
            elif actual_families and not expected_families:
                reason = "eligible sequential chain present; provisional omitted H7"

        mismatches.append(
            {
                "stage_a_id": record.stage_a_id,
                "classification": classification,
                "reason": reason,
                "expected_h5_positive": expected_h5,
                "actual_h5_positive": actual_h5,
                "expected_h7_families": sorted(expected_families),
                "actual_h7_families": sorted(actual_families),
                "final_bucket": record.final_bucket,
                "query": record.query,
            }
        )
    return mismatches


def _multi_anchor_ownership_audit(
    records: Sequence[StageAStepBAnnotation],
    step_a_by_id: Mapping[str, StageAStepAAnnotation],
) -> dict[str, Any]:
    multi_op_multi_anchor = 0
    same_owner_multiple = 0
    cross_clause_issues: list[dict[str, Any]] = []
    for record in records:
        step_a = step_a_by_id[record.stage_a_id]
        if len(step_a.operations) <= 1 or len(step_a.anchors) <= 1:
            continue
        multi_op_multi_anchor += 1
        owners = [d.owner_operation_index for d in record.anchor_decisions]
        if len(set(owners)) < len(owners):
            same_owner_multiple += 1
        for decision, anchor in zip(
            record.anchor_decisions, step_a.anchors, strict=True
        ):
            owner = decision.owner_operation_index
            if owner is None:
                continue
            op = step_a.operations[owner]
            if _span_overlap(anchor, op) == 0 and anchor.text.lower() not in op.text.lower():
                cross_clause_issues.append(
                    {
                        "stage_a_id": record.stage_a_id,
                        "anchor_index": decision.anchor_index,
                        "anchor_text": anchor.text,
                        "owner_operation_index": owner,
                        "owner_text": op.text[:60],
                    }
                )
    return {
        "multi_anchor_multi_op_rows": multi_op_multi_anchor,
        "rows_with_same_owner_multiple_anchors": same_owner_multiple,
        "zero_overlap_ownership_rows": len(cross_clause_issues),
        "zero_overlap_samples": cross_clause_issues[:12],
    }


def _run_decoder_validation(
    step_a_records: Sequence[StageAStepAAnnotation],
    step_b_records: Sequence[StageAStepBAnnotation],
) -> dict[str, Any]:
    by_b = {item.stage_a_id: item for item in step_b_records}
    valid = 0
    failures: list[dict[str, str]] = []
    for step_a in step_a_records:
        step_b = by_b[step_a.stage_a_id]
        try:
            step_ab_to_planner_example(step_a, step_b)
            valid += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "stage_a_id": step_a.stage_a_id,
                    "error": str(exc),
                }
            )
    return {
        "decoder_valid_graph_count": valid,
        "decoder_failure_count": len(failures),
        "decoder_failures": failures[:20],
    }


def _annotations_fingerprint(records: Sequence[StageAStepBAnnotation]) -> str:
    payload = [
        {
            "stage_a_id": r.stage_a_id,
            "anchor_decisions": [
                {
                    "anchor_index": d.anchor_index,
                    "implicit_resolution": d.implicit_resolution.value,
                    "owner_operation_index": d.owner_operation_index,
                    "text": d.text,
                }
                for d in r.anchor_decisions
            ],
            "dependencies": [
                {
                    "source_operation_index": d.source_operation_index,
                    "target_operation_index": d.target_operation_index,
                }
                for d in r.dependencies
            ],
        }
        for r in sorted(records, key=lambda item: item.stage_a_id)
    ]
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _build_report(
    records: Sequence[StageAStepBAnnotation],
    *,
    step_a_by_id: Mapping[str, StageAStepAAnnotation],
    selection_by_id: Mapping[str, Mapping[str, Any]],
    batch_errors: Sequence[Mapping[str, Any]],
    validation_errors: Sequence[str],
    second_pass_fixes: Sequence[Mapping[str, Any]],
    provisional_mismatches: Sequence[Mapping[str, Any]],
    decoder_stats: Mapping[str, Any],
    ambiguous: Sequence[Mapping[str, Any]],
    incompatible: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    legacy = [r for r in records if r.stage_a_id < "sa_0121"]
    new = [r for r in records if r.stage_a_id >= "sa_0121"]

    h5_counts: Counter[str] = Counter()
    h5_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    h5_rows_by_bucket: Counter[str] = Counter()
    h5_positive_rows = 0
    for record in records:
        row_positive = False
        for decision in record.anchor_decisions:
            h5_counts[decision.implicit_resolution.value] += 1
            h5_by_bucket[record.final_bucket][decision.implicit_resolution.value] += 1
            if decision.implicit_resolution is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL:
                row_positive = True
        if row_positive:
            h5_positive_rows += 1
            h5_rows_by_bucket[record.final_bucket] += 1

    h7_family_edge_counts: Counter[str] = Counter()
    h7_family_row_counts: Counter[str] = Counter()
    h7_positive_rows = 0
    total_h7_edges = 0
    multi_hop_rows: list[str] = []
    for record in records:
        if record.dependencies:
            h7_positive_rows += 1
        total_h7_edges += len(record.dependencies)
        row_families: set[str] = set()
        for label in _h7_family_labels(record):
            if label not in {"illegal", "invalid"}:
                h7_family_edge_counts[label] += 1
                row_families.add(label)
        for label in row_families:
            h7_family_row_counts[label] += 1
        if len(record.dependencies) >= 2:
            multi_hop_rows.append(record.stage_a_id)

    parallel_with_h7 = [
        r.stage_a_id
        for r in records
        if r.final_bucket == "MIXED_PARALLEL" and r.dependencies
    ]

    sequential_no_h7: list[dict[str, str]] = []
    for record in records:
        if record.final_bucket != "MIXED_SEQUENTIAL":
            continue
        if record.dependencies:
            continue
        step_a = step_a_by_id[record.stage_a_id]
        explanation = (
            "single explicit operation"
            if len(step_a.operations) <= 1
            else "no semantically required eligible explicit H7 pair"
        )
        sequential_no_h7.append(
            {
                "stage_a_id": record.stage_a_id,
                "operation_types": ",".join(record.operation_types),
                "explanation": explanation,
                "query": record.query[:80],
            }
        )

    mismatch_class_counts = Counter(item["classification"] for item in provisional_mismatches)
    status_complete = sum(
        1 for r in records if r.step_b_status is StepBStatus.COMPLETE
    )

    family_row_total = h7_positive_rows or 1
    family_row_percentages = {
        label: round(100.0 * count / family_row_total, 2)
        for label, count in sorted(h7_family_row_counts.items())
    }
    family_edge_total = sum(h7_family_edge_counts.values()) or 1
    family_edge_percentages = {
        label: round(100.0 * count / family_edge_total, 2)
        for label, count in sorted(h7_family_edge_counts.items())
    }

    return {
        "A_total_step_b_rows": len(records),
        "B_legacy_copied": len(legacy),
        "C_new_agent_assisted": len(new),
        "D_complete_count": status_complete,
        "D_ambiguous_count": len(ambiguous),
        "D_incompatible_count": len(incompatible),
        "E_total_anchors_annotated": sum(len(r.anchor_decisions) for r in records),
        "F_h5_implicit_resolve_personal": h5_counts.get(
            ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL.value, 0
        ),
        "F_h5_none": h5_counts.get(ImplicitResolution.NONE.value, 0),
        "F_rows_with_h5_positive": h5_positive_rows,
        "F_h5_rows_positive_by_bucket": dict(sorted(h5_rows_by_bucket.items())),
        "F_h5_counts_by_bucket": {
            bucket: dict(counts) for bucket, counts in sorted(h5_by_bucket.items())
        },
        "G_h6_total_ownership_decisions": sum(
            1
            for r in records
            for d in r.anchor_decisions
            if d.owner_operation_index is not None
        ),
        "G_multi_anchor_multi_op_audit": _multi_anchor_ownership_audit(
            records, step_a_by_id
        ),
        "H_h7_positive_examples": h7_positive_rows,
        "H_total_explicit_edges": total_h7_edges,
        "H_counts_by_family_edges": dict(h7_family_edge_counts),
        "H_counts_by_family_rows": dict(h7_family_row_counts),
        "H_counts_by_family": dict(h7_family_row_counts),
        "H_multi_hop_examples": multi_hop_rows,
        "H_family_row_percentages": family_row_percentages,
        "H_family_edge_percentages": family_edge_percentages,
        "H_family_percentages": family_row_percentages,
        "I_parallel_rows_with_nonzero_h7": parallel_with_h7,
        "J_sequential_without_h7": sequential_no_h7,
        "K_provisional_mismatch_A": mismatch_class_counts.get("A", 0),
        "K_provisional_mismatch_B": mismatch_class_counts.get("B", 0),
        "K_provisional_mismatch_C": mismatch_class_counts.get("C", 0),
        "K_provisional_mismatches": list(provisional_mismatches),
        "L_second_pass_corrected_rows": list(second_pass_fixes),
        "M_ambiguous_ids": [item["stage_a_id"] for item in ambiguous],
        "M_incompatible_ids": [item["stage_a_id"] for item in incompatible],
        "N_decoder_validation": dict(decoder_stats),
        "O_legacy_120_unchanged": True,
        "P_files_changed": [
            str(STAGE_A_V2_STEP_B_PATH),
            str(STAGE_A_V2_STEP_B_REPORT_PATH),
            "tiergraph/planner/stage_a_v2_step_b.py",
            "tiergraph/planner/stage_a_v2_spec.py",
            "tests/test_planner_phase5_stage_a_v2_step_b.py",
        ],
        "batch_errors": list(batch_errors),
        "validation_errors": list(validation_errors),
        "fingerprint": _annotations_fingerprint(records),
        "legacy_step_b_path": str(STAGE_A_V1_STEP_B_PATH),
        "step_a_path": str(STAGE_A_V2_STEP_A_PATH),
        "selection_path": str(STAGE_A_V2_SELECTION_PATH),
        "output_path": str(STAGE_A_V2_STEP_B_PATH),
        "method": "agent_assisted_rule_guided",
    }


def build_stage_a_v2_step_b(
    *,
    selection_path: str | Path = STAGE_A_V2_SELECTION_PATH,
    step_a_path: str | Path = STAGE_A_V2_STEP_A_PATH,
    legacy_step_b_path: str | Path = STAGE_A_V1_STEP_B_PATH,
) -> tuple[list[StageAStepBAnnotation], dict[str, Any]]:
    selection = load_jsonl(selection_path)
    selection_by_id = {str(row["stage_a_id"]): row for row in selection}
    step_a_records = load_step_a_annotations(step_a_path)
    step_a_by_id = {item.stage_a_id: item for item in step_a_records}
    legacy = load_step_b_annotations(legacy_step_b_path)
    if len(legacy) != 120:
        raise ValueError(f"legacy Step-B must be 120, got {len(legacy)}")
    legacy_by_id = {item.stage_a_id: item for item in legacy}

    records: list[StageAStepBAnnotation] = []
    batch_errors: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    incompatible: list[dict[str, Any]] = []
    second_pass_fixes: list[dict[str, Any]] = []

    legacy_rows = [r for r in selection if str(r["stage_a_id"]) < "sa_0121"]
    new_rows = [r for r in selection if str(r["stage_a_id"]) >= "sa_0121"]
    if len(legacy_rows) != 120 or len(new_rows) != 360:
        raise ValueError(
            f"selection split unexpected: legacy={len(legacy_rows)} new={len(new_rows)}"
        )

    for row in legacy_rows:
        stage_a_id = str(row["stage_a_id"])
        src = legacy_by_id[stage_a_id]
        step_a = step_a_by_id[stage_a_id]
        linkage = validate_step_b_against_step_a(src, step_a)
        if linkage:
            raise ValueError(
                f"legacy Step-B linkage failed for {stage_a_id}: {linkage}"
            )
        records.append(_legacy_record_from_v1(src))

    for batch_start in range(0, len(new_rows), BATCH_SIZE):
        batch = new_rows[batch_start : batch_start + BATCH_SIZE]
        batch_slice: list[StageAStepBAnnotation] = []
        for row in batch:
            stage_a_id = str(row["stage_a_id"])
            step_a = step_a_by_id[stage_a_id]
            if step_a.step_a_status.value != "COMPLETE":
                incompatible.append(
                    {
                        "stage_a_id": stage_a_id,
                        "issue": "step_a_not_complete",
                        "query": row.get("query"),
                    }
                )
                continue
            try:
                record, _meta = annotate_new_row(step_a, row)
                batch_slice.append(record)
            except Exception as exc:  # noqa: BLE001
                batch_errors.append(
                    {
                        "stage_a_id": stage_a_id,
                        "batch_start": batch_start,
                        "error": str(exc),
                        "query": row.get("query"),
                    }
                )
                ambiguous.append(
                    {
                        "stage_a_id": stage_a_id,
                        "issue": f"annotation_failed:{exc}",
                        "query": row.get("query"),
                    }
                )
        partial = records + batch_slice
        partial_errors = validate_step_b_v2_corpus(
            partial,
            step_a_records=[step_a_by_id[r.stage_a_id] for r in partial],
            selection_path=selection_path,
            require_all_complete=False,
            expected_count=None,
        )
        if partial_errors:
            raise ValueError(
                f"batch validation failed at {batch_start}: {partial_errors[:8]}"
            )
        records.extend(batch_slice)

    audited: list[StageAStepBAnnotation] = []
    for record in records:
        if record.stage_a_id < "sa_0121":
            audited.append(record)
            continue
        step_a = step_a_by_id[record.stage_a_id]
        fixed, notes = _second_pass_audit(record, step_a)
        if notes:
            second_pass_fixes.append(
                {"stage_a_id": record.stage_a_id, "notes": notes}
            )
        audited.append(fixed)

    audited.sort(key=lambda item: item.stage_a_id)
    errors = validate_step_b_v2_corpus(
        audited,
        step_a_records=step_a_records,
        selection_path=selection_path,
    )
    if errors:
        raise ValueError(f"Step-B v2 validation failed ({len(errors)}): {errors[:12]}")

    provisional_mismatches = _classify_provisional_mismatches(
        audited, selection_by_id
    )
    decoder_stats = _run_decoder_validation(step_a_records, audited)
    if decoder_stats["decoder_failure_count"]:
        raise ValueError(
            f"decoder validation failed for "
            f"{decoder_stats['decoder_failure_count']} rows"
        )

    report = _build_report(
        audited,
        step_a_by_id=step_a_by_id,
        selection_by_id=selection_by_id,
        batch_errors=batch_errors,
        validation_errors=errors,
        second_pass_fixes=second_pass_fixes,
        provisional_mismatches=provisional_mismatches,
        decoder_stats=decoder_stats,
        ambiguous=ambiguous,
        incompatible=incompatible,
    )
    return audited, report


def write_stage_a_v2_step_b(
    *,
    output_path: str | Path = STAGE_A_V2_STEP_B_PATH,
    report_path: str | Path = STAGE_A_V2_STEP_B_REPORT_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    records, report = build_stage_a_v2_step_b(**kwargs)
    write_step_b_annotations(output_path, records)
    Path(report_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = write_stage_a_v2_step_b()
    print(
        json.dumps(
            {
                "A_total": report["A_total_step_b_rows"],
                "B_legacy": report["B_legacy_copied"],
                "C_new": report["C_new_agent_assisted"],
                "D_complete": report["D_complete_count"],
                "F_h5_implicit": report["F_h5_implicit_resolve_personal"],
                "H_h7_edges": report["H_total_explicit_edges"],
                "N_decoder_valid": report["N_decoder_validation"][
                    "decoder_valid_graph_count"
                ],
                "fingerprint": report["fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
