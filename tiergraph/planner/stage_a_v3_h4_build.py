"""Build Stage-A v3 from v2 TRAIN+DEV under frozen H4_REFEXPR_V1.

Applies SAFE_MECHANICAL corrections plus **explicit recorded human decisions**
(approved R6 shorten / R5 remove / final replace). Detector output may queue
HUMAN_REVIEW candidates; queuing never decides labels. Ambiguous cases are
never automatically decided.

TEST isolation:
- Split manifest may be read solely for TRAIN/DEV membership.
- Step-A / Step-B annotation rows are json.loads'd only when the raw line
  contains an allowed TRAIN/DEV ``stage_a_id``.
- TEST annotation rows are never parsed, validated, rewritten, or written.
- v3 annotation files materialize TRAIN+DEV only (432 rows) until a separate
  blind TEST migration.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from tiergraph.planner.annotation_step_a import (
    StageAStepAAnnotation,
    StepAAnchor,
    fingerprint_file,
    write_step_a_annotations,
)
from tiergraph.planner.annotation_step_b import (
    StageAStepBAnnotation,
    StepBAnchorDecision,
    validate_step_b_against_step_a,
    write_step_b_annotations,
)
from tiergraph.planner.annotations import ImplicitResolution
from tiergraph.planner.h4_refexpr_v1 import (
    AFFECTED_DEV_IDS,
    AFFECTED_TRAIN_IDS,
    APPROVED_EDIT_IDS,
    APPROVED_FINAL_REPLACE,
    APPROVED_R4_KEEP,
    APPROVED_R5_REMOVE,
    APPROVED_R6_SHORTEN,
    H4_ANCHOR_CONTRACT,
    HUMAN_REVIEW_DEV_IDS,
    HUMAN_REVIEW_IDS,
    HUMAN_REVIEW_TRAIN_IDS,
    SAFE_MECHANICAL_DEV_IDS,
    SAFE_MECHANICAL_IDS,
    SAFE_MECHANICAL_TRAIN_IDS,
    SOURCE_CORPUS_V2,
    UNRESOLVED_HUMAN_REVIEW_IDS,
    HumanReviewFlag,
    classify_human_review_reasons,
    is_safe_mechanical_anchor_text,
    primary_review_rule,
    shorten_anchor_to_text,
    shorten_safe_mechanical_anchor,
    sync_step_b_text_preserve_h5_h6,
)
from tiergraph.planner.stage_a_to_corpus import (
    step_ab_to_planner_example,
    validate_step_ab_linkage,
)
from tiergraph.planner.stage_a_v2_spec import (
    STAGE_A_V2_SELECTION_PATH,
    STAGE_A_V2_SPLIT_PATH,
    STAGE_A_V2_STEP_A_PATH,
    STAGE_A_V2_STEP_B_PATH,
)
from tiergraph.planner.stage_a_v3_spec import (
    STAGE_A_V3_BUILD_REPORT_PATH,
    STAGE_A_V3_DEV_SIZE,
    STAGE_A_V3_HUMAN_REVIEW_QUEUE_PATH,
    STAGE_A_V3_MATERIALIZED_SIZE,
    STAGE_A_V3_SPLIT_PATH,
    STAGE_A_V3_STEP_A_PATH,
    STAGE_A_V3_STEP_B_PATH,
    STAGE_A_V3_TEST_ANNOTATION_MIGRATION,
    STAGE_A_V3_TRAIN_DEV_H4_STATUS,
    STAGE_A_V3_TRAIN_SIZE,
    STAGE_A_V3_UNRESOLVED_REVIEW_PATH,
)

TModel = TypeVar("TModel", bound=BaseModel)


def annotation_corpus_fingerprint(
    step_a_path: str | Path,
    step_b_path: str | Path,
) -> str:
    digest = hashlib.sha256()
    digest.update(Path(step_a_path).read_bytes())
    digest.update(b"\0")
    digest.update(Path(step_b_path).read_bytes())
    return digest.hexdigest()


def load_train_dev_split_ids(
    split_path: str | Path = STAGE_A_V2_SPLIT_PATH,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return TRAIN/DEV ID sets from the split manifest.

    TEST membership rows are skipped after seeing ``split == \"test\"``; their
    stage_a_id values are never retained or returned.
    """
    train: set[str] = set()
    dev: set[str] = set()
    with Path(split_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            split_name = str(row["split"])
            if split_name == "train":
                train.add(str(row["stage_a_id"]))
            elif split_name == "dev":
                dev.add(str(row["stage_a_id"]))
            # skip test without retaining the id
    return frozenset(train), frozenset(dev)


def load_annotation_rows_for_ids(
    path: str | Path,
    ids: frozenset[str] | set[str],
    *,
    model_cls: type[TModel],
) -> dict[str, TModel]:
    """Parse JSONL annotation rows only for the requested ``stage_a_id`` set.

    A line is ``json.loads``'d only when its raw text contains an allowed id
    needle. All other lines (including every TEST annotation row) are skipped
    without parsing.
    """
    allowed = frozenset(ids)
    if not allowed:
        return {}
    needles = {sid: f'"stage_a_id": "{sid}"' for sid in allowed}
    out: dict[str, TModel] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            matched_sid: str | None = None
            for sid, needle in needles.items():
                if needle in line:
                    matched_sid = sid
                    break
            if matched_sid is None:
                continue
            payload = json.loads(line)
            if str(payload.get("stage_a_id")) != matched_sid:
                raise RuntimeError(
                    f"{path}: stage_a_id mismatch for needle {matched_sid!r}"
                )
            if matched_sid in out:
                raise RuntimeError(f"{path}: duplicate stage_a_id {matched_sid}")
            out[matched_sid] = model_cls.model_validate(payload)
    missing = allowed - set(out)
    if missing:
        raise RuntimeError(
            f"{path}: missing requested ids: {sorted(missing)[:10]}"
        )
    extra = set(out) - allowed
    if extra:
        raise RuntimeError(f"{path}: unexpected ids loaded: {sorted(extra)[:10]}")
    return out


def _reason_human_review_required(rule: str, reasons: tuple[str, ...]) -> str:
    if rule == "R5":
        return (
            "Function-word / non-referring gold anchor; deterministic removal "
            "not proven safe for H5/H6 and graph structure"
        )
    if "elevator_going" in reasons:
        return (
            "Elevator/going postmodifier boundary is ambiguous; "
            "auto-shortening could change referent or H5 carrier"
        )
    if "pred_tail" in reasons or "this_the_non_right" in reasons:
        return (
            "Demonstrative+NP with trailing predicate/adjective material; "
            "pronoun-vs-determiner boundary needs human judgment"
        )
    if "state_in_np" in reasons:
        return (
            "State description inside NP ('on or off' / similar); "
            "boundary vs predicate split is ambiguous"
        )
    if "the_one" in reasons:
        return (
            "Unusual anaphor ('the one …'); may be sole IMPLICIT carrier — "
            "review carefully without auto-rewrite"
        )
    return "Ambiguous H4 boundary under H4_REFEXPR_V1; queued for human review"


def build_human_review_queue(
    step_a_by_id: Mapping[str, StageAStepAAnnotation],
    step_b_by_id: Mapping[str, StageAStepBAnnotation],
    *,
    train_ids: frozenset[str],
    dev_ids: frozenset[str],
) -> list[HumanReviewFlag]:
    """Queue HUMAN_REVIEW anchors. Never mutates annotations."""
    flags: list[HumanReviewFlag] = []
    for stage_a_id in sorted(HUMAN_REVIEW_IDS):
        if stage_a_id in train_ids:
            split_name = "train"
        elif stage_a_id in dev_ids:
            split_name = "dev"
        else:
            raise ValueError(
                f"HUMAN_REVIEW id {stage_a_id} is not in TRAIN/DEV "
                "(TEST must not be queued here)"
            )
        if stage_a_id in SAFE_MECHANICAL_IDS:
            raise ValueError(
                f"{stage_a_id} cannot be both SAFE_MECHANICAL and HUMAN_REVIEW"
            )
        step_a = step_a_by_id[stage_a_id]
        step_b = step_b_by_id[stage_a_id]
        found = False
        for anchor, decision in zip(
            step_a.anchors, step_b.anchor_decisions, strict=True
        ):
            reasons = classify_human_review_reasons(anchor.text)
            if not reasons:
                continue
            found = True
            rule = primary_review_rule(reasons)
            flags.append(
                HumanReviewFlag(
                    stage_a_id=stage_a_id,
                    split=split_name,
                    query=step_a.query,
                    anchor_index=anchor.anchor_index,
                    existing_anchor=anchor.text,
                    char_start=anchor.char_start,
                    char_end=anchor.char_end,
                    flagged_issue=",".join(reasons),
                    rule=rule,
                    h5=decision.implicit_resolution.value,
                    h6_owner=decision.owner_operation_index,
                    reason_human_review_required=_reason_human_review_required(
                        rule, reasons
                    ),
                )
            )
        if not found:
            raise ValueError(
                f"HUMAN_REVIEW id {stage_a_id} has no detector-matching anchor"
            )
    return flags


def apply_safe_mechanical_to_pair(
    step_a: StageAStepAAnnotation,
    step_b: StageAStepBAnnotation,
) -> tuple[StageAStepAAnnotation, StageAStepBAnnotation, list[dict[str, Any]]]:
    """Apply SAFE_MECHANICAL shortenings; preserve H5/H6 on every decision."""
    if step_a.stage_a_id not in SAFE_MECHANICAL_IDS:
        raise ValueError(f"{step_a.stage_a_id} is not in SAFE_MECHANICAL set")

    new_anchors = []
    new_decisions = []
    changes: list[dict[str, Any]] = []
    shortened = 0

    for anchor, decision in zip(
        step_a.anchors, step_b.anchor_decisions, strict=True
    ):
        if is_safe_mechanical_anchor_text(anchor.text):
            old_h5 = decision.implicit_resolution
            old_h6 = decision.owner_operation_index
            new_anchor = shorten_safe_mechanical_anchor(anchor, step_a.query)
            new_decision = sync_step_b_text_preserve_h5_h6(
                decision, new_anchor.text
            )
            if new_decision.implicit_resolution != old_h5:
                raise RuntimeError("H5 must be preserved under SAFE_MECHANICAL")
            if new_decision.owner_operation_index != old_h6:
                raise RuntimeError("H6 must be preserved under SAFE_MECHANICAL")
            changes.append(
                {
                    "stage_a_id": step_a.stage_a_id,
                    "anchor_index": anchor.anchor_index,
                    "operation": "shorten",
                    "old_text": anchor.text,
                    "new_text": new_anchor.text,
                    "old_char_start": anchor.char_start,
                    "old_char_end": anchor.char_end,
                    "new_char_start": new_anchor.char_start,
                    "new_char_end": new_anchor.char_end,
                    "h5_preserved": old_h5.value,
                    "h6_preserved": old_h6,
                }
            )
            shortened += 1
            new_anchors.append(new_anchor)
            new_decisions.append(new_decision)
        else:
            new_anchors.append(anchor)
            new_decisions.append(decision)

    if shortened != 1:
        raise ValueError(
            f"{step_a.stage_a_id}: expected exactly 1 SAFE_MECHANICAL shorten, "
            f"got {shortened}"
        )

    new_a = step_a.model_copy(update={"anchors": tuple(new_anchors)})
    new_b = step_b.model_copy(update={"anchor_decisions": tuple(new_decisions)})
    return new_a, new_b, changes


def apply_approved_r6_shorten(
    step_a: StageAStepAAnnotation,
    step_b: StageAStepBAnnotation,
    *,
    old_text: str,
    new_text: str,
) -> tuple[StageAStepAAnnotation, StageAStepBAnnotation, dict[str, Any]]:
    """Apply one approved R6 boundary shortening; preserve H5/H6."""
    matched = [
        (i, anc)
        for i, anc in enumerate(step_a.anchors)
        if anc.text == old_text
    ]
    if len(matched) != 1:
        raise ValueError(
            f"{step_a.stage_a_id}: expected exactly one anchor {old_text!r}, "
            f"found {len(matched)}"
        )
    idx, anchor = matched[0]
    decision = step_b.anchor_decisions[idx]
    old_h5 = decision.implicit_resolution
    old_h6 = decision.owner_operation_index
    new_anchor = shorten_anchor_to_text(anchor, step_a.query, new_text)
    new_decision = sync_step_b_text_preserve_h5_h6(decision, new_text)
    if new_decision.implicit_resolution != old_h5:
        raise RuntimeError("H5 must be preserved under approved R6 shorten")
    if new_decision.owner_operation_index != old_h6:
        raise RuntimeError("H6 must be preserved under approved R6 shorten")

    anchors = list(step_a.anchors)
    decisions = list(step_b.anchor_decisions)
    anchors[idx] = new_anchor
    decisions[idx] = new_decision
    change = {
        "stage_a_id": step_a.stage_a_id,
        "anchor_index": idx,
        "operation": "approved_r6_shorten",
        "old_text": old_text,
        "new_text": new_text,
        "old_char_start": anchor.char_start,
        "old_char_end": anchor.char_end,
        "new_char_start": new_anchor.char_start,
        "new_char_end": new_anchor.char_end,
        "h5_preserved": old_h5.value,
        "h6_preserved": old_h6,
    }
    return (
        step_a.model_copy(update={"anchors": tuple(anchors)}),
        step_b.model_copy(update={"anchor_decisions": tuple(decisions)}),
        change,
    )


def apply_approved_r5_remove(
    step_a: StageAStepAAnnotation,
    step_b: StageAStepBAnnotation,
    *,
    remove_text: str,
) -> tuple[StageAStepAAnnotation, StageAStepBAnnotation, dict[str, Any]]:
    """Remove one approved R5 function-word anchor; reindex survivors."""
    matched = [
        (i, anc, dec)
        for i, (anc, dec) in enumerate(
            zip(step_a.anchors, step_b.anchor_decisions, strict=True)
        )
        if anc.text == remove_text
    ]
    if len(matched) != 1:
        raise ValueError(
            f"{step_a.stage_a_id}: expected exactly one anchor {remove_text!r}, "
            f"found {len(matched)}"
        )
    idx, anchor, decision = matched[0]
    new_anchors = []
    new_decisions = []
    for i, (anc, dec) in enumerate(
        zip(step_a.anchors, step_b.anchor_decisions, strict=True)
    ):
        if i == idx:
            continue
        new_index = len(new_anchors)
        new_anchors.append(anc.model_copy(update={"anchor_index": new_index}))
        new_decisions.append(dec.model_copy(update={"anchor_index": new_index}))

    change = {
        "stage_a_id": step_a.stage_a_id,
        "anchor_index": idx,
        "operation": "approved_r5_remove",
        "old_text": remove_text,
        "new_text": None,
        "old_char_start": anchor.char_start,
        "old_char_end": anchor.char_end,
        "h5_removed_with_anchor": decision.implicit_resolution.value,
        "h6_removed_with_anchor": decision.owner_operation_index,
        "anchors_remaining": len(new_anchors),
    }
    new_a = step_a.model_copy(update={"anchors": tuple(new_anchors)})
    new_b = step_b.model_copy(
        update={
            "n_anchors": len(new_anchors),
            "anchor_decisions": tuple(new_decisions),
        }
    )
    return new_a, new_b, change


def apply_approved_final_replace(
    step_a: StageAStepAAnnotation,
    step_b: StageAStepBAnnotation,
    *,
    old_text: str,
    new_anchor_specs: Sequence[Mapping[str, Any]],
) -> tuple[StageAStepAAnnotation, StageAStepBAnnotation, dict[str, Any]]:
    """Replace one H4 anchor with one or more anchors at explicit offsets.

    Does not assume the replacement starts at the old ``char_start``. Surviving
    non-replaced anchors are kept; the full set is re-sorted by span and
    reindexed. Step-B H7 dependencies are preserved.
    """
    matched = [
        i for i, anc in enumerate(step_a.anchors) if anc.text == old_text
    ]
    if len(matched) != 1:
        raise ValueError(
            f"{step_a.stage_a_id}: expected exactly one anchor {old_text!r}, "
            f"found {len(matched)}"
        )
    old_idx = matched[0]
    old_anchor = step_a.anchors[old_idx]

    survivors_a = [
        anc for i, anc in enumerate(step_a.anchors) if i != old_idx
    ]
    survivors_b = [
        dec for i, dec in enumerate(step_b.anchor_decisions) if i != old_idx
    ]

    built: list[tuple[StepAAnchor, StepBAnchorDecision]] = []
    for spec in new_anchor_specs:
        text = str(spec["text"])
        char_start = int(spec["char_start"])
        char_end = int(spec["char_end"])
        if step_a.query[char_start:char_end] != text:
            raise ValueError(
                f"{step_a.stage_a_id}: offset mismatch for {text!r}: "
                f"query[{char_start}:{char_end}]="
                f"{step_a.query[char_start:char_end]!r}"
            )
        h5 = ImplicitResolution(str(spec["h5"]))
        h6 = int(spec["h6"])
        # placeholder index; reassigned after sort
        built.append(
            (
                StepAAnchor(
                    anchor_index=0,
                    text=text,
                    char_start=char_start,
                    char_end=char_end,
                ),
                StepBAnchorDecision(
                    anchor_index=0,
                    text=text,
                    implicit_resolution=h5,
                    owner_operation_index=h6,
                ),
            )
        )

    combined: list[tuple[StepAAnchor, StepBAnchorDecision]] = []
    for anc, dec in zip(survivors_a, survivors_b, strict=True):
        combined.append((anc, dec))
    combined.extend(built)
    combined.sort(key=lambda pair: (pair[0].char_start, pair[0].char_end))

    new_anchors: list[StepAAnchor] = []
    new_decisions: list[StepBAnchorDecision] = []
    for index, (anc, dec) in enumerate(combined):
        new_anchors.append(anc.model_copy(update={"anchor_index": index}))
        new_decisions.append(
            dec.model_copy(
                update={
                    "anchor_index": index,
                    "text": anc.text if index >= 0 else dec.text,
                }
            )
        )
        # Keep decision text aligned to anchor text after relocation.
        new_decisions[-1] = new_decisions[-1].model_copy(
            update={"text": new_anchors[-1].text}
        )

    # For newly built decisions, H5/H6 already set; for survivors preserve.
    # Re-apply explicit H5/H6 from specs onto matching texts after sort.
    by_text_spec = {str(spec["text"]): spec for spec in new_anchor_specs}
    for i, anc in enumerate(new_anchors):
        if anc.text in by_text_spec:
            spec = by_text_spec[anc.text]
            new_decisions[i] = StepBAnchorDecision(
                anchor_index=i,
                text=anc.text,
                implicit_resolution=ImplicitResolution(str(spec["h5"])),
                owner_operation_index=int(spec["h6"]),
            )

    change = {
        "stage_a_id": step_a.stage_a_id,
        "operation": "approved_final_replace",
        "old_text": old_text,
        "old_char_start": old_anchor.char_start,
        "old_char_end": old_anchor.char_end,
        "new_anchors": [
            {
                "text": anc.text,
                "char_start": anc.char_start,
                "char_end": anc.char_end,
                "h5": dec.implicit_resolution.value,
                "h6": dec.owner_operation_index,
            }
            for anc, dec in zip(new_anchors, new_decisions, strict=True)
            if anc.text in by_text_spec
        ],
        "n_anchors_before": len(step_a.anchors),
        "n_anchors_after": len(new_anchors),
        "h7_preserved": [
            {
                "source_operation_index": dep.source_operation_index,
                "target_operation_index": dep.target_operation_index,
            }
            for dep in step_b.dependencies
        ],
    }
    new_a = step_a.model_copy(update={"anchors": tuple(new_anchors)})
    new_b = step_b.model_copy(
        update={
            "n_anchors": len(new_anchors),
            "anchor_decisions": tuple(new_decisions),
            # dependencies unchanged
            "dependencies": step_b.dependencies,
        }
    )
    return new_a, new_b, change


_UNRESOLVED_NEEDED: dict[str, str] = {}


def build_unresolved_review_report(
    step_a_by_id: Mapping[str, StageAStepAAnnotation],
    step_b_by_id: Mapping[str, StageAStepBAnnotation],
    *,
    train_ids: frozenset[str],
    dev_ids: frozenset[str],
) -> list[dict[str, Any]]:
    """Document unresolved HUMAN_REVIEW cases (no edits)."""
    rows: list[dict[str, Any]] = []
    for stage_a_id in sorted(UNRESOLVED_HUMAN_REVIEW_IDS):
        if stage_a_id in train_ids:
            split_name = "train"
        elif stage_a_id in dev_ids:
            split_name = "dev"
        else:
            raise ValueError(f"unresolved id {stage_a_id} not in TRAIN/DEV")
        step_a = step_a_by_id[stage_a_id]
        step_b = step_b_by_id[stage_a_id]
        rows.append(
            {
                "stage_a_id": stage_a_id,
                "split": split_name,
                "query": step_a.query,
                "operations": [
                    {
                        "operation_index": op.operation_index,
                        "operator_type": op.operator_type.value,
                        "text": op.text,
                        "char_start": op.char_start,
                        "char_end": op.char_end,
                    }
                    for op in step_a.operations
                ],
                "anchors": [
                    {
                        "anchor_index": anc.anchor_index,
                        "text": anc.text,
                        "char_start": anc.char_start,
                        "char_end": anc.char_end,
                        "h5": dec.implicit_resolution.value,
                        "h6_owner": dec.owner_operation_index,
                    }
                    for anc, dec in zip(
                        step_a.anchors, step_b.anchor_decisions, strict=True
                    )
                ],
                "dependencies": [
                    {
                        "source_operation_index": dep.source_operation_index,
                        "target_operation_index": dep.target_operation_index,
                    }
                    for dep in step_b.dependencies
                ],
                "semantic_change_needed": _UNRESOLVED_NEEDED.get(stage_a_id, ""),
                "human_approved_applied": False,
                "h4_anchor_contract": H4_ANCHOR_CONTRACT,
                "source_corpus": SOURCE_CORPUS_V2,
            }
        )
    return rows


def validate_train_dev_v3(
    step_a_records: Sequence[StageAStepAAnnotation],
    step_b_records: Sequence[StageAStepBAnnotation],
    *,
    train_ids: frozenset[str],
    dev_ids: frozenset[str],
    selection_path: str | Path = STAGE_A_V2_SELECTION_PATH,
) -> list[str]:
    """Validate TRAIN+DEV annotations only (TEST never evaluated)."""
    errors: list[str] = []
    keep = train_ids | dev_ids

    if len(train_ids) != STAGE_A_V3_TRAIN_SIZE:
        errors.append(f"train id count {len(train_ids)} != {STAGE_A_V3_TRAIN_SIZE}")
    if len(dev_ids) != STAGE_A_V3_DEV_SIZE:
        errors.append(f"dev id count {len(dev_ids)} != {STAGE_A_V3_DEV_SIZE}")
    if len(step_a_records) != len(keep) or len(step_b_records) != len(keep):
        errors.append(
            f"materialized annotation counts "
            f"A={len(step_a_records)} B={len(step_b_records)} "
            f"!= TRAIN+DEV {len(keep)}"
        )

    by_a = {r.stage_a_id: r for r in step_a_records}
    by_b = {r.stage_a_id: r for r in step_b_records}
    if set(by_a) != keep or set(by_b) != keep:
        errors.append("materialized ids are not exactly TRAIN∪DEV")
    if sorted(by_a) != sorted(by_b):
        errors.append("TRAIN/DEV Step-A / Step-B id sets differ")

    # Selection rows: parse only TRAIN/DEV ids (never TEST selection rows).
    selection_rows = load_annotation_rows_for_ids(
        selection_path,
        keep,
        model_cls=_SelectionRow,
    )

    for stage_a_id in sorted(keep):
        step_a = by_a.get(stage_a_id)
        step_b = by_b.get(stage_a_id)
        if step_a is None or step_b is None:
            errors.append(f"{stage_a_id}: missing TRAIN/DEV Step-A or Step-B")
            continue
        frozen = selection_rows[stage_a_id]
        if step_a.query != frozen.query:
            errors.append(f"{stage_a_id}: query differs from selection")
        try:
            StageAStepAAnnotation.model_validate(step_a.model_dump(mode="python"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{stage_a_id}: Step-A invalid: {exc}")
        errors.extend(validate_step_b_against_step_a(step_b, step_a))
        try:
            validate_step_ab_linkage(step_a, step_b)
            step_ab_to_planner_example(step_a, step_b)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{stage_a_id}: graph/linkage invalid: {exc}")

    for stage_a_id in sorted(AFFECTED_TRAIN_IDS | AFFECTED_DEV_IDS):
        if stage_a_id not in keep:
            errors.append(f"affected id {stage_a_id} not in TRAIN/DEV")
            continue
        step_a = by_a[stage_a_id]
        for anchor in step_a.anchors:
            slice_text = step_a.query[anchor.char_start:anchor.char_end]
            if slice_text != anchor.text:
                errors.append(
                    f"{stage_a_id}: offset/text mismatch "
                    f"{anchor.text!r} vs {slice_text!r}"
                )

    return errors


class _SelectionRow(BaseModel):
    """Minimal selection fields needed for TRAIN/DEV query checks."""

    stage_a_id: str
    query: str


def build_stage_a_v3_h4_refexpr_v1(
    *,
    root: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Build v3 TRAIN+DEV annotations from v2; SAFE_MECHANICAL only."""
    root_path = Path(root) if root is not None else Path.cwd()

    v2_step_a_path = root_path / STAGE_A_V2_STEP_A_PATH
    v2_step_b_path = root_path / STAGE_A_V2_STEP_B_PATH
    v2_split_path = root_path / STAGE_A_V2_SPLIT_PATH

    v2_a_before = fingerprint_file(v2_step_a_path)
    v2_b_before = fingerprint_file(v2_step_b_path)
    v2_split_bytes = v2_split_path.read_bytes()
    # Byte fingerprint of frozen v2 sources (no JSON parse of annotation rows).
    v2_corpus_fp = annotation_corpus_fingerprint(v2_step_a_path, v2_step_b_path)

    train_ids, dev_ids = load_train_dev_split_ids(v2_split_path)
    keep = train_ids | dev_ids
    if len(train_ids) != STAGE_A_V3_TRAIN_SIZE:
        raise RuntimeError(f"expected {STAGE_A_V3_TRAIN_SIZE} train ids")
    if len(dev_ids) != STAGE_A_V3_DEV_SIZE:
        raise RuntimeError(f"expected {STAGE_A_V3_DEV_SIZE} dev ids")
    if len(keep) != STAGE_A_V3_MATERIALIZED_SIZE:
        raise RuntimeError(
            f"expected {STAGE_A_V3_MATERIALIZED_SIZE} TRAIN+DEV ids, got {len(keep)}"
        )
    if train_ids & SAFE_MECHANICAL_DEV_IDS:
        raise RuntimeError("SAFE_MECHANICAL DEV id found in TRAIN")
    if dev_ids & SAFE_MECHANICAL_TRAIN_IDS:
        raise RuntimeError("SAFE_MECHANICAL TRAIN id found in DEV")
    if not SAFE_MECHANICAL_TRAIN_IDS <= train_ids:
        raise RuntimeError("SAFE_MECHANICAL TRAIN ids missing from TRAIN split")
    if not SAFE_MECHANICAL_DEV_IDS <= dev_ids:
        raise RuntimeError("SAFE_MECHANICAL DEV ids missing from DEV split")
    if not HUMAN_REVIEW_TRAIN_IDS <= train_ids:
        raise RuntimeError("HUMAN_REVIEW TRAIN ids missing from TRAIN split")
    if not HUMAN_REVIEW_DEV_IDS <= dev_ids:
        raise RuntimeError("HUMAN_REVIEW DEV ids missing from DEV split")

    # Parse TRAIN+DEV annotation rows only — never TEST.
    by_a = load_annotation_rows_for_ids(
        v2_step_a_path, keep, model_cls=StageAStepAAnnotation
    )
    by_b = load_annotation_rows_for_ids(
        v2_step_b_path, keep, model_cls=StageAStepBAnnotation
    )
    n_step_a_parsed = len(by_a)
    n_step_b_parsed = len(by_b)
    if n_step_a_parsed != STAGE_A_V3_MATERIALIZED_SIZE:
        raise RuntimeError(
            f"parsed {n_step_a_parsed} Step-A rows, expected "
            f"{STAGE_A_V3_MATERIALIZED_SIZE}"
        )
    if n_step_b_parsed != STAGE_A_V3_MATERIALIZED_SIZE:
        raise RuntimeError(
            f"parsed {n_step_b_parsed} Step-B rows, expected "
            f"{STAGE_A_V3_MATERIALIZED_SIZE}"
        )

    # Snapshot gold before edits for unchanged-set checks.
    unchanged_ids = UNRESOLVED_HUMAN_REVIEW_IDS | APPROVED_R4_KEEP
    unchanged_a_before = {
        sid: by_a[sid].model_dump(mode="json") for sid in unchanged_ids
    }
    unchanged_b_before = {
        sid: by_b[sid].model_dump(mode="json") for sid in unchanged_ids
    }

    # Queue snapshot uses pre-edit labels (includes approved + unresolved).
    review_flags = build_human_review_queue(
        by_a, by_b, train_ids=train_ids, dev_ids=dev_ids
    )
    if len(review_flags) != len(HUMAN_REVIEW_IDS):
        raise RuntimeError(
            f"expected {len(HUMAN_REVIEW_IDS)} review flags, got {len(review_flags)}"
        )

    mechanical_changes: list[dict[str, Any]] = []
    approved_r6_changes: list[dict[str, Any]] = []
    approved_r5_changes: list[dict[str, Any]] = []
    approved_final_changes: list[dict[str, Any]] = []
    anchors_shortened_train = 0
    anchors_shortened_dev = 0
    anchors_removed = 0
    anchors_expanded = 0
    approved_r6_shortened = 0
    approved_r5_removed = 0
    approved_final_replaced = 0
    approved_final_anchors_added_net = 0

    for stage_a_id in sorted(SAFE_MECHANICAL_IDS):
        new_a, new_b, changes = apply_safe_mechanical_to_pair(
            by_a[stage_a_id], by_b[stage_a_id]
        )
        by_a[stage_a_id] = new_a
        by_b[stage_a_id] = new_b
        mechanical_changes.extend(changes)
        if stage_a_id in SAFE_MECHANICAL_TRAIN_IDS:
            anchors_shortened_train += len(changes)
        else:
            anchors_shortened_dev += len(changes)

    for stage_a_id in sorted(APPROVED_R6_SHORTEN):
        old_text, new_text = APPROVED_R6_SHORTEN[stage_a_id]
        new_a, new_b, change = apply_approved_r6_shorten(
            by_a[stage_a_id],
            by_b[stage_a_id],
            old_text=old_text,
            new_text=new_text,
        )
        by_a[stage_a_id] = new_a
        by_b[stage_a_id] = new_b
        approved_r6_changes.append(change)
        approved_r6_shortened += 1

    for stage_a_id in sorted(APPROVED_R5_REMOVE):
        new_a, new_b, change = apply_approved_r5_remove(
            by_a[stage_a_id],
            by_b[stage_a_id],
            remove_text=APPROVED_R5_REMOVE[stage_a_id],
        )
        by_a[stage_a_id] = new_a
        by_b[stage_a_id] = new_b
        approved_r5_changes.append(change)
        approved_r5_removed += 1
        anchors_removed += 1

    for stage_a_id in sorted(APPROVED_FINAL_REPLACE):
        spec = APPROVED_FINAL_REPLACE[stage_a_id]
        new_a, new_b, change = apply_approved_final_replace(
            by_a[stage_a_id],
            by_b[stage_a_id],
            old_text=str(spec["old_text"]),
            new_anchor_specs=spec["new_anchors"],  # type: ignore[arg-type]
        )
        by_a[stage_a_id] = new_a
        by_b[stage_a_id] = new_b
        approved_final_changes.append(change)
        approved_final_replaced += 1
        net = int(change["n_anchors_after"]) - int(change["n_anchors_before"])
        approved_final_anchors_added_net += net
        if net > 0:
            anchors_expanded += net

    for stage_a_id in unchanged_ids:
        if by_a[stage_a_id].model_dump(mode="json") != unchanged_a_before[stage_a_id]:
            raise RuntimeError(f"unchanged HUMAN_REVIEW Step-A {stage_a_id} was modified")
        if by_b[stage_a_id].model_dump(mode="json") != unchanged_b_before[stage_a_id]:
            raise RuntimeError(f"unchanged HUMAN_REVIEW Step-B {stage_a_id} was modified")

    unresolved_rows = build_unresolved_review_report(
        by_a, by_b, train_ids=train_ids, dev_ids=dev_ids
    )

    out_a = tuple(by_a[sid] for sid in sorted(by_a))
    out_b = tuple(by_b[sid] for sid in sorted(by_b))
    if len(out_a) != STAGE_A_V3_MATERIALIZED_SIZE:
        raise RuntimeError("v3 Step-A materialized size drifted")
    if len(out_b) != STAGE_A_V3_MATERIALIZED_SIZE:
        raise RuntimeError("v3 Step-B materialized size drifted")

    validation_errors = validate_train_dev_v3(
        out_a,
        out_b,
        train_ids=train_ids,
        dev_ids=dev_ids,
        selection_path=root_path / STAGE_A_V2_SELECTION_PATH,
    )
    if validation_errors:
        raise RuntimeError(
            "v3 TRAIN/DEV validation failed:\n" + "\n".join(validation_errors[:40])
        )

    v3_step_a_path = root_path / STAGE_A_V3_STEP_A_PATH
    v3_step_b_path = root_path / STAGE_A_V3_STEP_B_PATH
    v3_split_path = root_path / STAGE_A_V3_SPLIT_PATH
    report_path = root_path / STAGE_A_V3_BUILD_REPORT_PATH
    queue_path = root_path / STAGE_A_V3_HUMAN_REVIEW_QUEUE_PATH
    unresolved_path = root_path / STAGE_A_V3_UNRESOLVED_REVIEW_PATH

    n_test_annotations_materialized = 0

    if write:
        write_step_a_annotations(v3_step_a_path, out_a)
        write_step_b_annotations(v3_step_b_path, out_b)
        v3_split_path.parent.mkdir(parents=True, exist_ok=True)
        # Frozen split membership (includes TEST assignment rows as membership
        # only). Annotation files still omit TEST content.
        v3_split_path.write_bytes(v2_split_bytes)

        def _decision_status(stage_a_id: str) -> str:
            if stage_a_id in APPROVED_R6_SHORTEN:
                return "applied_r6_shorten"
            if stage_a_id in APPROVED_R5_REMOVE:
                return "applied_r5_remove"
            if stage_a_id in APPROVED_FINAL_REPLACE:
                return "applied_final_replace"
            if stage_a_id in APPROVED_R4_KEEP:
                return "kept_r4"
            if stage_a_id in UNRESOLVED_HUMAN_REVIEW_IDS:
                return "unresolved"
            return "unknown"

        queue_lines = [
            json.dumps(
                {
                    "stage_a_id": flag.stage_a_id,
                    "split": flag.split,
                    "query": flag.query,
                    "anchor_index": flag.anchor_index,
                    "existing_anchor": flag.existing_anchor,
                    "char_start": flag.char_start,
                    "char_end": flag.char_end,
                    "proposed_flagged_issue": flag.flagged_issue,
                    "rule": flag.rule,
                    "h5": flag.h5,
                    "h6_owner": flag.h6_owner,
                    "reason_human_review_required": flag.reason_human_review_required,
                    "decision_status": _decision_status(flag.stage_a_id),
                    # True only when an explicit recorded human decision was applied.
                    "human_approved_applied": flag.stage_a_id in APPROVED_EDIT_IDS,
                    "h4_anchor_contract": H4_ANCHOR_CONTRACT,
                    "source_corpus": SOURCE_CORPUS_V2,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for flag in review_flags
        ]
        queue_path.write_text(
            "\n".join(queue_lines) + ("\n" if queue_lines else ""),
            encoding="utf-8",
        )
        unresolved_lines = [
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in unresolved_rows
        ]
        unresolved_path.write_text(
            "\n".join(unresolved_lines) + ("\n" if unresolved_lines else ""),
            encoding="utf-8",
        )

    v2_a_after = fingerprint_file(v2_step_a_path)
    v2_b_after = fingerprint_file(v2_step_b_path)
    if v2_a_after != v2_a_before or v2_b_after != v2_b_before:
        raise RuntimeError("Stage-A v2 files mutated during v3 build")

    if write:
        v3_corpus_fp = annotation_corpus_fingerprint(v3_step_a_path, v3_step_b_path)
        v3_a_fp = fingerprint_file(v3_step_a_path)
        v3_b_fp = fingerprint_file(v3_step_b_path)
        written_a = len(out_a)
        written_b = len(out_b)
    else:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ta = tmp_path / "a.jsonl"
            tb = tmp_path / "b.jsonl"
            write_step_a_annotations(ta, out_a)
            write_step_b_annotations(tb, out_b)
            v3_corpus_fp = annotation_corpus_fingerprint(ta, tb)
            v3_a_fp = fingerprint_file(ta)
            v3_b_fp = fingerprint_file(tb)
            written_a = len(out_a)
            written_b = len(out_b)

    if v3_corpus_fp == v2_corpus_fp:
        raise RuntimeError("v3 annotation fingerprint unexpectedly equals v2")
    if written_a != STAGE_A_V3_MATERIALIZED_SIZE or written_b != STAGE_A_V3_MATERIALIZED_SIZE:
        raise RuntimeError("v3 write counts must be TRAIN+DEV only")

    report: dict[str, Any] = {
        "h4_anchor_contract": H4_ANCHOR_CONTRACT,
        "source_corpus": SOURCE_CORPUS_V2,
        "train_dev_h4_status": STAGE_A_V3_TRAIN_DEV_H4_STATUS,
        "test_annotation_migration": STAGE_A_V3_TEST_ANNOTATION_MIGRATION,
        "n_train": len(train_ids),
        "n_dev": len(dev_ids),
        "n_test_annotations_materialized": n_test_annotations_materialized,
        "n_step_a_rows_parsed": n_step_a_parsed,
        "n_step_b_rows_parsed": n_step_b_parsed,
        "n_step_a_rows_written": written_a,
        "n_step_b_rows_written": written_b,
        "v2_annotation_fingerprint": v2_corpus_fp,
        "v3_annotation_fingerprint": v3_corpus_fp,
        "v2_step_a_fingerprint": {"nbytes": v2_a_before[0], "sha256": v2_a_before[1]},
        "v2_step_b_fingerprint": {"nbytes": v2_b_before[0], "sha256": v2_b_before[1]},
        "v3_step_a_fingerprint": {"nbytes": v3_a_fp[0], "sha256": v3_a_fp[1]},
        "v3_step_b_fingerprint": {"nbytes": v3_b_fp[0], "sha256": v3_b_fp[1]},
        "train_count": len(train_ids),
        "dev_count": len(dev_ids),
        "safe_mechanical": {
            "train_examples": sorted(SAFE_MECHANICAL_TRAIN_IDS),
            "dev_examples": sorted(SAFE_MECHANICAL_DEV_IDS),
            "anchors_shortened_train": anchors_shortened_train,
            "anchors_shortened_dev": anchors_shortened_dev,
            "anchors_removed": 0,
            "anchors_expanded": anchors_expanded,
            "changes": mechanical_changes,
        },
        "approved_human_review": {
            "r6_shortened": approved_r6_shortened,
            "r5_removed": approved_r5_removed,
            "final_replaced": approved_final_replaced,
            "final_net_anchor_delta": approved_final_anchors_added_net,
            "r4_kept": sorted(APPROVED_R4_KEEP),
            "r6_changes": approved_r6_changes,
            "r5_changes": approved_r5_changes,
            "final_changes": approved_final_changes,
        },
        "human_review": {
            "train_examples": sorted(HUMAN_REVIEW_TRAIN_IDS),
            "dev_examples": sorted(HUMAN_REVIEW_DEV_IDS),
            "queue_count": len(review_flags),
            "unresolved_ids": sorted(UNRESOLVED_HUMAN_REVIEW_IDS),
            "unresolved_count": len(unresolved_rows),
            # Detector queues candidates; only explicit human decisions apply.
            "human_approved_edits_only": True,
        },
        "totals": {
            "anchors_shortened_safe_mechanical": (
                anchors_shortened_train + anchors_shortened_dev
            ),
            "anchors_shortened_approved_r6": approved_r6_shortened,
            "anchors_removed_approved_r5": approved_r5_removed,
            "anchors_final_replaced_examples": approved_final_replaced,
            "anchors_final_net_delta": approved_final_anchors_added_net,
            "anchors_expanded": anchors_expanded,
            "anchors_removed_total": anchors_removed,
        },
        "affected_revalidated": {
            "train": sorted(AFFECTED_TRAIN_IDS),
            "dev": sorted(AFFECTED_DEV_IDS),
        },
        "v2_unchanged": True,
        "paths": {
            "step_a": str(STAGE_A_V3_STEP_A_PATH).replace("\\", "/"),
            "step_b": str(STAGE_A_V3_STEP_B_PATH).replace("\\", "/"),
            "split": str(STAGE_A_V3_SPLIT_PATH).replace("\\", "/"),
            "human_review_queue": str(STAGE_A_V3_HUMAN_REVIEW_QUEUE_PATH).replace(
                "\\", "/"
            ),
            "unresolved_review": str(STAGE_A_V3_UNRESOLVED_REVIEW_PATH).replace(
                "\\", "/"
            ),
            "build_report": str(STAGE_A_V3_BUILD_REPORT_PATH).replace("\\", "/"),
        },
    }

    if write:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        from tiergraph.planner.stage_a_v3_spec import STAGE_A_V3_ANNOTATION_FINGERPRINT

        if v3_corpus_fp != STAGE_A_V3_ANNOTATION_FINGERPRINT:
            raise RuntimeError(
                "frozen TRAIN/DEV fingerprint drift: "
                f"got {v3_corpus_fp}, expected {STAGE_A_V3_ANNOTATION_FINGERPRINT}"
            )
        # Idempotent: refresh constant only if it somehow drifted to placeholder.
        _update_spec_fingerprint(root_path, v3_corpus_fp)

    return report


def _update_spec_fingerprint(root: Path, fingerprint: str) -> None:
    import re

    spec_path = root / "tiergraph" / "planner" / "stage_a_v3_spec.py"
    text = spec_path.read_text(encoding="utf-8")
    replacement = (
        "STAGE_A_V3_ANNOTATION_FINGERPRINT: Final[str] = (\n"
        f'    "{fingerprint}"\n'
        ")"
    )
    text2, n = re.subn(
        r"STAGE_A_V3_ANNOTATION_FINGERPRINT: Final\[str\] = \(\n"
        r'    "(?:PLACEHOLDER_SET_BY_BUILD|[0-9a-fA-F]+)"\n'
        r"\)",
        replacement,
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("failed to update STAGE_A_V3_ANNOTATION_FINGERPRINT")
    spec_path.write_text(text2, encoding="utf-8")


__all__ = [
    "annotation_corpus_fingerprint",
    "apply_approved_final_replace",
    "apply_approved_r5_remove",
    "apply_approved_r6_shorten",
    "apply_safe_mechanical_to_pair",
    "build_human_review_queue",
    "build_stage_a_v3_h4_refexpr_v1",
    "build_unresolved_review_report",
    "load_annotation_rows_for_ids",
    "load_train_dev_split_ids",
    "validate_train_dev_v3",
]
