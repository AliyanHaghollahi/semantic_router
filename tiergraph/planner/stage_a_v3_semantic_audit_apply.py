"""Apply approved Stage-A v3 semantic-audit corrections (sa_0249/sa_0258/sa_0344).

Corrections are the frozen READY_TO_APPLY decisions from the Stage-A v3
semantic audit. Does not touch Stage-A v2 paths or TEST annotations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tiergraph.enums import OperatorType, QueryType
from tiergraph.planner.annotation_step_a import (
    StageAStepAAnnotation,
    StepAAnchor,
    StepAOperation,
    load_step_a_annotations,
    write_step_a_annotations,
)
from tiergraph.planner.annotation_step_b import (
    StageAStepBAnnotation,
    StepBAnchorDecision,
    load_step_b_annotations,
    validate_step_b_against_step_a,
    write_step_b_annotations,
)
from tiergraph.planner.annotations import ImplicitResolution
from tiergraph.planner.stage_a_to_corpus import step_ab_to_planner_example
from tiergraph.planner.stage_a_v3_h4_build import (
    annotation_corpus_fingerprint,
    _update_spec_fingerprint,
)
from tiergraph.planner.stage_a_v3_spec import (
    STAGE_A_V3_BUILD_REPORT_PATH,
    STAGE_A_V3_MATERIALIZED_SIZE,
    STAGE_A_V3_STEP_A_PATH,
    STAGE_A_V3_STEP_B_PATH,
)

CORRECTED_IDS: frozenset[str] = frozenset({"sa_0249", "sa_0258", "sa_0344"})


def _apply_sa_0249(
    step_a: StageAStepAAnnotation, step_b: StageAStepBAnnotation
) -> tuple[StageAStepAAnnotation, StageAStepBAnnotation]:
    """SAFE_AUTO: RETRIEVE→DESCRIBE; H4 'the room safely'→'the room'."""
    assert step_a.query.startswith("Do I have enough space to walk through the room")
    op0 = step_a.operations[0]
    new_op = StepAOperation(
        operation_index=0,
        text=op0.text,
        char_start=op0.char_start,
        char_end=op0.char_end,
        operator_type=OperatorType.DESCRIBE_ENVIRONMENT,
    )
    new_anchor = StepAAnchor(
        anchor_index=0,
        text="the room",
        char_start=39,
        char_end=47,
    )
    assert step_a.query[39:47] == "the room"
    new_a = step_a.model_copy(
        update={
            "operations": (new_op,),
            "anchors": (new_anchor,),
            "derived_query_type": QueryType.ENVIRONMENTAL,
        }
    )
    new_b = step_b.model_copy(
        update={
            "operation_types": ("DESCRIBE_ENVIRONMENT",),
            "n_operations": 1,
            "n_anchors": 1,
            "anchor_decisions": (
                StepBAnchorDecision(
                    anchor_index=0,
                    implicit_resolution=ImplicitResolution.NONE,
                    owner_operation_index=0,
                    text="the room",
                ),
            ),
            "dependencies": (),
        }
    )
    return new_a, new_b


def _apply_sa_0258(
    step_a: StageAStepAAnnotation, step_b: StageAStepBAnnotation
) -> tuple[StageAStepAAnnotation, StageAStepBAnnotation]:
    """HUMAN_REVIEW final: H4 'the entrance' only; H5 NONE."""
    assert step_a.query == "What is my position relative to the entrance?"
    new_anchor = StepAAnchor(
        anchor_index=0,
        text="the entrance",
        char_start=32,
        char_end=44,
    )
    assert step_a.query[32:44] == "the entrance"
    new_a = step_a.model_copy(update={"anchors": (new_anchor,)})
    new_b = step_b.model_copy(
        update={
            "n_anchors": 1,
            "anchor_decisions": (
                StepBAnchorDecision(
                    anchor_index=0,
                    implicit_resolution=ImplicitResolution.NONE,
                    owner_operation_index=0,
                    text="the entrance",
                ),
            ),
            "dependencies": (),
        }
    )
    return new_a, new_b


def _apply_sa_0344(
    step_a: StageAStepAAnnotation, step_b: StageAStepBAnnotation
) -> tuple[StageAStepAAnnotation, StageAStepBAnnotation]:
    """HUMAN_REVIEW final: DESCRIBE+IDENTIFY; add 'what I ordered' IMPLICIT."""
    q = step_a.query
    assert q == "What is written on this receipt and does it match what I ordered?"
    ops = (
        StepAOperation(
            operation_index=0,
            text="What is written on this receipt",
            char_start=0,
            char_end=31,
            operator_type=OperatorType.DESCRIBE_ENVIRONMENT,
        ),
        StepAOperation(
            operation_index=1,
            text="does it match what I ordered",
            char_start=36,
            char_end=64,
            operator_type=OperatorType.IDENTIFY_ENVIRONMENTAL,
        ),
    )
    anchors = (
        StepAAnchor(
            anchor_index=0, text="this receipt", char_start=19, char_end=31
        ),
        StepAAnchor(
            anchor_index=1, text="what I ordered", char_start=50, char_end=64
        ),
    )
    assert q[19:31] == "this receipt"
    assert q[50:64] == "what I ordered"
    new_a = step_a.model_copy(
        update={
            "operations": ops,
            "anchors": anchors,
            "final_bucket": "MIXED_PARALLEL",
            "derived_query_type": QueryType.MIXED,
        }
    )
    new_b = step_b.model_copy(
        update={
            "final_bucket": "MIXED_PARALLEL",
            "n_operations": 2,
            "n_anchors": 2,
            "operation_types": (
                "DESCRIBE_ENVIRONMENT",
                "IDENTIFY_ENVIRONMENTAL",
            ),
            "anchor_decisions": (
                StepBAnchorDecision(
                    anchor_index=0,
                    implicit_resolution=ImplicitResolution.NONE,
                    owner_operation_index=0,
                    text="this receipt",
                ),
                StepBAnchorDecision(
                    anchor_index=1,
                    implicit_resolution=ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL,
                    owner_operation_index=1,
                    text="what I ordered",
                ),
            ),
            "dependencies": (),
        }
    )
    return new_a, new_b


_APPLIERS = {
    "sa_0249": _apply_sa_0249,
    "sa_0258": _apply_sa_0258,
    "sa_0344": _apply_sa_0344,
}


def apply_semantic_audit_corrections(
    *,
    root: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Apply the three approved corrections; optionally write + refresh fingerprint."""
    root = root or Path(__file__).resolve().parents[2]
    step_a_path = root / STAGE_A_V3_STEP_A_PATH
    step_b_path = root / STAGE_A_V3_STEP_B_PATH

    before_fp = annotation_corpus_fingerprint(step_a_path, step_b_path)
    records_a = list(load_step_a_annotations(step_a_path))
    records_b = list(load_step_b_annotations(step_b_path))
    if len(records_a) != STAGE_A_V3_MATERIALIZED_SIZE:
        raise RuntimeError(
            f"expected {STAGE_A_V3_MATERIALIZED_SIZE} step-a rows, got {len(records_a)}"
        )
    if len(records_b) != STAGE_A_V3_MATERIALIZED_SIZE:
        raise RuntimeError(
            f"expected {STAGE_A_V3_MATERIALIZED_SIZE} step-b rows, got {len(records_b)}"
        )

    by_a = {r.stage_a_id: r for r in records_a}
    by_b = {r.stage_a_id: r for r in records_b}
    missing = CORRECTED_IDS - set(by_a)
    if missing:
        raise RuntimeError(f"missing stage_a_ids in v3 corpus: {sorted(missing)}")

    applied: dict[str, dict[str, Any]] = {}
    pending_a: dict[str, StageAStepAAnnotation] = {}
    pending_b: dict[str, StageAStepBAnnotation] = {}
    for stage_a_id, applier in _APPLIERS.items():
        new_a, new_b = applier(by_a[stage_a_id], by_b[stage_a_id])
        # Real contract: validate_step_b_against_step_a(step_b, step_a).
        linkage_errors = validate_step_b_against_step_a(new_b, new_a)
        if linkage_errors:
            raise RuntimeError(
                f"{stage_a_id}: Step-A/B linkage validation failed: "
                + "; ".join(linkage_errors)
            )
        example = step_ab_to_planner_example(new_a, new_b)
        pending_a[stage_a_id] = new_a
        pending_b[stage_a_id] = new_b
        applied[stage_a_id] = {
            "query": new_a.query,
            "final_bucket": new_a.final_bucket,
            "operators": [op.operator_type.value for op in new_a.operations],
            "anchors": [
                {
                    "text": a.text,
                    "start": a.char_start,
                    "end": a.char_end,
                    "h5": d.implicit_resolution.value,
                    "owner": d.owner_operation_index,
                }
                for a, d in zip(new_a.anchors, new_b.anchor_decisions, strict=True)
            ],
            "n_h7": len(new_b.dependencies),
            "planner_query_type": example.planner_labels.query_type.value,
        }

    # Commit in-memory only after every correction validated.
    by_a.update(pending_a)
    by_b.update(pending_b)

    out_a = tuple(by_a[sid] for sid in sorted(by_a))
    out_b = tuple(by_b[sid] for sid in sorted(by_b))

    if write:
        write_step_a_annotations(step_a_path, out_a)
        write_step_b_annotations(step_b_path, out_b)
        after_fp = annotation_corpus_fingerprint(step_a_path, step_b_path)
        _update_spec_fingerprint(root, after_fp)
        _patch_build_report(root, before_fp, after_fp, applied)
    else:
        after_fp = before_fp

    return {
        "before_fingerprint": before_fp,
        "after_fingerprint": after_fp if write else None,
        "corrected_ids": sorted(CORRECTED_IDS),
        "applied": applied,
        "wrote": write,
    }


def _patch_build_report(
    root: Path,
    before_fp: str,
    after_fp: str,
    applied: dict[str, dict[str, Any]],
) -> None:
    report_path = root / STAGE_A_V3_BUILD_REPORT_PATH
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["v3_annotation_fingerprint"] = after_fp
    report["semantic_audit_corrections"] = {
        "source": "stage_a_v3_semantic_audit_sa_0249_0258_0344",
        "before_fingerprint": before_fp,
        "after_fingerprint": after_fp,
        "corrected_ids": sorted(CORRECTED_IDS),
        "applied": applied,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    report = apply_semantic_audit_corrections(write=True)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
