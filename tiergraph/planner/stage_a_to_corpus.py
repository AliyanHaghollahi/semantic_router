"""Convert frozen Stage-A Step-A + Step-B gold into PlannerSemanticAnnotation.

Joins COMPLETE Step-A / Step-B records and maps them onto the existing corpus
schema consumed by :func:`semantic_annotation_to_planner_example`. Graph
materialization (including synthetic ``RESOLVE_PERSONAL``) remains exclusively
in :class:`~tiergraph.planner.decode.GraphDecoder`.
"""

from __future__ import annotations

import re
from pathlib import Path

from tiergraph.enums import OperatorType
from tiergraph.planner.annotation_step_a import (
    DEFAULT_STEP_A_ANNOTATIONS_PATH,
    EXPECTED_STAGE_A_COUNT,
    StageAStepAAnnotation,
    StepAStatus,
    fingerprint_file,
    load_step_a_annotations,
)
from tiergraph.planner.annotation_step_b import (
    DEFAULT_STEP_B_ANNOTATIONS_PATH,
    StageAStepBAnnotation,
    StepBStatus,
    load_step_b_annotations,
)
from tiergraph.planner.annotations import PlannerExample
from tiergraph.planner.corpus import (
    PlannerBucket,
    PlannerSemanticAnnotation,
    SemanticAnchorSpan,
    SemanticDependency,
    SemanticOperationSpan,
    semantic_annotation_to_planner_example,
)
from tiergraph.planner.decode import GraphDecoder
from tiergraph.planner.naming import SlotNamingError, normalize_base_name


# Determiners / possessives commonly prefixed on Stage-A H4 phrases. Stripped
# only when raw ``normalize_base_name`` rejects the surface (apostrophe, etc.).
_LEADING_DET = re.compile(
    r"^(?:my|your|his|her|their|our|this|that|these|those|the|a|an)\s+",
    re.IGNORECASE,
)
_NON_V1_CHARS = re.compile(r"[^a-zA-Z0-9\s]+")

FINAL_BUCKET_TO_PLANNER_BUCKET: dict[str, PlannerBucket] = {
    "Personal": PlannerBucket.PERSONAL,
    "Environmental": PlannerBucket.ENVIRONMENTAL,
    "MIXED_IMPLICIT": PlannerBucket.MIXED_IMPLICIT,
    "MIXED_PARALLEL": PlannerBucket.MIXED_EXPLICIT_PARALLEL,
    "MIXED_SEQUENTIAL": PlannerBucket.MIXED_SEQUENTIAL,
}

FINAL_BUCKET_TO_CLASSIFICATION: dict[str, str] = {
    "Personal": "Personal",
    "Environmental": "Environmental",
    "MIXED_IMPLICIT": "Mixed",
    "MIXED_PARALLEL": "Mixed",
    "MIXED_SEQUENTIAL": "Mixed",
}


def final_bucket_to_planner_bucket(final_bucket: str) -> PlannerBucket:
    try:
        return FINAL_BUCKET_TO_PLANNER_BUCKET[final_bucket]
    except KeyError as exc:
        raise ValueError(f"unknown final_bucket: {final_bucket!r}") from exc


def final_bucket_to_classification_label(final_bucket: str) -> str:
    try:
        return FINAL_BUCKET_TO_CLASSIFICATION[final_bucket]
    except KeyError as exc:
        raise ValueError(f"unknown final_bucket: {final_bucket!r}") from exc


def derive_anchor_normalized_name(anchor_text: str) -> str:
    """Derive a SLOT_NAMING_V1 base from an H4 surface string.

    Prefers ``normalize_base_name`` on the literal text. When punctuation or
    leading determiners block V1, strips those minimally and retries. Does not
    invent corpus-specific synonyms.
    """
    if type(anchor_text) is not str or not anchor_text.strip():
        raise ValueError("anchor_text must be a nonblank string")
    try:
        return normalize_base_name(anchor_text)
    except SlotNamingError:
        stripped = _LEADING_DET.sub("", anchor_text).strip() or anchor_text
        cleaned = _NON_V1_CHARS.sub(" ", stripped)
        try:
            return normalize_base_name(cleaned)
        except SlotNamingError as exc:
            raise ValueError(
                f"cannot derive SLOT_NAMING_V1 normalized_name from "
                f"anchor text {anchor_text!r}"
            ) from exc


def validate_step_ab_linkage(
    step_a: StageAStepAAnnotation,
    step_b: StageAStepBAnnotation,
) -> None:
    """Raise ``ValueError`` when Step-A / Step-B records are not joinable."""
    if step_a.stage_a_id != step_b.stage_a_id:
        raise ValueError(
            f"stage_a_id mismatch: {step_a.stage_a_id!r} vs {step_b.stage_a_id!r}"
        )
    prefix = step_a.stage_a_id
    if step_a.step_a_status is not StepAStatus.COMPLETE:
        raise ValueError(f"{prefix}: Step A is not COMPLETE")
    if step_b.step_b_status is not StepBStatus.COMPLETE:
        raise ValueError(f"{prefix}: Step B is not COMPLETE")
    if step_a.query != step_b.query:
        raise ValueError(f"{prefix}: query mismatch between Step A and Step B")
    if step_a.source_id != step_b.source_id:
        raise ValueError(f"{prefix}: source_id mismatch between Step A and Step B")
    if step_a.candidate_id != step_b.candidate_id:
        raise ValueError(
            f"{prefix}: candidate_id mismatch between Step A and Step B"
        )
    if step_a.final_bucket != step_b.final_bucket:
        raise ValueError(
            f"{prefix}: final_bucket mismatch between Step A and Step B"
        )
    if step_b.n_operations != len(step_a.operations):
        raise ValueError(
            f"{prefix}: n_operations {step_b.n_operations} != "
            f"Step-A operation count {len(step_a.operations)}"
        )
    if step_b.n_anchors != len(step_a.anchors):
        raise ValueError(
            f"{prefix}: n_anchors {step_b.n_anchors} != "
            f"Step-A anchor count {len(step_a.anchors)}"
        )
    expected_types = tuple(op.operator_type.value for op in step_a.operations)
    if step_b.operation_types != expected_types:
        raise ValueError(
            f"{prefix}: operation_types drifted from Step-A H3 "
            f"{expected_types!r} vs {step_b.operation_types!r}"
        )
    if len(step_b.anchor_decisions) != len(step_a.anchors):
        raise ValueError(f"{prefix}: anchor_decisions count mismatch")
    for decision, anchor in zip(
        step_b.anchor_decisions, step_a.anchors, strict=True
    ):
        if decision.anchor_index != anchor.anchor_index:
            raise ValueError(
                f"{prefix}: anchor_index mismatch "
                f"{decision.anchor_index} vs {anchor.anchor_index}"
            )
        if decision.text is not None and decision.text != anchor.text:
            raise ValueError(
                f"{prefix}: anchor[{anchor.anchor_index}] text mismatch "
                f"{decision.text!r} vs {anchor.text!r}"
            )
        if decision.owner_operation_index is None:
            raise ValueError(
                f"{prefix}: COMPLETE Step B missing H6 owner for "
                f"anchor[{anchor.anchor_index}]"
            )


def step_ab_to_semantic_annotation(
    step_a: StageAStepAAnnotation,
    step_b: StageAStepBAnnotation,
) -> PlannerSemanticAnnotation:
    """Map one COMPLETE Step-A + Step-B pair to ``PlannerSemanticAnnotation``."""
    validate_step_ab_linkage(step_a, step_b)

    operations = tuple(
        SemanticOperationSpan(
            char_start=operation.char_start,
            char_end=operation.char_end,
            operator_type=operation.operator_type,
        )
        for operation in step_a.operations
    )
    anchors = tuple(
        SemanticAnchorSpan(
            char_start=anchor.char_start,
            char_end=anchor.char_end,
            normalized_name=derive_anchor_normalized_name(anchor.text),
            owner_operation_index=decision.owner_operation_index,
            implicit_resolution=decision.implicit_resolution,
        )
        for anchor, decision in zip(
            step_a.anchors, step_b.anchor_decisions, strict=True
        )
    )
    dependencies = tuple(
        SemanticDependency(
            source_operation_index=dependency.source_operation_index,
            target_operation_index=dependency.target_operation_index,
        )
        for dependency in step_b.dependencies
    )
    return PlannerSemanticAnnotation(
        source_query_id=step_a.stage_a_id,
        semantic_group_id=step_a.semantic_group,
        query=step_a.query,
        source_classification_label=final_bucket_to_classification_label(
            step_a.final_bucket
        ),
        planner_bucket=final_bucket_to_planner_bucket(step_a.final_bucket),
        operations=operations,
        anchors=anchors,
        dependencies=dependencies,
        template_id=step_a.template_group,
        paraphrase_id=None,
        split=None,
    )


def _enrich_planner_example_metadata(
    example: PlannerExample,
    step_a: StageAStepAAnnotation,
) -> PlannerExample:
    metadata = dict(example.metadata)
    metadata["stage_a_id"] = step_a.stage_a_id
    metadata["final_bucket"] = step_a.final_bucket
    metadata["semantic_group"] = step_a.semantic_group
    metadata["template_group"] = step_a.template_group
    metadata["source_kind"] = step_a.source_kind
    if step_a.source_id is not None:
        metadata["source_id"] = step_a.source_id
    if step_a.candidate_id is not None:
        metadata["candidate_id"] = step_a.candidate_id
    return example.model_copy(update={"metadata": metadata})


def step_ab_to_planner_example(
    step_a: StageAStepAAnnotation,
    step_b: StageAStepBAnnotation,
    *,
    decoder: GraphDecoder | None = None,
) -> PlannerExample:
    """Step-A+B → semantic annotation → GraphDecoder → ``PlannerExample``."""
    annotation = step_ab_to_semantic_annotation(step_a, step_b)
    example = semantic_annotation_to_planner_example(
        annotation,
        decoder=decoder,
        example_id=step_a.stage_a_id,
    )
    return _enrich_planner_example_metadata(example, step_a)


def load_stage_a_planner_examples(
    step_a_path: str | Path = DEFAULT_STEP_A_ANNOTATIONS_PATH,
    step_b_path: str | Path = DEFAULT_STEP_B_ANNOTATIONS_PATH,
    *,
    decoder: GraphDecoder | None = None,
) -> tuple[PlannerExample, ...]:
    """Load frozen Step-A+B gold and convert to validated ``PlannerExample``s."""
    step_a_path = Path(step_a_path)
    step_b_path = Path(step_b_path)
    step_a_before = fingerprint_file(step_a_path)
    step_b_before = fingerprint_file(step_b_path)

    step_a_records = load_step_a_annotations(step_a_path)
    step_b_records = load_step_b_annotations(step_b_path)
    if len(step_a_records) != EXPECTED_STAGE_A_COUNT:
        raise ValueError(
            f"expected {EXPECTED_STAGE_A_COUNT} Step-A records, "
            f"got {len(step_a_records)}"
        )
    if len(step_b_records) != EXPECTED_STAGE_A_COUNT:
        raise ValueError(
            f"expected {EXPECTED_STAGE_A_COUNT} Step-B records, "
            f"got {len(step_b_records)}"
        )

    by_b = {item.stage_a_id: item for item in step_b_records}
    if len(by_b) != len(step_b_records):
        raise ValueError("duplicate stage_a_id in Step B")
    if sorted(by_b) != sorted(item.stage_a_id for item in step_a_records):
        raise ValueError("Step-A / Step-B stage_a_id sets do not match")

    decoder = decoder or GraphDecoder()
    examples: list[PlannerExample] = []
    for step_a in sorted(step_a_records, key=lambda item: item.stage_a_id):
        examples.append(
            step_ab_to_planner_example(step_a, by_b[step_a.stage_a_id], decoder=decoder)
        )

    if fingerprint_file(step_a_path) != step_a_before:
        raise ValueError("Step-A annotation file mutated during conversion")
    if fingerprint_file(step_b_path) != step_b_before:
        raise ValueError("Step-B annotation file mutated during conversion")
    if len(examples) != EXPECTED_STAGE_A_COUNT:
        raise ValueError(
            f"expected {EXPECTED_STAGE_A_COUNT} PlannerExamples, got {len(examples)}"
        )
    return tuple(examples)


def count_explicit_h7_edges(example: PlannerExample) -> int:
    """Count explicit op→op edges (excludes impl_* and FUSE endpoints)."""
    answer_ops = {
        node.node_id
        for node in example.graph.nodes
        if node.operator is not OperatorType.FUSE
        and not node.node_id.startswith("impl_")
    }
    count = 0
    for edge in example.graph.edges:
        if edge.source_node_id in answer_ops and edge.target_node_id in answer_ops:
            count += 1
    return count


__all__ = [
    "FINAL_BUCKET_TO_CLASSIFICATION",
    "FINAL_BUCKET_TO_PLANNER_BUCKET",
    "count_explicit_h7_edges",
    "derive_anchor_normalized_name",
    "final_bucket_to_classification_label",
    "final_bucket_to_planner_bucket",
    "load_stage_a_planner_examples",
    "step_ab_to_planner_example",
    "step_ab_to_semantic_annotation",
    "validate_step_ab_linkage",
]
