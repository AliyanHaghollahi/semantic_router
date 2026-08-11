"""Phase-5 planner corpus helpers: candidates and semantic annotations.

Humans annotate only semantic decisions. Slot names, tiers, transfer policies,
node IDs, mandatory implicit edges, FUSE wiring, and task strings are produced
by :class:`~tiergraph.planner.decode.GraphDecoder`.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from tiergraph.enums import OperatorType, _CanonicalWireEnum
from tiergraph.models import TierGraphSchema
from tiergraph.planner.annotations import (
    ImplicitResolution,
    OperationSpanLabel,
    PlannerExample,
    SlotAnchorLabel,
    _OPERATOR_SEMANTICS,
)
from tiergraph.planner.decode import (
    GraphDecoder,
    PlannerDecodeError,
    PlannerPredictions,
    PredictedAnchor,
    PredictedOperation,
)
from tiergraph.planner.operator_io import is_h7_pair_eligible


SOURCE_CLASSIFICATION_LABELS: tuple[str, ...] = (
    "Personal",
    "Environmental",
    "Mixed",
)
DEFAULT_CANDIDATE_SEED = 20260811
DEFAULT_PERSONAL_CANDIDATES = 30
DEFAULT_ENVIRONMENTAL_CANDIDATES = 30


class PlannerBucket(_CanonicalWireEnum, str, Enum):
    """Stage-A structural bucket assigned during human semantic annotation."""

    PERSONAL = "personal"
    ENVIRONMENTAL = "environmental"
    MIXED_IMPLICIT = "mixed_implicit"
    MIXED_EXPLICIT_PARALLEL = "mixed_explicit_parallel"
    MIXED_SEQUENTIAL = "mixed_sequential"


class AnnotationStatus(_CanonicalWireEnum, str, Enum):
    UNREVIEWED = "unreviewed"
    IN_PROGRESS = "in_progress"
    ANNOTATED = "annotated"
    REJECTED = "rejected"


class _CorpusSchema(TierGraphSchema):
    """Strict corpus models; updates revalidate."""

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


class SemanticOperationSpan(_CorpusSchema):
    """One explicit answer operation annotated by character offsets."""

    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    operator_type: OperatorType

    @model_validator(mode="after")
    def _validate_span(self) -> "SemanticOperationSpan":
        if self.char_end <= self.char_start:
            raise ValueError("operation span must be nonempty")
        if self.operator_type is OperatorType.FUSE:
            raise ValueError("operation spans cannot use FUSE")
        return self


class SemanticAnchorSpan(_CorpusSchema):
    """One owned slot anchor annotated by character offsets."""

    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    normalized_name: str
    owner_operation_index: int = Field(ge=0)
    implicit_resolution: ImplicitResolution

    @field_validator("normalized_name")
    @classmethod
    def _nonblank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("normalized_name must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_span(self) -> "SemanticAnchorSpan":
        if self.char_end <= self.char_start:
            raise ValueError("anchor span must be nonempty")
        return self


class SemanticDependency(_CorpusSchema):
    """One explicit learned dependency between answer operations (H7)."""

    source_operation_index: int = Field(ge=0)
    target_operation_index: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_pair(self) -> "SemanticDependency":
        if self.source_operation_index == self.target_operation_index:
            raise ValueError("dependency self-loop is invalid")
        return self


class PlannerSemanticAnnotation(_CorpusSchema):
    """Human semantic annotation prior to GraphDecoder materialization."""

    source_query_id: str
    semantic_group_id: str
    query: str
    source_classification_label: str
    planner_bucket: PlannerBucket
    operations: tuple[SemanticOperationSpan, ...]
    anchors: tuple[SemanticAnchorSpan, ...] = ()
    dependencies: tuple[SemanticDependency, ...] = ()
    template_id: str | None = None
    paraphrase_id: str | None = None
    split: str | None = None

    @field_validator("source_query_id", "semantic_group_id", "query")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("source_classification_label")
    @classmethod
    def _classification_label(cls, value: str) -> str:
        if value not in SOURCE_CLASSIFICATION_LABELS:
            raise ValueError(
                "source_classification_label must be one of "
                f"{SOURCE_CLASSIFICATION_LABELS}"
            )
        return value

    @field_validator("operations", "anchors", "dependencies", mode="before")
    @classmethod
    def _tupleize(cls, value: object) -> object:
        if type(value) is list:
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_indices_and_spans(self) -> "PlannerSemanticAnnotation":
        if not self.operations:
            raise ValueError("operations must not be empty")
        n_ops = len(self.operations)
        query_len = len(self.query)

        for index, operation in enumerate(self.operations):
            if operation.char_end > query_len:
                raise ValueError(
                    f"operation span out of bounds at index {index}"
                )

        for index, anchor in enumerate(self.anchors):
            if anchor.char_end > query_len:
                raise ValueError(f"anchor span out of bounds at index {index}")
            if anchor.owner_operation_index >= n_ops:
                raise ValueError(
                    f"anchor owner_operation_index out of range at index {index}"
                )

        seen_deps: set[tuple[int, int]] = set()
        for index, dependency in enumerate(self.dependencies):
            if dependency.source_operation_index >= n_ops:
                raise ValueError(
                    f"dependency source_operation_index out of range at {index}"
                )
            if dependency.target_operation_index >= n_ops:
                raise ValueError(
                    f"dependency target_operation_index out of range at {index}"
                )
            pair = (
                dependency.source_operation_index,
                dependency.target_operation_index,
            )
            if pair in seen_deps:
                raise ValueError(f"duplicate dependency pair {pair}")
            seen_deps.add(pair)
            source_op = self.operations[dependency.source_operation_index].operator_type
            target_op = self.operations[dependency.target_operation_index].operator_type
            if not is_h7_pair_eligible(source_op, target_op):
                raise ValueError(
                    "structurally impossible H7 pair under "
                    f"OPERATOR_IO_CONTRACT_V1: {source_op.value} -> {target_op.value}"
                )
        return self


class StageACandidate(_CorpusSchema):
    """Unannotated Stage-A review candidate (no planner labels)."""

    source_query_id: str
    semantic_group_id: str
    query: str
    source_classification_label: str
    annotation_status: AnnotationStatus = AnnotationStatus.UNREVIEWED
    planner_bucket: PlannerBucket | None = None
    split: str | None = None

    @field_validator("source_query_id", "semantic_group_id", "query")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("source_classification_label")
    @classmethod
    def _classification_label(cls, value: str) -> str:
        if value not in SOURCE_CLASSIFICATION_LABELS:
            raise ValueError(
                "source_classification_label must be one of "
                f"{SOURCE_CLASSIFICATION_LABELS}"
            )
        return value


def normalize_query_key(query: str) -> str:
    """Normalize only for duplicate detection: casefold + whitespace collapse."""
    if type(query) is not str:
        raise TypeError("query must be a string")
    return " ".join(query.casefold().split())


def load_classification_rows(path: str | Path) -> tuple[dict[str, str], ...]:
    """Load ``dataset/training_data.json`` rows as ``{query, label}`` mappings."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("classification dataset must be a JSON array")
    rows: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"row {index} must be an object")
        if "query" not in item or "label" not in item:
            raise ValueError(f"row {index} must contain query and label")
        query = item["query"]
        label = item["label"]
        if type(query) is not str or type(label) is not str:
            raise ValueError(f"row {index} query/label must be strings")
        if label not in SOURCE_CLASSIFICATION_LABELS:
            raise ValueError(f"row {index} has unknown label {label!r}")
        rows.append({"query": query, "label": label})
    return tuple(rows)


def build_unique_query_pool(
    rows: Sequence[Mapping[str, str]],
) -> tuple[StageACandidate, ...]:
    """Deduplicate to unique original queries with stable ``src_####`` IDs.

    Duplicate detection uses :func:`normalize_query_key`. Conflicting labels
    for the same normalized key raise ``ValueError``. The first-seen original
    query text (file order) is preserved. IDs follow sorted normalized keys.
    """
    groups: dict[str, list[tuple[int, str, str]]] = {}
    for index, row in enumerate(rows):
        query = row["query"]
        label = row["label"]
        key = normalize_query_key(query)
        if not key:
            raise ValueError(f"blank normalized query at row {index}")
        groups.setdefault(key, []).append((index, query, label))

    uniques: list[StageACandidate] = []
    for ordinal, key in enumerate(sorted(groups.keys()), start=1):
        members = groups[key]
        labels = {label for _index, _query, label in members}
        if len(labels) != 1:
            raise ValueError(
                "conflicting classification labels for normalized query "
                f"{key!r}: {sorted(labels)}"
            )
        members_sorted = sorted(members, key=lambda item: item[0])
        _index, original_query, label = members_sorted[0]
        source_id = f"src_{ordinal:04d}"
        uniques.append(
            StageACandidate(
                source_query_id=source_id,
                semantic_group_id=source_id,
                query=original_query,
                source_classification_label=label,
                annotation_status=AnnotationStatus.UNREVIEWED,
                planner_bucket=None,
                split=None,
            )
        )
    return tuple(uniques)


def select_stage_a_candidates(
    unique_pool: Sequence[StageACandidate],
    *,
    seed: int = DEFAULT_CANDIDATE_SEED,
    n_personal: int = DEFAULT_PERSONAL_CANDIDATES,
    n_environmental: int = DEFAULT_ENVIRONMENTAL_CANDIDATES,
) -> tuple[StageACandidate, ...]:
    """Select the Stage-A review pool (not final labeled distribution).

    Takes ``n_personal`` Personal, ``n_environmental`` Environmental, and
    **all** Mixed unique queries. Selection uses a fixed RNG seed over
    ID-sorted pools.
    """
    by_label: dict[str, list[StageACandidate]] = {
        label: [] for label in SOURCE_CLASSIFICATION_LABELS
    }
    for item in unique_pool:
        by_label[item.source_classification_label].append(item)
    for items in by_label.values():
        items.sort(key=lambda candidate: candidate.source_query_id)

    if len(by_label["Personal"]) < n_personal:
        raise ValueError(
            f"need {n_personal} Personal uniques, found {len(by_label['Personal'])}"
        )
    if len(by_label["Environmental"]) < n_environmental:
        raise ValueError(
            f"need {n_environmental} Environmental uniques, "
            f"found {len(by_label['Environmental'])}"
        )

    rng = random.Random(seed)
    personal = sorted(
        rng.sample(by_label["Personal"], n_personal),
        key=lambda candidate: candidate.source_query_id,
    )
    environmental = sorted(
        rng.sample(by_label["Environmental"], n_environmental),
        key=lambda candidate: candidate.source_query_id,
    )
    mixed = list(by_label["Mixed"])
    selected = personal + environmental + mixed
    selected.sort(key=lambda candidate: candidate.source_query_id)
    return tuple(selected)


def write_candidates_jsonl(
    path: str | Path,
    candidates: Sequence[StageACandidate],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False)
        for candidate in candidates
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_candidates_jsonl(path: str | Path) -> tuple[StageACandidate, ...]:
    path = Path(path)
    records: list[StageACandidate] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(StageACandidate.model_validate(json.loads(line)))
        except Exception as exc:  # noqa: BLE001 - surface line context
            raise ValueError(
                f"invalid candidate at {path}:{line_number}: {exc}"
            ) from exc
    return tuple(records)


def semantic_annotation_to_predictions(
    annotation: PlannerSemanticAnnotation,
) -> PlannerPredictions:
    """Map human semantic fields to :class:`PlannerPredictions`."""
    operations = tuple(
        PredictedOperation(
            start=operation.char_start,
            end=operation.char_end,
            operator=operation.operator_type,
        )
        for operation in annotation.operations
    )
    anchors = tuple(
        PredictedAnchor(
            start=anchor.char_start,
            end=anchor.char_end,
            text=annotation.query[anchor.char_start : anchor.char_end],
            owner_index=anchor.owner_operation_index,
            implicit_resolution=anchor.implicit_resolution,
            normalized_name=anchor.normalized_name,
        )
        for anchor in annotation.anchors
    )
    dependency_pairs = frozenset(
        (dependency.source_operation_index, dependency.target_operation_index)
        for dependency in annotation.dependencies
    )
    return PlannerPredictions(
        operations=operations,
        anchors=anchors,
        dependency_pairs=dependency_pairs,
        aux_query_type=None,
    )


def semantic_annotation_to_planner_example(
    annotation: PlannerSemanticAnnotation,
    *,
    decoder: GraphDecoder | None = None,
    example_id: str | None = None,
) -> PlannerExample:
    """Convert semantic annotation → GraphDecoder → validated PlannerExample.

    Raises :class:`PlannerDecodeError` with the ``source_query_id`` when
    decoding fails.
    """
    decoder = decoder or GraphDecoder()
    example_id = example_id or annotation.source_query_id
    predictions = semantic_annotation_to_predictions(annotation)
    try:
        decoded = decoder.decode(
            predictions,
            query=annotation.query,
            graph_id=example_id,
        )
    except PlannerDecodeError as exc:
        raise PlannerDecodeError(
            f"GraphDecoder rejected annotation {annotation.source_query_id}: {exc}"
        ) from exc

    operation_labels: list[OperationSpanLabel] = []
    for index, operation in enumerate(annotation.operations):
        node_id = f"op_{index + 1}"
        operation_labels.append(
            OperationSpanLabel(
                node_id=node_id,
                semantic_type=_OPERATOR_SEMANTICS[operation.operator_type],
                start=operation.char_start,
                end=operation.char_end,
                operator=operation.operator_type,
            )
        )

    anchor_labels: list[SlotAnchorLabel] = []
    implicit_counter = 0
    for index, anchor in enumerate(annotation.anchors):
        owner_node_id = f"op_{anchor.owner_operation_index + 1}"
        implicit_node_id: str | None = None
        if anchor.implicit_resolution is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL:
            implicit_counter += 1
            implicit_node_id = f"impl_{implicit_counter}"
        anchor_labels.append(
            SlotAnchorLabel(
                anchor_id=f"a{index + 1}",
                start=anchor.char_start,
                end=anchor.char_end,
                text=annotation.query[anchor.char_start : anchor.char_end],
                normalized_name=anchor.normalized_name,
                owner_node_id=owner_node_id,
                implicit_resolution=anchor.implicit_resolution,
                implicit_node_id=implicit_node_id,
            )
        )

    metadata: dict[str, Any] = {
        "source_query_id": annotation.source_query_id,
        "semantic_group_id": annotation.semantic_group_id,
        "source_classification_label": annotation.source_classification_label,
        "planner_bucket": annotation.planner_bucket.value,
    }
    if annotation.template_id is not None:
        metadata["template_id"] = annotation.template_id
    if annotation.paraphrase_id is not None:
        metadata["paraphrase_id"] = annotation.paraphrase_id
    if annotation.split is not None:
        metadata["split"] = annotation.split

    return PlannerExample.model_validate(
        {
            "example_id": example_id,
            "query": annotation.query,
            "graph": decoded.graph.model_dump(mode="json"),
            "fusion_plan": (
                None
                if decoded.fusion_plan is None
                else decoded.fusion_plan.model_dump(mode="json")
            ),
            "planner_labels": {
                "query_type": decoded.graph.query_type.value,
                "operation_spans": [
                    label.model_dump(mode="json") for label in operation_labels
                ],
                "slot_anchors": [
                    label.model_dump(mode="json") for label in anchor_labels
                ],
            },
            "metadata": metadata,
        }
    )


def load_semantic_annotations_jsonl(
    path: str | Path,
) -> tuple[PlannerSemanticAnnotation, ...]:
    path = Path(path)
    records: list[PlannerSemanticAnnotation] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(
                PlannerSemanticAnnotation.model_validate(json.loads(line))
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"invalid semantic annotation at {path}:{line_number}: {exc}"
            ) from exc
    return tuple(records)


__all__ = [
    "AnnotationStatus",
    "DEFAULT_CANDIDATE_SEED",
    "DEFAULT_ENVIRONMENTAL_CANDIDATES",
    "DEFAULT_PERSONAL_CANDIDATES",
    "PlannerBucket",
    "PlannerSemanticAnnotation",
    "SOURCE_CLASSIFICATION_LABELS",
    "SemanticAnchorSpan",
    "SemanticDependency",
    "SemanticOperationSpan",
    "StageACandidate",
    "build_unique_query_pool",
    "load_candidates_jsonl",
    "load_classification_rows",
    "load_semantic_annotations_jsonl",
    "normalize_query_key",
    "select_stage_a_candidates",
    "semantic_annotation_to_planner_example",
    "semantic_annotation_to_predictions",
    "write_candidates_jsonl",
]
