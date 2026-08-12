"""Authored Stage-A MIXED_SEQUENTIAL candidates and human review state.

These candidates are provenance-aware authored examples for human review.
They are never silently inserted into ``dataset/training_data.json``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from tiergraph.enums import OperatorType, SlotType, _CanonicalWireEnum
from tiergraph.models import TierGraphSchema
from tiergraph.planner.corpus import normalize_query_key
from tiergraph.planner.operator_io import (
    h7_dependency_slot_type,
    is_h7_pair_eligible,
)


DEFAULT_AUTHORED_SEQUENTIAL_CANDIDATES_PATH = Path(
    "dataset/planner/stage_a_authored_sequential_candidates.jsonl"
)
DEFAULT_AUTHORED_SEQUENTIAL_REVIEWS_PATH = Path(
    "dataset/planner/stage_a_authored_sequential_reviews.jsonl"
)
DEFAULT_TRAIN_PATH = Path("dataset/training_data.json")
EXPECTED_AUTHORED_SEQUENTIAL_COUNT = 20
SOURCE_KIND_AUTHORED = "authored_stage_a"
PLANNER_BUCKET_MIXED_SEQUENTIAL = "MIXED_SEQUENTIAL"

CHOICE_TO_STATUS: dict[str, "AuthoredReviewStatus"] = {}

_OPERATOR_NAMES = {op.value: op for op in OperatorType if op is not OperatorType.FUSE}
_SLOT_NAMES = {slot.value: slot for slot in SlotType}


class AuthoredReviewStatus(_CanonicalWireEnum, str, Enum):
    UNREVIEWED = "UNREVIEWED"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


CHOICE_TO_STATUS.update(
    {
        "1": AuthoredReviewStatus.ACCEPT,
        "2": AuthoredReviewStatus.REJECT,
    }
)


class _AuthoredSchema(TierGraphSchema):
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


def parse_dependency_edge(edge: str) -> tuple[OperatorType, OperatorType]:
    """Parse ``SOURCE -> TARGET`` edge notation into operator enums."""
    if type(edge) is not str or "->" not in edge:
        raise ValueError(f"dependency edge must look like 'OP_A -> OP_B', got {edge!r}")
    left, right = edge.split("->", 1)
    source_name = left.strip()
    target_name = right.strip()
    if source_name not in _OPERATOR_NAMES or target_name not in _OPERATOR_NAMES:
        raise ValueError(
            f"dependency edge uses unsupported operators: {edge!r}"
        )
    source = _OPERATOR_NAMES[source_name]
    target = _OPERATOR_NAMES[target_name]
    if source is OperatorType.FUSE or target is OperatorType.FUSE:
        raise ValueError(f"FUSE cannot appear in dependency edges: {edge!r}")
    if not is_h7_pair_eligible(source, target):
        raise ValueError(
            "dependency edge is not allowed under OPERATOR_IO_CONTRACT_V1: "
            f"{edge!r}"
        )
    return source, target


def validate_dependency_spec(
    *,
    operations: Sequence[str],
    edges: Sequence[str],
    typed_values: Sequence[str],
) -> None:
    """Validate operations/edges/typed values against OPERATOR_IO_CONTRACT_V1."""
    if not operations:
        raise ValueError("intended_operations must be non-empty")
    if not edges:
        raise ValueError(
            "MIXED_SEQUENTIAL requires at least one non-fusion dependency edge"
        )
    if len(edges) != len(typed_values):
        raise ValueError(
            "intended_dependency_edges and intended_typed_values must align"
        )
    for op_name in operations:
        if op_name == OperatorType.FUSE.value:
            raise ValueError("intended_operations must not include FUSE")
        if op_name not in _OPERATOR_NAMES:
            raise ValueError(f"unsupported intended operation: {op_name!r}")
    for edge, typed in zip(edges, typed_values, strict=True):
        source, target = parse_dependency_edge(edge)
        expected = h7_dependency_slot_type(source, target)
        if typed not in _SLOT_NAMES:
            raise ValueError(f"unsupported typed value: {typed!r}")
        if _SLOT_NAMES[typed] is not expected:
            raise ValueError(
                f"typed value {typed!r} does not match V1 transfer for {edge!r} "
                f"(expected {expected.value})"
            )
        if source.value not in operations or target.value not in operations:
            raise ValueError(
                f"dependency edge {edge!r} references an operation missing from "
                "intended_operations"
            )


class AuthoredSequentialCandidate(_AuthoredSchema):
    """One authored MIXED_SEQUENTIAL candidate awaiting human review."""

    candidate_id: str
    query: str
    source_kind: str
    planner_bucket: str
    template_group: str
    semantic_group: str
    authoring_reason: str
    intended_personal_requirement: str
    intended_environmental_requirement: str
    personal_necessity_reason: str
    environmental_necessity_reason: str
    intended_operations: tuple[str, ...] = Field(min_length=1)
    intended_dependency_edges: tuple[str, ...] = Field(min_length=1)
    intended_typed_values: tuple[str, ...] = Field(min_length=1)
    dependency_family: str
    review_status: AuthoredReviewStatus = AuthoredReviewStatus.UNREVIEWED

    @field_validator(
        "candidate_id",
        "query",
        "source_kind",
        "planner_bucket",
        "template_group",
        "semantic_group",
        "authoring_reason",
        "intended_personal_requirement",
        "intended_environmental_requirement",
        "personal_necessity_reason",
        "environmental_necessity_reason",
        "dependency_family",
    )
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if type(value) is not str or not value.strip():
            raise ValueError("must be a nonblank string")
        return value

    @field_validator(
        "intended_operations",
        "intended_dependency_edges",
        "intended_typed_values",
        mode="before",
    )
    @classmethod
    def _tupleize(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_authored_contract(self) -> "AuthoredSequentialCandidate":
        if self.source_kind != SOURCE_KIND_AUTHORED:
            raise ValueError(
                f"source_kind must be {SOURCE_KIND_AUTHORED!r}, "
                f"got {self.source_kind!r}"
            )
        if self.planner_bucket != PLANNER_BUCKET_MIXED_SEQUENTIAL:
            raise ValueError(
                f"planner_bucket must be {PLANNER_BUCKET_MIXED_SEQUENTIAL!r}, "
                f"got {self.planner_bucket!r}"
            )
        if not self.candidate_id.startswith("auth_seq_"):
            raise ValueError(
                "candidate_id must start with 'auth_seq_' "
                f"(got {self.candidate_id!r})"
            )
        validate_dependency_spec(
            operations=self.intended_operations,
            edges=self.intended_dependency_edges,
            typed_values=self.intended_typed_values,
        )
        return self


class AuthoredSequentialReview(_AuthoredSchema):
    """Persisted human decision for one authored sequential candidate."""

    candidate_id: str
    query: str
    source_kind: str
    planner_bucket: str
    template_group: str
    semantic_group: str
    authoring_reason: str
    intended_personal_requirement: str
    intended_environmental_requirement: str
    personal_necessity_reason: str
    environmental_necessity_reason: str
    intended_operations: tuple[str, ...] = Field(min_length=1)
    intended_dependency_edges: tuple[str, ...] = Field(min_length=1)
    intended_typed_values: tuple[str, ...] = Field(min_length=1)
    dependency_family: str
    review_status: AuthoredReviewStatus

    @field_validator(
        "candidate_id",
        "query",
        "source_kind",
        "planner_bucket",
        "template_group",
        "semantic_group",
        "authoring_reason",
        "intended_personal_requirement",
        "intended_environmental_requirement",
        "personal_necessity_reason",
        "environmental_necessity_reason",
        "dependency_family",
    )
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if type(value) is not str or not value.strip():
            raise ValueError("must be a nonblank string")
        return value

    @field_validator(
        "intended_operations",
        "intended_dependency_edges",
        "intended_typed_values",
        mode="before",
    )
    @classmethod
    def _tupleize(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def _validate_decision(self) -> "AuthoredSequentialReview":
        if self.review_status is AuthoredReviewStatus.UNREVIEWED:
            raise ValueError("persisted reviews cannot use UNREVIEWED")
        if self.source_kind != SOURCE_KIND_AUTHORED:
            raise ValueError(f"source_kind must be {SOURCE_KIND_AUTHORED!r}")
        if self.planner_bucket != PLANNER_BUCKET_MIXED_SEQUENTIAL:
            raise ValueError(
                f"planner_bucket must be {PLANNER_BUCKET_MIXED_SEQUENTIAL!r}"
            )
        validate_dependency_spec(
            operations=self.intended_operations,
            edges=self.intended_dependency_edges,
            typed_values=self.intended_typed_values,
        )
        return self


def load_authored_sequential_candidates(
    path: str | Path,
) -> tuple[AuthoredSequentialCandidate, ...]:
    path = Path(path)
    records: list[AuthoredSequentialCandidate] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(
                AuthoredSequentialCandidate.model_validate(json.loads(line))
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"invalid authored sequential candidate at {path}:{line_number}: "
                f"{exc}"
            ) from exc
    return tuple(records)


def write_authored_sequential_candidates(
    path: str | Path,
    candidates: Sequence[AuthoredSequentialCandidate],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(candidates, key=lambda item: item.candidate_id)
    lines = [
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
        for item in ordered
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_authored_sequential_reviews(
    path: str | Path,
) -> tuple[AuthoredSequentialReview, ...]:
    path = Path(path)
    if not path.is_file():
        return ()
    records: list[AuthoredSequentialReview] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = AuthoredSequentialReview.model_validate(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"invalid authored sequential review at {path}:{line_number}: "
                f"{exc}"
            ) from exc
        if record.candidate_id in seen:
            raise ValueError(
                f"duplicate candidate_id in reviews: {record.candidate_id}"
            )
        seen.add(record.candidate_id)
        records.append(record)
    return tuple(records)


def write_authored_sequential_reviews(
    path: str | Path,
    reviews: Sequence[AuthoredSequentialReview],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(reviews, key=lambda item: item.candidate_id)
    seen: set[str] = set()
    for record in ordered:
        if record.candidate_id in seen:
            raise ValueError(
                f"duplicate candidate_id rejected: {record.candidate_id}"
            )
        seen.add(record.candidate_id)
    lines = [
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
        for item in ordered
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def validate_authored_sequential_candidate_set(
    candidates: Sequence[AuthoredSequentialCandidate],
    *,
    train_path: str | Path | None = DEFAULT_TRAIN_PATH,
    expected_count: int = EXPECTED_AUTHORED_SEQUENTIAL_COUNT,
) -> None:
    """Validate identity, provenance, V1 edges, and non-injection into train."""
    if len(candidates) != expected_count:
        raise ValueError(
            f"expected exactly {expected_count} authored sequential candidates, "
            f"found {len(candidates)}"
        )
    ids = [item.candidate_id for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate_id in authored sequential set")
    keys = [normalize_query_key(item.query) for item in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate normalized query in authored sequential set")
    for item in candidates:
        if item.source_kind != SOURCE_KIND_AUTHORED:
            raise ValueError(f"bad source_kind on {item.candidate_id}")
        if item.planner_bucket != PLANNER_BUCKET_MIXED_SEQUENTIAL:
            raise ValueError(f"bad planner_bucket on {item.candidate_id}")
        if item.review_status not in AuthoredReviewStatus:
            raise ValueError(f"bad review_status on {item.candidate_id}")
        if not item.intended_dependency_edges:
            raise ValueError(
                f"fusion-only/no-edge candidate rejected: {item.candidate_id}"
            )
        validate_dependency_spec(
            operations=item.intended_operations,
            edges=item.intended_dependency_edges,
            typed_values=item.intended_typed_values,
        )
        if not item.intended_personal_requirement.strip():
            raise ValueError(f"missing personal requirement: {item.candidate_id}")
        if not item.intended_environmental_requirement.strip():
            raise ValueError(
                f"missing environmental requirement: {item.candidate_id}"
            )
        if not item.personal_necessity_reason.strip():
            raise ValueError(
                f"missing personal_necessity_reason: {item.candidate_id}"
            )
        if not item.environmental_necessity_reason.strip():
            raise ValueError(
                f"missing environmental_necessity_reason: {item.candidate_id}"
            )

    if train_path is not None and Path(train_path).is_file():
        train_queries = {
            normalize_query_key(row["query"])
            for row in json.loads(Path(train_path).read_text(encoding="utf-8"))
        }
        overlap = sorted(
            item.candidate_id
            for item in candidates
            if normalize_query_key(item.query) in train_queries
        )
        if overlap:
            raise ValueError(
                "authored sequential candidates must not appear in "
                "training_data.json: " + ", ".join(overlap)
            )


def summarize_authored_sequential_reviews(
    candidates: Sequence[AuthoredSequentialCandidate],
    reviews: Sequence[AuthoredSequentialReview],
) -> dict[str, Any]:
    by_status = Counter(item.review_status.value for item in reviews)
    reviewed_ids = {item.candidate_id for item in reviews}
    unknown = reviewed_ids - {item.candidate_id for item in candidates}
    if unknown:
        raise ValueError(
            "reviews reference unknown authored sequential candidates: "
            + ", ".join(sorted(unknown))
        )
    family_counts = Counter(item.dependency_family for item in candidates)
    return {
        "total": len(candidates),
        "reviewed": len(reviewed_ids),
        "remaining": len(candidates) - len(reviewed_ids),
        "ACCEPT": by_status.get(AuthoredReviewStatus.ACCEPT.value, 0),
        "REJECT": by_status.get(AuthoredReviewStatus.REJECT.value, 0),
        "UNREVIEWED": len(candidates) - len(reviewed_ids),
        "dependency_family_counts": dict(sorted(family_counts.items())),
    }


def format_authored_sequential_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        f"Reviewed: {summary['reviewed']} / {summary['total']}",
        f"ACCEPT: {summary['ACCEPT']}",
        f"REJECT: {summary['REJECT']}",
        f"Remaining: {summary['remaining']}",
        "dependency_family_counts:",
    ]
    for family, count in summary.get("dependency_family_counts", {}).items():
        lines.append(f"  {family}: {count}")
    return "\n".join(lines)


def parse_authored_sequential_review_command(
    raw: str,
) -> tuple[str, AuthoredReviewStatus | None]:
    token = raw.strip().lower()
    if token in CHOICE_TO_STATUS:
        return "assign", CHOICE_TO_STATUS[token]
    if token in {"s", "skip"}:
        return "skip", None
    if token in {"b", "back"}:
        return "back", None
    if token in {"p", "progress", "summary"}:
        return "summary", None
    if token in {"q", "quit"}:
        return "quit", None
    raise ValueError(f"unknown command: {raw!r}")


class AuthoredSequentialReviewSession:
    """Resumable ACCEPT/REJECT review over authored sequential candidates."""

    def __init__(
        self,
        candidates: Sequence[AuthoredSequentialCandidate],
        *,
        reviews_path: str | Path,
        existing_reviews: Sequence[AuthoredSequentialReview] | None = None,
        train_path: str | Path | None = DEFAULT_TRAIN_PATH,
    ) -> None:
        validate_authored_sequential_candidate_set(
            candidates, train_path=train_path
        )
        ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
        self.candidates = ordered
        self.reviews_path = Path(reviews_path)
        self._by_id = {item.candidate_id: item for item in ordered}
        self._reviews: dict[str, AuthoredSequentialReview] = {}
        if existing_reviews is None:
            existing_reviews = load_authored_sequential_reviews(self.reviews_path)
        for record in existing_reviews:
            candidate = self._by_id.get(record.candidate_id)
            if candidate is None:
                raise ValueError(
                    "review references unknown authored sequential candidate: "
                    f"{record.candidate_id}"
                )
            if record.candidate_id in self._reviews:
                raise ValueError(
                    f"duplicate candidate_id rejected: {record.candidate_id}"
                )
            if record.query != candidate.query:
                raise ValueError(
                    "review query must match candidate exactly: "
                    f"{record.candidate_id}"
                )
            self._reviews[record.candidate_id] = record
        self._history: list[str] = []
        self._cursor = self._first_unreviewed_index()

    @property
    def reviews(self) -> tuple[AuthoredSequentialReview, ...]:
        return tuple(
            self._reviews[item.candidate_id]
            for item in self.candidates
            if item.candidate_id in self._reviews
        )

    def _first_unreviewed_index(self) -> int:
        for index, item in enumerate(self.candidates):
            if item.candidate_id not in self._reviews:
                return index
        return len(self.candidates)

    def current(self) -> AuthoredSequentialCandidate | None:
        if self._cursor >= len(self.candidates):
            return None
        return self.candidates[self._cursor]

    def current_position(self) -> tuple[int, int]:
        if self._cursor >= len(self.candidates):
            return len(self.candidates), len(self.candidates)
        return self._cursor + 1, len(self.candidates)

    def summary(self) -> dict[str, Any]:
        return summarize_authored_sequential_reviews(self.candidates, self.reviews)

    def save(self) -> None:
        write_authored_sequential_reviews(self.reviews_path, self.reviews)

    def apply_status(
        self,
        status: AuthoredReviewStatus,
        *,
        persist: bool = True,
    ) -> AuthoredSequentialReview:
        if status is AuthoredReviewStatus.UNREVIEWED:
            raise ValueError("invalid review value: UNREVIEWED")
        candidate = self.current()
        if candidate is None:
            raise ValueError("no remaining authored sequential candidates to review")
        record = AuthoredSequentialReview(
            candidate_id=candidate.candidate_id,
            query=candidate.query,
            source_kind=candidate.source_kind,
            planner_bucket=candidate.planner_bucket,
            template_group=candidate.template_group,
            semantic_group=candidate.semantic_group,
            authoring_reason=candidate.authoring_reason,
            intended_personal_requirement=candidate.intended_personal_requirement,
            intended_environmental_requirement=(
                candidate.intended_environmental_requirement
            ),
            personal_necessity_reason=candidate.personal_necessity_reason,
            environmental_necessity_reason=(
                candidate.environmental_necessity_reason
            ),
            intended_operations=candidate.intended_operations,
            intended_dependency_edges=candidate.intended_dependency_edges,
            intended_typed_values=candidate.intended_typed_values,
            dependency_family=candidate.dependency_family,
            review_status=status,
        )
        self._reviews[candidate.candidate_id] = record
        self._history.append(candidate.candidate_id)
        self._cursor = self._first_unreviewed_index()
        if persist:
            self.save()
        return record

    def apply_choice(
        self,
        choice: str,
        *,
        persist: bool = True,
    ) -> AuthoredSequentialReview:
        key = choice.strip().lower()
        if key not in CHOICE_TO_STATUS:
            raise ValueError(f"invalid review value: {choice!r}")
        return self.apply_status(CHOICE_TO_STATUS[key], persist=persist)

    def skip(self) -> AuthoredSequentialCandidate | None:
        if self._cursor >= len(self.candidates):
            return None
        self._cursor += 1
        while (
            self._cursor < len(self.candidates)
            and self.candidates[self._cursor].candidate_id in self._reviews
        ):
            self._cursor += 1
        return self.current()

    def back(self, *, persist: bool = True) -> AuthoredSequentialCandidate | None:
        if not self._history:
            if self._cursor <= 0:
                return self.current()
            self._cursor = max(0, self._cursor - 1)
            return self.current()
        candidate_id = self._history.pop()
        self._reviews.pop(candidate_id, None)
        for index, item in enumerate(self.candidates):
            if item.candidate_id == candidate_id:
                self._cursor = index
                break
        if persist:
            self.save()
        return self.current()


def _cand(
    *,
    candidate_id: str,
    query: str,
    template_group: str,
    semantic_group: str,
    authoring_reason: str,
    personal: str,
    environmental: str,
    personal_necessity_reason: str,
    environmental_necessity_reason: str,
    operations: Sequence[str],
    edges: Sequence[str],
    typed_values: Sequence[str],
    dependency_family: str,
) -> AuthoredSequentialCandidate:
    return AuthoredSequentialCandidate(
        candidate_id=candidate_id,
        query=query,
        source_kind=SOURCE_KIND_AUTHORED,
        planner_bucket=PLANNER_BUCKET_MIXED_SEQUENTIAL,
        template_group=template_group,
        semantic_group=semantic_group,
        authoring_reason=authoring_reason,
        intended_personal_requirement=personal,
        intended_environmental_requirement=environmental,
        personal_necessity_reason=personal_necessity_reason,
        environmental_necessity_reason=environmental_necessity_reason,
        intended_operations=tuple(operations),
        intended_dependency_edges=tuple(edges),
        intended_typed_values=tuple(typed_values),
        dependency_family=dependency_family,
        review_status=AuthoredReviewStatus.UNREVIEWED,
    )


# Canonical authored Stage-A MIXED_SEQUENTIAL candidate set (human-authored).
# All queries require genuine Personal AND Environmental necessity plus >=1 V1 edge.
AUTHORED_SEQUENTIAL_CANDIDATES: tuple[AuthoredSequentialCandidate, ...] = (
    _cand(
        candidate_id="auth_seq_001",
        query="Which of these bottles is my prescription?",
        template_group="resolve_identify_prescription_bottle",
        semantic_group="medication_identity_scene",
        authoring_reason=(
            "IDENTIFY among bottles requires the resolved prescription reference."
        ),
        personal="prescribed medication identity",
        environmental="visible bottles to identify among",
        personal_necessity_reason=(
            "Without the prescription identity, no bottle can be selected as 'mine'."
        ),
        environmental_necessity_reason=(
            "The answer is which observed bottle matches; scene identification is required."
        ),
        operations=("RESOLVE_PERSONAL", "IDENTIFY_ENVIRONMENTAL"),
        edges=("RESOLVE_PERSONAL -> IDENTIFY_ENVIRONMENTAL",),
        typed_values=("RESOLVED_REFERENCE",),
        dependency_family="resolve_to_identify",
    ),
    _cand(
        candidate_id="auth_seq_002",
        query="Which suitcase on this carousel is the one registered under my name?",
        template_group="resolve_identify_luggage_carousel",
        semantic_group="luggage_identity_scene",
        authoring_reason=(
            "IDENTIFY among carousel suitcases requires the resolved luggage/name reference."
        ),
        personal="luggage registration / passenger name",
        environmental="suitcases on the carousel",
        personal_necessity_reason=(
            "The matching criterion is the personal registration/name; without it, "
            "no suitcase can be identified as the user's."
        ),
        environmental_necessity_reason=(
            "The answer names/selects an observed suitcase on the carousel."
        ),
        operations=("RESOLVE_PERSONAL", "IDENTIFY_ENVIRONMENTAL"),
        edges=("RESOLVE_PERSONAL -> IDENTIFY_ENVIRONMENTAL",),
        typed_values=("RESOLVED_REFERENCE",),
        dependency_family="resolve_to_identify",
    ),
    _cand(
        candidate_id="auth_seq_003",
        query="Which dish on this table is the one from my order?",
        template_group="resolve_identify_ordered_dish",
        semantic_group="food_order_identity_scene",
        authoring_reason=(
            "IDENTIFY among dishes requires resolving the user's order contents."
        ),
        personal="food order contents",
        environmental="dishes on the table",
        personal_necessity_reason=(
            "Order contents define which dish is correct; personal data is the match key."
        ),
        environmental_necessity_reason=(
            "The answer selects among dishes currently on the table."
        ),
        operations=("RESOLVE_PERSONAL", "IDENTIFY_ENVIRONMENTAL"),
        edges=("RESOLVE_PERSONAL -> IDENTIFY_ENVIRONMENTAL",),
        typed_values=("RESOLVED_REFERENCE",),
        dependency_family="resolve_to_identify",
    ),
    _cand(
        candidate_id="auth_seq_004",
        query="Which of these seats is the one on my reservation?",
        template_group="resolve_identify_reserved_seat",
        semantic_group="reservation_seat_identity_scene",
        authoring_reason=(
            "IDENTIFY among seats requires the resolved reservation seat reference."
        ),
        personal="seat reservation record",
        environmental="visible candidate seats",
        personal_necessity_reason=(
            "Reservation seat data is required to know which seat is correct."
        ),
        environmental_necessity_reason=(
            "The answer identifies one of the seats present in the current scene."
        ),
        operations=("RESOLVE_PERSONAL", "IDENTIFY_ENVIRONMENTAL"),
        edges=("RESOLVE_PERSONAL -> IDENTIFY_ENVIRONMENTAL",),
        typed_values=("RESOLVED_REFERENCE",),
        dependency_family="resolve_to_identify",
    ),
    _cand(
        candidate_id="auth_seq_005",
        query=(
            "What landmark is shown on this plaque, and where is that landmark "
            "relative to my hotel?"
        ),
        template_group="identify_locate_landmark_vs_hotel",
        semantic_group="landmark_locate_personal_hotel",
        authoring_reason=(
            "Landmark identity comes from the plaque; locating that landmark and "
            "the personal hotel are both required for the relative-location answer."
        ),
        personal="user hotel identity/location",
        environmental="plaque text and landmark location in the environment",
        personal_necessity_reason=(
            "Relative position to 'my hotel' requires resolving/locating the hotel; "
            "the answer is undefined without personal hotel data."
        ),
        environmental_necessity_reason=(
            "Plaque identification is required for 'what landmark', and environmental "
            "locate of that landmark is required for the relative geometry."
        ),
        operations=(
            "IDENTIFY_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL",
            "RESOLVE_PERSONAL",
        ),
        edges=(
            "IDENTIFY_ENVIRONMENTAL -> LOCATE_ENVIRONMENTAL",
            "RESOLVE_PERSONAL -> LOCATE_ENVIRONMENTAL",
        ),
        typed_values=("ENVIRONMENTAL_FACT", "RESOLVED_REFERENCE"),
        dependency_family="identify_to_locate",
    ),
    _cand(
        candidate_id="auth_seq_006",
        query="Which counter here is assigned to my airline, and where is that counter?",
        template_group="resolve_identify_locate_airline_counter",
        semantic_group="airline_counter_identify_locate",
        authoring_reason=(
            "Personal airline/flight assignment resolves which airline; IDENTIFY "
            "selects the matching counter; LOCATE then consumes that environmental fact."
        ),
        personal="airline / flight assignment",
        environmental="visible check-in/service counters",
        personal_necessity_reason=(
            "Which counter is correct depends on the user's airline assignment; "
            "without it, counter identity is unknown."
        ),
        environmental_necessity_reason=(
            "Must identify the matching counter in the scene and locate that counter."
        ),
        operations=(
            "RESOLVE_PERSONAL",
            "IDENTIFY_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL",
        ),
        edges=(
            "RESOLVE_PERSONAL -> IDENTIFY_ENVIRONMENTAL",
            "IDENTIFY_ENVIRONMENTAL -> LOCATE_ENVIRONMENTAL",
        ),
        typed_values=("RESOLVED_REFERENCE", "ENVIRONMENTAL_FACT"),
        dependency_family="resolve_identify_locate",
    ),
    _cand(
        candidate_id="auth_seq_007",
        query="Which door here is the entrance listed for my appointment?",
        template_group="resolve_identify_appointment_entrance",
        semantic_group="appointment_entrance_identify",
        authoring_reason=(
            "Appointment data resolves the expected entrance; IDENTIFY selects the "
            "matching door in the current scene."
        ),
        personal="appointment entrance / clinic destination",
        environmental="visible doors/entrances",
        personal_necessity_reason=(
            "The listed entrance comes from the appointment record; without it, "
            "no door can be chosen as the appointment entrance."
        ),
        environmental_necessity_reason=(
            "The answer identifies which observed door matches that entrance."
        ),
        operations=("RESOLVE_PERSONAL", "IDENTIFY_ENVIRONMENTAL"),
        edges=("RESOLVE_PERSONAL -> IDENTIFY_ENVIRONMENTAL",),
        typed_values=("RESOLVED_REFERENCE",),
        dependency_family="resolve_to_identify",
    ),
    _cand(
        candidate_id="auth_seq_008",
        query="Where is the room listed for my appointment?",
        template_group="resolve_locate_appointment_room",
        semantic_group="appointment_room_locate",
        authoring_reason=(
            "Appointment room must be resolved from personal data before it can be "
            "located in the environment."
        ),
        personal="appointment room assignment",
        environmental="building/clinic interior to locate the room",
        personal_necessity_reason=(
            "The target room identity is stored in the appointment; locate cannot "
            "know which room without resolving it."
        ),
        environmental_necessity_reason=(
            "The answer is an environmental location of that room in the building."
        ),
        operations=("RESOLVE_PERSONAL", "LOCATE_ENVIRONMENTAL"),
        edges=("RESOLVE_PERSONAL -> LOCATE_ENVIRONMENTAL",),
        typed_values=("RESOLVED_REFERENCE",),
        dependency_family="resolve_to_locate",
    ),
    _cand(
        candidate_id="auth_seq_009",
        query="Where is the clinic listed in my appointment and how do I get there?",
        template_group="resolve_locate_navigate_appointment_clinic",
        semantic_group="appointment_clinic_wayfinding",
        authoring_reason=(
            "Clinic identity comes from the appointment; locate then navigate."
        ),
        personal="appointment clinic assignment",
        environmental="clinic location and route from here",
        personal_necessity_reason=(
            "Which clinic to find is determined by the appointment record."
        ),
        environmental_necessity_reason=(
            "Must locate that clinic in the environment and produce navigation."
        ),
        operations=(
            "RESOLVE_PERSONAL",
            "LOCATE_ENVIRONMENTAL",
            "NAVIGATE_TO",
        ),
        edges=(
            "RESOLVE_PERSONAL -> LOCATE_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL -> NAVIGATE_TO",
        ),
        typed_values=("RESOLVED_REFERENCE", "LOCATION"),
        dependency_family="resolve_locate_navigate",
    ),
    _cand(
        candidate_id="auth_seq_010",
        query="Where is the pickup point assigned to my order and how do I get there?",
        template_group="resolve_locate_navigate_order_pickup",
        semantic_group="order_pickup_wayfinding",
        authoring_reason=(
            "Order assignment resolves which pickup point; locate then navigate."
        ),
        personal="order-assigned pickup point",
        environmental="pickup point location and route",
        personal_necessity_reason=(
            "The pickup point identity is assigned by the order; without resolving "
            "it, the destination is unknown."
        ),
        environmental_necessity_reason=(
            "Must locate that pickup point and navigate to its LOCATION."
        ),
        operations=(
            "RESOLVE_PERSONAL",
            "LOCATE_ENVIRONMENTAL",
            "NAVIGATE_TO",
        ),
        edges=(
            "RESOLVE_PERSONAL -> LOCATE_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL -> NAVIGATE_TO",
        ),
        typed_values=("RESOLVED_REFERENCE", "LOCATION"),
        dependency_family="resolve_locate_navigate",
    ),
    _cand(
        candidate_id="auth_seq_011",
        query=(
            "Which baggage belt is listed for my flight on this arrivals board, "
            "and where is that belt?"
        ),
        template_group="resolve_identify_locate_baggage_belt",
        semantic_group="flight_baggage_belt_identify_locate",
        authoring_reason=(
            "Flight identity resolves which listing to use; IDENTIFY reads the "
            "matching belt from the board; LOCATE finds that belt in the hall."
        ),
        personal="flight identity / booking",
        environmental="arrivals board listings and baggage belts",
        personal_necessity_reason=(
            "Which board row/belt applies depends on the user's flight."
        ),
        environmental_necessity_reason=(
            "Must identify the belt from the board and locate that belt in the hall."
        ),
        operations=(
            "RESOLVE_PERSONAL",
            "IDENTIFY_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL",
        ),
        edges=(
            "RESOLVE_PERSONAL -> IDENTIFY_ENVIRONMENTAL",
            "IDENTIFY_ENVIRONMENTAL -> LOCATE_ENVIRONMENTAL",
        ),
        typed_values=("RESOLVED_REFERENCE", "ENVIRONMENTAL_FACT"),
        dependency_family="resolve_identify_locate",
    ),
    _cand(
        candidate_id="auth_seq_012",
        query=(
            "Find my reserved seat row on this cabin map and read what is printed "
            "beside that row."
        ),
        template_group="resolve_describe_cabin_map_row",
        semantic_group="seat_row_directory_describe",
        authoring_reason=(
            "DESCRIBE of the map entry beside the reserved row requires the "
            "resolved seat-row reference."
        ),
        personal="reserved seat / row assignment",
        environmental="cabin map text beside the matching row",
        personal_necessity_reason=(
            "Which map row to inspect is determined by the reservation."
        ),
        environmental_necessity_reason=(
            "The requested printed details exist only on the cabin map in view."
        ),
        operations=("RESOLVE_PERSONAL", "DESCRIBE_ENVIRONMENT"),
        edges=("RESOLVE_PERSONAL -> DESCRIBE_ENVIRONMENT",),
        typed_values=("RESOLVED_REFERENCE",),
        dependency_family="resolve_to_describe",
    ),
    _cand(
        candidate_id="auth_seq_013",
        query="Where is the locker bank assigned to my reservation?",
        template_group="resolve_locate_reservation_locker_bank",
        semantic_group="reservation_locker_bank_locate",
        authoring_reason=(
            "Reservation assigns the locker bank; LOCATE consumes that reference."
        ),
        personal="locker-bank assignment from reservation",
        environmental="locker area to locate the assigned bank",
        personal_necessity_reason=(
            "Which locker bank to find is stored in the reservation."
        ),
        environmental_necessity_reason=(
            "The answer is the environmental location of that locker bank."
        ),
        operations=("RESOLVE_PERSONAL", "LOCATE_ENVIRONMENTAL"),
        edges=("RESOLVE_PERSONAL -> LOCATE_ENVIRONMENTAL",),
        typed_values=("RESOLVED_REFERENCE",),
        dependency_family="resolve_to_locate",
    ),
    _cand(
        candidate_id="auth_seq_014",
        query="Where is my rental car stall in this garage and how do I get there?",
        template_group="resolve_locate_navigate_rental_stall",
        semantic_group="rental_stall_wayfinding",
        authoring_reason=(
            "Rental assignment resolves the stall; locate then navigate in the garage."
        ),
        personal="assigned rental car stall",
        environmental="garage stall location and route",
        personal_necessity_reason=(
            "Stall identity comes from the rental assignment; without it the "
            "destination is unknown."
        ),
        environmental_necessity_reason=(
            "Must locate that stall in the garage and navigate to it."
        ),
        operations=(
            "RESOLVE_PERSONAL",
            "LOCATE_ENVIRONMENTAL",
            "NAVIGATE_TO",
        ),
        edges=(
            "RESOLVE_PERSONAL -> LOCATE_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL -> NAVIGATE_TO",
        ),
        typed_values=("RESOLVED_REFERENCE", "LOCATION"),
        dependency_family="resolve_locate_navigate",
    ),
    _cand(
        candidate_id="auth_seq_015",
        query="Where is my appointment room and how do I get there?",
        template_group="resolve_locate_navigate_appointment_room",
        semantic_group="appointment_room_wayfinding",
        authoring_reason=(
            "Appointment room is resolved, located, then used as NAVIGATE input."
        ),
        personal="appointment room assignment",
        environmental="room location and route in the building",
        personal_necessity_reason=(
            "The destination room is defined by the appointment record."
        ),
        environmental_necessity_reason=(
            "Must locate that room environmentally and navigate to its LOCATION."
        ),
        operations=(
            "RESOLVE_PERSONAL",
            "LOCATE_ENVIRONMENTAL",
            "NAVIGATE_TO",
        ),
        edges=(
            "RESOLVE_PERSONAL -> LOCATE_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL -> NAVIGATE_TO",
        ),
        typed_values=("RESOLVED_REFERENCE", "LOCATION"),
        dependency_family="resolve_locate_navigate",
    ),
    _cand(
        candidate_id="auth_seq_016",
        query="Where is my hotel room and how do I get there from the lobby?",
        template_group="resolve_locate_navigate_hotel_room",
        semantic_group="hotel_room_wayfinding",
        authoring_reason=(
            "Hotel room reference is resolved, located from the lobby, then navigated to."
        ),
        personal="assigned hotel room",
        environmental="hotel interior location and route from lobby",
        personal_necessity_reason=(
            "Room number/identity is personal assignment data required to know the target."
        ),
        environmental_necessity_reason=(
            "Must locate the room in the hotel and produce navigation from the lobby."
        ),
        operations=(
            "RESOLVE_PERSONAL",
            "LOCATE_ENVIRONMENTAL",
            "NAVIGATE_TO",
        ),
        edges=(
            "RESOLVE_PERSONAL -> LOCATE_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL -> NAVIGATE_TO",
        ),
        typed_values=("RESOLVED_REFERENCE", "LOCATION"),
        dependency_family="resolve_locate_navigate",
    ),
    _cand(
        candidate_id="auth_seq_017",
        query="Where is my reserved train platform and how do I get there?",
        template_group="resolve_locate_navigate_reserved_platform",
        semantic_group="reserved_platform_wayfinding",
        authoring_reason=(
            "Reserved platform is resolved from booking, located, then navigated to."
        ),
        personal="reserved train platform assignment",
        environmental="platform location and route in the station",
        personal_necessity_reason=(
            "Platform identity is determined by the reservation/booking."
        ),
        environmental_necessity_reason=(
            "Must locate that platform in the station and navigate to it."
        ),
        operations=(
            "RESOLVE_PERSONAL",
            "LOCATE_ENVIRONMENTAL",
            "NAVIGATE_TO",
        ),
        edges=(
            "RESOLVE_PERSONAL -> LOCATE_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL -> NAVIGATE_TO",
        ),
        typed_values=("RESOLVED_REFERENCE", "LOCATION"),
        dependency_family="resolve_locate_navigate",
    ),
    _cand(
        candidate_id="auth_seq_018",
        query="Where is my grocery pickup locker and how do I get there?",
        template_group="resolve_locate_navigate_pickup_locker",
        semantic_group="pickup_locker_wayfinding",
        authoring_reason=(
            "Assigned locker is resolved, located, then navigated to."
        ),
        personal="assigned grocery pickup locker",
        environmental="locker location and route in the store/lot",
        personal_necessity_reason=(
            "Locker identity is personal assignment data required as the destination."
        ),
        environmental_necessity_reason=(
            "Must locate that locker environmentally and navigate to its LOCATION."
        ),
        operations=(
            "RESOLVE_PERSONAL",
            "LOCATE_ENVIRONMENTAL",
            "NAVIGATE_TO",
        ),
        edges=(
            "RESOLVE_PERSONAL -> LOCATE_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL -> NAVIGATE_TO",
        ),
        typed_values=("RESOLVED_REFERENCE", "LOCATION"),
        dependency_family="resolve_locate_navigate",
    ),
    _cand(
        candidate_id="auth_seq_019",
        query=(
            "Find my doctor's name on this wall directory and read the suite "
            "details listed beside that entry."
        ),
        template_group="resolve_describe_directory_doctor",
        semantic_group="directory_lookup_personal_doctor",
        authoring_reason=(
            "DESCRIBE of the matching directory entry requires the resolved doctor reference."
        ),
        personal="doctor identity from personal/medical contacts",
        environmental="wall directory suite details beside that name",
        personal_necessity_reason=(
            "Which directory entry to read is determined by the user's doctor identity."
        ),
        environmental_necessity_reason=(
            "Suite details are environmental text on the directory beside that entry."
        ),
        operations=("RESOLVE_PERSONAL", "DESCRIBE_ENVIRONMENT"),
        edges=("RESOLVE_PERSONAL -> DESCRIBE_ENVIRONMENT",),
        typed_values=("RESOLVED_REFERENCE",),
        dependency_family="resolve_to_describe",
    ),
    _cand(
        candidate_id="auth_seq_020",
        query=(
            "Which pharmacy window handles my prescription refill, and where is "
            "that window?"
        ),
        template_group="resolve_identify_locate_pharmacy_window",
        semantic_group="prescription_window_identify_locate",
        authoring_reason=(
            "Prescription/refill assignment resolves which window; IDENTIFY selects "
            "it in the scene; LOCATE finds that window."
        ),
        personal="prescription refill / assigned pharmacy window",
        environmental="pharmacy windows in the current scene",
        personal_necessity_reason=(
            "Which window is correct depends on the personal refill/assignment data."
        ),
        environmental_necessity_reason=(
            "Must identify the matching window among those present and locate it."
        ),
        operations=(
            "RESOLVE_PERSONAL",
            "IDENTIFY_ENVIRONMENTAL",
            "LOCATE_ENVIRONMENTAL",
        ),
        edges=(
            "RESOLVE_PERSONAL -> IDENTIFY_ENVIRONMENTAL",
            "IDENTIFY_ENVIRONMENTAL -> LOCATE_ENVIRONMENTAL",
        ),
        typed_values=("RESOLVED_REFERENCE", "ENVIRONMENTAL_FACT"),
        dependency_family="resolve_identify_locate",
    ),
)

def default_authored_sequential_candidates() -> tuple[
    AuthoredSequentialCandidate, ...
]:
    validate_authored_sequential_candidate_set(AUTHORED_SEQUENTIAL_CANDIDATES)
    return AUTHORED_SEQUENTIAL_CANDIDATES


__all__ = [
    "AUTHORED_SEQUENTIAL_CANDIDATES",
    "AuthoredReviewStatus",
    "AuthoredSequentialCandidate",
    "AuthoredSequentialReview",
    "AuthoredSequentialReviewSession",
    "CHOICE_TO_STATUS",
    "DEFAULT_AUTHORED_SEQUENTIAL_CANDIDATES_PATH",
    "DEFAULT_AUTHORED_SEQUENTIAL_REVIEWS_PATH",
    "DEFAULT_TRAIN_PATH",
    "EXPECTED_AUTHORED_SEQUENTIAL_COUNT",
    "PLANNER_BUCKET_MIXED_SEQUENTIAL",
    "SOURCE_KIND_AUTHORED",
    "default_authored_sequential_candidates",
    "format_authored_sequential_summary",
    "load_authored_sequential_candidates",
    "load_authored_sequential_reviews",
    "parse_authored_sequential_review_command",
    "parse_dependency_edge",
    "summarize_authored_sequential_reviews",
    "validate_authored_sequential_candidate_set",
    "validate_dependency_spec",
    "write_authored_sequential_candidates",
    "write_authored_sequential_reviews",
]
