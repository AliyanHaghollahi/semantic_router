"""Human Mixed-candidate review state (no auto-labeling).

Assigns only high-level planner buckets for Mixed Stage-A candidates.
Does not invent operations, anchors, ownership, dependencies, or graphs.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Self

from pydantic import field_validator

from tiergraph.enums import _CanonicalWireEnum
from tiergraph.models import TierGraphSchema
from tiergraph.planner.corpus import StageACandidate, load_candidates_jsonl


STAGE_A_BUCKET_TARGET = 24

CHOICE_TO_BUCKET: dict[str, "MixedReviewBucket"] = {}


class MixedReviewBucket(_CanonicalWireEnum, str, Enum):
    """High-level Mixed review labels for Stage-A triage."""

    MIXED_IMPLICIT = "mixed_implicit"
    MIXED_PARALLEL = "mixed_parallel"
    MIXED_SEQUENTIAL = "mixed_sequential"
    NOT_SUITABLE = "not_suitable"


CHOICE_TO_BUCKET.update(
    {
        "1": MixedReviewBucket.MIXED_IMPLICIT,
        "2": MixedReviewBucket.MIXED_PARALLEL,
        "3": MixedReviewBucket.MIXED_SEQUENTIAL,
        "4": MixedReviewBucket.NOT_SUITABLE,
    }
)

BUCKET_DEFINITIONS: tuple[tuple[MixedReviewBucket, str], ...] = (
    (
        MixedReviewBucket.MIXED_IMPLICIT,
        "surface query appears like one task, but correct execution requires "
        "an implicit personal/environmental subtask. Example: "
        '"Where is my gate?"',
    ),
    (
        MixedReviewBucket.MIXED_PARALLEL,
        "contains two or more explicit answer operations that do not depend "
        "on each other's outputs. Example: "
        '"What medication am I holding and is there a pharmacy nearby?"',
    ),
    (
        MixedReviewBucket.MIXED_SEQUENTIAL,
        "contains explicit operations where one answer operation needs the "
        "output of another. Example: "
        '"What is my gate and how do I get there?"',
    ),
    (
        MixedReviewBucket.NOT_SUITABLE,
        "old Mixed classification does not correspond cleanly to one of these "
        "planner structures or is ambiguous/bad data.",
    ),
)


class ReviewStatus(_CanonicalWireEnum, str, Enum):
    REVIEWED = "reviewed"


class _ReviewSchema(TierGraphSchema):
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


class MixedReviewRecord(_ReviewSchema):
    """One human Mixed-bucket decision (no planner structure labels)."""

    source_query_id: str
    semantic_group_id: str
    query: str
    source_classification_label: str
    planner_bucket: MixedReviewBucket
    review_status: ReviewStatus = ReviewStatus.REVIEWED

    @field_validator(
        "source_query_id",
        "semantic_group_id",
        "query",
        "source_classification_label",
    )
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("source_classification_label")
    @classmethod
    def _must_be_mixed(cls, value: str) -> str:
        if value != "Mixed":
            raise ValueError("Mixed reviews require source_classification_label=Mixed")
        return value


def load_mixed_candidates(
    candidates_path: str | Path,
) -> tuple[StageACandidate, ...]:
    """Load Mixed candidates only, in file order."""
    candidates = load_candidates_jsonl(candidates_path)
    mixed = tuple(
        item
        for item in candidates
        if item.source_classification_label == "Mixed"
    )
    return mixed


def load_mixed_reviews(path: str | Path) -> tuple[MixedReviewRecord, ...]:
    path = Path(path)
    if not path.is_file():
        return ()
    records: list[MixedReviewRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = MixedReviewRecord.model_validate(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"invalid mixed review at {path}:{line_number}: {exc}"
            ) from exc
        if record.source_query_id in seen:
            raise ValueError(
                f"duplicate source_query_id in reviews: {record.source_query_id}"
            )
        seen.add(record.source_query_id)
        records.append(record)
    return tuple(records)


def write_mixed_reviews(
    path: str | Path,
    reviews: Sequence[MixedReviewRecord],
) -> None:
    """Write reviews deterministically ordered by source_query_id."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(reviews, key=lambda item: item.source_query_id)
    seen: set[str] = set()
    for record in ordered:
        if record.source_query_id in seen:
            raise ValueError(
                f"duplicate source_query_id rejected: {record.source_query_id}"
            )
        seen.add(record.source_query_id)
    lines = [
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
        for record in ordered
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def summarize_mixed_reviews(
    mixed_candidates: Sequence[StageACandidate],
    reviews: Sequence[MixedReviewRecord],
) -> dict[str, Any]:
    """Return progress counts for Mixed review."""
    total = len(mixed_candidates)
    counts = Counter(item.planner_bucket.value for item in reviews)
    reviewed_ids = {item.source_query_id for item in reviews}
    candidate_ids = {item.source_query_id for item in mixed_candidates}
    unknown = reviewed_ids - candidate_ids
    if unknown:
        raise ValueError(
            "reviews reference unknown Mixed candidates: "
            + ", ".join(sorted(unknown))
        )
    remaining = total - len(reviewed_ids)
    targets = {
        bucket.value: {
            "count": counts.get(bucket.value, 0),
            "target": STAGE_A_BUCKET_TARGET,
            "reached_target": counts.get(bucket.value, 0) >= STAGE_A_BUCKET_TARGET,
        }
        for bucket in (
            MixedReviewBucket.MIXED_IMPLICIT,
            MixedReviewBucket.MIXED_PARALLEL,
            MixedReviewBucket.MIXED_SEQUENTIAL,
        )
    }
    return {
        "total_mixed": total,
        "reviewed": len(reviewed_ids),
        "remaining": remaining,
        "counts": {
            MixedReviewBucket.MIXED_IMPLICIT.value: counts.get(
                MixedReviewBucket.MIXED_IMPLICIT.value, 0
            ),
            MixedReviewBucket.MIXED_PARALLEL.value: counts.get(
                MixedReviewBucket.MIXED_PARALLEL.value, 0
            ),
            MixedReviewBucket.MIXED_SEQUENTIAL.value: counts.get(
                MixedReviewBucket.MIXED_SEQUENTIAL.value, 0
            ),
            MixedReviewBucket.NOT_SUITABLE.value: counts.get(
                MixedReviewBucket.NOT_SUITABLE.value, 0
            ),
        },
        "targets": targets,
    }


def format_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        f"Reviewed: {summary['reviewed']} / {summary['total_mixed']}",
        "",
        f"MIXED_IMPLICIT: {summary['counts']['mixed_implicit']}",
        f"MIXED_PARALLEL: {summary['counts']['mixed_parallel']}",
        f"MIXED_SEQUENTIAL: {summary['counts']['mixed_sequential']}",
        f"NOT_SUITABLE: {summary['counts']['not_suitable']}",
        f"Remaining: {summary['remaining']}",
        "",
        "Stage-A targets (24 each; review may continue for spares):",
    ]
    for key, label in (
        ("mixed_implicit", "MIXED_IMPLICIT"),
        ("mixed_parallel", "MIXED_PARALLEL"),
        ("mixed_sequential", "MIXED_SEQUENTIAL"),
    ):
        info = summary["targets"][key]
        status = "yes" if info["reached_target"] else "no"
        lines.append(
            f"  {label}: {info['count']} / {info['target']}  reached={status}"
        )
    return "\n".join(lines)


class MixedReviewSession:
    """Mutable review session with save/resume and back navigation."""

    def __init__(
        self,
        mixed_candidates: Sequence[StageACandidate],
        *,
        reviews_path: str | Path,
        existing_reviews: Sequence[MixedReviewRecord] | None = None,
    ) -> None:
        if not mixed_candidates:
            raise ValueError("mixed_candidates must not be empty")
        ids = [item.source_query_id for item in mixed_candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate Mixed candidate source_query_id")
        for item in mixed_candidates:
            if item.source_classification_label != "Mixed":
                raise ValueError(
                    "non-Mixed candidate passed to MixedReviewSession: "
                    f"{item.source_query_id}"
                )
        self.mixed_candidates = tuple(mixed_candidates)
        self.reviews_path = Path(reviews_path)
        self._by_id = {item.source_query_id: item for item in self.mixed_candidates}
        self._reviews: dict[str, MixedReviewRecord] = {}
        if existing_reviews is None:
            existing_reviews = load_mixed_reviews(self.reviews_path)
        for record in existing_reviews:
            if record.source_query_id not in self._by_id:
                raise ValueError(
                    "review references unknown Mixed candidate: "
                    f"{record.source_query_id}"
                )
            if record.source_query_id in self._reviews:
                raise ValueError(
                    f"duplicate source_query_id rejected: {record.source_query_id}"
                )
            # Preserve exact candidate query text.
            candidate = self._by_id[record.source_query_id]
            if record.query != candidate.query:
                raise ValueError(
                    "review query text must match candidate exactly: "
                    f"{record.source_query_id}"
                )
            self._reviews[record.source_query_id] = record
        self._history: list[str] = []
        self._cursor = self._first_unreviewed_index()

    @property
    def reviews(self) -> tuple[MixedReviewRecord, ...]:
        return tuple(
            self._reviews[item.source_query_id]
            for item in self.mixed_candidates
            if item.source_query_id in self._reviews
        )

    def _first_unreviewed_index(self) -> int:
        for index, item in enumerate(self.mixed_candidates):
            if item.source_query_id not in self._reviews:
                return index
        return len(self.mixed_candidates)

    def current(self) -> StageACandidate | None:
        if self._cursor >= len(self.mixed_candidates):
            return None
        return self.mixed_candidates[self._cursor]

    def current_position(self) -> tuple[int, int]:
        """1-based position among Mixed candidates and total."""
        if self._cursor >= len(self.mixed_candidates):
            return len(self.mixed_candidates), len(self.mixed_candidates)
        return self._cursor + 1, len(self.mixed_candidates)

    def summary(self) -> dict[str, Any]:
        return summarize_mixed_reviews(self.mixed_candidates, self.reviews)

    def save(self) -> None:
        write_mixed_reviews(self.reviews_path, self.reviews)

    def apply_bucket(
        self,
        bucket: MixedReviewBucket,
        *,
        persist: bool = True,
    ) -> MixedReviewRecord:
        candidate = self.current()
        if candidate is None:
            raise ValueError("no remaining Mixed candidates to review")
        record = MixedReviewRecord(
            source_query_id=candidate.source_query_id,
            semantic_group_id=candidate.semantic_group_id,
            query=candidate.query,
            source_classification_label="Mixed",
            planner_bucket=bucket,
            review_status=ReviewStatus.REVIEWED,
        )
        self._reviews[candidate.source_query_id] = record
        self._history.append(candidate.source_query_id)
        self._cursor = self._first_unreviewed_index()
        if persist:
            self.save()
        return record

    def apply_choice(self, choice: str, *, persist: bool = True) -> MixedReviewRecord:
        key = choice.strip().lower()
        if key not in CHOICE_TO_BUCKET:
            raise ValueError(f"unknown review choice: {choice!r}")
        return self.apply_bucket(CHOICE_TO_BUCKET[key], persist=persist)

    def skip(self) -> StageACandidate | None:
        """Advance cursor past the current item without saving a review."""
        if self._cursor >= len(self.mixed_candidates):
            return None
        self._cursor += 1
        while (
            self._cursor < len(self.mixed_candidates)
            and self.mixed_candidates[self._cursor].source_query_id in self._reviews
        ):
            self._cursor += 1
        return self.current()

    def back(self, *, persist: bool = True) -> StageACandidate | None:
        """Undo the latest decision and return to that candidate."""
        if not self._history:
            # Fall back to previous Mixed index if any.
            if self._cursor <= 0:
                return self.current()
            self._cursor = max(0, self._cursor - 1)
            return self.current()
        source_id = self._history.pop()
        self._reviews.pop(source_id, None)
        for index, item in enumerate(self.mixed_candidates):
            if item.source_query_id == source_id:
                self._cursor = index
                break
        if persist:
            self.save()
        return self.current()


def parse_review_command(raw: str) -> tuple[str, MixedReviewBucket | None]:
    """Parse a CLI token into an action name and optional bucket."""
    token = raw.strip().lower()
    if token in CHOICE_TO_BUCKET:
        return "assign", CHOICE_TO_BUCKET[token]
    if token in {"s", "skip"}:
        return "skip", None
    if token in {"b", "back"}:
        return "back", None
    if token in {"q", "quit"}:
        return "quit", None
    if token in {"p", "progress", "summary"}:
        return "summary", None
    raise ValueError(f"unknown command: {raw!r}")


__all__ = [
    "BUCKET_DEFINITIONS",
    "CHOICE_TO_BUCKET",
    "MixedReviewBucket",
    "MixedReviewRecord",
    "MixedReviewSession",
    "ReviewStatus",
    "STAGE_A_BUCKET_TARGET",
    "format_summary",
    "load_mixed_candidates",
    "load_mixed_reviews",
    "parse_review_command",
    "summarize_mixed_reviews",
    "write_mixed_reviews",
]
