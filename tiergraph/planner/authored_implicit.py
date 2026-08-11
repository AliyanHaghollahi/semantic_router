"""Authored Stage-A MIXED_IMPLICIT candidates and human review state.

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

from pydantic import field_validator, model_validator

from tiergraph.enums import _CanonicalWireEnum
from tiergraph.models import TierGraphSchema
from tiergraph.planner.corpus import normalize_query_key


DEFAULT_AUTHORED_CANDIDATES_PATH = Path(
    "dataset/planner/stage_a_authored_implicit_candidates.jsonl"
)
DEFAULT_AUTHORED_REVIEWS_PATH = Path(
    "dataset/planner/stage_a_authored_implicit_reviews.jsonl"
)
DEFAULT_TRAIN_PATH = Path("dataset/training_data.json")
EXPECTED_AUTHORED_COUNT = 12
SOURCE_KIND_AUTHORED = "authored_stage_a"
PLANNER_BUCKET_MIXED_IMPLICIT = "MIXED_IMPLICIT"

CHOICE_TO_STATUS: dict[str, "AuthoredReviewStatus"] = {}


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


class AuthoredImplicitCandidate(_AuthoredSchema):
    """One authored MIXED_IMPLICIT candidate awaiting human review."""

    candidate_id: str
    query: str
    source_kind: str
    planner_bucket: str
    template_group: str
    authoring_reason: str
    intended_personal_requirement: str
    intended_environmental_requirement: str
    review_status: AuthoredReviewStatus = AuthoredReviewStatus.UNREVIEWED

    @field_validator(
        "candidate_id",
        "query",
        "source_kind",
        "planner_bucket",
        "template_group",
        "authoring_reason",
        "intended_personal_requirement",
        "intended_environmental_requirement",
    )
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if type(value) is not str or not value.strip():
            raise ValueError("must be a nonblank string")
        return value

    @model_validator(mode="after")
    def _validate_authored_contract(self) -> "AuthoredImplicitCandidate":
        if self.source_kind != SOURCE_KIND_AUTHORED:
            raise ValueError(
                f"source_kind must be {SOURCE_KIND_AUTHORED!r}, "
                f"got {self.source_kind!r}"
            )
        if self.planner_bucket != PLANNER_BUCKET_MIXED_IMPLICIT:
            raise ValueError(
                f"planner_bucket must be {PLANNER_BUCKET_MIXED_IMPLICIT!r}, "
                f"got {self.planner_bucket!r}"
            )
        if not self.candidate_id.startswith("auth_imp_"):
            raise ValueError(
                "candidate_id must start with 'auth_imp_' "
                f"(got {self.candidate_id!r})"
            )
        return self


class AuthoredImplicitReview(_AuthoredSchema):
    """Persisted human decision for one authored candidate.

    Stores provenance copies so review files remain self-describing without
    mutating the candidate JSONL.
    """

    candidate_id: str
    query: str
    source_kind: str
    planner_bucket: str
    template_group: str
    authoring_reason: str
    intended_personal_requirement: str
    intended_environmental_requirement: str
    review_status: AuthoredReviewStatus

    @field_validator(
        "candidate_id",
        "query",
        "source_kind",
        "planner_bucket",
        "template_group",
        "authoring_reason",
        "intended_personal_requirement",
        "intended_environmental_requirement",
    )
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if type(value) is not str or not value.strip():
            raise ValueError("must be a nonblank string")
        return value

    @model_validator(mode="after")
    def _validate_decision(self) -> "AuthoredImplicitReview":
        if self.review_status is AuthoredReviewStatus.UNREVIEWED:
            raise ValueError("persisted reviews cannot use UNREVIEWED")
        if self.source_kind != SOURCE_KIND_AUTHORED:
            raise ValueError(
                f"source_kind must be {SOURCE_KIND_AUTHORED!r}"
            )
        if self.planner_bucket != PLANNER_BUCKET_MIXED_IMPLICIT:
            raise ValueError(
                f"planner_bucket must be {PLANNER_BUCKET_MIXED_IMPLICIT!r}"
            )
        return self


def load_authored_candidates(
    path: str | Path,
) -> tuple[AuthoredImplicitCandidate, ...]:
    path = Path(path)
    records: list[AuthoredImplicitCandidate] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(
                AuthoredImplicitCandidate.model_validate(json.loads(line))
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"invalid authored candidate at {path}:{line_number}: {exc}"
            ) from exc
    return tuple(records)


def write_authored_candidates(
    path: str | Path,
    candidates: Sequence[AuthoredImplicitCandidate],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(candidates, key=lambda item: item.candidate_id)
    lines = [
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
        for item in ordered
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_authored_reviews(path: str | Path) -> tuple[AuthoredImplicitReview, ...]:
    path = Path(path)
    if not path.is_file():
        return ()
    records: list[AuthoredImplicitReview] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = AuthoredImplicitReview.model_validate(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"invalid authored review at {path}:{line_number}: {exc}"
            ) from exc
        if record.candidate_id in seen:
            raise ValueError(
                f"duplicate candidate_id in reviews: {record.candidate_id}"
            )
        seen.add(record.candidate_id)
        records.append(record)
    return tuple(records)


def write_authored_reviews(
    path: str | Path,
    reviews: Sequence[AuthoredImplicitReview],
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


def validate_authored_candidate_set(
    candidates: Sequence[AuthoredImplicitCandidate],
    *,
    train_path: str | Path | None = DEFAULT_TRAIN_PATH,
    expected_count: int = EXPECTED_AUTHORED_COUNT,
) -> None:
    """Validate identity, provenance, and non-injection into training data."""
    if len(candidates) != expected_count:
        raise ValueError(
            f"expected exactly {expected_count} authored candidates, "
            f"found {len(candidates)}"
        )
    ids = [item.candidate_id for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate_id in authored set")
    keys = [normalize_query_key(item.query) for item in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate normalized query in authored set")
    for item in candidates:
        if item.source_kind != SOURCE_KIND_AUTHORED:
            raise ValueError(f"bad source_kind on {item.candidate_id}")
        if item.planner_bucket != PLANNER_BUCKET_MIXED_IMPLICIT:
            raise ValueError(f"bad planner_bucket on {item.candidate_id}")
        if item.review_status not in AuthoredReviewStatus:
            raise ValueError(f"bad review_status on {item.candidate_id}")

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
                "authored candidates must not appear in training_data.json: "
                + ", ".join(overlap)
            )


def summarize_authored_reviews(
    candidates: Sequence[AuthoredImplicitCandidate],
    reviews: Sequence[AuthoredImplicitReview],
) -> dict[str, Any]:
    by_status = Counter(item.review_status.value for item in reviews)
    reviewed_ids = {item.candidate_id for item in reviews}
    unknown = reviewed_ids - {item.candidate_id for item in candidates}
    if unknown:
        raise ValueError(
            "reviews reference unknown authored candidates: "
            + ", ".join(sorted(unknown))
        )
    return {
        "total": len(candidates),
        "reviewed": len(reviewed_ids),
        "remaining": len(candidates) - len(reviewed_ids),
        "ACCEPT": by_status.get(AuthoredReviewStatus.ACCEPT.value, 0),
        "REJECT": by_status.get(AuthoredReviewStatus.REJECT.value, 0),
        "UNREVIEWED": len(candidates) - len(reviewed_ids),
    }


def format_authored_summary(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"Reviewed: {summary['reviewed']} / {summary['total']}",
            f"ACCEPT: {summary['ACCEPT']}",
            f"REJECT: {summary['REJECT']}",
            f"Remaining: {summary['remaining']}",
        ]
    )


def parse_authored_review_command(
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


class AuthoredImplicitReviewSession:
    """Resumable ACCEPT/REJECT review over authored candidates."""

    def __init__(
        self,
        candidates: Sequence[AuthoredImplicitCandidate],
        *,
        reviews_path: str | Path,
        existing_reviews: Sequence[AuthoredImplicitReview] | None = None,
        train_path: str | Path | None = DEFAULT_TRAIN_PATH,
    ) -> None:
        validate_authored_candidate_set(candidates, train_path=train_path)
        ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
        self.candidates = ordered
        self.reviews_path = Path(reviews_path)
        self._by_id = {item.candidate_id: item for item in ordered}
        self._reviews: dict[str, AuthoredImplicitReview] = {}
        if existing_reviews is None:
            existing_reviews = load_authored_reviews(self.reviews_path)
        for record in existing_reviews:
            candidate = self._by_id.get(record.candidate_id)
            if candidate is None:
                raise ValueError(
                    "review references unknown authored candidate: "
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
    def reviews(self) -> tuple[AuthoredImplicitReview, ...]:
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

    def current(self) -> AuthoredImplicitCandidate | None:
        if self._cursor >= len(self.candidates):
            return None
        return self.candidates[self._cursor]

    def current_position(self) -> tuple[int, int]:
        if self._cursor >= len(self.candidates):
            return len(self.candidates), len(self.candidates)
        return self._cursor + 1, len(self.candidates)

    def summary(self) -> dict[str, Any]:
        return summarize_authored_reviews(self.candidates, self.reviews)

    def save(self) -> None:
        write_authored_reviews(self.reviews_path, self.reviews)

    def apply_status(
        self,
        status: AuthoredReviewStatus,
        *,
        persist: bool = True,
    ) -> AuthoredImplicitReview:
        if status is AuthoredReviewStatus.UNREVIEWED:
            raise ValueError("invalid review value: UNREVIEWED")
        candidate = self.current()
        if candidate is None:
            raise ValueError("no remaining authored candidates to review")
        record = AuthoredImplicitReview(
            candidate_id=candidate.candidate_id,
            query=candidate.query,
            source_kind=candidate.source_kind,
            planner_bucket=candidate.planner_bucket,
            template_group=candidate.template_group,
            authoring_reason=candidate.authoring_reason,
            intended_personal_requirement=candidate.intended_personal_requirement,
            intended_environmental_requirement=(
                candidate.intended_environmental_requirement
            ),
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
    ) -> AuthoredImplicitReview:
        key = choice.strip().lower()
        if key not in CHOICE_TO_STATUS:
            raise ValueError(f"invalid review value: {choice!r}")
        return self.apply_status(CHOICE_TO_STATUS[key], persist=persist)

    def skip(self) -> AuthoredImplicitCandidate | None:
        if self._cursor >= len(self.candidates):
            return None
        self._cursor += 1
        while (
            self._cursor < len(self.candidates)
            and self.candidates[self._cursor].candidate_id in self._reviews
        ):
            self._cursor += 1
        return self.current()

    def back(self, *, persist: bool = True) -> AuthoredImplicitCandidate | None:
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


# Canonical authored Stage-A MIXED_IMPLICIT candidate set (human-authored).
AUTHORED_IMPLICIT_CANDIDATES: tuple[AuthoredImplicitCandidate, ...] = (
    AuthoredImplicitCandidate(
        candidate_id="auth_imp_001",
        query="Is this snack safe for my allergies?",
        source_kind=SOURCE_KIND_AUTHORED,
        planner_bucket=PLANNER_BUCKET_MIXED_IMPLICIT,
        template_group="allergies_food_item",
        authoring_reason=(
            "One surface safety question requiring allergy profile plus "
            "identification of the observed snack."
        ),
        intended_personal_requirement="user allergy profile",
        intended_environmental_requirement="identify this snack / its ingredients",
        review_status=AuthoredReviewStatus.UNREVIEWED,
    ),
    AuthoredImplicitCandidate(
        candidate_id="auth_imp_002",
        query="Is this the medication I am supposed to take tonight?",
        source_kind=SOURCE_KIND_AUTHORED,
        planner_bucket=PLANNER_BUCKET_MIXED_IMPLICIT,
        template_group="medication_tonight_schedule",
        authoring_reason=(
            "Single verification of a held medication against tonight's "
            "prescribed dose schedule."
        ),
        intended_personal_requirement="tonight's prescribed medication schedule",
        intended_environmental_requirement="identify this medication",
        review_status=AuthoredReviewStatus.UNREVIEWED,
    ),
    AuthoredImplicitCandidate(
        candidate_id="auth_imp_003",
        query="Is this the entrance for my appointment?",
        source_kind=SOURCE_KIND_AUTHORED,
        planner_bucket=PLANNER_BUCKET_MIXED_IMPLICIT,
        template_group="appointment_entrance",
        authoring_reason=(
            "One entrance-check question needing appointment destination "
            "plus current entrance observation."
        ),
        intended_personal_requirement="appointment location / clinic destination",
        intended_environmental_requirement="identify this entrance",
        review_status=AuthoredReviewStatus.UNREVIEWED,
    ),
    AuthoredImplicitCandidate(
        candidate_id="auth_imp_004",
        query="Does this seat match my reservation?",
        source_kind=SOURCE_KIND_AUTHORED,
        planner_bucket=PLANNER_BUCKET_MIXED_IMPLICIT,
        template_group="reservation_seat",
        authoring_reason=(
            "Single seat-match question requiring reservation details and "
            "observation of the current seat."
        ),
        intended_personal_requirement="seat reservation record",
        intended_environmental_requirement="identify this seat",
        review_status=AuthoredReviewStatus.UNREVIEWED,
    ),
    AuthoredImplicitCandidate(
        candidate_id="auth_imp_005",
        query="Is this item the one on my shopping list?",
        source_kind=SOURCE_KIND_AUTHORED,
        planner_bucket=PLANNER_BUCKET_MIXED_IMPLICIT,
        template_group="shopping_list_item",
        authoring_reason=(
            "One list-membership check over an observed store item against "
            "the user's shopping list."
        ),
        intended_personal_requirement="shopping list contents",
        intended_environmental_requirement="identify this item",
        review_status=AuthoredReviewStatus.UNREVIEWED,
    ),
    AuthoredImplicitCandidate(
        candidate_id="auth_imp_006",
        query="Does this label match the dosage I was prescribed?",
        source_kind=SOURCE_KIND_AUTHORED,
        planner_bucket=PLANNER_BUCKET_MIXED_IMPLICIT,
        template_group="prescription_label_dosage",
        authoring_reason=(
            "Single dosage-consistency check between a visible label and "
            "prescribed dosage."
        ),
        intended_personal_requirement="prescribed dosage instructions",
        intended_environmental_requirement="read this medication label",
        review_status=AuthoredReviewStatus.UNREVIEWED,
    ),
    AuthoredImplicitCandidate(
        candidate_id="auth_imp_007",
        query="Does this boarding pass match my flight today?",
        source_kind=SOURCE_KIND_AUTHORED,
        planner_bucket=PLANNER_BUCKET_MIXED_IMPLICIT,
        template_group="travel_boarding_pass",
        authoring_reason=(
            "One boarding-pass verification against today's flight booking."
        ),
        intended_personal_requirement="today's flight booking details",
        intended_environmental_requirement="read this boarding pass",
        review_status=AuthoredReviewStatus.UNREVIEWED,
    ),
    AuthoredImplicitCandidate(
        candidate_id="auth_imp_008",
        query="Is this suitcase the one registered under my name?",
        source_kind=SOURCE_KIND_AUTHORED,
        planner_bucket=PLANNER_BUCKET_MIXED_IMPLICIT,
        template_group="personal_belonging_luggage",
        authoring_reason=(
            "Single belonging check tying an observed suitcase to the user's "
            "registered luggage identity."
        ),
        intended_personal_requirement="luggage registration / passenger name",
        intended_environmental_requirement="identify this suitcase",
        review_status=AuthoredReviewStatus.UNREVIEWED,
    ),
    AuthoredImplicitCandidate(
        candidate_id="auth_imp_009",
        query="Does this room number match my meeting location?",
        source_kind=SOURCE_KIND_AUTHORED,
        planner_bucket=PLANNER_BUCKET_MIXED_IMPLICIT,
        template_group="schedule_meeting_room",
        authoring_reason=(
            "One room-match question requiring calendar meeting location and "
            "current room number observation."
        ),
        intended_personal_requirement="meeting location from schedule",
        intended_environmental_requirement="read this room number",
        review_status=AuthoredReviewStatus.UNREVIEWED,
    ),
    AuthoredImplicitCandidate(
        candidate_id="auth_imp_010",
        query="Can I eat this dish with my lactose restriction?",
        source_kind=SOURCE_KIND_AUTHORED,
        planner_bucket=PLANNER_BUCKET_MIXED_IMPLICIT,
        template_group="dietary_restriction_dish",
        authoring_reason=(
            "Single dietary-safety question needing lactose restriction and "
            "observed dish contents."
        ),
        intended_personal_requirement="lactose / dietary restriction profile",
        intended_environmental_requirement="identify this dish / ingredients",
        review_status=AuthoredReviewStatus.UNREVIEWED,
    ),
    AuthoredImplicitCandidate(
        candidate_id="auth_imp_011",
        query="Is this platform the one for my reserved train?",
        source_kind=SOURCE_KIND_AUTHORED,
        planner_bucket=PLANNER_BUCKET_MIXED_IMPLICIT,
        template_group="train_platform_reservation",
        authoring_reason=(
            "One platform-check question requiring reserved train details and "
            "current platform observation."
        ),
        intended_personal_requirement="reserved train booking",
        intended_environmental_requirement="identify this platform",
        review_status=AuthoredReviewStatus.UNREVIEWED,
    ),
    AuthoredImplicitCandidate(
        candidate_id="auth_imp_012",
        query="Is this package the takeout order under my account?",
        source_kind=SOURCE_KIND_AUTHORED,
        planner_bucket=PLANNER_BUCKET_MIXED_IMPLICIT,
        template_group="order_pickup_account",
        authoring_reason=(
            "Single pickup verification of an observed package against the "
            "user's takeout order account."
        ),
        intended_personal_requirement="takeout order under user account",
        intended_environmental_requirement="identify this package",
        review_status=AuthoredReviewStatus.UNREVIEWED,
    ),
)


def default_authored_candidates() -> tuple[AuthoredImplicitCandidate, ...]:
    validate_authored_candidate_set(AUTHORED_IMPLICIT_CANDIDATES)
    return AUTHORED_IMPLICIT_CANDIDATES


__all__ = [
    "AUTHORED_IMPLICIT_CANDIDATES",
    "AuthoredImplicitCandidate",
    "AuthoredImplicitReview",
    "AuthoredImplicitReviewSession",
    "AuthoredReviewStatus",
    "CHOICE_TO_STATUS",
    "DEFAULT_AUTHORED_CANDIDATES_PATH",
    "DEFAULT_AUTHORED_REVIEWS_PATH",
    "DEFAULT_TRAIN_PATH",
    "EXPECTED_AUTHORED_COUNT",
    "PLANNER_BUCKET_MIXED_IMPLICIT",
    "SOURCE_KIND_AUTHORED",
    "default_authored_candidates",
    "format_authored_summary",
    "load_authored_candidates",
    "load_authored_reviews",
    "parse_authored_review_command",
    "summarize_authored_reviews",
    "validate_authored_candidate_set",
    "write_authored_candidates",
    "write_authored_reviews",
]
