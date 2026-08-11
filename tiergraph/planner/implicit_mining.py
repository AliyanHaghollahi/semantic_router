"""Conservative mining of possible MIXED_IMPLICIT candidates.

This module ranks Personal/Environmental unique queries that *may* be
structurally implicit-mixed for human review. It never mutates ground-truth
classification labels and never invents planner structure labels.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tiergraph.planner.corpus import (
    DEFAULT_CANDIDATE_SEED,
    StageACandidate,
    build_unique_query_pool,
    load_classification_rows,
    normalize_query_key,
)
from tiergraph.planner.mixed_review import load_mixed_reviews


DEFAULT_IMPLICIT_LIMIT = 80
DEFAULT_TRAIN_PATH = Path("dataset/training_data.json")
DEFAULT_REVIEWS_PATH = Path("dataset/planner/stage_a_mixed_reviews.jsonl")
DEFAULT_OUTPUT_PATH = Path("dataset/planner/stage_a_implicit_candidates.jsonl")

# Strong PERSONAL evidence only: stored/user-specific facts or belongings.
# Generic first-person / egocentric language (I, me, where I am sitting) is NOT
# personal evidence by itself.
_STRONG_PERSONAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "personal:my_medication",
        re.compile(
            r"\bmy\s+(medication|medications|medicine|prescription|prescriptions)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "personal:medication_list",
        re.compile(r"\b(on my list|my list|medication list|medicine list)\b", re.IGNORECASE),
    ),
    (
        "personal:allergy_profile",
        re.compile(r"\ballerg(?:y|ies|ic)\b", re.IGNORECASE),
    ),
    (
        "personal:dietary_restriction",
        re.compile(r"\b(dietary|diet restriction|food restriction)\b", re.IGNORECASE),
    ),
    (
        "personal:my_appointment",
        re.compile(r"\bmy\s+(appointment|doctor|physician|specialist)\b", re.IGNORECASE),
    ),
    (
        "personal:my_reservation",
        re.compile(r"\bmy\s+(reservation|booking|hotel)\b", re.IGNORECASE),
    ),
    (
        "personal:my_travel",
        re.compile(
            r"\bmy\s+(flight|gate|seat|ticket|itinerary|boarding pass)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "personal:my_order",
        re.compile(r"\bmy\s+(order|purchase|package)\b", re.IGNORECASE),
    ),
    (
        "personal:my_address",
        re.compile(r"\bmy\s+(saved\s+)?address\b", re.IGNORECASE),
    ),
    (
        "personal:my_schedule",
        re.compile(r"\bmy\s+(schedule|calendar|meeting)\b", re.IGNORECASE),
    ),
    (
        "personal:my_usual",
        re.compile(r"\bmy\s+(usual|brand|size|preferred)\b", re.IGNORECASE),
    ),
    (
        "personal:my_record",
        re.compile(
            r"\bmy\s+(insurance|passport|diagnosis|blood type|health record|patient)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "personal:prescription_instructions",
        re.compile(
            r"\b(supposed to take|take this|empty stomach|dosage|refill)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "personal:belonging_is_this_my",
        re.compile(r"\bis this my\b", re.IGNORECASE),
    ),
)

# ENVIRONMENTAL evidence must be CURRENT/OBSERVED scene input.
# A physical place/object named only as a property of personal data
# (e.g. "my flight's terminal", "my usual pharmacy") does NOT qualify.
_ENVIRONMENTAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "environmental:deictic_referent",
        # this/that/these/those (+ optional adjective) + physical referent,
        # excluding temporal "this afternoon/morning/...".
        re.compile(
            r"\b(?:this|these|that|those)(?!\s+"
            r"(?:afternoon|morning|evening|night|week|month|year|time|weekend))"
            r"(?:\s+\w+){0,2}\s+"
            r"(?:medication|medicine|pill|pills|sign|menu|object|bottle|desk|"
            r"room|floor|entrance|terminal|clinic|pharmacy|shelf|door|label|"
            r"board|map|path|hallway|counter|screen|machine|building|street|"
            r"hydrant|lawn|form|item|package|box|bag|ticket|document)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "environmental:deictic_bare_object",
        # "take this" / "on this" / "is this" where the deictic stands alone
        # as the observed referent (common for held objects).
        re.compile(
            r"\b(?:take|on|is|does|contains?|holding|read|identify|look(?:ing)? at)"
            r"\s+(?:this|that|these|those)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "environmental:visible_scene",
        re.compile(
            r"\b(visible|in (?:front of|view)|looking at|what does this|"
            r"read this|read the|nearby|around me|in this room)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "environmental:observed_medication",
        re.compile(
            r"\b(this|that|these|those)\s+(medication|medicine|pill|pills)\b",
            re.IGNORECASE,
        ),
    ),
)

# Bonus only after both sides already qualify. Never creates personal evidence.
_DECISION_BONUS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "decision_form:is_this_for_me",
        re.compile(r"\bis this\b.+\b(for me|mine)\b", re.IGNORECASE),
    ),
    (
        "decision_form:is_this_the_right",
        re.compile(r"\bis this the (right|correct|one)\b", re.IGNORECASE),
    ),
    (
        "decision_form:does_this_match",
        re.compile(r"\bdoes this match\b", re.IGNORECASE),
    ),
    (
        "decision_form:am_i_allergic_to_observed",
        re.compile(
            r"\bam i allergic\b.+\b(this|these|that|here|menu|food|item)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "decision_form:is_this_my",
        re.compile(r"\bis this my\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True, slots=True)
class ImplicitMineCandidate:
    """One mined Personal/Environmental query for implicit-mixed review."""

    source_id: str
    query: str
    original_label: str
    normalized_query: str
    score: int
    mining_reasons: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "query": self.query,
            "original_label": self.original_label,
            "normalized_query": self.normalized_query,
            "score": self.score,
            "mining_reasons": list(self.mining_reasons),
        }


def _pattern_hits(
    query: str,
    patterns: Sequence[tuple[str, re.Pattern[str]]],
) -> list[str]:
    return [name for name, pattern in patterns if pattern.search(query)]


def score_implicit_candidate(query: str) -> tuple[int, tuple[str, ...]]:
    """Score a query as a possible implicit-mixed case.

    Requires both:
    - strong personal-context evidence (profile/schedule/belonging/prescription), and
    - environmental/observed/deictic evidence.

    Generic first-person pronouns and egocentric spatial phrasing alone never
    satisfy the personal side. Decision-form patterns are bonuses only.
    """
    if type(query) is not str or not query.strip():
        return 0, ()

    personal_hits = _pattern_hits(query, _STRONG_PERSONAL_PATTERNS)
    environmental_hits = _pattern_hits(query, _ENVIRONMENTAL_PATTERNS)

    if not personal_hits or not environmental_hits:
        return 0, ()

    reasons: list[str] = []
    score = 10
    reasons.extend(personal_hits)
    reasons.extend(environmental_hits)

    # Modest bonus for additional independent cues on each side.
    score += min(3, max(0, len(personal_hits) - 1))
    score += min(3, max(0, len(environmental_hits) - 1))

    for reason, pattern in _DECISION_BONUS_PATTERNS:
        if pattern.search(query):
            score += 5
            reasons.append(reason)

    return score, tuple(reasons)


def excluded_normalized_queries(
    *,
    unique_pool: Sequence[StageACandidate],
    reviews_path: str | Path | None = None,
) -> set[str]:
    """Normalized queries that must not be mined.

    Excludes original Mixed uniques and any query already present in completed
    Mixed review data (by exact normalized text).
    """
    excluded = {
        normalize_query_key(item.query)
        for item in unique_pool
        if item.source_classification_label == "Mixed"
    }
    if reviews_path is not None and Path(reviews_path).is_file():
        for record in load_mixed_reviews(reviews_path):
            excluded.add(normalize_query_key(record.query))
    return excluded


def iter_personal_environmental_uniques(
    unique_pool: Sequence[StageACandidate],
    *,
    excluded_keys: Iterable[str] = (),
) -> tuple[StageACandidate, ...]:
    """Return Personal/Environmental uniques not in ``excluded_keys``."""
    blocked = set(excluded_keys)
    selected: list[StageACandidate] = []
    seen_keys: set[str] = set()
    for item in unique_pool:
        if item.source_classification_label not in {"Personal", "Environmental"}:
            continue
        key = normalize_query_key(item.query)
        if key in blocked or key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(item)
    return tuple(selected)


def mine_implicit_candidates(
    unique_pool: Sequence[StageACandidate],
    *,
    excluded_keys: Iterable[str] = (),
    limit: int = DEFAULT_IMPLICIT_LIMIT,
    seed: int = DEFAULT_CANDIDATE_SEED,
) -> tuple[ImplicitMineCandidate, ...]:
    """Rank and shortlist possible implicit-mixed candidates.

    Deterministic: sort by descending score, then seeded shuffle within equal
    score groups, then ``source_id`` as a final stable key.
    """
    if limit < 0:
        raise ValueError("limit must be >= 0")

    pool = iter_personal_environmental_uniques(
        unique_pool,
        excluded_keys=excluded_keys,
    )
    scored: list[ImplicitMineCandidate] = []
    for item in pool:
        score, reasons = score_implicit_candidate(item.query)
        if score <= 0:
            continue
        # Never mutate labels; echo the original classification only.
        scored.append(
            ImplicitMineCandidate(
                source_id=item.source_query_id,
                query=item.query,
                original_label=item.source_classification_label,
                normalized_query=normalize_query_key(item.query),
                score=score,
                mining_reasons=reasons,
            )
        )

    by_score: dict[int, list[ImplicitMineCandidate]] = {}
    for candidate in scored:
        by_score.setdefault(candidate.score, []).append(candidate)

    ordered: list[ImplicitMineCandidate] = []
    rng = random.Random(seed)
    for score in sorted(by_score.keys(), reverse=True):
        group = list(by_score[score])
        group.sort(key=lambda item: item.source_id)
        rng.shuffle(group)
        ordered.extend(group)

    if limit == 0:
        return ()
    return tuple(ordered[:limit])


def summarize_implicit_mining(
    unique_pool: Sequence[StageACandidate],
    *,
    excluded_keys: Iterable[str] = (),
    limit: int = DEFAULT_IMPLICIT_LIMIT,
    seed: int = DEFAULT_CANDIDATE_SEED,
) -> dict[str, int]:
    """Counts for ``--summary`` without writing files."""
    personal = sum(
        1 for item in unique_pool if item.source_classification_label == "Personal"
    )
    environmental = sum(
        1
        for item in unique_pool
        if item.source_classification_label == "Environmental"
    )
    eligible = iter_personal_environmental_uniques(
        unique_pool,
        excluded_keys=excluded_keys,
    )
    qualifying = 0
    for item in eligible:
        score, _reasons = score_implicit_candidate(item.query)
        if score > 0:
            qualifying += 1
    shortlist = mine_implicit_candidates(
        unique_pool,
        excluded_keys=excluded_keys,
        limit=limit,
        seed=seed,
    )
    return {
        "unique_personal": personal,
        "unique_environmental": environmental,
        "eligible_personal_environmental": len(eligible),
        "satisfying_mining_criteria": qualifying,
        "shortlist_written": len(shortlist),
        "limit": limit,
    }


def write_implicit_candidates_jsonl(
    path: str | Path,
    candidates: Sequence[ImplicitMineCandidate],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(candidate.to_json_dict(), ensure_ascii=False)
        for candidate in candidates
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_implicit_candidates_jsonl(
    path: str | Path,
) -> tuple[ImplicitMineCandidate, ...]:
    path = Path(path)
    records: list[ImplicitMineCandidate] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            records.append(
                ImplicitMineCandidate(
                    source_id=payload["source_id"],
                    query=payload["query"],
                    original_label=payload["original_label"],
                    normalized_query=payload["normalized_query"],
                    score=int(payload["score"]),
                    mining_reasons=tuple(payload["mining_reasons"]),
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"invalid implicit candidate at {path}:{line_number}: {exc}"
            ) from exc
    return tuple(records)


def build_mining_inputs(
    *,
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    reviews_path: str | Path | None = DEFAULT_REVIEWS_PATH,
) -> tuple[tuple[StageACandidate, ...], set[str]]:
    """Load unique pool + exclusion set using existing corpus helpers."""
    rows = load_classification_rows(train_path)
    unique_pool = build_unique_query_pool(rows)
    excluded = excluded_normalized_queries(
        unique_pool=unique_pool,
        reviews_path=reviews_path,
    )
    return unique_pool, excluded


__all__ = [
    "DEFAULT_IMPLICIT_LIMIT",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_REVIEWS_PATH",
    "DEFAULT_TRAIN_PATH",
    "ImplicitMineCandidate",
    "build_mining_inputs",
    "excluded_normalized_queries",
    "iter_personal_environmental_uniques",
    "load_implicit_candidates_jsonl",
    "mine_implicit_candidates",
    "score_implicit_candidate",
    "summarize_implicit_mining",
    "write_implicit_candidates_jsonl",
]
