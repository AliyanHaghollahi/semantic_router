"""H4_REFEXPR_V1: boundary-only referring-expression annotation contract.

H4 controls anchor **boundaries** only. Boundary rewrites must not
automatically change H5 (implicit_resolution) or H6 (owner_operation_index).
Possessive NP boundaries are independent of H5; do not infer
IMPLICIT_RESOLVE_PERSONAL from a possessive span alone.

Rewrite policy:
- SAFE_MECHANICAL: deterministic high-confidence shortenings
  (``this|that the right|correct …`` → bare demonstrative).
- HUMAN_REVIEW detection: detectors may **queue** ambiguous anchors for
  human review. Queuing never decides a label by itself.
- Only **explicit recorded human decisions** (approved R6/R5/final tables,
  or explicit keep such as ``sa_0404``) are applied to annotations.
- No ambiguous case is automatically decided.

TRAIN/DEV under this contract is frozen; TEST annotation migration remains
pending and must stay blind to model outputs/metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from tiergraph.planner.annotation_step_a import StepAAnchor
from tiergraph.planner.annotation_step_b import StepBAnchorDecision

H4_ANCHOR_CONTRACT: Final[str] = "H4_REFEXPR_V1"
SOURCE_CORPUS_V2: Final[str] = "stage_a_v2"

# ---------------------------------------------------------------------------
# Contract rules (documentation + machine-checkable SAFE subset)
# ---------------------------------------------------------------------------

CONTRACT_RULES: Final[dict[str, str]] = {
    "R1": (
        "Standalone demonstrative pronouns: boundary is exact "
        "this/that/these/those when used as a pronoun "
        "(e.g. 'Is this the right terminal?' → 'this'). "
        "Do not mis-apply to determiner NPs ('this tray' is R2)."
    ),
    "R2": (
        "Demonstrative determiner + NP: minimal complete NP "
        "(demonstrative + head + name-level modifiers only). "
        "Keep 'this tray' / 'this elevator'; exclude clause predicates."
    ),
    "R3": (
        "Possessive referring NPs (boundary only): possessive + full NP "
        "('my ticket'). H5 is independent of the possessive boundary."
    ),
    "R4": (
        "Locative / anaphoric expressions: here/there; 'the one …' only when "
        "that anaphor is the intended entity. Preserve H5/H6."
    ),
        "R5": (
            "Function-word / non-referring anchors (Does/What/Which/where/…): "
            "not valid H4 referring expressions. Detectors queue such cases for "
            "HUMAN_REVIEW; only an explicit recorded human decision may remove "
            "or replace them. Always revalidate H5/H6 and graph after any edit."
        ),
    "R6": (
        "Predicate material outside the anchor: under R1 exclude "
        "'the right/correct X' from the span; under R2 exclude clause "
        "predicates not part of the entity name. SAFE_MECHANICAL covers only "
        "the unambiguous this|that + the + right|correct merge."
    ),
}

# Unambiguous pronoun-vs-determiner merge: "this the right terminal" → "this"
SAFE_MECHANICAL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(this|that)\s+the\s+(right|correct)\b.*$",
    re.IGNORECASE,
)

# HUMAN_REVIEW detectors: queue candidates for human judgment only.
# Detection never applies a rewrite by itself.
_FUNC_WORD: Final[re.Pattern[str]] = re.compile(
    r"^(where|which|what|who|does|do|did|is|are|am|can|will|how)$",
    re.IGNORECASE,
)
_PRED_TAIL: Final[re.Pattern[str]] = re.compile(
    r"\b(safe|compatible|visible|listed|delayed|taken|crowded|quiet|"
    r"organized|busy|correct|right)\s*$",
    re.IGNORECASE,
)
_STATE_IN: Final[re.Pattern[str]] = re.compile(
    r"\bon or off\b|\bbusy or quiet\b",
    re.IGNORECASE,
)
_ELEV_GOING: Final[re.Pattern[str]] = re.compile(
    r"^(this|that)\s+\w+(?:\s+\w+){0,3}\s+going\b",
    re.IGNORECASE,
)
_THIS_THE_OTHER: Final[re.Pattern[str]] = re.compile(
    r"^(this|that)\s+the\s+(?!right\b|correct\b)",
    re.IGNORECASE,
)

# Frozen Phase-1 ID sets (TRAIN/DEV only; TEST unused)
SAFE_MECHANICAL_TRAIN_IDS: Final[frozenset[str]] = frozenset(
    {
        "sa_0263",
        "sa_0342",
        "sa_0369",
        "sa_0371",
        "sa_0383",
        "sa_0398",
        "sa_0399",
        "sa_0405",
    }
)
SAFE_MECHANICAL_DEV_IDS: Final[frozenset[str]] = frozenset({"sa_0393"})
SAFE_MECHANICAL_IDS: Final[frozenset[str]] = (
    SAFE_MECHANICAL_TRAIN_IDS | SAFE_MECHANICAL_DEV_IDS
)

HUMAN_REVIEW_TRAIN_IDS: Final[frozenset[str]] = frozenset(
    {
        "sa_0138",
        "sa_0141",
        "sa_0160",
        "sa_0166",
        "sa_0167",
        "sa_0168",
        "sa_0265",
        "sa_0267",
        "sa_0269",
        "sa_0288",
        "sa_0314",
        "sa_0322",
        "sa_0330",
        "sa_0339",
        "sa_0456",
    }
)
HUMAN_REVIEW_DEV_IDS: Final[frozenset[str]] = frozenset(
    {
        "sa_0216",
        "sa_0237",
        "sa_0262",
        "sa_0404",
        "sa_0464",
    }
)
HUMAN_REVIEW_IDS: Final[frozenset[str]] = (
    HUMAN_REVIEW_TRAIN_IDS | HUMAN_REVIEW_DEV_IDS
)

# ---------------------------------------------------------------------------
# Explicit recorded human decisions (TRAIN/DEV only; frozen).
# ---------------------------------------------------------------------------

# R6: pure boundary shortenings; preserve H5/H6.
APPROVED_R6_SHORTEN: Final[dict[str, tuple[str, str]]] = {
    "sa_0237": ("the TV screen on or off", "the TV screen"),
    "sa_0265": ("this plated dish safe", "this plated dish"),
    "sa_0267": ("this soup compatible", "this soup"),
    "sa_0269": ("this salad dressing safe", "this salad dressing"),
    "sa_0288": (
        "this pastry fit the egg-free restriction listed",
        "this pastry",
    ),
    "sa_0314": (
        "this will-call desk the pickup point listed",
        "this will-call desk",
    ),
    "sa_0322": (
        "this pharmacy sticker show the frequency listed",
        "this pharmacy sticker",
    ),
    "sa_0330": (
        "this track sign match the platform listed",
        "this track sign",
    ),
}

# R5: remove flagged non-referring function-word anchors entirely.
APPROVED_R5_REMOVE: Final[dict[str, str]] = {
    "sa_0138": "What",
    "sa_0141": "What",
    "sa_0160": "What",
    "sa_0166": "What",
    "sa_0167": "What",
    "sa_0168": "What",
    "sa_0216": "Which",
}

# R4: keep current anaphor unchanged for now.
APPROVED_R4_KEEP: Final[frozenset[str]] = frozenset({"sa_0404"})

# Explicit human finalizations (offsets must match query exactly).
# Each entry replaces ``old_text`` with one or more new anchors (may relocate).
APPROVED_FINAL_REPLACE: Final[dict[str, dict[str, object]]] = {
    "sa_0262": {
        "old_text": "Does",
        "new_anchors": (
            {
                "text": "it",
                "char_start": 5,
                "char_end": 7,
                "h5": "NONE",
                "h6": 0,
            },
        ),
    },
    "sa_0339": {
        "old_text": "this bus going to my hotel and",
        "new_anchors": (
            {
                "text": "this bus",
                "char_start": 3,
                "char_end": 11,
                "h5": "NONE",
                "h6": 0,
            },
            {
                "text": "my hotel",
                "char_start": 21,
                "char_end": 29,
                "h5": "IMPLICIT_RESOLVE_PERSONAL",
                "h6": 0,
            },
        ),
    },
    "sa_0456": {
        "old_text": "which",
        "new_anchors": (
            {
                "text": "bottle",
                "char_start": 29,
                "char_end": 35,
                "h5": "IMPLICIT_RESOLVE_PERSONAL",
                "h6": 0,
            },
        ),
    },
    "sa_0464": {
        "old_text": "where",
        "new_anchors": (
            {
                "text": "the assigned exam room",
                "char_start": 31,
                "char_end": 53,
                "h5": "IMPLICIT_RESOLVE_PERSONAL",
                "h6": 0,
            },
        ),
    },
}

# No remaining unresolved HUMAN_REVIEW cases after finalization.
UNRESOLVED_HUMAN_REVIEW_IDS: Final[frozenset[str]] = frozenset()

APPROVED_EDIT_IDS: Final[frozenset[str]] = frozenset(
    set(APPROVED_R6_SHORTEN)
    | set(APPROVED_R5_REMOVE)
    | set(APPROVED_FINAL_REPLACE)
)

AFFECTED_TRAIN_IDS: Final[frozenset[str]] = (
    SAFE_MECHANICAL_TRAIN_IDS | HUMAN_REVIEW_TRAIN_IDS
)
AFFECTED_DEV_IDS: Final[frozenset[str]] = (
    SAFE_MECHANICAL_DEV_IDS | HUMAN_REVIEW_DEV_IDS
)


@dataclass(frozen=True, slots=True)
class HumanReviewFlag:
    """One HUMAN_REVIEW queue entry from detector output.

    Queued for human judgment; not an automatic rewrite instruction.
    """

    stage_a_id: str
    split: str
    query: str
    anchor_index: int
    existing_anchor: str
    char_start: int
    char_end: int
    flagged_issue: str
    rule: str
    h5: str
    h6_owner: int | None
    reason_human_review_required: str


def is_safe_mechanical_anchor_text(text: str) -> bool:
    return SAFE_MECHANICAL_PATTERN.match(" ".join(text.split())) is not None


def classify_human_review_reasons(text: str) -> tuple[str, ...]:
    """Return rule tags that put this gold anchor on the HUMAN_REVIEW queue."""
    t = " ".join(text.split())
    tl = t.lower()
    if is_safe_mechanical_anchor_text(t):
        return ()
    reasons: list[str] = []
    if _FUNC_WORD.match(tl):
        reasons.append("R5_func")
    if _THIS_THE_OTHER.match(t):
        reasons.append("this_the_non_right")
    if tl.startswith(("this ", "that ")) and _PRED_TAIL.search(tl):
        reasons.append("pred_tail")
    if _STATE_IN.search(tl):
        reasons.append("state_in_np")
    if _ELEV_GOING.match(tl):
        reasons.append("elevator_going")
    if tl == "the one" or tl.startswith("the one "):
        reasons.append("the_one")
    return tuple(reasons)


def primary_review_rule(reasons: tuple[str, ...]) -> str:
    if not reasons:
        return "unknown"
    if "R5_func" in reasons:
        return "R5"
    if "elevator_going" in reasons:
        return "R6"
    if "pred_tail" in reasons or "this_the_non_right" in reasons:
        return "R6"
    if "state_in_np" in reasons:
        return "R6"
    if "the_one" in reasons:
        return "R4"
    return reasons[0]


def shorten_safe_mechanical_anchor(
    anchor: StepAAnchor,
    query: str,
) -> StepAAnchor:
    """Shorten SAFE_MECHANICAL gold text to the demonstrative token only."""
    text = " ".join(anchor.text.split())
    match = SAFE_MECHANICAL_PATTERN.match(text)
    if match is None:
        raise ValueError(
            f"anchor text is not SAFE_MECHANICAL: {anchor.text!r}"
        )
    new_text = match.group(1)
    if query[anchor.char_start : anchor.char_start + len(new_text)] != new_text:
        raise ValueError(
            f"query slice does not start with demonstrative {new_text!r} "
            f"at {anchor.char_start} for text {anchor.text!r}"
        )
    new_end = anchor.char_start + len(new_text)
    if query[anchor.char_start:new_end] != new_text:
        raise ValueError(
            f"shortened span mismatch: expected {new_text!r}, "
            f"got {query[anchor.char_start:new_end]!r}"
        )
    return anchor.model_copy(update={"text": new_text, "char_end": new_end})


def shorten_anchor_to_text(
    anchor: StepAAnchor,
    query: str,
    new_text: str,
) -> StepAAnchor:
    """Shorten an anchor to ``new_text`` at the same ``char_start`` (prefix)."""
    if not new_text:
        raise ValueError("new_text must be nonempty")
    if query[anchor.char_start : anchor.char_start + len(new_text)] != new_text:
        raise ValueError(
            f"query does not contain {new_text!r} at {anchor.char_start} "
            f"(old text {anchor.text!r})"
        )
    if not anchor.text.startswith(new_text):
        raise ValueError(
            f"new_text {new_text!r} is not a prefix of old text {anchor.text!r}"
        )
    new_end = anchor.char_start + len(new_text)
    return anchor.model_copy(update={"text": new_text, "char_end": new_end})


def sync_step_b_text_preserve_h5_h6(
    decision: StepBAnchorDecision,
    new_text: str,
) -> StepBAnchorDecision:
    """Update audit text only; leave H5/H6 untouched."""
    return decision.model_copy(
        update={
            "text": new_text,
            "implicit_resolution": decision.implicit_resolution,
            "owner_operation_index": decision.owner_operation_index,
        }
    )


def is_determiner_np_demo(text: str) -> bool:
    """True when demonstrative is used as determiner of a fuller NP (R2)."""
    t = " ".join(text.split())
    if is_safe_mechanical_anchor_text(t):
        return False
    return bool(re.match(r"^(this|that|these|those)\s+\S", t, re.I))


def is_bare_demonstrative_pronoun(text: str) -> bool:
    return bool(re.fullmatch(r"(?i)this|that|these|those", " ".join(text.split())))


def is_possessive_np(text: str) -> bool:
    return bool(re.match(r"(?i)^(my|your|his|her|our|their)\s+\S", " ".join(text.split())))


__all__ = [
    "AFFECTED_DEV_IDS",
    "AFFECTED_TRAIN_IDS",
    "APPROVED_EDIT_IDS",
    "APPROVED_FINAL_REPLACE",
    "APPROVED_R4_KEEP",
    "APPROVED_R5_REMOVE",
    "APPROVED_R6_SHORTEN",
    "CONTRACT_RULES",
    "H4_ANCHOR_CONTRACT",
    "HUMAN_REVIEW_DEV_IDS",
    "HUMAN_REVIEW_IDS",
    "HUMAN_REVIEW_TRAIN_IDS",
    "HumanReviewFlag",
    "SAFE_MECHANICAL_DEV_IDS",
    "SAFE_MECHANICAL_IDS",
    "SAFE_MECHANICAL_PATTERN",
    "SAFE_MECHANICAL_TRAIN_IDS",
    "SOURCE_CORPUS_V2",
    "UNRESOLVED_HUMAN_REVIEW_IDS",
    "classify_human_review_reasons",
    "is_bare_demonstrative_pronoun",
    "is_determiner_np_demo",
    "is_possessive_np",
    "is_safe_mechanical_anchor_text",
    "primary_review_rule",
    "shorten_anchor_to_text",
    "shorten_safe_mechanical_anchor",
    "sync_step_b_text_preserve_h5_h6",
]
