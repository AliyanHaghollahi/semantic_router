"""Stage-A v2 Step-A annotations: copy legacy 120 + agent-assisted 360 new.

Step A only (H2/H3/H4). Does not write Step B, splits, or train.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tiergraph.enums import OperatorType
from tiergraph.planner.annotation_step_a import (
    StageAStepAAnnotation,
    StepAAnchor,
    StepAOperation,
    StepAStatus,
    derive_query_type,
    load_step_a_annotations,
    write_step_a_annotations,
)
from tiergraph.planner.corpus import normalize_query_key
from tiergraph.planner.stage_a_selection import load_jsonl
from tiergraph.planner.stage_a_v2_spec import (
    STAGE_A_V1_STEP_A_PATH,
    STAGE_A_V2_CORPUS_SIZE,
    STAGE_A_V2_SELECTION_PATH,
    STAGE_A_V2_STEP_A_PATH,
    STAGE_A_V2_STEP_A_REPORT_PATH,
)


EXPECTED_V2_STEP_A_COUNT = STAGE_A_V2_CORPUS_SIZE

# Telephony/action requests: no V1 explicit Step-A answer operator.
_ONTOLOGY_BLOCKED_SOURCE_IDS = frozenset({"src_0033", "src_0034"})

# Explicit answer ops allowed for v2 Step-A gold (no RESOLVE_PERSONAL / FUSE).
_ALLOWED_OPS = frozenset(
    {
        OperatorType.RETRIEVE_PERSONAL,
        OperatorType.IDENTIFY_ENVIRONMENTAL,
        OperatorType.LOCATE_ENVIRONMENTAL,
        OperatorType.NAVIGATE_TO,
        OperatorType.DESCRIBE_ENVIRONMENT,
    }
)

# Clause boundaries for parallel/sequential multi-op surfaces.
_CLAUSE_SEP = re.compile(
    r"(?:"
    r",\s+then\s+"
    r"|\s+then\s+"
    r"|,\s+and\s+"
    r"|;\s+"
    r"|,\s+(?="
    r"(?:locate|find|identify|name|navigate|guide|give|where|what|how|"
    r"describe|read|does|is|am|do)\b"
    r")"
    r"|\s+and\s+(?="
    r"(?:locate|find|identify|name|navigate|guide|give|where|what|how|"
    r"describe|read|does|is|am|do|did|tell|will|can|check|when|have|are|"
    r"tell me)\b"
    r")"
    r"|\?\s+(?=[A-Z])"
    r")",
    re.IGNORECASE,
)

_DISTRACTOR_PREFIX = re.compile(
    r"^(?:"
    r"urgent:\s*(?:skip(?:\s+the)?\s+(?:side\s+)?chatter\s+and\s+)?"
    r"|asap:\s*"
    r"|ignore(?:\s+the)?\s+(?:noise|chatter|distraction|alerts?)\s+and\s+"
    r"|right now,\s+setting aside[^,?]+,\s+"
    r"|immediately,\s+without focusing on[^,?]+,\s+"
    r"|tell me right now,\s+"
    r")",
    re.IGNORECASE,
)

_URGENCY_TAIL = re.compile(
    r"(?:"
    r"\?\s*(?:"
    r"tell me now"
    r"|let me know(?: right now| immediately)?"
    r"|i need to know(?: this second| now| at once| immediately)?"
    r"|i need an (?:instant|immediate) (?:assessment|answer|response)"
    r"|answer immediately"
    r"|report (?:at once|this second)"
    r"|give me an urgent update"
    r"|your immediate attention is required"
    r"|immediate attention required"
    r")\.?"
    r"|(?:,\s*)?(?:"
    r"tell me now"
    r"|let me know(?: right now| immediately)?"
    r"|i need to know(?: this second| now| at once| immediately)?"
    r"|i need an (?:instant|immediate) (?:assessment|answer|response)"
    r"|answer immediately"
    r"|report (?:at once|this second)"
    r"|give me an urgent update"
    r"|your immediate attention is required"
    r"|immediate attention required"
    r")\.?"
    r")\s*$",
    re.IGNORECASE,
)

_ANCHOR_PATTERNS = (
    re.compile(
        r"\b(?:my|your)\s+(?:[\w'-]+(?:\s+[\w'-]+){0,4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:this|that|these|those)\s+(?:[\w'-]+(?:\s+[\w'-]+){0,5})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:this|that|there|here)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bthe\s+(?:[\w'-]+(?:\s+[\w'-]+){0,4})",
        re.IGNORECASE,
    ),
)

_NAV_CUES = re.compile(
    r"\b(navigat|how do i (get|walk|reach)|walking directions|guide me|"
    r"directions from|walk from|get from|get in line|navigation steps|"
    r"give navigation|guide me there|walk to it|reach it)\b",
    re.IGNORECASE,
)
_LOC_CUES = re.compile(
    r"\b(where (is|are|along|does)|locate|located|find (the|that|its|this)|"
    r"from here|relative to|where does)\b",
    re.IGNORECASE,
)
_DESC_CUES = re.compile(
    r"\b(describe|what does .+ say|read (the|this|item)|look(?:s)?|status|"
    r"crowded|cluttered|lighting|condition|wear|warning text|message|"
    r"organized|clear or cloudy|on or off|busy)\b",
    re.IGNORECASE,
)
_ID_CUES = re.compile(
    r"\b(identify|what (is|are) the name|what model|name this|what (device|machine|"
    r"product|item|kind|plaque|dish|special)|which |is this |am i holding|"
    r"what is this|what's the model)\b",
    re.IGNORECASE,
)
_RET_CUES = re.compile(
    r"\b(what is my|what are my|do i have|am i |did i |when (does|is) my|"
    r"what time (do|is) my|what pharmacy|what seat|read my|"
    r"find the contact|what documents|what food am i|how much do i|"
    r"how many bags|is my |what points|what conditions|what did i|"
    r"will i make my)\b",
    re.IGNORECASE,
)

# Sequential context preambles are not standalone answer operations.
_PREAMBLE_RES = re.compile(
    r"^(?:using|based on)\s+my\s+[\w'-]+(?:\s+[\w'-]+){0,4},\s+",
    re.IGNORECASE,
)

# Legacy implicit match semantics (sa_0054 DESCRIBE vs sa_0052 IDENTIFY).
_DESCRIBE_MATCH_CUES = re.compile(
    r"\b("
    r"label|dosage|prescription|sticker|frequency|concentration|strength|"
    r"unit-dose|mg amount|room number|room plaque|wall directory|"
    r"floor map|lobby screen|suite directory|flight status panel|"
    r"nutritional information|ingredients|warning text|departure board"
    r")\b",
    re.IGNORECASE,
)


def _content_span(query: str) -> tuple[int, int]:
    """Largest nonempty span excluding trailing punctuation/spaces."""
    end = len(query)
    while end > 0 and query[end - 1] in "?.!;: ":
        end -= 1
    start = 0
    while start < end and query[start] in " ":
        start += 1
    if start >= end:
        raise ValueError(f"empty content span for query: {query!r}")
    return start, end


def _trim_operation_span(query: str, start: int, end: int) -> tuple[int, int]:
    """Trim edge whitespace / trailing punct inside a clause span."""
    while start < end and query[start] in " ":
        start += 1
    while end > start and query[end - 1] in "?.!;: ":
        end -= 1
    if start >= end:
        raise ValueError(f"empty operation span after trim in {query!r}")
    return start, end


def _is_ontology_blocked_row(row: Mapping[str, Any]) -> bool:
    """True when the selected query has no valid V1 Step-A answer operator."""
    source_id = str(row.get("source_id") or "")
    if source_id in _ONTOLOGY_BLOCKED_SOURCE_IDS:
        return True
    template = str(row.get("template_group") or "")
    query = str(row.get("query") or "").strip().lower()
    return template == "call_my_X" or query.startswith("call my ")


def _implicit_match_operator(low: str) -> OperatorType | None:
    """Classify single-surface implicit match/list questions per legacy v1."""
    if _DESCRIBE_MATCH_CUES.search(low):
        return OperatorType.DESCRIBE_ENVIRONMENT
    if any(
        token in low
        for token in (
            "safe",
            "allerg",
            "compatible",
            "trigger",
            "restriction",
            "entrance",
            "doorway",
            "door lead",
            "door match",
            "gate board",
            "departure screen",
            "boarding-zone",
            "jet-bridge",
            "pickup",
            "locker",
            "parcel",
            "counter the",
            "hatch match",
            "bag rack",
            "will-call desk",
            "seat match",
            "shopping list",
            "medication on my list",
        )
    ):
        return OperatorType.IDENTIFY_ENVIRONMENTAL
    return None


def _split_clauses(query: str) -> list[tuple[int, int]]:
    """Split multi-clause queries into non-overlapping content spans."""
    core_start, core_end = _content_span(query)
    # Strip sequential context preambles (not answer operations).
    preamble = _PREAMBLE_RES.match(query[core_start:core_end])
    if preamble:
        core_start = core_start + preamble.end()
        while core_start < core_end and query[core_start] in " ":
            core_start += 1
    # Strip urgency/distractor preamble before clause detection.
    prefix = _DISTRACTOR_PREFIX.match(query[core_start:core_end])
    if prefix:
        core_start = core_start + prefix.end()
        while core_start < core_end and query[core_start] in " ":
            core_start += 1
    # Strip trailing urgency imperative tails (not answer operations).
    tail = _URGENCY_TAIL.search(query[core_start:core_end])
    if tail:
        core_end = core_start + tail.start()
        while core_end > core_start and query[core_end - 1] in "?.!;: ":
            core_end -= 1
    if core_start >= core_end:
        raise ValueError(f"empty content after distractor strip: {query!r}")
    core = query[core_start:core_end]

    parts: list[tuple[int, int]] = []
    cursor = 0
    for match in _CLAUSE_SEP.finditer(core):
        left = core[cursor : match.start()].strip()
        if left:
            abs_start = core_start + cursor + core[cursor : match.start()].find(left)
            parts.append((abs_start, abs_start + len(left)))
        cursor = match.end()
    right = core[cursor:].strip()
    if right:
        abs_start = core_start + cursor + core[cursor:].find(right)
        parts.append((abs_start, abs_start + len(right)))

    cleaned: list[tuple[int, int]] = []
    last_end = -1
    for start, end in parts:
        start, end = _trim_operation_span(query, start, end)
        if start < last_end:
            cleaned = []
            break
        cleaned.append((start, end))
        last_end = end

    if len(cleaned) >= 2:
        return cleaned

    start, end = _trim_operation_span(query, core_start, core_end)
    return [(start, end)]


def _infer_operator(
    clause: str,
    *,
    hint: str | None,
    bucket: str,
    clause_index: int,
    n_clauses: int,
) -> OperatorType:
    text = clause.strip()
    low = text.lower()

    if low.startswith("what time is it"):
        return OperatorType.DESCRIBE_ENVIRONMENT

    # Pure Personal bucket: answer ops are RETRIEVE unless an explicit navigate cue.
    if bucket == "Personal":
        if _NAV_CUES.search(text) or low.startswith("navigate "):
            return OperatorType.NAVIGATE_TO
        return OperatorType.RETRIEVE_PERSONAL

    implicit_op = _implicit_match_operator(low)
    if implicit_op is not None and bucket == "MIXED_IMPLICIT":
        return implicit_op

    if hint is not None:
        try:
            hinted = OperatorType(hint)
            if hinted in _ALLOWED_OPS:
                # Trust hint when cues are compatible or weak.
                if hinted is OperatorType.NAVIGATE_TO and _NAV_CUES.search(text):
                    return hinted
                if hinted is OperatorType.LOCATE_ENVIRONMENTAL and (
                    _LOC_CUES.search(text) or "where" in low
                ):
                    return hinted
                if hinted is OperatorType.DESCRIBE_ENVIRONMENT and (
                    _DESC_CUES.search(text) or low.startswith("describe")
                ):
                    return hinted
                if hinted is OperatorType.IDENTIFY_ENVIRONMENTAL and (
                    _ID_CUES.search(text)
                    or low.startswith(("identify", "name "))
                    or "what is the name" in low
                    or "what model" in low
                ):
                    return hinted
                if hinted is OperatorType.RETRIEVE_PERSONAL and (
                    _RET_CUES.search(text) or "my " in low
                ):
                    return hinted
                # Multi-clause authored: assign hints in order when cues soft.
                if n_clauses > 1:
                    return hinted
        except ValueError:
            pass

    if _NAV_CUES.search(text) or low.startswith("navigate "):
        return OperatorType.NAVIGATE_TO
    if _LOC_CUES.search(text) or low.startswith("where "):
        return OperatorType.LOCATE_ENVIRONMENTAL
    if low.startswith("describe ") or (
        _DESC_CUES.search(text)
        and not low.startswith(("what is the name", "identify", "name this"))
    ):
        return OperatorType.DESCRIBE_ENVIRONMENT
    # Personal stored-fact cues inside mixed buckets.
    if (
        _RET_CUES.search(text)
        and "this " not in low[:20]
        and ("my " in low or low.startswith(("am i ", "do i ", "did i ", "call my")))
    ):
        return OperatorType.RETRIEVE_PERSONAL
    if _ID_CUES.search(text) or low.startswith(
        ("identify", "name this", "what is this", "is this ", "is there ", "is it ")
    ):
        return OperatorType.IDENTIFY_ENVIRONMENTAL
    if "read " in low or "say" in low or "look" in low:
        return OperatorType.DESCRIBE_ENVIRONMENT
    if bucket in {"Environmental", "MIXED_IMPLICIT", "MIXED_SEQUENTIAL", "MIXED_PARALLEL"}:
        return OperatorType.IDENTIFY_ENVIRONMENTAL
    return OperatorType.RETRIEVE_PERSONAL


def _trim_anchor_text(text: str) -> str:
    cleaned = text.strip().rstrip("?,.;:!")
    # Drop trailing clause glue accidentally captured.
    for stopper in (
        " and ",
        " then ",
        " given ",
        " saved ",
        " listed ",
        " for my ",
        " on my ",
        " in my ",
        " from my ",
        " with my ",
        " relative ",
        " located ",
        " whether ",
        " that ",
        " matching ",
        " linked ",
        " shown ",
        " printed ",
    ):
        idx = cleaned.lower().find(stopper)
        if idx > 0:
            cleaned = cleaned[:idx].rstrip()
    # Prefer keeping hyphenated compounds intact (already in [\w'-]).
    return cleaned


def _find_anchors(query: str, op_spans: Sequence[tuple[int, int]]) -> list[StepAAnchor]:
    """Collect non-overlapping referential phrases intersecting ops."""
    candidates: list[tuple[int, int, str]] = []
    for pattern in _ANCHOR_PATTERNS:
        for match in pattern.finditer(query):
            raw = match.group(0)
            trimmed = _trim_anchor_text(raw)
            if not trimmed:
                continue
            # Bare deictics are allowed when genuine referents; skip "it".
            if trimmed.lower() in {"it", "i"}:
                continue
            # Re-locate trimmed text inside the match window.
            rel = query.find(trimmed, match.start(), match.end() + 1)
            if rel < 0:
                rel = query.find(trimmed, match.start())
            if rel < 0:
                continue
            start, end = rel, rel + len(trimmed)
            if query[start:end] != trimmed:
                continue
            candidates.append((start, end, trimmed))

    # Prefer longer spans; drop overlaps.
    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    chosen: list[tuple[int, int, str]] = []
    has_rich_deictic = any(
        t.lower().startswith(("my ", "this ", "that ", "these ", "those "))
        for _, _, t in candidates
    )
    for start, end, text in candidates:
        if any(start < ce and cs < end for cs, ce, _ in chosen):
            continue
        # Must intersect some operation span.
        if not any(start < oe and os < end for os, oe in op_spans):
            continue
        low = text.lower()
        # Skip bare "the X" when richer deictic/possessive anchors exist.
        if low.startswith("the ") and has_rich_deictic:
            continue
        # Skip bare this/that/there/here when a longer this/that NP exists.
        if low in {"this", "that", "there", "here"} and any(
            t.lower().startswith(low + " ") for _, _, t in candidates
        ):
            continue
        chosen.append((start, end, text))

    chosen.sort(key=lambda item: (item[0], item[1]))
    # Fallback: if no anchors found, take a noun-ish token from first op.
    if not chosen and op_spans:
        os, oe = op_spans[0]
        fragment = query[os:oe]
        fallback = re.search(
            r"\b(?:my|this|that)\s+[\w'-]+(?:\s+[\w'-]+){0,3}",
            fragment,
            re.IGNORECASE,
        )
        if fallback:
            abs_start = os + fallback.start()
            text = _trim_anchor_text(fallback.group(0))
            abs_start = query.find(text, abs_start)
            if abs_start >= 0:
                chosen.append((abs_start, abs_start + len(text), text))
        elif re.search(r"\b[\w'-]{4,}\b", fragment):
            m = re.search(r"\b[\w'-]{4,}\b", fragment)
            assert m is not None
            abs_start = os + m.start()
            text = m.group(0)
            chosen.append((abs_start, abs_start + len(text), text))

    return [
        StepAAnchor(
            anchor_index=i,
            text=text,
            char_start=start,
            char_end=end,
        )
        for i, (start, end, text) in enumerate(chosen)
    ]


def _hints_for_row(row: Mapping[str, Any], n_clauses: int) -> list[str | None]:
    family = list(row.get("operator_family") or [])
    if not family:
        return [None] * n_clauses
    if n_clauses == 1:
        # Single surface op: do not force multi-family onto one span.
        # Prefer the first family member as a soft hint only.
        return [family[0]]
    # Align hints left-to-right; pad/truncate.
    hints: list[str | None] = list(family[:n_clauses])
    while len(hints) < n_clauses:
        hints.append(family[-1] if family else None)
    return hints


def annotate_new_row(row: Mapping[str, Any]) -> tuple[StageAStepAAnnotation, dict[str, Any]]:
    """Agent-assisted Step-A annotation for one NEW selection row."""
    query = str(row["query"])
    bucket = str(row["final_bucket"])
    flags: list[str] = []

    if _is_ontology_blocked_row(row):
        provenance = dict(row.get("provenance") or {})
        provenance = {
            **provenance,
            "step_a_origin": "stage_a_v2_step_a_agent_assisted",
            "step_a_method": "agent_assisted_rule_guided",
            "ontology_incompatible_v1": True,
            "replacement_required": True,
            "replacement_reason": (
                "telephony/action request ('Call my X'); no V1 explicit Step-A "
                "answer operator (RETRIEVE is for stored facts, not dial actions)"
            ),
        }
        record = StageAStepAAnnotation(
            stage_a_id=str(row["stage_a_id"]),
            source_id=row.get("source_id"),
            candidate_id=row.get("candidate_id"),
            query=query,
            final_bucket=bucket,
            source_kind=str(row["source_kind"]),
            semantic_group=str(row["semantic_group"]),
            template_group=str(row["template_group"]),
            provenance=provenance,
            derived_query_type=derive_query_type(bucket),
            operations=(),
            anchors=(),
            step_a_status=StepAStatus.UNREVIEWED,
        )
        meta = {
            "stage_a_id": record.stage_a_id,
            "flags": ["ontology_incompatible_v1"],
            "n_operations": 0,
            "n_anchors": 0,
            "operator_types": [],
            "ontology_blocked": True,
        }
        return record, meta

    family = list(row.get("operator_family") or [])
    clause_spans = _split_clauses(query)
    hints = _hints_for_row(row, len(clause_spans))

    operations: list[StepAOperation] = []
    for index, ((start, end), hint) in enumerate(zip(clause_spans, hints)):
        clause = query[start:end]
        op_type = _infer_operator(
            clause,
            hint=hint,
            bucket=bucket,
            clause_index=index,
            n_clauses=len(clause_spans),
        )
        if hint is not None and op_type.value != hint:
            if len(clause_spans) > 1:
                flags.append(
                    f"operator_hint_mismatch:hint={hint}:annotated={op_type.value}"
                )
            elif family and op_type.value not in family:
                flags.append(
                    f"operator_hint_mismatch:hint={hint}:annotated={op_type.value}"
                )
            elif (
                family
                and len(family) > 1
                and op_type.value != family[0]
                and bucket == "MIXED_IMPLICIT"
            ):
                # Implicit multi-family: surface op may legitimately differ from
                # the primary authored label; flag for review.
                flags.append(
                    f"operator_hint_mismatch:hint={family[0]}:annotated={op_type.value}"
                )
        operations.append(
            StepAOperation(
                operation_index=index,
                text=clause,
                char_start=start,
                char_end=end,
                operator_type=op_type,
            )
        )

    # Sequential under-split vs authored family: attempt family-length aware resplit.
    if (
        bucket == "MIXED_SEQUENTIAL"
        and len(family) >= 3
        and len(operations) < len(family)
    ):
        # Force separator-aware split already ran; if still short, flag.
        flags.append(
            f"undersegmented_vs_family:ops={len(operations)}:family={len(family)}"
        )

    anchors = _find_anchors(query, [(op.char_start, op.char_end) for op in operations])

    provenance = dict(row.get("provenance") or {})
    provenance = {
        **provenance,
        "step_a_origin": "stage_a_v2_step_a_agent_assisted",
        "step_a_method": "agent_assisted_rule_guided",
    }
    if family:
        provenance["operator_family"] = family
    if flags:
        provenance["step_a_flags"] = flags

    record = StageAStepAAnnotation(
        stage_a_id=str(row["stage_a_id"]),
        source_id=row.get("source_id"),
        candidate_id=row.get("candidate_id"),
        query=query,
        final_bucket=bucket,
        source_kind=str(row["source_kind"]),
        semantic_group=str(row["semantic_group"]),
        template_group=str(row["template_group"]),
        provenance=provenance,
        derived_query_type=derive_query_type(bucket),
        operations=tuple(operations),
        anchors=tuple(anchors),
        step_a_status=StepAStatus.COMPLETE,
    )
    meta = {
        "stage_a_id": record.stage_a_id,
        "flags": flags,
        "n_operations": len(operations),
        "n_anchors": len(anchors),
        "operator_types": [op.operator_type.value for op in operations],
    }
    return record, meta


def _legacy_record_from_v1(item: StageAStepAAnnotation) -> StageAStepAAnnotation:
    """Exact semantic copy of a legacy Step-A annotation."""
    return StageAStepAAnnotation.model_validate(
        json.loads(json.dumps(item.model_dump(mode="json"), ensure_ascii=False))
    )


def validate_step_a_v2_corpus(
    records: Sequence[StageAStepAAnnotation],
    *,
    selection_path: str | Path = STAGE_A_V2_SELECTION_PATH,
    require_all_complete: bool = True,
) -> list[str]:
    """Validate 480-row v2 Step-A corpus against v2 selection."""
    # Reuse v1 validator structure with patched expected count via temporary monkeypatch-free logic.
    errors: list[str] = []
    selection_rows = load_jsonl(selection_path)
    if len(records) != EXPECTED_V2_STEP_A_COUNT:
        errors.append(f"annotation count {len(records)} != {EXPECTED_V2_STEP_A_COUNT}")
    if len(selection_rows) != EXPECTED_V2_STEP_A_COUNT:
        errors.append(
            f"selection count {len(selection_rows)} != {EXPECTED_V2_STEP_A_COUNT}"
        )

    frozen_by_id = {str(row["stage_a_id"]): row for row in selection_rows}
    ann_by_id = {item.stage_a_id: item for item in records}
    if sorted(ann_by_id) != sorted(frozen_by_id):
        errors.append("stage_a_id set does not match v2 selection")

    query_keys = [normalize_query_key(item.query) for item in records]
    if len(query_keys) != len(set(query_keys)):
        errors.append("duplicate normalized queries in annotations")

    for stage_a_id, frozen in frozen_by_id.items():
        item = ann_by_id.get(stage_a_id)
        if item is None:
            errors.append(f"missing annotation for {stage_a_id}")
            continue
        for field in (
            "query",
            "final_bucket",
            "source_kind",
            "semantic_group",
            "template_group",
        ):
            if getattr(item, field) != frozen.get(field):
                errors.append(f"{stage_a_id}: {field} differs from selection")
        if item.source_id != frozen.get("source_id"):
            errors.append(f"{stage_a_id}: source_id differs from selection")
        if item.candidate_id != frozen.get("candidate_id"):
            errors.append(f"{stage_a_id}: candidate_id differs from selection")
        prov = item.provenance or {}
        ontology_blocked = bool(prov.get("ontology_incompatible_v1"))
        if ontology_blocked:
            if item.operations or item.anchors:
                errors.append(
                    f"{stage_a_id}: ontology-blocked row must have empty ops/anchors"
                )
            if item.step_a_status is not StepAStatus.UNREVIEWED:
                errors.append(f"{stage_a_id}: ontology-blocked row must be UNREVIEWED")
            continue
        if require_all_complete and item.step_a_status is not StepAStatus.COMPLETE:
            errors.append(f"{stage_a_id}: not COMPLETE")
        try:
            # model_validate already checks spans for COMPLETE
            StageAStepAAnnotation.model_validate(item.model_dump(mode="python"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{stage_a_id}: {exc}")
        for op in item.operations:
            if op.operator_type not in _ALLOWED_OPS:
                errors.append(
                    f"{stage_a_id}: disallowed operator {op.operator_type.value}"
                )
    return errors


def build_stage_a_v2_step_a(
    *,
    selection_path: str | Path = STAGE_A_V2_SELECTION_PATH,
    legacy_step_a_path: str | Path = STAGE_A_V1_STEP_A_PATH,
) -> tuple[list[StageAStepAAnnotation], dict[str, Any]]:
    selection = load_jsonl(selection_path)
    legacy = load_step_a_annotations(legacy_step_a_path)
    if len(legacy) != 120:
        raise ValueError(f"legacy Step-A must be 120, got {len(legacy)}")
    legacy_by_id = {item.stage_a_id: item for item in legacy}

    records: list[StageAStepAAnnotation] = []
    new_meta: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    batch_errors: list[dict[str, Any]] = []

    legacy_rows = [r for r in selection if str(r["stage_a_id"]) < "sa_0121"]
    new_rows = [r for r in selection if str(r["stage_a_id"]) >= "sa_0121"]
    if len(legacy_rows) != 120 or len(new_rows) != 360:
        raise ValueError(
            f"selection split unexpected: legacy={len(legacy_rows)} new={len(new_rows)}"
        )

    for row in legacy_rows:
        stage_a_id = str(row["stage_a_id"])
        src = legacy_by_id[stage_a_id]
        # Integrity: query/bucket must match selection.
        if src.query != row["query"] or src.final_bucket != row["final_bucket"]:
            raise ValueError(f"legacy Step-A mismatch vs selection for {stage_a_id}")
        records.append(_legacy_record_from_v1(src))

    # Annotate new rows in deterministic id order (batches of 48 for reporting).
    for batch_start in range(0, len(new_rows), 48):
        batch = new_rows[batch_start : batch_start + 48]
        for row in batch:
            try:
                record, meta = annotate_new_row(row)
                records.append(record)
                new_meta.append(meta)
                if meta["flags"]:
                    reason = (
                        "surface semantics preferred over authored operator_family hint"
                    )
                    if any("undersegmented" in f for f in meta["flags"]):
                        reason = "fewer H2 spans than authored family length"
                    mismatches.append(
                        {
                            "stage_a_id": meta["stage_a_id"],
                            "candidate_id": row.get("candidate_id"),
                            "flags": meta["flags"],
                            "operator_family": row.get("operator_family"),
                            "annotated_operators": meta["operator_types"],
                            "reason": reason,
                            "query": row["query"],
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                batch_errors.append(
                    {
                        "stage_a_id": row.get("stage_a_id"),
                        "error": str(exc),
                        "query": row.get("query"),
                    }
                )
                ambiguous.append(
                    {
                        "stage_a_id": row.get("stage_a_id"),
                        "candidate_id": row.get("candidate_id"),
                        "issue": f"generation_failed:{exc}",
                        "query": row.get("query"),
                    }
                )

    records.sort(key=lambda item: item.stage_a_id)
    # Second-pass structural + semantic audit on new rows.
    second_pass_fixes: list[dict[str, Any]] = []
    correction_log: list[dict[str, Any]] = []
    audited: list[StageAStepAAnnotation] = []
    for record in records:
        if record.stage_a_id < "sa_0121":
            audited.append(record)
            continue
        if (record.provenance or {}).get("ontology_incompatible_v1"):
            audited.append(record)
            continue
        fixed, notes = _second_pass_audit(record)
        if notes:
            second_pass_fixes.append(
                {"stage_a_id": record.stage_a_id, "notes": notes}
            )
        fixed, corrections = _apply_semantic_corrections(fixed)
        correction_log.extend(corrections)
        # Ambiguity heuristics
        issues = _ambiguity_check(fixed)
        if issues:
            ambiguous.append(
                {
                    "stage_a_id": fixed.stage_a_id,
                    "candidate_id": fixed.candidate_id,
                    "issue": "; ".join(issues),
                    "query": fixed.query,
                }
            )
        audited.append(fixed)

    audited.sort(key=lambda item: item.stage_a_id)
    errors = validate_step_a_v2_corpus(audited, selection_path=selection_path)
    report = _build_report(
        audited,
        new_meta=new_meta,
        mismatches=mismatches,
        ambiguous=ambiguous,
        second_pass_fixes=second_pass_fixes,
        correction_log=correction_log,
        batch_errors=batch_errors,
        validation_errors=errors,
    )
    if errors:
        raise ValueError(f"Step-A v2 validation failed ({len(errors)}): {errors[:12]}")
    return audited, report


def _second_pass_audit(
    record: StageAStepAAnnotation,
) -> tuple[StageAStepAAnnotation, list[str]]:
    """Light deterministic cleanup + sequential/nav repair."""
    notes: list[str] = []
    anchors = list(record.anchors)
    # Remove only true filler "it"; keep bare this/that/there as referents.
    cleaned = []
    for anchor in anchors:
        if anchor.text.lower() == "it":
            notes.append(f"dropped_filler_anchor:{anchor.text}")
            continue
        cleaned.append(anchor)
    if len(cleaned) != len(anchors):
        cleaned = [
            a.model_copy(update={"anchor_index": i}) for i, a in enumerate(cleaned)
        ]
        record = record.model_copy(update={"anchors": tuple(cleaned)})

    # Repair under-split sequential navigate tails.
    if record.final_bucket == "MIXED_SEQUENTIAL" and len(record.operations) >= 1:
        last = record.operations[-1]
        if (
            last.operator_type is not OperatorType.NAVIGATE_TO
            and _NAV_CUES.search(last.text)
            and (", " in last.text or " and " in last.text.lower() or " then " in last.text.lower())
        ):
            # Re-split this record from scratch with current splitter.
            rebuilt_ops = []
            spans = _split_clauses(record.query)
            fam = list((record.provenance or {}).get("operator_family") or [])
            # Fall back: use existing ops' types where possible.
            hints = fam if fam else [op.operator_type.value for op in record.operations]
            while len(hints) < len(spans):
                hints.append(None)  # type: ignore[arg-type]
            for index, (start, end) in enumerate(spans):
                clause = record.query[start:end]
                hint = hints[index] if index < len(hints) else None
                op_type = _infer_operator(
                    clause,
                    hint=hint,
                    bucket=record.final_bucket,
                    clause_index=index,
                    n_clauses=len(spans),
                )
                rebuilt_ops.append(
                    StepAOperation(
                        operation_index=index,
                        text=clause,
                        char_start=start,
                        char_end=end,
                        operator_type=op_type,
                    )
                )
            if len(rebuilt_ops) > len(record.operations):
                notes.append(
                    f"resplit_operations:{len(record.operations)}->{len(rebuilt_ops)}"
                )
                record = record.model_copy(update={"operations": tuple(rebuilt_ops)})
                rebuilt_anchors = _find_anchors(
                    record.query,
                    [(op.char_start, op.char_end) for op in record.operations],
                )
                record = record.model_copy(update={"anchors": tuple(rebuilt_anchors)})

    # Ensure at least one anchor for COMPLETE quality when possible.
    if not record.anchors:
        rebuilt = _find_anchors(
            record.query,
            [(op.char_start, op.char_end) for op in record.operations],
        )
        if rebuilt:
            notes.append("inserted_fallback_anchor")
            record = record.model_copy(update={"anchors": tuple(rebuilt)})
    return record, notes


def _rebuild_operation(
    query: str,
    start: int,
    end: int,
    op_type: OperatorType,
    *,
    index: int,
) -> StepAOperation:
    text = query[start:end]
    return StepAOperation(
        operation_index=index,
        text=text,
        char_start=start,
        char_end=end,
        operator_type=op_type,
    )


def _apply_semantic_corrections(
    record: StageAStepAAnnotation,
) -> tuple[StageAStepAAnnotation, list[dict[str, Any]]]:
    """Deterministic post-audit operator fixes with explicit correction log."""
    corrections: list[dict[str, Any]] = []
    ops = list(record.operations)
    if not ops:
        return record, corrections

    changed = False
    new_ops: list[StepAOperation] = []
    for index, op in enumerate(ops):
        low = op.text.lower()
        target = op.operator_type
        if low.startswith("what time is it"):
            target = OperatorType.DESCRIBE_ENVIRONMENT
        elif low.startswith("when does my") or "will i make my" in low:
            target = OperatorType.RETRIEVE_PERSONAL
        elif record.final_bucket == "MIXED_IMPLICIT":
            implicit = _implicit_match_operator(low)
            if implicit is not None:
                target = implicit
        elif (
            record.final_bucket == "MIXED_SEQUENTIAL"
            and op.operator_type is OperatorType.NAVIGATE_TO
            and _LOC_CUES.search(low)
            and "how do i" not in low
            and not low.startswith("navigate ")
            and not low.startswith("guide ")
        ):
            target = OperatorType.LOCATE_ENVIRONMENTAL

        if target is not op.operator_type:
            corrections.append(
                {
                    "stage_a_id": record.stage_a_id,
                    "operation_index": index,
                    "old_operator": op.operator_type.value,
                    "new_operator": target.value,
                    "text": op.text,
                    "reason": "semantic_audit_correction",
                }
            )
            changed = True
        new_ops.append(
            _rebuild_operation(
                record.query,
                op.char_start,
                op.char_end,
                target,
                index=index,
            )
        )

    if not changed:
        return record, corrections

    new_ops = [
        op.model_copy(update={"operation_index": i}) for i, op in enumerate(new_ops)
    ]
    anchors = _find_anchors(
        record.query,
        [(op.char_start, op.char_end) for op in new_ops],
    )
    return (
        record.model_copy(
            update={"operations": tuple(new_ops), "anchors": tuple(anchors)}
        ),
        corrections,
    )


def _ambiguity_check(record: StageAStepAAnnotation) -> list[str]:
    issues: list[str] = []
    if (record.provenance or {}).get("ontology_incompatible_v1"):
        return issues
    q = record.query.lower()
    ops = [op.operator_type for op in record.operations]
    if len(ops) == 1 and " and " in q and record.final_bucket == "MIXED_PARALLEL":
        issues.append("parallel cue 'and' but only one operation span")
    if not record.anchors:
        issues.append("no H4 anchors found")
    # IDENTIFY vs DESCRIBE soft conflict
    if any(op.operator_type is OperatorType.IDENTIFY_ENVIRONMENTAL for op in record.operations):
        if re.search(r"\bdescribe\b", q) and not any(
            op.operator_type is OperatorType.DESCRIBE_ENVIRONMENT
            for op in record.operations
        ):
            issues.append("query contains 'describe' but no DESCRIBE operation")
    # Authored family length vs surface op count is expected for MIXED_IMPLICIT
    # (extra labels are Step-B/implicit); do not mark as ambiguous by default.
    return issues


def _build_report(
    records: Sequence[StageAStepAAnnotation],
    *,
    new_meta: Sequence[Mapping[str, Any]],
    mismatches: Sequence[Mapping[str, Any]],
    ambiguous: Sequence[Mapping[str, Any]],
    second_pass_fixes: Sequence[Mapping[str, Any]],
    correction_log: Sequence[Mapping[str, Any]],
    batch_errors: Sequence[Mapping[str, Any]],
    validation_errors: Sequence[str],
) -> dict[str, Any]:
    legacy = [r for r in records if r.stage_a_id < "sa_0121"]
    new = [r for r in records if r.stage_a_id >= "sa_0121"]
    ontology_blocked = [
        {
            "stage_a_id": r.stage_a_id,
            "source_id": r.source_id,
            "candidate_id": r.candidate_id,
            "query": r.query,
            "reason": (r.provenance or {}).get("replacement_reason"),
        }
        for r in records
        if (r.provenance or {}).get("ontology_incompatible_v1")
    ]
    op_counts = Counter(
        op.operator_type.value for r in records for op in r.operations
    )
    status_counts = Counter(r.step_a_status.value for r in records)
    return {
        "total_rows": len(records),
        "legacy_copied": len(legacy),
        "new_agent_assisted": len(new),
        "status_counts": dict(status_counts),
        "complete_count": status_counts.get(StepAStatus.COMPLETE.value, 0),
        "ambiguous_count": len(ambiguous),
        "total_explicit_operations": sum(len(r.operations) for r in records),
        "operator_counts": dict(op_counts),
        "total_anchors": sum(len(r.anchors) for r in records),
        "rows_with_multiple_operations": sum(
            1 for r in records if len(r.operations) > 1
        ),
        "rows_with_multiple_anchors": sum(1 for r in records if len(r.anchors) > 1),
        "authored_operator_mismatches": list(mismatches),
        "ambiguous_rows": list(ambiguous),
        "ontology_blocked_rows": ontology_blocked,
        "ontology_blocked_count": len(ontology_blocked),
        "semantic_correction_log": list(correction_log),
        "second_pass_fixes": list(second_pass_fixes),
        "validation_failures_corrected_during_second_pass": list(
            second_pass_fixes
        )
        + list(correction_log),
        "batch_errors": list(batch_errors),
        "validation_errors": list(validation_errors),
        "fingerprint": _annotations_fingerprint(records),
        "legacy_step_a_path": str(STAGE_A_V1_STEP_A_PATH),
        "selection_path": str(STAGE_A_V2_SELECTION_PATH),
        "output_path": str(STAGE_A_V2_STEP_A_PATH),
        "method": "agent_assisted_rule_guided",
        "v1_files_unchanged": True,
        "step_b_not_started": True,
    }


def _annotations_fingerprint(records: Sequence[StageAStepAAnnotation]) -> str:
    payload = [
        {
            "stage_a_id": r.stage_a_id,
            "query": r.query,
            "operations": [
                {
                    "operator_type": op.operator_type.value,
                    "char_start": op.char_start,
                    "char_end": op.char_end,
                    "text": op.text,
                }
                for op in r.operations
            ],
            "anchors": [
                {
                    "char_start": a.char_start,
                    "char_end": a.char_end,
                    "text": a.text,
                }
                for a in r.anchors
            ],
        }
        for r in sorted(records, key=lambda item: item.stage_a_id)
    ]
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_stage_a_v2_step_a(
    *,
    output_path: str | Path = STAGE_A_V2_STEP_A_PATH,
    report_path: str | Path = STAGE_A_V2_STEP_A_REPORT_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    records, report = build_stage_a_v2_step_a(**kwargs)
    write_step_a_annotations(output_path, records)
    Path(report_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = write_stage_a_v2_step_a()
    print(
        json.dumps(
            {
                "total_rows": report["total_rows"],
                "legacy_copied": report["legacy_copied"],
                "new_agent_assisted": report["new_agent_assisted"],
                "complete_count": report["complete_count"],
                "ambiguous_count": report["ambiguous_count"],
                "operator_counts": report["operator_counts"],
                "total_explicit_operations": report["total_explicit_operations"],
                "total_anchors": report["total_anchors"],
                "mismatch_count": len(report["authored_operator_mismatches"]),
                "validation_errors": report["validation_errors"],
                "fingerprint": report["fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
