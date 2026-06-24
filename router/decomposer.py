"""
router/decomposer.py — C2: Mixed Query Decomposer
==================================================
Splits a Mixed query into exactly 2 sub-queries:
  - personal_subquery  → routed to Edge
  - environmental_subquery → routed to Fog

Strategy (in order):
  1. Conjunction split ("and", "but also", "as well as")
  2. Possessive-trigger detection ("my X ... and Y")
  3. spaCy dependency parse (if model available)
  4. SLM fallback (prompt Ollama edge model)

Implicit query detection:
  "Where is my gate?" → implicit personal sub-query (booking lookup)
  → synthesized: personal="What is my gate number?", env="Where is gate {N}?"
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Conjunctions that signal a mixed query boundary
CONJUNCTIONS = [
    " and ",
    " but also ",
    " as well as ",
    " also ",
    " while ",
    ", and ",
]

# Possessive triggers that mark a personal sub-query segment
POSSESSIVE_TRIGGERS = [
    r"\bmy\b",
    r"\bi have\b",
    r"\bdo i\b",
    r"\bam i\b",
    r"\bi am\b",
    r"\bmine\b",
]

# Implicit personal query patterns: "Where is my X?" etc.
IMPLICIT_PERSONAL_PATTERNS = [
    (r"where\s+is\s+my\s+(\w+)", "location_lookup"),
    (r"what\s+time\s+is\s+my\s+(\w+)", "time_lookup"),
    (r"when\s+is\s+my\s+(\w+)", "time_lookup"),
    (r"what\s+is\s+my\s+(\w+)\s+number", "id_lookup"),
]


@dataclass
class DecompositionResult:
    personal_subquery: str
    environmental_subquery: str
    method: str                        # how it was split
    implicit_personal: bool = False    # was personal sub-query synthesized?
    confidence: float = 1.0
    notes: str = ""

    def __repr__(self):
        return (
            f"DecompositionResult(\n"
            f"  personal='{self.personal_subquery}',\n"
            f"  environmental='{self.environmental_subquery}',\n"
            f"  method={self.method!r}, implicit={self.implicit_personal}\n)"
        )


class MixedQueryDecomposer:
    """
    Decomposes a Mixed query into personal + environmental sub-queries.
    
    Usage:
        decomposer = MixedQueryDecomposer()
        result = decomposer.decompose("What medication am I holding and is there a pharmacy nearby?")
    """

    def __init__(self, use_spacy: bool = True, slm_client=None):
        self.slm_client = slm_client   # optional OllamaClient for fallback
        self._nlp = None
        self._spacy_available = False
        if use_spacy:
            self._try_load_spacy()

    def _try_load_spacy(self):
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            self._spacy_available = True
            logger.info("spaCy en_core_web_sm loaded.")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm\n"
                "Falling back to rule-based decomposer."
            )

    # ── Public API ────────────────────────────────────────────────

    def decompose(self, query: str) -> DecompositionResult:
        """Main entry point. Try strategies in order."""
        query = query.strip()

        # 1. Check for implicit personal query
        implicit = self._detect_implicit(query)
        if implicit:
            return implicit

        # 2. Conjunction split
        result = self._conjunction_split(query)
        if result:
            return result

        # 3. spaCy dependency parse
        if self._spacy_available:
            result = self._spacy_split(query)
            if result:
                return result

        # 4. SLM fallback
        if self.slm_client:
            result = self._slm_split(query)
            if result:
                return result

        # 5. Last resort: treat whole query as personal (safe)
        logger.warning("Decomposer fallback: treating whole query as personal.")
        return DecompositionResult(
            personal_subquery=query,
            environmental_subquery="",
            method="fallback_personal",
            confidence=0.5,
            notes="Could not decompose; entire query routed to edge for safety.",
        )

    # ── Strategy 1: Implicit Personal Detection ───────────────────

    def _detect_implicit(self, query: str) -> Optional[DecompositionResult]:
        """
        Detect queries where a personal lookup is IMPLICIT.
        e.g. "Where is my gate?" → needs booking lookup first
        """
        q_lower = query.lower()
        for pattern, query_type in IMPLICIT_PERSONAL_PATTERNS:
            m = re.search(pattern, q_lower)
            if m:
                noun = m.group(1) if m.lastindex else "information"
                personal_q  = f"What is my {noun}?"
                env_q       = f"{query}".strip()  # original remains as env context
                logger.debug("Implicit personal detected: %s → '%s'", query_type, noun)
                return DecompositionResult(
                    personal_subquery=personal_q,
                    environmental_subquery=env_q,
                    method="implicit_possessive_rule",
                    implicit_personal=True,
                    confidence=0.90,
                    notes=f"Type: {query_type}, entity: {noun}",
                )
        return None

    # ── Strategy 2: Conjunction Split ─────────────────────────────

    def _conjunction_split(self, query: str) -> Optional[DecompositionResult]:
        q_lower = query.lower()
        for conj in CONJUNCTIONS:
            idx = q_lower.find(conj)
            if idx == -1:
                continue

            left  = query[:idx].strip()
            right = query[idx + len(conj):].strip()

            if not left or not right:
                continue

            # Decide which side is personal vs environmental
            left_is_personal  = self._has_possessive(left)
            right_is_personal = self._has_possessive(right)

            if left_is_personal and not right_is_personal:
                return DecompositionResult(
                    personal_subquery=left,
                    environmental_subquery=right,
                    method="conjunction_split",
                    confidence=0.88,
                )
            elif right_is_personal and not left_is_personal:
                return DecompositionResult(
                    personal_subquery=right,
                    environmental_subquery=left,
                    method="conjunction_split",
                    confidence=0.88,
                )
            elif left_is_personal and right_is_personal:
                # Both personal — the environmental part is the non-possessive half
                # This is a personal-heavy mixed query; env part is likely implicit
                return DecompositionResult(
                    personal_subquery=left,
                    environmental_subquery=right,
                    method="conjunction_split_both_personal",
                    confidence=0.72,
                    notes="Both halves appear personal; env routing may be redundant.",
                )
            else:
                # Neither has possessive — but we found a conjunction in a Mixed query.
                # Assign by order: treat first clause as environmental (description-like)
                return DecompositionResult(
                    personal_subquery=right,
                    environmental_subquery=left,
                    method="conjunction_split_heuristic",
                    confidence=0.65,
                )

        return None

    def _has_possessive(self, text: str) -> bool:
        t = text.lower()
        return any(re.search(p, t) for p in POSSESSIVE_TRIGGERS)

    # ── Strategy 3: spaCy Dependency Parse ────────────────────────

    def _spacy_split(self, query: str) -> Optional[DecompositionResult]:
        doc = self._nlp(query)

        # Find clausal boundaries: ROOT verbs and their subtrees
        clauses = []
        for token in doc:
            if token.dep_ in ("ROOT", "conj") and token.pos_ == "VERB":
                subtree_tokens = sorted(token.subtree, key=lambda t: t.i)
                clause_text = " ".join(t.text for t in subtree_tokens).strip()
                if clause_text:
                    clauses.append(clause_text)

        if len(clauses) >= 2:
            personal_clauses = [c for c in clauses if self._has_possessive(c)]
            env_clauses      = [c for c in clauses if not self._has_possessive(c)]

            if personal_clauses and env_clauses:
                return DecompositionResult(
                    personal_subquery=" ".join(personal_clauses),
                    environmental_subquery=" ".join(env_clauses),
                    method="spacy_dependency_parse",
                    confidence=0.85,
                )

        return None

    # ── Strategy 4: SLM Fallback ──────────────────────────────────

    def _slm_split(self, query: str) -> Optional[DecompositionResult]:
        prompt = f"""You are a query decomposer. Split the following mixed query into exactly two sub-queries:
1. A PERSONAL sub-query (about the user's own data, health, documents, bookings, contacts)
2. An ENVIRONMENTAL sub-query (about physical surroundings, nearby places, objects, signs)

Mixed query: "{query}"

Respond in JSON only:
{{"personal": "...", "environmental": "..."}}"""

        try:
            response = self.slm_client.generate(prompt)
            import json
            data = json.loads(response.strip())
            if "personal" in data and "environmental" in data:
                return DecompositionResult(
                    personal_subquery=data["personal"],
                    environmental_subquery=data["environmental"],
                    method="slm_fallback",
                    confidence=0.80,
                )
        except Exception as e:
            logger.warning("SLM decomposer failed: %s", e)

        return None
