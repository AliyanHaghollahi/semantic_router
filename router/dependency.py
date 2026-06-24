"""
router/dependency.py — C3: Dependency Detector
===============================================
Determines if two sub-queries should be dispatched:
  - PARALLEL: both sent simultaneously (asyncio.gather)
  - SEQUENTIAL: edge query runs first, result injected into fog prompt

Rules (in order):
  1. Pronoun reference rule: if environmental sub-query contains an
     unresolved pronoun that could only be resolved by the personal
     sub-query result → SEQUENTIAL
  2. Explicit data-dependency markers ("based on that", "if so", etc.)
  3. Implicit personal query flag → always SEQUENTIAL (need personal
     answer first to construct the env query)
  4. Default: PARALLEL
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Pronouns that may refer to personal query result
REFERENTIAL_PRONOUNS = re.compile(
    r"\b(it|that|this|those|them|they|he|she|its|their)\b"
    r"|\bthere\b(?!\s+is\b|\s+are\b|\s+was\b|\s+were\b)",  # "there" but not existential "there is/are"
    re.IGNORECASE,
)

# Explicit dependency markers
SEQUENTIAL_MARKERS = [
    "based on that",
    "based on the result",
    "if so",
    "if yes",
    "if i do",
    "if it is",
    "if i have",
    "using that",
    "with that information",
    "given that",
    "accordingly",
]


@dataclass
class DependencyResult:
    mode: str          # "parallel" | "sequential"
    reason: str        # human-readable explanation
    inject_personal: bool = False  # if sequential, inject edge answer into fog prompt

    def is_sequential(self) -> bool:
        return self.mode == "sequential"

    def __repr__(self):
        return f"DependencyResult(mode={self.mode!r}, reason={self.reason!r})"


class DependencyDetector:
    """
    Decides if two sub-queries can be dispatched in parallel or must run sequentially.

    Usage:
        detector = DependencyDetector()
        result = detector.detect(
            personal_subquery="What medication am I taking?",
            environmental_subquery="Is there a pharmacy nearby?",
            implicit_personal=False,
        )
        # → DependencyResult(mode='parallel', ...)
    """

    def detect(
        self,
        personal_subquery: str,
        environmental_subquery: str,
        implicit_personal: bool = False,
    ) -> DependencyResult:

        # Rule 1: Implicit personal query → ALWAYS sequential
        # The environmental query cannot be properly formed until the personal
        # answer is known (e.g., gate number must be retrieved before asking
        # where gate D34 is physically located).
        if implicit_personal:
            logger.debug("Sequential: implicit personal query requires edge-first.")
            return DependencyResult(
                mode="sequential",
                reason="Implicit personal sub-query: edge answer needed to construct fog query.",
                inject_personal=True,
            )

        # Rule 2: Pronoun in environmental sub-query with no referent
        env_lower = environmental_subquery.lower()
        personal_lower = personal_subquery.lower()

        env_pronouns = [p for p in REFERENTIAL_PRONOUNS.findall(env_lower) if p.strip()]
        if env_pronouns:
            # Check if the referent could plausibly come from the personal sub-query
            # Heuristic: if personal sub-query contains a noun and env has a pronoun
            # with no obvious image referent → sequential
            has_image_referent = any(
                w in env_lower for w in ["this sign", "this object", "this label",
                                         "what i'm holding", "in front of me"]
            )
            if not has_image_referent:
                logger.debug("Sequential: env sub-query has unresolved pronoun(s): %s", env_pronouns)
                return DependencyResult(
                    mode="sequential",
                    reason=f"Environmental sub-query contains unresolved pronoun(s): {env_pronouns}. "
                           f"Inject edge answer before fog dispatch.",
                    inject_personal=True,
                )

        # Rule 3: Explicit sequential markers in environmental sub-query
        for marker in SEQUENTIAL_MARKERS:
            if marker in env_lower:
                logger.debug("Sequential: found marker '%s' in env sub-query.", marker)
                return DependencyResult(
                    mode="sequential",
                    reason=f"Sequential dependency marker found: '{marker}'.",
                    inject_personal=True,
                )

        # Rule 4: Default → PARALLEL
        logger.debug("Parallel: no dependency detected.")
        return DependencyResult(
            mode="parallel",
            reason="No pronoun reference or dependency marker detected. Dispatching in parallel.",
            inject_personal=False,
        )
