"""
edge/session.py — Session Manager
===================================
Rolling {query, label, response} buffer stored on edge.
Handles pronoun resolution for follow-up queries.

"Where is it?" after "What medication am I taking?"
→ inject last session turn to resolve "it"

Rules (from system design):
  - Image present + pronoun → image resolves reference, skip injection
  - No image + unresolvable pronoun → inject last N session turns
  - Buffer size: configurable (default 10 turns)
"""

import re
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)

UNRESOLVABLE_PRONOUNS = re.compile(
    r"^(where is it|what is it|how about it|tell me more|and what about|"
    r"is it nearby|does it have|can i|if so|based on that)\b",
    re.IGNORECASE,
)

PRONOUN_PATTERN = re.compile(
    r"\b(it|that|this|those|them|they|there)\b", re.IGNORECASE
)


@dataclass
class SessionTurn:
    query: str
    label: str          # Personal / Environmental / Mixed
    response: str
    turn_id: int


class SessionManager:
    """
    Manages conversation context for the edge device.
    
    Usage:
        session = SessionManager(buffer_size=10)
        
        # After each turn:
        session.add_turn(query, label, response)
        
        # Before routing:
        enriched_query, injected = session.maybe_inject(query, has_image=False)
    """

    def __init__(self, buffer_size: int = 10):
        self.buffer: deque[SessionTurn] = deque(maxlen=buffer_size)
        self._turn_count = 0

    def add_turn(self, query: str, label: str, response: str):
        self._turn_count += 1
        turn = SessionTurn(
            query=query,
            label=label,
            response=response,
            turn_id=self._turn_count,
        )
        self.buffer.append(turn)
        logger.debug("Session: added turn %d (%s)", self._turn_count, label)

    def maybe_inject(
        self, query: str, has_image: bool = False
    ) -> tuple:
        """
        Decide if session context should be injected into this query.
        
        Returns:
            (enriched_query: str, injected: bool)
        """
        if has_image:
            # Image resolves 'this/it/that' — no injection needed
            logger.debug("Session: image present, skipping injection.")
            return query, False

        if not self.buffer:
            return query, False

        # Check if query has unresolvable pronoun
        if not self._has_unresolvable_pronoun(query):
            return query, False

        # Inject last 2 turns as context
        recent = list(self.buffer)[-2:]
        context_lines = []
        for turn in recent:
            context_lines.append(
                f"[Turn {turn.turn_id}] User: {turn.query} | "
                f"Label: {turn.label} | Response: {turn.response}"
            )
        context_str = "\n".join(context_lines)
        enriched = f"[Previous context:\n{context_str}\n]\n\nCurrent query: {query}"
        logger.debug("Session: injected %d turns into query.", len(recent))
        return enriched, True

    def _has_unresolvable_pronoun(self, query: str) -> bool:
        """True if query starts with or heavily relies on a pronoun with no local referent."""
        if UNRESOLVABLE_PRONOUNS.match(query.strip()):
            return True
        # Also check: query is very short + has pronoun
        words = query.strip().split()
        if len(words) <= 6 and PRONOUN_PATTERN.search(query):
            return True
        return False

    def get_recent(self, n: int = 3) -> List[SessionTurn]:
        return list(self.buffer)[-n:]

    def clear(self):
        self.buffer.clear()
        self._turn_count = 0

    @property
    def turn_count(self) -> int:
        return self._turn_count
