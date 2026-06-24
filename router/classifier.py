"""
router/classifier.py — C1: Query Classifier
============================================
5-rule routing pipeline wrapping a swappable ML backend.

Rule 0   — Implicit Mixed patterns (e.g. "Where is my gate?")
Rule 0.5 — Explicit Mixed: conjunction + personal signal + env signal
Rule 1   — Keyword backstop → force Personal (privacy enforcement)
Rule 2+3 — ML backend (predict_proba) + low-confidence handler

The ML backend is selected via config.yaml → classifier_backend.
All backends implement BaseQueryClassifier and are in router/classifiers/.
"""

import logging
import re
import time

logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    5-rule query classifier. Wraps any BaseQueryClassifier backend.

    Usage:
        clf = QueryClassifier()                          # uses config default
        clf = QueryClassifier(backend="setfit")          # explicit backend
        clf.load_or_train("dataset/training_data.json")
        result = clf.predict("Where is my gate?")
        # → ClassificationResult(label='Mixed', confidence=0.90, via=implicit_mixed_pattern)
    """

    def __init__(self, backend: str = None, encoder_name: str = None):
        # Load config
        try:
            from config import cfg
        except Exception:
            cfg = {}

        self.privacy_keywords      = [kw.lower() for kw in cfg.get("privacy_keywords", [])]
        self.confidence_threshold  = cfg.get("classifier_confidence_threshold", 0.65)
        self.use_keyword_backstop  = cfg.get("use_keyword_backstop", True)

        backend_name   = backend or cfg.get("classifier_backend", "minilm_lr")
        encoder        = encoder_name or cfg.get("classifier_model", "all-MiniLM-L6-v2")

        from router.classifiers import get_classifier
        self._backend = get_classifier(backend_name, encoder_name=encoder)
        logger.info("QueryClassifier: backend=%s", backend_name)

    # ── Public API (mirrors old interface — pipeline.py unchanged) ─

    def load_or_train(self, dataset_path: str = None):
        self._backend.load_or_train(dataset_path)

    def train(self, queries: list, labels: list):
        self._backend.train(queries, labels)

    def save(self):
        self._backend.save()

    @property
    def backend_name(self) -> str:
        return self._backend.__class__.__name__

    # ── 5-Rule Predict ────────────────────────────────────────────

    def predict(self, query: str) -> "ClassificationResult":
        t0 = time.perf_counter()

        # Rule 0: Implicit Mixed
        # Query looks Personal but structurally needs a personal lookup THEN
        # environmental navigation — e.g. "Where is my gate?" (gate # from
        # booking → physical gate location).
        if self._is_implicit_mixed(query):
            return ClassificationResult(
                label="Mixed", confidence=0.90,
                latency_ms=(time.perf_counter() - t0) * 1000,
                triggered_by="implicit_mixed_pattern",
            )

        # Rule 0.5: Explicit Mixed
        # Conjunction + personal signal + environmental signal.
        # Fires BEFORE keyword backstop so "my appointment ... this building"
        # routes to Mixed instead of being force-Personal by the backstop.
        if self._is_explicit_mixed(query):
            return ClassificationResult(
                label="Mixed", confidence=0.88,
                latency_ms=(time.perf_counter() - t0) * 1000,
                triggered_by="explicit_mixed_conjunction",
            )

        # Rule 1: Keyword backstop (hard privacy enforcement)
        if self.use_keyword_backstop:
            q_lower = query.lower()
            for kw in self.privacy_keywords:
                if kw in q_lower:
                    return ClassificationResult(
                        label="Personal", confidence=1.0,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        triggered_by="keyword_backstop",
                        keyword=kw,
                    )

        # Rule 2+3: ML backend
        if not self._backend.is_ready:
            return self._rule_based_fallback(query, t0)

        prob_dict = self._backend.predict_proba(query)
        top_label = max(prob_dict, key=prob_dict.get)
        top_conf  = float(prob_dict[top_label])

        # Rule 2: Low-confidence handler
        # Mixed gets lower mass by nature (ambiguous class) — trust it even at
        # low confidence. For Environmental: only force-Personal when the query
        # also has personal signals (privacy risk); pure env queries go to fog.
        if top_conf < self.confidence_threshold:
            latency = (time.perf_counter() - t0) * 1000
            logger.debug(
                "Low confidence %.2f (top=%s) probs=%s", top_conf, top_label,
                {k: f"{v:.2f}" for k, v in prob_dict.items()},
            )
            if top_label == "Mixed":
                return ClassificationResult(
                    label="Mixed", confidence=top_conf, latency_ms=latency,
                    triggered_by="low_confidence_mixed", all_probs=prob_dict,
                )
            if top_label == "Environmental":
                has_personal = any(re.search(p, query.lower()) for p in self._PERSONAL_RE)
                if not has_personal:
                    return ClassificationResult(
                        label="Environmental", confidence=top_conf, latency_ms=latency,
                        triggered_by="low_confidence_environmental", all_probs=prob_dict,
                    )
            return ClassificationResult(
                label="Personal", confidence=top_conf, latency_ms=latency,
                triggered_by="low_confidence_safe_default", all_probs=prob_dict,
            )

        return ClassificationResult(
            label=top_label, confidence=top_conf,
            latency_ms=(time.perf_counter() - t0) * 1000,
            triggered_by="classifier", all_probs=prob_dict,
        )

    # ── Rule pattern data ─────────────────────────────────────────

    _IMPLICIT_MIXED_PATTERNS = [
        r"where\s+is\s+my\s+\w+",
        r"how\s+(do\s+i|can\s+i)\s+get\s+to\s+my\s+\w+",
        r"which\s+(terminal|gate|exit|building|floor|room)\s+(is|for)\s+my\s+\w+",
        r"what\s+(floor|building|room|terminal)\s+is\s+my\s+\w+\s+(on|in|at)",
        r"(how\s+far|how\s+long)\s+(is|to)\s+my\s+\w+",
        r"is\s+this\s+(the\s+)?\w+(\s+\w+)?\s+(for|with|of)\s+my\s+\w+",
        r"is\s+this\s+(the\s+)?right\s+\w+\s+for\s+my\s+\w+",
    ]

    _CONJUNCTIONS = [" and ", " but also ", " as well as ", ", and ", " while "]
    _PERSONAL_RE  = [r"\bmy\b", r"\bam\s+i\b", r"\bi'?m\b", r"\bi\s+have\b", r"\bdo\s+i\b"]
    _ENV_KEYWORDS = [
        "nearby", "nearest", "in front", "is there a", "is there an",
        "around me", "over there", "how far", "which way", "which direction",
        "this place", "this sign", "this building", "this restaurant", "this menu",
        "this terminal", "this clinic", "this pharmacy", "this pill", "this food",
        "this dish", "this road", "this street", "this floor", "this gate",
        "this screen", "this entrance", "this counter", "this section", "this area",
        "the gate", "the exit", "the pharmacy", "the clinic", "the hospital",
        "right building", "right floor", "right terminal", "right direction",
        "right gate", "right place", "right counter", "right entrance",
        "the price", "available here", "listed here", "what time is it",
    ]

    def _is_implicit_mixed(self, query: str) -> bool:
        q = query.lower()
        return any(re.search(p, q) for p in self._IMPLICIT_MIXED_PATTERNS)

    def _is_explicit_mixed(self, query: str) -> bool:
        q = query.lower()
        if not any(c in q for c in self._CONJUNCTIONS):
            return False
        return (
            any(re.search(p, q) for p in self._PERSONAL_RE) and
            any(s in q for s in self._ENV_KEYWORDS)
        )

    def _rule_based_fallback(self, query: str, t0: float) -> "ClassificationResult":
        q = query.lower()
        personal_tokens = [
            "my ", "i have", "i am", "am i", "i'm", "do i",
            "my name", "my health", "my doctor", "my medication",
            "my id", "passport", "i need my", "do i have",
        ]
        env_tokens = [
            "what is this", "is this", "what does", "where is the", "where is",
            "what sign", "which building", "around me", "what color", "how far",
            "nearest", "nearby", "in front of", "is there a", "over there",
        ]
        p = sum(1 for t in personal_tokens if t in q)
        e = sum(1 for t in env_tokens if t in q)
        label = "Mixed" if (p > 0 and e > 0) else ("Personal" if p >= e else "Environmental")
        return ClassificationResult(
            label=label, confidence=0.70,
            latency_ms=(time.perf_counter() - t0) * 1000,
            triggered_by="rule_based_fallback",
        )


class ClassificationResult:
    def __init__(
        self,
        label: str,
        confidence: float,
        latency_ms: float,
        triggered_by: str = "classifier",
        keyword: str = None,
        all_probs: dict = None,
    ):
        self.label        = label
        self.confidence   = confidence
        self.latency_ms   = latency_ms
        self.triggered_by = triggered_by
        self.keyword      = keyword
        self.all_probs    = all_probs or {}

    def __repr__(self):
        return (
            f"ClassificationResult(label={self.label!r}, "
            f"confidence={self.confidence:.3f}, "
            f"latency={self.latency_ms:.1f}ms, "
            f"via={self.triggered_by})"
        )

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "triggered_by": self.triggered_by,
            "keyword": self.keyword,
            "all_probs": self.all_probs,
        }
