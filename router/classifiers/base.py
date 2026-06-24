"""
router/classifiers/base.py — Abstract Base Classifier
======================================================
All classifier backends must implement this interface.
The 5-rule pipeline in QueryClassifier (classifier.py) calls
predict_proba() and gets a probability dict back — it doesn't
care which backend is running.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseQueryClassifier(ABC):
    """
    Abstract base for all query classifier backends.

    Each backend is responsible for:
      - Training on (queries, labels) pairs
      - Returning class probabilities for a single query
      - Saving/loading its model files

    The 5-rule routing logic (keyword backstop, implicit Mixed detection,
    low-confidence fallback) lives in QueryClassifier and is backend-agnostic.
    """

    LABELS = ["Personal", "Environmental", "Mixed"]

    # ── Required interface ────────────────────────────────────────

    @abstractmethod
    def train(self, queries: list, labels: list) -> None:
        """Train on (queries, labels) pairs. Labels must be in LABELS."""
        ...

    @abstractmethod
    def predict_proba(self, query: str) -> dict:
        """
        Return class probability dict for a single query.

        Returns:
            {"Personal": 0.40, "Environmental": 0.16, "Mixed": 0.44}
        """
        ...

    @abstractmethod
    def load_or_train(self, dataset_path: str = None) -> None:
        """Load saved model if it exists, else train from dataset file."""
        ...

    @abstractmethod
    def save(self) -> None:
        """Persist model weights to disk."""
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True if the model is loaded and ready for inference."""
        ...

    # ── Optional overrides ────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def model_size_mb(self) -> float:
        """Disk size of saved model files in MB (excluding shared encoder)."""
        return self._dir_size_mb(self.model_dir) if hasattr(self, "model_dir") else 0.0

    @staticmethod
    def _dir_size_mb(path) -> float:
        p = Path(path)
        if not p.exists():
            return 0.0
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return round(total / (1024 * 1024), 2)
