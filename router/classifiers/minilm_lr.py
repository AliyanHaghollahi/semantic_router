"""
router/classifiers/minilm_lr.py — Frozen MiniLM + Logistic Regression
=======================================================================
Encodes queries with a frozen SentenceTransformer (all-MiniLM-L6-v2),
then trains a scikit-learn LogisticRegression head on top.

This is the default backend. Fastest to train (seconds on CPU),
smallest model footprint.

Saved to: models/classifier/minilm_lr/head.pkl
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from router.classifiers.base import BaseQueryClassifier

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "classifier" / "minilm_lr"

# Backward-compatible path (original location before multi-backend refactor)
_LEGACY_PATH = Path(__file__).parent.parent.parent / "models" / "classifier" / "head.pkl"


class _TFIDFFallback:
    """TF-IDF + SVD encoder when SentenceTransformer is unavailable."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self._vec = TfidfVectorizer(max_features=2000, sublinear_tf=True)
        self._svd = TruncatedSVD(n_components=384, random_state=42)
        self._fitted = False

    def encode(self, texts: list) -> np.ndarray:
        if not self._fitted:
            X = self._vec.fit_transform(texts)
            n = min(384, X.shape[1] - 1) if X.shape[1] > 1 else 1
            self._svd.set_params(n_components=n)
            vecs = self._svd.fit_transform(X).astype("float32")
            self._fitted = True
        else:
            X = self._vec.transform(texts)
            vecs = self._svd.transform(X).astype("float32")
        if vecs.shape[1] < 384:
            pad = np.zeros((vecs.shape[0], 384 - vecs.shape[1]), "float32")
            vecs = np.hstack([vecs, pad])
        return vecs


class MiniLMLRClassifier(BaseQueryClassifier):
    """
    Frozen SentenceTransformer encoder + scikit-learn LogisticRegression head.

    Training: ~2–5 seconds on CPU for 100 examples.
    Inference: ~5–10 ms per query.
    """

    model_dir = MODEL_DIR

    def __init__(self, encoder_name: str = "all-MiniLM-L6-v2"):
        self.encoder_name = encoder_name
        self._encoder = None
        self._head: LogisticRegression = None
        self._loaded = False

    # ── BaseQueryClassifier interface ─────────────────────────────

    def train(self, queries: list, labels: list) -> None:
        logger.info("MiniLM+LR: training on %d samples...", len(queries))
        embeddings = self._encode(queries)
        self._head = LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced", solver="lbfgs"
        )
        self._head.fit(embeddings, labels)
        self._loaded = True
        self.save()
        logger.info("MiniLM+LR: training complete.")

    def predict_proba(self, query: str) -> dict:
        embedding = self._encode([query])
        probs = self._head.predict_proba(embedding)[0]
        return {cls: float(p) for cls, p in zip(self._head.classes_, probs)}

    def load_or_train(self, dataset_path: str = None) -> None:
        if self._try_load():
            return
        if dataset_path:
            logger.info("MiniLM+LR: training from %s", dataset_path)
            with open(dataset_path) as f:
                data = json.load(f)
            self.train([d["query"] for d in data], [d["label"] for d in data])
        else:
            logger.info("MiniLM+LR: no saved model, no dataset — rule-based only.")
            self._init_encoder()

    def save(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / "head.pkl", "wb") as f:
            pickle.dump(self._head, f)
        with open(MODEL_DIR / "metadata.json", "w") as f:
            json.dump(
                {
                    "encoder_name": self.encoder_name,
                    "encoder_kind": self._encoder_kind(),
                },
                f,
                indent=2,
            )
        logger.info("MiniLM+LR: saved to %s", MODEL_DIR)

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._head is not None

    # ── Internals ─────────────────────────────────────────────────

    def _try_load(self) -> bool:
        head_path = MODEL_DIR / "head.pkl"
        # Fall back to legacy location for backward compatibility
        if not head_path.exists() and _LEGACY_PATH.exists():
            head_path = _LEGACY_PATH
        if not head_path.exists():
            return False
        try:
            self._init_encoder()
            if not self._metadata_matches(head_path.parent):
                return False
            with open(head_path, "rb") as f:
                self._head = pickle.load(f)
            self._loaded = True
            logger.info("MiniLM+LR: loaded from %s", head_path)
            return True
        except Exception as e:
            logger.warning("MiniLM+LR: failed to load — %s", e)
            return False

    def _init_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.encoder_name)
                logger.info("MiniLM+LR: encoder loaded (%s)", self.encoder_name)
            except Exception as e:
                logger.warning("SentenceTransformer unavailable (%s). Using TF-IDF fallback.", e)
                self._encoder = _TFIDFFallback()

    def _encode(self, texts: list) -> np.ndarray:
        self._init_encoder()
        if isinstance(self._encoder, _TFIDFFallback):
            return self._encoder.encode(texts)
        return self._encoder.encode(texts, batch_size=32, show_progress_bar=False)

    def _encoder_kind(self) -> str:
        return "tfidf_fallback" if isinstance(self._encoder, _TFIDFFallback) else "sentence_transformer"

    def _metadata_matches(self, model_dir: Path) -> bool:
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            logger.info("MiniLM+LR: no metadata found; loading legacy head.")
            return True
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning("MiniLM+LR: failed to read metadata - %s", e)
            return False

        expected = {
            "encoder_name": self.encoder_name,
            "encoder_kind": self._encoder_kind(),
        }
        if metadata != expected:
            logger.warning(
                "MiniLM+LR: saved head metadata %s does not match current encoder %s; retraining.",
                metadata,
                expected,
            )
            return False
        return True
