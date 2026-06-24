"""
router/classifiers/cosine_clf.py — Cosine Similarity Baseline
==============================================================
Nearest-centroid classifier using cosine similarity.

Training: compute the mean L2-normalized embedding for each class.
Inference: embed the query, find the closest centroid via dot product,
           convert similarities to probabilities via softmax.

This is the simplest possible semantic classifier and serves as a
lower-bound baseline. No gradient updates, no hyperparameters.

Saved to: models/classifier/cosine/centroids.pkl
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np

from router.classifiers.base import BaseQueryClassifier

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "classifier" / "cosine"


class CosineClassifier(BaseQueryClassifier):
    """
    Nearest-centroid classifier using cosine similarity.

    Training: ~1 second for any dataset size.
    Inference: ~5 ms per query.
    No hyperparameters to tune.
    """

    model_dir = MODEL_DIR

    def __init__(self, encoder_name: str = "all-MiniLM-L6-v2"):
        self.encoder_name = encoder_name
        self._encoder = None
        self._centroids: dict = {}   # {label: unit-norm centroid vector}
        self._classes: list = []
        self._loaded = False

    # ── BaseQueryClassifier interface ─────────────────────────────

    def train(self, queries: list, labels: list) -> None:
        logger.info("Cosine: computing class centroids from %d samples...", len(queries))
        embeddings = self._encode(queries)
        embeddings = self._l2_normalize(embeddings)

        self._classes = sorted(set(labels))
        self._centroids = {}
        for label in self._classes:
            idx = [i for i, l in enumerate(labels) if l == label]
            class_vecs = embeddings[idx]
            centroid = class_vecs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            self._centroids[label] = centroid / norm if norm > 0 else centroid

        self._loaded = True
        self.save()
        logger.info("Cosine: centroids computed for classes: %s", self._classes)

    def predict_proba(self, query: str) -> dict:
        embedding = self._encode([query])[0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Cosine similarity = dot product of unit vectors
        sims = {label: float(np.dot(embedding, c)) for label, c in self._centroids.items()}

        # Softmax over similarities for calibrated probabilities
        vals = np.array(list(sims.values()))
        # Temperature scaling: higher T = softer distribution
        T = 0.1
        exp_vals = np.exp((vals - vals.max()) / T)
        probs = exp_vals / exp_vals.sum()

        return {label: float(p) for label, p in zip(sims.keys(), probs)}

    def load_or_train(self, dataset_path: str = None) -> None:
        if self._try_load():
            return
        if dataset_path:
            logger.info("Cosine: training from %s", dataset_path)
            with open(dataset_path) as f:
                data = json.load(f)
            self.train([d["query"] for d in data], [d["label"] for d in data])
        else:
            logger.info("Cosine: no saved model, no dataset.")
            self._init_encoder()

    def save(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / "centroids.pkl", "wb") as f:
            pickle.dump({"centroids": self._centroids, "classes": self._classes}, f)
        logger.info("Cosine: saved to %s", MODEL_DIR)

    @property
    def is_ready(self) -> bool:
        return self._loaded and bool(self._centroids)

    # ── Internals ─────────────────────────────────────────────────

    def _try_load(self) -> bool:
        path = MODEL_DIR / "centroids.pkl"
        if not path.exists():
            return False
        try:
            self._init_encoder()
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._centroids = data["centroids"]
            self._classes   = data["classes"]
            self._loaded    = True
            logger.info("Cosine: loaded centroids from %s", path)
            return True
        except Exception as e:
            logger.warning("Cosine: failed to load — %s", e)
            return False

    def _init_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_name)
            logger.info("Cosine: encoder loaded (%s)", self.encoder_name)

    def _encode(self, texts: list) -> np.ndarray:
        self._init_encoder()
        return self._encoder.encode(texts, batch_size=32, show_progress_bar=False)

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return matrix / norms
