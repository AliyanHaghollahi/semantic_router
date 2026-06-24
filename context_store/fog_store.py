"""
context_store/fog_store.py — Fog Context Store (FAISS)
=======================================================
Environmental knowledge base on the fog server.
Stores: floor plans, POIs, location ontologies, object taxonomies.

Uses FAISS for semantic vector retrieval.
Falls back to keyword search if FAISS not available.

Usage:
    store = FogContextStore()
    store.add_document("The pharmacy is on the second floor, west wing.")
    context = store.retrieve("Is there a pharmacy nearby?")
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try importing FAISS; fall back to numpy cosine similarity
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed. Using numpy fallback. "
                   "Install with: pip install faiss-cpu")


class FogContextStore:
    """
    Semantic knowledge store for environmental context.
    Runs on the fog server (or locally for simulation).
    """

    def __init__(
        self,
        index_path: str = "data/fog_index.faiss",
        metadata_path: str = "data/fog_metadata.json",
        encoder_name: str = "all-MiniLM-L6-v2",
        dim: int = 384,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.dim = dim
        self.encoder_name = encoder_name

        self._encoder = None
        self._index = None
        self._metadata: List[dict] = []

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_or_init()

    # ── Encoder ───────────────────────────────────────────────────

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.encoder_name)
            except Exception as e:
                logger.warning("SentenceTransformer unavailable (%s). Using TF-IDF fallback.", e)
                self._encoder = _TFIDFEncoder(self.dim)
        return self._encoder

    def _encode(self, texts: List[str]) -> np.ndarray:
        enc = self._get_encoder()
        if isinstance(enc, _TFIDFEncoder):
            return enc.encode(texts)
        vecs = enc.encode(texts, show_progress_bar=False)
        return vecs.astype("float32")

    # ── Index I/O ─────────────────────────────────────────────────

    def _load_or_init(self):
        if self.index_path.exists() and self.metadata_path.exists():
            self._load()
        else:
            self._init_empty()

    def _init_empty(self):
        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(self.dim)  # inner product (cosine after norm)
        else:
            self._index = NumpyFlatIndex(self.dim)
        self._metadata = []
        logger.info("FogContextStore: initialized empty index (dim=%d).", self.dim)

    def _load(self):
        try:
            if FAISS_AVAILABLE:
                self._index = faiss.read_index(str(self.index_path))
            else:
                self._index = NumpyFlatIndex.load(str(self.index_path))

            with open(self.metadata_path) as f:
                self._metadata = json.load(f)
            logger.info("FogContextStore: loaded %d entries.", len(self._metadata))
        except Exception as e:
            logger.warning("Failed to load fog store: %s. Re-initializing.", e)
            self._init_empty()

    def save(self):
        if FAISS_AVAILABLE:
            faiss.write_index(self._index, str(self.index_path))
        else:
            self._index.save(str(self.index_path))
        with open(self.metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)
        logger.info("FogContextStore: saved %d entries.", len(self._metadata))

    # ── Write API ─────────────────────────────────────────────────

    def add_document(self, text: str, metadata: dict = None):
        vec = self._encode([text])
        # Normalize for cosine similarity via inner product
        faiss_norm = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = vec / (faiss_norm + 1e-9)
        self._index.add(vec)
        self._metadata.append({
            "text": text,
            **(metadata or {}),
        })

    def add_documents(self, docs: List[dict]):
        """Batch add. Each doc must have 'text' key."""
        texts = [d["text"] for d in docs]
        vecs = self._encode(texts)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norms + 1e-9)
        self._index.add(vecs)
        self._metadata.extend(docs)

    # ── Retrieval ─────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Return top-k relevant context chunks as a formatted string."""
        if len(self._metadata) == 0:
            return ""
        q_vec = self._encode([query])
        q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
        q_vec = q_vec / (q_norm + 1e-9)

        k = min(top_k, len(self._metadata))
        scores, indices = self._index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            if score < 0.3:  # relevance threshold
                continue
            results.append(self._metadata[idx]["text"])

        return "\n".join(results) if results else ""

    @property
    def size(self) -> int:
        return len(self._metadata)


# ── TF-IDF Fallback Encoder (no HuggingFace needed) ─────────────

class _TFIDFEncoder:
    """
    Minimal TF-IDF + SVD encoder for offline/no-model environments.
    Produces vectors in the same dim as MiniLM for compatibility.
    On your laptop this is replaced automatically by the real MiniLM model.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim
        self._fitted = False
        self._vocab: dict = {}
        self._idf: np.ndarray = None

    def encode(self, texts: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        if not hasattr(self, '_vectorizer'):
            self._vectorizer = TfidfVectorizer(max_features=min(1000, max(100, self.dim * 2)))
            self._svd = TruncatedSVD(n_components=self.dim, random_state=42)

        try:
            if not self._fitted or len(texts) > 1:
                X = self._vectorizer.fit_transform(texts)
                # Pad if fewer features than dim
                if X.shape[1] < self.dim:
                    out = X.toarray().astype("float32")
                    pad = np.zeros((out.shape[0], self.dim - out.shape[1]), dtype="float32")
                    return np.hstack([out, pad])
                vecs = self._svd.fit_transform(X).astype("float32")
                self._fitted = True
                return vecs
            else:
                X = self._vectorizer.transform(texts)
                if X.shape[1] < self.dim:
                    out = X.toarray().astype("float32")
                    pad = np.zeros((out.shape[0], self.dim - out.shape[1]), dtype="float32")
                    return np.hstack([out, pad])
                return self._svd.transform(X).astype("float32")
        except Exception:
            # Ultimate fallback: random projection
            return np.random.randn(len(texts), self.dim).astype("float32") * 0.1


# ── Numpy Fallback Index ──────────────────────────────────────────

class NumpyFlatIndex:
    """Drop-in FAISS replacement using numpy cosine similarity."""

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = np.empty((0, dim), dtype="float32")

    def add(self, vecs: np.ndarray):
        self._vectors = np.vstack([self._vectors, vecs]) if self._vectors.shape[0] > 0 else vecs

    def search(self, query: np.ndarray, k: int):
        if self._vectors.shape[0] == 0:
            return np.array([[]] ), np.array([[]])
        scores = (self._vectors @ query.T).squeeze()
        if scores.ndim == 0:
            scores = np.array([float(scores)])
        top_k = min(k, len(scores))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return scores[top_idx].reshape(1, -1), top_idx.reshape(1, -1)

    def save(self, path: str):
        np.save(path, self._vectors)

    @classmethod
    def load(cls, path: str):
        obj = cls.__new__(cls)
        data = np.load(path + ".npy" if not path.endswith(".npy") else path)
        obj._vectors = data
        obj.dim = data.shape[1] if data.ndim > 1 else 384
        return obj
