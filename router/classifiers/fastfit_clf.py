"""
router/classifiers/fastfit_clf.py — FastFit Classifier
=======================================================
FastFit (IBM Research) trains a text classifier by repeatedly
computing similarity between query and label embeddings — no
contrastive pair sampling needed. Trains in seconds even on CPU.

Install:
    pip install fastfit

Saved to: models/classifier/fastfit/
"""

import json
import logging
from pathlib import Path

import numpy as np

from router.classifiers.base import BaseQueryClassifier

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "classifier" / "fastfit"


class FastFitClassifier(BaseQueryClassifier):
    """
    FastFit few-shot classifier (IBM Research).

    Training: ~10–60 seconds on CPU for 100 examples.
    Inference: ~10–30 ms per query via HuggingFace pipeline.

    FastFit is optimised for few-shot (<100 examples/class) classification
    and uses a repeated-classification loss that is faster to train than
    contrastive approaches like SetFit.
    """

    model_dir = MODEL_DIR

    def __init__(self, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder_name = encoder_name
        self._pipeline = None
        self._classes: list = []
        self._loaded = False

    # ── BaseQueryClassifier interface ─────────────────────────────

    def train(self, queries: list, labels: list) -> None:
        self._require_fastfit()
        from fastfit import FastFitTrainer
        from datasets import Dataset, DatasetDict
        from sklearn.model_selection import train_test_split

        logger.info("FastFit: training on %d samples...", len(queries))
        self._classes = sorted(set(labels))

        # FastFit requires a validation split
        train_q, val_q, train_l, val_l = train_test_split(
            queries, labels, test_size=0.15, stratify=labels, random_state=42
        )

        train_ds = Dataset.from_dict({"text": train_q, "label": train_l})
        val_ds   = Dataset.from_dict({"text": val_q,   "label": val_l})

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        trainer = FastFitTrainer(
            model_name_or_path=self.encoder_name,
            label_column_name="label",
            text_column_name="text",
            num_train_epochs=20,
            num_iterations=10,
            dataset=DatasetDict({"train": train_ds, "validation": val_ds}),
            output_dir=str(MODEL_DIR),
        )
        model = trainer.train()
        model.save_pretrained(str(MODEL_DIR))

        self._load_pipeline()
        self._loaded = True
        logger.info("FastFit: training complete.")

    def predict_proba(self, query: str) -> dict:
        results = self._pipeline(query, return_all_scores=True)
        # pipeline returns [{"label": ..., "score": ...}, ...]
        return {r["label"]: float(r["score"]) for r in results[0]}

    def load_or_train(self, dataset_path: str = None) -> None:
        if self._try_load():
            return
        if dataset_path:
            logger.info("FastFit: training from %s", dataset_path)
            with open(dataset_path) as f:
                data = json.load(f)
            self.train([d["query"] for d in data], [d["label"] for d in data])
        else:
            logger.info("FastFit: no saved model, no dataset.")

    def save(self) -> None:
        # FastFitTrainer saves during training — this is a no-op
        pass

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._pipeline is not None

    # ── Internals ─────────────────────────────────────────────────

    def _try_load(self) -> bool:
        if not (MODEL_DIR / "config.json").exists():
            return False
        try:
            self._require_fastfit()
            self._load_pipeline()
            self._loaded = True
            logger.info("FastFit: loaded from %s", MODEL_DIR)
            return True
        except Exception as e:
            logger.warning("FastFit: failed to load — %s", e)
            return False

    def _load_pipeline(self):
        from transformers import pipeline
        self._pipeline = pipeline(
            "text-classification",
            model=str(MODEL_DIR),
            tokenizer=self.encoder_name,
            device=-1,  # CPU
        )

    @staticmethod
    def _require_fastfit():
        try:
            import fastfit  # noqa: F401
        except ImportError:
            raise ImportError(
                "FastFit is not installed. Run: pip install fastfit"
            )
