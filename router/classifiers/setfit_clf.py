"""
router/classifiers/setfit_clf.py — SetFit Few-Shot Classifier
==============================================================
Uses Hugging Face SetFit to fine-tune a SentenceTransformer with
contrastive (pair) learning, then trains a classification head.

SetFit is designed for few-shot scenarios (8–64 examples per class)
and typically outperforms frozen LR heads at the same data size.

Install:
    pip install setfit

Saved to: models/classifier/setfit/
"""

import json
import logging
from pathlib import Path

import numpy as np

from router.classifiers.base import BaseQueryClassifier

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "classifier" / "setfit"


class SetFitClassifier(BaseQueryClassifier):
    """
    SetFit few-shot classifier.

    Training: fine-tunes SentenceTransformer with contrastive pairs (~30s–2min on CPU).
    Inference: ~10–20 ms per query.

    Recommended when you have 20–100 labelled examples per class
    and want better Mixed recall than frozen LR provides.
    """

    model_dir = MODEL_DIR

    def __init__(self, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder_name = encoder_name
        self._model = None
        self._classes: list = []
        self._loaded = False

    # ── BaseQueryClassifier interface ─────────────────────────────

    def train(self, queries: list, labels: list) -> None:
        self._require_setfit()
        from setfit import SetFitModel, Trainer, TrainingArguments
        from datasets import Dataset

        logger.info("SetFit: training on %d samples...", len(queries))
        self._classes = sorted(set(labels))

        train_dataset = Dataset.from_dict({"text": queries, "label": labels})

        self._model = SetFitModel.from_pretrained(
            self.encoder_name,
            labels=self._classes,
        )

        args = TrainingArguments(
            batch_size=16,
            num_epochs=1,
            num_setfit_iterations=20,
            evaluation_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
        )

        trainer = Trainer(
            model=self._model,
            args=args,
            train_dataset=train_dataset,
        )
        trainer.train()

        self._loaded = True
        self.save()
        logger.info("SetFit: training complete.")

    def predict_proba(self, query: str) -> dict:
        probs_tensor = self._model.predict_proba([query])[0]
        # probs_tensor is ordered by self._model.labels
        labels = self._model.labels
        probs = probs_tensor.numpy() if hasattr(probs_tensor, "numpy") else np.array(probs_tensor)
        return {label: float(p) for label, p in zip(labels, probs)}

    def load_or_train(self, dataset_path: str = None) -> None:
        if self._try_load():
            return
        if dataset_path:
            logger.info("SetFit: training from %s", dataset_path)
            with open(dataset_path) as f:
                data = json.load(f)
            self.train([d["query"] for d in data], [d["label"] for d in data])
        else:
            logger.info("SetFit: no saved model, no dataset.")

    def save(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(MODEL_DIR))
        logger.info("SetFit: saved to %s", MODEL_DIR)

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._model is not None

    # ── Internals ─────────────────────────────────────────────────

    def _try_load(self) -> bool:
        if not (MODEL_DIR / "config.json").exists():
            return False
        try:
            self._require_setfit()
            from setfit import SetFitModel
            self._model = SetFitModel.from_pretrained(str(MODEL_DIR))
            self._classes = self._model.labels
            self._loaded = True
            logger.info("SetFit: loaded from %s", MODEL_DIR)
            return True
        except Exception as e:
            logger.warning("SetFit: failed to load — %s", e)
            return False

    @staticmethod
    def _require_setfit():
        try:
            import setfit  # noqa: F401
        except ImportError:
            raise ImportError(
                "SetFit is not installed. Run: pip install setfit"
            )
