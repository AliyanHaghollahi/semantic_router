"""
router/classifiers/__init__.py — Backend Factory
=================================================
Returns the right BaseQueryClassifier subclass for a given backend name.

Supported backends:
    "minilm_lr"  — Frozen MiniLM encoder + LogisticRegression (default)
    "cosine"     — Nearest-centroid cosine similarity baseline
    "setfit"     — SetFit few-shot fine-tuning  (pip install setfit)
    "fastfit"    — FastFit IBM classifier        (pip install fastfit)

Usage:
    from router.classifiers import get_classifier
    clf = get_classifier("setfit")
    clf.load_or_train("dataset/training_data.json")
"""

from router.classifiers.base import BaseQueryClassifier

BACKENDS = ["minilm_lr", "cosine", "setfit", "fastfit"]


def get_classifier(backend: str = "minilm_lr", encoder_name: str = "all-MiniLM-L6-v2") -> BaseQueryClassifier:
    """
    Factory — returns an uninitialised classifier backend.
    Call load_or_train() on the result before using predict_proba().
    """
    backend = backend.lower().strip()

    if backend == "minilm_lr":
        from router.classifiers.minilm_lr import MiniLMLRClassifier
        return MiniLMLRClassifier(encoder_name=encoder_name)

    if backend == "cosine":
        from router.classifiers.cosine_clf import CosineClassifier
        return CosineClassifier(encoder_name=encoder_name)

    if backend == "setfit":
        from router.classifiers.setfit_clf import SetFitClassifier
        return SetFitClassifier(encoder_name=f"sentence-transformers/{encoder_name}")

    if backend == "fastfit":
        from router.classifiers.fastfit_clf import FastFitClassifier
        return FastFitClassifier(encoder_name=f"sentence-transformers/{encoder_name}")

    raise ValueError(
        f"Unknown classifier backend: {backend!r}. "
        f"Choose from: {BACKENDS}"
    )
