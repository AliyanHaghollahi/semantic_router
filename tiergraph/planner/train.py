"""Thin Stage-A training / evaluation helpers for the H1-H7 planner.

Wires existing encode -> gold batch -> ``forward_train`` -> ``planner_loss``
APIs. Does not redefine targets, batching, or architecture.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tiergraph.planner.align import BIO_IGNORE, BIO_O, TokenCharSpan
from tiergraph.planner.annotation_step_a import (
    DEFAULT_STEP_A_ANNOTATIONS_PATH,
    EXPECTED_STAGE_A_COUNT,
    fingerprint_file,
)
from tiergraph.planner.annotation_step_b import DEFAULT_STEP_B_ANNOTATIONS_PATH
from tiergraph.planner.annotations import PlannerExample
from tiergraph.planner.batching import (
    GoldStructureBatch,
    build_gold_batch_from_examples,
)
from tiergraph.planner.encoder import DEFAULT_MINILM_MODEL, MiniLMFeatureEncoder
from tiergraph.planner.loss import PlannerLossBreakdown, planner_loss
from tiergraph.planner.model import PlannerHeadOutputs, PlannerModel, decode_bio_spans
from tiergraph.planner.stage_a_split import (
    DEFAULT_DEV_SIZE,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    StageASplitResult,
    group_holdout_split,
)
from tiergraph.planner.stage_a_to_corpus import load_stage_a_planner_examples
from tiergraph.planner.stage_a_v2_spec import (
    STAGE_A_V2_CORPUS_SIZE,
    STAGE_A_V2_DEV_SIZE,
    STAGE_A_V2_SPLIT_FINGERPRINT,
    STAGE_A_V2_SPLIT_SEED,
    STAGE_A_V2_STEP_A_PATH,
    STAGE_A_V2_STEP_B_PATH,
    STAGE_A_V2_TEST_SIZE,
    STAGE_A_V2_TRAIN_SIZE,
)
from tiergraph.planner.stage_a_v2_split import regenerate_stage_a_v2_split_report


DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8
HEAD_KEYS = ("h1", "h2", "h3", "h4", "h5", "h6", "h7")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Serializable training configuration."""

    seed: int = DEFAULT_SPLIT_SEED
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    device: str = "cpu"
    output_dir: str = "artifacts/planner_train"
    smoke: bool = False
    smoke_train_batches: int = 2
    smoke_eval_batches: int = 1
    max_length: int = 128
    encoder_model_name: str = DEFAULT_MINILM_MODEL
    step_a_path: str = str(DEFAULT_STEP_A_ANNOTATIONS_PATH)
    step_b_path: str = str(DEFAULT_STEP_B_ANNOTATIONS_PATH)
    corpus_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


@dataclass
class LossMeter:
    """Running mean of total + per-head losses."""

    count: int = 0
    total: float = 0.0
    heads: dict[str, float] = field(
        default_factory=lambda: {key: 0.0 for key in HEAD_KEYS}
    )

    def update(self, breakdown: PlannerLossBreakdown) -> None:
        self.count += 1
        self.total += float(breakdown.total.detach().item())
        for key in HEAD_KEYS:
            self.heads[key] += float(getattr(breakdown, key).detach().item())

    def means(self) -> dict[str, float]:
        if self.count == 0:
            return {"total": 0.0, **{key: 0.0 for key in HEAD_KEYS}}
        return {
            "total": self.total / self.count,
            **{key: self.heads[key] / self.count for key in HEAD_KEYS},
        }


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    """Gold-structure teacher-forced evaluation metrics."""

    mean_loss: dict[str, float]
    h1_accuracy: float
    h3_operator_accuracy: float
    h5_accuracy: float
    h6_ownership_accuracy: float
    h7_precision: float
    h7_recall: float
    h7_f1: float
    h2_span_precision: float
    h2_span_recall: float
    h2_span_f1: float
    h4_span_precision: float
    h4_span_recall: float
    h4_span_f1: float
    n_examples: int
    n_batches: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_loss": dict(self.mean_loss),
            "h1_accuracy": self.h1_accuracy,
            "h3_operator_accuracy": self.h3_operator_accuracy,
            "h5_accuracy": self.h5_accuracy,
            "h6_ownership_accuracy": self.h6_ownership_accuracy,
            "h7_precision": self.h7_precision,
            "h7_recall": self.h7_recall,
            "h7_f1": self.h7_f1,
            "h2_span_precision": self.h2_span_precision,
            "h2_span_recall": self.h2_span_recall,
            "h2_span_f1": self.h2_span_f1,
            "h4_span_precision": self.h4_span_precision,
            "h4_span_recall": self.h4_span_recall,
            "h4_span_f1": self.h4_span_f1,
            "n_examples": self.n_examples,
            "n_batches": self.n_batches,
            "note": (
                "Metrics use gold-structure teacher-forced head outputs; "
                "not exact predicted-graph accuracy."
            ),
        }


def set_seed(seed: int) -> None:
    """Seed Python, NumPy (if present), and Torch RNGs."""
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_annotations_unchanged(
    *,
    step_a_path: Path,
    step_b_path: Path,
    before_a: tuple[int, str],
    before_b: tuple[int, str],
) -> None:
    after_a = fingerprint_file(step_a_path)
    after_b = fingerprint_file(step_b_path)
    if after_a != before_a:
        raise RuntimeError(f"Step-A annotation file mutated: {step_a_path}")
    if after_b != before_b:
        raise RuntimeError(f"Step-B annotation file mutated: {step_b_path}")


def load_and_split_stage_a(
    config: TrainConfig,
) -> tuple[StageASplitResult, tuple[int, str], tuple[int, str]]:
    """Load 120 examples, split 96/12/12, and fingerprint frozen annotations."""
    step_a_path = Path(config.step_a_path)
    step_b_path = Path(config.step_b_path)
    before_a = fingerprint_file(step_a_path)
    before_b = fingerprint_file(step_b_path)
    examples = load_stage_a_planner_examples(step_a_path, step_b_path)
    if len(examples) != EXPECTED_STAGE_A_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_STAGE_A_COUNT} examples, got {len(examples)}"
        )
    split = group_holdout_split(
        examples,
        train_size=DEFAULT_TRAIN_SIZE,
        dev_size=DEFAULT_DEV_SIZE,
        test_size=DEFAULT_TEST_SIZE,
        seed=config.seed,
    )
    if (
        len(split.train) != DEFAULT_TRAIN_SIZE
        or len(split.dev) != DEFAULT_DEV_SIZE
        or len(split.test) != DEFAULT_TEST_SIZE
    ):
        raise RuntimeError(
            f"unexpected split sizes train={len(split.train)} "
            f"dev={len(split.dev)} test={len(split.test)}"
        )
    if not split.fingerprint:
        raise RuntimeError("split fingerprint missing")
    assert_annotations_unchanged(
        step_a_path=step_a_path,
        step_b_path=step_b_path,
        before_a=before_a,
        before_b=before_b,
    )
    return split, before_a, before_b


def load_and_split_stage_a_v2(
    config: TrainConfig,
) -> tuple[StageASplitResult, tuple[int, str], tuple[int, str]]:
    """Load 480 v2 examples and apply the frozen publication split."""
    step_a_path = Path(config.step_a_path)
    step_b_path = Path(config.step_b_path)
    before_a = fingerprint_file(step_a_path)
    before_b = fingerprint_file(step_b_path)
    result = regenerate_stage_a_v2_split_report(
        step_a_path=step_a_path,
        step_b_path=step_b_path,
        seed=STAGE_A_V2_SPLIT_SEED,
    )
    if (
        len(result.train) != STAGE_A_V2_TRAIN_SIZE
        or len(result.dev) != STAGE_A_V2_DEV_SIZE
        or len(result.test) != STAGE_A_V2_TEST_SIZE
    ):
        raise RuntimeError(
            f"unexpected v2 split sizes train={len(result.train)} "
            f"dev={len(result.dev)} test={len(result.test)}"
        )
    if len(result.train) + len(result.dev) + len(result.test) != STAGE_A_V2_CORPUS_SIZE:
        raise RuntimeError(
            f"expected {STAGE_A_V2_CORPUS_SIZE} examples across splits, "
            f"got {len(result.train) + len(result.dev) + len(result.test)}"
        )
    if result.fingerprint != STAGE_A_V2_SPLIT_FINGERPRINT:
        raise RuntimeError(
            "v2 split fingerprint mismatch: "
            f"got {result.fingerprint}, expected {STAGE_A_V2_SPLIT_FINGERPRINT}"
        )
    assert_annotations_unchanged(
        step_a_path=step_a_path,
        step_b_path=step_b_path,
        before_a=before_a,
        before_b=before_b,
    )
    return result, before_a, before_b


def load_and_split_for_config(
    config: TrainConfig,
) -> tuple[StageASplitResult, tuple[int, str], tuple[int, str]]:
    if config.corpus_version == "v2":
        return load_and_split_stage_a_v2(config)
    return load_and_split_stage_a(config)


def head_parameters(model: PlannerModel) -> list[nn.Parameter]:
    """Learned planner-head parameters (excludes frozen encoder internals)."""
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def encoder_parameters(model: PlannerModel) -> list[nn.Parameter]:
    """Underlying frozen MiniLM parameters when loaded; else empty."""
    encoder_model = getattr(model.encoder, "_model", None)
    if encoder_model is None:
        return []
    return list(encoder_model.parameters())


def count_parameters(parameters: Iterable[nn.Parameter]) -> tuple[int, int]:
    """Return (numel, n_tensors)."""
    tensors = list(parameters)
    return sum(int(p.numel()) for p in tensors), len(tensors)


def assert_encoder_frozen(model: PlannerModel) -> None:
    """Fail if any encoder parameter is trainable or present in head grads."""
    for parameter in encoder_parameters(model):
        if parameter.requires_grad:
            raise RuntimeError("encoder parameter unexpectedly requires_grad=True")
    head_ids = {id(parameter) for parameter in head_parameters(model)}
    for parameter in encoder_parameters(model):
        if id(parameter) in head_ids:
            raise RuntimeError("encoder parameter leaked into trainable head set")


def build_optimizer(
    model: PlannerModel,
    *,
    lr: float,
) -> torch.optim.Optimizer:
    """AdamW over trainable heads only."""
    assert_encoder_frozen(model)
    params = head_parameters(model)
    if not params:
        raise RuntimeError("no trainable head parameters found")
    encoder_ids = {id(parameter) for parameter in encoder_parameters(model)}
    for parameter in params:
        if id(parameter) in encoder_ids:
            raise RuntimeError("refusing to optimize a frozen encoder parameter")
    return torch.optim.AdamW(params, lr=lr)


def iter_example_batches(
    examples: Sequence[PlannerExample],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    max_batches: int | None = None,
    shuffle: bool = True,
) -> list[tuple[PlannerExample, ...]]:
    """Deterministic epoch batches (shuffle is seed+epoch reproducible)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = list(range(len(examples)))
    if shuffle:
        rng = random.Random(seed + 1_000_003 * epoch)
        rng.shuffle(indices)
    batches: list[tuple[PlannerExample, ...]] = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        batches.append(tuple(examples[index] for index in chunk))
        if max_batches is not None and len(batches) >= max_batches:
            break
    return batches


def encode_gold_batch(
    model: PlannerModel,
    examples: Sequence[PlannerExample],
) -> tuple[Any, tuple[tuple[TokenCharSpan, ...], ...], GoldStructureBatch]:
    """One encoder forward + gold targets for a batch of examples."""
    texts = [example.query for example in examples]
    features = model.encode(texts)
    token_views = model.encoder.token_char_spans_for_batch(features)
    _targets, gold = build_gold_batch_from_examples(examples, features, token_views)
    return features, token_views, gold


def train_step(
    model: PlannerModel,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[PlannerExample],
) -> PlannerLossBreakdown:
    """One forward/backward/optimizer step on a gold batch."""
    model.train()
    assert_encoder_frozen(model)
    features, _token_views, gold = encode_gold_batch(model, examples)
    outputs = model.forward_train(features, gold)
    breakdown = planner_loss(outputs, gold)
    if not torch.isfinite(breakdown.total):
        raise RuntimeError(f"non-finite total loss: {breakdown.total.item()!r}")
    optimizer.zero_grad(set_to_none=True)
    breakdown.total.backward()
    for parameter in encoder_parameters(model):
        if parameter.grad is not None:
            raise RuntimeError("encoder received gradients")
    optimizer.step()
    return breakdown


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _span_prf_from_bio(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_views: Sequence[Sequence[TokenCharSpan]],
    token_loss_mask: torch.Tensor,
) -> tuple[int, int, int]:
    """Accumulate span-level TP/FP/FN for BIO heads (content tokens only)."""
    tp = fp = fn = 0
    pred_ids = logits.argmax(dim=-1)
    for batch_index, tokens in enumerate(token_views):
        gold_labels: list[int] = []
        pred_labels: list[int] = []
        for token_index, token in enumerate(tokens):
            if not bool(token_loss_mask[batch_index, token_index]):
                gold_labels.append(BIO_O)
                pred_labels.append(BIO_O)
                continue
            gold = int(labels[batch_index, token_index].item())
            pred = int(pred_ids[batch_index, token_index].item())
            if gold == BIO_IGNORE:
                gold = BIO_O
            if not token.is_content:
                gold = BIO_O
                pred = BIO_O
            gold_labels.append(gold)
            pred_labels.append(pred)
        gold_spans = set(decode_bio_spans(gold_labels, tokens))
        pred_spans = set(decode_bio_spans(pred_labels, tokens))
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
    return tp, fp, fn


def accumulate_head_metrics(
    outputs: PlannerHeadOutputs,
    gold: GoldStructureBatch,
    token_views: Sequence[Sequence[TokenCharSpan]],
    counters: dict[str, Any],
) -> None:
    """Update running metric counters from one teacher-forced batch."""
    pred_h1 = outputs.query_type_logits.argmax(dim=-1)
    counters["h1_correct"] += int((pred_h1 == gold.query_type_labels).sum().item())
    counters["h1_total"] += int(gold.query_type_labels.numel())

    if gold.op_valid.any():
        pred_h3 = outputs.op_type_logits.argmax(dim=-1)
        mask = gold.op_valid
        counters["h3_correct"] += int(
            (pred_h3[mask] == gold.op_type_labels[mask]).sum().item()
        )
        counters["h3_total"] += int(mask.sum().item())

    if gold.anc_valid.any():
        pred_h5 = outputs.impl_logits.argmax(dim=-1)
        mask = gold.anc_valid
        counters["h5_correct"] += int(
            (pred_h5[mask] == gold.impl_labels[mask]).sum().item()
        )
        counters["h5_total"] += int(mask.sum().item())

        pred_h6 = outputs.own_logits.argmax(dim=-1)
        row_ok = gold.anc_valid & gold.own_mask.any(dim=-1)
        if bool(row_ok.any()):
            counters["h6_correct"] += int(
                (pred_h6[row_ok] == gold.own_labels[row_ok]).sum().item()
            )
            counters["h6_total"] += int(row_ok.sum().item())

    if gold.dep_mask.any():
        pred_pos = outputs.dep_logits > 0.0
        gold_pos = gold.dep_labels > 0.5
        mask = gold.dep_mask
        counters["h7_tp"] += int((pred_pos & gold_pos & mask).sum().item())
        counters["h7_fp"] += int((pred_pos & ~gold_pos & mask).sum().item())
        counters["h7_fn"] += int((~pred_pos & gold_pos & mask).sum().item())

    h2_tp, h2_fp, h2_fn = _span_prf_from_bio(
        outputs.op_bio_logits,
        gold.op_bio_labels,
        token_views,
        gold.token_loss_mask,
    )
    counters["h2_tp"] += h2_tp
    counters["h2_fp"] += h2_fp
    counters["h2_fn"] += h2_fn
    h4_tp, h4_fp, h4_fn = _span_prf_from_bio(
        outputs.anc_bio_logits,
        gold.anc_bio_labels,
        token_views,
        gold.token_loss_mask,
    )
    counters["h4_tp"] += h4_tp
    counters["h4_fp"] += h4_fp
    counters["h4_fn"] += h4_fn


def _empty_metric_counters() -> dict[str, int]:
    return {
        "h1_correct": 0,
        "h1_total": 0,
        "h3_correct": 0,
        "h3_total": 0,
        "h5_correct": 0,
        "h5_total": 0,
        "h6_correct": 0,
        "h6_total": 0,
        "h7_tp": 0,
        "h7_fp": 0,
        "h7_fn": 0,
        "h2_tp": 0,
        "h2_fp": 0,
        "h2_fn": 0,
        "h4_tp": 0,
        "h4_fp": 0,
        "h4_fn": 0,
    }


def evaluate_examples(
    model: PlannerModel,
    examples: Sequence[PlannerExample],
    *,
    batch_size: int,
    seed: int,
    max_batches: int | None = None,
) -> EvalMetrics:
    """Teacher-forced evaluation on gold structures (not free decode)."""
    model.eval()
    meter = LossMeter()
    counters = _empty_metric_counters()
    n_examples = 0
    batches = iter_example_batches(
        examples,
        batch_size=batch_size,
        seed=seed,
        epoch=0,
        max_batches=max_batches,
        shuffle=False,
    )
    with torch.no_grad():
        for batch in batches:
            features, token_views, gold = encode_gold_batch(model, batch)
            outputs = model.forward_train(features, gold)
            breakdown = planner_loss(outputs, gold)
            if not torch.isfinite(breakdown.total):
                raise RuntimeError("non-finite eval loss")
            meter.update(breakdown)
            accumulate_head_metrics(outputs, gold, token_views, counters)
            n_examples += len(batch)

    h7_p, h7_r, h7_f1 = _prf(counters["h7_tp"], counters["h7_fp"], counters["h7_fn"])
    h2_p, h2_r, h2_f1 = _prf(counters["h2_tp"], counters["h2_fp"], counters["h2_fn"])
    h4_p, h4_r, h4_f1 = _prf(counters["h4_tp"], counters["h4_fp"], counters["h4_fn"])

    def _acc(correct_key: str, total_key: str) -> float:
        total = counters[total_key]
        return (counters[correct_key] / total) if total else 0.0

    return EvalMetrics(
        mean_loss=meter.means(),
        h1_accuracy=_acc("h1_correct", "h1_total"),
        h3_operator_accuracy=_acc("h3_correct", "h3_total"),
        h5_accuracy=_acc("h5_correct", "h5_total"),
        h6_ownership_accuracy=_acc("h6_correct", "h6_total"),
        h7_precision=h7_p,
        h7_recall=h7_r,
        h7_f1=h7_f1,
        h2_span_precision=h2_p,
        h2_span_recall=h2_r,
        h2_span_f1=h2_f1,
        h4_span_precision=h4_p,
        h4_span_recall=h4_r,
        h4_span_f1=h4_f1,
        n_examples=n_examples,
        n_batches=len(batches),
    )


def save_checkpoint(
    path: Path,
    *,
    model: PlannerModel,
    config: TrainConfig,
    split_fingerprint: str,
    best_dev_loss: float,
    best_dev_metrics: Mapping[str, Any] | None,
    epoch: int,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Save head ``state_dict`` plus training metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_head_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "seed": config.seed,
        "split_fingerprint": split_fingerprint,
        "best_dev_loss": best_dev_loss,
        "best_dev_metrics": dict(best_dev_metrics) if best_dev_metrics else None,
        "epoch": epoch,
    }
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, path)


def build_model(
    config: TrainConfig,
    *,
    encoder: MiniLMFeatureEncoder | None = None,
) -> PlannerModel:
    """Construct PlannerModel with frozen MiniLM encoder on ``config.device``."""
    if encoder is None:
        encoder = MiniLMFeatureEncoder(
            model_name=config.encoder_model_name,
            max_length=config.max_length,
            device=config.device,
        )
    model = PlannerModel(encoder)
    model.to(torch.device(config.device))
    if encoder.is_loaded:
        assert_encoder_frozen(model)
    return model


def load_checkpoint(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a planner checkpoint payload saved by ``save_checkpoint``."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint must be a dict payload: {path}")
    if "model_head_state_dict" not in payload:
        raise KeyError(f"checkpoint missing model_head_state_dict: {path}")
    return payload


def config_from_checkpoint(
    payload: Mapping[str, Any],
    *,
    device: str | None = None,
    batch_size: int | None = None,
) -> TrainConfig:
    """Rebuild ``TrainConfig`` from checkpoint metadata."""
    raw = dict(payload.get("config") or {})
    if device is not None:
        raw["device"] = device
    if batch_size is not None:
        raw["batch_size"] = batch_size
    allowed = set(TrainConfig.__dataclass_fields__)
    filtered = {key: value for key, value in raw.items() if key in allowed}
    if "seed" not in filtered and "seed" in payload:
        filtered["seed"] = int(payload["seed"])
    return TrainConfig(**filtered)


@dataclass(frozen=True, slots=True)
class CheckpointEvalResult:
    """Held-out evaluation result for a loaded checkpoint."""

    checkpoint_path: str
    split_name: str
    split_fingerprint: str
    expected_fingerprint: str | None
    n_examples: int
    eval_mode: str
    metrics: EvalMetrics | Any
    checkpoint_seed: int
    checkpoint_best_dev_loss: float | None
    checkpoint_epoch: int | None

    def to_dict(self) -> dict[str, Any]:
        metrics_dict = (
            self.metrics.to_dict()
            if hasattr(self.metrics, "to_dict")
            else dict(self.metrics)
        )
        if self.eval_mode == "teacher-forced":
            label = "teacher-forced gold-structure metrics (not exact graph accuracy)"
        else:
            label = (
                "free predicted-graph metrics "
                "(predict_structures -> GraphDecoder; not teacher-forced)"
            )
        return {
            "checkpoint_path": self.checkpoint_path,
            "split_name": self.split_name,
            "split_fingerprint": self.split_fingerprint,
            "expected_fingerprint": self.expected_fingerprint,
            "n_examples": self.n_examples,
            "eval_mode": self.eval_mode,
            "metrics": metrics_dict,
            "checkpoint_seed": self.checkpoint_seed,
            "checkpoint_best_dev_loss": self.checkpoint_best_dev_loss,
            "checkpoint_epoch": self.checkpoint_epoch,
            "label": label,
        }


EXPECTED_STAGE_A_SPLIT_FINGERPRINT = (
    "7adb7e6a1f2080d965092097207f2e084d24d4a659c4042c27575fc8fac70478"
)


def evaluate_checkpoint(
    checkpoint_path: Path | str,
    *,
    split_name: str = "test",
    seed: int | None = None,
    device: str = "cpu",
    batch_size: int | None = None,
    expected_fingerprint: str | None = EXPECTED_STAGE_A_SPLIT_FINGERPRINT,
    model: PlannerModel | None = None,
    split: StageASplitResult | None = None,
    eval_mode: str = "teacher-forced",
) -> CheckpointEvalResult:
    """Load a head checkpoint and evaluate one split.

    ``eval_mode='teacher-forced'`` uses gold structures (unchanged).
    ``eval_mode='free'`` uses ``predict_structures`` → ``GraphDecoder``.
    """
    if split_name not in {"train", "dev", "test"}:
        raise ValueError(f"split_name must be train/dev/test, got {split_name!r}")
    if eval_mode not in {"teacher-forced", "free"}:
        raise ValueError(
            f"eval_mode must be 'teacher-forced' or 'free', got {eval_mode!r}"
        )
    checkpoint_path = Path(checkpoint_path)
    payload = load_checkpoint(checkpoint_path, map_location=device)
    config = config_from_checkpoint(payload, device=device, batch_size=batch_size)
    if seed is not None:
        config = TrainConfig(**{**config.to_dict(), "seed": seed})
    set_seed(config.seed)

    if split is None:
        split, _before_a, _before_b = load_and_split_for_config(config)
    if expected_fingerprint is not None and split.fingerprint != expected_fingerprint:
        raise RuntimeError(
            "split fingerprint mismatch: "
            f"got {split.fingerprint}, expected {expected_fingerprint}"
        )
    examples = {
        "train": split.train,
        "dev": split.dev,
        "test": split.test,
    }[split_name]
    expected_test_sizes = {
        EXPECTED_STAGE_A_SPLIT_FINGERPRINT: DEFAULT_TEST_SIZE,
        STAGE_A_V2_SPLIT_FINGERPRINT: STAGE_A_V2_TEST_SIZE,
    }
    if split_name == "test" and expected_fingerprint in expected_test_sizes:
        expected_n = expected_test_sizes[expected_fingerprint]
        if len(examples) != expected_n:
            raise RuntimeError(
                f"expected {expected_n} test examples, got {len(examples)}"
            )

    if model is None:
        model = build_model(config)
        _ = model.encode([examples[0].query])
        assert_encoder_frozen(model)
    missing = model.load_state_dict(payload["model_head_state_dict"], strict=True)
    if missing.missing_keys or missing.unexpected_keys:
        raise RuntimeError(
            "checkpoint state_dict mismatch: "
            f"missing={missing.missing_keys} unexpected={missing.unexpected_keys}"
        )
    assert_encoder_frozen(model)

    if eval_mode == "teacher-forced":
        metrics: EvalMetrics | Any = evaluate_examples(
            model,
            examples,
            batch_size=config.batch_size,
            seed=config.seed,
            max_batches=None,
        )
    else:
        from tiergraph.planner.free_eval import evaluate_free_examples

        metrics = evaluate_free_examples(
            model,
            examples,
            batch_size=config.batch_size,
            seed=config.seed,
            max_batches=None,
        )
    return CheckpointEvalResult(
        checkpoint_path=str(checkpoint_path),
        split_name=split_name,
        split_fingerprint=split.fingerprint,
        expected_fingerprint=expected_fingerprint,
        n_examples=len(examples),
        eval_mode=eval_mode,
        metrics=metrics,
        checkpoint_seed=int(payload.get("seed", config.seed)),
        checkpoint_best_dev_loss=(
            float(payload["best_dev_loss"])
            if payload.get("best_dev_loss") is not None
            else None
        ),
        checkpoint_epoch=(
            int(payload["epoch"]) if payload.get("epoch") is not None else None
        ),
    )


@dataclass(frozen=True, slots=True)
class TrainRunResult:
    """Summary of a training run."""

    config: TrainConfig
    split_fingerprint: str
    trainable_params: int
    frozen_encoder_params: int
    history: tuple[dict[str, Any], ...]
    best_dev_loss: float
    best_dev_metrics: dict[str, Any] | None
    final_train_loss: dict[str, float] | None
    checkpoint_path: str | None
    smoke_first_loss: float | None = None
    smoke_last_loss: float | None = None


def run_training(
    config: TrainConfig,
    *,
    model: PlannerModel | None = None,
    split: StageASplitResult | None = None,
    annotation_fingerprints: tuple[tuple[int, str], tuple[int, str]] | None = None,
) -> TrainRunResult:
    """Run the thin Stage-A training loop (supports ``--smoke``)."""
    set_seed(config.seed)
    step_a_path = Path(config.step_a_path)
    step_b_path = Path(config.step_b_path)
    if split is None:
        split, before_a, before_b = load_and_split_for_config(config)
    else:
        if annotation_fingerprints is None:
            before_a = fingerprint_file(step_a_path)
            before_b = fingerprint_file(step_b_path)
        else:
            before_a, before_b = annotation_fingerprints

    if model is None:
        model = build_model(config)
        _ = model.encode([split.train[0].query])
        assert_encoder_frozen(model)

    optimizer = build_optimizer(model, lr=config.lr)
    trainable_n, _ = count_parameters(head_parameters(model))
    frozen_n, _ = count_parameters(encoder_parameters(model))

    max_train_batches = config.smoke_train_batches if config.smoke else None
    max_eval_batches = config.smoke_eval_batches if config.smoke else None
    epochs = 1 if config.smoke else config.epochs

    history: list[dict[str, Any]] = []
    best_dev_loss = float("inf")
    best_dev_metrics: dict[str, Any] | None = None
    final_train_loss: dict[str, float] | None = None
    smoke_first_loss: float | None = None
    smoke_last_loss: float | None = None
    checkpoint_path: Path | None = None
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_meter = LossMeter()
        batches = iter_example_batches(
            split.train,
            batch_size=config.batch_size,
            seed=config.seed,
            epoch=epoch,
            max_batches=max_train_batches,
            shuffle=True,
        )
        for batch in batches:
            breakdown = train_step(model, optimizer, batch)
            loss_value = float(breakdown.total.detach().item())
            if smoke_first_loss is None:
                smoke_first_loss = loss_value
            smoke_last_loss = loss_value
            train_meter.update(breakdown)
        final_train_loss = train_meter.means()

        dev_metrics = evaluate_examples(
            model,
            split.dev,
            batch_size=config.batch_size,
            seed=config.seed,
            max_batches=max_eval_batches,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": final_train_loss,
            "dev": dev_metrics.to_dict(),
        }
        history.append(epoch_record)
        if dev_metrics.mean_loss["total"] < best_dev_loss:
            best_dev_loss = dev_metrics.mean_loss["total"]
            best_dev_metrics = dev_metrics.to_dict()
            checkpoint_path = output_dir / "best.pt"
            save_checkpoint(
                checkpoint_path,
                model=model,
                config=config,
                split_fingerprint=split.fingerprint,
                best_dev_loss=best_dev_loss,
                best_dev_metrics=best_dev_metrics,
                epoch=epoch,
                extra={
                    "trainable_params": trainable_n,
                    "frozen_encoder_params": frozen_n,
                },
            )

        assert_annotations_unchanged(
            step_a_path=step_a_path,
            step_b_path=step_b_path,
            before_a=before_a,
            before_b=before_b,
        )

    final_path = output_dir / ("smoke.pt" if config.smoke else "last.pt")
    save_checkpoint(
        final_path,
        model=model,
        config=config,
        split_fingerprint=split.fingerprint,
        best_dev_loss=best_dev_loss,
        best_dev_metrics=best_dev_metrics,
        epoch=epochs - 1,
        extra={
            "trainable_params": trainable_n,
            "frozen_encoder_params": frozen_n,
            "history": history,
        },
    )
    if checkpoint_path is None:
        checkpoint_path = final_path

    config_path = output_dir / "train_config.json"
    config_path.write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "split_fingerprint": split.fingerprint,
                "device": str(config.device),
                "trainable_params": trainable_n,
                "frozen_encoder_params": frozen_n,
                "best_dev_loss": best_dev_loss,
                "best_dev_metrics": best_dev_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return TrainRunResult(
        config=config,
        split_fingerprint=split.fingerprint,
        trainable_params=trainable_n,
        frozen_encoder_params=frozen_n,
        history=tuple(history),
        best_dev_loss=best_dev_loss,
        best_dev_metrics=best_dev_metrics,
        final_train_loss=final_train_loss,
        checkpoint_path=str(checkpoint_path),
        smoke_first_loss=smoke_first_loss,
        smoke_last_loss=smoke_last_loss,
    )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_LR",
    "EXPECTED_STAGE_A_SPLIT_FINGERPRINT",
    "STAGE_A_V2_SPLIT_FINGERPRINT",
    "CheckpointEvalResult",
    "EvalMetrics",
    "LossMeter",
    "TrainConfig",
    "TrainRunResult",
    "accumulate_head_metrics",
    "assert_annotations_unchanged",
    "assert_encoder_frozen",
    "build_model",
    "build_optimizer",
    "config_from_checkpoint",
    "count_parameters",
    "encode_gold_batch",
    "encoder_parameters",
    "evaluate_checkpoint",
    "evaluate_examples",
    "head_parameters",
    "iter_example_batches",
    "load_and_split_stage_a",
    "load_and_split_stage_a_v2",
    "load_and_split_for_config",
    "load_checkpoint",
    "run_training",
    "save_checkpoint",
    "set_seed",
    "train_step",
]
