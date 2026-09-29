#!/usr/bin/env python3
"""TRAIN-only 5-fold comparison of H4.2 candidate selection against H4.1.

Each job trains on four frozen TRAIN folds for five epochs and scores the
final model on the held-out TRAIN fold. H4.2 is an extra selector on the same
frozen MiniLM forward. The H4.1 BIO loss stays at lambda_start=0.0 and
lambda_end=0.25. DEV and TEST annotations are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_H4_SPEC = importlib.util.spec_from_file_location(
    "run_planner_h4_cv",
    ROOT / "scripts" / "run_planner_h4_cv.py",
)
if _H4_SPEC is None or _H4_SPEC.loader is None:
    raise RuntimeError("H4.1 CV driver is missing")
h4cv = importlib.util.module_from_spec(_H4_SPEC)
_H4_SPEC.loader.exec_module(h4cv)

from tiergraph.planner.annotation_step_a import StageAStepAAnnotation  # noqa: E402
from tiergraph.planner.annotation_step_b import StageAStepBAnnotation  # noqa: E402
from tiergraph.planner.batching import INDEX_TO_IMPLICIT  # noqa: E402
from tiergraph.planner.decode import (  # noqa: E402
    GraphDecoder,
    PlannerDecodeError,
    PredictedAnchor,
    PlannerPredictions,
)
from tiergraph.planner.free_eval import (  # noqa: E402
    evaluate_free_predictions,
    predict_batch,
)
from tiergraph.planner.h42_selector import (  # noqa: E402
    H42_MAX_SPAN_TOKENS,
    H42CandidateSelector,
    candidate_targets,
    candidate_token_mask,
    enumerate_word_spans,
    h42_loss,
    operation_token_mask,
    select_candidates,
)
from tiergraph.planner.loss import planner_loss  # noqa: E402
from tiergraph.planner.model import masked_mean_pool  # noqa: E402
from tiergraph.planner.naming import derive_anchor_normalized_name  # noqa: E402
from tiergraph.planner.stage_a_to_corpus import step_ab_to_planner_example  # noqa: E402
from tiergraph.planner.stage_a_v3_h4_build import load_annotation_rows_for_ids  # noqa: E402
from tiergraph.planner.targets import build_planner_targets  # noqa: E402
from tiergraph.planner.train import (  # noqa: E402
    TrainConfig,
    assert_encoder_frozen,
    build_model,
    build_optimizer,
    encoder_parameters,
    iter_example_batches,
)

CV_FOLD_FINGERPRINT = h4cv.CV_FOLD_FINGERPRINT
SPLIT_FINGERPRINT = h4cv.SPLIT_FINGERPRINT
ANNOTATION_FINGERPRINT = h4cv.ANNOTATION_FINGERPRINT
BASE_SEED = h4cv.BASE_SEED
EPOCHS = h4cv.EPOCHS
BATCH_SIZE = h4cv.BATCH_SIZE
LR = h4cv.LR
DEVICE = h4cv.DEVICE
DECODE = h4cv.DECODE
FINAL_EPOCH_ONLY = True
LAMBDA_START = 0.0
LAMBDA_END = 0.25
MEANINGFUL_F1_GAIN = h4cv.TIE_F1_GAP
H41_RESULTS = ROOT / "artifacts" / "planner_h4_cv" / "per_fold_results.jsonl"
DEFAULT_MANIFEST = h4cv.DEFAULT_MANIFEST
DEFAULT_OUTPUT = ROOT / "artifacts" / "planner_h42_cv"
METADATA_NAME = "experiment_metadata.json"
PER_FOLD_NAME = "per_fold_results.jsonl"
SUMMARY_NAME = "summary.json"
SELECTION_RULE = (
    "Keep a candidate when its score is strictly greater than NONE. "
    "No threshold grid."
)


def fold_seed(fold: int) -> int:
    return h4cv.fold_seed(fold, BASE_SEED)


def id_hash(ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_train_only_split(
    train_ids: Sequence[str],
    eval_ids: Sequence[str],
    *,
    train_universe: set[str],
    dev_ids: set[str],
    test_ids: set[str],
) -> None:
    """Fail if a fold leaks into DEV, TEST, or its own training ids."""
    train_set = set(train_ids)
    eval_set = set(eval_ids)
    if train_set & eval_set:
        raise RuntimeError("held-out fold overlaps its training ids")
    if not train_set <= train_universe or not eval_set <= train_universe:
        raise RuntimeError("fold ids are not a subset of TRAIN")
    if train_set & dev_ids or eval_set & dev_ids:
        raise RuntimeError("DEV was requested; refusing to continue")
    if train_set & test_ids or eval_set & test_ids:
        raise RuntimeError("TEST was requested; refusing to continue")
    if train_set | eval_set != train_universe:
        raise RuntimeError("a fold does not cover the frozen TRAIN set")


def fold_jobs(mapping: dict[str, int], split_ids: dict[str, set[str]]) -> list[dict[str, Any]]:
    """One job per frozen fold. Lambda is fixed at the selected H4.1 pair."""
    jobs = []
    covered: list[str] = []
    for fold in range(5):
        eval_ids = sorted(sid for sid, assigned in mapping.items() if assigned == fold)
        train_ids = sorted(sid for sid, assigned in mapping.items() if assigned != fold)
        assert_train_only_split(
            train_ids,
            eval_ids,
            train_universe=set(mapping),
            dev_ids=split_ids["dev"],
            test_ids=split_ids["test"],
        )
        if set(covered) & set(eval_ids):
            raise RuntimeError("evaluation fold repeats a TRAIN id")
        covered.extend(eval_ids)
        jobs.append(
            {
                "fold": fold,
                "seed": fold_seed(fold),
                "lambda_start": LAMBDA_START,
                "lambda_end": LAMBDA_END,
                "n_train": len(train_ids),
                "n_holdout": len(eval_ids),
                "train_ids_hash": id_hash(train_ids),
                "eval_ids_hash": id_hash(eval_ids),
                "train_ids": train_ids,
                "eval_ids": eval_ids,
            }
        )
    if set(covered) != set(mapping) or len(covered) != len(mapping):
        raise RuntimeError("the five held-out folds do not cover TRAIN exactly once")
    if len(jobs) != 5:
        raise RuntimeError(f"expected 5 fold jobs, got {len(jobs)}")
    return jobs


def supervised_anchor_pairs(targets: Any) -> list[tuple[int, int, int]]:
    """Gold ``(start, end, supervised owner)`` labels. Not used to build spans."""
    pairs = []
    for anchor, owner in zip(
        targets.supervised_anchors,
        targets.ownership_owner_indices,
        strict=True,
    ):
        pairs.append((int(anchor.start), int(anchor.end), int(owner)))
    return pairs


def free_h42_anchor_spans(
    selector: H42CandidateSelector,
    token_embeddings: Any,
    tokens: Any,
    query: str,
    operation_spans: Sequence[tuple[int, int]],
) -> tuple[tuple[Any, ...], tuple[tuple[int, ...], ...]]:
    """Select anchors from query candidates and already computed embeddings.

    Operation spans may be predicted. Gold spans are not an input.
    Returns selected ``(span, owner)`` pairs and the raw per-operation index
    tuples so NONE operations can be counted.
    """
    spans = enumerate_word_spans(query, max_tokens=H42_MAX_SPAN_TOKENS)
    if not operation_spans or not spans:
        return (), tuple(() for _ in operation_spans)
    candidate_mask = candidate_token_mask(spans, tokens)
    operation_mask = operation_token_mask(operation_spans, tokens)
    logits, none_logits = selector.score(
        token_embeddings,
        operation_mask,
        candidate_mask,
    )
    chosen = select_candidates(logits, none_logits)
    selected = []
    for owner, indexes in enumerate(chosen):
        for index in indexes:
            selected.append((spans[index], owner))
    return tuple(selected), chosen


def anchors_from_selection(
    query: str,
    selected: Sequence[tuple[Any, int]],
    implicit_indexes: Sequence[int],
) -> tuple[PredictedAnchor, ...]:
    """Build owned anchors. Implicit labels come from the existing H5 head."""
    if len(implicit_indexes) != len(selected):
        raise ValueError("one implicit label is required per selected span")
    anchors = []
    for (span, owner), impl_index in zip(selected, implicit_indexes, strict=True):
        text = query[span.start : span.end]
        normalized_name = None
        if text:
            try:
                normalized_name = derive_anchor_normalized_name(text)
            except ValueError:
                normalized_name = None
        anchors.append(
            PredictedAnchor(
                start=span.start,
                end=span.end,
                text=text,
                owner_index=owner,
                implicit_resolution=INDEX_TO_IMPLICIT[int(impl_index)],
                normalized_name=normalized_name,
            )
        )
    return tuple(anchors)


def span_prf(predicted: set[tuple[int, int]], gold: set[tuple[int, int]]) -> dict[str, float]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "h4_precision": precision,
        "h4_recall": recall,
        "h4_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def exact_anchor_set_match(
    predicted: set[tuple[int, int]],
    gold: set[tuple[int, int]],
) -> bool:
    return predicted == gold


def compare_means(
    h42_f1: float,
    h41_f1: float,
    h42_valid: float,
    h41_valid: float,
) -> dict[str, Any]:
    """Apply the pre-registered keep rule. Thresholds are not tuned on DEV."""
    gain = float(h42_f1) - float(h41_f1)
    valid_drop = float(h41_valid) - float(h42_valid)
    keep = gain >= MEANINGFUL_F1_GAIN and valid_drop <= MEANINGFUL_F1_GAIN
    return {
        "h42_mean_h4_f1": float(h42_f1),
        "h41_mean_h4_f1": float(h41_f1),
        "f1_gain": gain,
        "meaningful_f1_gain": MEANINGFUL_F1_GAIN,
        "valid_graph_drop": valid_drop,
        "h42_worth_keeping": keep,
        "dev_used_for_decision": False,
    }


def load_h41_baseline(path: Path = H41_RESULTS) -> list[dict[str, Any]] | None:
    """Selected H4.1 fold rows, if the frozen CV artifacts are already present."""
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if float(row["lambda_start"]) != LAMBDA_START or float(row["lambda_end"]) != LAMBDA_END:
            continue
        if str(row.get("cv_fold_fingerprint")) != CV_FOLD_FINGERPRINT:
            raise RuntimeError("H4.1 baseline fingerprint does not match the frozen folds")
        rows.append(row)
    rows.sort(key=lambda row: int(row["fold"]))
    if len(rows) != 5:
        raise RuntimeError(f"expected 5 H4.1 selected fold rows, got {len(rows)}")
    return rows


def experiment_metadata(jobs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    return {
        "git_commit": commit,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "cv_fold_fingerprint": CV_FOLD_FINGERPRINT,
        "annotation_fingerprint": ANNOTATION_FINGERPRINT,
        "base_seed": BASE_SEED,
        "fold_seeds": {str(job["fold"]): job["seed"] for job in jobs},
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "optimizer": "AdamW",
        "device": DEVICE,
        "decode": DECODE,
        "final_epoch_only": FINAL_EPOCH_ONLY,
        "lambda_start": LAMBDA_START,
        "lambda_end": LAMBDA_END,
        "h42_max_span_tokens": H42_MAX_SPAN_TOKENS,
        "h42_selection_rule": SELECTION_RULE,
        "train_eval_hashes": [
            {
                "fold": job["fold"],
                "train_ids_hash": job["train_ids_hash"],
                "eval_ids_hash": job["eval_ids_hash"],
            }
            for job in jobs
        ],
        "DEV_USED": False,
        "TEST_USED": False,
        "h42_worth_keeping_rule": (
            f"mean held-out H4 span F1 gains at least {MEANINGFUL_F1_GAIN} "
            "and valid-graph rate does not drop by more than that amount"
        ),
    }


def h42_batch_loss(
    selector: H42CandidateSelector,
    token_embeddings: Any,
    token_views: Sequence[Any],
    examples: Sequence[Any],
) -> Any:
    """Multi-label H4.2 loss on embeddings from the current planner forward."""
    import torch

    losses = []
    for index, (example, tokens) in enumerate(zip(examples, token_views, strict=True)):
        targets = build_planner_targets(example, tokens)
        operations = tuple((item.start, item.end) for item in targets.supervised_operations)
        if not operations:
            continue
        spans = enumerate_word_spans(example.query, max_tokens=H42_MAX_SPAN_TOKENS)
        candidate_mask = candidate_token_mask(spans, tokens).to(token_embeddings.device)
        operation_mask = operation_token_mask(operations, tokens).to(token_embeddings.device)
        representable = candidate_mask.sum(dim=1) > 0
        usable_operations = operation_mask.sum(dim=1) > 0
        logits, none_logits = selector.score(
            token_embeddings[index],
            operation_mask,
            candidate_mask,
        )
        cand_targets, none_targets = candidate_targets(
            spans,
            len(operations),
            supervised_anchor_pairs(targets),
        )
        losses.append(
            h42_loss(
                logits,
                none_logits,
                cand_targets.to(logits.device),
                none_targets.to(logits.device),
                representable,
                usable_operations,
            )
        )
    if not losses:
        return selector.none_vector.sum() * 0.0
    return torch.stack(losses).mean()


def train_h42_step(
    model: Any,
    selector: H42CandidateSelector,
    planner_optimizer: Any,
    selector_optimizer: Any,
    examples: Sequence[Any],
    *,
    lambda_start: float,
    lambda_end: float,
) -> None:
    """One planner step plus H4.2 on a single frozen MiniLM forward."""
    import torch

    if (float(lambda_start), float(lambda_end)) != (LAMBDA_START, LAMBDA_END):
        raise RuntimeError("H4.2 CV does not search lambdas")
    model.train()
    selector.train()
    assert_encoder_frozen(model)
    features, token_views, gold = encode_gold_batch(model, examples)
    shared = features.token_embeddings.detach().clone()
    outputs = model.forward_train(features, gold)
    breakdown = planner_loss(
        outputs,
        gold,
        h4_boundary_lambda_start=lambda_start,
        h4_boundary_lambda_end=lambda_end,
    )
    selector_loss = h42_batch_loss(selector, shared, token_views, examples)
    total = breakdown.total + selector_loss
    if not torch.isfinite(total):
        raise RuntimeError(f"non-finite total loss: {total.item()!r}")
    planner_optimizer.zero_grad(set_to_none=True)
    selector_optimizer.zero_grad(set_to_none=True)
    total.backward()
    for parameter in encoder_parameters(model):
        if parameter.grad is not None:
            raise RuntimeError("encoder received gradients")
    planner_optimizer.step()
    selector_optimizer.step()


def _implicit_indexes(model: Any, token_embeddings: Any, tokens: Any, selected: Sequence[Any]) -> list[int]:
    if not selected:
        return []
    masks = candidate_token_mask(tuple(span for span, _owner in selected), tokens)
    pooled = masked_mean_pool(
        token_embeddings.detach().unsqueeze(0),
        masks.unsqueeze(0).to(token_embeddings.device),
    ).squeeze(0)
    return [int(index) for index in model.impl_head(pooled).argmax(dim=-1).tolist()]


def predictions_with_h42(
    model: Any,
    selector: H42CandidateSelector,
    token_embeddings: Any,
    tokens: Any,
    query: str,
    bio_predictions: PlannerPredictions,
) -> tuple[PlannerPredictions, int, int]:
    """Replace BIO anchors with H4.2 selections. Operations stay free Viterbi."""
    operation_spans = tuple((op.start, op.end) for op in bio_predictions.operations)
    selected, chosen = free_h42_anchor_spans(
        selector,
        token_embeddings,
        tokens,
        query,
        operation_spans,
    )
    anchors = anchors_from_selection(
        query,
        selected,
        _implicit_indexes(model, token_embeddings, tokens, selected),
    )
    replaced = PlannerPredictions(
        operations=bio_predictions.operations,
        anchors=anchors,
        dependency_pairs=bio_predictions.dependency_pairs,
        aux_query_type=bio_predictions.aux_query_type,
    )
    return replaced, len(anchors), sum(1 for indexes in chosen if len(indexes) == 0)


def _slice_counts() -> dict[str, dict[str, int]]:
    return {}


def _add_slice(
    slices: dict[str, dict[str, int]],
    name: str,
    predicted: set[tuple[int, int]],
    gold: set[tuple[int, int]],
) -> None:
    row = slices.setdefault(name, {"n": 0, "exact": 0, "tp": 0, "fp": 0, "fn": 0})
    scored = span_prf(predicted, gold)
    row["n"] += 1
    row["exact"] += int(exact_anchor_set_match(predicted, gold))
    row["tp"] += int(scored["tp"])
    row["fp"] += int(scored["fp"])
    row["fn"] += int(scored["fn"])


def evaluate_holdout(
    model: Any,
    selector: H42CandidateSelector,
    examples: Sequence[Any],
) -> dict[str, Any]:
    """Free Viterbi operations plus H4.2 anchors, after the final epoch only."""
    import torch

    from tiergraph.pilot.gold_learned_harness import (
        GoldExampleView,
        compare_pair,
        make_stub_executor,
        op_group_for_graph,
    )

    model.eval()
    selector.eval()
    decoder = GraphDecoder()
    executor = make_stub_executor()
    pairs = []
    n_predicted = 0
    n_none = 0
    n_exact = 0
    slices = _slice_counts()
    n_semantic = 0
    n_tier = 0
    with torch.no_grad():
        for batch in iter_example_batches(
            examples,
            batch_size=BATCH_SIZE,
            seed=0,
            epoch=0,
            shuffle=False,
        ):
            features, items = predict_batch(model, batch, bio_decode_mode=DECODE)
            token_views = model.encoder.token_char_spans_for_batch(features)
            for index, (example, bio) in enumerate(zip(batch, items, strict=True)):
                replaced, n_anchors, n_empty_ops = predictions_with_h42(
                    model,
                    selector,
                    features.token_embeddings[index],
                    token_views[index],
                    example.query,
                    bio,
                )
                pairs.append((example, replaced))
                n_predicted += n_anchors
                n_none += n_empty_ops
                predicted = {(anchor.start, anchor.end) for anchor in replaced.anchors}
                gold = {
                    (anchor.start, anchor.end)
                    for anchor in example.planner_labels.slot_anchors
                }
                n_exact += int(exact_anchor_set_match(predicted, gold))
                bucket = str(example.metadata.get("final_bucket") or "UNKNOWN")
                group = op_group_for_graph(example.graph)
                _add_slice(slices, f"bucket:{bucket}", predicted, gold)
                _add_slice(slices, f"ops:{group}", predicted, gold)
                decoded_graph = None
                try:
                    decoded_graph = decoder.decode(
                        replaced,
                        query=example.query,
                        graph_id=f"pred::{example.example_id}",
                    ).graph
                except PlannerDecodeError:
                    decoded_graph = None
                record = compare_pair(
                    GoldExampleView(
                        example_id=example.example_id,
                        query=example.query,
                        graph=example.graph,
                        final_bucket=bucket,
                    ),
                    decoded_graph,
                    predict_ms=0.0,
                    executor=executor,
                )
                n_semantic += int(record.response_semantic_match)
                n_tier += int(record.tier_routing)
    metrics = evaluate_free_predictions(pairs, decoder=decoder)
    n_examples = len(examples)
    if int(metrics.n_examples) != n_examples:
        raise RuntimeError("held-out evaluation did not cover the fold")
    return {
        "metrics": metrics,
        "n_predicted_anchors": n_predicted,
        "n_none_operations": n_none,
        "exact_anchor_set_accuracy": n_exact / n_examples if n_examples else 0.0,
        "semantic_match_rate": n_semantic / n_examples if n_examples else 0.0,
        "tier_routing_rate": n_tier / n_examples if n_examples else 0.0,
        "slices": slices,
    }


def _load_train_examples(ids: set[str]) -> dict[str, Any]:
    step_a = load_annotation_rows_for_ids(
        ROOT / h4cv.STAGE_A_V3_STEP_A_PATH,
        ids,
        model_cls=StageAStepAAnnotation,
    )
    step_b = load_annotation_rows_for_ids(
        ROOT / h4cv.STAGE_A_V3_STEP_B_PATH,
        ids,
        model_cls=StageAStepBAnnotation,
    )
    examples = {}
    for sid in sorted(ids):
        example = step_ab_to_planner_example(step_a[sid], step_b[sid], use_semantic_h1=True)
        if example.example_id != sid:
            raise RuntimeError(f"example id {example.example_id} != {sid}")
        examples[sid] = example
    return examples


def execute_fold(
    job: dict[str, Any],
    examples_by_id: dict[str, Any],
    split_ids: dict[str, set[str]],
) -> dict[str, Any]:
    """Train one fresh fold model. The held-out fold is never in this training set."""
    import torch

    fold = int(job["fold"])
    seed = int(job["seed"])
    if seed != fold_seed(fold):
        raise RuntimeError("refusing a mismatched fold seed")
    if (float(job["lambda_start"]), float(job["lambda_end"])) != (LAMBDA_START, LAMBDA_END):
        raise RuntimeError("H4.2 CV does not search lambdas")
    train_ids = list(job["train_ids"])
    eval_ids = list(job["eval_ids"])
    assert_train_only_split(
        train_ids,
        eval_ids,
        train_universe=set(split_ids["train"]),
        dev_ids=split_ids["dev"],
        test_ids=split_ids["test"],
    )
    if id_hash(train_ids) != job["train_ids_hash"] or id_hash(eval_ids) != job["eval_ids_hash"]:
        raise RuntimeError("fold id hash does not match the planned split")
    train_examples = [examples_by_id[sid] for sid in train_ids]
    holdout = [examples_by_id[sid] for sid in eval_ids]
    torch.manual_seed(seed)
    model = build_model(
        TrainConfig(
            seed=seed,
            device=DEVICE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
        )
    )
    selector = H42CandidateSelector(model.hidden_size).to(torch.device(DEVICE))
    planner_optimizer = build_optimizer(model, lr=LR)
    selector_optimizer = torch.optim.AdamW(selector.parameters(), lr=LR)

    def train_epoch(epoch: int) -> None:
        for batch in iter_example_batches(
            train_examples,
            batch_size=BATCH_SIZE,
            seed=seed,
            epoch=epoch,
            shuffle=True,
        ):
            train_h42_step(
                model,
                selector,
                planner_optimizer,
                selector_optimizer,
                batch,
                lambda_start=LAMBDA_START,
                lambda_end=LAMBDA_END,
            )

    train_started = time.perf_counter()
    h4cv.run_fold_epochs(epochs=EPOCHS, train_epoch=train_epoch)
    train_seconds = time.perf_counter() - train_started
    eval_started = time.perf_counter()
    evaluated = evaluate_holdout(model, selector, holdout)
    eval_seconds = time.perf_counter() - eval_started
    metrics = evaluated["metrics"]
    return {
        "lambda_start": LAMBDA_START,
        "lambda_end": LAMBDA_END,
        "fold": fold,
        "seed": seed,
        "n_train": len(train_examples),
        "n_holdout": len(holdout),
        "train_ids_hash": job["train_ids_hash"],
        "eval_ids_hash": job["eval_ids_hash"],
        "h4_precision": metrics.anchor_span_precision,
        "h4_recall": metrics.anchor_span_recall,
        "h4_f1": metrics.anchor_span_f1,
        "exact_anchor_set_accuracy": evaluated["exact_anchor_set_accuracy"],
        "h5_accuracy_span_aligned": metrics.h5_accuracy_span_aligned,
        "h6_ownership_accuracy_span_aligned": metrics.h6_ownership_accuracy_span_aligned,
        "operation_span_f1": metrics.operation_span_f1,
        "operation_joint_f1": metrics.operation_joint_f1,
        "query_type_accuracy": metrics.query_type_accuracy,
        "exact_graph_rate": metrics.canonical_exact_graph_accuracy,
        "valid_graph_rate": metrics.valid_graph_rate,
        "tier_routing_rate": evaluated["tier_routing_rate"],
        "semantic_match_rate": evaluated["semantic_match_rate"],
        "n_predicted_anchors": evaluated["n_predicted_anchors"],
        "n_none_operations": evaluated["n_none_operations"],
        "slices": evaluated["slices"],
        "training_seconds": train_seconds,
        "evaluation_seconds": eval_seconds,
        "epochs": EPOCHS,
        "final_epoch_only": True,
        "decode": DECODE,
        "DEV_USED": False,
        "TEST_USED": False,
        "cv_fold_fingerprint": CV_FOLD_FINGERPRINT,
    }


def aggregate_against_h41(
    rows: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]] | None,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["fold"]))
    f1 = [float(row["h4_f1"]) for row in ordered]
    summary: dict[str, Any] = {
        "n_folds": len(ordered),
        "mean_h4_f1": statistics.fmean(f1) if f1 else None,
        "mean_h4_precision": statistics.fmean(float(row["h4_precision"]) for row in ordered) if ordered else None,
        "mean_h4_recall": statistics.fmean(float(row["h4_recall"]) for row in ordered) if ordered else None,
        "mean_exact_anchor_set_accuracy": statistics.fmean(
            float(row["exact_anchor_set_accuracy"]) for row in ordered
        ) if ordered else None,
        "mean_operation_joint_f1": statistics.fmean(float(row["operation_joint_f1"]) for row in ordered) if ordered else None,
        "mean_valid_graph_rate": statistics.fmean(float(row["valid_graph_rate"]) for row in ordered) if ordered else None,
        "mean_tier_routing_rate": statistics.fmean(float(row["tier_routing_rate"]) for row in ordered) if ordered else None,
        "mean_semantic_match_rate": statistics.fmean(float(row["semantic_match_rate"]) for row in ordered) if ordered else None,
        "DEV_USED": False,
        "TEST_USED": False,
    }
    if baseline is None or not ordered:
        summary["h41_baseline"] = None
        summary["comparison"] = None
        return summary
    by_fold = {int(row["fold"]): row for row in baseline}
    if set(by_fold) != {int(row["fold"]) for row in ordered}:
        raise RuntimeError("H4.1 baseline folds do not match the H4.2 jobs")
    summary["h41_baseline"] = {
        "lambda_start": LAMBDA_START,
        "lambda_end": LAMBDA_END,
        "mean_h4_f1": statistics.fmean(float(by_fold[int(row["fold"])]["h4_f1"]) for row in ordered),
        "mean_valid_graph_rate": statistics.fmean(
            float(by_fold[int(row["fold"])]["valid_graph_rate"]) for row in ordered
        ),
    }
    summary["comparison"] = compare_means(
        summary["mean_h4_f1"],
        summary["h41_baseline"]["mean_h4_f1"],
        summary["mean_valid_graph_rate"],
        summary["h41_baseline"]["mean_valid_graph_rate"],
    )
    return summary


def print_dry_run(
    jobs: Sequence[dict[str, Any]],
    baseline: list[dict[str, Any]] | None,
    metadata: dict[str, Any],
) -> None:
    print(f"N_JOBS {len(jobs)}")
    print("FOLD_SIZES " + json.dumps([job["n_holdout"] for job in jobs]))
    print("SEEDS " + " ".join(f"{job['fold']}:{job['seed']}" for job in jobs))
    print(f"LAMBDA {LAMBDA_START:.2f} {LAMBDA_END:.2f}")
    print(f"EPOCHS {EPOCHS}")
    print(f"BATCH_SIZE {BATCH_SIZE}")
    print(f"LR {LR}")
    print("OPTIMIZER AdamW")
    print(f"DEVICE {DEVICE}")
    print(f"DECODE {DECODE}")
    print(f"FINAL_EPOCH_ONLY {str(FINAL_EPOCH_ONLY).lower()}")
    print(f"H42_MAX_SPAN_TOKENS {H42_MAX_SPAN_TOKENS}")
    print(f"SELECTION {SELECTION_RULE}")
    print("DEV_USED false")
    print("TEST_USED false")
    print(f"GIT_COMMIT {metadata['git_commit']}")
    print(f"SPLIT_FINGERPRINT {SPLIT_FINGERPRINT}")
    print(f"CV_FOLD_FINGERPRINT {CV_FOLD_FINGERPRINT}")
    if baseline is None:
        print(f"H41_BASELINE missing {H41_RESULTS}")
    else:
        mean_f1 = statistics.fmean(float(row["h4_f1"]) for row in baseline)
        print(f"H41_BASELINE folds={len(baseline)} mean_h4_f1={mean_f1:.6f}")
    for job in jobs:
        print(
            "job "
            f"fold={job['fold']} seed={job['seed']} "
            f"n_train={job['n_train']} n_holdout={job['n_holdout']} "
            f"train={job['train_ids_hash']} eval={job['eval_ids_hash']}"
        )


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the five frozen fold jobs without training or loading annotations",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="run the five TRAIN folds; omitted by the review dry-run",
    )
    args = parser.parse_args(argv)
    if args.dry_run and args.train:
        raise RuntimeError("choose either --dry-run or --train")
    split_ids = h4cv.load_split_ids(ROOT / h4cv.STAGE_A_V3_SPLIT_PATH)
    payload = h4cv.load_fold_manifest(args.manifest)
    mapping = h4cv.validate_fold_manifest(payload, split_ids)
    jobs = fold_jobs(mapping, split_ids)
    metadata = experiment_metadata(jobs)
    if args.dry_run:
        print_dry_run(jobs, load_h41_baseline(), metadata)
        return
    if not args.train:
        raise RuntimeError(
            "pass --dry-run to inspect the five folds, or --train after review; "
            "this driver does not fall back to DEV"
        )
    if set(mapping) & split_ids["dev"] or set(mapping) & split_ids["test"]:
        raise RuntimeError("DEV or TEST ids are in the fold map")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    examples = _load_train_examples(set(mapping))
    rows = []
    results_path = args.output_dir / PER_FOLD_NAME
    for job in jobs:
        row = execute_fold(job, examples, split_ids)
        rows.append(row)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = aggregate_against_h41(rows, load_h41_baseline())
    (args.output_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def encode_gold_batch(model: Any, examples: Sequence[Any]) -> Any:
    """Local name so the training step has exactly one encoder entry point."""
    from tiergraph.planner.train import encode_gold_batch as _encode_gold_batch

    return _encode_gold_batch(model, examples)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
