#!/usr/bin/env python3
"""Thin reproducible training/evaluation entrypoint for the H1-H7 planner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.stage_a_split import DEFAULT_SPLIT_SEED
from tiergraph.planner.stage_a_v2_spec import (
    STAGE_A_V2_SPLIT_FINGERPRINT,
    STAGE_A_V2_SPLIT_SEED,
    STAGE_A_V2_STEP_A_PATH,
    STAGE_A_V2_STEP_B_PATH,
)
from tiergraph.planner.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    EXPECTED_STAGE_A_SPLIT_FINGERPRINT,
    TrainConfig,
    evaluate_checkpoint,
    load_and_split_for_config,
    run_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/planner_train"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a tiny real train/eval loop (CPU-friendly)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="skip training; evaluate a checkpoint on a held-out split",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint path for --eval-only (e.g. artifacts/planner_run1/best.pt)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test"),
        default="test",
        help="split to evaluate in --eval-only mode (default: test)",
    )
    parser.add_argument(
        "--eval-mode",
        choices=("teacher-forced", "free"),
        default="teacher-forced",
        help=(
            "teacher-forced: gold-structure head metrics; "
            "free: predict_structures -> GraphDecoder predicted-graph metrics"
        ),
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="use Stage-A v2 corpus (480 examples, frozen 384/48/48 split)",
    )
    parser.add_argument(
        "--expected-fingerprint",
        type=str,
        default=None,
        help="require this Stage-A split fingerprint (empty string disables)",
    )
    return parser


def _resolve_cli_defaults(args: argparse.Namespace) -> tuple[str | None, TrainConfig]:
    if args.v2:
        expected = (
            args.expected_fingerprint
            if args.expected_fingerprint is not None
            else STAGE_A_V2_SPLIT_FINGERPRINT
        )
        seed = args.seed if args.seed != DEFAULT_SPLIT_SEED else STAGE_A_V2_SPLIT_SEED
        config = TrainConfig(
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            output_dir=str(args.output_dir),
            smoke=bool(args.smoke),
            corpus_version="v2",
            step_a_path=str(STAGE_A_V2_STEP_A_PATH),
            step_b_path=str(STAGE_A_V2_STEP_B_PATH),
        )
    else:
        expected = (
            args.expected_fingerprint
            if args.expected_fingerprint is not None
            else EXPECTED_STAGE_A_SPLIT_FINGERPRINT
        )
        config = TrainConfig(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            output_dir=str(args.output_dir),
            smoke=bool(args.smoke),
        )
    if expected == "":
        expected = None
    return expected, config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected, config = _resolve_cli_defaults(args)
    if args.eval_only:
        if args.checkpoint is None:
            print("ERROR: --eval-only requires --checkpoint", file=sys.stderr)
            return 2
        split = None
        if args.v2:
            split, _before_a, _before_b = load_and_split_for_config(config)
        result = evaluate_checkpoint(
            args.checkpoint,
            split_name=args.split,
            seed=config.seed,
            device=args.device,
            batch_size=config.batch_size,
            expected_fingerprint=expected,
            split=split,
            eval_mode=args.eval_mode,
        )
        print("mode: eval-only")
        print(f"eval_mode: {result.eval_mode}")
        print(f"label: {result.to_dict()['label']}")
        print(f"checkpoint: {result.checkpoint_path}")
        print(f"split: {result.split_name}")
        print(f"n_examples: {result.n_examples}")
        print(f"split_fingerprint: {result.split_fingerprint}")
        print(f"expected_fingerprint: {result.expected_fingerprint}")
        print(f"checkpoint_seed: {result.checkpoint_seed}")
        print(f"checkpoint_epoch: {result.checkpoint_epoch}")
        print(f"checkpoint_best_dev_loss: {result.checkpoint_best_dev_loss}")
        print("metrics:", json.dumps(result.metrics.to_dict(), sort_keys=True))
        return 0

    print("config:", json.dumps(config.to_dict(), sort_keys=True))
    result = run_training(config)
    print(f"device: {config.device}")
    print(f"split_fingerprint: {result.split_fingerprint}")
    print(f"trainable_params: {result.trainable_params}")
    print(f"frozen_encoder_params: {result.frozen_encoder_params}")
    if result.final_train_loss is not None:
        print("train_loss:", json.dumps(result.final_train_loss, sort_keys=True))
    if result.smoke_first_loss is not None:
        print(f"smoke_first_loss: {result.smoke_first_loss:.6f}")
        print(f"smoke_last_loss: {result.smoke_last_loss:.6f}")
    if result.best_dev_metrics is not None:
        print("best_dev:", json.dumps(result.best_dev_metrics, sort_keys=True))
    if result.history:
        best_epoch = min(
            range(len(result.history)),
            key=lambda index: result.history[index]["dev"]["mean_loss"]["total"],
        )
        print(f"best_epoch: {best_epoch}")
    print(f"checkpoint: {result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
