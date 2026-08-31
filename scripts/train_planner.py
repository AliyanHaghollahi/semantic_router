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
from tiergraph.planner.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    TrainConfig,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=str(args.output_dir),
        smoke=bool(args.smoke),
    )
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
    print(f"checkpoint: {result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
