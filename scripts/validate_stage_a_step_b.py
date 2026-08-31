#!/usr/bin/env python3
"""Validate Stage-A Step-B annotations against frozen COMPLETE Step A."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.annotation_step_a import (
    DEFAULT_STEP_A_ANNOTATIONS_PATH,
    fingerprint_file,
    load_step_a_annotations,
)
from tiergraph.planner.annotation_step_b import (
    DEFAULT_STEP_B_ANNOTATIONS_PATH,
    ensure_step_b_annotations_initialized,
    format_step_b_progress,
    load_step_b_annotations,
    summarize_step_b_progress,
    validate_step_b_corpus,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step-a",
        type=Path,
        default=ROOT / DEFAULT_STEP_A_ANNOTATIONS_PATH,
    )
    parser.add_argument(
        "--step-b",
        type=Path,
        default=ROOT / DEFAULT_STEP_B_ANNOTATIONS_PATH,
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="initialize Step B from Step A if missing",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="require all 120 Step-B examples to be COMPLETE",
    )
    args = parser.parse_args(argv)

    before = fingerprint_file(args.step_a)
    step_a_records = load_step_a_annotations(args.step_a)
    if args.init or not args.step_b.is_file():
        records = ensure_step_b_annotations_initialized(
            step_a_path=args.step_a,
            step_b_path=args.step_b,
        )
    else:
        records = load_step_b_annotations(args.step_b)
    after = fingerprint_file(args.step_a)
    if before != after:
        print(
            "ERROR: frozen Step-A annotation file was mutated during validation",
            file=sys.stderr,
        )
        return 1

    errors = validate_step_b_corpus(
        records,
        step_a_records=step_a_records,
        step_a_path=args.step_a,
        require_all_complete=args.require_complete,
        step_a_fingerprint=before,
    )
    summary = summarize_step_b_progress(records)
    print(format_step_b_progress(summary))
    print(f"step_b_rows: {len(records)}")
    print(f"step_a_fingerprint: {before[1][:16]}...")
    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("VALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
