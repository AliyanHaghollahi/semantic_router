#!/usr/bin/env python3
"""Validate Stage-A final selection and spares manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.stage_a_selection import (
    DEFAULT_MIXED_REVIEWS_PATH,
    DEFAULT_SELECTION_PATH,
    DEFAULT_SPARES_PATH,
    format_selection_summary,
    load_jsonl,
    summarize_selection,
    validate_stage_a_selection,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / DEFAULT_SELECTION_PATH,
    )
    parser.add_argument(
        "--spares",
        type=Path,
        default=ROOT / DEFAULT_SPARES_PATH,
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=ROOT / DEFAULT_MIXED_REVIEWS_PATH,
    )
    args = parser.parse_args(argv)

    if not args.selection.is_file():
        print(f"ERROR: missing selection file: {args.selection}", file=sys.stderr)
        return 1
    if not args.spares.is_file():
        print(f"ERROR: missing spares file: {args.spares}", file=sys.stderr)
        return 1

    selected = load_jsonl(args.selection)
    spares = load_jsonl(args.spares)
    errors = validate_stage_a_selection(
        selected, spares, reviews_path=args.reviews
    )
    summary = summarize_selection(selected, spares)
    print(format_selection_summary(summary))
    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("VALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
