#!/usr/bin/env python3
"""Validate Stage-A Step-A annotation corpus against the frozen selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.align import TokenCharSpan
from tiergraph.planner.annotation_step_a import (
    DEFAULT_FROZEN_SELECTION_PATH,
    DEFAULT_STEP_A_ANNOTATIONS_PATH,
    ensure_step_a_annotations_initialized,
    fingerprint_file,
    format_step_a_progress,
    load_step_a_annotations,
    summarize_step_a_progress,
    validate_step_a_corpus,
)


def _synthetic_token_view(query: str) -> tuple[TokenCharSpan, ...]:
    """Deterministic per-character-ish view for offline alignment checks."""
    tokens: list[TokenCharSpan] = [
        TokenCharSpan(None, None, is_special=True, is_padding=False),
    ]
    for index, _char in enumerate(query):
        if index % 2 == 1:
            continue
        end = min(index + 2, len(query))
        tokens.append(
            TokenCharSpan(index, end, is_special=False, is_padding=False)
        )
    tokens.append(TokenCharSpan(None, None, is_special=True, is_padding=False))
    return tuple(tokens)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / DEFAULT_FROZEN_SELECTION_PATH,
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / DEFAULT_STEP_A_ANNOTATIONS_PATH,
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="initialize annotations from frozen selection if missing",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="require all 120 examples to be COMPLETE",
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="skip tokenizer/alignment checks",
    )
    args = parser.parse_args(argv)

    before = fingerprint_file(args.selection)
    if args.init or not args.annotations.is_file():
        records = ensure_step_a_annotations_initialized(
            selection_path=args.selection,
            annotations_path=args.annotations,
        )
    else:
        records = load_step_a_annotations(args.annotations)
    after = fingerprint_file(args.selection)
    if before != after:
        print(
            "ERROR: frozen selection file was mutated during validation",
            file=sys.stderr,
        )
        return 1

    token_factory = None if args.skip_alignment else _synthetic_token_view
    errors = validate_step_a_corpus(
        records,
        selection_path=args.selection,
        token_view_factory=token_factory,
        require_all_complete=args.require_complete,
    )
    summary = summarize_step_a_progress(records)
    print(format_step_a_progress(summary))
    print(f"annotation_rows: {len(records)}")
    print(f"frozen_selection_fingerprint: {before[1][:16]}...")
    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("VALIDATION OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
