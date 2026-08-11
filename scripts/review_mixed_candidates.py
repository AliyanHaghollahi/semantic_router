#!/usr/bin/env python3
"""Interactive human review for Stage-A Mixed candidates.

Assigns ONLY high-level buckets:
  MIXED_IMPLICIT / MIXED_PARALLEL / MIXED_SEQUENTIAL / NOT_SUITABLE

Does not invent operations, anchors, ownership, dependencies, or graphs.
Does not modify ``dataset/planner/stage_a_candidates.jsonl``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.mixed_review import (
    BUCKET_DEFINITIONS,
    MixedReviewSession,
    format_summary,
    load_mixed_candidates,
    load_mixed_reviews,
    parse_review_command,
    summarize_mixed_reviews,
)


DEFAULT_CANDIDATES = ROOT / "dataset" / "planner" / "stage_a_candidates.jsonl"
DEFAULT_REVIEWS = ROOT / "dataset" / "planner" / "stage_a_mixed_reviews.jsonl"


def _print_definitions() -> None:
    print("Definitions:")
    for bucket, text in BUCKET_DEFINITIONS:
        print(f"  {bucket.name}:")
        print(f"    {text}")
    print()


def _print_prompt(session: MixedReviewSession) -> None:
    candidate = session.current()
    if candidate is None:
        print("All Mixed candidates have been reviewed.")
        print(format_summary(session.summary()))
        return
    position, total = session.current_position()
    print("-" * 50)
    print(f"Mixed candidate {position} / {total}")
    print(f"ID: {candidate.source_query_id}")
    print()
    print("QUERY:")
    print(candidate.query)
    print()
    print("Choose:")
    print()
    print("1 = MIXED_IMPLICIT")
    print("2 = MIXED_PARALLEL")
    print("3 = MIXED_SEQUENTIAL")
    print("4 = NOT_SUITABLE")
    print("s = skip")
    print("b = back")
    print("p = progress summary")
    print("q = save and quit")
    print("-" * 50)


def run_summary(candidates_path: Path, reviews_path: Path) -> int:
    mixed = load_mixed_candidates(candidates_path)
    reviews = load_mixed_reviews(reviews_path)
    summary = summarize_mixed_reviews(mixed, reviews)
    print(format_summary(summary))
    return 0


def run_interactive(candidates_path: Path, reviews_path: Path) -> int:
    mixed = load_mixed_candidates(candidates_path)
    if len(mixed) != 127:
        print(
            f"warning: expected 127 Mixed candidates, found {len(mixed)}",
            file=sys.stderr,
        )
    session = MixedReviewSession(mixed, reviews_path=reviews_path)
    _print_definitions()
    print(format_summary(session.summary()))
    print()

    try:
        while True:
            if session.current() is None:
                print("Review complete.")
                print(format_summary(session.summary()))
                session.save()
                return 0
            _print_prompt(session)
            try:
                raw = input("> ")
            except EOFError:
                print("\nEOF — saving and quitting.")
                session.save()
                print(format_summary(session.summary()))
                return 0
            try:
                action, bucket = parse_review_command(raw)
            except ValueError as exc:
                print(f"{exc}. Try 1/2/3/4/s/b/p/q.")
                continue
            if action == "assign":
                assert bucket is not None
                record = session.apply_bucket(bucket, persist=True)
                print(
                    f"saved {record.source_query_id} -> {record.planner_bucket.name}"
                )
            elif action == "skip":
                session.skip()
                print("skipped")
            elif action == "back":
                previous = session.back(persist=True)
                if previous is None:
                    print("nothing to go back to")
                else:
                    print(f"back to {previous.source_query_id}")
            elif action == "summary":
                print(format_summary(session.summary()))
            elif action == "quit":
                session.save()
                print("saved.")
                print(format_summary(session.summary()))
                return 0
    except KeyboardInterrupt:
        print("\nInterrupted — saving.")
        session.save()
        print(format_summary(session.summary()))
        return 130


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES,
        help="Stage-A candidates JSONL (read-only)",
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=DEFAULT_REVIEWS,
        help="Mixed review output JSONL",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print review progress without starting interactive mode",
    )
    args = parser.parse_args(argv)

    if args.summary:
        return run_summary(args.candidates, args.reviews)
    return run_interactive(args.candidates, args.reviews)


if __name__ == "__main__":
    raise SystemExit(main())
