#!/usr/bin/env python3
"""Interactive ACCEPT/REJECT review for authored MIXED_SEQUENTIAL candidates.

Does not modify the candidate JSONL or training_data.json. Decisions are saved
to ``dataset/planner/stage_a_authored_sequential_reviews.jsonl``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.authored_sequential import (
    DEFAULT_AUTHORED_SEQUENTIAL_CANDIDATES_PATH,
    DEFAULT_AUTHORED_SEQUENTIAL_REVIEWS_PATH,
    AuthoredSequentialReviewSession,
    format_authored_sequential_summary,
    load_authored_sequential_candidates,
    load_authored_sequential_reviews,
    parse_authored_sequential_review_command,
    summarize_authored_sequential_reviews,
    validate_authored_sequential_candidate_set,
)


def run_summary(candidates_path: Path, reviews_path: Path) -> int:
    candidates = load_authored_sequential_candidates(candidates_path)
    validate_authored_sequential_candidate_set(candidates)
    reviews = load_authored_sequential_reviews(reviews_path)
    print(
        format_authored_sequential_summary(
            summarize_authored_sequential_reviews(candidates, reviews)
        )
    )
    return 0


def _print_prompt(session: AuthoredSequentialReviewSession) -> None:
    candidate = session.current()
    if candidate is None:
        print("All authored sequential candidates have been reviewed.")
        print(format_authored_sequential_summary(session.summary()))
        return
    position, total = session.current_position()
    print("-" * 50)
    print(f"Authored sequential candidate {position} / {total}")
    print(f"ID: {candidate.candidate_id}")
    print(f"dependency_family: {candidate.dependency_family}")
    print(f"template_group: {candidate.template_group}")
    print(f"semantic_group: {candidate.semantic_group}")
    print()
    print("QUERY:")
    print(candidate.query)
    print()
    print("operations:")
    for op in candidate.intended_operations:
        print(f"  - {op}")
    print("dependency edges:")
    for edge, typed in zip(
        candidate.intended_dependency_edges,
        candidate.intended_typed_values,
        strict=True,
    ):
        print(f"  - {edge}  [{typed}]")
    print()
    print(f"personal requirement: {candidate.intended_personal_requirement}")
    print(
        f"environmental requirement: "
        f"{candidate.intended_environmental_requirement}"
    )
    print(f"personal necessity: {candidate.personal_necessity_reason}")
    print(
        f"environmental necessity: "
        f"{candidate.environmental_necessity_reason}"
    )
    print()
    print("Choose:")
    print("1 = ACCEPT")
    print("2 = REJECT")
    print("s = skip")
    print("b = back")
    print("p = progress summary")
    print("q = save and quit")
    print("-" * 50)


def run_interactive(candidates_path: Path, reviews_path: Path) -> int:
    candidates = load_authored_sequential_candidates(candidates_path)
    session = AuthoredSequentialReviewSession(
        candidates,
        reviews_path=reviews_path,
    )
    print(format_authored_sequential_summary(session.summary()))
    print()
    try:
        while True:
            if session.current() is None:
                print("Review complete.")
                print(format_authored_sequential_summary(session.summary()))
                session.save()
                return 0
            _print_prompt(session)
            try:
                raw = input("> ")
            except EOFError:
                print("\nEOF — saving and quitting.")
                session.save()
                print(format_authored_sequential_summary(session.summary()))
                return 0
            try:
                action, status = parse_authored_sequential_review_command(raw)
            except ValueError as exc:
                print(f"{exc}. Try 1/2/s/b/p/q.")
                continue
            if action == "assign":
                assert status is not None
                record = session.apply_status(status, persist=True)
                print(
                    f"saved {record.candidate_id} -> {record.review_status.value}"
                )
            elif action == "skip":
                session.skip()
                print("skipped")
            elif action == "back":
                previous = session.back(persist=True)
                if previous is None:
                    print("nothing to go back to")
                else:
                    print(f"back to {previous.candidate_id}")
            elif action == "summary":
                print(format_authored_sequential_summary(session.summary()))
            elif action == "quit":
                session.save()
                print("saved.")
                print(format_authored_sequential_summary(session.summary()))
                return 0
    except KeyboardInterrupt:
        print("\nInterrupted — saving.")
        session.save()
        print(format_authored_sequential_summary(session.summary()))
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / DEFAULT_AUTHORED_SEQUENTIAL_CANDIDATES_PATH,
        help="authored sequential candidates JSONL",
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=ROOT / DEFAULT_AUTHORED_SEQUENTIAL_REVIEWS_PATH,
        help="authored sequential reviews JSONL",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print review progress and exit",
    )
    args = parser.parse_args(argv)
    if args.summary:
        return run_summary(args.candidates, args.reviews)
    return run_interactive(args.candidates, args.reviews)


if __name__ == "__main__":
    raise SystemExit(main())
