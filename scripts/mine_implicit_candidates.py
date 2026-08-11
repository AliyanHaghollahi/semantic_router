#!/usr/bin/env python3
"""Mine Personal/Environmental queries that may be MIXED_IMPLICIT.

Heuristic shortlist for human review only. Never mutates ground-truth labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.corpus import DEFAULT_CANDIDATE_SEED
from tiergraph.planner.implicit_mining import (
    DEFAULT_IMPLICIT_LIMIT,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_REVIEWS_PATH,
    DEFAULT_TRAIN_PATH,
    build_mining_inputs,
    mine_implicit_candidates,
    summarize_implicit_mining,
    write_implicit_candidates_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / DEFAULT_TRAIN_PATH,
        help="classification dataset JSON",
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=ROOT / DEFAULT_REVIEWS_PATH,
        help="completed Mixed review JSONL (excluded by normalized query)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / DEFAULT_OUTPUT_PATH,
        help="shortlist JSONL output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_IMPLICIT_LIMIT,
        help="max candidates to write (default 80)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CANDIDATE_SEED,
        help="tie-break seed (default Phase-5 seed 20260811)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print mining summary without writing output",
    )
    args = parser.parse_args(argv)

    unique_pool, excluded = build_mining_inputs(
        train_path=args.input,
        reviews_path=args.reviews if args.reviews.is_file() else None,
    )
    summary = summarize_implicit_mining(
        unique_pool,
        excluded_keys=excluded,
        limit=args.limit,
        seed=args.seed,
    )

    print(f"unique Personal:              {summary['unique_personal']}")
    print(f"unique Environmental:         {summary['unique_environmental']}")
    print(
        f"eligible Personal+Environmental "
        f"(after exclusions): {summary['eligible_personal_environmental']}"
    )
    print(
        f"satisfying mining criteria:   {summary['satisfying_mining_criteria']}"
    )
    print(f"shortlist size (limit={summary['limit']}): {summary['shortlist_written']}")

    if args.summary:
        return 0

    candidates = mine_implicit_candidates(
        unique_pool,
        excluded_keys=excluded,
        limit=args.limit,
        seed=args.seed,
    )
    write_implicit_candidates_jsonl(args.output, candidates)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
