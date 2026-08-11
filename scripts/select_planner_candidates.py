#!/usr/bin/env python3
"""Build the Stage-A planner candidate review pool.

Reads ``dataset/training_data.json``, deduplicates to unique queries, and writes
``dataset/planner/stage_a_candidates.jsonl``.

This does NOT assign planner_bucket / Mixed structural categories and does NOT
fabricate planner labels.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.corpus import (
    DEFAULT_CANDIDATE_SEED,
    DEFAULT_ENVIRONMENTAL_CANDIDATES,
    DEFAULT_PERSONAL_CANDIDATES,
    build_unique_query_pool,
    load_classification_rows,
    select_stage_a_candidates,
    write_candidates_jsonl,
)


DEFAULT_INPUT = ROOT / "dataset" / "training_data.json"
DEFAULT_OUTPUT = ROOT / "dataset" / "planner" / "stage_a_candidates.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="classification JSON path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="candidate JSONL output path",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CANDIDATE_SEED)
    parser.add_argument(
        "--n-personal",
        type=int,
        default=DEFAULT_PERSONAL_CANDIDATES,
    )
    parser.add_argument(
        "--n-environmental",
        type=int,
        default=DEFAULT_ENVIRONMENTAL_CANDIDATES,
    )
    args = parser.parse_args(argv)

    rows = load_classification_rows(args.input)
    unique_pool = build_unique_query_pool(rows)
    candidates = select_stage_a_candidates(
        unique_pool,
        seed=args.seed,
        n_personal=args.n_personal,
        n_environmental=args.n_environmental,
    )
    write_candidates_jsonl(args.output, candidates)

    unique_counts = Counter(
        item.source_classification_label for item in unique_pool
    )
    candidate_counts = Counter(
        item.source_classification_label for item in candidates
    )
    print(f"classification rows: {len(rows)}")
    print(f"unique queries:      {len(unique_pool)}")
    for label in ("Personal", "Environmental", "Mixed"):
        print(f"  unique {label:13}: {unique_counts[label]}")
    print(f"candidate pool:      {len(candidates)}")
    for label in ("Personal", "Environmental", "Mixed"):
        print(f"  candidate {label:9}: {candidate_counts[label]}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
