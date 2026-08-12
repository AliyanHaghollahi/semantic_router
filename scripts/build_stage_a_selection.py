#!/usr/bin/env python3
"""Build the final Stage-A selection (120) and spares manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.authored_implicit import (
    DEFAULT_AUTHORED_CANDIDATES_PATH,
    DEFAULT_AUTHORED_REVIEWS_PATH,
)
from tiergraph.planner.authored_sequential import (
    DEFAULT_AUTHORED_SEQUENTIAL_CANDIDATES_PATH,
    DEFAULT_AUTHORED_SEQUENTIAL_REVIEWS_PATH,
)
from tiergraph.planner.implicit_mining import (
    DEFAULT_OUTPUT_PATH as DEFAULT_IMPLICIT_CANDIDATES_PATH,
)
from tiergraph.planner.stage_a_selection import (
    DEFAULT_CANDIDATES_PATH,
    DEFAULT_MIXED_REVIEWS_PATH,
    DEFAULT_SELECTION_PATH,
    DEFAULT_SPARES_PATH,
    build_and_write_stage_a_selection,
    format_selection_summary,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / DEFAULT_CANDIDATES_PATH,
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=ROOT / DEFAULT_MIXED_REVIEWS_PATH,
    )
    parser.add_argument(
        "--mined",
        type=Path,
        default=ROOT / DEFAULT_IMPLICIT_CANDIDATES_PATH,
    )
    parser.add_argument(
        "--authored-implicit-candidates",
        type=Path,
        default=ROOT / DEFAULT_AUTHORED_CANDIDATES_PATH,
    )
    parser.add_argument(
        "--authored-implicit-reviews",
        type=Path,
        default=ROOT / DEFAULT_AUTHORED_REVIEWS_PATH,
    )
    parser.add_argument(
        "--authored-sequential-candidates",
        type=Path,
        default=ROOT / DEFAULT_AUTHORED_SEQUENTIAL_CANDIDATES_PATH,
    )
    parser.add_argument(
        "--authored-sequential-reviews",
        type=Path,
        default=ROOT / DEFAULT_AUTHORED_SEQUENTIAL_REVIEWS_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / DEFAULT_SELECTION_PATH,
    )
    parser.add_argument(
        "--spares",
        type=Path,
        default=ROOT / DEFAULT_SPARES_PATH,
    )
    args = parser.parse_args(argv)

    summary = build_and_write_stage_a_selection(
        candidates_path=args.candidates,
        reviews_path=args.reviews,
        mined_path=args.mined,
        authored_implicit_candidates_path=args.authored_implicit_candidates,
        authored_implicit_reviews_path=args.authored_implicit_reviews,
        authored_sequential_candidates_path=args.authored_sequential_candidates,
        authored_sequential_reviews_path=args.authored_sequential_reviews,
        selection_path=args.output,
        spares_path=args.spares,
    )
    print(format_selection_summary(summary))
    print(f"wrote: {args.output}")
    print(f"wrote: {args.spares}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
