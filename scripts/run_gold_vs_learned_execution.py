#!/usr/bin/env python3
"""Compare gold ExecutionGraphs with learned graphs on one stub backend.

Both conditions use GraphExecutor and the frozen user_response layer.
The learned planner receives only the query. fusion_plan is never passed.

Dry-run prints the pinned checkpoint, the TRAIN count, and the metric names.
It does not load annotations, a checkpoint, or TEST.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.pilot.gold_learned_harness import (
    METRIC_NAMES,
    GoldExampleView,
    compare_pair,
    make_stub_executor,
    summarize,
)
from tiergraph.planner.stage_a_v3_spec import (
    STAGE_A_V3_SPLIT_PATH,
    STAGE_A_V3_STEP_A_PATH,
    STAGE_A_V3_STEP_B_PATH,
    STAGE_A_V3_TRAIN_SIZE,
)

H4_CHECKPOINT = Path("artifacts/planner_h4_dev_confirmation/final.pt")
DEFAULT_OUTPUT = ROOT / "artifacts" / "gold_vs_learned_execution"


def train_ids(split_path: Path) -> list[str]:
    """TRAIN membership ids. DEV and TEST ids are not retained."""
    ids: list[str] = []
    with Path(split_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            name = str(row["split"])
            if name == "train":
                ids.append(str(row["stage_a_id"]))
            elif name not in {"dev", "test"}:
                raise RuntimeError(f"unknown split label {name!r}")
    if len(ids) != STAGE_A_V3_TRAIN_SIZE or len(set(ids)) != len(ids):
        raise RuntimeError(f"expected {STAGE_A_V3_TRAIN_SIZE} unique TRAIN ids, got {len(set(ids))}")
    return ids


def load_train_gold_examples(root: Path = ROOT) -> list[GoldExampleView]:
    """Materialize TRAIN gold graphs only. TEST annotation rows are not parsed."""
    from tiergraph.planner.annotation_step_a import StageAStepAAnnotation
    from tiergraph.planner.annotation_step_b import StageAStepBAnnotation
    from tiergraph.planner.stage_a_to_corpus import step_ab_to_planner_example
    from tiergraph.planner.stage_a_v3_h4_build import load_annotation_rows_for_ids

    ids = train_ids(root / STAGE_A_V3_SPLIT_PATH)
    allowed = set(ids)
    step_a = load_annotation_rows_for_ids(
        root / STAGE_A_V3_STEP_A_PATH,
        allowed,
        model_cls=StageAStepAAnnotation,
    )
    step_b = load_annotation_rows_for_ids(
        root / STAGE_A_V3_STEP_B_PATH,
        allowed,
        model_cls=StageAStepBAnnotation,
    )
    if set(step_a) != allowed or set(step_b) != allowed:
        raise RuntimeError("annotation load did not return exactly the TRAIN ids")
    examples: list[GoldExampleView] = []
    for sid in ids:
        example = step_ab_to_planner_example(
            step_a[sid],
            step_b[sid],
            use_semantic_h1=True,
        )
        if example.example_id != sid or example.graph.original_query != example.query:
            raise RuntimeError(f"gold example {sid} is inconsistent")
        examples.append(
            GoldExampleView(
                example_id=example.example_id,
                query=example.query,
                graph=example.graph,
                final_bucket=str(example.metadata["final_bucket"]),
            )
        )
    return examples


def run_comparison(
    examples: list[GoldExampleView],
    *,
    checkpoint: Path,
) -> dict[str, Any]:
    """Score every gold example against a learned graph from its query alone."""
    from tiergraph.pilot.learned_planner_pilot import (
        load_planner_for_physical_pilot,
        predict_execution_graph,
    )

    if not checkpoint.is_file():
        raise RuntimeError(f"H4.1 checkpoint is missing: {checkpoint}")
    model = load_planner_for_physical_pilot(checkpoint, device="cpu")
    executor = make_stub_executor()
    records = []
    for example in examples:
        outcome = predict_execution_graph(
            model,
            query=example.query,
            graph_id=f"pred::{example.example_id}",
        )
        records.append(
            compare_pair(
                example,
                outcome.graph,
                predict_ms=outcome.planner_latency_ms,
                executor=executor,
            )
        )
    summary = summarize(records)
    summary["checkpoint"] = str(checkpoint)
    summary["n_train"] = len(examples)
    summary["test_used"] = False
    summary["dev_used"] = False
    return {"summary": summary, "records": records}


def print_dry_run(checkpoint: Path, n_train: int) -> None:
    print(f"CHECKPOINT {checkpoint.as_posix()}")
    print(f"CHECKPOINT_EXISTS {str(checkpoint.is_file()).lower()}")
    print(f"N_TRAIN {n_train}")
    print("METRICS " + " ".join(METRIC_NAMES))
    print("SLICES final_bucket op_group")
    print("FUSION_PLAN none")
    print("BACKEND operator_stub")
    print("DEV_USED false")
    print("TEST_USED false")


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=H4_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the frozen protocol without loading a model or executing graphs",
    )
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    ids = train_ids(ROOT / STAGE_A_V3_SPLIT_PATH)
    if args.dry_run:
        print_dry_run(Path(args.checkpoint), len(ids))
        return
    examples = load_train_gold_examples()
    if len(examples) != len(ids):
        raise RuntimeError("TRAIN gold graphs do not match the TRAIN membership")
    payload = run_comparison(examples, checkpoint=checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "per_example.jsonl").open("w", encoding="utf-8") as handle:
        for record in payload["records"]:
            handle.write(json.dumps(record.__dict__, sort_keys=True) + "\n")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
