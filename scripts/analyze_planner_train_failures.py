#!/usr/bin/env python3
"""Attribute TRAIN semantic failures to planner components.

Uses the frozen H4.1 checkpoint and the same TRAIN gold graphs and graph
decoding as the gold-vs-learned harness. DEV and TEST are not read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.pilot.gold_learned_harness import GoldExampleView
from tiergraph.pilot.train_error_attribution import (
    H1_QUERY_TYPE,
    attribute_failure,
    graph_summary,
    make_executor,
    op_group,
    oracle_repairs,
    semantic_success,
    summarize_attributions,
)

H4_CHECKPOINT = Path("artifacts/planner_h4_dev_confirmation/final.pt")
DEFAULT_OUTPUT = ROOT / "artifacts" / "planner_train_error_attribution"


def _gold_driver():
    spec = importlib.util.spec_from_file_location(
        "run_gold_vs_learned_execution",
        ROOT / "scripts" / "run_gold_vs_learned_execution.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("gold-vs-learned harness script is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze(
    examples: list[GoldExampleView],
    *,
    checkpoint: Path,
) -> dict[str, Any]:
    """Score TRAIN examples and attribute semantic failures. No parameter update."""
    driver = _gold_driver()
    from tiergraph.pilot.learned_planner_pilot import (
        load_planner_for_physical_pilot,
        predict_execution_graph,
    )

    if not checkpoint.is_file():
        raise RuntimeError(f"H4.1 checkpoint is missing: {checkpoint}")
    model = load_planner_for_physical_pilot(checkpoint, device="cpu")
    executor = make_executor()
    rows = []
    for example in examples:
        outcome = predict_execution_graph(
            model,
            query=example.query,
            graph_id=f"pred::{example.example_id}",
        )
        success = semantic_success(
            example,
            outcome.graph,
            executor=executor,
            predict_ms=outcome.planner_latency_ms,
        )
        rows.append(
            _row(
                example,
                outcome.graph,
                learned_decoded=bool(outcome.predicted_graph_valid and outcome.graph is not None),
                semantic_success=success,
                executor=executor,
            )
        )
    del driver
    summary = summarize_attributions(rows)
    summary["checkpoint"] = str(checkpoint)
    return {"summary": summary, "failures": [row for row in rows if not row["semantic_success"]]}


def _row(
    example: GoldExampleView,
    learned,
    *,
    learned_decoded: bool,
    semantic_success: bool,
    executor,
) -> dict[str, Any]:
    gold_summary = graph_summary(example.graph)
    learned_summary = None if learned is None else graph_summary(learned)
    if semantic_success:
        return {
            "stage_a_id": example.example_id,
            "query": example.query,
            "final_bucket": example.final_bucket,
            "op_group": op_group(example.graph),
            "semantic_success": True,
            "categories": [],
            "primary": None,
            "oracle_applicable": False,
            "oracle": None,
            "gold_graph_summary": gold_summary,
            "learned_graph_summary": learned_summary,
        }
    attributed = attribute_failure(
        example.graph,
        learned,
        learned_decoded=learned_decoded,
    )
    pairs = attributed["pairs"]
    oracle = None
    applicable = bool(attributed["alignment_safe"] and learned is not None and pairs)
    if applicable:
        oracle = oracle_repairs(example.graph, learned, pairs, executor)
    categories = [
        category for category in attributed["categories"] if category != H1_QUERY_TYPE or True
    ]
    return {
        "stage_a_id": example.example_id,
        "query": example.query,
        "final_bucket": example.final_bucket,
        "op_group": op_group(example.graph),
        "semantic_success": False,
        "categories": categories,
        "primary": attributed["primary"],
        "oracle_applicable": applicable,
        "oracle": oracle,
        "gold_graph_summary": gold_summary,
        "learned_graph_summary": learned_summary,
    }


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=H4_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    driver = _gold_driver()
    ids = driver.train_ids(ROOT / driver.STAGE_A_V3_SPLIT_PATH)
    examples = driver.load_train_gold_examples()
    if len(examples) != len(ids):
        raise RuntimeError("TRAIN gold graphs do not match the TRAIN membership")
    payload = analyze(examples, checkpoint=checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "per_failure.jsonl").open("w", encoding="utf-8") as handle:
        for row in payload["failures"]:
            handle.write(json.dumps(_public_failure(row), sort_keys=True) + "\n")


def _public_failure(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage_a_id": row["stage_a_id"],
        "query": row["query"],
        "final_bucket": row["final_bucket"],
        "op_group": row["op_group"],
        "gold_graph_summary": row["gold_graph_summary"],
        "learned_graph_summary": row["learned_graph_summary"],
        "mismatch_categories": row["categories"],
        "primary": row["primary"],
        "oracle_applicable": row["oracle_applicable"],
        "oracle": row["oracle"],
    }


def main() -> None:
    run()


if __name__ == "__main__":
    main()
