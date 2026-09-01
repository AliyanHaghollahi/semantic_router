"""Learned H1-H7 planner path for physical end-to-end TierGraph pilot."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tiergraph.graph import ExecutionGraph
from tiergraph.planner.decode import GraphDecoder, PlannerDecodeError
from tiergraph.planner.free_eval import classify_decode_error
from tiergraph.planner.model import PlannerModel
from tiergraph.planner.train import (
    assert_encoder_frozen,
    build_model,
    config_from_checkpoint,
    load_checkpoint,
)
from tiergraph.pilot.physical_harness import (
    ORCHESTRATOR_DEVICE,
    PilotQuerySpec,
    build_deployment_locations,
)

GRAPH_SOURCE_LEARNED = "learned_h1_h7_predicted"
DEFAULT_CHECKPOINT_PATH = Path("artifacts/planner_v2_run1/best.pt")
DEFAULT_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_LENGTH = 128


@dataclass(frozen=True, slots=True)
class PlannerPredictionOutcome:
    planner_latency_ms: float
    graph: ExecutionGraph | None
    predicted_graph_valid: bool
    decode_failure_reason: str | None
    decode_failure_bucket: str | None
    predicted_execution_mode: str | None


def describe_learned_graph_source() -> dict[str, Any]:
    return {
        "graph_source": GRAPH_SOURCE_LEARNED,
        "uses_learned_planner": True,
        "description": (
            "query -> MiniLMFeatureEncoder -> trained PlannerModel H1-H7 -> "
            "predict_structures() -> GraphDecoder -> predicted ExecutionGraph -> "
            "GraphExecutor. Gold example.graph and gold fusion_plan are not used."
        ),
    }


def build_learned_deployment_locations(
    *,
    checkpoint_path: str | Path,
) -> dict[str, str]:
    locations = build_deployment_locations()
    locations["planner_location"] = (
        f"{ORCHESTRATOR_DEVICE} CPU (learned H1-H7, {checkpoint_path})"
    )
    locations["graph_source"] = GRAPH_SOURCE_LEARNED
    return locations


def load_planner_for_physical_pilot(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> PlannerModel:
    """Load planner heads strictly from a training checkpoint on CPU."""
    checkpoint_path = Path(checkpoint_path)
    payload = load_checkpoint(checkpoint_path, map_location=device)
    config = config_from_checkpoint(payload, device=device)
    config_dict = config.to_dict()
    config_dict["encoder_model_name"] = DEFAULT_ENCODER_MODEL
    config_dict["max_length"] = DEFAULT_MAX_LENGTH
    config_dict["device"] = device
    from tiergraph.planner.train import TrainConfig

    config = TrainConfig(**config_dict)
    model = build_model(config)
    _ = model.encode(["planner warmup"])
    assert_encoder_frozen(model)
    missing = model.load_state_dict(payload["model_head_state_dict"], strict=True)
    if missing.missing_keys or missing.unexpected_keys:
        raise RuntimeError(
            "checkpoint state_dict mismatch: "
            f"missing={missing.missing_keys} unexpected={missing.unexpected_keys}"
        )
    model.eval()
    return model


def predict_execution_graph(
    model: PlannerModel,
    *,
    query: str,
    graph_id: str,
    decoder: GraphDecoder | None = None,
) -> PlannerPredictionOutcome:
    """Run learned planner inference and decode to a predicted ExecutionGraph."""
    decoder = decoder or GraphDecoder()
    t0 = time.perf_counter()
    with torch.inference_mode():
        features = model.encode([query])
        predicted_batch = model.predict_structures(features)
        predictions = predicted_batch.items[0]
    planner_latency_ms = (time.perf_counter() - t0) * 1000
    try:
        decoded = decoder.decode(
            predictions,
            query=query,
            graph_id=graph_id,
        )
    except PlannerDecodeError as exc:
        message = str(exc)
        return PlannerPredictionOutcome(
            planner_latency_ms=planner_latency_ms,
            graph=None,
            predicted_graph_valid=False,
            decode_failure_reason=message,
            decode_failure_bucket=classify_decode_error(message),
            predicted_execution_mode=None,
        )
    graph = decoded.graph
    return PlannerPredictionOutcome(
        planner_latency_ms=planner_latency_ms,
        graph=graph,
        predicted_graph_valid=True,
        decode_failure_reason=None,
        decode_failure_bucket=None,
        predicted_execution_mode=graph.execution_mode(),
    )


def build_decode_failure_record(
    *,
    spec: PilotQuerySpec,
    outcome: PlannerPredictionOutcome,
    deployment_locations: dict[str, str],
) -> dict[str, Any]:
    """Trial record when planner decode fails before physical execution."""
    return {
        "query_id": spec.query_id,
        "bucket": spec.bucket,
        "query": spec.query,
        "example_id": spec.example_id,
        "source_split": spec.source_split,
        "provenance": spec.provenance,
        "execution_graph_id": f"pred::{spec.query_id}",
        "execution_mode_expected": spec.execution_mode_expected,
        "is_mixed": spec.bucket.startswith("MIXED_"),
        "deployment_locations": deployment_locations,
        "graph_source": GRAPH_SOURCE_LEARNED,
        "planner_runs": True,
        "planner_latency_ms": outcome.planner_latency_ms,
        "predicted_graph_valid": False,
        "predicted_decode_failure_reason": outcome.decode_failure_reason,
        "predicted_decode_failure_bucket": outcome.decode_failure_bucket,
        "predicted_execution_mode": None,
        "evaluation_labels": {
            "gold_example_id": spec.example_id,
            "gold_execution_mode_expected": spec.execution_mode_expected,
        },
        "success": False,
        "failure_reason": outcome.decode_failure_reason,
        "fog_contacted": False,
        "personal_fog_isolation_ok": True,
        "edge_calls": [],
        "fog_calls": [],
        "latencies": {
            "planner_latency_ms": outcome.planner_latency_ms,
            "end_to_end_latency_ms": outcome.planner_latency_ms,
            "end_to_end_ms": outcome.planner_latency_ms,
        },
    }
