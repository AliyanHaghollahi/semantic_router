"""Deterministic graph-aware fusion and template response generation.

Fusion reads structured node results and the graph's answer-node
dependencies. Rendering turns that record into one sentence. Neither step
calls a model. ``GraphExecutionResult.final_response`` stays the raw sink or
concatenate string already produced by the executor.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from tiergraph.enums import NodeSemanticType, OperatorType, SlotType, Tier
from tiergraph.graph import ExecutionGraph, SemanticNode
from tiergraph.models import TierResult

FusionMode = Literal["none", "sequential", "parallel", "hybrid"]
Comparison = Literal["match", "mismatch"]

_PERSONAL_SLOTS = frozenset(
    {
        SlotType.RESOLVED_REFERENCE,
        SlotType.PERSONAL_FACT,
        SlotType.PERSONAL_RECORD,
    }
)


@dataclass(frozen=True)
class BranchResult:
    """One answer node's principal structured output."""

    node_id: str
    operator: OperatorType
    tier: Tier
    slot_name: str
    slot_type: SlotType
    text: str


@dataclass(frozen=True)
class StructuredFusion:
    """Graph-aware combination of answer results. ``none`` means no fusion."""

    mode: FusionMode
    branches: tuple[BranchResult, ...]
    comparison: Comparison | None = None


@dataclass(frozen=True)
class UserFacingResponse:
    """Template sentence for one finished execution."""

    text: str
    fusion: StructuredFusion


def fuse_structured_results(
    graph: ExecutionGraph,
    results: Mapping[str, TierResult],
) -> StructuredFusion:
    """Classify the answer graph and attach a comparison only for parallel mixes."""
    answers = _answer_nodes(graph)
    branches = tuple(_branch(node, results) for node in answers)
    if len(answers) <= 1:
        return StructuredFusion(mode="none", branches=branches, comparison=None)

    ordered, mode = _schedule_answers(graph, answers)
    ordered_branches = tuple(_branch(node, results) for node in ordered)
    comparison = None
    if mode == "parallel":
        comparison = _compare_tiers(ordered_branches)
    return StructuredFusion(
        mode=mode,
        branches=ordered_branches,
        comparison=comparison,
    )


def render_user_response(fusion: StructuredFusion) -> str:
    """Render one deterministic sentence from a fusion record."""
    if not fusion.branches:
        return ""
    if fusion.mode == "none":
        return _single_sentence(fusion.branches[0])
    if fusion.mode == "parallel":
        sentences = [_single_sentence(branch) for branch in fusion.branches]
        if fusion.comparison == "match":
            sentences.append("They match.")
        elif fusion.comparison == "mismatch":
            sentences.append("They do not match.")
        return " ".join(sentences)
    sentences = []
    for index, branch in enumerate(fusion.branches):
        has_predecessor = index > 0 and fusion.mode in {"sequential", "hybrid"}
        if branch.operator is OperatorType.NAVIGATE_TO and has_predecessor:
            sentences.append(_navigation_clause(branch.text))
        else:
            sentences.append(_single_sentence(branch))
    return " ".join(sentence for sentence in sentences if sentence)


def compose_user_response(
    graph: ExecutionGraph,
    results: Mapping[str, TierResult],
) -> UserFacingResponse:
    """Fuse structured results, then render the template. No model call."""
    fusion = fuse_structured_results(graph, results)
    return UserFacingResponse(text=render_user_response(fusion), fusion=fusion)


def _answer_nodes(graph: ExecutionGraph) -> tuple[SemanticNode, ...]:
    return tuple(
        node
        for node in graph.nodes
        if node.semantic_type is not NodeSemanticType.CONTROL
        and node.operator is not OperatorType.FUSE
    )


def _branch(node: SemanticNode, results: Mapping[str, TierResult]) -> BranchResult:
    slot_name = next(iter(node.produced_outputs))
    result = results.get(node.node_id)
    if result is None or slot_name not in result.outputs:
        raise ValueError(
            f"missing structured output {node.node_id}.{slot_name}"
        )
    return BranchResult(
        node_id=node.node_id,
        operator=node.operator,
        tier=node.tier,
        slot_name=slot_name,
        slot_type=node.produced_outputs[slot_name],
        text=_text(result.outputs[slot_name]),
    )


def _schedule_answers(
    graph: ExecutionGraph,
    answers: tuple[SemanticNode, ...],
) -> tuple[tuple[SemanticNode, ...], FusionMode]:
    by_id = {node.node_id: node for node in answers}
    answer_ids = set(by_id)
    successors: dict[str, list[str]] = {node_id: [] for node_id in answer_ids}
    indegree = {node_id: 0 for node_id in answer_ids}
    for edge in graph.edges:
        if edge.source_node_id in answer_ids and edge.target_node_id in answer_ids:
            successors[edge.source_node_id].append(edge.target_node_id)
            indegree[edge.target_node_id] += 1
    if all(degree == 0 for degree in indegree.values()):
        return answers, "parallel"

    order_index = {node.node_id: index for index, node in enumerate(graph.nodes)}
    ready = deque(
        sorted(
            (node_id for node_id, degree in indegree.items() if degree == 0),
            key=order_index.__getitem__,
        )
    )
    ordered: list[SemanticNode] = []
    branched = False
    while ready:
        if len(ready) > 1:
            branched = True
        node_id = ready.popleft()
        ordered.append(by_id[node_id])
        for successor_id in successors[node_id]:
            indegree[successor_id] -= 1
            if indegree[successor_id] == 0:
                ready.append(successor_id)
        ready = deque(sorted(ready, key=order_index.__getitem__))
    mode: FusionMode = "hybrid" if branched else "sequential"
    return tuple(ordered), mode


def _compare_tiers(branches: tuple[BranchResult, ...]) -> Comparison | None:
    personal = next((branch for branch in branches if branch.tier is Tier.EDGE), None)
    environmental = next(
        (branch for branch in branches if branch.tier is Tier.FOG),
        None,
    )
    if personal is None or environmental is None:
        return None
    same = _normalized(personal.text) == _normalized(environmental.text)
    return "match" if same else "mismatch"


def _single_sentence(branch: BranchResult) -> str:
    value = _plain(branch.text)
    subject = _subject(branch.slot_name)
    if branch.operator is OperatorType.NAVIGATE_TO:
        return _period(f"The directions are: {value}")
    if branch.tier is Tier.EDGE or branch.slot_type in _PERSONAL_SLOTS:
        return _period(f"Your {subject} is {value}")
    return _period(f"The {subject} is {value}")


def _navigation_clause(text: str) -> str:
    return _period(f"To get there, {_plain(text)}")


def _subject(slot_name: str) -> str:
    name = slot_name.strip().lower().replace("-", " ").replace("_", " ")
    for suffix in (" identifier", " reference"):
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    name = " ".join(name.split())
    return name or "result"


def _plain(text: str) -> str:
    value = " ".join(text.split())
    while value and value[-1] in ".!?":
        value = value[:-1].rstrip()
    return value


def _period(text: str) -> str:
    value = text.strip()
    if not value:
        return ""
    if value[-1] in ".!?":
        return value
    return value + "."


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _text(value: JsonValue) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=True)
