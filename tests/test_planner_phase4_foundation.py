"""Phase-4 step-1 tests: alignment, operator I/O, naming, tasks, targets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiergraph.enums import OperatorType, QueryType, SlotType
from tiergraph.planner.align import (
    BIO_B,
    BIO_I,
    BIO_IGNORE,
    BIO_O,
    TokenCharSpan,
    TruncationKind,
    align_char_span,
    align_char_spans,
    encode_bio_labels,
)
from tiergraph.planner.annotations import ImplicitResolution, PlannerExample
from tiergraph.planner.naming import (
    DEFAULT_BASE_BY_OPERATOR_V1,
    SlotNamingError,
    default_base_for_operator,
    fuse_input_slot_name,
    fuse_output_slot_name,
    normalize_base_name,
    principal_slot_name,
)
from tiergraph.planner.operator_io import (
    OPERATOR_IO_CONTRACT_V1,
    is_h7_pair_eligible,
    principal_output_type,
)
from tiergraph.planner.annotations import _PRINCIPAL_OUTPUT_TYPES
from tiergraph.planner.targets import build_planner_targets
from tiergraph.planner.tasks import render_answer_task, render_fuse_task


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "planner" / "where_is_my_gate.json"
)


def _gate_example() -> PlannerExample:
    return PlannerExample.model_validate(
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    )


def _tokens_for_query(query: str) -> tuple[TokenCharSpan, ...]:
    """Deterministic fake tokenizer: CLS, per-character content, SEP, PAD.

    Per-character content tokens make span coverage exact and avoid accidental
    partial-truncation from whitespace gaps in the test double.
    """
    tokens: list[TokenCharSpan] = [
        TokenCharSpan(None, None, is_special=True, is_padding=False),
    ]
    for index, _char in enumerate(query):
        # Pair adjacent characters into crude subwords when possible so B/I
        # across subwords is exercised for spans longer than one character.
        if index % 2 == 1:
            continue
        end = min(index + 2, len(query))
        tokens.append(
            TokenCharSpan(index, end, is_special=False, is_padding=False)
        )
    tokens.append(TokenCharSpan(None, None, is_special=True, is_padding=False))
    tokens.append(TokenCharSpan(None, None, is_special=False, is_padding=True))
    return tuple(tokens)


# ---------------------------------------------------------------------------
# ALIGNMENT
# ---------------------------------------------------------------------------


def test_content_tokens_align_to_char_spans():
    query = "Where is my gate?"
    tokens = _tokens_for_query(query)
    gate_start = query.index("gate")
    aligned = align_char_span(gate_start, gate_start + 4, tokens)
    assert aligned.representable
    assert aligned.truncation_kind is TruncationKind.NONE
    assert aligned.token_indices
    covered = set()
    for index in aligned.token_indices:
        token = tokens[index]
        assert token.is_content
        covered.update(range(token.char_start, token.char_end))
    assert set(range(gate_start, gate_start + 4)).issubset(covered)


def test_special_tokens_ignored_in_bio():
    query = "Where is my gate?"
    tokens = _tokens_for_query(query)
    spans, _ = align_char_spans([(0, len(query))], tokens)
    bio = encode_bio_labels(spans, tokens)
    for token, label in zip(tokens, bio.labels, strict=True):
        if token.is_special:
            assert label == BIO_IGNORE


def test_padding_tokens_ignored_in_bio():
    query = "Where is my gate?"
    tokens = _tokens_for_query(query)
    spans, _ = align_char_spans([(0, len(query))], tokens)
    bio = encode_bio_labels(spans, tokens)
    for token, label in zip(tokens, bio.labels, strict=True):
        if token.is_padding:
            assert label == BIO_IGNORE


def test_bio_assignment_deterministic_across_subwords():
    query = "Where is my gate?"
    tokens = _tokens_for_query(query)
    spans, _ = align_char_spans([(0, len(query))], tokens)
    bio = encode_bio_labels(spans, tokens)
    content_labels = [
        label
        for token, label in zip(tokens, bio.labels, strict=True)
        if token.is_content
    ]
    assert content_labels[0] == BIO_B
    assert all(label == BIO_I for label in content_labels[1:])
    # outside content remains O only for uncovered content — here all covered
    assert BIO_O not in content_labels


def test_fully_truncated_span_masked_and_reported():
    tokens = (
        TokenCharSpan(None, None, True, False),
        TokenCharSpan(0, 5, False, False),
        TokenCharSpan(None, None, True, False),
    )
    aligned = align_char_span(10, 14, tokens)
    assert not aligned.representable
    assert aligned.truncation_kind is TruncationKind.FULL
    assert aligned.token_indices == ()
    _, stats = align_char_spans([(10, 14)], tokens)
    assert stats.n_fully_truncated == 1
    assert stats.n_representable == 0


def test_partially_truncated_span_masked_not_clipped():
    tokens = (
        TokenCharSpan(None, None, True, False),
        TokenCharSpan(0, 5, False, False),
        TokenCharSpan(None, None, True, False),
    )
    # Span overlaps retained chars [0,5) but also needs [5,8).
    aligned = align_char_span(3, 8, tokens)
    assert not aligned.representable
    assert aligned.truncation_kind is TruncationKind.PARTIAL
    assert aligned.token_indices == ()
    bio = encode_bio_labels((aligned,), tokens)
    content_labels = [
        label
        for token, label in zip(tokens, bio.labels, strict=True)
        if token.is_content
    ]
    assert content_labels == [BIO_O]
    assert bio.stats.n_partially_truncated == 1


def _hf_like_word_tokens(query: str = "Where is my gate?") -> tuple[TokenCharSpan, ...]:
    """BERT-like offsets: word pieces only; whitespace gaps uncovered."""
    assert query == "Where is my gate?"
    return (
        TokenCharSpan(None, None, True, False),  # CLS
        TokenCharSpan(0, 5, False, False),  # Where
        TokenCharSpan(6, 8, False, False),  # is
        TokenCharSpan(9, 11, False, False),  # my
        TokenCharSpan(12, 16, False, False),  # gate
        TokenCharSpan(16, 17, False, False),  # ?
        TokenCharSpan(None, None, True, False),  # SEP
        TokenCharSpan(None, None, False, True),  # PAD
    )


def test_whitespace_gaps_do_not_make_multiword_span_unrepresentable():
    query = "Where is my gate?"
    tokens = _hf_like_word_tokens(query)
    # Spaces at offsets 5, 8, 11 are uncovered — must not imply truncation.
    aligned = align_char_span(0, len(query), tokens)
    assert aligned.representable
    assert aligned.truncation_kind is TruncationKind.NONE
    assert len(aligned.token_indices) == 5


def test_genuine_token_truncation_makes_span_unrepresentable():
    query = "Where is my gate?"
    # Retained frontier ends after "my"; "gate?" dropped by truncation.
    tokens = (
        TokenCharSpan(None, None, True, False),
        TokenCharSpan(0, 5, False, False),
        TokenCharSpan(6, 8, False, False),
        TokenCharSpan(9, 11, False, False),
        TokenCharSpan(None, None, True, False),
    )
    aligned = align_char_span(0, len(query), tokens)
    assert not aligned.representable
    assert aligned.truncation_kind is TruncationKind.PARTIAL
    assert aligned.token_indices == ()
    gate_start = query.index("gate")
    gate = align_char_span(gate_start, gate_start + 4, tokens)
    assert not gate.representable
    assert gate.truncation_kind is TruncationKind.FULL


# ---------------------------------------------------------------------------
# OPERATOR I/O
# ---------------------------------------------------------------------------


def test_principal_output_reuses_annotation_contract():
    for operator, expected in _PRINCIPAL_OUTPUT_TYPES.items():
        assert principal_output_type(operator) is expected
        assert OPERATOR_IO_CONTRACT_V1[operator].principal_output is expected


def test_legal_h7_pair_accepted():
    assert is_h7_pair_eligible(
        OperatorType.IDENTIFY_ENVIRONMENTAL,
        OperatorType.LOCATE_ENVIRONMENTAL,
    )
    assert is_h7_pair_eligible(
        OperatorType.LOCATE_ENVIRONMENTAL,
        OperatorType.NAVIGATE_TO,
    )


def test_structurally_impossible_h7_pair_rejected():
    assert not is_h7_pair_eligible(
        OperatorType.RETRIEVE_PERSONAL,
        OperatorType.LOCATE_ENVIRONMENTAL,
    )
    assert not is_h7_pair_eligible(
        OperatorType.NAVIGATE_TO,
        OperatorType.RESOLVE_PERSONAL,
    )
    assert not is_h7_pair_eligible(
        OperatorType.FUSE,
        OperatorType.LOCATE_ENVIRONMENTAL,
    )


def test_operator_io_has_no_slot_selection_api():
    spec = OPERATOR_IO_CONTRACT_V1[OperatorType.LOCATE_ENVIRONMENTAL]
    assert spec.allowed_learned_input_types == frozenset(
        {SlotType.RESOLVED_REFERENCE, SlotType.ENVIRONMENTAL_FACT}
    )
    # Contract exposes types only; naming is a separate deterministic util.
    assert not hasattr(spec, "choose_slot")
    assert not hasattr(spec, "select_input")


# ---------------------------------------------------------------------------
# OVERLAP / TARGETS (gate fixture)
# ---------------------------------------------------------------------------


def test_operation_anchor_overlap_accepted_in_targets():
    example = _gate_example()
    op = example.planner_labels.operation_spans[0]
    anchor = example.planner_labels.slot_anchors[0]
    assert op.start <= anchor.start < anchor.end <= op.end
    tokens = _tokens_for_query(example.query)
    targets = build_planner_targets(example, tokens)
    assert targets.operations[0].representable
    assert targets.anchors[0].representable


def test_op_op_overlap_still_rejected_by_annotation_contract():
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    data["planner_labels"]["operation_spans"].append(
        {
            "node_id": "q2",
            "semantic_type": "environmental",
            "start": 5,
            "end": 17,
            "operator": "LOCATE_ENVIRONMENTAL",
        }
    )
    # Duplicate node id / overlap — annotation validation must fail.
    with pytest.raises(Exception):
        PlannerExample.model_validate(data)


def test_gate_fixture_operation_and_anchor_targets():
    example = _gate_example()
    tokens = _tokens_for_query(example.query)
    targets = build_planner_targets(example, tokens)
    assert targets.query_type is QueryType.MIXED
    assert len(targets.operations) == 1
    assert targets.operations[0].operator is OperatorType.LOCATE_ENVIRONMENTAL
    assert targets.operator_labels == (OperatorType.LOCATE_ENVIRONMENTAL,)
    assert len(targets.anchors) == 1
    assert targets.anchors[0].text == "gate"
    assert targets.implicit_labels == (
        ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL,
    )
    assert targets.ownership_owner_indices == (0,)


def test_implicit_owner_excluded_from_h7_targets():
    example = _gate_example()
    tokens = _tokens_for_query(example.query)
    targets = build_planner_targets(example, tokens)
    # Only one explicit op => no H7 pairs at all.
    assert targets.h7_pairs == ()
    # Graph mandatory edge is implicit->owner, not an H7 label.
    assert len(example.graph.edges) == 1
    assert example.graph.edges[0].source_node_id == "q1"


def test_impossible_h7_pairs_masked_in_targets():
    # Build a two-op environmental example with no learned edge.
    query = "Read this sign then describe this room"
    split = query.index(" then ")
    right = split + len(" then ")
    sign_start = query.index("sign")
    room_start = query.index("room")
    data = {
        "example_id": "env-two-001",
        "query": query,
        "graph": {
            "schema_version": "1.0",
            "graph_id": "env-two-001",
            "original_query": query,
            "query_type": "Environmental",
            "nodes": [
                {
                    "node_id": "q1",
                    "semantic_type": "environmental",
                    "operator": "IDENTIFY_ENVIRONMENTAL",
                    "tier": "fog",
                    "task": "Identify the sign",
                    "required_inputs": {},
                    "produced_outputs": {"sign_fact": "ENVIRONMENTAL_FACT"},
                    "status": "pending",
                    "metadata": {},
                },
                {
                    "node_id": "q2",
                    "semantic_type": "environmental",
                    "operator": "DESCRIBE_ENVIRONMENT",
                    "tier": "fog",
                    "task": "Describe the room",
                    "required_inputs": {},
                    "produced_outputs": {"room_scene": "SCENE_DESCRIPTION"},
                    "status": "pending",
                    "metadata": {},
                },
                {
                    "node_id": "fuse",
                    "semantic_type": "control",
                    "operator": "FUSE",
                    "tier": "edge",
                    "task": "Fuse the terminal answers",
                    "required_inputs": {
                        "q1__sign_fact": "ENVIRONMENTAL_FACT",
                        "q2__room_scene": "SCENE_DESCRIPTION",
                    },
                    "produced_outputs": {"response": "FINAL_RESPONSE"},
                    "status": "pending",
                    "metadata": {},
                },
            ],
            "edges": [
                {
                    "source_node_id": "q1",
                    "source_slot": "sign_fact",
                    "target_node_id": "fuse",
                    "target_slot": "q1__sign_fact",
                    "transfer_policy": "direct",
                },
                {
                    "source_node_id": "q2",
                    "source_slot": "room_scene",
                    "target_node_id": "fuse",
                    "target_slot": "q2__room_scene",
                    "transfer_policy": "direct",
                },
            ],
            "metadata": {},
        },
        "fusion_plan": {
            "schema_version": "1.0",
            "plan_id": "plan-env-two",
            "graph_id": "env-two-001",
            "fusion_node_id": "fuse",
            "strategy": "validated_slm",
            "required_slots": {
                "q1__sign_fact": "ENVIRONMENTAL_FACT",
                "q2__room_scene": "SCENE_DESCRIPTION",
            },
            "ordered_slots": ["q1__sign_fact", "q2__room_scene"],
            "max_sentences": 2,
            "spoken_style": True,
            "instructions": "Fuse the typed answers into a concise response.",
            "metadata": {},
        },
        "planner_labels": {
            "query_type": "Environmental",
            "operation_spans": [
                {
                    "node_id": "q1",
                    "semantic_type": "environmental",
                    "start": 0,
                    "end": split,
                    "operator": "IDENTIFY_ENVIRONMENTAL",
                },
                {
                    "node_id": "q2",
                    "semantic_type": "environmental",
                    "start": right,
                    "end": len(query),
                    "operator": "DESCRIBE_ENVIRONMENT",
                },
            ],
            "slot_anchors": [
                {
                    "anchor_id": "a1",
                    "start": sign_start,
                    "end": sign_start + 4,
                    "text": "sign",
                    "normalized_name": "sign",
                    "owner_node_id": "q1",
                    "implicit_resolution": "NONE",
                    "implicit_node_id": None,
                },
                {
                    "anchor_id": "a2",
                    "start": room_start,
                    "end": room_start + 4,
                    "text": "room",
                    "normalized_name": "room",
                    "owner_node_id": "q2",
                    "implicit_resolution": "NONE",
                    "implicit_node_id": None,
                },
            ],
        },
        "metadata": {},
    }
    example = PlannerExample.model_validate(data)
    tokens = _tokens_for_query(query)
    targets = build_planner_targets(example, tokens)
    # DESCRIBE -> IDENTIFY is ineligible (SCENE_DESCRIPTION not accepted).
    describe_to_identify = [
        pair
        for pair in targets.h7_pairs
        if pair.source_node_id == "q2" and pair.target_node_id == "q1"
    ]
    assert len(describe_to_identify) == 1
    assert describe_to_identify[0].masked
    assert describe_to_identify[0].mask_reason == "structurally_ineligible"
    # IDENTIFY -> DESCRIBE is eligible (ENVIRONMENTAL_FACT allowed) and gold-
    # negative because no explicit learned edge exists.
    identify_to_describe = [
        pair
        for pair in targets.h7_pairs
        if pair.source_node_id == "q1" and pair.target_node_id == "q2"
    ]
    assert identify_to_describe[0].eligible
    assert not identify_to_describe[0].masked
    assert identify_to_describe[0].label == 0.0


def test_downstream_targets_use_gold_spans():
    example = _gate_example()
    tokens = _tokens_for_query(example.query)
    targets = build_planner_targets(example, tokens)
    gold_op = example.planner_labels.operation_spans[0]
    assert targets.operations[0].start == gold_op.start
    assert targets.operations[0].end == gold_op.end
    gold_anchor = example.planner_labels.slot_anchors[0]
    assert targets.anchors[0].start == gold_anchor.start
    assert targets.anchors[0].owner_node_id == gold_anchor.owner_node_id
    assert targets.anchors[0].owner_index_full == 0
    assert targets.anchors[0].owner_index_supervised == 0


def test_truncation_masks_downstream_supervision():
    example = _gate_example()
    # Retain only the first few characters so the gate anchor is truncated.
    tokens = (
        TokenCharSpan(None, None, True, False),
        TokenCharSpan(0, 5, False, False),
        TokenCharSpan(None, None, True, False),
    )
    targets = build_planner_targets(example, tokens)
    assert targets.n_masked_operations == 1
    assert targets.n_masked_anchors == 1
    assert targets.operator_labels == ()
    assert targets.implicit_labels == ()
    assert targets.ownership_owner_indices == ()
    assert targets.anchors[0].owner_index_full == 0
    assert targets.anchors[0].owner_index_supervised is None
    assert targets.alignment_stats.n_partially_truncated + (
        targets.alignment_stats.n_fully_truncated
    ) >= 2


def test_supervised_index_space_survives_truncated_middle_operation():
    """Ownership/H7 indices must use representable_ops, not full gold indices."""
    query = "Read this sign then locate this gate then navigate there"
    first_end = query.index(" then locate")
    mid_start = first_end + len(" then ")
    mid_end = query.index(" then navigate")
    last_start = mid_end + len(" then ")
    sign_start = query.index("sign")
    gate_start = query.index("gate")
    there_start = query.index("there")
    data = {
        "example_id": "trunc-mid-001",
        "query": query,
        "graph": {
            "schema_version": "1.0",
            "graph_id": "trunc-mid-001",
            "original_query": query,
            "query_type": "Environmental",
            "nodes": [
                {
                    "node_id": "q1",
                    "semantic_type": "environmental",
                    "operator": "IDENTIFY_ENVIRONMENTAL",
                    "tier": "fog",
                    "task": "Identify the sign",
                    "required_inputs": {},
                    "produced_outputs": {"sign_fact": "ENVIRONMENTAL_FACT"},
                    "status": "pending",
                    "metadata": {},
                },
                {
                    "node_id": "q2",
                    "semantic_type": "environmental",
                    "operator": "LOCATE_ENVIRONMENTAL",
                    "tier": "fog",
                    "task": "Locate the resolved gate",
                    "required_inputs": {"sign_fact": "ENVIRONMENTAL_FACT"},
                    "produced_outputs": {"gate_location": "LOCATION"},
                    "status": "pending",
                    "metadata": {},
                },
                {
                    "node_id": "q3",
                    "semantic_type": "environmental",
                    "operator": "NAVIGATE_TO",
                    "tier": "fog",
                    "task": "Navigate to the destination",
                    "required_inputs": {"gate_location": "LOCATION"},
                    "produced_outputs": {
                        "destination_navigation": "NAVIGATION_INSTRUCTION"
                    },
                    "status": "pending",
                    "metadata": {},
                },
            ],
            "edges": [
                {
                    "source_node_id": "q1",
                    "source_slot": "sign_fact",
                    "target_node_id": "q2",
                    "target_slot": "sign_fact",
                    "transfer_policy": "direct",
                },
                {
                    "source_node_id": "q2",
                    "source_slot": "gate_location",
                    "target_node_id": "q3",
                    "target_slot": "gate_location",
                    "transfer_policy": "direct",
                },
            ],
            "metadata": {},
        },
        "fusion_plan": None,
        "planner_labels": {
            "query_type": "Environmental",
            "operation_spans": [
                {
                    "node_id": "q1",
                    "semantic_type": "environmental",
                    "start": 0,
                    "end": first_end,
                    "operator": "IDENTIFY_ENVIRONMENTAL",
                },
                {
                    "node_id": "q2",
                    "semantic_type": "environmental",
                    "start": mid_start,
                    "end": mid_end,
                    "operator": "LOCATE_ENVIRONMENTAL",
                },
                {
                    "node_id": "q3",
                    "semantic_type": "environmental",
                    "start": last_start,
                    "end": len(query),
                    "operator": "NAVIGATE_TO",
                },
            ],
            "slot_anchors": [
                {
                    "anchor_id": "a1",
                    "start": sign_start,
                    "end": sign_start + 4,
                    "text": "sign",
                    "normalized_name": "sign",
                    "owner_node_id": "q1",
                    "implicit_resolution": "NONE",
                    "implicit_node_id": None,
                },
                {
                    "anchor_id": "a2",
                    "start": gate_start,
                    "end": gate_start + 4,
                    "text": "gate",
                    "normalized_name": "gate",
                    "owner_node_id": "q2",
                    "implicit_resolution": "NONE",
                    "implicit_node_id": None,
                },
                {
                    "anchor_id": "a3",
                    "start": there_start,
                    "end": there_start + 5,
                    "text": "there",
                    "normalized_name": "destination",
                    "owner_node_id": "q3",
                    "implicit_resolution": "NONE",
                    "implicit_node_id": None,
                },
            ],
        },
        "metadata": {},
    }
    example = PlannerExample.model_validate(data)
    # Retain first and last clauses; middle locate span dropped from offsets.
    tokens = (
        TokenCharSpan(None, None, True, False),
        TokenCharSpan(0, 4, False, False),
        TokenCharSpan(sign_start, sign_start + 4, False, False),
        TokenCharSpan(last_start, last_start + 8, False, False),
        TokenCharSpan(there_start, there_start + 5, False, False),
        TokenCharSpan(None, None, True, False),
    )
    targets = build_planner_targets(example, tokens)
    assert targets.operations[1].representable is False
    assert targets.operations[1].index_full == 1
    assert targets.operations[1].index_supervised is None
    assert [op.node_id for op in targets.supervised_operations] == ["q1", "q3"]
    assert targets.operations[0].index_supervised == 0
    assert targets.operations[2].index_supervised == 1

    gate_anchor = next(a for a in targets.anchors if a.anchor_id == "a2")
    assert gate_anchor.owner_index_full == 1
    assert gate_anchor.owner_index_supervised is None

    dest_anchor = next(a for a in targets.anchors if a.anchor_id == "a3")
    assert dest_anchor.owner_index_full == 2
    assert dest_anchor.owner_index_supervised == 1
    assert targets.ownership_owner_indices == (0, 1)
    assert targets.ownership_anchor_ids == ("a1", "a3")

    # Surviving supervised pair q1->q3 is ineligible (ENVIRONMENTAL_FACT ∉ NAVIGATE).
    pair_q1_q3 = [
        pair
        for pair in targets.h7_pairs
        if pair.source_node_id == "q1" and pair.target_node_id == "q3"
    ]
    assert len(pair_q1_q3) == 1
    assert pair_q1_q3[0].source_index == 0
    assert pair_q1_q3[0].target_index == 1
    assert pair_q1_q3[0].masked
    assert pair_q1_q3[0].mask_reason == "structurally_ineligible"
    assert all(
        pair.source_index in {0, 1} and pair.target_index in {0, 1}
        for pair in targets.h7_pairs
    )

# ---------------------------------------------------------------------------
# NAMING / TASKS
# ---------------------------------------------------------------------------


def test_slot_naming_v1_deterministic():
    assert (
        principal_slot_name(base_name="gate", slot_type=SlotType.LOCATION)
        == "gate_location"
    )
    assert (
        principal_slot_name(
            base_name="Gate",
            slot_type=SlotType.RESOLVED_REFERENCE,
        )
        == "gate_identifier"
    )
    assert fuse_output_slot_name() == "response"
    assert (
        fuse_input_slot_name(source_node_id="op_1", source_slot="gate_location")
        == "op_1__gate_location"
    )
    assert DEFAULT_BASE_BY_OPERATOR_V1[OperatorType.DESCRIBE_ENVIRONMENT] == "scene"
    assert default_base_for_operator(OperatorType.IDENTIFY_ENVIRONMENTAL) == (
        "environment"
    )


def test_slot_naming_v1_fails_explicitly():
    with pytest.raises(SlotNamingError):
        principal_slot_name(base_name="!!!", slot_type=SlotType.LOCATION)
    with pytest.raises(SlotNamingError):
        principal_slot_name(
            base_name="gate",
            slot_type=SlotType.FINAL_RESPONSE,
        )


@pytest.mark.parametrize(
    "bad_name",
    [
        "user's",
        "gate-id",
        "café",
        "",
        "   ",
    ],
)
def test_slot_naming_v1_rejects_unsupported_normalized_forms(bad_name):
    with pytest.raises(SlotNamingError):
        normalize_base_name(bad_name)


def test_task_template_v1_deterministic():
    assert (
        render_answer_task(
            operator=OperatorType.RESOLVE_PERSONAL,
            base_name="gate",
        )
        == "Resolve the user's gate identifier"
    )
    assert (
        render_answer_task(
            operator=OperatorType.LOCATE_ENVIRONMENTAL,
            base_name="gate",
        )
        == "Locate the resolved gate"
    )
    assert render_fuse_task() == "Fuse the terminal answers"
