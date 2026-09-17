"""Focused tests for free-inference punctuation-safe SLOT_NAMING_V1 derivation."""

from __future__ import annotations

import pytest

from tiergraph.enums import OperatorType
from tiergraph.planner.annotations import ImplicitResolution
from tiergraph.planner.decode import (
    GraphDecoder,
    PlannerPredictions,
    PredictedAnchor,
    PredictedOperation,
)
from tiergraph.planner.naming import (
    SlotNamingError,
    derive_anchor_normalized_name,
    normalize_base_name,
)
from tiergraph.planner.stage_a_to_corpus import (
    derive_anchor_normalized_name as derive_from_corpus_module,
)


def test_derive_apostrophe_doctor_office():
    assert derive_anchor_normalized_name("my doctor's office") == "doctor_s_office"


def test_derive_apostrophe_phone_number():
    assert (
        derive_anchor_normalized_name("my doctor's phone number")
        == "doctor_s_phone_number"
    )


def test_derive_hyphenated_neighborhood():
    assert (
        derive_anchor_normalized_name("this neighborhood clean and well-kept")
        == "neighborhood_clean_and_well_kept"
    )


def test_derive_alphanumeric_unchanged_path():
    assert derive_anchor_normalized_name("gate") == "gate"
    assert derive_anchor_normalized_name("right floor") == "right_floor"


def test_normalize_base_name_still_rejects_raw_apostrophe():
    with pytest.raises(SlotNamingError):
        normalize_base_name("my doctor's office")


def test_stage_a_to_corpus_reexports_same_helper():
    assert derive_from_corpus_module is derive_anchor_normalized_name
    assert derive_from_corpus_module("my doctor's office") == "doctor_s_office"


def test_graph_decoder_accepts_none_normalized_name_with_apostrophe():
    query = "Where is my doctor's office?"
    start, end = 9, 27
    assert query[start:end] == "my doctor's office"
    predictions = PlannerPredictions(
        operations=(
            PredictedOperation(
                start=0,
                end=len(query) - 1,
                operator=OperatorType.LOCATE_ENVIRONMENTAL,
            ),
        ),
        anchors=(
            PredictedAnchor(
                start=start,
                end=end,
                text="my doctor's office",
                owner_index=0,
                implicit_resolution=ImplicitResolution.NONE,
                normalized_name=None,
            ),
        ),
        dependency_pairs=frozenset(),
        aux_query_type=None,
    )
    decoded = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="pred::doctor_office",
    )
    slot_names = [
        slot
        for node in decoded.graph.nodes
        for slot in list(node.produced_outputs.keys())
        + list(node.required_inputs.keys())
    ]
    assert any("doctor_s_office" in name for name in slot_names)


def test_graph_decoder_over_wide_span_offsets_unchanged():
    query = "Is this neighborhood clean and well-kept?"
    start, end = 3, 40
    surface = query[start:end]
    assert surface == "this neighborhood clean and well-kept"
    predictions = PlannerPredictions(
        operations=(
            PredictedOperation(
                start=0,
                end=len(query) - 1,
                operator=OperatorType.IDENTIFY_ENVIRONMENTAL,
            ),
        ),
        anchors=(
            PredictedAnchor(
                start=start,
                end=end,
                text=surface,
                owner_index=0,
                implicit_resolution=ImplicitResolution.NONE,
                normalized_name=None,
            ),
        ),
        dependency_pairs=frozenset(),
        aux_query_type=None,
    )
    decoded = GraphDecoder().decode(
        predictions,
        query=query,
        graph_id="pred::over_wide",
    )
    assert derive_anchor_normalized_name(surface) == "neighborhood_clean_and_well_kept"
    assert decoded.graph.nodes
    assert predictions.anchors[0].start == start
    assert predictions.anchors[0].end == end
    assert predictions.anchors[0].text == surface


def test_predict_structures_naming_branch_preserves_span_geometry():
    from tiergraph.planner.model import PlannerModel

    span_text = "my doctor's office"
    name = derive_anchor_normalized_name(span_text)
    assert name == "doctor_s_office"
    anchor = PredictedAnchor(
        start=9,
        end=27,
        text=span_text,
        owner_index=0,
        implicit_resolution=ImplicitResolution.NONE,
        normalized_name=name,
    )
    assert anchor.start == 9 and anchor.end == 27
    assert anchor.text == span_text
    assert callable(PlannerModel.predict_structures)
