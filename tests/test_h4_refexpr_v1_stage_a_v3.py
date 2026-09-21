"""Focused tests for H4_REFEXPR_V1 contract and Stage-A v3 Phase-1 scaffold."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiergraph.planner.annotation_step_a import (
    StageAStepAAnnotation,
    StepAAnchor,
    fingerprint_file,
    load_step_a_annotations,
)
from tiergraph.planner.annotation_step_b import (
    StageAStepBAnnotation,
    StepBAnchorDecision,
    load_step_b_annotations,
)
from tiergraph.planner.annotations import ImplicitResolution
from tiergraph.planner.h4_refexpr_v1 import (
    APPROVED_FINAL_REPLACE,
    APPROVED_R4_KEEP,
    APPROVED_R5_REMOVE,
    APPROVED_R6_SHORTEN,
    HUMAN_REVIEW_IDS,
    SAFE_MECHANICAL_IDS,
    UNRESOLVED_HUMAN_REVIEW_IDS,
    classify_human_review_reasons,
    is_bare_demonstrative_pronoun,
    is_determiner_np_demo,
    is_possessive_np,
    is_safe_mechanical_anchor_text,
    shorten_safe_mechanical_anchor,
    sync_step_b_text_preserve_h5_h6,
)
from tiergraph.planner.stage_a_v2_spec import (
    STAGE_A_V2_SPLIT_PATH,
    STAGE_A_V2_STEP_A_PATH,
    STAGE_A_V2_STEP_B_PATH,
)
from tiergraph.planner.stage_a_v3_h4_build import (
    annotation_corpus_fingerprint,
    build_stage_a_v3_h4_refexpr_v1,
    load_annotation_rows_for_ids,
    load_train_dev_split_ids,
)
from tiergraph.planner.stage_a_v3_spec import (
    STAGE_A_V3_ANNOTATION_FINGERPRINT,
    STAGE_A_V3_BUILD_REPORT_PATH,
    STAGE_A_V3_DEV_SIZE,
    STAGE_A_V3_HUMAN_REVIEW_QUEUE_PATH,
    STAGE_A_V3_MATERIALIZED_SIZE,
    STAGE_A_V3_SPLIT_PATH,
    STAGE_A_V3_STEP_A_PATH,
    STAGE_A_V3_STEP_B_PATH,
    STAGE_A_V3_TEST_ANNOTATION_MIGRATION,
    STAGE_A_V3_TRAIN_SIZE,
    STAGE_A_V3_UNRESOLVED_REVIEW_PATH,
)

ROOT = Path(__file__).resolve().parents[1]


def test_standalone_demonstrative_pronoun_is_bare_this_that():
    assert is_bare_demonstrative_pronoun("this")
    assert is_bare_demonstrative_pronoun("that")
    assert is_safe_mechanical_anchor_text("this the right terminal")
    query = "Is this the right terminal?"
    anchor = StepAAnchor(
        anchor_index=0,
        text="this the right terminal",
        char_start=3,
        char_end=26,
    )
    shortened = shorten_safe_mechanical_anchor(anchor, query)
    assert shortened.text == "this"
    assert is_bare_demonstrative_pronoun(shortened.text)
    assert query[shortened.char_start:shortened.char_end] == "this"


def test_demonstrative_determiner_np_remains_full_np():
    assert is_determiner_np_demo("this tray")
    assert not is_bare_demonstrative_pronoun("this tray")
    assert not is_safe_mechanical_anchor_text("this tray")
    assert classify_human_review_reasons("this tray") == ()


def test_possessive_np_remains_intact():
    assert is_possessive_np("my ticket")
    assert classify_human_review_reasons("my ticket") == ()
    assert not is_safe_mechanical_anchor_text("my doctor's office")


def test_safe_mechanical_preserves_h5_h6_on_text_sync():
    decision = StepBAnchorDecision(
        anchor_index=0,
        text="this the right terminal",
        implicit_resolution=ImplicitResolution.NONE,
        owner_operation_index=0,
    )
    synced = sync_step_b_text_preserve_h5_h6(decision, "this")
    assert synced.text == "this"
    assert synced.implicit_resolution is ImplicitResolution.NONE
    assert synced.owner_operation_index == 0


def test_build_applies_approved_only_leaves_unresolved():
    report = build_stage_a_v3_h4_refexpr_v1(root=ROOT, write=False)
    changed_mech = {
        change["stage_a_id"] for change in report["safe_mechanical"]["changes"]
    }
    assert changed_mech == set(SAFE_MECHANICAL_IDS)
    assert report["approved_human_review"]["r6_shortened"] == len(APPROVED_R6_SHORTEN)
    assert report["approved_human_review"]["r5_removed"] == len(APPROVED_R5_REMOVE)
    assert report["approved_human_review"]["final_replaced"] == len(
        APPROVED_FINAL_REPLACE
    )
    assert report["human_review"]["unresolved_ids"] == []
    assert report["human_review"]["unresolved_count"] == 0
    assert report["n_test_annotations_materialized"] == 0
    assert report["n_step_a_rows_parsed"] == STAGE_A_V3_MATERIALIZED_SIZE
    assert report["test_annotation_migration"] == STAGE_A_V3_TEST_ANNOTATION_MIGRATION
    assert STAGE_A_V3_TEST_ANNOTATION_MIGRATION == "pending_blind_migration"
    from tiergraph.planner.stage_a_v3_spec import STAGE_A_V3_TRAIN_DEV_H4_STATUS

    assert STAGE_A_V3_TRAIN_DEV_H4_STATUS == "frozen"
    assert report["train_dev_h4_status"] == "frozen"
    assert report["human_review"]["human_approved_edits_only"] is True
    assert "auto_rewritten_approved_only" not in report["human_review"]


@pytest.mark.parametrize("stage_a_id", sorted(APPROVED_R4_KEEP))
def test_r4_keep_match_v2(stage_a_id):
    v3_a_path = ROOT / STAGE_A_V3_STEP_A_PATH
    if not v3_a_path.is_file():
        pytest.skip("v3 corpus not built yet")
    ids = frozenset({stage_a_id})
    v2 = load_annotation_rows_for_ids(
        ROOT / STAGE_A_V2_STEP_A_PATH, ids, model_cls=StageAStepAAnnotation
    )
    v3 = load_annotation_rows_for_ids(
        v3_a_path, ids, model_cls=StageAStepAAnnotation
    )
    assert v3[stage_a_id].model_dump(mode="json") == v2[stage_a_id].model_dump(
        mode="json"
    )


@pytest.mark.parametrize("stage_a_id", sorted(APPROVED_FINAL_REPLACE))
def test_approved_final_replace_on_disk(stage_a_id):
    v3_a_path = ROOT / STAGE_A_V3_STEP_A_PATH
    if not v3_a_path.is_file():
        pytest.skip("v3 corpus not built yet")
    spec = APPROVED_FINAL_REPLACE[stage_a_id]
    ids = frozenset({stage_a_id})
    v3 = load_annotation_rows_for_ids(
        v3_a_path, ids, model_cls=StageAStepAAnnotation
    )
    v3_b = load_annotation_rows_for_ids(
        ROOT / STAGE_A_V3_STEP_B_PATH, ids, model_cls=StageAStepBAnnotation
    )
    texts = [a.text for a in v3[stage_a_id].anchors]
    assert str(spec["old_text"]) not in texts
    expected = list(spec["new_anchors"])  # type: ignore[arg-type]
    assert len(v3[stage_a_id].anchors) == len(expected)
    for anc, exp, dec in zip(
        v3[stage_a_id].anchors,
        sorted(expected, key=lambda s: (s["char_start"], s["char_end"])),
        v3_b[stage_a_id].anchor_decisions,
        strict=True,
    ):
        assert anc.text == exp["text"]
        assert anc.char_start == exp["char_start"]
        assert anc.char_end == exp["char_end"]
        assert v3[stage_a_id].query[anc.char_start:anc.char_end] == anc.text
        assert dec.implicit_resolution.value == exp["h5"]
        assert dec.owner_operation_index == exp["h6"]
    # H7 preserved for sa_0456 / sa_0464
    if stage_a_id in {"sa_0456", "sa_0464"}:
        deps = [
            (d.source_operation_index, d.target_operation_index)
            for d in v3_b[stage_a_id].dependencies
        ]
        assert deps == [(0, 1)]
    if stage_a_id == "sa_0339":
        assert [a.text for a in v3[stage_a_id].anchors] == ["this bus", "my hotel"]
        assert v3_b[stage_a_id].anchor_decisions[0].implicit_resolution.value == "NONE"
        assert (
            v3_b[stage_a_id].anchor_decisions[1].implicit_resolution.value
            == "IMPLICIT_RESOLVE_PERSONAL"
        )


@pytest.mark.parametrize("stage_a_id,old_text,new_text", [
    (sid, old, new) for sid, (old, new) in sorted(APPROVED_R6_SHORTEN.items())
])
def test_approved_r6_shortened_on_disk(stage_a_id, old_text, new_text):
    v3_a_path = ROOT / STAGE_A_V3_STEP_A_PATH
    if not v3_a_path.is_file():
        pytest.skip("v3 corpus not built yet")
    ids = frozenset({stage_a_id})
    v3 = load_annotation_rows_for_ids(
        v3_a_path, ids, model_cls=StageAStepAAnnotation
    )
    v3_b = load_annotation_rows_for_ids(
        ROOT / STAGE_A_V3_STEP_B_PATH, ids, model_cls=StageAStepBAnnotation
    )
    v2_b = load_annotation_rows_for_ids(
        ROOT / STAGE_A_V2_STEP_B_PATH, ids, model_cls=StageAStepBAnnotation
    )
    texts = [a.text for a in v3[stage_a_id].anchors]
    assert new_text in texts
    assert old_text not in texts
    # H5/H6 preserved on shortened anchor index 0 for these examples
    idx = next(
        i for i, a in enumerate(v3[stage_a_id].anchors) if a.text == new_text
    )
    assert (
        v3_b[stage_a_id].anchor_decisions[idx].implicit_resolution
        == v2_b[stage_a_id].anchor_decisions[idx].implicit_resolution
    )
    assert (
        v3_b[stage_a_id].anchor_decisions[idx].owner_operation_index
        == v2_b[stage_a_id].anchor_decisions[idx].owner_operation_index
    )


@pytest.mark.parametrize("stage_a_id,remove_text", sorted(APPROVED_R5_REMOVE.items()))
def test_approved_r5_removed_on_disk(stage_a_id, remove_text):
    v3_a_path = ROOT / STAGE_A_V3_STEP_A_PATH
    if not v3_a_path.is_file():
        pytest.skip("v3 corpus not built yet")
    ids = frozenset({stage_a_id})
    v3 = load_annotation_rows_for_ids(
        v3_a_path, ids, model_cls=StageAStepAAnnotation
    )
    texts = [a.text for a in v3[stage_a_id].anchors]
    assert remove_text not in texts
    assert len(v3[stage_a_id].anchors) == 0


def test_v2_files_unchanged_byte_for_byte():
    a_path = ROOT / STAGE_A_V2_STEP_A_PATH
    b_path = ROOT / STAGE_A_V2_STEP_B_PATH
    before_a = fingerprint_file(a_path)
    before_b = fingerprint_file(b_path)
    bytes_a = a_path.read_bytes()
    bytes_b = b_path.read_bytes()
    assert fingerprint_file(a_path) == before_a
    assert fingerprint_file(b_path) == before_b
    assert a_path.read_bytes() == bytes_a
    assert b_path.read_bytes() == bytes_b


def test_frozen_annotation_fingerprint_exact():
    expected = "1af0b450623eb2ed9d26e0bcf42fa255fe1a26e6a0e1118582c0603abb3af9ca"
    assert STAGE_A_V3_ANNOTATION_FINGERPRINT == expected
    v3_a = ROOT / STAGE_A_V3_STEP_A_PATH
    v3_b = ROOT / STAGE_A_V3_STEP_B_PATH
    if not v3_a.is_file() or not v3_b.is_file():
        pytest.skip("v3 corpus not built yet")
    assert annotation_corpus_fingerprint(v3_a, v3_b) == expected


def test_v3_fingerprint_distinct_from_v2_when_built():
    v3_a = ROOT / STAGE_A_V3_STEP_A_PATH
    v3_b = ROOT / STAGE_A_V3_STEP_B_PATH
    if not v3_a.is_file() or not v3_b.is_file():
        pytest.skip("v3 corpus not built yet")
    v2_fp = annotation_corpus_fingerprint(
        ROOT / STAGE_A_V2_STEP_A_PATH, ROOT / STAGE_A_V2_STEP_B_PATH
    )
    v3_fp = annotation_corpus_fingerprint(v3_a, v3_b)
    assert v3_fp != v2_fp
    assert STAGE_A_V3_ANNOTATION_FINGERPRINT == v3_fp


def test_v3_materializes_train_dev_only_no_test_annotations():
    v3_a = ROOT / STAGE_A_V3_STEP_A_PATH
    v3_b = ROOT / STAGE_A_V3_STEP_B_PATH
    report_path = ROOT / STAGE_A_V3_BUILD_REPORT_PATH
    if not v3_a.is_file() or not v3_b.is_file() or not report_path.is_file():
        pytest.skip("v3 corpus not built yet")

    step_a = load_step_a_annotations(v3_a)
    step_b = load_step_b_annotations(v3_b)
    assert len(step_a) == STAGE_A_V3_MATERIALIZED_SIZE == 432
    assert len(step_b) == STAGE_A_V3_MATERIALIZED_SIZE == 432

    train_ids, dev_ids = load_train_dev_split_ids(ROOT / STAGE_A_V3_SPLIT_PATH)
    assert len(train_ids) == STAGE_A_V3_TRAIN_SIZE == 384
    assert len(dev_ids) == STAGE_A_V3_DEV_SIZE == 48
    assert {r.stage_a_id for r in step_a} == (train_ids | dev_ids)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["n_test_annotations_materialized"] == 0
    assert report["n_step_a_rows_written"] == 432
    assert report["totals"]["anchors_shortened_approved_r6"] == 8
    assert report["totals"]["anchors_removed_approved_r5"] == 7
    assert report["totals"]["anchors_final_replaced_examples"] == 4
    assert report["human_review"]["unresolved_count"] == 0


def test_v3_train_dev_counts_and_split_membership_match_v2():
    v3_split = ROOT / STAGE_A_V3_SPLIT_PATH
    if not v3_split.is_file():
        pytest.skip("v3 corpus not built yet")
    assert v3_split.read_bytes() == (ROOT / STAGE_A_V2_SPLIT_PATH).read_bytes()
    train_ids, dev_ids = load_train_dev_split_ids(v3_split)
    v2_train, v2_dev = load_train_dev_split_ids(ROOT / STAGE_A_V2_SPLIT_PATH)
    assert train_ids == v2_train
    assert dev_ids == v2_dev


def test_mechanical_shortening_on_disk_and_h5_h6_preserved():
    v3_a_path = ROOT / STAGE_A_V3_STEP_A_PATH
    report_path = ROOT / STAGE_A_V3_BUILD_REPORT_PATH
    if not v3_a_path.is_file() or not report_path.is_file():
        pytest.skip("v3 corpus not built yet")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["safe_mechanical"]["anchors_shortened_train"] == 8
    assert report["safe_mechanical"]["anchors_shortened_dev"] == 1

    mech_ids = frozenset(SAFE_MECHANICAL_IDS)
    v3_a = load_annotation_rows_for_ids(
        v3_a_path, mech_ids, model_cls=StageAStepAAnnotation
    )
    v3_b = load_annotation_rows_for_ids(
        ROOT / STAGE_A_V3_STEP_B_PATH, mech_ids, model_cls=StageAStepBAnnotation
    )
    v2_b = load_annotation_rows_for_ids(
        ROOT / STAGE_A_V2_STEP_B_PATH, mech_ids, model_cls=StageAStepBAnnotation
    )
    for change in report["safe_mechanical"]["changes"]:
        sid = change["stage_a_id"]
        idx = change["anchor_index"]
        anchor = v3_a[sid].anchors[idx]
        assert anchor.text == change["new_text"]
        assert is_bare_demonstrative_pronoun(anchor.text)
        decision = v3_b[sid].anchor_decisions[idx]
        old = v2_b[sid].anchor_decisions[idx]
        assert decision.implicit_resolution == old.implicit_resolution
        assert decision.owner_operation_index == old.owner_operation_index


def test_unresolved_review_report_on_disk():
    path = ROOT / STAGE_A_V3_UNRESOLVED_REVIEW_PATH
    if not path.is_file():
        pytest.skip("v3 corpus not built yet")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows == []
    assert UNRESOLVED_HUMAN_REVIEW_IDS == frozenset()


def test_human_review_queue_on_disk_fields():
    queue_path = ROOT / STAGE_A_V3_HUMAN_REVIEW_QUEUE_PATH
    if not queue_path.is_file():
        pytest.skip("v3 corpus not built yet")
    rows = [
        json.loads(line)
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == len(HUMAN_REVIEW_IDS)
    for row in rows:
        assert row["stage_a_id"] in HUMAN_REVIEW_IDS
        assert row["decision_status"] in {
            "applied_r6_shorten",
            "applied_r5_remove",
            "applied_final_replace",
            "kept_r4",
            "unresolved",
        }
        assert "human_approved_applied" in row
        assert "auto_rewritten" not in row
        if row["decision_status"] in {
            "applied_r6_shorten",
            "applied_r5_remove",
            "applied_final_replace",
        }:
            assert row["human_approved_applied"] is True
        else:
            assert row["human_approved_applied"] is False
