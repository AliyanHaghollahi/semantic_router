"""Stage-A v3 (H4_REFEXPR_V1) corpus paths and metadata.

v3 keeps the same underlying query IDs and split membership as Stage-A v2.
Annotations may differ under H4_REFEXPR_V1; therefore v3 has its own
annotation/corpus fingerprint. Stage-A v2 paths must remain untouched.

TRAIN/DEV H4_REFEXPR_V1 annotations are **frozen**. TEST annotation migration
remains pending and must stay blind (no TEST annotation content access until
an explicit blind migration step).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from tiergraph.planner.h4_refexpr_v1 import H4_ANCHOR_CONTRACT, SOURCE_CORPUS_V2
from tiergraph.planner.stage_a_v2_spec import (
    STAGE_A_V2_CORPUS_SIZE,
    STAGE_A_V2_DEV_SIZE,
    STAGE_A_V2_SELECTION_PATH,
    STAGE_A_V2_SPLIT_FINGERPRINT,
    STAGE_A_V2_SPLIT_SEED,
    STAGE_A_V2_TEST_SIZE,
    STAGE_A_V2_TRAIN_SIZE,
)

STAGE_A_V3_CORPUS_SIZE: Final[int] = STAGE_A_V2_CORPUS_SIZE
STAGE_A_V3_TRAIN_SIZE: Final[int] = STAGE_A_V2_TRAIN_SIZE
STAGE_A_V3_DEV_SIZE: Final[int] = STAGE_A_V2_DEV_SIZE
STAGE_A_V3_TEST_SIZE: Final[int] = STAGE_A_V2_TEST_SIZE
# Annotation files materialize TRAIN+DEV only until blind TEST migration.
STAGE_A_V3_MATERIALIZED_SIZE: Final[int] = (
    STAGE_A_V3_TRAIN_SIZE + STAGE_A_V3_DEV_SIZE
)
# TRAIN/DEV contract freeze complete; TEST still not migrated.
STAGE_A_V3_TRAIN_DEV_H4_STATUS: Final[str] = "frozen"
STAGE_A_V3_TEST_ANNOTATION_MIGRATION: Final[str] = "pending_blind_migration"
STAGE_A_V3_SPLIT_SEED: Final[int] = STAGE_A_V2_SPLIT_SEED
# Same assignment as v2 → same split membership fingerprint.
STAGE_A_V3_SPLIT_FINGERPRINT: Final[str] = STAGE_A_V2_SPLIT_FINGERPRINT

STAGE_A_V3_H4_ANCHOR_CONTRACT: Final[str] = H4_ANCHOR_CONTRACT
STAGE_A_V3_SOURCE_CORPUS: Final[str] = SOURCE_CORPUS_V2

# Selection is unchanged (queries/IDs); reuse v2 selection path read-only.
STAGE_A_V3_SELECTION_PATH: Final[Path] = STAGE_A_V2_SELECTION_PATH

STAGE_A_V3_STEP_A_PATH: Final[Path] = Path(
    "dataset/planner/stage_a_v3_step_a_annotations.jsonl"
)
STAGE_A_V3_STEP_B_PATH: Final[Path] = Path(
    "dataset/planner/stage_a_v3_step_b_annotations.jsonl"
)
STAGE_A_V3_SPLIT_PATH: Final[Path] = Path(
    "dataset/planner/stage_a_v3_split.jsonl"
)
STAGE_A_V3_BUILD_REPORT_PATH: Final[Path] = Path(
    "dataset/planner/stage_a_v3_h4_refexpr_v1_build_report.json"
)
STAGE_A_V3_HUMAN_REVIEW_QUEUE_PATH: Final[Path] = Path(
    "dataset/planner/stage_a_v3_h4_refexpr_v1_human_review_queue.jsonl"
)
STAGE_A_V3_UNRESOLVED_REVIEW_PATH: Final[Path] = Path(
    "dataset/planner/stage_a_v3_h4_refexpr_v1_unresolved_review.jsonl"
)

# Frozen TRAIN+DEV H4_REFEXPR_V1 annotation fingerprint (Step-A ∥ Step-B).
STAGE_A_V3_ANNOTATION_FINGERPRINT: Final[str] = (
    "1af0b450623eb2ed9d26e0bcf42fa255fe1a26e6a0e1118582c0603abb3af9ca"
)

__all__ = [
    "STAGE_A_V3_ANNOTATION_FINGERPRINT",
    "STAGE_A_V3_BUILD_REPORT_PATH",
    "STAGE_A_V3_CORPUS_SIZE",
    "STAGE_A_V3_DEV_SIZE",
    "STAGE_A_V3_H4_ANCHOR_CONTRACT",
    "STAGE_A_V3_HUMAN_REVIEW_QUEUE_PATH",
    "STAGE_A_V3_MATERIALIZED_SIZE",
    "STAGE_A_V3_SELECTION_PATH",
    "STAGE_A_V3_SOURCE_CORPUS",
    "STAGE_A_V3_SPLIT_FINGERPRINT",
    "STAGE_A_V3_SPLIT_PATH",
    "STAGE_A_V3_SPLIT_SEED",
    "STAGE_A_V3_STEP_A_PATH",
    "STAGE_A_V3_STEP_B_PATH",
    "STAGE_A_V3_TEST_ANNOTATION_MIGRATION",
    "STAGE_A_V3_TEST_SIZE",
    "STAGE_A_V3_TRAIN_DEV_H4_STATUS",
    "STAGE_A_V3_TRAIN_SIZE",
    "STAGE_A_V3_UNRESOLVED_REVIEW_PATH",
]
