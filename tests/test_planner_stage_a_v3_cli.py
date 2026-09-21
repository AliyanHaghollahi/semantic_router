"""Focused tests for Stage-A v3 CLI / train corpus selection."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tiergraph.planner.annotation_step_a import fingerprint_file
from tiergraph.planner.stage_a_v2_spec import (
    STAGE_A_V2_SPLIT_FINGERPRINT,
    STAGE_A_V2_STEP_A_PATH,
    STAGE_A_V2_STEP_B_PATH,
)
from tiergraph.planner.stage_a_v3_h4_build import annotation_corpus_fingerprint
from tiergraph.planner.stage_a_v3_spec import (
    STAGE_A_V3_ANNOTATION_FINGERPRINT,
    STAGE_A_V3_DEV_SIZE,
    STAGE_A_V3_MATERIALIZED_SIZE,
    STAGE_A_V3_SPLIT_FINGERPRINT,
    STAGE_A_V3_STEP_A_PATH,
    STAGE_A_V3_STEP_B_PATH,
    STAGE_A_V3_TRAIN_SIZE,
)
from tiergraph.planner.train import (
    EXPECTED_STAGE_A_SPLIT_FINGERPRINT,
    TrainConfig,
    load_and_split_for_config,
    load_and_split_stage_a_v2,
    load_and_split_stage_a_v3,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "train_planner.py"


def _import_cli():
    sys.path.insert(0, str(ROOT / "scripts"))
    import train_planner as cli  # type: ignore

    return cli


def test_cli_help_lists_v2_and_v3():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--v2" in completed.stdout
    assert "--v3" in completed.stdout
    assert "Stage-A v2 frozen corpus" in completed.stdout
    assert "Stage-A v3 frozen H4_REFEXPR_V1 TRAIN+DEV corpus" in completed.stdout


def test_v2_and_v3_mutually_exclusive():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--v2", "--v3", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    # argparse rejects mutually exclusive before help in some versions;
    # force conflict via parse of non-help.
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--v2", "--v3", "--eval-only", "--checkpoint", "x"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "not allowed with argument" in (completed.stderr + completed.stdout).lower() or (
        "mutually exclusive" in (completed.stderr + completed.stdout).lower()
    )


def test_cli_default_paths_unchanged():
    cli = _import_cli()
    args = cli.build_parser().parse_args([])
    expected, config = cli._resolve_cli_defaults(args)
    assert expected == EXPECTED_STAGE_A_SPLIT_FINGERPRINT
    assert config.corpus_version == "v1"
    assert "stage_a_step_a_annotations.jsonl" in config.step_a_path.replace("\\", "/")
    assert "stage_a_v2" not in config.step_a_path
    assert "stage_a_v3" not in config.step_a_path


def test_cli_v2_path_selection_unchanged():
    cli = _import_cli()
    args = cli.build_parser().parse_args(["--v2"])
    expected, config = cli._resolve_cli_defaults(args)
    assert expected == STAGE_A_V2_SPLIT_FINGERPRINT
    assert config.corpus_version == "v2"
    assert Path(config.step_a_path) == Path(STAGE_A_V2_STEP_A_PATH)
    assert Path(config.step_b_path) == Path(STAGE_A_V2_STEP_B_PATH)
    assert config.h2_bio_class_weights is None
    assert config.h4_bio_class_weights is None


def test_cli_v3_path_and_fingerprint_selection():
    cli = _import_cli()
    args = cli.build_parser().parse_args(["--v3"])
    expected, config = cli._resolve_cli_defaults(args)
    assert expected == STAGE_A_V3_SPLIT_FINGERPRINT
    assert config.corpus_version == "v3"
    assert Path(config.step_a_path) == Path(STAGE_A_V3_STEP_A_PATH)
    assert Path(config.step_b_path) == Path(STAGE_A_V3_STEP_B_PATH)
    assert config.h2_bio_class_weights is None
    assert config.h4_bio_class_weights is None


def test_load_and_split_stage_a_v3_train_dev_only_no_test_annotations():
    step_a = ROOT / STAGE_A_V3_STEP_A_PATH
    step_b = ROOT / STAGE_A_V3_STEP_B_PATH
    if not step_a.is_file() or not step_b.is_file():
        pytest.skip("v3 corpus not built yet")

    before_a = fingerprint_file(step_a)
    before_b = fingerprint_file(step_b)
    config = TrainConfig(
        corpus_version="v3",
        step_a_path=str(step_a),
        step_b_path=str(step_b),
        device="cpu",
    )
    split, after_a, after_b = load_and_split_stage_a_v3(config)
    assert after_a == before_a
    assert after_b == before_b
    assert len(split.train) == STAGE_A_V3_TRAIN_SIZE
    assert len(split.dev) == STAGE_A_V3_DEV_SIZE
    assert len(split.test) == 0
    assert len(split.train) + len(split.dev) == STAGE_A_V3_MATERIALIZED_SIZE
    assert split.fingerprint == STAGE_A_V3_SPLIT_FINGERPRINT
    assert (
        annotation_corpus_fingerprint(step_a, step_b)
        == STAGE_A_V3_ANNOTATION_FINGERPRINT
        == "1af0b450623eb2ed9d26e0bcf42fa255fe1a26e6a0e1118582c0603abb3af9ca"
    )
    assert split.report["n_test_annotations_materialized"] == 0
    assert split.report["test_annotation_migration"] == "pending_blind_migration"


def test_load_and_split_for_config_routes_v3():
    step_a = ROOT / STAGE_A_V3_STEP_A_PATH
    if not step_a.is_file():
        pytest.skip("v3 corpus not built yet")
    split, _, _ = load_and_split_for_config(
        TrainConfig(
            corpus_version="v3",
            step_a_path=str(STAGE_A_V3_STEP_A_PATH),
            step_b_path=str(STAGE_A_V3_STEP_B_PATH),
            device="cpu",
        )
    )
    assert len(split.test) == 0
    assert len(split.train) == 384


def test_v2_load_still_materializes_test():
    split, _, _ = load_and_split_stage_a_v2(
        TrainConfig(
            corpus_version="v2",
            step_a_path=str(ROOT / STAGE_A_V2_STEP_A_PATH),
            step_b_path=str(ROOT / STAGE_A_V2_STEP_B_PATH),
            device="cpu",
        )
    )
    assert len(split.test) == 48
    assert split.fingerprint == STAGE_A_V2_SPLIT_FINGERPRINT


def test_cli_v3_rejects_eval_only_test_split():
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--v3",
            "--eval-only",
            "--checkpoint",
            "artifacts/does_not_matter.pt",
            "--split",
            "test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "TEST annotations are not materialized" in completed.stderr
