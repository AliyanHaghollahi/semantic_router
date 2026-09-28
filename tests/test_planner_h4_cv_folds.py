"""Frozen Stage-A v3 TRAIN-only H4 cross-validation folds."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "build_planner_h4_cv_manifest",
    ROOT / "scripts" / "build_planner_h4_cv_manifest.py",
)
assert _SPEC is not None and _SPEC.loader is not None
manifest_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(manifest_mod)

MANIFEST_PATH = ROOT / "dataset" / "planner" / "stage_a_v3_h4_cv_folds.json"


def _load_manifest() -> dict:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _mapping(payload: dict) -> dict[str, int]:
    rows = payload["assignments"]
    assert [row["stage_a_id"] for row in rows] == sorted(row["stage_a_id"] for row in rows)
    return {row["stage_a_id"]: int(row["fold"]) for row in rows}


def test_manifest_matches_deterministic_regeneration():
    first = manifest_mod.build_manifest()
    second = manifest_mod.build_manifest()
    stored = _load_manifest()
    assert first["assignments"] == second["assignments"]
    assert first["cv_fold_fingerprint"] == second["cv_fold_fingerprint"]
    assert first["assignments"] == stored["assignments"]
    assert first["cv_fold_fingerprint"] == stored["cv_fold_fingerprint"]
    encoded = json.dumps(stored, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    assert MANIFEST_PATH.read_bytes() == encoded
    assert hashlib.sha256(encoded).hexdigest() == manifest_mod.manifest_sha256(stored)


def test_train_coverage_is_complete_and_disjoint():
    payload = _load_manifest()
    mapping = _mapping(payload)
    split_ids = manifest_mod.load_split_ids(
        ROOT / manifest_mod.STAGE_A_V3_SPLIT_PATH
    )
    assert len(mapping) == 384
    assert len(set(mapping)) == 384
    assert set(mapping) == split_ids["train"]
    assert set(mapping.values()) == {0, 1, 2, 3, 4}


def test_no_dev_or_test_ids():
    mapping = _mapping(_load_manifest())
    split_ids = manifest_mod.load_split_ids(
        ROOT / manifest_mod.STAGE_A_V3_SPLIT_PATH
    )
    assert not (set(mapping) & split_ids["dev"])
    assert not (set(mapping) & split_ids["test"])
    assert split_ids["dev"]
    assert split_ids["test"]


def test_leakage_groups_stay_inside_one_fold():
    mapping = _mapping(_load_manifest())
    rows = manifest_mod.load_train_rows(
        ROOT / manifest_mod.STAGE_A_V3_STEP_A_PATH,
        set(mapping),
    )
    units = manifest_mod.build_units(rows)
    anchorless = []
    for unit in units:
        folds = {mapping[sid] for sid in unit["ids"]}
        assert len(folds) == 1
        anchorless.extend(unit["anchorless"])
    assert len(anchorless) == 6
    assert len({mapping[sid] for sid in anchorless}) == 1


def test_fold_sizes_and_objective():
    payload = _load_manifest()
    mapping = _mapping(payload)
    sizes = [sum(fold == index for fold in mapping.values()) for index in range(5)]
    assert sizes == [77, 77, 77, 77, 76]
    assert payload["fold_sizes"] == [77, 77, 77, 77, 76]
    assert round(float(payload["objective_value"]), 2) == 539.89
    assert payload["schema_version"] == "planner_h4_cv_folds_v1"
    assert payload["n_examples"] == 384
    assert payload["n_folds"] == 5
    assert "template_group" in payload["grouping_rule"]
    assert "authored_holdout_family" in payload["grouping_rule"]
    assert payload["split_fingerprint"] == (
        "ad221c67fb08290582f863bad682e88d1de7bfd1e127c790aee868030229cd10"
    )
    assert payload["annotation_fingerprint"] == (
        "30ffae30f24c987c5a77a50fa2d9faa9054198e3d4273ccbecdf09d4a7e0c397"
    )


def test_cv_fingerprint_is_stable():
    payload = _load_manifest()
    mapping = _mapping(payload)
    digest = manifest_mod.cv_fold_fingerprint(mapping)
    assert digest == payload["cv_fold_fingerprint"]
    assert digest == manifest_mod.cv_fold_fingerprint(dict(reversed(list(mapping.items()))))
    assert len(digest) == 64
