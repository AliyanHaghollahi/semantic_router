#!/usr/bin/env python3
"""Freeze the Stage-A v3 TRAIN-only H4 cross-validation folds.

Reproduces the selected assignment (objective 539.89). It does not search
for a better fold map. DEV and TEST annotations are not loaded. DEV and TEST
ids are read from the split manifest only so they can be rejected.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.stage_a_v2_spec import resolve_authored_holdout_family
from tiergraph.planner.stage_a_v2_split import _UnionFind, enrich_row_for_split
from tiergraph.planner.stage_a_v3_spec import (
    STAGE_A_V3_ANNOTATION_FINGERPRINT,
    STAGE_A_V3_SPLIT_FINGERPRINT,
    STAGE_A_V3_SPLIT_PATH,
    STAGE_A_V3_STEP_A_PATH,
    STAGE_A_V3_TRAIN_SIZE,
)

SCHEMA_VERSION = "planner_h4_cv_folds_v1"
GROUPING_RULE = (
    "connected components of template_group UNION authored_holdout_family"
)
OBJECTIVE_DEFINITION = (
    "For each fold of size n, sum over the five final_bucket counts of "
    "4 * (count - global_count * n / 384)^2, plus the unweighted squared "
    "deviation of anchor buckets 0/1/2/3+ and of single-op versus multi-op. "
    "Assignment is largest-group-first greedy into folds 0..4, then "
    "best-improvement moves, pair swaps, and 2-for-1 exchanges while every "
    "fold size stays at most 77. This is the frozen procedure, not a search "
    "for a new map."
)
EXPECTED_OBJECTIVE = 539.89
EXPECTED_FOLD_SIZES = (77, 77, 77, 77, 76)
# Personal, Environmental, MIXED_IMPLICIT, MIXED_PARALLEL, MIXED_SEQUENTIAL,
# a0, a1, a2, a3, single, multi.
EXPECTED_FOLD_PROFILES = (
    (16, 14, 14, 20, 13, 0, 45, 25, 7, 48, 29),
    (14, 16, 17, 15, 15, 6, 45, 24, 2, 48, 29),
    (22, 13, 14, 13, 15, 0, 47, 27, 3, 49, 28),
    (12, 17, 16, 15, 17, 0, 46, 26, 5, 48, 29),
    (13, 17, 16, 14, 16, 0, 46, 26, 4, 48, 28),
)
STRUCTURAL_IMBALANCE_NOTES = (
    "All 6 anchorless TRAIN examples sit in one leakage component, template other_pe, and therefore one fold.",
    "what_is_my_X is one leakage component and forces one fold to contain at least 22 Personal examples.",
    "coord_is_and is one leakage component and forces one fold to contain at least 20 MIXED_PARALLEL examples.",
    "MIXED_SEQUENTIAL counts in the frozen map range from 13 to 17.",
)
MANIFEST_PATH = ROOT / "dataset" / "planner" / "stage_a_v3_h4_cv_folds.json"
SPLIT_FINGERPRINT = (
    "ad221c67fb08290582f863bad682e88d1de7bfd1e127c790aee868030229cd10"
)
ANNOTATION_FINGERPRINT = (
    "30ffae30f24c987c5a77a50fa2d9faa9054198e3d4273ccbecdf09d4a7e0c397"
)

BUCKETS = (
    "Personal",
    "Environmental",
    "MIXED_IMPLICIT",
    "MIXED_PARALLEL",
    "MIXED_SEQUENTIAL",
)
KEYS = (
    tuple(f"b:{bucket}" for bucket in BUCKETS)
    + ("a0", "a1", "a2", "a3", "op:single", "op:multi")
)
WEIGHTS = (4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def _anchor_bucket(n_anchors: int) -> str:
    if n_anchors <= 0:
        return "a0"
    if n_anchors == 1:
        return "a1"
    if n_anchors == 2:
        return "a2"
    return "a3"


def load_split_ids(split_path: Path) -> dict[str, set[str]]:
    """Return split -> ids from the frozen membership file. No annotations."""
    ids: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    with split_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            split_name = str(row["split"])
            if split_name not in ids:
                raise RuntimeError(f"unknown split label {split_name!r}")
            ids[split_name].add(str(row["stage_a_id"]))
    return ids


def load_train_rows(step_a_path: Path, train_ids: set[str]) -> dict[str, dict]:
    """Parse Step-A rows for TRAIN ids only. Other lines are not decoded."""
    needles = {sid: f'"stage_a_id": "{sid}"' for sid in train_ids}
    rows: dict[str, dict] = {}
    with step_a_path.open(encoding="utf-8") as handle:
        for line in handle:
            matched = next(
                (sid for sid, needle in needles.items() if needle in line),
                None,
            )
            if matched is None:
                continue
            payload = json.loads(line)
            if str(payload.get("stage_a_id")) != matched:
                raise RuntimeError(f"stage_a_id mismatch for {matched}")
            if matched in rows:
                raise RuntimeError(f"duplicate TRAIN stage_a_id {matched}")
            rows[matched] = payload
    missing = train_ids - set(rows)
    if missing:
        raise RuntimeError(f"missing TRAIN annotations: {sorted(missing)[:5]}")
    return rows


def _fold_cost(vec: list[int], n: int, global_counts: list[int]) -> float:
    if n == 0:
        return 0.0
    cost = 0.0
    for index, weight in enumerate(WEIGHTS):
        diff = vec[index] - global_counts[index] * n / 384.0
        cost += weight * diff * diff
    return cost


def _add(dst: list[int], src: tuple[int, ...] | list[int], sign: int = 1) -> None:
    for index, value in enumerate(src):
        dst[index] += sign * value


def build_units(rows: dict[str, dict]) -> list[dict]:
    """Leakage components: template_group union authored_holdout_family."""
    enriched = {sid: enrich_row_for_split(row) for sid, row in rows.items()}
    union = _UnionFind()
    links: dict[str, list[str]] = defaultdict(list)
    for sid, row in enriched.items():
        union.add(sid)
        links["tg:" + str(row["template_group"])].append(sid)
        family = resolve_authored_holdout_family(row)
        if family:
            links["fam:" + family].append(sid)
    for members in links.values():
        anchor = members[0]
        for other in members[1:]:
            union.union(anchor, other)
    grouped: dict[str, list[str]] = defaultdict(list)
    for sid in enriched:
        grouped[union.find(sid)].append(sid)

    units = []
    for member_ids in grouped.values():
        ordered = tuple(sorted(member_ids))
        profile: Counter[str] = Counter()
        anchorless: list[str] = []
        for sid in ordered:
            row = enriched[sid]
            n_anchors = len(row.get("anchors") or [])
            n_ops = len(row.get("operations") or [])
            profile["b:" + str(row["final_bucket"])] += 1
            profile[_anchor_bucket(n_anchors)] += 1
            profile["op:" + ("single" if n_ops <= 1 else "multi")] += 1
            if n_anchors == 0:
                anchorless.append(sid)
        units.append(
            {
                "ids": ordered,
                "key": ordered[0],
                "vec": tuple(profile[key] for key in KEYS),
                "n": len(ordered),
                "anchorless": tuple(anchorless),
            }
        )
    units.sort(key=lambda unit: (-unit["n"], unit["key"]))
    return units


def _objective(counts: list[list[int]], sizes: list[int], global_counts: list[int]) -> float:
    return sum(
        _fold_cost(counts[fold], sizes[fold], global_counts) for fold in range(5)
    )


def _materialize(
    units: list[dict],
    assign: list[int],
) -> tuple[list[list[int]], list[int]]:
    counts = [[0] * 11 for _ in range(5)]
    sizes = [0] * 5
    for index, fold in enumerate(assign):
        _add(counts[fold], units[index]["vec"])
        sizes[fold] += units[index]["n"]
    return counts, sizes


def _greedy(units: list[dict], global_counts: list[int]) -> list[int]:
    """Largest groups first, preferring folds 0, 1, 2, 3, 4 on ties."""
    assign = [-1] * len(units)
    counts = [[0] * 11 for _ in range(5)]
    sizes = [0] * 5
    for index, unit in enumerate(units):
        base = _objective(counts, sizes, global_counts)
        best: tuple[tuple[float, int, int], int] | None = None
        for fold in range(5):
            if sizes[fold] + unit["n"] > 77:
                continue
            updated = counts[fold][:]
            _add(updated, unit["vec"])
            cost = (
                base
                - _fold_cost(counts[fold], sizes[fold], global_counts)
                + _fold_cost(updated, sizes[fold] + unit["n"], global_counts)
            )
            candidate = (cost, sizes[fold] + unit["n"], fold)
            if best is None or candidate < best[0]:
                best = (candidate, fold)
        if best is None:
            raise RuntimeError(f"no legal fold for group {unit['key']}")
        fold = best[1]
        assign[index] = fold
        _add(counts[fold], unit["vec"])
        sizes[fold] += unit["n"]
    return assign


def _search(
    units: list[dict],
    assign: list[int],
    global_counts: list[int],
    *,
    max_steps: int = 40,
) -> list[int]:
    """Frozen best-improvement search. Stops at the first local optimum."""
    counts, sizes = _materialize(units, assign)
    steps = 0
    while steps < max_steps:
        base = _objective(counts, sizes, global_counts)
        best: tuple[tuple, tuple] | None = None
        members = [
            [index for index, fold in enumerate(assign) if fold == fold_id]
            for fold_id in range(5)
        ]
        for fold, group_ids in enumerate(members):
            for index in group_ids:
                size = units[index]["n"]
                vec = units[index]["vec"]
                for dest in range(5):
                    if dest == fold or sizes[dest] + size > 77:
                        continue
                    source_vec = counts[fold][:]
                    dest_vec = counts[dest][:]
                    _add(source_vec, vec, -1)
                    _add(dest_vec, vec)
                    cost = (
                        base
                        - _fold_cost(counts[fold], sizes[fold], global_counts)
                        - _fold_cost(counts[dest], sizes[dest], global_counts)
                        + _fold_cost(source_vec, sizes[fold] - size, global_counts)
                        + _fold_cost(dest_vec, sizes[dest] + size, global_counts)
                    )
                    candidate = (cost, "move", units[index]["key"], dest)
                    if best is None or candidate < best[0]:
                        best = (candidate, ("move", index, fold, dest))
        for left in range(5):
            for right in range(left + 1, 5):
                for left_index in members[left]:
                    for right_index in members[right]:
                        left_n = units[left_index]["n"]
                        right_n = units[right_index]["n"]
                        new_left = sizes[left] - left_n + right_n
                        new_right = sizes[right] - right_n + left_n
                        if new_left > 77 or new_right > 77:
                            continue
                        left_vec = counts[left][:]
                        right_vec = counts[right][:]
                        _add(left_vec, units[left_index]["vec"], -1)
                        _add(left_vec, units[right_index]["vec"])
                        _add(right_vec, units[right_index]["vec"], -1)
                        _add(right_vec, units[left_index]["vec"])
                        cost = (
                            base
                            - _fold_cost(counts[left], sizes[left], global_counts)
                            - _fold_cost(counts[right], sizes[right], global_counts)
                            + _fold_cost(left_vec, new_left, global_counts)
                            + _fold_cost(right_vec, new_right, global_counts)
                        )
                        candidate = (
                            cost,
                            "swap",
                            units[left_index]["key"],
                            units[right_index]["key"],
                        )
                        if best is None or candidate < best[0]:
                            best = (
                                candidate,
                                ("swap", left_index, right_index, left, right),
                            )
        for source in range(5):
            source_members = members[source]
            for dest in range(5):
                if source == dest:
                    continue
                for first_pos in range(len(source_members)):
                    first = source_members[first_pos]
                    for second_pos in range(first_pos + 1, len(source_members)):
                        second = source_members[second_pos]
                        leave = units[first]["n"] + units[second]["n"]
                        for third in members[dest]:
                            enter = units[third]["n"]
                            new_source = sizes[source] - leave + enter
                            new_dest = sizes[dest] - enter + leave
                            if new_source > 77 or new_dest > 77:
                                continue
                            source_vec = counts[source][:]
                            dest_vec = counts[dest][:]
                            _add(source_vec, units[first]["vec"], -1)
                            _add(source_vec, units[second]["vec"], -1)
                            _add(source_vec, units[third]["vec"])
                            _add(dest_vec, units[third]["vec"], -1)
                            _add(dest_vec, units[first]["vec"])
                            _add(dest_vec, units[second]["vec"])
                            cost = (
                                base
                                - _fold_cost(counts[source], sizes[source], global_counts)
                                - _fold_cost(counts[dest], sizes[dest], global_counts)
                                + _fold_cost(source_vec, new_source, global_counts)
                                + _fold_cost(dest_vec, new_dest, global_counts)
                            )
                            candidate = (
                                cost,
                                "2for1",
                                units[first]["key"],
                                units[second]["key"],
                                units[third]["key"],
                            )
                            if best is None or candidate < best[0]:
                                best = (
                                    candidate,
                                    ("x", first, second, third, source, dest),
                                )
        if best is None:
            break
        step = best[1]
        if step[0] == "move":
            _, index, fold, dest = step
            _add(counts[fold], units[index]["vec"], -1)
            sizes[fold] -= units[index]["n"]
            _add(counts[dest], units[index]["vec"])
            sizes[dest] += units[index]["n"]
            assign[index] = dest
        elif step[0] == "swap":
            _, left_index, right_index, left, right = step
            _add(counts[left], units[left_index]["vec"], -1)
            _add(counts[left], units[right_index]["vec"])
            _add(counts[right], units[right_index]["vec"], -1)
            _add(counts[right], units[left_index]["vec"])
            sizes[left] = sizes[left] - units[left_index]["n"] + units[right_index]["n"]
            sizes[right] = sizes[right] - units[right_index]["n"] + units[left_index]["n"]
            assign[left_index] = right
            assign[right_index] = left
        else:
            _, first, second, third, source, dest = step
            _add(counts[source], units[first]["vec"], -1)
            _add(counts[source], units[second]["vec"], -1)
            _add(counts[source], units[third]["vec"])
            _add(counts[dest], units[third]["vec"], -1)
            _add(counts[dest], units[first]["vec"])
            _add(counts[dest], units[second]["vec"])
            sizes[source] = (
                sizes[source]
                - units[first]["n"]
                - units[second]["n"]
                + units[third]["n"]
            )
            sizes[dest] = (
                sizes[dest]
                - units[third]["n"]
                + units[first]["n"]
                + units[second]["n"]
            )
            assign[first] = dest
            assign[second] = dest
            assign[third] = source
        steps += 1
    return assign


def assign_train_folds(rows: dict[str, dict]) -> tuple[dict[str, int], float, list[dict]]:
    """Return stage_a_id -> fold, objective, and leakage units."""
    units = build_units(rows)
    global_counts = [0] * 11
    for unit in units:
        for index, value in enumerate(unit["vec"]):
            global_counts[index] += value
    assign = _search(units, _greedy(units, global_counts), global_counts)
    mapping: dict[str, int] = {}
    for unit, fold in zip(units, assign, strict=True):
        for sid in unit["ids"]:
            mapping[sid] = int(fold)
    counts, sizes = _materialize(units, assign)
    objective = _objective(counts, sizes, global_counts)
    profiles = []
    for fold in range(5):
        profiles.append(tuple(counts[fold]) + (sizes[fold],))
    for fold, expected in enumerate(EXPECTED_FOLD_PROFILES):
        if tuple(counts[fold]) != expected or sizes[fold] != EXPECTED_FOLD_SIZES[fold]:
            raise RuntimeError(
                "frozen fold profile mismatch at "
                f"fold {fold}: got {tuple(counts[fold])} n={sizes[fold]}"
            )
    if round(objective, 2) != EXPECTED_OBJECTIVE:
        raise RuntimeError(f"objective {objective} != {EXPECTED_OBJECTIVE}")
    _ = profiles
    return mapping, objective, units


def cv_fold_fingerprint(mapping: dict[str, int]) -> str:
    """SHA256 of schema version plus stage_a_id, fold sorted by id."""
    lines = [SCHEMA_VERSION]
    for stage_a_id, fold in sorted(mapping.items()):
        lines.append(f"{stage_a_id}\t{int(fold)}")
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_manifest(root: Path | None = None) -> dict:
    root = ROOT if root is None else root
    if STAGE_A_V3_SPLIT_FINGERPRINT != SPLIT_FINGERPRINT:
        raise RuntimeError("split fingerprint constant drifted")
    if STAGE_A_V3_ANNOTATION_FINGERPRINT != ANNOTATION_FINGERPRINT:
        raise RuntimeError("annotation fingerprint constant drifted")
    split_ids = load_split_ids(root / STAGE_A_V3_SPLIT_PATH)
    train_ids = split_ids["train"]
    if len(train_ids) != STAGE_A_V3_TRAIN_SIZE:
        raise RuntimeError(f"TRAIN size {len(train_ids)} != {STAGE_A_V3_TRAIN_SIZE}")
    rows = load_train_rows(root / STAGE_A_V3_STEP_A_PATH, train_ids)
    mapping, objective, units = assign_train_folds(rows)
    _assert_assignment(mapping, units, split_ids, objective)
    fingerprint = cv_fold_fingerprint(mapping)
    return {
        "schema_version": SCHEMA_VERSION,
        "n_examples": 384,
        "n_folds": 5,
        "grouping_rule": GROUPING_RULE,
        "objective_definition": OBJECTIVE_DEFINITION,
        "objective_value": objective,
        "fold_sizes": list(EXPECTED_FOLD_SIZES),
        "structural_imbalance_notes": list(STRUCTURAL_IMBALANCE_NOTES),
        "split_fingerprint": SPLIT_FINGERPRINT,
        "annotation_fingerprint": ANNOTATION_FINGERPRINT,
        "cv_fold_fingerprint": fingerprint,
        "assignments": [
            {"fold": mapping[sid], "stage_a_id": sid}
            for sid in sorted(mapping)
        ],
    }


def _assert_assignment(
    mapping: dict[str, int],
    units: list[dict],
    split_ids: dict[str, set[str]],
    objective: float,
) -> None:
    if len(mapping) != 384 or set(mapping) != split_ids["train"]:
        raise RuntimeError("assignment is not exactly the 384 TRAIN ids")
    if set(mapping) & split_ids["dev"] or set(mapping) & split_ids["test"]:
        raise RuntimeError("DEV or TEST id entered the fold map")
    if set(mapping.values()) != {0, 1, 2, 3, 4}:
        raise RuntimeError(f"unexpected folds {sorted(set(mapping.values()))}")
    sizes = [sum(fold == index for fold in mapping.values()) for index in range(5)]
    if tuple(sizes) != EXPECTED_FOLD_SIZES:
        raise RuntimeError(f"fold sizes {sizes}")
    for unit in units:
        folds = {mapping[sid] for sid in unit["ids"]}
        if len(folds) != 1:
            raise RuntimeError(f"leakage component crosses folds: {unit['key']}")
    anchorless = [sid for unit in units for sid in unit["anchorless"]]
    if len(anchorless) != 6:
        raise RuntimeError(f"expected 6 anchorless TRAIN examples, got {len(anchorless)}")
    if len({mapping[sid] for sid in anchorless}) != 1:
        raise RuntimeError("anchorless examples were split across folds")
    if round(objective, 2) != EXPECTED_OBJECTIVE:
        raise RuntimeError(f"objective {objective}")


def manifest_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    return hashlib.sha256(encoded).hexdigest()


def write_manifest(path: Path | None = None) -> tuple[dict, str]:
    payload = build_manifest()
    path = MANIFEST_PATH if path is None else path
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.write_bytes(encoded)
    return payload, hashlib.sha256(encoded).hexdigest()


def main() -> None:
    payload, digest = write_manifest()
    print(f"CV_FOLD_FINGERPRINT {payload['cv_fold_fingerprint']}")
    print(f"MANIFEST_SHA256 {digest}")
    print(f"OBJECTIVE {payload['objective_value']}")
    print(f"FOLD_SIZES {payload['fold_sizes']}")


if __name__ == "__main__":
    main()
