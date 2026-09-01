"""Deterministic publication-oriented train/dev/test split for Stage-A v2 (480).

Hard holdout units merge ``hard_holdout_atoms`` (semantic_group plus authored
holdout-family links). Quarantined / publication-ineligible rows never enter
test. Reuses the v1 ``group_holdout_split`` packing search with v2 geometry.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

from tiergraph.enums import OperatorType, QueryType
from tiergraph.planner.annotation_step_a import (
    StageAStepAAnnotation,
    load_step_a_annotations,
)
from tiergraph.planner.annotation_step_b import (
    StageAStepBAnnotation,
    load_step_b_annotations,
)
from tiergraph.planner.annotations import ImplicitResolution, PlannerExample
from tiergraph.planner.stage_a_selection import load_jsonl
from tiergraph.planner.stage_a_split import (
    SPLITS,
    StageASplitResult,
    _assignment_fingerprint,
)
from tiergraph.planner.stage_a_to_corpus import (
    count_explicit_h7_edges,
    final_bucket_to_classification_label,
    step_ab_to_planner_example,
)
from tiergraph.planner.stage_a_v2_spec import (
    H7_SPLIT_FLOOR_DEV,
    H7_SPLIT_FLOOR_TEST,
    LEGAL_H7_FAMILY_LABELS,
    STAGE_A_V2_BUCKETS,
    STAGE_A_V2_CORPUS_SIZE,
    STAGE_A_V2_DEV_SIZE,
    STAGE_A_V2_SELECTION_PATH,
    STAGE_A_V2_SPLIT_PATH,
    STAGE_A_V2_SPLIT_REPORT_PATH,
    STAGE_A_V2_SPLIT_SEED,
    STAGE_A_V2_STEP_A_PATH,
    STAGE_A_V2_STEP_B_PATH,
    STAGE_A_V2_TEST_IDS_PATH,
    STAGE_A_V2_TEST_SIZE,
    STAGE_A_V2_TRAIN_SIZE,
    example_is_quarantined_for_publication_test,
    hard_holdout_atoms,
    is_authored_source,
    resolve_authored_holdout_family,
    resolve_authored_template_family,
)

H5_SPLIT_FLOOR_DEV: int = 5
H5_SPLIT_FLOOR_TEST: int = 5


@dataclass(frozen=True, slots=True)
class HoldoutComponent:
    """One atomic holdout unit for v2 (union of hard_holdout_atoms)."""

    component_id: str
    example_ids: tuple[str, ...]
    final_bucket: str
    atoms: frozenset[str]
    query_types: tuple[str, ...]
    h7_positive_count: int
    h5_positive_count: int
    test_assignable: bool

    @property
    def size(self) -> int:
        return len(self.example_ids)


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        if item not in self._parent:
            self._parent[item] = item

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self._parent[root] != root:
            self._parent[root] = self._parent[self._parent[root]]
            root = self._parent[root]
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root


def enrich_row_for_split(
    selection_row: Mapping[str, Any],
    *,
    step_a: StageAStepAAnnotation | None = None,
) -> dict[str, Any]:
    """Merge selection + Step-A provenance for holdout / quarantine metadata."""
    row = dict(selection_row)
    provenance: dict[str, Any] = {}
    if step_a is not None and step_a.provenance:
        provenance.update(step_a.provenance)
    if isinstance(row.get("provenance"), Mapping):
        provenance.update(row["provenance"])
    for key in (
        "authored_template_family",
        "authored_holdout_family",
        "scenario_family",
        "publication_test_eligible",
    ):
        if row.get(key) is None and key in provenance:
            row[key] = provenance[key]
    source_kind = str(row.get("source_kind") or "")
    if is_authored_source(source_kind) and resolve_authored_template_family(row) is None:
        template_group = str(row.get("template_group") or "").strip()
        if template_group:
            row["authored_template_family"] = template_group
    if row.get("publication_test_eligible") is None:
        row["publication_test_eligible"] = not example_is_quarantined_for_publication_test(
            row
        )
    return row


def _h5_positive(step_b: StageAStepBAnnotation) -> bool:
    return any(
        decision.implicit_resolution is ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
        for decision in step_b.anchor_decisions
    )


def _h7_families_for_record(
    step_b: StageAStepBAnnotation,
) -> tuple[str, ...]:
    labels: list[str] = []
    for dep in step_b.dependencies:
        try:
            source = OperatorType(step_b.operation_types[dep.source_operation_index])
            target = OperatorType(step_b.operation_types[dep.target_operation_index])
        except (IndexError, ValueError):
            continue
        label = f"{source.value}->{target.value}"
        if label in LEGAL_H7_FAMILY_LABELS:
            labels.append(label)
    return tuple(sorted(set(labels)))


def load_stage_a_v2_planner_examples(
    *,
    selection_path: str | Path = STAGE_A_V2_SELECTION_PATH,
    step_a_path: str | Path = STAGE_A_V2_STEP_A_PATH,
    step_b_path: str | Path = STAGE_A_V2_STEP_B_PATH,
) -> tuple[PlannerExample, ...]:
    selection = load_jsonl(selection_path)
    step_a_records = load_step_a_annotations(step_a_path)
    step_b_records = load_step_b_annotations(step_b_path)
    if len(selection) != STAGE_A_V2_CORPUS_SIZE:
        raise ValueError(f"selection count {len(selection)} != {STAGE_A_V2_CORPUS_SIZE}")
    if len(step_a_records) != STAGE_A_V2_CORPUS_SIZE:
        raise ValueError(f"Step-A count {len(step_a_records)} != {STAGE_A_V2_CORPUS_SIZE}")
    if len(step_b_records) != STAGE_A_V2_CORPUS_SIZE:
        raise ValueError(f"Step-B count {len(step_b_records)} != {STAGE_A_V2_CORPUS_SIZE}")

    sel_by_id = {str(row["stage_a_id"]): row for row in selection}
    step_b_by_id = {item.stage_a_id: item for item in step_b_records}
    examples: list[PlannerExample] = []
    for step_a in sorted(step_a_records, key=lambda item: item.stage_a_id):
        split_row = enrich_row_for_split(sel_by_id[step_a.stage_a_id], step_a=step_a)
        step_b = step_b_by_id[step_a.stage_a_id]
        example = step_ab_to_planner_example(step_a, step_b)
        metadata = dict(example.metadata)
        metadata.update(
            {
                "stage_a_id": step_a.stage_a_id,
                "final_bucket": step_a.final_bucket,
                "semantic_group": step_a.semantic_group,
                "template_group": step_a.template_group,
                "source_kind": split_row.get("source_kind"),
                "publication_test_eligible": bool(
                    split_row.get("publication_test_eligible")
                ),
                "h5_positive": _h5_positive(step_b),
                "h7_positive": bool(step_b.dependencies),
                "h7_families": list(_h7_families_for_record(step_b)),
                "authored_template_family": resolve_authored_template_family(split_row),
                "authored_holdout_family": resolve_authored_holdout_family(split_row),
            }
        )
        examples.append(example.model_copy(update={"metadata": metadata}))
    if len(examples) != STAGE_A_V2_CORPUS_SIZE:
        raise ValueError(f"expected {STAGE_A_V2_CORPUS_SIZE} PlannerExamples")
    return tuple(examples)


def build_holdout_components(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    examples_by_id: Mapping[str, PlannerExample],
) -> tuple[HoldoutComponent, ...]:
    uf = _UnionFind()
    atoms_by_example: dict[str, frozenset[str]] = {}
    for stage_a_id, row in rows_by_id.items():
        uf.add(stage_a_id)
        atoms_by_example[stage_a_id] = hard_holdout_atoms(row)
    atom_to_examples: dict[str, list[str]] = defaultdict(list)
    for stage_a_id, atoms in atoms_by_example.items():
        for atom in atoms:
            atom_to_examples[atom].append(stage_a_id)
    for example_ids in atom_to_examples.values():
        anchor = example_ids[0]
        for other in example_ids[1:]:
            uf.union(anchor, other)

    grouped: dict[str, list[str]] = defaultdict(list)
    for stage_a_id in rows_by_id:
        grouped[uf.find(stage_a_id)].append(stage_a_id)

    components: list[HoldoutComponent] = []
    for root, example_ids in grouped.items():
        ordered_ids = tuple(sorted(example_ids))
        buckets = {str(rows_by_id[sid]["final_bucket"]) for sid in ordered_ids}
        if len(buckets) != 1:
            bucket = "+".join(sorted(buckets))
        else:
            bucket = next(iter(buckets))
        atoms: set[str] = set()
        for sid in ordered_ids:
            atoms.update(atoms_by_example[sid])
        examples = [examples_by_id[sid] for sid in ordered_ids]
        test_assignable = all(
            bool(rows_by_id[sid].get("publication_test_eligible"))
            for sid in ordered_ids
        )
        components.append(
            HoldoutComponent(
                component_id=root,
                example_ids=ordered_ids,
                final_bucket=bucket,
                atoms=frozenset(atoms),
                query_types=tuple(ex.graph.query_type.value for ex in examples),
                h7_positive_count=sum(
                    1 for ex in examples if count_explicit_h7_edges(ex) > 0
                ),
                h5_positive_count=sum(
                    1 for ex in examples if bool(ex.metadata.get("h5_positive"))
                ),
                test_assignable=test_assignable,
            )
        )
    return tuple(sorted(components, key=lambda item: item.component_id))


def _bucket_quotas_v2(
    *,
    train_size: int = STAGE_A_V2_TRAIN_SIZE,
    dev_size: int = STAGE_A_V2_DEV_SIZE,
    test_size: int = STAGE_A_V2_TEST_SIZE,
    per_bucket: int = 96,
) -> tuple[dict[str, dict[str, int]], ...]:
    if train_size + dev_size + test_size != per_bucket * len(STAGE_A_V2_BUCKETS):
        raise ValueError("split sizes must cover the v2 corpus")

    local_options: list[tuple[int, int, int]] = []
    for train_n in range(76, 79):
        for dev_n in range(9, 12):
            test_n = per_bucket - train_n - dev_n
            if 9 <= test_n <= 11:
                local_options.append((train_n, dev_n, test_n))
    if not local_options:
        raise ValueError("no local bucket quota options for v2")

    quotas: list[dict[str, dict[str, int]]] = []

    def rec(
        index: int,
        rem_train: int,
        rem_dev: int,
        rem_test: int,
        acc: dict[str, dict[str, int]],
    ) -> None:
        if index == len(STAGE_A_V2_BUCKETS):
            if rem_train == rem_dev == rem_test == 0:
                quotas.append({bucket: dict(counts) for bucket, counts in acc.items()})
            return
        bucket = STAGE_A_V2_BUCKETS[index]
        buckets_left = len(STAGE_A_V2_BUCKETS) - index
        for train_n, dev_n, test_n in local_options:
            if train_n > rem_train or dev_n > rem_dev or test_n > rem_test:
                continue
            if rem_train - train_n > 78 * (buckets_left - 1):
                continue
            if rem_dev - dev_n > 11 * (buckets_left - 1):
                continue
            if rem_test - test_n > 11 * (buckets_left - 1):
                continue
            if rem_train - train_n < 76 * (buckets_left - 1):
                continue
            if rem_dev - dev_n < 9 * (buckets_left - 1):
                continue
            if rem_test - test_n < 9 * (buckets_left - 1):
                continue
            acc[bucket] = {"train": train_n, "dev": dev_n, "test": test_n}
            rec(
                index + 1,
                rem_train - train_n,
                rem_dev - dev_n,
                rem_test - test_n,
                acc,
            )
            del acc[bucket]

    rec(0, train_size, dev_size, test_size, {})
    if not quotas:
        raise ValueError("no feasible v2 bucket quotas")

    def quota_key(q: Mapping[str, Mapping[str, int]]) -> tuple:
        seq = q["MIXED_SEQUENTIAL"]
        seq_eval_ok = 0 if seq["dev"] >= 2 and seq["test"] >= 2 else 1
        seq_target = (
            abs(seq["train"] - 77) + abs(seq["dev"] - 10) + abs(seq["test"] - 9)
        )
        balance = sum(
            abs(counts["train"] - 77)
            + abs(counts["dev"] - 10)
            + abs(counts["test"] - 9)
            for counts in q.values()
        )
        return (
            seq_eval_ok,
            seq_target,
            balance,
            tuple(
                (bucket, q[bucket]["train"], q[bucket]["dev"], q[bucket]["test"])
                for bucket in STAGE_A_V2_BUCKETS
            ),
        )

    return tuple(sorted(quotas, key=quota_key))


def _component_bucket_counts(
    comp: HoldoutComponent,
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> Counter[str]:
    return Counter(str(rows_by_id[sid]["final_bucket"]) for sid in comp.example_ids)


def _is_multi_bucket_component(comp: HoldoutComponent) -> bool:
    return "+" in comp.final_bucket


def _pack_multi_bucket_components(
    components: Sequence[HoldoutComponent],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    quota: Mapping[str, Mapping[str, int]],
    *,
    rng: Random,
) -> dict[str, str] | None:
    """Assign cross-bucket holdout components under per-bucket quotas."""
    multi = [comp for comp in components if _is_multi_bucket_component(comp)]
    if not multi:
        return {}

    rem_global = {
        "train": STAGE_A_V2_TRAIN_SIZE,
        "dev": STAGE_A_V2_DEV_SIZE,
        "test": STAGE_A_V2_TEST_SIZE,
    }
    rem_bucket = {
        bucket: {
            "train": quota[bucket]["train"],
            "dev": quota[bucket]["dev"],
            "test": quota[bucket]["test"],
        }
        for bucket in STAGE_A_V2_BUCKETS
    }
    h7_counts = {split: 0 for split in SPLITS}
    h5_counts = {split: 0 for split in SPLITS}
    bucket_counts_by_comp = {
        comp.component_id: _component_bucket_counts(comp, rows_by_id)
        for comp in multi
    }

    ordered = list(multi)
    ordered.sort(
        key=lambda comp: (
            -comp.size,
            -comp.h7_positive_count,
            comp.component_id,
        )
    )
    by_size: dict[int, list[HoldoutComponent]] = defaultdict(list)
    for comp in ordered:
        by_size[comp.size].append(comp)
    ordered = []
    for size in sorted(by_size, reverse=True):
        group = by_size[size]
        rng.shuffle(group)
        ordered.extend(group)

    assignment: dict[str, str] = {}

    def can_place(comp: HoldoutComponent, split: str) -> bool:
        if split == "test" and not comp.test_assignable:
            return False
        if comp.size > rem_global[split]:
            return False
        for bucket, count in bucket_counts_by_comp[comp.component_id].items():
            if count > rem_bucket[bucket][split]:
                return False
        return True

    def place(comp: HoldoutComponent, split: str) -> None:
        rem_global[split] -= comp.size
        for bucket, count in bucket_counts_by_comp[comp.component_id].items():
            rem_bucket[bucket][split] -= count
        h7_counts[split] += comp.h7_positive_count
        h5_counts[split] += comp.h5_positive_count
        for example_id in comp.example_ids:
            assignment[example_id] = split

    def unplace(comp: HoldoutComponent, split: str) -> None:
        rem_global[split] += comp.size
        for bucket, count in bucket_counts_by_comp[comp.component_id].items():
            rem_bucket[bucket][split] += count
        h7_counts[split] -= comp.h7_positive_count
        h5_counts[split] -= comp.h5_positive_count
        for example_id in comp.example_ids:
            del assignment[example_id]

    def rec(index: int) -> bool:
        if index == len(ordered):
            return True
        comp = ordered[index]
        split_order = sorted(
            SPLITS,
            key=lambda split: (
                0 if can_place(comp, split) else 1,
                -rem_global[split],
                rng.random(),
            ),
        )
        for split in split_order:
            if not can_place(comp, split):
                continue
            place(comp, split)
            if rec(index + 1):
                return True
            unplace(comp, split)
        return False

    if not rec(0):
        return None
    return assignment


def _remaining_bucket_quota(
    quota: Mapping[str, Mapping[str, int]],
    assignment: Mapping[str, str],
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    used = {
        bucket: {"train": 0, "dev": 0, "test": 0}
        for bucket in STAGE_A_V2_BUCKETS
    }
    for example_id, split in assignment.items():
        bucket = str(rows_by_id[example_id]["final_bucket"])
        used[bucket][split] += 1
    return {
        bucket: {
            split: quota[bucket][split] - used[bucket][split]
            for split in SPLITS
        }
        for bucket in STAGE_A_V2_BUCKETS
    }


def _pack_split_hybrid(
    components: Sequence[HoldoutComponent],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    quota: Mapping[str, Mapping[str, int]],
    *,
    rng: Random,
) -> dict[str, str] | None:
    """Pack cross-bucket components globally, then per-bucket singles."""
    multi_assignment = _pack_multi_bucket_components(
        components, rows_by_id, quota, rng=rng
    )
    if multi_assignment is None:
        return None

    single_by_bucket: dict[str, list[HoldoutComponent]] = defaultdict(list)
    for comp in components:
        if not _is_multi_bucket_component(comp):
            single_by_bucket[comp.final_bucket].append(comp)

    rem_quota = _remaining_bucket_quota(quota, multi_assignment, rows_by_id)
    assignment = dict(multi_assignment)

    for bucket in STAGE_A_V2_BUCKETS:
        comps = single_by_bucket[bucket]
        expected = sum(rem_quota[bucket][split] for split in SPLITS)
        if sum(comp.size for comp in comps) != expected:
            return None
        min_h7 = None
        if bucket == "MIXED_SEQUENTIAL":
            min_h7 = {
                "train": 1,
                "dev": 2 if rem_quota[bucket]["dev"] >= 2 else 1,
                "test": 2 if rem_quota[bucket]["test"] >= 2 else 1,
            }
        packed = _pack_bucket_components_v2(
            comps,
            train_n=rem_quota[bucket]["train"],
            dev_n=rem_quota[bucket]["dev"],
            test_n=rem_quota[bucket]["test"],
            min_h7_per_split=min_h7,
            rng=rng,
        )
        if packed is None:
            return None
        for split, example_ids in packed.items():
            for example_id in example_ids:
                if example_id in assignment:
                    return None
                assignment[example_id] = split

    if len(assignment) != STAGE_A_V2_CORPUS_SIZE:
        return None
    return assignment


def _pack_bucket_components_v2(
    components: Sequence[HoldoutComponent],
    *,
    train_n: int,
    dev_n: int,
    test_n: int,
    min_h7_per_split: Mapping[str, int] | None,
    rng: Random,
) -> dict[str, tuple[str, ...]] | None:
    """Pack one bucket's holdout components with publication-test eligibility."""
    total = sum(comp.size for comp in components)
    if total != train_n + dev_n + test_n:
        return None

    by_size: dict[int, list[HoldoutComponent]] = defaultdict(list)
    for comp in components:
        by_size[comp.size].append(comp)
    ordered: list[HoldoutComponent] = []
    for size in sorted(by_size, reverse=True):
        group = sorted(
            by_size[size],
            key=lambda comp: (-comp.h7_positive_count, comp.component_id),
        )
        rng.shuffle(group)
        ordered.extend(group)

    targets = {"train": train_n, "dev": dev_n, "test": test_n}
    min_h7 = {
        split: int(min_h7_per_split.get(split, 0)) if min_h7_per_split else 0
        for split in SPLITS
    }
    comp_split: dict[str, str] = {}

    def h7_ok() -> bool:
        if min_h7_per_split is None:
            return True
        remaining_h7 = sum(
            comp.h7_positive_count
            for comp in ordered
            if comp.component_id not in comp_split
        )
        deficit = 0
        for split in SPLITS:
            need = min_h7[split]
            if need <= 0 or targets[split] <= 0:
                continue
            have = sum(
                comp.h7_positive_count
                for comp in ordered
                if comp_split.get(comp.component_id) == split
            )
            if have < need:
                deficit += need - have
        return deficit <= remaining_h7

    def finish_ok() -> bool:
        if min_h7_per_split is None:
            return True
        for split in SPLITS:
            need = min_h7[split]
            if need <= 0 or targets[split] <= 0:
                continue
            have = sum(
                comp.h7_positive_count
                for comp in ordered
                if comp_split.get(comp.component_id) == split
            )
            if have < need:
                return False
        return True

    def rec(index: int, rem: dict[str, int]) -> bool:
        if index == len(ordered):
            return rem["train"] == rem["dev"] == rem["test"] == 0 and finish_ok()
        comp = ordered[index]
        split_order = sorted(
            SPLITS,
            key=lambda split: (
                0 if rem[split] >= comp.size else 1,
                1 if split == "test" and not comp.test_assignable else 0,
                -rem[split],
                rng.random(),
            ),
        )
        for split in split_order:
            if rem[split] < comp.size:
                continue
            if split == "test" and not comp.test_assignable:
                continue
            comp_split[comp.component_id] = split
            rem[split] -= comp.size
            if h7_ok() and rec(index + 1, rem):
                return True
            rem[split] += comp.size
            del comp_split[comp.component_id]
        return False

    if not rec(0, dict(targets)):
        return None
    out: dict[str, list[str]] = {split: [] for split in SPLITS}
    for comp in ordered:
        out[comp_split[comp.component_id]].extend(comp.example_ids)
    return {split: tuple(sorted(ids)) for split, ids in out.items()}


def _score_assignment_v2(
    examples_by_id: Mapping[str, PlannerExample],
    assignment: Mapping[str, str],
    *,
    train_size: int,
    dev_size: int,
    test_size: int,
) -> tuple[int, ...]:
    by_split: dict[str, list[PlannerExample]] = {split: [] for split in SPLITS}
    for example_id, split in assignment.items():
        by_split[split].append(examples_by_id[example_id])
    sizes = {split: len(items) for split, items in by_split.items()}
    if sizes != {"train": train_size, "dev": dev_size, "test": test_size}:
        return (10**9,)

    h7 = {
        split: sum(1 for ex in items if count_explicit_h7_edges(ex) > 0)
        for split, items in by_split.items()
    }
    h5 = {
        split: sum(1 for ex in items if bool(ex.metadata.get("h5_positive")))
        for split, items in by_split.items()
    }
    if h7["dev"] < H7_SPLIT_FLOOR_DEV or h7["test"] < H7_SPLIT_FLOOR_TEST:
        return (10**9 - 1,)
    if h5["dev"] < H5_SPLIT_FLOOR_DEV or h5["test"] < H5_SPLIT_FLOOR_TEST:
        return (10**9 - 2,)
    if h7["train"] < (sum(h7.values()) // 2 + sum(h7.values()) % 2):
        return (10**9 - 3,)

    bucket_penalty = 0
    for split, items in by_split.items():
        counts = Counter(str(ex.metadata["final_bucket"]) for ex in items)
        expected = sizes[split] / len(STAGE_A_V2_BUCKETS)
        bucket_penalty += sum(
            abs(counts[bucket] - expected) for bucket in STAGE_A_V2_BUCKETS
        )

    h1_penalty = 0
    for split, items in by_split.items():
        counts = Counter(ex.graph.query_type.value for ex in items)
        h1_penalty += abs(counts.get(QueryType.PERSONAL.value, 0) - 0.2 * sizes[split])
        h1_penalty += abs(
            counts.get(QueryType.ENVIRONMENTAL.value, 0) - 0.2 * sizes[split]
        )
        h1_penalty += abs(counts.get(QueryType.MIXED.value, 0) - 0.6 * sizes[split])

    h7_target = abs(h7["train"] - 63) + abs(h7["dev"] - 8) + abs(h7["test"] - 8)
    return (
        int(bucket_penalty * 100),
        int(h1_penalty * 100),
        h7_target,
        tuple(sorted(assignment.items())),
    )


def _atom_leakage_for_prefix(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    assignment: Mapping[str, str],
    *,
    prefix: str,
) -> dict[str, list[str]]:
    atom_to_splits: dict[str, set[str]] = defaultdict(set)
    for stage_a_id, split in assignment.items():
        row = rows_by_id[stage_a_id]
        for atom in hard_holdout_atoms(row):
            if atom.startswith(prefix):
                atom_to_splits[atom].add(split)
    return {
        atom: sorted(splits)
        for atom, splits in sorted(atom_to_splits.items())
        if len(splits) > 1
    }


def _atom_leakage(
    rows_by_id: Mapping[str, Mapping[str, Any]],
    assignment: Mapping[str, str],
) -> dict[str, list[str]]:
    atom_to_splits: dict[str, set[str]] = defaultdict(set)
    for stage_a_id, split in assignment.items():
        row = rows_by_id[stage_a_id]
        for atom in hard_holdout_atoms(row):
            atom_to_splits[atom].add(split)
    return {
        atom: sorted(splits)
        for atom, splits in sorted(atom_to_splits.items())
        if len(splits) > 1
    }


def _build_v2_report(
    *,
    examples_by_id: Mapping[str, PlannerExample],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    assignment: Mapping[str, str],
    components: Sequence[HoldoutComponent],
    seed: int,
    fingerprint: str,
) -> dict[str, Any]:
    by_split: dict[str, list[PlannerExample]] = {split: [] for split in SPLITS}
    for example_id, split in assignment.items():
        by_split[split].append(examples_by_id[example_id])

    def split_stats(items: Sequence[PlannerExample]) -> dict[str, Any]:
        h7_families: Counter[str] = Counter()
        for ex in items:
            for label in ex.metadata.get("h7_families") or []:
                h7_families[str(label)] += 1
        source_kind = Counter(
            str(ex.metadata.get("source_kind") or "unknown") for ex in items
        )
        return {
            "n": len(items),
            "final_bucket": dict(
                sorted(Counter(str(ex.metadata["final_bucket"]) for ex in items).items())
            ),
            "h1_classification_label": dict(
                sorted(
                    Counter(
                        final_bucket_to_classification_label(
                            str(ex.metadata["final_bucket"])
                        )
                        for ex in items
                    ).items()
                )
            ),
            "decoded_graph_query_type": dict(
                sorted(Counter(ex.graph.query_type.value for ex in items).items())
            ),
            "h5_positive_rows": sum(
                1 for ex in items if bool(ex.metadata.get("h5_positive"))
            ),
            "h7_positive": sum(
                1 for ex in items if count_explicit_h7_edges(ex) > 0
            ),
            "h7_positive_rows": sum(
                1 for ex in items if count_explicit_h7_edges(ex) > 0
            ),
            "h7_family_counts": dict(sorted(h7_families.items())),
            "source_kind": dict(sorted(source_kind.items())),
            "semantic_groups": len({str(ex.metadata["semantic_group"]) for ex in items}),
            "template_groups": len({str(ex.metadata["template_group"]) for ex in items}),
            "example_ids": [ex.example_id for ex in sorted(items, key=lambda e: e.example_id)],
        }

    semantic_leakage = _atom_leakage_for_prefix(
        rows_by_id, assignment, prefix="semantic:"
    )
    authored_leakage = _atom_leakage_for_prefix(
        rows_by_id, assignment, prefix="authored_holdout_family:"
    )

    template_to_splits: dict[str, set[str]] = defaultdict(set)
    for example_id, split in assignment.items():
        template_to_splits[str(rows_by_id[example_id]["template_group"])].add(split)
    template_overlap = [
        {"template_group": template, "splits": sorted(splits)}
        for template, splits in sorted(template_to_splits.items())
        if len(splits) > 1
    ]

    test_ids = tuple(
        sorted(example_id for example_id, split in assignment.items() if split == "test")
    )
    quarantined_in_test = [
        example_id
        for example_id in test_ids
        if not bool(rows_by_id[example_id].get("publication_test_eligible"))
    ]

    all_examples = list(examples_by_id.values())
    h1_classification_total = dict(
        sorted(
            Counter(
                final_bucket_to_classification_label(str(ex.metadata["final_bucket"]))
                for ex in all_examples
            ).items()
        )
    )
    decoded_graph_query_type_total = dict(
        sorted(Counter(ex.graph.query_type.value for ex in all_examples).items())
    )

    return {
        "seed": seed,
        "fingerprint": fingerprint,
        "sizes": {split: len(by_split[split]) for split in SPLITS},
        "h1_classification_label_total": h1_classification_total,
        "decoded_graph_query_type_total": decoded_graph_query_type_total,
        "h1_reporting_note": (
            "h1_classification_label uses final_bucket_to_classification_label "
            "(canonical Stage-A H1). decoded_graph_query_type counts "
            "PlannerExample.graph.query_type from GraphDecoder answer nodes."
        ),
        "by_split": {split: split_stats(by_split[split]) for split in SPLITS},
        "semantic_group_leakage": semantic_leakage,
        "authored_holdout_family_leakage": authored_leakage,
        "template_group_overlap": template_overlap,
        "template_group_overlap_count": len(template_overlap),
        "n_holdout_components": len(components),
        "n_h7_positive_total": sum(
            1 for ex in examples_by_id.values() if count_explicit_h7_edges(ex) > 0
        ),
        "n_h5_positive_total": sum(
            1 for ex in examples_by_id.values() if bool(ex.metadata.get("h5_positive"))
        ),
        "test_ids": list(test_ids),
        "quarantined_in_test": quarantined_in_test,
        "example_to_split": dict(sorted(assignment.items())),
    }


def validate_split_result_v2(
    result: StageASplitResult,
    *,
    rows_by_id: Mapping[str, Mapping[str, Any]],
    train_size: int = STAGE_A_V2_TRAIN_SIZE,
    dev_size: int = STAGE_A_V2_DEV_SIZE,
    test_size: int = STAGE_A_V2_TEST_SIZE,
) -> list[str]:
    errors: list[str] = []
    report = result.report
    sizes = report["sizes"]
    if sizes != {"train": train_size, "dev": dev_size, "test": test_size}:
        errors.append(f"unexpected sizes: {sizes}")
    if (
        len(result.train) != train_size
        or len(result.dev) != dev_size
        or len(result.test) != test_size
    ):
        errors.append("result tuple lengths mismatch")
    if report.get("semantic_group_leakage"):
        errors.append(f"semantic_group leakage: {report['semantic_group_leakage']}")
    if report.get("authored_holdout_family_leakage"):
        errors.append(
            "authored holdout leakage: "
            f"{report['authored_holdout_family_leakage']}"
        )
    if report.get("quarantined_in_test"):
        errors.append(f"quarantined rows in test: {report['quarantined_in_test']}")
    all_ids = [ex.example_id for ex in (*result.train, *result.dev, *result.test)]
    if len(all_ids) != len(set(all_ids)):
        errors.append("duplicate example_id across splits")
    by_split = report["by_split"]
    assert isinstance(by_split, dict)
    if int(by_split["dev"]["h7_positive_rows"]) < H7_SPLIT_FLOOR_DEV:
        errors.append(
            f"dev H7 floor failed: {by_split['dev']['h7_positive_rows']} "
            f"< {H7_SPLIT_FLOOR_DEV}"
        )
    if int(by_split["test"]["h7_positive_rows"]) < H7_SPLIT_FLOOR_TEST:
        errors.append(
            f"test H7 floor failed: {by_split['test']['h7_positive_rows']} "
            f"< {H7_SPLIT_FLOOR_TEST}"
        )
    if int(by_split["dev"]["h5_positive_rows"]) < H5_SPLIT_FLOOR_DEV:
        errors.append(
            f"dev H5 floor failed: {by_split['dev']['h5_positive_rows']} "
            f"< {H5_SPLIT_FLOOR_DEV}"
        )
    if int(by_split["test"]["h5_positive_rows"]) < H5_SPLIT_FLOOR_TEST:
        errors.append(
            f"test H5 floor failed: {by_split['test']['h5_positive_rows']} "
            f"< {H5_SPLIT_FLOOR_TEST}"
        )
    test_ids = report.get("test_ids")
    if isinstance(test_ids, list):
        for example_id in test_ids:
            if not bool(rows_by_id[example_id].get("publication_test_eligible")):
                errors.append(f"{example_id} is not publication-test eligible")
    return errors


def group_holdout_split_v2(
    examples: Sequence[PlannerExample],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    *,
    train_size: int = STAGE_A_V2_TRAIN_SIZE,
    dev_size: int = STAGE_A_V2_DEV_SIZE,
    test_size: int = STAGE_A_V2_TEST_SIZE,
    seed: int = STAGE_A_V2_SPLIT_SEED,
    max_quota_candidates: int = 128,
    trials_per_quota: int = 48,
) -> StageASplitResult:
    if train_size + dev_size + test_size != len(examples):
        raise ValueError("split sizes must equal number of examples")
    examples_by_id = {ex.example_id: ex for ex in examples}
    if len(examples_by_id) != len(examples):
        raise ValueError("duplicate example_id in split input")

    components = build_holdout_components(rows_by_id, examples_by_id)

    quota_candidates = _bucket_quotas_v2(
        train_size=train_size, dev_size=dev_size, test_size=test_size
    )
    rng_root = Random(seed)
    ranked_quotas = list(quota_candidates[:max_quota_candidates])
    if len(quota_candidates) > max_quota_candidates:
        rest = list(quota_candidates[max_quota_candidates:])
        rng_root.shuffle(rest)
        ranked_quotas.extend(rest[: max(0, max_quota_candidates // 2)])

    best: tuple[tuple[int, ...], dict[str, str]] | None = None
    trial_counter = 0
    for quota_index, quota in enumerate(ranked_quotas):
        for trial in range(trials_per_quota):
            trial_counter += 1
            trial_seed = seed + 1009 * quota_index + 17 * trial
            rng = Random(trial_seed)
            assignment = _pack_split_hybrid(
                components,
                rows_by_id,
                quota,
                rng=rng,
            )
            if assignment is None or len(assignment) != len(examples_by_id):
                continue
            score = _score_assignment_v2(
                examples_by_id,
                assignment,
                train_size=train_size,
                dev_size=dev_size,
                test_size=test_size,
            )
            if score[0] >= 10**9 - 3:
                continue
            if best is None or score < best[0]:
                best = (score, dict(assignment))

    if best is None:
        raise RuntimeError(
            "failed to find a valid v2 holdout split; "
            f"tried {trial_counter} quota/trial combinations with seed={seed}"
        )

    assignment = best[1]
    fingerprint = _assignment_fingerprint(assignment, seed)
    report = _build_v2_report(
        examples_by_id=examples_by_id,
        rows_by_id=rows_by_id,
        assignment=assignment,
        components=components,
        seed=seed,
        fingerprint=fingerprint,
    )
    result = StageASplitResult(
        train=tuple(
            examples_by_id[example_id]
            for example_id in sorted(
                eid for eid, split in assignment.items() if split == "train"
            )
        ),
        dev=tuple(
            examples_by_id[example_id]
            for example_id in sorted(
                eid for eid, split in assignment.items() if split == "dev"
            )
        ),
        test=tuple(
            examples_by_id[example_id]
            for example_id in sorted(
                eid for eid, split in assignment.items() if split == "test"
            )
        ),
        seed=seed,
        fingerprint=fingerprint,
        report=report,
    )
    errors = validate_split_result_v2(result, rows_by_id=rows_by_id)
    if errors:
        raise RuntimeError("v2 split validation failed: " + "; ".join(errors))
    return result


def load_stage_a_v2_assignment(
    split_path: str | Path = STAGE_A_V2_SPLIT_PATH,
) -> dict[str, str]:
    """Load a frozen stage_a_id -> split assignment from JSONL."""
    assignment: dict[str, str] = {}
    for row in load_jsonl(split_path):
        assignment[str(row["stage_a_id"])] = str(row["split"])
    if len(assignment) != STAGE_A_V2_CORPUS_SIZE:
        raise ValueError(
            f"assignment count {len(assignment)} != {STAGE_A_V2_CORPUS_SIZE}"
        )
    return assignment


def regenerate_stage_a_v2_split_report(
    *,
    assignment: Mapping[str, str] | None = None,
    split_path: str | Path = STAGE_A_V2_SPLIT_PATH,
    selection_path: str | Path = STAGE_A_V2_SELECTION_PATH,
    step_a_path: str | Path = STAGE_A_V2_STEP_A_PATH,
    step_b_path: str | Path = STAGE_A_V2_STEP_B_PATH,
    seed: int = STAGE_A_V2_SPLIT_SEED,
) -> StageASplitResult:
    """Rebuild split report/result from frozen assignments without re-packing."""
    if assignment is None:
        assignment = load_stage_a_v2_assignment(split_path)
    if len(assignment) != STAGE_A_V2_CORPUS_SIZE:
        raise ValueError(
            f"assignment count {len(assignment)} != {STAGE_A_V2_CORPUS_SIZE}"
        )

    selection = load_jsonl(selection_path)
    step_a_by_id = {
        item.stage_a_id: item for item in load_step_a_annotations(step_a_path)
    }
    rows_by_id = {
        str(row["stage_a_id"]): enrich_row_for_split(
            row, step_a=step_a_by_id.get(str(row["stage_a_id"]))
        )
        for row in selection
    }
    examples = load_stage_a_v2_planner_examples(
        selection_path=selection_path,
        step_a_path=step_a_path,
        step_b_path=step_b_path,
    )
    examples_by_id = {ex.example_id: ex for ex in examples}
    if set(assignment) != set(examples_by_id):
        raise ValueError("assignment example ids do not match corpus")

    components = build_holdout_components(rows_by_id, examples_by_id)
    fingerprint = _assignment_fingerprint(dict(assignment), seed)
    report = _build_v2_report(
        examples_by_id=examples_by_id,
        rows_by_id=rows_by_id,
        assignment=assignment,
        components=components,
        seed=seed,
        fingerprint=fingerprint,
    )
    result = StageASplitResult(
        train=tuple(
            examples_by_id[example_id]
            for example_id in sorted(
                eid for eid, split in assignment.items() if split == "train"
            )
        ),
        dev=tuple(
            examples_by_id[example_id]
            for example_id in sorted(
                eid for eid, split in assignment.items() if split == "dev"
            )
        ),
        test=tuple(
            examples_by_id[example_id]
            for example_id in sorted(
                eid for eid, split in assignment.items() if split == "test"
            )
        ),
        seed=seed,
        fingerprint=fingerprint,
        report=report,
    )
    errors = validate_split_result_v2(result, rows_by_id=rows_by_id)
    if errors:
        raise RuntimeError("v2 split validation failed: " + "; ".join(errors))
    return result


def write_stage_a_v2_split_report(
    *,
    report_path: str | Path = STAGE_A_V2_SPLIT_REPORT_PATH,
    test_ids_path: str | Path = STAGE_A_V2_TEST_IDS_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    """Rewrite report/test-id artifacts from frozen split assignments."""
    result = regenerate_stage_a_v2_split_report(**kwargs)
    Path(report_path).write_text(
        json.dumps(result.report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(test_ids_path).write_text(
        json.dumps(
            {
                "seed": result.seed,
                "fingerprint": result.fingerprint,
                "test_ids": result.report["test_ids"],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result.report


def build_stage_a_v2_split(
    *,
    selection_path: str | Path = STAGE_A_V2_SELECTION_PATH,
    step_a_path: str | Path = STAGE_A_V2_STEP_A_PATH,
    step_b_path: str | Path = STAGE_A_V2_STEP_B_PATH,
    seed: int = STAGE_A_V2_SPLIT_SEED,
) -> tuple[StageASplitResult, dict[str, Mapping[str, Any]]]:
    selection = load_jsonl(selection_path)
    step_a_by_id = {
        item.stage_a_id: item for item in load_step_a_annotations(step_a_path)
    }
    rows_by_id = {
        str(row["stage_a_id"]): enrich_row_for_split(
            row, step_a=step_a_by_id.get(str(row["stage_a_id"]))
        )
        for row in selection
    }
    examples = load_stage_a_v2_planner_examples(
        selection_path=selection_path,
        step_a_path=step_a_path,
        step_b_path=step_b_path,
    )
    result = group_holdout_split_v2(examples, rows_by_id, seed=seed)
    return result, rows_by_id


def write_stage_a_v2_split(
    *,
    output_path: str | Path = STAGE_A_V2_SPLIT_PATH,
    report_path: str | Path = STAGE_A_V2_SPLIT_REPORT_PATH,
    test_ids_path: str | Path = STAGE_A_V2_TEST_IDS_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    result, _rows_by_id = build_stage_a_v2_split(**kwargs)
    lines = [
        json.dumps(
            {
                "stage_a_id": example_id,
                "split": split,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for example_id, split in sorted(result.report["example_to_split"].items())
    ]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(report_path).write_text(
        json.dumps(result.report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(test_ids_path).write_text(
        json.dumps(
            {
                "seed": result.seed,
                "fingerprint": result.fingerprint,
                "test_ids": result.report["test_ids"],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result.report


def main() -> None:
    report = write_stage_a_v2_split()
    print(
        json.dumps(
            {
                "sizes": report["sizes"],
                "fingerprint": report["fingerprint"],
                "seed": report["seed"],
                "h7_positive": {
                    split: report["by_split"][split]["h7_positive_rows"]
                    for split in SPLITS
                },
                "h5_positive": {
                    split: report["by_split"][split]["h5_positive_rows"]
                    for split in SPLITS
                },
                "semantic_group_leakage": report["semantic_group_leakage"],
                "authored_holdout_family_leakage": report["authored_holdout_family_leakage"],
                "quarantined_in_test": report["quarantined_in_test"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
