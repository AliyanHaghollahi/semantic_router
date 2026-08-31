"""Deterministic semantic_group holdout splits for Stage-A planner gold.

``semantic_group`` is the hard leakage barrier: every example in a group is
assigned to exactly one of train / dev / test. ``template_group`` overlap is
reported but not forbidden.

The default search builds exact 96/12/12 assignments while balancing
``final_bucket`` quotas and placing H7-positive examples in every split.
When feasible it prefers ~11/2/2 H7 positives (train/dev/test), with at
least 2 in each of dev and test and a clear train majority of the 15.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random

from tiergraph.enums import QueryType
from tiergraph.planner.annotations import PlannerExample
from tiergraph.planner.stage_a_to_corpus import count_explicit_h7_edges


DEFAULT_SPLIT_SEED = 20260831
DEFAULT_TRAIN_SIZE = 96
DEFAULT_DEV_SIZE = 12
DEFAULT_TEST_SIZE = 12
FINAL_BUCKETS: tuple[str, ...] = (
    "Personal",
    "Environmental",
    "MIXED_IMPLICIT",
    "MIXED_PARALLEL",
    "MIXED_SEQUENTIAL",
)
SPLITS: tuple[str, ...] = ("train", "dev", "test")


@dataclass(frozen=True, slots=True)
class SemanticGroupComponent:
    """One hard holdout unit: all examples sharing a ``semantic_group``."""

    semantic_group: str
    example_ids: tuple[str, ...]
    final_bucket: str
    query_types: tuple[str, ...]
    h7_positive_count: int

    @property
    def size(self) -> int:
        return len(self.example_ids)


@dataclass(frozen=True, slots=True)
class StageASplitResult:
    """Deterministic train/dev/test assignment over Stage-A planner examples."""

    train: tuple[PlannerExample, ...]
    dev: tuple[PlannerExample, ...]
    test: tuple[PlannerExample, ...]
    seed: int
    fingerprint: str
    report: dict[str, object]

    def split_for(self, example_id: str) -> str:
        mapping = self.report["example_to_split"]
        assert isinstance(mapping, dict)
        return str(mapping[example_id])


def _require_metadata(example: PlannerExample) -> tuple[str, str, str]:
    meta = example.metadata
    try:
        stage_a_id = str(meta["stage_a_id"])
        semantic_group = str(meta["semantic_group"])
        final_bucket = str(meta["final_bucket"])
    except KeyError as exc:
        raise ValueError(
            f"PlannerExample {example.example_id!r} missing split metadata: {exc}"
        ) from exc
    if not stage_a_id.strip() or not semantic_group.strip() or not final_bucket.strip():
        raise ValueError(
            f"PlannerExample {example.example_id!r} has blank split metadata"
        )
    if final_bucket not in FINAL_BUCKETS:
        raise ValueError(
            f"PlannerExample {example.example_id!r} has unknown final_bucket "
            f"{final_bucket!r}"
        )
    return stage_a_id, semantic_group, final_bucket


def build_semantic_group_components(
    examples: Sequence[PlannerExample],
) -> tuple[SemanticGroupComponent, ...]:
    """Group examples by ``semantic_group`` (order-invariant)."""
    by_group: dict[str, list[PlannerExample]] = defaultdict(list)
    for example in examples:
        _stage_a_id, semantic_group, _final_bucket = _require_metadata(example)
        by_group[semantic_group].append(example)

    components: list[SemanticGroupComponent] = []
    for semantic_group in sorted(by_group):
        items = sorted(by_group[semantic_group], key=lambda ex: ex.example_id)
        buckets = {str(ex.metadata["final_bucket"]) for ex in items}
        if len(buckets) != 1:
            raise ValueError(
                f"semantic_group {semantic_group!r} spans multiple final_buckets: "
                f"{sorted(buckets)}"
            )
        components.append(
            SemanticGroupComponent(
                semantic_group=semantic_group,
                example_ids=tuple(ex.example_id for ex in items),
                final_bucket=next(iter(buckets)),
                query_types=tuple(ex.graph.query_type.value for ex in items),
                h7_positive_count=sum(
                    1 for ex in items if count_explicit_h7_edges(ex) > 0
                ),
            )
        )
    return tuple(components)


def _bucket_quotas(
    *,
    train_size: int,
    dev_size: int,
    test_size: int,
    per_bucket: int = 24,
) -> tuple[dict[str, dict[str, int]], ...]:
    """Enumerate compact near-ideal per-bucket train/dev/test quotas.

    Restricts each bucket to train∈{18..21}, dev∈{1..4}, test∈{1..4} with
    train+dev+test=24, then keeps combinations that hit the global sizes.
    Prefers MIXED_SEQUENTIAL quotas with ≥2 slots in both dev and test so
    H7 can land ~11/2/2 instead of a fragile 1/1 eval holdout.
    """
    if train_size + dev_size + test_size != per_bucket * len(FINAL_BUCKETS):
        raise ValueError("split sizes must cover the full Stage-A corpus")

    local_options: list[tuple[int, int, int]] = []
    for train_n in range(18, 22):
        for dev_n in range(1, 5):
            test_n = per_bucket - train_n - dev_n
            if 1 <= test_n <= 4:
                local_options.append((train_n, dev_n, test_n))
    if not local_options:
        raise ValueError("no local bucket quota options")

    quotas: list[dict[str, dict[str, int]]] = []

    def rec(index: int, rem_train: int, rem_dev: int, rem_test: int, acc: dict) -> None:
        if index == len(FINAL_BUCKETS):
            if rem_train == rem_dev == rem_test == 0:
                quotas.append(
                    {bucket: dict(counts) for bucket, counts in acc.items()}
                )
            return
        bucket = FINAL_BUCKETS[index]
        buckets_left = len(FINAL_BUCKETS) - index
        for train_n, dev_n, test_n in local_options:
            if train_n > rem_train or dev_n > rem_dev or test_n > rem_test:
                continue
            if rem_train - train_n > 21 * (buckets_left - 1):
                continue
            if rem_dev - dev_n > 4 * (buckets_left - 1):
                continue
            if rem_test - test_n > 4 * (buckets_left - 1):
                continue
            if rem_train - train_n < 18 * (buckets_left - 1):
                continue
            if rem_dev - dev_n < 1 * (buckets_left - 1):
                continue
            if rem_test - test_n < 1 * (buckets_left - 1):
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
        raise ValueError("no feasible near-ideal bucket quotas")

    def quota_key(q: Mapping[str, Mapping[str, int]]) -> tuple:
        seq = q["MIXED_SEQUENTIAL"]
        # Prefer sequential slots that can host ≥2 H7 in both eval splits.
        seq_eval_ok = 0 if seq["dev"] >= 2 and seq["test"] >= 2 else 1
        # 20/2/2 leaves room for exact H7 11/2/2 (all 9 non-H7 sequential in train).
        seq_target = (
            abs(seq["train"] - 20) + abs(seq["dev"] - 2) + abs(seq["test"] - 2)
        )
        balance = sum(
            abs(counts["train"] - 19)
            + abs(counts["dev"] - 2)
            + abs(counts["test"] - 3)
            for counts in q.values()
        )
        return (
            seq_eval_ok,
            seq_target,
            balance,
            tuple(
                (bucket, q[bucket]["train"], q[bucket]["dev"], q[bucket]["test"])
                for bucket in FINAL_BUCKETS
            ),
        )

    return tuple(sorted(quotas, key=quota_key))


def _min_h7_targets(
    *,
    train_n: int,
    dev_n: int,
    test_n: int,
) -> dict[str, int]:
    """Minimum H7 positives per split for MIXED_SEQUENTIAL packing."""
    return {
        "train": 1 if train_n >= 1 else 0,
        "dev": 2 if dev_n >= 2 else (1 if dev_n >= 1 else 0),
        "test": 2 if test_n >= 2 else (1 if test_n >= 1 else 0),
    }


def _pack_bucket_components(
    components: Sequence[SemanticGroupComponent],
    *,
    train_n: int,
    dev_n: int,
    test_n: int,
    min_h7_per_split: Mapping[str, int] | None,
    rng: Random,
) -> dict[str, tuple[str, ...]] | None:
    """Pack one bucket's components into exact train/dev/test sizes."""
    total = sum(comp.size for comp in components)
    if total != train_n + dev_n + test_n:
        return None
    # Size-descending first (required for exact packing with size-4 groups),
    # then seeded shuffle only within equal-size ties.
    by_size: dict[int, list[SemanticGroupComponent]] = defaultdict(list)
    for comp in components:
        by_size[comp.size].append(comp)
    ordered: list[SemanticGroupComponent] = []
    for size in sorted(by_size, reverse=True):
        group = sorted(
            by_size[size],
            key=lambda comp: (-comp.h7_positive_count, comp.semantic_group),
        )
        rng.shuffle(group)
        ordered.extend(group)

    targets = {"train": train_n, "dev": dev_n, "test": test_n}
    min_h7 = {
        split: int(min_h7_per_split.get(split, 0)) if min_h7_per_split else 0
        for split in SPLITS
    }
    assignment: dict[str, str] = {}

    def h7_ok(partial: Mapping[str, Sequence[SemanticGroupComponent]]) -> bool:
        if min_h7_per_split is None:
            return True
        remaining_h7 = sum(
            comp.h7_positive_count
            for comp in ordered
            if comp.semantic_group not in assignment
        )
        deficit = 0
        for split in SPLITS:
            need = min_h7[split]
            if need <= 0 or targets[split] <= 0:
                continue
            have = sum(comp.h7_positive_count for comp in partial.get(split, ()))
            if have < need:
                deficit += need - have
        return deficit <= remaining_h7

    def finish_ok(partial: Mapping[str, Sequence[SemanticGroupComponent]]) -> bool:
        if min_h7_per_split is None:
            return True
        for split in SPLITS:
            need = min_h7[split]
            if need <= 0 or targets[split] <= 0:
                continue
            have = sum(comp.h7_positive_count for comp in partial.get(split, ()))
            if have < need:
                return False
        return True

    def rec(index: int, rem: dict[str, int]) -> bool:
        if index == len(ordered):
            return rem["train"] == rem["dev"] == rem["test"] == 0 and finish_ok(
                {
                    split: tuple(
                        comp
                        for comp in ordered
                        if assignment.get(comp.semantic_group) == split
                    )
                    for split in SPLITS
                }
            )
        comp = ordered[index]
        # Try splits in seeded order, preferring the most underfilled first.
        split_order = sorted(
            SPLITS,
            key=lambda split: (
                0 if rem[split] >= comp.size else 1,
                -rem[split],
                rng.random(),
            ),
        )
        for split in split_order:
            if rem[split] < comp.size:
                continue
            assignment[comp.semantic_group] = split
            rem[split] -= comp.size
            partial = {
                name: tuple(
                    item
                    for item in ordered
                    if assignment.get(item.semantic_group) == name
                )
                for name in SPLITS
            }
            if h7_ok(partial) and rec(index + 1, rem):
                return True
            rem[split] += comp.size
            del assignment[comp.semantic_group]
        return False

    if not rec(0, dict(targets)):
        return None
    out: dict[str, list[str]] = {split: [] for split in SPLITS}
    for comp in ordered:
        out[assignment[comp.semantic_group]].extend(comp.example_ids)
    return {split: tuple(sorted(ids)) for split, ids in out.items()}


def _score_assignment(
    examples_by_id: Mapping[str, PlannerExample],
    assignment: Mapping[str, str],
    *,
    train_size: int,
    dev_size: int,
    test_size: int,
) -> tuple[int, ...]:
    """Lower is better. Hard-invalid candidates return a large sentinel."""
    by_split: dict[str, list[PlannerExample]] = {split: [] for split in SPLITS}
    for example_id, split in assignment.items():
        by_split[split].append(examples_by_id[example_id])
    sizes = {split: len(items) for split, items in by_split.items()}
    if (
        sizes["train"] != train_size
        or sizes["dev"] != dev_size
        or sizes["test"] != test_size
    ):
        return (10**9,)

    h7 = {
        split: sum(1 for ex in items if count_explicit_h7_edges(ex) > 0)
        for split, items in by_split.items()
    }
    if min(h7.values()) < 1:
        return (10**9 - 1,)
    # Prefer resilient eval holdouts: ≥2 H7 in both dev and test when feasible.
    if h7["dev"] < 2 or h7["test"] < 2:
        return (10**9 - 2,)
    if h7["train"] < 8:
        return (10**9 - 3,)

    bucket_penalty = 0
    for split, items in by_split.items():
        counts = Counter(str(ex.metadata["final_bucket"]) for ex in items)
        expected = sizes[split] / len(FINAL_BUCKETS)
        bucket_penalty += sum(abs(counts[bucket] - expected) for bucket in FINAL_BUCKETS)

    h1_penalty = 0
    for split, items in by_split.items():
        counts = Counter(ex.graph.query_type.value for ex in items)
        # Target mix ≈ corpus rates: P/E 0.2 each, Mixed 0.6
        h1_penalty += abs(counts.get(QueryType.PERSONAL.value, 0) - 0.2 * sizes[split])
        h1_penalty += abs(
            counts.get(QueryType.ENVIRONMENTAL.value, 0) - 0.2 * sizes[split]
        )
        h1_penalty += abs(counts.get(QueryType.MIXED.value, 0) - 0.6 * sizes[split])

    # Target ~11/2/2; 9/3/3 remains reachable but loses to this objective.
    h7_target_penalty = (
        abs(h7["train"] - 11) + abs(h7["dev"] - 2) + abs(h7["test"] - 2)
    )
    fingerprint_key = tuple(sorted(assignment.items()))
    return (
        int(bucket_penalty * 100),
        int(h1_penalty * 100),
        h7_target_penalty,
        fingerprint_key,
    )


def _assignment_fingerprint(assignment: Mapping[str, str], seed: int) -> str:
    payload = {
        "seed": seed,
        "assignment": [
            {"example_id": example_id, "split": split}
            for example_id, split in sorted(assignment.items())
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _build_report(
    examples_by_id: Mapping[str, PlannerExample],
    assignment: Mapping[str, str],
    *,
    seed: int,
    fingerprint: str,
    components: Sequence[SemanticGroupComponent],
) -> dict[str, object]:
    by_split: dict[str, list[PlannerExample]] = {split: [] for split in SPLITS}
    for example_id, split in assignment.items():
        by_split[split].append(examples_by_id[example_id])

    def split_stats(items: Sequence[PlannerExample]) -> dict[str, object]:
        return {
            "n": len(items),
            "final_bucket": dict(
                sorted(Counter(str(ex.metadata["final_bucket"]) for ex in items).items())
            ),
            "query_type": dict(
                sorted(Counter(ex.graph.query_type.value for ex in items).items())
            ),
            "h7_positive": sum(1 for ex in items if count_explicit_h7_edges(ex) > 0),
            "semantic_groups": len({str(ex.metadata["semantic_group"]) for ex in items}),
            "template_groups": len({str(ex.metadata["template_group"]) for ex in items}),
            "example_ids": [ex.example_id for ex in sorted(items, key=lambda e: e.example_id)],
        }

    group_to_splits: dict[str, set[str]] = defaultdict(set)
    template_to_splits: dict[str, set[str]] = defaultdict(set)
    for example_id, split in assignment.items():
        example = examples_by_id[example_id]
        group_to_splits[str(example.metadata["semantic_group"])].add(split)
        template_to_splits[str(example.metadata["template_group"])].add(split)

    semantic_leakage = sorted(
        group for group, splits in group_to_splits.items() if len(splits) > 1
    )
    template_overlap = sorted(
        {
            template: sorted(splits)
            for template, splits in template_to_splits.items()
            if len(splits) > 1
        }.items()
    )

    return {
        "seed": seed,
        "fingerprint": fingerprint,
        "sizes": {split: len(by_split[split]) for split in SPLITS},
        "by_split": {split: split_stats(by_split[split]) for split in SPLITS},
        "semantic_group_leakage": semantic_leakage,
        "template_group_overlap": [
            {"template_group": template, "splits": splits}
            for template, splits in template_overlap
        ],
        "template_group_overlap_count": len(template_overlap),
        "n_semantic_groups": len(components),
        "n_h7_positive_total": sum(
            1 for ex in examples_by_id.values() if count_explicit_h7_edges(ex) > 0
        ),
        "example_to_split": dict(sorted(assignment.items())),
    }


def validate_split_result(
    result: StageASplitResult,
    *,
    train_size: int = DEFAULT_TRAIN_SIZE,
    dev_size: int = DEFAULT_DEV_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
) -> list[str]:
    """Return human-readable validation errors (empty if OK)."""
    errors: list[str] = []
    report = result.report
    sizes = report["sizes"]
    assert isinstance(sizes, dict)
    expected = {"train": train_size, "dev": dev_size, "test": test_size}
    if sizes != expected:
        errors.append(f"unexpected sizes: {sizes} (expected {expected})")
    if (
        len(result.train) != train_size
        or len(result.dev) != dev_size
        or len(result.test) != test_size
    ):
        errors.append(
            "result tuple lengths mismatch: "
            f"train={len(result.train)} dev={len(result.dev)} "
            f"test={len(result.test)}"
        )
    if report["semantic_group_leakage"]:
        errors.append(
            f"semantic_group leakage: {report['semantic_group_leakage']}"
        )
    all_ids = [ex.example_id for ex in (*result.train, *result.dev, *result.test)]
    if len(all_ids) != len(set(all_ids)):
        errors.append("duplicate example_id across splits")
    by_split = report["by_split"]
    assert isinstance(by_split, dict)
    for split in SPLITS:
        stats = by_split[split]
        assert isinstance(stats, dict)
        if int(stats["h7_positive"]) < 1:
            errors.append(f"{split} has no H7-positive examples")
    train_h7 = int(by_split["train"]["h7_positive"])  # type: ignore[index]
    total_h7 = int(report["n_h7_positive_total"])
    if train_h7 < (total_h7 // 2 + total_h7 % 2):
        errors.append(
            f"train H7 majority failed: train={train_h7} total={total_h7}"
        )
    # Stage-A feasibility corpus: ≥2 H7 in each eval split when totals allow.
    if (
        train_size == DEFAULT_TRAIN_SIZE
        and dev_size == DEFAULT_DEV_SIZE
        and test_size == DEFAULT_TEST_SIZE
        and total_h7 >= 15
    ):
        for split in ("dev", "test"):
            split_h7 = int(by_split[split]["h7_positive"])  # type: ignore[index]
            if split_h7 < 2:
                errors.append(
                    f"{split} H7 floor failed: {split_h7} < 2 (target ~11/2/2)"
                )
    return errors


def group_holdout_split(
    examples: Sequence[PlannerExample],
    *,
    train_size: int = DEFAULT_TRAIN_SIZE,
    dev_size: int = DEFAULT_DEV_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SPLIT_SEED,
    max_quota_candidates: int = 64,
    trials_per_quota: int = 24,
) -> StageASplitResult:
    """Assign examples to train/dev/test under semantic_group holdout.

    Search is deterministic given ``seed``. Input order does not affect the
    result: examples are indexed by ``example_id`` / ``stage_a_id``.
    """
    if train_size + dev_size + test_size != len(examples):
        raise ValueError(
            f"split sizes {train_size}+{dev_size}+{test_size} != n_examples "
            f"{len(examples)}"
        )
    examples_by_id = {ex.example_id: ex for ex in examples}
    if len(examples_by_id) != len(examples):
        raise ValueError("duplicate example_id in split input")
    for example in examples:
        stage_a_id, _, _ = _require_metadata(example)
        if stage_a_id != example.example_id:
            raise ValueError(
                f"example_id {example.example_id!r} != metadata stage_a_id "
                f"{stage_a_id!r}"
            )

    components = build_semantic_group_components(examples)
    by_bucket: dict[str, list[SemanticGroupComponent]] = defaultdict(list)
    for comp in components:
        by_bucket[comp.final_bucket].append(comp)

    quota_candidates = _bucket_quotas(
        train_size=train_size, dev_size=dev_size, test_size=test_size
    )
    rng_root = Random(seed)
    ranked_quotas = list(quota_candidates[:max_quota_candidates])
    # Keep sequential-capable quotas first (already sorted), then sample extras.
    if len(quota_candidates) > max_quota_candidates:
        rest = list(quota_candidates[max_quota_candidates:])
        rng_root.shuffle(rest)
        ranked_quotas.extend(rest[: max(0, max_quota_candidates // 2)])

    best: tuple[tuple, dict[str, str]] | None = None
    trial_counter = 0
    for quota_index, quota in enumerate(ranked_quotas):
        for trial in range(trials_per_quota):
            trial_counter += 1
            trial_seed = seed + 1009 * quota_index + 17 * trial
            rng = Random(trial_seed)
            assignment: dict[str, str] = {}
            failed = False
            for bucket in FINAL_BUCKETS:
                min_h7 = None
                if bucket == "MIXED_SEQUENTIAL":
                    min_h7 = _min_h7_targets(
                        train_n=quota[bucket]["train"],
                        dev_n=quota[bucket]["dev"],
                        test_n=quota[bucket]["test"],
                    )
                packed = _pack_bucket_components(
                    by_bucket[bucket],
                    train_n=quota[bucket]["train"],
                    dev_n=quota[bucket]["dev"],
                    test_n=quota[bucket]["test"],
                    min_h7_per_split=min_h7,
                    rng=rng,
                )
                if packed is None:
                    failed = True
                    break
                for split, ids in packed.items():
                    for example_id in ids:
                        assignment[example_id] = split
            if failed or len(assignment) != len(examples_by_id):
                continue
            score = _score_assignment(
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
            "failed to find a valid semantic_group holdout split; "
            f"tried {trial_counter} quota/trial combinations with seed={seed}"
        )

    assignment = best[1]
    fingerprint = _assignment_fingerprint(assignment, seed)
    report = _build_report(
        examples_by_id,
        assignment,
        seed=seed,
        fingerprint=fingerprint,
        components=components,
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
    errors = validate_split_result(
        result,
        train_size=train_size,
        dev_size=dev_size,
        test_size=test_size,
    )
    if errors:
        raise RuntimeError("split validation failed: " + "; ".join(errors))
    return result


def format_split_report(result: StageASplitResult) -> str:
    """Pretty-print the split report for CLI / logs."""
    report = result.report
    lines = [
        f"seed: {report['seed']}",
        f"fingerprint: {report['fingerprint']}",
        f"sizes: {report['sizes']}",
        f"semantic_group_leakage: {report['semantic_group_leakage']}",
        f"template_group_overlap_count: {report['template_group_overlap_count']}",
        f"n_semantic_groups: {report['n_semantic_groups']}",
        f"n_h7_positive_total: {report['n_h7_positive_total']}",
    ]
    by_split = report["by_split"]
    assert isinstance(by_split, dict)
    for split in SPLITS:
        stats = by_split[split]
        assert isinstance(stats, dict)
        lines.append(f"{split}:")
        lines.append(f"  n={stats['n']}")
        lines.append(f"  final_bucket={stats['final_bucket']}")
        lines.append(f"  query_type={stats['query_type']}")
        lines.append(f"  h7_positive={stats['h7_positive']}")
        lines.append(
            f"  semantic_groups={stats['semantic_groups']} "
            f"template_groups={stats['template_groups']}"
        )
    if report["template_group_overlap"]:
        lines.append("template_group_overlap:")
        for item in report["template_group_overlap"]:  # type: ignore[union-attr]
            lines.append(
                f"  {item['template_group']}: {item['splits']}"
            )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_DEV_SIZE",
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_TRAIN_SIZE",
    "FINAL_BUCKETS",
    "SemanticGroupComponent",
    "StageASplitResult",
    "build_semantic_group_components",
    "format_split_report",
    "group_holdout_split",
    "validate_split_result",
]
