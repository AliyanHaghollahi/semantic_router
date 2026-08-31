#!/usr/bin/env python3
"""Interactive Step-B annotation over frozen COMPLETE Step-A examples.

Annotates H5/H6/H7 only. Never mutates Step-A gold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.planner.annotation_step_a import (
    DEFAULT_STEP_A_ANNOTATIONS_PATH,
    fingerprint_file,
    load_step_a_annotations,
)
from tiergraph.planner.annotation_step_b import (
    DEFAULT_STEP_B_ANNOTATIONS_PATH,
    StepBAnnotationSession,
    demo_step_b_interaction,
    ensure_step_b_annotations_initialized,
    format_step_b_help,
    format_step_b_preview,
    format_step_b_progress,
    parse_step_b_command,
)
from tiergraph.planner.annotations import ImplicitResolution


def _prompt_h5() -> ImplicitResolution:
    print("H5 ImplicitResolution:")
    print("  0 = NONE")
    print("  1 = IMPLICIT_RESOLVE_PERSONAL")
    while True:
        raw = input("h5> ").strip()
        if raw in {"0", "none", "NONE"}:
            return ImplicitResolution.NONE
        if raw in {"1", "implicit_resolve_personal", "IMPLICIT_RESOLVE_PERSONAL"}:
            return ImplicitResolution.IMPLICIT_RESOLVE_PERSONAL
        try:
            return ImplicitResolution(raw)
        except ValueError:
            print("invalid H5; try again")


def _print_current(session: StepBAnnotationSession) -> None:
    record = session.current()
    step_a = session.current_step_a()
    if record is None or step_a is None:
        print("No UNREVIEWED Step-B examples remaining.")
        print(format_step_b_progress(session.summary()))
        return
    position, total = session.current_position()
    print("-" * 60)
    print(f"Step-B example {position} / {total}")
    print(format_step_b_preview(record, step_a=step_a))
    print()
    print("Controls:")
    print("i = set H5 for an anchor_index")
    print("o = set H6 owner for an anchor_index")
    print("d = add H7 dependency (src tgt)")
    print("x = remove H7 dependency (src tgt)")
    print("v = preview")
    print("h = H5/H6/H7 help")
    print("s = skip")
    print("b = back")
    print("e = reopen COMPLETE example (stage_a_id)")
    print("p = progress")
    print("c = confirm COMPLETE")
    print("q = save and quit")
    print("-" * 60)


def run_interactive(step_a_path: Path, step_b_path: Path) -> int:
    step_a_before = fingerprint_file(step_a_path)
    step_a_records = load_step_a_annotations(step_a_path)
    records = ensure_step_b_annotations_initialized(
        step_a_path=step_a_path,
        step_b_path=step_b_path,
    )
    if fingerprint_file(step_a_path) != step_a_before:
        print("ERROR: Step-A file mutated during init", file=sys.stderr)
        return 1

    session = StepBAnnotationSession(
        records,
        step_a_records=step_a_records,
        step_b_path=step_b_path,
        step_a_path=step_a_path,
    )
    print(format_step_b_progress(session.summary()))
    print()
    try:
        while True:
            if session.current() is None:
                print("No UNREVIEWED examples remaining.")
                print(format_step_b_progress(session.summary()))
                session.save()
                return 0
            _print_current(session)
            try:
                raw = input("> ")
            except EOFError:
                print("\nEOF — saving and quitting.")
                session.save()
                return 0
            try:
                action, _ = parse_step_b_command(raw)
            except ValueError as exc:
                print(f"{exc}. Try i/o/d/x/v/h/s/b/e/p/c/q.")
                continue
            try:
                if action == "set_h5":
                    anchor_index = int(input("anchor_index> ").strip())
                    value = _prompt_h5()
                    session.set_h5(anchor_index, value)
                    print(f"set H5 anchor[{anchor_index}]={value.value}")
                elif action == "set_h6":
                    anchor_index = int(input("anchor_index> ").strip())
                    owner = int(input("owner_operation_index> ").strip())
                    session.set_h6(anchor_index, owner)
                    print(f"set H6 anchor[{anchor_index}] -> op[{owner}]")
                elif action == "add_dependency":
                    source = int(input("source_operation_index> ").strip())
                    target = int(input("target_operation_index> ").strip())
                    session.add_dependency(source, target)
                    print(f"added H7 {source} -> {target}")
                elif action == "remove_dependency":
                    source = int(input("source_operation_index> ").strip())
                    target = int(input("target_operation_index> ").strip())
                    session.remove_dependency(source, target)
                    print(f"removed H7 {source} -> {target}")
                elif action == "preview":
                    print(
                        format_step_b_preview(
                            session.current(),
                            step_a=session.current_step_a(),
                        )
                    )
                elif action == "help":
                    print(format_step_b_help())
                elif action == "skip":
                    session.skip()
                elif action == "back":
                    session.back()
                elif action == "reopen":
                    stage_a_id = input("stage_a_id> ").strip()
                    session.reopen(stage_a_id)
                    print(f"reopened {stage_a_id}")
                elif action == "progress":
                    print(format_step_b_progress(session.summary()))
                elif action == "complete":
                    session.mark_complete()
                    print("marked COMPLETE")
                    session.skip()
                elif action == "quit":
                    session.save()
                    print("saved; quitting")
                    return 0
            except Exception as exc:  # noqa: BLE001 - interactive UX
                print(f"error: {exc}")
    except KeyboardInterrupt:
        print("\nCtrl+C — saving and quitting.")
        session.save()
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step-a",
        type=Path,
        default=ROOT / DEFAULT_STEP_A_ANNOTATIONS_PATH,
    )
    parser.add_argument(
        "--step-b",
        type=Path,
        default=ROOT / DEFAULT_STEP_B_ANNOTATIONS_PATH,
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print progress and exit",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="print an in-memory dry-run interaction demo and exit",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="initialize Step-B JSONL from frozen Step A and exit",
    )
    args = parser.parse_args(argv)

    if args.demo:
        print(demo_step_b_interaction())
        return 0

    before = fingerprint_file(args.step_a)
    records = ensure_step_b_annotations_initialized(
        step_a_path=args.step_a,
        step_b_path=args.step_b,
    )
    after = fingerprint_file(args.step_a)
    if before != after:
        print("ERROR: Step-A file mutated", file=sys.stderr)
        return 1

    if args.init_only or args.summary:
        from tiergraph.planner.annotation_step_b import summarize_step_b_progress

        print(format_step_b_progress(summarize_step_b_progress(records)))
        print(f"step_b_rows: {len(records)}")
        print(f"step_a_fingerprint: {before[1][:16]}...")
        return 0

    return run_interactive(args.step_a, args.step_b)


if __name__ == "__main__":
    raise SystemExit(main())
