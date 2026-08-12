#!/usr/bin/env python3
"""Interactive Step-A annotation over the frozen Stage-A 120 examples.

Annotates operation spans, OperatorType, and anchor spans only.
Does not modify ``dataset/planner/stage_a_final_selection.jsonl``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tiergraph.enums import OperatorType
from tiergraph.planner.annotation_step_a import (
    ANSWER_OPERATORS,
    DEFAULT_FROZEN_SELECTION_PATH,
    DEFAULT_STEP_A_ANNOTATIONS_PATH,
    H4_ANCHOR_GUIDANCE,
    StepAAnnotationSession,
    demo_step_a_interaction,
    ensure_step_a_annotations_initialized,
    find_substring_occurrences,
    format_annotation_preview,
    format_operator_help,
    format_query_with_word_indexes,
    format_step_a_progress,
    parse_step_a_command,
)


def _prompt_operator() -> OperatorType:
    print("OperatorType:")
    for index, op in enumerate(ANSWER_OPERATORS, start=1):
        print(f"  {index} = {op.value}")
    while True:
        raw = input("operator (number or name)> ").strip()
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(ANSWER_OPERATORS):
                return ANSWER_OPERATORS[index - 1]
        try:
            op = OperatorType(raw)
        except ValueError:
            print("invalid operator; try again")
            continue
        if op is OperatorType.FUSE or op not in ANSWER_OPERATORS:
            print("FUSE / unknown operator not allowed")
            continue
        return op


def _prompt_occurrence(query: str, substring: str) -> int:
    occurrences = find_substring_occurrences(query, substring)
    if not occurrences:
        raise ValueError(f"substring not found: {substring!r}")
    if len(occurrences) == 1:
        return 0
    print(f"Found {len(occurrences)} occurrences:")
    for index, (start, end) in enumerate(occurrences):
        print(f"  {index}: [{start}:{end}] {query[start:end]!r}")
    while True:
        raw = input("occurrence index> ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(occurrences):
            return int(raw)
        print("invalid occurrence index")


def _print_current(session: StepAAnnotationSession) -> None:
    record = session.current()
    if record is None:
        print("All examples are COMPLETE (or none remaining UNREVIEWED).")
        print(format_step_a_progress(session.summary()))
        return
    position, total = session.current_position()
    print("-" * 60)
    print(f"Step-A example {position} / {total}")
    print(f"stage_a_id: {record.stage_a_id}")
    print(f"final_bucket: {record.final_bucket}")
    print(f"derived_query_type: {record.derived_query_type.value}")
    print(f"status: {record.step_a_status.value}")
    print()
    print("QUERY:")
    print(record.query)
    print("words:", format_query_with_word_indexes(record.query))
    print()
    print(format_annotation_preview(record))
    print()
    print("Controls:")
    print("a = add operation")
    print("r = remove operation (by index)")
    print("n = finish operations / move to anchors (informational)")
    print("x = add anchor")
    print("d = remove anchor (by index)")
    print("v = preview")
    print("h = operator help")
    print("s = skip")
    print("b = back")
    print("e = reopen COMPLETE example for edit")
    print("p = progress")
    print("c = confirm COMPLETE")
    print("q = save and quit")
    print("-" * 60)


def run_interactive(selection_path: Path, annotations_path: Path) -> int:
    records = ensure_step_a_annotations_initialized(
        selection_path=selection_path,
        annotations_path=annotations_path,
    )
    session = StepAAnnotationSession(
        records,
        annotations_path=annotations_path,
        selection_path=selection_path,
    )
    print(format_step_a_progress(session.summary()))
    print()
    try:
        while True:
            if session.current() is None:
                print("No UNREVIEWED examples remaining.")
                print(format_step_a_progress(session.summary()))
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
                action, _ = parse_step_a_command(raw)
            except ValueError as exc:
                print(f"{exc}. Try a/r/n/x/d/v/h/s/b/e/p/c/q.")
                continue
            try:
                if action == "add_operation":
                    substring = input("operation substring> ")
                    occurrence = _prompt_occurrence(
                        session.current().query, substring
                    )
                    operator = _prompt_operator()
                    record = session.add_operation(
                        substring, operator, occurrence=occurrence
                    )
                    print(
                        f"added operation[{record.operations[-1].operation_index}]"
                    )
                elif action == "remove_operation":
                    index = int(input("operation_index> ").strip())
                    session.remove_operation(index)
                    print(f"removed operation[{index}]")
                elif action == "to_anchors":
                    print("Now add anchors with 'x' (or 'c' if none needed).")
                    print(H4_ANCHOR_GUIDANCE)
                elif action == "add_anchor":
                    print(H4_ANCHOR_GUIDANCE)
                    substring = input("anchor substring> ")
                    occurrence = _prompt_occurrence(
                        session.current().query, substring
                    )
                    record = session.add_anchor(substring, occurrence=occurrence)
                    print(f"added anchor[{record.anchors[-1].anchor_index}]")
                elif action == "remove_anchor":
                    index = int(input("anchor_index> ").strip())
                    session.remove_anchor(index)
                    print(f"removed anchor[{index}]")
                elif action == "preview":
                    print(format_annotation_preview(session.current()))
                elif action == "help":
                    print(format_operator_help())
                elif action == "skip":
                    session.skip()
                    print("skipped")
                elif action == "back":
                    previous = session.back()
                    if previous is None:
                        print("nothing to go back to")
                    else:
                        print(f"back to {previous.stage_a_id}")
                elif action == "reopen":
                    record = session.reopen_for_edit()
                    print(f"reopened {record.stage_a_id}")
                elif action == "progress":
                    print(format_step_a_progress(session.summary()))
                elif action == "complete":
                    record = session.mark_complete()
                    print(f"COMPLETE {record.stage_a_id}")
                elif action == "quit":
                    session.save()
                    print("saved.")
                    print(format_step_a_progress(session.summary()))
                    return 0
            except Exception as exc:  # noqa: BLE001
                print(f"error: {exc}")
    except KeyboardInterrupt:
        print("\nInterrupted — saving.")
        session.save()
        print(format_step_a_progress(session.summary()))
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / DEFAULT_FROZEN_SELECTION_PATH,
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / DEFAULT_STEP_A_ANNOTATIONS_PATH,
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
        help="initialize annotation JSONL from frozen selection and exit",
    )
    args = parser.parse_args(argv)

    if args.demo:
        print(demo_step_a_interaction())
        return 0

    records = ensure_step_a_annotations_initialized(
        selection_path=args.selection,
        annotations_path=args.annotations,
    )
    if args.init_only or args.summary:
        from tiergraph.planner.annotation_step_a import summarize_step_a_progress

        print(format_step_a_progress(summarize_step_a_progress(records)))
        print(f"annotations: {args.annotations}")
        return 0

    return run_interactive(args.selection, args.annotations)


if __name__ == "__main__":
    raise SystemExit(main())
