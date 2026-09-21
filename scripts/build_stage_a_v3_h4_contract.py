"""Build Stage-A v3 H4_REFEXPR_V1 corpus (Phase 1). Does not train or touch TEST."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiergraph.planner.stage_a_v3_h4_build import build_stage_a_v3_h4_refexpr_v1


def main() -> int:
    report = build_stage_a_v3_h4_refexpr_v1(root=ROOT, write=True)
    print(json.dumps(
        {
            "v2_annotation_fingerprint": report["v2_annotation_fingerprint"],
            "v3_annotation_fingerprint": report["v3_annotation_fingerprint"],
            "anchors_shortened_safe_mechanical": report["totals"][
                "anchors_shortened_safe_mechanical"
            ],
            "anchors_shortened_approved_r6": report["totals"][
                "anchors_shortened_approved_r6"
            ],
            "anchors_removed_approved_r5": report["totals"][
                "anchors_removed_approved_r5"
            ],
            "anchors_final_replaced_examples": report["totals"][
                "anchors_final_replaced_examples"
            ],
            "unresolved_count": report["human_review"]["unresolved_count"],
            "n_test_annotations_materialized": report[
                "n_test_annotations_materialized"
            ],
            "v2_unchanged": report["v2_unchanged"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
