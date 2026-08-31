"""Instantiate Stage-A v2 authored family specs into concrete reviewable candidates.

Todo 3A only: writes authored candidate JSONL for human review.
Does not write final selection, annotations, or splits.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tiergraph.planner.corpus import normalize_query_key
from tiergraph.planner.stage_a_selection import (
    infer_domain,
    infer_mixed_semantic_group,
    infer_pe_semantic_group,
    load_jsonl,
)
from tiergraph.planner.stage_a_v2_candidates import (
    STAGE_A_V2_AUTHORED_SPECS_PATH,
    STAGE_A_V2_CANDIDATES_PATH,
    load_v1_frozen_index,
    validate_candidate_row,
)
from tiergraph.planner.stage_a_v2_spec import (
    STAGE_A_V1_SELECTION_PATH,
    is_legal_h7_pair,
    parse_h7_family_label,
    resolve_authored_holdout_family,
)


STAGE_A_V2_AUTHORED_CANDIDATES_PATH = Path(
    "dataset/planner/stage_a_v2_authored_candidates.jsonl"
)
STAGE_A_V2_AUTHORED_INSTANTIATE_REPORT_PATH = Path(
    "dataset/planner/stage_a_v2_authored_instantiate_report.json"
)

REVIEW_STATUS = "needs_review"


# ---------------------------------------------------------------------------
# Query banks: exactly planned_paraphrases unique queries per family
# ---------------------------------------------------------------------------

def _build_query_banks() -> dict[str, tuple[str, ...]]:
    """Hand-authored, scenario-diverse queries keyed by authored_template_family."""

    banks: dict[str, tuple[str, ...]] = {}

    # ---- MIXED_IMPLICIT (82) ----
    banks["my_medication_scene_match"] = (
        "Is this white oval tablet the one matching my evening blood-pressure dose?",
        "Does this pink capsule match the medication I am scheduled to take tonight?",
        "Is the blister pack in my hand the prescription refill listed in my profile?",
        "Does this amber bottle match the statin my doctor currently has me on?",
        "Is this scored tablet the morning dose recorded in my medication list?",
        "Does this inhaler match the rescue medication saved in my health profile?",
        "Is this liquid syringe dose the pediatric antibiotic listed under my account?",
    )
    banks["my_allergy_menu_safe"] = (
        "Is this plated dish safe given my shellfish allergy profile?",
        "Would this dessert trigger the peanut allergy saved in my profile?",
        "Is this soup compatible with my lactose intolerance record?",
        "Does this sandwich contain anything flagged in my gluten-restriction profile?",
        "Is this salad dressing safe for my tree-nut allergy list?",
        "Would this sauce violate the sesame restriction in my dietary profile?",
    )
    banks["my_appointment_entrance_cue"] = (
        "Is this glass doorway the entrance for my cardiology appointment?",
        "Does this lobby door lead to the clinic listed on my appointment?",
        "Is this side entrance the one my dermatology visit says to use?",
        "Does this numbered door match the building entrance for my therapy session?",
        "Is this revolving door the arrival point saved for my specialist visit?",
        "Does this canopy entrance match the check-in location on my appointment?",
    )
    banks["my_reservation_seat_marker"] = (
        "Does this labeled chair match the seat on my reservation?",
        "Is this aisle seat marker the one assigned in my booking?",
        "Does this booth number match the table reserved under my name?",
        "Is this window seat the reservation seat stored in my ticket?",
        "Does this wheelchair-accessible seat match my reserved seat code?",
        "Is this row-letter plaque the seat marker from my reservation?",
    )
    banks["my_order_pickup_window"] = (
        "Is this counter the pickup window for my online order?",
        "Does this numbered hatch match the pickup location for my takeout?",
        "Is this drive-up window where my mobile order should be collected?",
        "Does this bag rack correspond to the pickup spot for my cafe order?",
        "Is this will-call desk the pickup point listed for my pharmacy order?",
        "Does this kiosk window match the collection counter for my preorder?",
    )
    banks["my_boarding_pass_gate_board"] = (
        "Does this gate board show the gate printed on my boarding pass?",
        "Is this departure screen listing the flight on my boarding pass?",
        "Does this boarding-zone sign match the zone on my boarding pass?",
        "Is this jet-bridge entrance the gate shown on my boarding pass?",
        "Does this flight status panel match the itinerary on my boarding pass?",
        "Is this standby board reflecting the same flight as my boarding pass?",
    )
    banks["my_luggage_carousel_tag"] = (
        "Is this black suitcase the bag tagged under my name?",
        "Does this hard-shell case match the luggage ID saved to my trip?",
        "Is this duffel with the ribbon the one registered to my booking?",
        "Does this checked bag match the claim tag in my travel profile?",
        "Is this spinner suitcase the luggage linked to my frequent-flyer account?",
    )
    banks["my_meeting_room_directory"] = (
        "Does this wall directory list the room for my next meeting?",
        "Is this floor map highlighting the conference room on my calendar?",
        "Does this room plaque match the meeting location in my schedule?",
        "Is this lobby screen showing the room booked on my calendar invite?",
        "Does this suite directory include the workspace reserved for my call?",
    )
    banks["my_dietary_restriction_dish"] = (
        "Does this stew respect the vegetarian restriction in my profile?",
        "Is this curry compatible with the halal preference saved for me?",
        "Would this broth violate the low-sodium limit in my dietary profile?",
        "Does this pastry fit the egg-free restriction listed under my account?",
        "Is this side dish compliant with the vegan diet stored in my profile?",
    )
    banks["my_prescription_label_dose"] = (
        "Does this bottle label match the dosage I was prescribed?",
        "Is the mg amount on this vial the dose recorded in my prescription?",
        "Does this pharmacy sticker show the frequency listed in my prescription?",
        "Is this dropper bottle labeled with the concentration my doctor prescribed?",
        "Does this unit-dose pack match the strength saved in my prescription file?",
    )
    banks["my_hotel_room_key_panel"] = (
        "Is this key panel the access point for my reserved hotel room?",
        "Does this door lock accept the key code for my booking?",
        "Is this elevator bank the one serving my reserved room floor?",
        "Does this hallway plaque match the room number on my hotel reservation?",
        "Is this RFID reader the entry panel for my reserved suite?",
    )
    banks["my_package_locker_bank"] = (
        "Is this locker bank where my delivery package should be retrieved?",
        "Does this parcel locker match the pickup code sent for my shipment?",
        "Is this Amazon locker the unit linked to my delivery notice?",
        "Does this outdoor locker wall hold the package listed in my account?",
        "Is this apartment mailroom locker the one assigned for my parcel?",
    )
    banks["my_train_platform_reservation"] = (
        "Is this platform the one assigned on my train reservation?",
        "Does this track sign match the platform listed on my ticket?",
        "Is this boarding area the reserved platform for my rail booking?",
        "Does this LED platform display show the train on my reservation?",
        "Is this coach marker the platform section printed on my ticket?",
    )
    banks["my_workspace_badge_reader"] = (
        "Is this badge reader the access point tied to my employee badge?",
        "Does this turnstile accept the badge credentials in my profile?",
        "Is this office door reader the one linked to my workspace access?",
        "Does this biometric panel match the entry method stored for my badge?",
        "Is this lobby scanner the badge checkpoint for my building access?",
    )
    banks["my_cart_item_shelf_match"] = (
        "Is this shelf item the product on my shopping list?",
        "Does this cereal box match an entry in my grocery list?",
        "Is this bottled water the brand saved on my shopping list?",
        "Does this produce bag correspond to an item on my shopping list?",
        "Is this cleaning spray listed under my current shopping list?",
    )

    # ---- MIXED_SEQUENTIAL (77) ----
    banks["identify_locate_plaque_vs_desk"] = (
        "What company name is on this desk plaque, and where is that desk in the office?",
        "Identify the logo on this nameplate, then locate the corresponding desk bay.",
        "Read which team is named on this plaque and tell me where that workstation sits.",
        "What department does this desk plaque identify, and where along this row is it?",
    )
    banks["identify_locate_menu_dish_station"] = (
        "What dish is pictured on this menu card, and where is its serving station?",
        "Identify the special on this chalkboard, then locate the counter that serves it.",
        "Which soup does this label name, and where is the soup station in this cafeteria?",
        "Name the pastry on this tag and locate the bakery case that holds it.",
    )
    banks["locate_navigate_clinic_corridor"] = (
        "Where is exam room 3B from here, and how do I walk there along this corridor?",
        "Locate the infusion suite on this floor, then give me navigation steps to reach it.",
        "Find the radiology check-in desk and navigate me there from this waiting area.",
    )
    banks["locate_navigate_platform_exit"] = (
        "Where is the platform exit relative to this bench, and how do I get there?",
        "Locate the station exit closest to track 4, then navigate me to it.",
    )
    banks["identify_describe_device_status"] = (
        "What device is this panel, and describe whether it shows an error or ready state.",
        "Identify this kiosk screen and describe the status message it is displaying.",
        "Name this printer model and describe whether it is jammed or idle.",
        "What machine is this control unit, and describe its current indicator lights.",
        "Identify this router front panel and describe the link-status LEDs you see.",
    )
    banks["identify_describe_signage_content"] = (
        "What kind of sign is this, and describe the warning text written on it.",
        "Identify this wall poster and describe the main instruction it gives.",
        "Name this overhead banner and describe the event details printed on it.",
        "What plaque is mounted here, and describe the hours listed on it.",
        "Identify this door sticker and describe the access rule it states.",
    )
    banks["identify_describe_packaging_label"] = (
        "What product is in this box, and describe the allergen line on its label.",
        "Identify this bottle and describe the storage instructions on the packaging.",
        "Name this pouch and describe the expiration date printed on it.",
        "What item is this carton, and describe the nutrition panel on the side.",
        "Identify this jar and describe the ingredient list on its label.",
    )
    banks["locate_describe_shelf_contents"] = (
        "Where is the clearance shelf, and describe what kinds of items are stocked there.",
        "Locate the endcap display and describe the products arranged on it.",
        "Find the refrigerated case by the entrance and describe what is currently inside.",
        "Where is aisle-7 mid-shelf, and describe the brands lined up there.",
        "Locate the returns bin and describe the items sitting in it right now.",
    )
    banks["locate_describe_room_layout"] = (
        "Where is the break room from this hallway, and describe how the seating is arranged.",
        "Locate the conference suite and describe the table layout inside.",
        "Find the quiet pod near the atrium and describe the furniture setup.",
        "Where is the training room, and describe whether it looks lecture-style or U-shaped.",
        "Locate the lounge alcove and describe the lighting and seating arrangement.",
    )
    banks["locate_describe_vehicle_bay"] = (
        "Where is loading bay C, and describe what vehicle is currently parked there.",
        "Locate the EV charging stall and describe the charger status you see.",
        "Find rental bay 12 and describe whether a car is present or the bay is empty.",
        "Where is the ambulance bay, and describe the markings painted on the pavement.",
    )
    banks["identify_locate_navigate_building_exit"] = (
        "What does this exit sign identify, where is that exit from here, and how do I walk to it?",
        "Identify the stairwell door label, locate that stairwell, then navigate me there.",
        "Name the emergency exit type on this placard, find its doorway, and guide me to it.",
        "What exit number is on this sign, where is that door, and how do I reach it from this lobby?",
        "Identify this fire-exit marker, locate the corresponding door, then give navigation steps.",
    )
    banks["identify_locate_navigate_counter_queue"] = (
        "What service does this counter sign name, where is that counter, and how do I get in line?",
        "Identify the ticketing desk label, locate the desk, then navigate me to the queue.",
        "Name the window listed on this board, find that window, and guide me there.",
        "What department is on this counter plaque, where does it sit, and how do I walk to it?",
        "Identify the returns counter header, locate it on this floor, then navigate me over.",
    )
    banks["resolve_locate_navigate_lab_draw_station"] = (
        "Using my lab appointment, where is the blood-draw station assigned to me, and how do I walk there?",
        "Based on my imaging order, locate the CT prep bay for my exam and navigate me there from check-in.",
        "Where is the phlebotomy chair listed for my morning draw, and how do I get there from this desk?",
        "Find the therapy room tied to my rehab appointment, then navigate me from the lobby to that door.",
    )
    banks["resolve_locate_navigate_rental_stall"] = (
        "Using my rental reservation, where is my assigned stall, and how do I navigate to it?",
        "Locate the parking stall on my car-rental booking and guide me there from the garage entry.",
        "Where is the stall number listed on my rental sheet, and how do I walk to it?",
    )
    banks["resolve_identify_locate_baggage_belt"] = (
        "Using my flight details, which carousel is mine, and where is that belt from here?",
        "Identify the baggage belt for my arriving flight and locate it in this claim hall.",
        "Which carousel matches my flight in my itinerary, and where does that belt sit?",
        "Name the claim belt for my booked flight and point out where it is located.",
    )
    banks["resolve_identify_describe_prescription_bottle"] = (
        "Using my prescription, which bottle is mine, and describe the dose text on its label.",
        "Identify the bottle that matches my prescribed medication and describe its warning strip.",
        "Which vial corresponds to my active prescription, and describe the directions printed?",
        "Find the bottle matching my prescription refill and describe the pharmacy sticker details.",
    )
    banks["resolve_locate_describe_appointment_room"] = (
        "Using my appointment, where is the assigned exam room, and describe the door markings.",
        "Locate the room listed on my clinic appointment and describe the entry plaque.",
        "Where is the suite for my scheduled visit, and describe what is posted beside the door?",
    )
    banks["resolve_only_identify_seat_marker"] = (
        "Which of these seat markers matches the seat on my reservation?",
        "Identify which labeled chair corresponds to my booked seat.",
        "Among these booth numbers, which one is the table on my reservation?",
        "Which row plaque matches the seat assignment in my ticket?",
    )
    banks["resolve_only_describe_cabin_map"] = (
        "Describe the cabin map section that corresponds to my reserved seat row.",
        "Using my seat assignment, describe how that row is drawn on this cabin diagram.",
        "Describe the legend area on this seat map that covers my booked cabin zone.",
    )

    # ---- H2/H3 hard cases (40) ----
    banks["urgency_distractor_scene"] = (
        "Quickly—ignore the flashing ad—describe whether this aisle looks crowded or clear.",
        "Right now, setting aside the loud announcement, describe the lighting in this hallway.",
        "Immediately, without focusing on the poster, describe how cluttered this counter is.",
        "Urgent: skip the side chatter and describe whether this doorway is blocked.",
    )
    banks["hyphenated_entity_identify"] = (
        "What is the name of this well-known check-in kiosk brand?",
        "Identify this state-of-the-art blood-pressure cuff on the cart.",
        "What model is this Wi-Fi-enabled label printer?",
        "Name this over-the-counter cold-and-flu display package.",
    )
    banks["identify_vs_describe_minimal_pair"] = (
        "What is the name of this machine?",
        "Describe how this machine looks and what status lights it shows.",
        "Identify the title printed on this binder spine.",
        "Describe the wear and condition of this binder cover.",
    )
    banks["retrieve_vs_describe_personal"] = (
        "What is my blood type on file?",
        "What emergency contact number do I have saved?",
        "What is my preferred pharmacy in my profile?",
        "What is my home mailing address on record?",
    )
    banks["locate_vs_identify_describe"] = (
        "Where is the nearest water fountain from here?",
        "Where along this corridor is the recycling station?",
        "Where is the accessible restroom relative to this elevator?",
        "Where is the visitor badge printer located on this floor?",
    )
    banks["navigate_direct_route"] = (
        "How do I get from this lobby desk to the south elevators?",
        "Give me walking directions from this gate area to baggage claim.",
        "How do I walk from pharmacy intake to the counseling window?",
        "Navigate me from the garage stairwell to the main entrance plaza.",
    )
    banks["retrieve_possessive_h5_none"] = (
        "What is my customer reference code?",
        "What is my membership ID?",
        "What is my employee badge number?",
        "What is my library card number on file?",
        "What is my preferred delivery address line?",
        "What is my preferred email address?",
        "What is my stored credit-card last four?",
        "What is my passport number on record?",
        "What is my primary care physician's name?",
        "What is my listed date of birth?",
        "What is my workplace department code?",
        "What is my saved Wi-Fi password at home?",
        "What is my frequent-flyer number?",
        "What is my insurance member ID?",
        "What is my emergency contact relationship?",
        "What is my preferred language setting?",
    )

    return banks


QUERY_BANKS: dict[str, tuple[str, ...]] = _build_query_banks()


def load_authored_family_specs(
    path: str | Path = STAGE_A_V2_AUTHORED_SPECS_PATH,
) -> list[dict[str, Any]]:
    return load_jsonl(path)


def _semantic_for_bucket(query: str, bucket: str, family: str) -> tuple[str, str]:
    if bucket in {"Personal", "Environmental"}:
        domain = infer_domain(query)
        template_group = family
        semantic_group = infer_pe_semantic_group(query, domain, template_group)
        return semantic_group, template_group
    domain = infer_domain(query)
    # Keep template_group identical to authored_template_family for leakage-safe holdout.
    template_group = family
    semantic_group = f"{domain}__{family}"
    # Prefer mixed semantic helper for natural-looking groups when PE-like.
    if bucket.startswith("MIXED"):
        # Still pin template_group to family ID; semantic can include domain.
        semantic_group = f"{infer_mixed_semantic_group(query, domain).split('__')[0]}__{family}"
    return semantic_group, template_group


def _candidate_id(family: str, index: int) -> str:
    safe = family.replace("-", "_")
    return f"auth_v2_{safe}_{index:02d}"


def instantiate_authored_candidates(
    *,
    specs_path: str | Path = STAGE_A_V2_AUTHORED_SPECS_PATH,
    inventory_path: str | Path = STAGE_A_V2_CANDIDATES_PATH,
    selection_path: str | Path = STAGE_A_V1_SELECTION_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize concrete authored candidates for all family specs."""
    specs = load_authored_family_specs(specs_path)
    v1 = load_v1_frozen_index(selection_path)
    inventory_keys: set[str] = set()
    if Path(inventory_path).is_file():
        for row in load_jsonl(inventory_path):
            inventory_keys.add(normalize_query_key(str(row["query"])))

    blocked = set(v1["query_keys"]) | inventory_keys
    candidates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    failed_families: list[dict[str, Any]] = []
    seen: set[str] = set()

    for spec in sorted(specs, key=lambda s: s["spec_id"]):
        family = str(spec["authored_template_family"])
        planned = int(spec["planned_paraphrases"])
        bucket = str(spec["proposed_final_bucket"])
        bank = QUERY_BANKS.get(family)
        if bank is None:
            failed_families.append(
                {"family": family, "reason": "missing_query_bank"}
            )
            continue
        if len(bank) != planned:
            failed_families.append(
                {
                    "family": family,
                    "reason": f"bank_size_{len(bank)}_!=_planned_{planned}",
                }
            )
            continue

        h7_families = list(spec.get("h7_families") or [])
        for label in h7_families:
            src, tgt = parse_h7_family_label(label)
            if not is_legal_h7_pair(src, tgt):
                failed_families.append(
                    {"family": family, "reason": f"illegal_h7_{label}"}
                )
                break
        else:
            for index, query in enumerate(bank, start=1):
                key = normalize_query_key(query)
                if key in blocked or key in seen:
                    conflicts.append(
                        {
                            "family": family,
                            "query": query,
                            "normalized_query": key,
                            "reason": (
                                "duplicate_against_frozen_v1_or_inventory"
                                if key in blocked
                                else "duplicate_within_authored_batch"
                            ),
                        }
                    )
                    continue
                if bucket == "MIXED_PARALLEL" and (
                    spec.get("h7_positive") or h7_families
                ):
                    conflicts.append(
                        {
                            "family": family,
                            "query": query,
                            "reason": "parallel_must_not_have_h7",
                        }
                    )
                    continue

                semantic_group, template_group = _semantic_for_bucket(
                    query, bucket, family
                )
                h5_positive = spec.get("h5_positive")
                if bucket == "MIXED_IMPLICIT":
                    h5_positive = True
                h7_positive = bool(h7_families) if bucket == "MIXED_SEQUENTIAL" else False
                if bucket == "MIXED_IMPLICIT":
                    h7_families_out: list[str] = []
                    h7_positive = False
                else:
                    h7_families_out = list(h7_families)

                holdout_family = resolve_authored_holdout_family(
                    {
                        "authored_template_family": family,
                        "authored_holdout_family": spec.get("authored_holdout_family"),
                    }
                )
                row = {
                    "candidate_id": _candidate_id(family, index),
                    "query": query,
                    "normalized_query": key,
                    "source_kind": "authored",
                    "source_id": None,
                    "authored_template_family": family,
                    "scenario_family": family,
                    "authored_holdout_family": holdout_family,
                    "semantic_group": semantic_group,
                    "template_group": template_group,
                    "final_bucket": bucket,
                    "proposed_final_bucket": bucket,
                    "operator_family": spec.get("operator_family"),
                    "h5_positive": h5_positive,
                    "h7_positive": h7_positive,
                    "h7_families": h7_families_out,
                    "multi_hop": bool(spec.get("multi_hop")),
                    "review_status": REVIEW_STATUS,
                    "spec_id": spec["spec_id"],
                    "spec_kind": spec["spec_kind"],
                    "acceptance_reason": None,
                    "rejection_reason": None,
                    "provenance": {
                        "origin": "stage_a_v2_authored_instantiate",
                        "spec_id": spec["spec_id"],
                        "authored_template_family": family,
                        "authored_holdout_family": holdout_family,
                        "planned_index": index,
                        "rationale": spec.get("rationale"),
                    },
                }
                errors = validate_candidate_row(row)
                if errors:
                    conflicts.append(
                        {
                            "family": family,
                            "candidate_id": row["candidate_id"],
                            "reason": "provenance_validation_failed",
                            "errors": errors,
                        }
                    )
                    continue
                seen.add(key)
                candidates.append(row)
            continue
        # illegal h7 broke inner for-else
        continue

    # Exact quota check per family
    by_family = Counter(r["authored_template_family"] for r in candidates)
    for spec in specs:
        family = str(spec["authored_template_family"])
        planned = int(spec["planned_paraphrases"])
        got = by_family.get(family, 0)
        if got != planned and family not in {f["family"] for f in failed_families}:
            if got < planned:
                failed_families.append(
                    {
                        "family": family,
                        "reason": f"underfilled_{got}_of_{planned}",
                    }
                )
            elif got > planned:
                failed_families.append(
                    {
                        "family": family,
                        "reason": f"exceeded_quota_{got}_of_{planned}",
                    }
                )

    candidates = sorted(
        candidates,
        key=lambda r: (r["proposed_final_bucket"], r["authored_template_family"], r["candidate_id"]),
    )

    report = {
        "concrete_authored_total": len(candidates),
        "counts_by_bucket": dict(
            Counter(r["proposed_final_bucket"] for r in candidates)
        ),
        "distinct_authored_template_families": sorted(
            {r["authored_template_family"] for r in candidates}
        ),
        "h5_positive_count": sum(1 for r in candidates if r.get("h5_positive") is True),
        "h5_negative_count": sum(1 for r in candidates if r.get("h5_positive") is False),
        "h5_unknown_count": sum(1 for r in candidates if r.get("h5_positive") is None),
        "h7_family_counts": dict(
            Counter(
                label
                for r in candidates
                for label in (r.get("h7_families") or [])
            )
        ),
        "h7_positive_count": sum(1 for r in candidates if r.get("h7_positive")),
        "multi_hop_count": sum(1 for r in candidates if r.get("multi_hop")),
        "duplicate_conflict_count": len(conflicts),
        "conflicts": conflicts,
        "failed_families": failed_families,
        "expected_totals": {
            "MIXED_IMPLICIT": 82,
            "MIXED_SEQUENTIAL": 77,
            "hard_cases": 40,
            "all": 199,
        },
    }
    return candidates, report


def write_authored_candidates(
    *,
    output_path: str | Path = STAGE_A_V2_AUTHORED_CANDIDATES_PATH,
    report_path: str | Path = STAGE_A_V2_AUTHORED_INSTANTIATE_REPORT_PATH,
    **kwargs: Any,
) -> dict[str, Any]:
    candidates, report = instantiate_authored_candidates(**kwargs)
    if report["failed_families"]:
        raise ValueError(f"authored instantiation failures: {report['failed_families']}")
    if report["duplicate_conflict_count"]:
        raise ValueError(
            f"authored instantiation conflicts: {report['conflicts'][:5]}"
        )
    if len(candidates) != report["expected_totals"]["all"]:
        raise ValueError(
            f"expected {report['expected_totals']['all']} authored candidates, "
            f"got {len(candidates)}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    Path(report_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = write_authored_candidates()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
