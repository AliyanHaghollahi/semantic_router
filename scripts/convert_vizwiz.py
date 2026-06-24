"""
scripts/convert_vizwiz.py
=========================
Converts Dataset204_Q3.json (VizWiz annotations) into the
training_data.json format used by the classifier, then merges
with the existing seed dataset.

VizWiz subclass → routing label mapping:
  object  → Environmental  (describing visible objects/surroundings)
  other   → Environmental  (spatial/navigation/scene questions)
  text    → Environmental  (reading visible text in the scene/image)

All 203 entries use both q1 and q2 → 406 new Environmental examples.
Merged with existing seed → final dataset printed at the end.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

def convert(vizwiz_path: str, output_path: str = None):
    with open(vizwiz_path) as f:
        raw = json.load(f)

    converted = []
    skipped = 0

    for entry in raw:
        subclass = entry.get("subclass", "").lower()
        image    = entry.get("image", "")

        # All subclasses in this file map to Environmental
        # (object = physical objects in scene,
        #  other  = spatial/scene description,
        #  text   = reading visible text from camera image)
        label = "Environmental"

        for q_key in ("q1", "q2"):
            q = entry.get(q_key, "").strip()
            if not q:
                skipped += 1
                continue
            converted.append({
                "query":    q,
                "label":    label,
                "source":   "vizwiz",
                "image":    image,
                "subclass": subclass,
                "class":    entry.get("class", ""),
            })

    print(f"Converted {len(converted)} queries from {len(raw)} VizWiz entries "
          f"({skipped} empty skipped)")

    # ── Merge with existing seed dataset ──────────────────────────
    seed_path = ROOT / "dataset" / "training_data.json"
    if seed_path.exists():
        with open(seed_path) as f:
            seed = json.load(f)
        print(f"Existing seed dataset: {len(seed)} examples")
    else:
        seed = []
        print("No existing seed dataset found — starting fresh")

    # Deduplicate: remove seed queries that also appear in VizWiz
    vizwiz_queries = {e["query"].lower().strip() for e in converted}
    seed_deduped = [e for e in seed if e["query"].lower().strip() not in vizwiz_queries]
    removed = len(seed) - len(seed_deduped)
    if removed:
        print(f"Removed {removed} duplicate(s) from seed")

    merged = seed_deduped + converted

    # ── Stats ──────────────────────────────────────────────────────
    from collections import Counter
    label_counts = Counter(e["label"] for e in merged)
    print(f"\nFinal merged dataset: {len(merged)} examples")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:15}: {count:4d}  ({count/len(merged)*100:.1f}%)")

    # ── Save ───────────────────────────────────────────────────────
    out = Path(output_path) if output_path else seed_path
    # Keep only the fields the classifier needs (query + label),
    # but preserve extra fields for reference
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nSaved to: {out}")
    return merged


if __name__ == "__main__":
    vizwiz_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "dataset" / "Dataset204_Q3.json")
    convert(vizwiz_path)


def add_balance_examples(output_path: str = None):
    """
    Add more Personal and Mixed examples to balance the dataset.
    Target: roughly 40% / 35% / 25% (Env / Personal / Mixed).
    """
    ROOT_PATH = Path(__file__).parent.parent
    out = Path(output_path) if output_path else ROOT_PATH / "dataset" / "training_data.json"

    with open(out) as f:
        data = json.load(f)

    extra_personal = [
        # Health
        {"query": "What pills am I supposed to take right now?", "label": "Personal"},
        {"query": "What is my current diagnosis?", "label": "Personal"},
        {"query": "How many times a day do I take this medication?", "label": "Personal"},
        {"query": "Does my health record show any recent changes?", "label": "Personal"},
        {"query": "What is my doctor's phone number?", "label": "Personal"},
        {"query": "What conditions am I being treated for?", "label": "Personal"},
        {"query": "What does my prescription say?", "label": "Personal"},
        {"query": "Am I due for any vaccines?", "label": "Personal"},
        {"query": "What is my blood type?", "label": "Personal"},
        {"query": "Do I have any upcoming lab tests scheduled?", "label": "Personal"},
        # Travel / bookings
        {"query": "What time does my flight leave?", "label": "Personal"},
        {"query": "Which terminal is my flight departing from?", "label": "Personal"},
        {"query": "What is my hotel reservation number?", "label": "Personal"},
        {"query": "How many bags am I allowed on my booking?", "label": "Personal"},
        {"query": "What seat did I book on this flight?", "label": "Personal"},
        {"query": "Is my ticket refundable?", "label": "Personal"},
        {"query": "What is my check-in time at the hotel?", "label": "Personal"},
        {"query": "What is my frequent flyer number?", "label": "Personal"},
        # Documents / identity
        {"query": "When does my driver's license expire?", "label": "Personal"},
        {"query": "What is my employee ID number?", "label": "Personal"},
        {"query": "What is my student ID?", "label": "Personal"},
        {"query": "Show me my health insurance details.", "label": "Personal"},
        {"query": "What is the name on my credit card?", "label": "Personal"},
        # Contacts / calendar
        {"query": "What is my sister's phone number?", "label": "Personal"},
        {"query": "What time is my meeting tomorrow?", "label": "Personal"},
        {"query": "Who is my emergency contact?", "label": "Personal"},
        {"query": "What did I have scheduled for today?", "label": "Personal"},
        {"query": "Remind me of my PIN number.", "label": "Personal"},
        {"query": "What is the address of my dentist?", "label": "Personal"},
        {"query": "Call my brother for me.", "label": "Personal"},
    ]

    extra_mixed = [
        {"query": "What is this pill and do I take it in the morning?", "label": "Mixed"},
        {"query": "Read this sign and tell me if this is my gate.", "label": "Mixed"},
        {"query": "What food is this and is it on my dietary restriction list?", "label": "Mixed"},
        {"query": "What does this label say and does it match my prescription?", "label": "Mixed"},
        {"query": "Is this the right building for my appointment?", "label": "Mixed"},
        {"query": "What floor is this and does it match my booking?", "label": "Mixed"},
        {"query": "How far is the pharmacy and do I need to refill my prescription?", "label": "Mixed"},
        {"query": "What restaurant is this and am I allowed to eat here given my allergies?", "label": "Mixed"},
        {"query": "Read the menu and tell me if there is anything I can eat.", "label": "Mixed"},
        {"query": "Is this my luggage and what does the tag say?", "label": "Mixed"},
        {"query": "What is written on this bottle and is this my medication?", "label": "Mixed"},
        {"query": "Where is the nearest exit and is it close to my gate?", "label": "Mixed"},
        {"query": "What time is it and will I make my appointment?", "label": "Mixed"},
        {"query": "Is this the right bus and where does it go?", "label": "Mixed"},
        {"query": "What is on this tray and can I eat any of it?", "label": "Mixed"},
        {"query": "Is this seat number correct for my ticket?", "label": "Mixed"},
        {"query": "What aisle is this and does the store have my brand of medication?", "label": "Mixed"},
        {"query": "Read this document and tell me if it mentions my name.", "label": "Mixed"},
        {"query": "Is this ATM working and do I have enough balance?", "label": "Mixed"},
        {"query": "What time does this place close and do I have time to get here from my appointment?", "label": "Mixed"},
        {"query": "What is this object and is it mine?", "label": "Mixed"},
        {"query": "Is this the correct entrance for my ticket type?", "label": "Mixed"},
        {"query": "What does this sign say and does it affect my route to the gate?", "label": "Mixed"},
        {"query": "Can I sit here and is my reservation for this section?", "label": "Mixed"},
        {"query": "What is the price and do I have enough on my card?", "label": "Mixed"},
        {"query": "Is the pharmacy open and does it carry my prescription?", "label": "Mixed"},
        {"query": "What bus stop is this and is it the one I need for my hotel?", "label": "Mixed"},
        {"query": "Is this elevator going to the right floor for my appointment?", "label": "Mixed"},
        {"query": "What color is this shirt and did I buy it before?", "label": "Mixed"},
        {"query": "Is this the correct form and does it have my name on it?", "label": "Mixed"},
    ]

    # Tag the source
    for e in extra_personal + extra_mixed:
        e["source"] = "augmented"

    data.extend(extra_personal)
    data.extend(extra_mixed)

    from collections import Counter
    label_counts = Counter(e["label"] for e in data)
    print(f"\nAfter balancing: {len(data)} examples")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:15}: {count:4d}  ({count/len(data)*100:.1f}%)")

    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to: {out}")


if __name__ == "__main__":
    vizwiz_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "dataset" / "Dataset204_Q3.json")
    convert(vizwiz_path)
    add_balance_examples()
