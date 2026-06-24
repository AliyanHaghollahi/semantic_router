"""
scripts/seed_edge_db.py — Seed the edge SQLite context store with realistic personal data.
Run once before the simulation: python scripts/seed_edge_db.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from context_store.edge_store import EdgeContextStore
from context_store.fog_store import FogContextStore


def seed_edge_store():
    print("Seeding Edge Context Store (SQLite)...")
    store = EdgeContextStore("data/edge_context.db")

    # Health profile
    store.add_medication("Lisinopril", "10mg, once daily, morning", "For blood pressure control")
    store.add_medication("Metformin", "500mg, twice daily with meals", "For type 2 diabetes")
    store.add_medication("Atorvastatin", "20mg, once daily at night", "Cholesterol management")
    store.add_allergy("Penicillin", "Causes hives and respiratory distress")
    store.add_allergy("Shellfish", "Anaphylactic reaction")
    store.add_allergy("Latex", "Contact dermatitis")
    store.add_health_condition("Type 2 Diabetes", "Diagnosed 2019, managed with Metformin")
    store.add_health_condition("Hypertension", "Stage 1, managed with Lisinopril")
    store.add_health_condition("Hyperlipidemia", "Managed with Atorvastatin")

    # Bookings
    store.add_flight(
        title="United Airlines UA447 → Dallas (DFW)",
        reference="UA447-DFW",
        departure="2026-06-09 15:45",
        gate="D34",
        seat="22A",
        destination="Dallas/Fort Worth International"
    )
    store.add_flight(
        title="American Airlines AA112 → New York (JFK)",
        reference="AA112-JFK",
        departure="2026-06-15 08:20",
        gate="B12",
        seat="14C",
        destination="John F. Kennedy International"
    )
    store.add_appointment(
        "Dr. Martinez – Cardiology Follow-up",
        "2026-06-10 14:00",
        location="UNT Health Science Center, Room 312",
        notes="Bring blood pressure log"
    )
    store.add_appointment(
        "Dentist – Cleaning",
        "2026-06-18 10:30",
        location="Denton Family Dental",
    )

    # Documents
    store.add_document("passport", "US Passport", masked_value="***-**-4729", expiry="2031-03-15")
    store.add_document("insurance", "BlueCross BlueShield", masked_value="BCB-4892-X", expiry="2026-12-31")
    store.add_document("id", "Texas Driver License", masked_value="***-***-2381", expiry="2028-08-20")

    # Contacts
    store.add_contact("Dr. Elena Martinez", phone="(940) 555-0182", relationship="Cardiologist")
    store.add_contact("Mom", phone="(940) 555-0199", relationship="Family")
    store.add_contact("Supervisor Aliyeh", email="aliyeh@unt.edu", relationship="Advisor")

    # Preferences
    store.set_preference("language", "English", "accessibility")
    store.set_preference("output_format", "spoken_audio", "accessibility")
    store.set_preference("dietary_restriction", "No shellfish, No penicillin antibiotics", "health")
    store.set_preference("font_size", "large", "display")

    print(f"  ✓ Edge store seeded.")
    store.close()


def seed_fog_store():
    print("Seeding Fog Context Store (FAISS/numpy)...")
    store = FogContextStore(
        index_path="data/fog_index.faiss",
        metadata_path="data/fog_metadata.json"
    )

    documents = [
        # Airport environment
        {"text": "Terminal D is the international departures terminal. Gates D1–D50 are on the main concourse.", "env": "airport"},
        {"text": "Gate D34 is located in Terminal D, main concourse, near the food court. Estimated walk from security: 5 minutes.", "env": "airport"},
        {"text": "CVS Pharmacy is located on the second floor of Terminal B, open 6 AM – 10 PM.", "env": "airport"},
        {"text": "The Airport Medical Center is near Gate B8. Open 24 hours for emergencies.", "env": "airport"},
        {"text": "Baggage claim for domestic flights is on Level 1, east side of the terminal.", "env": "airport"},
        {"text": "The moving walkway toward gates D30–D50 begins after the main atrium.", "env": "airport"},

        # Hospital / clinic
        {"text": "The cardiology department is on the 3rd floor, east wing. Take elevator bank C.", "env": "hospital"},
        {"text": "The pharmacy at UNT Health Science Center is in the ground floor lobby, open Monday–Friday 8am–6pm.", "env": "hospital"},
        {"text": "Patient check-in for outpatient appointments is at the main lobby, kiosk 3.", "env": "hospital"},

        # Supermarket / retail
        {"text": "Medications and health items are in Aisle 12 (pharmacy section) at the back of the store.", "env": "retail"},
        {"text": "The customer service desk is near the main entrance on your left.", "env": "retail"},
        {"text": "The exit is straight ahead through the automatic sliding doors.", "env": "retail"},

        # University campus
        {"text": "The HPCC Lab is in Discovery Park, Building E, Room E101.", "env": "campus"},
        {"text": "The UNT library is open until 10 PM on weekdays.", "env": "campus"},
        {"text": "Disability Services office is in Sage Hall, Room 167.", "env": "campus"},

        # General
        {"text": "Emergency exits are marked with green illuminated signs and are located at each end of the corridor.", "env": "general"},
        {"text": "Restrooms are typically located near elevator banks and escalators.", "env": "general"},
        {"text": "Accessible routes are marked with blue wheelchair symbols on the floor.", "env": "general"},
    ]

    store.add_documents(documents)
    store.save()
    print(f"  ✓ Fog store seeded with {store.size} documents.")


def seed_training_data():
    """Create a labeled training dataset for the classifier."""
    import json
    from pathlib import Path

    Path("dataset").mkdir(exist_ok=True)

    data = [
        # ── Personal ──────────────────────────────────────────────
        {"query": "What is my blood pressure medication?", "label": "Personal"},
        {"query": "How much Metformin do I take?", "label": "Personal"},
        {"query": "What are my allergies?", "label": "Personal"},
        {"query": "Am I allergic to penicillin?", "label": "Personal"},
        {"query": "What is my gate number for my flight today?", "label": "Personal"},
        {"query": "When is my doctor appointment?", "label": "Personal"},
        {"query": "What seat am I in on my flight?", "label": "Personal"},
        {"query": "What is my insurance plan ID?", "label": "Personal"},
        {"query": "Can you read my passport number?", "label": "Personal"},
        {"query": "What time is my cardiology appointment?", "label": "Personal"},
        {"query": "Do I have any meetings today?", "label": "Personal"},
        {"query": "What is my driver license expiration date?", "label": "Personal"},
        {"query": "Who is my cardiologist?", "label": "Personal"},
        {"query": "What food am I not allowed to eat?", "label": "Personal"},
        {"query": "What is my cholesterol medication?", "label": "Personal"},
        {"query": "Call my mom", "label": "Personal"},
        {"query": "What is my dietary restriction?", "label": "Personal"},
        {"query": "Read my upcoming calendar events", "label": "Personal"},
        {"query": "What medications do I take in the morning?", "label": "Personal"},
        {"query": "Do I have diabetes?", "label": "Personal"},
        {"query": "What is my health condition?", "label": "Personal"},
        {"query": "Show me my booking reference for the Dallas flight", "label": "Personal"},
        {"query": "What is my boarding time?", "label": "Personal"},
        {"query": "Find the contact info for Dr. Martinez", "label": "Personal"},
        {"query": "What is my next appointment location?", "label": "Personal"},

        # ── Environmental ─────────────────────────────────────────
        {"query": "What does this sign say?", "label": "Environmental"},
        {"query": "Is there a pharmacy nearby?", "label": "Environmental"},
        {"query": "What is in front of me?", "label": "Environmental"},
        {"query": "Where is the exit?", "label": "Environmental"},
        {"query": "What building am I in?", "label": "Environmental"},
        {"query": "How far is the nearest hospital?", "label": "Environmental"},
        {"query": "What color is the door to my left?", "label": "Environmental"},
        {"query": "Is this the right terminal?", "label": "Environmental"},
        {"query": "What restaurant is nearby?", "label": "Environmental"},
        {"query": "What does the label on this bottle say?", "label": "Environmental"},
        {"query": "Which direction should I walk to reach the gate?", "label": "Environmental"},
        {"query": "Are there stairs or an elevator here?", "label": "Environmental"},
        {"query": "What is this object I am holding?", "label": "Environmental"},
        {"query": "What floor am I on?", "label": "Environmental"},
        {"query": "Is the pharmacy open right now?", "label": "Environmental"},
        {"query": "What does the sign above the door say?", "label": "Environmental"},
        {"query": "How many steps are there in front of me?", "label": "Environmental"},
        {"query": "What is the name of this store?", "label": "Environmental"},
        {"query": "Is there a bench nearby?", "label": "Environmental"},
        {"query": "What time does this place close?", "label": "Environmental"},
        {"query": "Is this a one-way street?", "label": "Environmental"},
        {"query": "What language is this sign written in?", "label": "Environmental"},
        {"query": "Is the escalator going up or down?", "label": "Environmental"},
        {"query": "Are there people blocking the path?", "label": "Environmental"},
        {"query": "What is the nearest landmark?", "label": "Environmental"},

        # ── Mixed ─────────────────────────────────────────────────
        {"query": "What medication am I holding and is there a pharmacy nearby?", "label": "Mixed"},
        {"query": "Am I allergic to anything on this menu?", "label": "Mixed"},
        {"query": "Where is my gate and how do I get there?", "label": "Mixed"},
        {"query": "What food is this and can I eat it?", "label": "Mixed"},
        {"query": "Is this my prescription bottle and what does it say?", "label": "Mixed"},
        {"query": "Where is my doctor's office and am I going to the right floor?", "label": "Mixed"},
        {"query": "What are the ingredients here and do any conflict with my allergies?", "label": "Mixed"},
        {"query": "My flight departs at 3 PM and where is the gate from here?", "label": "Mixed"},
        {"query": "Am I holding the right medication and where can I get water to take it?", "label": "Mixed"},
        {"query": "What time is my appointment and how far is this clinic from here?", "label": "Mixed"},
        {"query": "Is this shellfish and am I allergic to it?", "label": "Mixed"},
        {"query": "What is my insurance and does this pharmacy accept it?", "label": "Mixed"},
        {"query": "How long until my flight and is this the right terminal?", "label": "Mixed"},
        {"query": "What does this pill look like and do I take this in the morning?", "label": "Mixed"},
        {"query": "I need to refill my prescription and where is the nearest CVS?", "label": "Mixed"},
        {"query": "What is my gate and is the moving walkway going in the right direction?", "label": "Mixed"},
        {"query": "What dish is this and am I allowed to eat it?", "label": "Mixed"},
        {"query": "What is my seat number and where is row 22 on this plane?", "label": "Mixed"},
        {"query": "Can I take my medication here and is there a water fountain nearby?", "label": "Mixed"},
        {"query": "What is my appointment time and is this the right building?", "label": "Mixed"},
    ]

    with open("dataset/training_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Training data created: {len(data)} samples ({sum(1 for d in data if d['label']=='Personal')} Personal, "
          f"{sum(1 for d in data if d['label']=='Environmental')} Environmental, "
          f"{sum(1 for d in data if d['label']=='Mixed')} Mixed)")


if __name__ == "__main__":
    import os, json
    os.makedirs("data", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

    # Only create training data if it does not already exist
    # Prevents overwriting your expanded VizWiz dataset
    if not os.path.exists("dataset/training_data.json"):
        seed_training_data()
    else:
        with open("dataset/training_data.json") as f:
            existing = json.load(f)
        print(f"  ✓ Training data already exists: {len(existing)} samples — skipping overwrite")

    seed_edge_store()
    seed_fog_store()

    # Train classifier on whatever dataset exists
    print("\nTraining classifier...")
    from router.classifier import QueryClassifier
    clf = QueryClassifier()
    clf.load_or_train(dataset_path="dataset/training_data.json")
    print("✓ Classifier trained and saved.")
    print("\n✓ All stores seeded. Ready to run: python run_simulation.py")