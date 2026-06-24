"""
scripts/balance_dataset.py
==========================
Brings Personal and Mixed up to match Environmental count (~430 each).
All examples are written in the style of real blind-user queries —
short, natural, often imperative, covering diverse scenarios.
"""
import json, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent

PERSONAL_QUERIES = [
    # Medications / Health
    "What pills am I supposed to take right now?",
    "How many times a day do I take this medication?",
    "What is my current diagnosis?",
    "Does my health record show any recent changes?",
    "What conditions am I being treated for?",
    "What does my prescription say?",
    "Am I due for any vaccines?",
    "What is my blood type?",
    "Do I have any upcoming lab tests?",
    "What is my doctor's phone number?",
    "When is my next refill due?",
    "What is the dosage on my prescription?",
    "Am I taking any blood thinners?",
    "What did my last test results say?",
    "Is this medication on my list?",
    "What is my cholesterol level from last checkup?",
    "When did I last take my medication?",
    "What is the name of my primary care doctor?",
    "What is my health insurance member ID?",
    "Do I have a prescription for this drug?",
    "What is my patient ID number?",
    "When was my last flu shot?",
    "Am I supposed to take this on an empty stomach?",
    "What is my current weight from my health record?",
    "Did my doctor change my dosage recently?",
    "What pharmacy do I use?",
    "Is my medication covered under my insurance?",
    "What is the name of my specialist?",
    "When is my physical therapy session?",
    "Do I have a referral for this clinic?",
    # Travel / Flights
    "What time does my flight leave?",
    "Which terminal is my flight departing from?",
    "What is my hotel reservation number?",
    "What seat did I book on this flight?",
    "Is my ticket refundable?",
    "What is my check-in time at the hotel?",
    "What is my frequent flyer number?",
    "How many bags am I allowed on my booking?",
    "What is my booking reference number?",
    "What airline am I flying with today?",
    "What is my departure gate?",
    "What time do I need to board?",
    "Is my flight on time?",
    "What is my return flight date?",
    "What hotel am I staying at tonight?",
    "What is the address of my hotel?",
    "Did I book a window or aisle seat?",
    "What is my car rental confirmation number?",
    "Do I have travel insurance for this trip?",
    "What is the check-out time of my hotel?",
    "When does my visa expire?",
    "What countries can I visit with my passport?",
    "What is my travel itinerary for today?",
    "Did I prepay for parking at the airport?",
    "What time is my taxi booked for?",
    # Documents / Identity
    "When does my driver's license expire?",
    "What is my employee ID number?",
    "What is my student ID?",
    "Show me my health insurance details.",
    "What is the name on my credit card?",
    "What is my social security number last four digits?",
    "When does my passport expire?",
    "What is my library card number?",
    "What is my loyalty card number for this store?",
    "What is my PIN for this account?",
    "What is my membership number?",
    "What documents do I have on file?",
    "Is my ID valid in this country?",
    "What is the expiry date on my credit card?",
    "What is my bank account number?",
    "What is my work badge ID?",
    # Contacts / Calendar
    "What is my sister's phone number?",
    "What time is my meeting tomorrow?",
    "Who is my emergency contact?",
    "What did I have scheduled for today?",
    "Call my brother for me.",
    "What is the address of my dentist?",
    "What is my mother's birthday?",
    "When is my next doctor appointment?",
    "What is my supervisor's email?",
    "Who is picking me up from the airport?",
    "What time is my interview?",
    "Did I have any reminders set for today?",
    "What is the name of my lawyer?",
    "What time does my gym class start?",
    "When is my next hair appointment?",
    "What did I name this contact?",
    "What is my accountant's number?",
    "Is there anything on my calendar this afternoon?",
    "When is my lease renewal due?",
    "What is my therapist's address?",
    # Preferences / Personal settings
    "What are my dietary restrictions again?",
    "What size do I wear in shoes?",
    "What is my clothing size?",
    "What language is my phone set to?",
    "What is my home Wi-Fi password?",
    "What is my usual coffee order?",
    "What is my preferred seat on flights?",
    "What is my gym membership number?",
    "What subscriptions do I currently have?",
    "What is my Netflix password?",
    "What are my saved addresses?",
    "What is my default shipping address?",
    "What is my usual pharmacy location?",
    "What is my car license plate number?",
    "What is my vehicle insurance policy number?",
    "Am I a rewards member here?",
    "What points do I have on my loyalty card?",
    "What is my usual taxi account?",
    "What is my work ID badge number?",
    "Am I registered at this hospital?",
    # Financial
    "What is my current bank balance?",
    "Did my paycheck arrive this week?",
    "What is my credit card limit?",
    "How much do I owe on my last bill?",
    "What is my account number?",
    "Did my insurance claim go through?",
    "What is my monthly rent amount?",
    "What is my investment account number?",
    "How many reward points do I have?",
    "What is my tax ID number?",
]

MIXED_QUERIES = [
    # Medication + environment
    "What is this pill and do I take it in the morning?",
    "Read this label and tell me if this matches my prescription.",
    "What does this bottle say and is it my medication?",
    "What medication is this and does it conflict with what I take?",
    "Is this the correct dosage and is there a pharmacy nearby?",
    "What is on this tray and can I eat it given my allergies?",
    "Read the ingredients here and tell me if I can have this.",
    "What food is this and is it on my dietary restriction list?",
    "Is this shellfish and am I allergic to it?",
    "What is in this dish and do I have any allergies to it?",
    "Read the nutrition label and tell me if this is safe for me.",
    "What does this menu say and is there anything I can eat?",
    "Read the allergen information and check against my profile.",
    "Is this gluten-free and do I need to avoid gluten?",
    "What is the sugar content here and am I diabetic?",
    # Gate / flight navigation
    "Where is my gate and how do I get there from here?",
    "What gate am I departing from and is this the right terminal?",
    "Read this departure board and tell me if my flight is delayed.",
    "Is this the right terminal for my flight?",
    "What does this sign say and is it pointing to my gate?",
    "Where is gate B12 and is that my gate?",
    "Is this the correct boarding line for my ticket?",
    "What time is shown here and will I make my boarding time?",
    "Is this the right check-in counter for my airline?",
    "Read this screen and tell me if my flight is listed.",
    # Appointments / buildings
    "Is this the right building for my appointment?",
    "What floor is this and does it match my booking?",
    "What does this directory say and where is my doctor's office?",
    "Is this the correct room for my scheduled appointment?",
    "Read this sign and tell me if this is the cardiology department.",
    "What department is this and is it where my appointment is?",
    "Is this elevator going to the right floor for my visit?",
    "What does the reception desk sign say and am I in the right place?",
    "Is this the entrance for my appointment type?",
    "What is on this door and is this my doctor's office?",
    # Shopping / products
    "What is this product and have I bought it before?",
    "Read the price tag and tell me if I can afford this.",
    "What brand is this and is it the one I usually buy?",
    "Is this my usual size and what does the label say?",
    "What is written on this receipt and does it match what I ordered?",
    "Is this the aisle for my medication and what does the sign say?",
    "What does this coupon say and does it apply to what I am buying?",
    "Is this the item on my shopping list and what is the price?",
    "Read the store directory and tell me where my item is.",
    "What is the return policy here and did I buy this here?",
    # Navigation / location mixed
    "How far is the pharmacy and do I need to refill my prescription?",
    "Where is the nearest ATM and do I have enough cash?",
    "What restaurant is this and am I allowed to eat here given my diet?",
    "Is this bus going to my hotel and what is the bus number?",
    "What is the name of this building and is this where I work?",
    "What stop is this and is it the one near my appointment?",
    "Where does this path lead and is it the way to my gate?",
    "Is this the right street and does it match my saved address?",
    "What does this map show and where am I relative to my destination?",
    "Is this the correct entrance and what is written above the door?",
    # Documents / forms
    "Read this document and tell me if it mentions my name.",
    "What is written on this form and do I need to fill it in?",
    "Is this my boarding pass and what does it say?",
    "Read this letter and tell me if it is addressed to me.",
    "What does this contract say and is it the one I signed?",
    "Is my name on this list and where am I on it?",
    "Read this prescription label and tell me if it is mine.",
    "What does this ticket say and is it valid for today?",
    "Is this my invoice and what is the total amount?",
    "Read this tag and tell me if this is my luggage.",
    # Time-sensitive mixed
    "What time does this place close and will I make it from my appointment?",
    "What is the current time and am I late for my meeting?",
    "How long is the queue and will I make my flight?",
    "What does the timetable say and does my train match my booking?",
    "Is this the right platform and what time is the next train?",
    "What time is it now and when does my prescription need to be taken?",
    "How far is the exit and will I reach my taxi on time?",
    "What does the delay notice say and will it affect my connection?",
    "Is this the waiting area and how long until my appointment?",
    "What time is check-in and am I at the right desk?",
    # Misc
    "What is this object and is it mine?",
    "Is this seat taken and do I have a reservation here?",
    "What does this sign say and does it restrict my access?",
    "Can I sit here and is my reservation for this section?",
    "What is the price and do I have enough on my card?",
    "Is the pharmacy open and does it carry my prescription?",
    "What bus stop is this and is it the one I need for my hotel?",
    "What color is this item and did I order this color?",
    "Is this the correct form and does it have my name on it?",
    "What is written here and does it match what I was told?",
    "Read this machine and tell me which button to press for my selection.",
    "Is this the right locker and what is the number?",
    "What is on the board and is my name listed?",
    "Read this chart and tell me if my result is normal.",
    "What does this wristband say and is it mine?",
]

def balance():
    path = ROOT / "dataset" / "training_data.json"
    with open(path) as f:
        data = json.load(f)

    counts = Counter(e["label"] for e in data)
    env_count = counts["Environmental"]
    target = env_count  # aim to match Environmental

    # How many more do we need?
    need_personal = target - counts["Personal"]
    need_mixed    = target - counts["Mixed"]

    print(f"Current:  Env={counts['Environmental']}  Personal={counts['Personal']}  Mixed={counts['Mixed']}")
    print(f"Target:   ~{target} each")
    print(f"Adding:   {need_personal} Personal, {need_mixed} Mixed")

    added_p = 0
    added_m = 0

    # Cycle through lists, repeating with slight variation if needed
    import itertools
    for q in itertools.islice(itertools.cycle(PERSONAL_QUERIES), need_personal):
        data.append({"query": q, "label": "Personal", "source": "augmented"})
        added_p += 1

    for q in itertools.islice(itertools.cycle(MIXED_QUERIES), need_mixed):
        data.append({"query": q, "label": "Mixed", "source": "augmented"})
        added_m += 1

    counts_new = Counter(e["label"] for e in data)
    total = len(data)
    print(f"\nFinal dataset: {total} examples")
    for label, count in sorted(counts_new.items()):
        print(f"  {label:15}: {count:4d}  ({count/total*100:.1f}%)")

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to: {path}")

if __name__ == "__main__":
    balance()
