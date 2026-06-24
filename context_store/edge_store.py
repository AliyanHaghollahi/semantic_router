"""
context_store/edge_store.py — Edge Context Store (SQLite)
==========================================================
Personal knowledge base stored on-device (smartphone / laptop).
Data NEVER leaves this store toward the fog server.

Tables:
  - health_profile   (medications, allergies, conditions, emergency contacts)
  - bookings         (flights, hotel reservations, appointments)
  - documents        (passport, ID, insurance — stored as masked references)
  - contacts         (name, phone, email, relationship)
  - preferences      (dietary, accessibility, language)

Usage:
    store = EdgeContextStore("data/edge_context.db")
    context = store.retrieve_context("What is my blood pressure medication?")
    # → "Medications: Lisinopril 10mg (morning), Metformin 500mg (twice daily)"
"""

import sqlite3
import json
import logging
import re
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class EdgeContextStore:

    def __init__(self, db_path: str = "data/edge_context.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("EdgeContextStore initialized at %s", self.db_path)

    # ── Schema ────────────────────────────────────────────────────

    def _init_schema(self):
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS health_profile (
            id INTEGER PRIMARY KEY,
            category TEXT NOT NULL,      -- 'medication', 'allergy', 'condition', 'emergency_contact'
            name TEXT NOT NULL,
            details TEXT,
            notes TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY,
            type TEXT NOT NULL,           -- 'flight', 'hotel', 'appointment'
            reference TEXT,
            title TEXT NOT NULL,
            location TEXT,
            datetime_str TEXT,
            gate TEXT,
            seat TEXT,
            notes TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            doc_type TEXT NOT NULL,       -- 'passport', 'id', 'insurance', 'prescription'
            label TEXT NOT NULL,          -- display name
            masked_value TEXT,            -- e.g. "***-**-1234" (never full value in logs)
            expiry TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            phone TEXT,
            email TEXT,
            relationship TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS preferences (
            key TEXT PRIMARY KEY,
            value TEXT,
            category TEXT
        );
        """)
        self._conn.commit()

    # ── Retrieval ─────────────────────────────────────────────────

    def retrieve_context(self, query: str) -> str:
        """
        Smart context retrieval: match query intent to the right table.
        Returns a formatted context string to inject into the edge LLM prompt.
        """
        q = query.lower()
        parts = []

        if any(w in q for w in ["medication", "medicine", "drug", "prescription", "pill", "dose"]):
            parts.append(self._get_medications())

        if any(w in q for w in ["allerg", "intoleran", "cannot eat", "can't eat"]):
            parts.append(self._get_allergies())

        if any(w in q for w in ["gate", "flight", "boarding", "depart", "arrival", "seat"]):
            parts.append(self._get_flight_bookings())

        if any(w in q for w in ["appointment", "doctor", "dentist", "meeting", "schedule"]):
            parts.append(self._get_appointments())

        if any(w in q for w in ["passport", "id card", "identity", "insurance"]):
            parts.append(self._get_documents())

        if any(w in q for w in ["contact", "phone", "call", "email"]):
            parts.append(self._get_contacts(query))

        if any(w in q for w in ["health", "condition", "diagnos", "blood pressure", "diabetes"]):
            parts.append(self._get_health_conditions())

        if not parts:
            # Generic: return preferences
            parts.append(self._get_preferences())

        context = "\n".join(p for p in parts if p)
        return context if context else "No relevant personal context found."

    def _get_medications(self) -> str:
        rows = self._conn.execute(
            "SELECT name, details, notes FROM health_profile WHERE category='medication'"
        ).fetchall()
        if not rows:
            return ""
        lines = [f"- {r['name']}: {r['details'] or ''} {r['notes'] or ''}".strip() for r in rows]
        return "Medications:\n" + "\n".join(lines)

    def _get_allergies(self) -> str:
        rows = self._conn.execute(
            "SELECT name, details FROM health_profile WHERE category='allergy'"
        ).fetchall()
        if not rows:
            return ""
        items = ", ".join(r["name"] for r in rows)
        return f"Known allergies: {items}"

    def _get_health_conditions(self) -> str:
        rows = self._conn.execute(
            "SELECT name, details FROM health_profile WHERE category='condition'"
        ).fetchall()
        if not rows:
            return ""
        lines = [f"- {r['name']}: {r['details'] or ''}".strip() for r in rows]
        return "Health conditions:\n" + "\n".join(lines)

    def _get_flight_bookings(self) -> str:
        rows = self._conn.execute(
            "SELECT title, reference, location, datetime_str, gate, seat, notes "
            "FROM bookings WHERE type='flight' ORDER BY datetime_str DESC LIMIT 3"
        ).fetchall()
        if not rows:
            return ""
        parts = []
        for r in rows:
            line = f"- {r['title']} ({r['reference'] or 'no ref'})"
            if r["datetime_str"]:
                line += f" at {r['datetime_str']}"
            if r["gate"]:
                line += f", Gate {r['gate']}"
            if r["seat"]:
                line += f", Seat {r['seat']}"
            if r["location"]:
                line += f" — {r['location']}"
            parts.append(line)
        return "Flight bookings:\n" + "\n".join(parts)

    def _get_appointments(self) -> str:
        rows = self._conn.execute(
            "SELECT title, location, datetime_str, notes "
            "FROM bookings WHERE type='appointment' ORDER BY datetime_str LIMIT 5"
        ).fetchall()
        if not rows:
            return ""
        lines = []
        for r in rows:
            line = f"- {r['title']}"
            if r["datetime_str"]:
                line += f" on {r['datetime_str']}"
            if r["location"]:
                line += f" at {r['location']}"
            lines.append(line)
        return "Appointments:\n" + "\n".join(lines)

    def _get_documents(self) -> str:
        rows = self._conn.execute(
            "SELECT doc_type, label, masked_value, expiry FROM documents"
        ).fetchall()
        if not rows:
            return ""
        lines = [
            f"- {r['doc_type'].capitalize()}: {r['label']} "
            f"({r['masked_value'] or 'on file'}"
            f"{', expires ' + r['expiry'] if r['expiry'] else ''})"
            for r in rows
        ]
        return "Documents on file:\n" + "\n".join(lines)

    def _get_contacts(self, query: str) -> str:
        # Simple name extraction from query
        rows = self._conn.execute("SELECT name, phone, email, relationship FROM contacts LIMIT 10").fetchall()
        if not rows:
            return ""
        lines = [
            f"- {r['name']} ({r['relationship'] or 'contact'}): {r['phone'] or r['email'] or 'no contact info'}"
            for r in rows
        ]
        return "Contacts:\n" + "\n".join(lines)

    def _get_preferences(self) -> str:
        rows = self._conn.execute("SELECT key, value FROM preferences LIMIT 10").fetchall()
        if not rows:
            return ""
        lines = [f"- {r['key']}: {r['value']}" for r in rows]
        return "User preferences:\n" + "\n".join(lines)

    # ── Write API ─────────────────────────────────────────────────

    def add_medication(self, name: str, details: str, notes: str = ""):
        self._conn.execute(
            "INSERT INTO health_profile (category, name, details, notes) VALUES (?,?,?,?)",
            ("medication", name, details, notes),
        )
        self._conn.commit()

    def add_allergy(self, name: str, details: str = ""):
        self._conn.execute(
            "INSERT INTO health_profile (category, name, details) VALUES (?,?,?)",
            ("allergy", name, details),
        )
        self._conn.commit()

    def add_health_condition(self, name: str, details: str = ""):
        self._conn.execute(
            "INSERT INTO health_profile (category, name, details) VALUES (?,?,?)",
            ("condition", name, details),
        )
        self._conn.commit()

    def add_flight(self, title: str, reference: str, departure: str,
                   gate: str = None, seat: str = None, destination: str = None):
        self._conn.execute(
            "INSERT INTO bookings (type, title, reference, datetime_str, gate, seat, location) "
            "VALUES (?,?,?,?,?,?,?)",
            ("flight", title, reference, departure, gate, seat, destination),
        )
        self._conn.commit()

    def add_appointment(self, title: str, when: str, location: str = "", notes: str = ""):
        self._conn.execute(
            "INSERT INTO bookings (type, title, datetime_str, location, notes) VALUES (?,?,?,?,?)",
            ("appointment", title, when, location, notes),
        )
        self._conn.commit()

    def add_document(self, doc_type: str, label: str, masked_value: str = "", expiry: str = ""):
        self._conn.execute(
            "INSERT INTO documents (doc_type, label, masked_value, expiry) VALUES (?,?,?,?)",
            (doc_type, label, masked_value, expiry),
        )
        self._conn.commit()

    def add_contact(self, name: str, phone: str = "", email: str = "", relationship: str = ""):
        self._conn.execute(
            "INSERT INTO contacts (name, phone, email, relationship) VALUES (?,?,?,?)",
            (name, phone, email, relationship),
        )
        self._conn.commit()

    def set_preference(self, key: str, value: str, category: str = "general"):
        self._conn.execute(
            "INSERT OR REPLACE INTO preferences (key, value, category) VALUES (?,?,?)",
            (key, value, category),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()
