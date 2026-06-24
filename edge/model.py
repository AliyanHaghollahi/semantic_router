"""
edge/model.py — C5 Edge: Model Client
======================================
Wraps Ollama running on localhost (your laptop).
Model: llama3.2:3b (quantized, ~2GB)

In simulation_mode=True, returns mock responses without Ollama.
"""

import asyncio
import json
import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class EdgeModelClient:
    """
    Client for the edge LLM (Ollama on laptop).
    
    Install Ollama: https://ollama.ai
    Pull model: ollama pull llama3.2:3b
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        timeout: float = 20.0,
        simulation_mode: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.simulation_mode = simulation_mode

    # ── Sync wrapper ──────────────────────────────────────────────

    def generate(self, query: str, context: str = "", image_b64: str = None) -> str:
        return asyncio.get_event_loop().run_until_complete(
            self.generate_async(query, context=context, image_b64=image_b64)
        )

    # ── Async (used by dispatcher) ────────────────────────────────

    async def generate_async(
        self, query: str, context: str = "", image_b64: str = None
    ) -> str:

        if self.simulation_mode:
            return self._simulate(query, context)

        return await self._call_ollama(query, context, image_b64)

    # ── Ollama API ────────────────────────────────────────────────

    async def _call_ollama(
        self, query: str, context: str, image_b64: Optional[str]
    ) -> str:
        system_prompt = (
            "You are a privacy-first personal assistant running on the user's smartphone. "
            "You have access to the user's personal data (health, documents, bookings, contacts). "
            "Answer concisely and accurately. Never speculate about external locations or surroundings."
        )

        messages = []
        if context:
            messages.append({"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}"})
        else:
            messages.append({"role": "system", "content": system_prompt})

        user_content = query
        if image_b64:
            # Phi-3.5 Vision / llava format
            messages.append({
                "role": "user",
                "content": user_content,
                "images": [image_b64],
            })
        else:
            messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat", json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"]
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            raise RuntimeError(f"Edge model error: {e}")

    # ── Simulation ────────────────────────────────────────────────

    def _simulate(self, query: str, context: str) -> str:
        """Return realistic mock responses for simulation/testing."""
        await_delay = 0.3  # simulate inference time
        time.sleep(await_delay)

        q_lower = query.lower()

        if "medication" in q_lower or "blood pressure" in q_lower:
            return (
                "Your current medication is Lisinopril 10mg, taken once daily in the morning. "
                "You also take Metformin 500mg twice daily with meals."
            )
        elif "gate" in q_lower and ("number" in q_lower or "what is" in q_lower):
            return "Your flight UA447 departs from Gate D34."
        elif "gate" in q_lower:
            return "Your boarding pass shows Gate D34 for flight UA447 to Dallas at 3:45 PM."
        elif "appointment" in q_lower or "schedule" in q_lower:
            return "Your next appointment is tomorrow at 2:00 PM — Dr. Martinez, cardiology follow-up."
        elif "passport" in q_lower:
            return "Your passport number is stored securely. Do you need me to display it privately?"
        elif "insurance" in q_lower:
            return "Your insurance is BlueCross BlueShield, plan ID BCB-4892-X."
        elif "allerg" in q_lower:
            return "According to your health profile, you are allergic to penicillin and shellfish."
        elif "contact" in q_lower:
            return "I found 3 contacts matching that name in your phone."
        else:
            return f"[Edge/Personal] Processed query: '{query}'. (Simulation mode — connect Ollama for real inference.)"


class FogModelClient:
    """
    Client for the fog LLM (Ollama on remote GPU server).
    
    On fog server, run:
      ollama pull llama3.2-vision:11b
      ollama serve --host 0.0.0.0
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11435",
        model: str = "llama3.2-vision:11b",
        timeout: float = 30.0,
        simulation_mode: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.simulation_mode = simulation_mode

    def generate(self, query: str, context: str = "", image_b64: str = None) -> str:
        return asyncio.get_event_loop().run_until_complete(
            self.generate_async(query, context=context, image_b64=image_b64)
        )

    async def generate_async(
        self, query: str, context: str = "", image_b64: str = None
    ) -> str:

        if self.simulation_mode:
            return self._simulate(query, context)

        return await self._call_ollama(query, context, image_b64)

    async def _call_ollama(
        self, query: str, context: str, image_b64: Optional[str]
    ) -> str:
        system_prompt = (
            "You are an environmental intelligence assistant running on a nearby GPU server. "
            "You help visually impaired users understand their physical surroundings. "
            "You do NOT have access to personal user data. "
            "Describe locations, objects, signs, and spatial context clearly and concisely."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if context:
            messages[0]["content"] += f"\n\nEnvironmental context:\n{context}"

        if image_b64:
            messages.append({
                "role": "user",
                "content": query,
                "images": [image_b64],
            })
        else:
            messages.append({"role": "user", "content": query})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 400},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat", json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"]
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Fog Ollama at {self.base_url}. "
                "Check fog server address in config.yaml."
            )
        except Exception as e:
            raise RuntimeError(f"Fog model error: {e}")

    def _simulate(self, query: str, context: str) -> str:
        time.sleep(0.5)  # simulate network + inference
        q_lower = query.lower()

        if "pharmacy" in q_lower:
            return (
                "There is a CVS Pharmacy approximately 120 meters ahead on your right, "
                "on the corner of 5th Avenue and Main Street. It is currently open until 9 PM."
            )
        elif "gate" in q_lower and "where" in q_lower:
            # Extract gate number from injected context
            gate = "D34"
            if "gate" in context.lower():
                import re
                m = re.search(r"gate\s+([A-Z]?\d+)", context, re.IGNORECASE)
                if m:
                    gate = m.group(1).upper()
            return (
                f"Gate {gate} is located in Terminal D. From your current position, "
                f"walk straight for 80 meters, then turn left at the main atrium. "
                f"Gate {gate} will be on your right. Estimated walking time: 4 minutes."
            )
        elif "sign" in q_lower or "what does" in q_lower:
            return (
                "The sign reads: 'Emergency Exit — Keep Clear'. "
                "It is mounted on a green background with white text, "
                "located above the door to your left."
            )
        elif "restaurant" in q_lower or "food" in q_lower or "eat" in q_lower:
            return (
                "I can see a café directly ahead called 'The Corner Brasserie'. "
                "There is also a sandwich counter on your right side of the corridor."
            )
        elif "holding" in q_lower or "this" in q_lower or "object" in q_lower:
            return (
                "You appear to be holding a small orange prescription bottle. "
                "The label is facing away from the camera — would you like to rotate it?"
            )
        elif "building" in q_lower or "where am i" in q_lower:
            return (
                "You are in the departures hall of Terminal D, near the security checkpoint. "
                "The main food court is 50 meters ahead."
            )
        else:
            if context and "gate" in context.lower():
                return (
                    f"Based on your personal information, I can help you navigate to the indicated location. "
                    f"The area is clearly marked with standard airport signage."
                )
            return f"[Fog/Environmental] Processed query: '{query}'. (Simulation mode — connect fog Ollama for real inference.)"
