"""
fog/server.py — Fog Node Server
================================
Run this on your FOG SERVER (GPU machine), NOT on your laptop.

This is an HTTP REST wrapper around Ollama's local inference.
The edge device (laptop) sends HTTP requests here.

Start the fog server:
  # On your fog/GPU server:
  pip install fastapi uvicorn httpx
  python fog/server.py

  # Or with uvicorn directly:
  uvicorn fog.server:app --host 0.0.0.0 --port 11435

Then update config.yaml:
  fog_server_url: "http://<your-server-ip>:11435"

Security note: In production, add API key authentication.
For dev/simulation, this runs without auth.
"""

import logging
import os
import time
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import httpx
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
FOG_MODEL  = os.environ.get("FOG_MODEL", "llama3.2-vision:11b")
HOST       = os.environ.get("FOG_HOST", "0.0.0.0")
PORT       = int(os.environ.get("FOG_PORT", "11435"))

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Semantic Router — Fog Node",
        description="Environmental LLM inference endpoint for the privacy-aware routing system.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # restrict to edge IP in production
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    class InferenceRequest(BaseModel):
        query: str
        context: Optional[str] = ""
        image_b64: Optional[str] = None
        model: Optional[str] = None
        max_tokens: Optional[int] = 400

    class InferenceResponse(BaseModel):
        response: str
        model: str
        latency_ms: float
        fog_node: str = "fog-server-01"

    @app.get("/health")
    async def health():
        """Health check — also verifies Ollama is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{OLLAMA_URL}/api/tags")
                ollama_ok = r.status_code == 200
        except Exception:
            ollama_ok = False
        return {
            "status": "ok",
            "ollama_reachable": ollama_ok,
            "fog_model": FOG_MODEL,
            "ollama_url": OLLAMA_URL,
        }

    @app.post("/generate", response_model=InferenceResponse)
    async def generate(req: InferenceRequest):
        """Main inference endpoint called by edge dispatcher."""
        t0 = time.perf_counter()
        model = req.model or FOG_MODEL

        system_prompt = (
            "You are an environmental intelligence assistant on a nearby GPU server. "
            "You help visually impaired users understand their physical surroundings. "
            "You do NOT have access to any personal user data — only environmental queries reach you. "
            "Be precise, spatial, and concise."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if req.context:
            messages[0]["content"] += f"\n\nEnvironmental knowledge base:\n{req.context}"

        user_msg: dict = {"role": "user", "content": req.query}
        if req.image_b64:
            user_msg["images"] = [req.image_b64]
        messages.append(user_msg)

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": req.max_tokens},
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                response_text = data["message"]["content"]
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Ollama at {OLLAMA_URL}. Run: ollama serve"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        latency = (time.perf_counter() - t0) * 1000
        logger.info("Fog inference: %.0fms | query='%s...'", latency, req.query[:50])

        return InferenceResponse(
            response=response_text,
            model=model,
            latency_ms=latency,
        )

    @app.get("/models")
    async def list_models():
        """List available models on the fog Ollama instance."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{OLLAMA_URL}/api/tags")
                return r.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))


def main():
    if not FASTAPI_AVAILABLE:
        print("ERROR: FastAPI not installed on fog server.")
        print("Run: pip install fastapi uvicorn httpx")
        return

    print(f"""
╔══════════════════════════════════════════════════════════╗
║         Semantic Router — Fog Node Server                ║
╠══════════════════════════════════════════════════════════╣
║  Listening on: http://{HOST}:{PORT}                    ║
║  Ollama URL:   {OLLAMA_URL:<40}║
║  Model:        {FOG_MODEL:<40}║
╠══════════════════════════════════════════════════════════╣
║  Health:   GET  /health                                  ║
║  Generate: POST /generate                                ║
╚══════════════════════════════════════════════════════════╝

Set on edge (config.yaml):  fog_server_url: "http://<this-server-ip>:{PORT}"
""")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
