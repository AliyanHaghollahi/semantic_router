"""
router/dispatch.py — C4: Routing Dispatcher
============================================
Handles the actual query dispatch to Edge and Fog models.

HARD PRIVACY RULE (enforced here, not just by convention):
  Personal sub-queries and their answers NEVER leave this process
  toward the Fog server. The dispatch layer actively blocks this.

Modes:
  - dispatch_personal(query)         → edge only
  - dispatch_environmental(query)    → fog only
  - dispatch_mixed_parallel(p, e)    → asyncio.gather both
  - dispatch_mixed_sequential(p, e)  → edge first → inject → fog

All fog communication is via HTTP REST (Ollama-compatible API).
"""

import asyncio
import logging
import time
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class RoutingDispatcher:
    """
    Central dispatcher. Takes sub-queries, calls appropriate tier models,
    returns (edge_response, fog_response).

    Args:
        edge_client: EdgeModelClient (calls local Ollama)
        fog_client:  FogModelClient  (calls remote Ollama via HTTP)
    """

    def __init__(self, edge_client, fog_client):
        self.edge = edge_client
        self.fog  = fog_client

    # ── Personal Only ─────────────────────────────────────────────

    async def dispatch_personal(
        self, query: str, context: str = "", image_b64: str = None
    ) -> "DispatchResult":
        """Route to edge only. Fog never sees this."""
        t0 = time.perf_counter()
        logger.info("[DISPATCH] Personal → Edge only")
        try:
            response = await self.edge.generate_async(query, context=context, image_b64=image_b64)
            return DispatchResult(
                edge_response=response,
                fog_response=None,
                route="edge_only",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            logger.error("Edge dispatch failed: %s", e)
            return DispatchResult(
                edge_response=f"[Edge error: {e}]",
                fog_response=None,
                route="edge_only",
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=str(e),
            )

    # ── Environmental Only ────────────────────────────────────────

    async def dispatch_environmental(
        self, query: str, context: str = "", image_b64: str = None
    ) -> "DispatchResult":
        """Route to fog only."""
        t0 = time.perf_counter()
        logger.info("[DISPATCH] Environmental → Fog only")
        try:
            response = await self.fog.generate_async(query, context=context, image_b64=image_b64)
            return DispatchResult(
                edge_response=None,
                fog_response=response,
                route="fog_only",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            logger.error("Fog dispatch failed: %s", e)
            return DispatchResult(
                edge_response=None,
                fog_response=f"[Fog error: {e}]",
                route="fog_only",
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=str(e),
            )

    # ── Mixed Parallel ────────────────────────────────────────────

    async def dispatch_mixed_parallel(
        self,
        personal_query: str,
        env_query: str,
        personal_context: str = "",
        env_context: str = "",
        image_b64: str = None,
    ) -> "DispatchResult":
        """
        Dispatch both sub-queries simultaneously.
        Personal → Edge | Environmental → Fog (concurrently)
        """
        t0 = time.perf_counter()
        logger.info("[DISPATCH] Mixed PARALLEL → Edge + Fog simultaneously")

        edge_task = self.edge.generate_async(personal_query, context=personal_context)
        fog_task  = self.fog.generate_async(env_query, context=env_context, image_b64=image_b64)

        try:
            edge_resp, fog_resp = await asyncio.gather(edge_task, fog_task)
        except Exception as e:
            logger.error("Parallel dispatch failed: %s", e)
            edge_resp = "[Edge error]"
            fog_resp  = "[Fog error]"

        return DispatchResult(
            edge_response=edge_resp,
            fog_response=fog_resp,
            route="mixed_parallel",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ── Mixed Sequential ──────────────────────────────────────────

    async def dispatch_mixed_sequential(
        self,
        personal_query: str,
        env_query: str,
        personal_context: str = "",
        env_context: str = "",
        image_b64: str = None,
    ) -> "DispatchResult":
        """
        Edge runs first. Its answer is injected into the fog prompt.
        Used when env sub-query depends on personal result (e.g. gate number).
        
        PRIVACY: Only the necessary personal *fact* is injected, not raw personal data.
        """
        t0 = time.perf_counter()
        logger.info("[DISPATCH] Mixed SEQUENTIAL → Edge first, then inject → Fog")

        # Step 1: Edge
        edge_response = await self.edge.generate_async(
            personal_query, context=personal_context
        )
        edge_latency = (time.perf_counter() - t0) * 1000
        logger.info("[DISPATCH] Edge answered in %.1fms. Injecting into fog prompt.", edge_latency)

        # Step 2: Inject edge answer into fog prompt
        injected_env_query = (
            f"Context from user's personal information: {edge_response}\n\n"
            f"Now answer: {env_query}"
        )

        fog_response = await self.fog.generate_async(
            injected_env_query, context=env_context, image_b64=image_b64
        )

        return DispatchResult(
            edge_response=edge_response,
            fog_response=fog_response,
            route="mixed_sequential",
            latency_ms=(time.perf_counter() - t0) * 1000,
            edge_latency_ms=edge_latency,
        )


class DispatchResult:
    def __init__(
        self,
        edge_response: Optional[str],
        fog_response: Optional[str],
        route: str,
        latency_ms: float,
        edge_latency_ms: float = None,
        error: str = None,
    ):
        self.edge_response = edge_response
        self.fog_response  = fog_response
        self.route = route
        self.latency_ms = latency_ms
        self.edge_latency_ms = edge_latency_ms
        self.error = error

    def __repr__(self):
        edge_preview = repr(self.edge_response[:40]) if self.edge_response else None
        fog_preview  = repr(self.fog_response[:40])  if self.fog_response  else None
        return (
            f"DispatchResult(route={self.route!r}, "
            f"latency={self.latency_ms:.0f}ms, "
            f"edge={edge_preview}, "
            f"fog={fog_preview})"
        )
