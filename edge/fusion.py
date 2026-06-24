"""
edge/fusion.py — C6: Response Fusion
=====================================
Fuses edge and fog responses into a single coherent answer.
Runs on-device (edge) using the edge SLM.

Privacy guarantee: fusion runs on edge, so personal context
from edge_response never needs to travel to fog.

Two fusion modes:
  - prompt_fusion: ask edge SLM to merge both answers
  - template_fusion: simple rule-based merge (faster, no LLM call)
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Templates for common fusion patterns
FUSION_TEMPLATES = {
    "parallel": (
        "Regarding your personal information: {edge}\n\n"
        "Regarding your surroundings: {fog}"
    ),
    "sequential": (
        "{edge}\n\nBased on this, here is the environmental information you need: {fog}"
    ),
    "personal_only": "{edge}",
    "fog_only": "{fog}",
}


class ResponseFuser:
    """
    Fuses edge and fog responses.
    
    Usage:
        fuser = ResponseFuser(edge_client=edge_model)
        final = await fuser.fuse(
            edge_response="Your gate is D34.",
            fog_response="Gate D34 is 80m ahead, turn left.",
            route="mixed_sequential",
            original_query="Where is my gate?",
        )
    """

    def __init__(self, edge_client=None, use_llm_fusion: bool = False):
        """
        Args:
            edge_client: EdgeModelClient (used only if use_llm_fusion=True)
            use_llm_fusion: If True, prompt the edge SLM to write the fusion.
                            If False, use template fusion (faster, more predictable).
        """
        self.edge_client = edge_client
        self.use_llm_fusion = use_llm_fusion

    async def fuse(
        self,
        edge_response: Optional[str],
        fog_response: Optional[str],
        route: str,
        original_query: str = "",
    ) -> "FusionResult":
        t0 = time.perf_counter()

        # Single-tier routes: no fusion needed
        if route == "edge_only" or fog_response is None:
            return FusionResult(
                text=edge_response or "",
                method="passthrough",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        if route == "fog_only" or edge_response is None:
            return FusionResult(
                text=fog_response or "",
                method="passthrough",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Mixed: fuse both
        if self.use_llm_fusion and self.edge_client:
            return await self._llm_fuse(edge_response, fog_response, route, original_query, t0)
        else:
            return self._template_fuse(edge_response, fog_response, route, t0)

    def _template_fuse(
        self, edge: str, fog: str, route: str, t0: float
    ) -> "FusionResult":
        mode = "sequential" if "sequential" in route else "parallel"
        template = FUSION_TEMPLATES.get(mode, FUSION_TEMPLATES["parallel"])
        text = template.format(edge=edge.strip(), fog=fog.strip())
        return FusionResult(
            text=text,
            method=f"template_{mode}",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    async def _llm_fuse(
        self, edge: str, fog: str, route: str, original_query: str, t0: float
    ) -> "FusionResult":
        prompt = (
            f"The user asked: \"{original_query}\"\n\n"
            f"Personal information (from on-device): {edge}\n\n"
            f"Environmental information (from nearby server): {fog}\n\n"
            f"Combine these into a single, natural, helpful answer for a visually impaired user. "
            f"Be concise. Do not repeat information unnecessarily."
        )
        try:
            fused = await self.edge_client.generate_async(prompt)
            return FusionResult(
                text=fused,
                method="llm_fusion",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            logger.warning("LLM fusion failed, falling back to template: %s", e)
            return self._template_fuse(edge, fog, route, t0)


class FusionResult:
    def __init__(self, text: str, method: str, latency_ms: float):
        self.text = text
        self.method = method
        self.latency_ms = latency_ms

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return f"FusionResult(method={self.method!r}, latency={self.latency_ms:.1f}ms, text={preview!r}...)"
