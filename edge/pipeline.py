"""
edge/pipeline.py — End-to-End Pipeline Orchestrator
=====================================================
This is the main entry point for a single query.
Runs entirely on the edge device (laptop/smartphone).

Pipeline:
  1. Session injection check
  2. C1 Classification
  3. Context retrieval (edge or fog store)
  4. Route:
     a. Personal → edge only
     b. Environmental → fog only
     c. Mixed → C2 Decompose → C3 Dependency → C4 Dispatch
  5. C6 Fusion (for mixed)
  6. Session update

Usage:
    pipeline = RoutingPipeline.from_config()
    result = await pipeline.process("What medication am I taking and is there a pharmacy nearby?")
    print(result.final_response)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tiergraph.executor import GraphExecutionResult


@dataclass
class PipelineResult:
    query: str
    classification: object          # ClassificationResult
    final_response: str
    route: str                      # edge_only | fog_only | mixed_parallel | mixed_sequential
    total_latency_ms: float
    decomposition: object = None    # DecompositionResult (if Mixed)
    dependency: object = None       # DependencyResult (if Mixed)
    dispatch: object = None         # DispatchResult
    fusion: object = None           # FusionResult (if Mixed)
    session_injected: bool = False
    context_used: str = ""

    def summary(self) -> str:
        lines = [
            f"Query:        {self.query}",
            f"Label:        {self.classification.label} (conf={self.classification.confidence:.2f}, via={self.classification.triggered_by})",
            f"Route:        {self.route}",
            f"Latency:      {self.total_latency_ms:.0f}ms",
            f"Session inj:  {self.session_injected}",
        ]
        if self.decomposition:
            lines += [
                f"Personal sub: {self.decomposition.personal_subquery}",
                f"Environ sub:  {self.decomposition.environmental_subquery}",
                f"Dependency:   {self.dependency.mode if self.dependency else 'N/A'}",
            ]
        lines.append(f"\nResponse:\n{self.final_response}")
        return "\n".join(lines)


@dataclass
class TierGraphPipelineResult:
    """Compatibility wrapper that preserves the research GraphExecutionResult."""

    graph_result: "GraphExecutionResult"
    pipeline_result: PipelineResult


class RoutingPipeline:
    """
    Full end-to-end routing pipeline.
    
    Construct via RoutingPipeline.from_config() for production,
    or inject components directly for testing.
    """

    def __init__(
        self,
        classifier,
        decomposer,
        dependency_detector,
        dispatcher,
        fuser,
        edge_store,
        fog_store,
        session_manager,
        *,
        tiergraph_enabled: bool = False,
        graph_executor=None,
    ):
        self.classifier = classifier
        self.decomposer = decomposer
        self.dep_detector = dependency_detector
        self.dispatcher = dispatcher
        self.fuser = fuser
        self.edge_store = edge_store
        self.fog_store = fog_store
        self.session = session_manager
        self.tiergraph_enabled = tiergraph_enabled
        self.graph_executor = graph_executor

    @classmethod
    def from_config(cls, config_override: dict = None):
        """
        Construct the full pipeline from config.yaml.
        Use config_override={'simulation_mode': False} for production.
        TierGraph oracle execution is opt-in via tiergraph_enabled=True.
        """
        try:
            from config import cfg
        except ImportError:
            cfg = {}

        if config_override:
            cfg.update(config_override)

        sim_mode = cfg.get("simulation_mode", True)
        tiergraph_enabled = bool(cfg.get("tiergraph_enabled", False))

        from router.classifier import QueryClassifier
        from router.decomposer import MixedQueryDecomposer
        from router.dependency import DependencyDetector
        from router.dispatch import RoutingDispatcher
        from edge.model import EdgeModelClient, FogModelClient
        from edge.fusion import ResponseFuser
        from edge.session import SessionManager
        from context_store.edge_store import EdgeContextStore
        from context_store.fog_store import FogContextStore

        edge_client = EdgeModelClient(
            base_url=cfg.get("edge_ollama_url", "http://localhost:11434"),
            model=cfg.get("edge_model", "llama3.2:3b"),
            timeout=cfg.get("edge_timeout_sec", 20.0),
            simulation_mode=sim_mode,
        )
        fog_client = FogModelClient(
            base_url=cfg.get("fog_server_url", "http://localhost:11435"),
            model=cfg.get("fog_model", "llama3.2-vision:11b"),
            timeout=cfg.get("fog_timeout_sec", 30.0),
            simulation_mode=sim_mode,
        )

        classifier = QueryClassifier(
            encoder_name=cfg.get("classifier_model", "all-MiniLM-L6-v2")
        )
        classifier.load_or_train(dataset_path="dataset/training_data.json")

        decomposer     = MixedQueryDecomposer(slm_client=edge_client if not sim_mode else None)
        dep_detector   = DependencyDetector()
        dispatcher     = RoutingDispatcher(edge_client=edge_client, fog_client=fog_client)
        fuser          = ResponseFuser(edge_client=edge_client, use_llm_fusion=True)
        edge_store     = EdgeContextStore(db_path=cfg.get("edge_db_path", "data/edge_context.db"))
        fog_store      = FogContextStore(
            index_path=cfg.get("fog_faiss_path", "data/fog_index.faiss"),
            metadata_path=cfg.get("fog_metadata_path", "data/fog_metadata.json"),
        )
        session_mgr    = SessionManager(buffer_size=cfg.get("session_buffer_size", 10))

        graph_executor = None
        if tiergraph_enabled:
            from tiergraph import GraphExecutor

            graph_executor = GraphExecutor(
                edge_client=edge_client,
                fog_client=fog_client,
                edge_context_fn=edge_store.retrieve_context,
                fog_context_fn=fog_store.retrieve,
            )

        return cls(
            classifier=classifier,
            decomposer=decomposer,
            dependency_detector=dep_detector,
            dispatcher=dispatcher,
            fuser=fuser,
            edge_store=edge_store,
            fog_store=fog_store,
            session_manager=session_mgr,
            tiergraph_enabled=tiergraph_enabled,
            graph_executor=graph_executor,
        )

    # ── Main Entry Point ──────────────────────────────────────────

    async def process(
        self,
        query: str,
        image_b64: str = None,
    ) -> PipelineResult:
        t0 = time.perf_counter()
        has_image = image_b64 is not None

        # Step 1: Session injection
        enriched_query, session_injected = self.session.maybe_inject(query, has_image=has_image)
        query_for_classification = query  # classify on original, not enriched
        query_for_inference = enriched_query

        # Step 2: Classify
        clf_result = self.classifier.predict(query_for_classification)
        label = clf_result.label
        logger.info("Classified: %s → %s (%.2f)", query[:60], label, clf_result.confidence)

        # Step 3: Route
        if label == "Personal":
            context = self.edge_store.retrieve_context(query)
            dispatch_result = await self.dispatcher.dispatch_personal(
                query_for_inference, context=context, image_b64=image_b64
            )
            fusion_result = await self.fuser.fuse(
                dispatch_result.edge_response, None, "edge_only", query
            )
            route = "edge_only"
            decomp_result = None
            dep_result = None

        elif label == "Environmental":
            context = self.fog_store.retrieve(query)
            dispatch_result = await self.dispatcher.dispatch_environmental(
                query_for_inference, context=context, image_b64=image_b64
            )
            fusion_result = await self.fuser.fuse(
                None, dispatch_result.fog_response, "fog_only", query
            )
            route = "fog_only"
            decomp_result = None
            dep_result = None

        else:  # Mixed
            # Step 3a: Decompose
            decomp_result = self.decomposer.decompose(query)
            logger.info("Decomposed: personal='%s' | env='%s'",
                        decomp_result.personal_subquery[:50],
                        decomp_result.environmental_subquery[:50])

            # Step 3b: Dependency detection
            dep_result = self.dep_detector.detect(
                decomp_result.personal_subquery,
                decomp_result.environmental_subquery,
                implicit_personal=decomp_result.implicit_personal,
            )
            logger.info("Dependency: %s (%s)", dep_result.mode, dep_result.reason[:60])

            # Step 3c: Dispatch
            personal_context = self.edge_store.retrieve_context(decomp_result.personal_subquery)
            env_context = self.fog_store.retrieve(decomp_result.environmental_subquery)

            if dep_result.is_sequential():
                dispatch_result = await self.dispatcher.dispatch_mixed_sequential(
                    decomp_result.personal_subquery,
                    decomp_result.environmental_subquery,
                    personal_context=personal_context,
                    env_context=env_context,
                    image_b64=image_b64,
                )
                route = "mixed_sequential"
            else:
                dispatch_result = await self.dispatcher.dispatch_mixed_parallel(
                    decomp_result.personal_subquery,
                    decomp_result.environmental_subquery,
                    personal_context=personal_context,
                    env_context=env_context,
                    image_b64=image_b64,
                )
                route = "mixed_parallel"

            # Step 4: Fuse
            fusion_result = await self.fuser.fuse(
                dispatch_result.edge_response,
                dispatch_result.fog_response,
                route,
                original_query=query,
            )

        # Step 5: Update session
        self.session.add_turn(
            query=query,
            label=label,
            response=fusion_result.text[:300],  # truncate for buffer
        )

        total_ms = (time.perf_counter() - t0) * 1000

        return PipelineResult(
            query=query,
            classification=clf_result,
            final_response=fusion_result.text,
            route=route,
            total_latency_ms=total_ms,
            decomposition=decomp_result,
            dependency=dep_result,
            dispatch=dispatch_result,
            fusion=fusion_result,
            session_injected=session_injected,
        )

    async def process_execution_graph(
        self,
        graph,
        *,
        image_b64: str = None,
        fusion_plan=None,
    ) -> TierGraphPipelineResult:
        """
        Opt-in oracle path: execute a preconstructed ExecutionGraph.

        Replaces legacy decomposer → dependency detector → mixed dispatcher
        for this call only. Does not predict graphs from free text.
        """
        if not self.tiergraph_enabled or self.graph_executor is None:
            raise RuntimeError(
                "TierGraph execution requires tiergraph_enabled=True "
                "and a configured graph_executor"
            )

        from tiergraph import GraphExecutionResult

        graph_result: GraphExecutionResult = await self.graph_executor.execute(
            graph,
            image_b64=image_b64,
            fusion_plan=fusion_plan,
        )

        route = _route_from_graph_result(graph_result)
        pipeline_result = PipelineResult(
            query=graph_result.original_query,
            classification=_SyntheticClassification(graph_result.query_type.value),
            final_response=graph_result.final_response,
            route=route,
            total_latency_ms=graph_result.total_latency_ms,
            decomposition=None,
            dependency=None,
            dispatch=None,
            fusion=_SyntheticFusion(
                text=graph_result.final_response,
                method=graph_result.fusion_method or "tiergraph_sink",
                latency_ms=0.0,
            ),
            session_injected=False,
        )
        self.session.add_turn(
            query=graph_result.original_query,
            label=graph_result.query_type.value,
            response=graph_result.final_response[:300],
        )
        return TierGraphPipelineResult(
            graph_result=graph_result,
            pipeline_result=pipeline_result,
        )

    def process_sync(self, query: str, image_b64: str = None) -> PipelineResult:
        """Synchronous wrapper for convenience."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.process(query, image_b64=image_b64))
        finally:
            loop.close()


class _SyntheticClassification:
    def __init__(self, label: str):
        self.label = label
        self.confidence = 1.0
        self.triggered_by = "tiergraph_execution_graph"


class _SyntheticFusion:
    def __init__(self, text: str, method: str, latency_ms: float):
        self.text = text
        self.method = method
        self.latency_ms = latency_ms


def _route_from_graph_result(graph_result) -> str:
    mode = graph_result.execution_mode
    tiers = {result.tier.value for result in graph_result.results.values()}
    if tiers == {"edge"}:
        return "edge_only"
    if tiers == {"fog"}:
        return "fog_only"
    if mode == "sequential":
        return "mixed_sequential"
    if mode == "parallel":
        return "mixed_parallel"
    return "mixed_hybrid"
