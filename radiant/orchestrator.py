"""
Pipeline orchestrator for Radiant Agentic RAG.

Coordinates the execution of all pipeline agents in the correct order,
handling errors gracefully and tracking metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from radiant.config import AppConfig
from radiant.utils.metrics import RunMetrics
from radiant.llm.client import LLMClient, LocalNLPModels
from radiant.storage.redis_store import RedisVectorStore, StoredDoc
from radiant.storage.bm25_index import PersistentBM25Index
from radiant.utils.conversation import ConversationManager
from radiant.agents import (
    AgentContext,
    AnswerSynthesisAgent,
    BM25RetrievalAgent,
    CriticAgent,
    CrossEncoderRerankingAgent,
    DenseRetrievalAgent,
    HierarchicalAutoMergingAgent,
    PlanningAgent,
    QueryDecompositionAgent,
    QueryExpansionAgent,
    QueryRewriteAgent,
    RRFAgent,
    WebSearchAgent,
    new_agent_context,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a complete pipeline run."""
    
    answer: str
    context: AgentContext
    metrics: RunMetrics
    success: bool = True
    error: Optional[str] = None

    @property
    def run_id(self) -> str:
        return self.context.run_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "answer": self.answer,
            "success": self.success,
            "error": self.error,
            "original_query": self.context.original_query,
            "decomposed_queries": self.context.decomposed_queries,
            "num_retrieved_docs": len(self.context.reranked),
            "warnings": self.context.warnings,
            "metrics": self.metrics.to_dict(),
        }


class RAGOrchestrator:
    """
    Orchestrates the complete RAG pipeline.
    
    Coordinates all agents in the correct sequence:
        1. Planning (optional)
        2. Query processing (decomposition, rewrite, expansion)
        3. Retrieval (dense + sparse)
        4. Fusion (RRF)
        5. Post-retrieval (auto-merge, rerank)
        6. Generation (synthesis, critique)
    """

    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        local: LocalNLPModels,
        store: RedisVectorStore,
        bm25_index: PersistentBM25Index,
        conversation_manager: Optional[ConversationManager] = None,
    ) -> None:
        """
        Initialize the orchestrator.
        
        Args:
            config: Application configuration
            llm: LLM client for chat completions
            local: Local NLP models (embedding, cross-encoder)
            store: Redis vector store
            bm25_index: BM25 index for sparse retrieval
            conversation_manager: Optional conversation manager
        """
        self._config = config
        self._pipeline_config = config.pipeline
        self._conversation = conversation_manager

        # Initialize agents
        self._planning_agent = PlanningAgent(
            llm, 
            web_search_enabled=config.web_search.enabled
        )
        self._decomposition_agent = QueryDecompositionAgent(llm, config.query)
        self._rewrite_agent = QueryRewriteAgent(llm)
        self._expansion_agent = QueryExpansionAgent(llm, config.query)
        
        self._dense_retrieval = DenseRetrievalAgent(store, local, config.retrieval)
        self._bm25_retrieval = BM25RetrievalAgent(bm25_index, config.retrieval)
        self._rrf_agent = RRFAgent(config.retrieval)
        
        # Web search agent (conditionally enabled)
        self._web_search_agent = WebSearchAgent(llm, config.web_search) if config.web_search.enabled else None
        
        self._automerge_agent = HierarchicalAutoMergingAgent(store, config.automerge)
        self._rerank_agent = CrossEncoderRerankingAgent(local, config.rerank)
        
        self._synthesis_agent = AnswerSynthesisAgent(
            llm, config.synthesis, conversation_manager
        )
        self._critic_agent = CriticAgent(llm, config.critic)

        logger.info(f"RAG orchestrator initialized (web_search={'enabled' if config.web_search.enabled else 'disabled'})")

    def run(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        plan_override: Optional[Dict[str, bool]] = None,
        retrieval_mode: str = "hybrid",
    ) -> PipelineResult:
        """
        Execute the complete RAG pipeline.
        
        Args:
            query: User query
            conversation_id: Optional conversation ID for history
            plan_override: Optional override for pipeline plan
            retrieval_mode: Retrieval mode - "hybrid" (default), "dense", or "bm25"
            
        Returns:
            PipelineResult with answer and metadata
        """
        # Validate retrieval mode
        valid_modes = {"hybrid", "dense", "bm25"}
        if retrieval_mode not in valid_modes:
            logger.warning(f"Invalid retrieval_mode '{retrieval_mode}', using 'hybrid'")
            retrieval_mode = "hybrid"
        
        # Initialize context and metrics
        ctx = new_agent_context(query, conversation_id)
        metrics = RunMetrics(run_id=ctx.run_id)
        
        logger.info(f"[{ctx.run_id}] Starting pipeline for query: {query[:100]}... (mode={retrieval_mode})")

        try:
            # Load conversation history if available
            if self._conversation and conversation_id:
                self._conversation.load_conversation(conversation_id)
                ctx.conversation_history = self._conversation.get_history_for_synthesis()

            # Phase 1: Planning
            plan = self._run_planning(ctx, metrics, plan_override)

            # Phase 2: Query Processing
            queries = self._run_query_processing(ctx, metrics, plan)

            # Phase 3: Retrieval
            self._run_retrieval(ctx, metrics, queries, plan, retrieval_mode)

            # Phase 4: Post-retrieval Processing
            self._run_post_retrieval(ctx, metrics, plan)

            # Phase 5: Generation
            answer = self._run_generation(ctx, metrics, plan)

            # Record conversation turn
            if self._conversation and conversation_id:
                self._conversation.add_user_query(query)
                self._conversation.add_assistant_response(answer)

            metrics.finish(
                query=query,
                answer_length=len(answer),
                num_docs_used=len(ctx.reranked),
            )

            return PipelineResult(
                answer=answer,
                context=ctx,
                metrics=metrics,
                success=True,
            )

        except Exception as e:
            logger.error(f"[{ctx.run_id}] Pipeline failed: {e}", exc_info=True)
            metrics.finish(error=str(e))
            
            return PipelineResult(
                answer=f"I apologize, but I encountered an error processing your query: {e}",
                context=ctx,
                metrics=metrics,
                success=False,
                error=str(e),
            )

    def _run_planning(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        plan_override: Optional[Dict[str, bool]],
    ) -> Dict[str, Any]:
        """Execute planning phase."""
        if plan_override:
            ctx.plan = plan_override
            return plan_override

        if not self._pipeline_config.use_planning:
            # Use default plan from pipeline config
            ctx.plan = {
                "use_decomposition": self._pipeline_config.use_decomposition,
                "use_rewrite": self._pipeline_config.use_rewrite,
                "use_expansion": self._pipeline_config.use_expansion,
                "use_rrf": self._pipeline_config.use_rrf,
                "use_automerge": self._pipeline_config.use_automerge,
                "use_rerank": self._pipeline_config.use_rerank,
                "use_critic": self._pipeline_config.use_critic,
            }
            return ctx.plan

        with metrics.track_step("PlanningAgent") as step:
            try:
                ctx.plan = self._planning_agent.run(ctx.original_query)
                step.extra["plan"] = ctx.plan
            except Exception as e:
                logger.warning(f"Planning failed, using defaults: {e}")
                metrics.mark_degraded("planning", str(e))
                ctx.plan = {
                    "use_decomposition": True,
                    "use_rewrite": True,
                    "use_expansion": True,
                    "use_rrf": True,
                    "use_automerge": True,
                    "use_rerank": True,
                    "use_critic": True,
                }

        return ctx.plan

    def _run_query_processing(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        plan: Dict[str, Any],
    ) -> List[str]:
        """Execute query processing phase."""
        queries = [ctx.original_query]

        # Query decomposition
        if plan.get("use_decomposition", True):
            with metrics.track_step("QueryDecompositionAgent") as step:
                try:
                    ctx.decomposed_queries = self._decomposition_agent.run(
                        ctx.original_query
                    )
                    queries = ctx.decomposed_queries
                    step.extra["num_queries"] = len(queries)
                except Exception as e:
                    logger.warning(f"Decomposition failed: {e}")
                    metrics.mark_degraded("decomposition", str(e))
                    ctx.decomposed_queries = [ctx.original_query]

        # Query rewriting
        if plan.get("use_rewrite", True):
            with metrics.track_step("QueryRewriteAgent") as step:
                try:
                    rewrites = []
                    for q in queries:
                        before, after = self._rewrite_agent.run(q)
                        rewrites.append((before, after))
                    ctx.rewrites = rewrites
                    queries = [after for _, after in rewrites]
                    step.extra["rewrites"] = len(rewrites)
                except Exception as e:
                    logger.warning(f"Rewrite failed: {e}")
                    metrics.mark_degraded("rewrite", str(e))

        # Query expansion
        if plan.get("use_expansion", True):
            with metrics.track_step("QueryExpansionAgent") as step:
                try:
                    all_expansions = []
                    for q in queries:
                        expansions = self._expansion_agent.run(q)
                        all_expansions.extend(expansions)
                    ctx.expansions = list(set(all_expansions))
                    step.extra["num_expansions"] = len(ctx.expansions)
                except Exception as e:
                    logger.warning(f"Expansion failed: {e}")
                    metrics.mark_degraded("expansion", str(e))

        return queries

    def _run_retrieval(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        queries: List[str],
        plan: Dict[str, Any],
        retrieval_mode: str = "hybrid",
    ) -> None:
        """Execute retrieval phase.
        
        Args:
            ctx: Agent context
            metrics: Metrics collector
            queries: Queries to search
            plan: Pipeline plan
            retrieval_mode: "hybrid", "dense", or "bm25"
        """
        # Combine queries with expansions for retrieval
        all_queries = queries + ctx.expansions

        # Dense retrieval (skip if bm25-only mode)
        if retrieval_mode in ("hybrid", "dense"):
            with metrics.track_step("DenseRetrievalAgent") as step:
                try:
                    all_dense: List[Tuple[StoredDoc, float]] = []
                    for q in all_queries:
                        results = self._dense_retrieval.run(q)
                        all_dense.extend(results)
                    
                    # Deduplicate, keeping best score
                    best_by_id: Dict[str, Tuple[StoredDoc, float]] = {}
                    for doc, score in all_dense:
                        existing = best_by_id.get(doc.doc_id)
                        if existing is None or score > existing[1]:
                            best_by_id[doc.doc_id] = (doc, score)
                    
                    ctx.dense_retrieved = sorted(
                        best_by_id.values(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    step.extra["num_retrieved"] = len(ctx.dense_retrieved)
                except Exception as e:
                    logger.error(f"Dense retrieval failed: {e}")
                    if retrieval_mode == "dense":
                        raise  # Fail if dense-only mode
                    # Continue with BM25 in hybrid mode
        else:
            step_info = metrics.track_step("DenseRetrievalAgent")
            step_info.__enter__()
            step_info.extra["skipped"] = True
            step_info.extra["reason"] = f"mode={retrieval_mode}"
            step_info.__exit__(None, None, None)

        # BM25 retrieval (skip if dense-only mode)
        if retrieval_mode in ("hybrid", "bm25"):
            with metrics.track_step("BM25RetrievalAgent") as step:
                try:
                    all_bm25: List[Tuple[StoredDoc, float]] = []
                    for q in all_queries:
                        results = self._bm25_retrieval.run(q)
                        all_bm25.extend(results)
                    
                    # Deduplicate
                    best_by_id: Dict[str, Tuple[StoredDoc, float]] = {}
                    for doc, score in all_bm25:
                        existing = best_by_id.get(doc.doc_id)
                        if existing is None or score > existing[1]:
                            best_by_id[doc.doc_id] = (doc, score)
                    
                    ctx.bm25_retrieved = sorted(
                        best_by_id.values(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    step.extra["num_retrieved"] = len(ctx.bm25_retrieved)
                except Exception as e:
                    logger.warning(f"BM25 retrieval failed: {e}")
                    if retrieval_mode == "bm25":
                        raise  # Fail if bm25-only mode
                    metrics.mark_degraded("bm25_retrieval", str(e))
        else:
            step_info = metrics.track_step("BM25RetrievalAgent")
            step_info.__enter__()
            step_info.extra["skipped"] = True
            step_info.extra["reason"] = f"mode={retrieval_mode}"
            step_info.__exit__(None, None, None)

        # Web search (if enabled and requested by plan)
        if self._web_search_agent and (plan.get("use_web_search", False) or 
            self._config.web_search.enabled):
            with metrics.track_step("WebSearchAgent") as step:
                try:
                    ctx.web_search_retrieved = self._web_search_agent.run(
                        ctx.original_query, plan
                    )
                    step.extra["num_retrieved"] = len(ctx.web_search_retrieved)
                    step.extra["urls_fetched"] = [
                        doc.meta.get("source_url", "unknown")
                        for doc, _ in ctx.web_search_retrieved
                    ]
                except Exception as e:
                    logger.warning(f"Web search failed: {e}")
                    metrics.mark_degraded("web_search", str(e))
                    ctx.web_search_retrieved = []
        else:
            # Skip web search - not enabled or not in plan
            if self._web_search_agent is None:
                pass  # Don't even log skipped step if not configured
            else:
                step_info = metrics.track_step("WebSearchAgent")
                step_info.__enter__()
                step_info.extra["skipped"] = True
                step_info.extra["reason"] = "not_triggered"
                step_info.__exit__(None, None, None)

        # RRF Fusion - combine all retrieval sources
        retrieval_lists = []
        
        if retrieval_mode in ("hybrid", "dense") and ctx.dense_retrieved:
            retrieval_lists.append(ctx.dense_retrieved)
        if retrieval_mode in ("hybrid", "bm25") and ctx.bm25_retrieved:
            retrieval_lists.append(ctx.bm25_retrieved)
        if ctx.web_search_retrieved:
            retrieval_lists.append(ctx.web_search_retrieved)
        
        if len(retrieval_lists) > 1 and plan.get("use_rrf", True):
            with metrics.track_step("RRFAgent") as step:
                try:
                    ctx.fused = self._rrf_agent.run(retrieval_lists)
                    step.extra["num_fused"] = len(ctx.fused)
                    step.extra["sources"] = len(retrieval_lists)
                except Exception as e:
                    logger.warning(f"RRF fusion failed: {e}")
                    metrics.mark_degraded("rrf", str(e))
                    # Fallback to first available list
                    ctx.fused = retrieval_lists[0] if retrieval_lists else []
        elif len(retrieval_lists) == 1:
            # Only one source, use directly
            ctx.fused = retrieval_lists[0]
        elif retrieval_mode == "dense":
            ctx.fused = ctx.dense_retrieved
        elif retrieval_mode == "bm25":
            ctx.fused = ctx.bm25_retrieved
        else:
            # Hybrid fallback
            ctx.fused = ctx.dense_retrieved or ctx.bm25_retrieved or []

    def _run_post_retrieval(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        plan: Dict[str, Any],
    ) -> None:
        """Execute post-retrieval processing phase."""
        current_docs = ctx.fused

        # Auto-merging
        if plan.get("use_automerge", True):
            with metrics.track_step("AutoMergingAgent") as step:
                try:
                    ctx.auto_merged = self._automerge_agent.run(current_docs)
                    current_docs = ctx.auto_merged
                    step.extra["num_docs"] = len(ctx.auto_merged)
                except Exception as e:
                    logger.warning(f"Auto-merge failed: {e}")
                    metrics.mark_degraded("automerge", str(e))
                    ctx.auto_merged = current_docs
        else:
            ctx.auto_merged = current_docs

        # Reranking
        if plan.get("use_rerank", True):
            with metrics.track_step("RerankingAgent") as step:
                try:
                    ctx.reranked = self._rerank_agent.run(
                        ctx.original_query,
                        ctx.auto_merged,
                    )
                    step.extra["num_reranked"] = len(ctx.reranked)
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}")
                    metrics.mark_degraded("rerank", str(e))
                    ctx.reranked = ctx.auto_merged
        else:
            ctx.reranked = ctx.auto_merged

    def _run_generation(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        plan: Dict[str, Any],
    ) -> str:
        """Execute generation phase."""
        # Extract documents for synthesis
        context_docs = [doc for doc, _ in ctx.reranked]

        if not context_docs:
            ctx.add_warning("No relevant documents found")
            return "I couldn't find any relevant documents to answer your question. Could you please rephrase or provide more context?"

        # Answer synthesis
        with metrics.track_step("AnswerSynthesisAgent") as step:
            ctx.final_answer = self._synthesis_agent.run(
                ctx.original_query,
                context_docs,
                ctx.conversation_history,
            )
            step.extra["answer_length"] = len(ctx.final_answer)

        # Critique
        if plan.get("use_critic", True) and self._config.critic.enabled:
            with metrics.track_step("CriticAgent") as step:
                try:
                    critique = self._critic_agent.run(
                        ctx.original_query,
                        ctx.final_answer,
                        context_docs,
                    )
                    ctx.critic_notes.append(critique)
                    step.extra["critic_ok"] = critique.get("ok", True)
                    
                    if not critique.get("ok", True):
                        issues = critique.get("issues", [])
                        if issues:
                            ctx.add_warning(f"Critic found issues: {', '.join(issues)}")
                except Exception as e:
                    logger.warning(f"Critic failed: {e}")
                    metrics.mark_degraded("critic", str(e))

        return ctx.final_answer


class SimplifiedOrchestrator:
    """
    Simplified orchestrator for quick queries without full pipeline.
    
    Useful for simple questions that don't need the full agentic flow.
    """

    def __init__(
        self,
        llm: LLMClient,
        local: LocalNLPModels,
        store: RedisVectorStore,
        config: AppConfig,
    ) -> None:
        """Initialize simplified orchestrator."""
        self._llm = llm
        self._local = local
        self._store = store
        self._config = config

    def run(self, query: str, top_k: int = 5) -> str:
        """
        Execute simplified RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Generated answer
        """
        # Embed query
        query_vec = self._local.embed_single(query)

        # Retrieve
        results = self._store.retrieve_by_embedding(query_vec, top_k=top_k)

        if not results:
            return "I couldn't find any relevant information to answer your question."

        # Format context
        context_parts = []
        for i, (doc, score) in enumerate(results, start=1):
            content = doc.content[:2000]
            context_parts.append(f"[{i}] {content}")

        context = "\n\n".join(context_parts)

        # Generate answer
        system = "Answer the question using the provided context. Be concise and accurate."
        user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        response = self._llm.chat(
            self._llm.create_messages(system, user),
            retry_on_error=True,
        )

        if not response.success:
            return f"Error generating answer: {response.error}"

        return response.content.strip()
