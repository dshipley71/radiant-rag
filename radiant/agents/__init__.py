"""
RAG pipeline agents package.

Provides specialized agents for each stage of the RAG pipeline:
    - Query processing (planning, decomposition, rewriting, expansion)
    - Retrieval (dense, sparse, web search)
    - Fusion (RRF)
    - Post-retrieval (auto-merging, reranking)
    - Generation (answer synthesis, critique)
    - Tools (calculator, code execution)
    - Strategy memory (adaptive retrieval)
    - Intelligent chunking (semantic document chunking)
    - Summarization (context compression)
    - Context evaluation (pre-generation quality gate)
"""

# Base context
from radiant.agents.base import AgentContext, new_agent_context

# Planning
from radiant.agents.planning import PlanningAgent

# Query processing
from radiant.agents.decomposition import QueryDecompositionAgent
from radiant.agents.rewrite import QueryRewriteAgent
from radiant.agents.expansion import QueryExpansionAgent

# Retrieval
from radiant.agents.dense import DenseRetrievalAgent
from radiant.agents.bm25 import BM25RetrievalAgent
from radiant.agents.web_search import WebSearchAgent

# Fusion
from radiant.agents.fusion import RRFAgent

# Post-retrieval
from radiant.agents.automerge import HierarchicalAutoMergingAgent
from radiant.agents.rerank import CrossEncoderRerankingAgent

# Generation
from radiant.agents.synthesis import AnswerSynthesisAgent
from radiant.agents.critic import CriticAgent

# Tools (agentic enhancements)
from radiant.agents.tools import (
    BaseTool,
    ToolType,
    ToolResult,
    ToolRegistry,
    ToolSelector,
    CalculatorTool,
    CodeExecutionTool,
    create_default_tool_registry,
)

# Strategy memory (agentic enhancements)
from radiant.agents.strategy_memory import (
    RetrievalStrategyMemory,
    StrategyOutcome,
    QueryPattern,
    QueryPatternExtractor,
)

# Intelligent chunking (new)
from radiant.agents.chunking import (
    IntelligentChunkingAgent,
    SemanticChunk,
    ChunkingResult,
)

# Summarization (new)
from radiant.agents.summarization import (
    SummarizationAgent,
    CompressedDocument,
    SummarizationResult,
)

# Context evaluation (new)
from radiant.agents.context_eval import (
    ContextEvaluationAgent,
    ContextEvaluation,
)

# Multi-hop reasoning (new)
from radiant.agents.multihop import (
    MultiHopReasoningAgent,
    MultiHopResult,
    ReasoningStep,
)

# Fact verification (new)
from radiant.agents.fact_verification import (
    FactVerificationAgent,
    FactVerificationResult,
    Claim,
    ClaimVerification,
    VerificationStatus,
)

# Citation tracking (new)
from radiant.agents.citation import (
    CitationTrackingAgent,
    CitedAnswer,
    Citation,
    SourceDocument,
    CitationStyle,
)

__all__ = [
    # Base
    "AgentContext",
    "new_agent_context",
    # Planning
    "PlanningAgent",
    # Query processing
    "QueryDecompositionAgent",
    "QueryRewriteAgent",
    "QueryExpansionAgent",
    # Retrieval
    "DenseRetrievalAgent",
    "BM25RetrievalAgent",
    "WebSearchAgent",
    # Fusion
    "RRFAgent",
    # Post-retrieval
    "HierarchicalAutoMergingAgent",
    "CrossEncoderRerankingAgent",
    # Generation
    "AnswerSynthesisAgent",
    "CriticAgent",
    # Tools
    "BaseTool",
    "ToolType",
    "ToolResult",
    "ToolRegistry",
    "ToolSelector",
    "CalculatorTool",
    "CodeExecutionTool",
    "create_default_tool_registry",
    # Strategy memory
    "RetrievalStrategyMemory",
    "StrategyOutcome",
    "QueryPattern",
    "QueryPatternExtractor",
    # Intelligent chunking
    "IntelligentChunkingAgent",
    "SemanticChunk",
    "ChunkingResult",
    # Summarization
    "SummarizationAgent",
    "CompressedDocument",
    "SummarizationResult",
    # Context evaluation
    "ContextEvaluationAgent",
    "ContextEvaluation",
    # Multi-hop reasoning
    "MultiHopReasoningAgent",
    "MultiHopResult",
    "ReasoningStep",
    # Fact verification
    "FactVerificationAgent",
    "FactVerificationResult",
    "Claim",
    "ClaimVerification",
    "VerificationStatus",
    # Citation tracking
    "CitationTrackingAgent",
    "CitedAnswer",
    "Citation",
    "SourceDocument",
    "CitationStyle",
]
