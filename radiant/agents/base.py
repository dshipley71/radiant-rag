"""
Base classes and context for RAG pipeline agents.

Provides the AgentContext dataclass that accumulates results
as the query flows through each pipeline stage.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """
    Context object passed through the RAG pipeline.

    Accumulates results from each agent stage.
    """

    run_id: str
    original_query: str
    conversation_id: Optional[str] = None

    # Query processing results
    plan: Dict[str, Any] = field(default_factory=dict)
    decomposed_queries: List[str] = field(default_factory=list)
    rewrites: List[Tuple[str, str]] = field(default_factory=list)
    expansions: List[str] = field(default_factory=list)

    # Retrieval results - using Any to avoid import issues
    dense_retrieved: List[Tuple[Any, float]] = field(default_factory=list)
    bm25_retrieved: List[Tuple[Any, float]] = field(default_factory=list)
    web_search_retrieved: List[Tuple[Any, float]] = field(default_factory=list)
    fused: List[Tuple[Any, float]] = field(default_factory=list)
    auto_merged: List[Tuple[Any, float]] = field(default_factory=list)
    reranked: List[Tuple[Any, float]] = field(default_factory=list)

    # Generation results
    final_answer: Optional[str] = None
    critic_notes: List[Dict[str, Any]] = field(default_factory=list)

    # Conversation history
    conversation_history: str = ""

    # Metadata
    warnings: List[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(f"[{self.run_id}] {message}")


def new_agent_context(
    query: str,
    conversation_id: Optional[str] = None,
) -> AgentContext:
    """
    Create a new agent context for a query.

    Args:
        query: User query
        conversation_id: Optional conversation ID

    Returns:
        New AgentContext instance
    """
    return AgentContext(
        run_id=str(uuid.uuid4()),
        original_query=query,
        conversation_id=conversation_id,
    )
