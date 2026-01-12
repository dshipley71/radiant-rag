"""
Query rewrite agent for RAG pipeline.

Transforms queries to improve retrieval effectiveness.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, TYPE_CHECKING

from radiant.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    LLMAgent,
)

if TYPE_CHECKING:
    from radiant.llm.client import LLMClient

logger = logging.getLogger(__name__)


class QueryRewriteAgent(LLMAgent):
    """
    Rewrites queries to improve retrieval effectiveness.

    Transforms queries to be more specific, remove ambiguity,
    or better match document terminology.
    """

    def __init__(
        self,
        llm: "LLMClient",
        enabled: bool = True,
    ) -> None:
        """
        Initialize the rewrite agent.
        
        Args:
            llm: LLM client for reasoning
            enabled: Whether the agent is enabled
        """
        super().__init__(llm=llm, enabled=enabled)

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "QueryRewriteAgent"

    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.QUERY_PROCESSING

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Rewrites queries to improve retrieval effectiveness"

    def _execute(
        self,
        query: str,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """
        Rewrite query for better retrieval.

        Args:
            query: Original query

        Returns:
            Tuple of (original_query, rewritten_query)
        """
        system = """You are a QueryRewriteAgent.
Rewrite the query to maximize retrieval precision while preserving the original meaning.

Consider:
- Making implicit concepts explicit
- Using more specific terminology
- Removing filler words
- Clarifying ambiguous references

Return a JSON object: {"before": "original query", "after": "rewritten query"}"""

        user = f"Query: {query}\n\nReturn JSON only."

        result = self._chat_json(
            system=system,
            user=user,
            default={"before": query, "after": query},
            expected_type=dict,
        )

        if not result:
            return query, query

        before = str(result.get("before", query)).strip() or query
        after = str(result.get("after", query)).strip() or query

        if before != after:
            self.logger.info(
                "Query rewritten",
                original=before[:50],
                rewritten=after[:50],
            )

        return before, after

    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[Tuple[str, str]]:
        """
        Return original query on error.
        """
        query = kwargs.get("query", "")
        self.logger.warning(f"Rewrite failed, using original: {error}")
        return (query, query) if query else ("", "")
