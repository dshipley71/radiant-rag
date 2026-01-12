"""
Query expansion agent for RAG pipeline.

Generates related terms and synonyms to improve recall.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, TYPE_CHECKING

from radiant.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    LLMAgent,
)

if TYPE_CHECKING:
    from radiant.config import QueryConfig
    from radiant.llm.client import LLMClient

logger = logging.getLogger(__name__)


class QueryExpansionAgent(LLMAgent):
    """
    Expands queries with related terms and synonyms.

    Generates additional search terms to improve recall.
    """

    def __init__(
        self,
        llm: "LLMClient",
        config: "QueryConfig",
        enabled: bool = True,
    ) -> None:
        """
        Initialize the expansion agent.
        
        Args:
            llm: LLM client for reasoning
            config: Query configuration
            enabled: Whether the agent is enabled
        """
        super().__init__(llm=llm, enabled=enabled)
        self._config = config

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "QueryExpansionAgent"

    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.QUERY_PROCESSING

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Expands queries with related terms and synonyms"

    def _execute(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate query expansions.

        Args:
            query: Original query

        Returns:
            List of expansion terms
        """
        system = """You are a QueryExpansionAgent.
Generate concise expansions (synonyms, related terms, key entities) that could improve retrieval.

Guidelines:
- Include synonyms and related terminology
- Include relevant entities (people, places, concepts)
- Keep expansions concise (1-3 words each)
- Avoid generic terms that won't help retrieval

Return a JSON array of strings."""

        user = f"Query: {query}\n\nReturn JSON array of 3-8 expansion terms."

        result = self._chat_json(
            system=system,
            user=user,
            default=[],
            expected_type=list,
        )

        if isinstance(result, list):
            expansions = [str(x).strip() for x in result if isinstance(x, str) and x.strip()]
            final_expansions = expansions[: self._config.max_expansions]
            
            if final_expansions:
                self.logger.info(
                    "Query expanded",
                    original=query[:50],
                    num_expansions=len(final_expansions),
                )
            
            return final_expansions

        return []

    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[List[str]]:
        """
        Return empty list on error.
        """
        self.logger.warning(f"Expansion failed: {error}")
        return []
