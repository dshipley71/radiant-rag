"""
Query decomposition agent for RAG pipeline.

Breaks complex queries into simpler sub-queries for better retrieval.
"""

from __future__ import annotations

import logging
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant.config import QueryConfig
    from radiant.llm.client import LLMClient

logger = logging.getLogger(__name__)


class QueryDecompositionAgent:
    """
    Decomposes complex queries into simpler sub-queries.

    Useful for multi-part questions or queries requiring
    information from multiple sources.
    """

    def __init__(self, llm: "LLMClient", config: "QueryConfig") -> None:
        self._llm = llm
        self._config = config

    def run(self, query: str) -> List[str]:
        """
        Decompose query into sub-queries.

        Args:
            query: Original query

        Returns:
            List of sub-queries (may be single element if no decomposition needed)
        """
        system = """You are a QueryDecompositionAgent.
If the query is complex or contains multiple distinct questions, decompose it into independent sub-queries.
If the query is simple and doesn't need decomposition, return it as-is.

Return a JSON array of strings. Each string should be a complete, self-contained query.
Maximum sub-queries: Return at most 5 sub-queries.

Examples:
- "What is Python and how does it compare to Java?" -> ["What is Python?", "How does Python compare to Java?"]
- "Tell me about climate change" -> ["Tell me about climate change"]"""

        user = f"Query: {query}\n\nReturn JSON array only."

        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default=[query],
            expected_type=list,
        )

        if not response.success:
            return [query]

        # Validate and clean results
        if isinstance(result, list):
            queries = [str(q).strip() for q in result if isinstance(q, str) and q.strip()]
            if queries:
                return queries[: self._config.max_decomposed_queries]

        return [query]
