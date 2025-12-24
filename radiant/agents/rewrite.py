"""
Query rewrite agent for RAG pipeline.

Transforms queries to improve retrieval effectiveness.
"""

from __future__ import annotations

import logging
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant.llm.client import LLMClient

logger = logging.getLogger(__name__)


class QueryRewriteAgent:
    """
    Rewrites queries to improve retrieval effectiveness.

    Transforms queries to be more specific, remove ambiguity,
    or better match document terminology.
    """

    def __init__(self, llm: "LLMClient") -> None:
        self._llm = llm

    def run(self, query: str) -> Tuple[str, str]:
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

        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={"before": query, "after": query},
            expected_type=dict,
        )

        if not response.success or not result:
            return query, query

        before = str(result.get("before", query)).strip() or query
        after = str(result.get("after", query)).strip() or query

        return before, after
