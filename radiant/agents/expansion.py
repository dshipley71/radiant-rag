"""
Query expansion agent for RAG pipeline.

Generates related terms and synonyms to improve recall.
"""

from __future__ import annotations

import logging
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant.config import QueryConfig
    from radiant.llm.client import LLMClient

logger = logging.getLogger(__name__)


class QueryExpansionAgent:
    """
    Expands queries with related terms and synonyms.

    Generates additional search terms to improve recall.
    """

    def __init__(self, llm: "LLMClient", config: "QueryConfig") -> None:
        self._llm = llm
        self._config = config

    def run(self, query: str) -> List[str]:
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

        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default=[],
            expected_type=list,
        )

        if not response.success:
            return []

        if isinstance(result, list):
            expansions = [str(x).strip() for x in result if isinstance(x, str) and x.strip()]
            return expansions[: self._config.max_expansions]

        return []
