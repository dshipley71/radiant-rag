"""
Planning agent for RAG pipeline.

Analyzes queries and produces execution plans that control
which pipeline features are activated.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant.llm.client import LLMClient

logger = logging.getLogger(__name__)


class PlanningAgent:
    """
    Planning agent that decides which pipeline features to use.

    Analyzes the query and produces a plan that enables/disables
    various pipeline stages based on query characteristics.
    """

    def __init__(self, llm: "LLMClient", web_search_enabled: bool = False) -> None:
        self._llm = llm
        self._web_search_enabled = web_search_enabled

    def run(self, query: str) -> Dict[str, Any]:
        """
        Generate execution plan for query.

        Args:
            query: User query

        Returns:
            Plan dictionary with feature flags
        """
        web_search_instruction = ""
        if self._web_search_enabled:
            web_search_instruction = """
- use_web_search: Search the web for current/recent information (use for queries about recent events, news, current status, or when indexed content may be outdated)
"""

        system = f"""You are a PlanningAgent for an agentic RAG system.
Analyze the query and produce a JSON plan that decides which features to use.

Available features:
- use_decomposition: Break complex queries into sub-queries
- use_rewrite: Rewrite query for better retrieval
- use_expansion: Add synonyms and related terms
- use_rrf: Use Reciprocal Rank Fusion to combine retrieval methods
- use_automerge: Merge child chunks into parent documents when appropriate
- use_rerank: Use cross-encoder reranking
- use_critic: Evaluate answer quality{web_search_instruction}

Consider:
- Simple factual queries may not need decomposition
- Queries with multiple parts benefit from decomposition
- Ambiguous queries benefit from rewriting and expansion
- Complex queries benefit from all features
- Queries about recent events, news, or current information benefit from web search
- Queries with words like "latest", "recent", "current", "today", "news" may need web search

Return ONLY a JSON object with boolean values for each feature."""

        user = f"Query: {query}\n\nReturn JSON plan only."

        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={},
            expected_type=dict,
        )

        # Ensure all required keys exist with defaults
        default_plan = {
            "use_decomposition": True,
            "use_rewrite": True,
            "use_expansion": True,
            "use_rrf": True,
            "use_automerge": True,
            "use_rerank": True,
            "use_critic": True,
            "use_web_search": False,
        }

        if not response.success or not result:
            logger.warning("Planning agent failed, using default plan")
            return default_plan

        # Merge with defaults
        for key in default_plan:
            if key not in result:
                result[key] = default_plan[key]
            else:
                result[key] = bool(result[key])

        # Only allow web search if enabled in config
        if not self._web_search_enabled:
            result["use_web_search"] = False

        return result
