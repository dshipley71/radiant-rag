"""
Critic agent for RAG pipeline.

Evaluates answer quality and suggests improvements.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant.config import CriticConfig
    from radiant.llm.client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent:
    """
    Evaluates answer quality and suggests improvements.

    Checks for relevance, faithfulness to context, and coverage.
    """

    def __init__(
        self,
        llm: "LLMClient",
        config: "CriticConfig",
    ) -> None:
        self._llm = llm
        self._config = config

    def run(
        self,
        query: str,
        answer: str,
        context_docs: List[Any],
    ) -> Dict[str, Any]:
        """
        Critique the generated answer.

        Args:
            query: Original query
            answer: Generated answer
            context_docs: Context documents used

        Returns:
            Critique dictionary with ok flag and issues
        """
        max_docs = self._config.max_context_docs
        max_chars = self._config.max_doc_chars

        # Format context
        context_parts = []
        for i, doc in enumerate(context_docs[:max_docs], start=1):
            content = doc.content[:max_chars] if len(doc.content) > max_chars else doc.content
            context_parts.append(f"[DOC {i}] {content}")

        context = "\n\n".join(context_parts)

        system = """You are a CriticAgent for RAG systems.
Evaluate the answer for:
1. Relevance: Does it address the question?
2. Faithfulness: Is it supported by the context?
3. Coverage: Does it cover all important aspects?
4. Accuracy: Are there any factual errors?

Return a JSON object:
{
  "ok": true/false,
  "relevance_score": 0-10,
  "faithfulness_score": 0-10,
  "coverage_score": 0-10,
  "issues": ["list of issues"],
  "suggested_improvements": ["list of suggestions"]
}"""

        user = f"""QUERY:
{query}

CONTEXT:
{context}

ANSWER:
{answer}

Return JSON critique only."""

        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={"ok": True, "issues": []},
            expected_type=dict,
        )

        if not response.success:
            return {"ok": True, "issues": ["critic_failed"], "suggested_improvements": []}

        # Ensure required fields
        if "ok" not in result:
            result["ok"] = True
        if "issues" not in result:
            result["issues"] = []
        if "suggested_improvements" not in result:
            result["suggested_improvements"] = []

        return result
