"""
Answer synthesis agent for RAG pipeline.

Generates coherent answers from retrieved context.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant.config import SynthesisConfig
    from radiant.llm.client import LLMClient
    from radiant.utils.conversation import ConversationManager

logger = logging.getLogger(__name__)


class AnswerSynthesisAgent:
    """
    Synthesizes answers from retrieved context.

    Generates coherent answers grounded in the retrieved documents.
    """

    def __init__(
        self,
        llm: "LLMClient",
        config: "SynthesisConfig",
        conversation_manager: Optional["ConversationManager"] = None,
    ) -> None:
        self._llm = llm
        self._config = config
        self._conversation = conversation_manager

    def run(
        self,
        query: str,
        docs: List[Any],
        conversation_history: str = "",
    ) -> str:
        """
        Synthesize answer from context.

        Args:
            query: User query
            docs: Retrieved documents
            conversation_history: Optional conversation history

        Returns:
            Generated answer
        """
        max_docs = self._config.max_context_docs
        max_chars = self._config.max_doc_chars

        # Format context
        context_parts = []
        for i, doc in enumerate(docs[:max_docs], start=1):
            content = doc.content[:max_chars] if len(doc.content) > max_chars else doc.content
            source = doc.meta.get("source_path", "unknown")
            context_parts.append(f"[DOC {i}] (Source: {source})\n{content}")

        context = "\n\n".join(context_parts)

        # Build system prompt
        system = """You are a RAG Answer Agent.
Answer the question using ONLY the provided context documents.

Guidelines:
- Base your answer strictly on the provided context
- If the context doesn't contain sufficient information, say so clearly
- Be concise but complete
- Cite document numbers when referencing specific information (e.g., "According to [DOC 1]...")
- If multiple documents provide relevant information, synthesize them coherently"""

        # Build user prompt
        user_parts = []

        if conversation_history and self._config.include_history:
            user_parts.append(f"CONVERSATION HISTORY:\n{conversation_history}\n")

        user_parts.append(f"QUESTION:\n{query}\n")
        user_parts.append(f"CONTEXT:\n{context}\n")
        user_parts.append("ANSWER:")

        user = "\n".join(user_parts)

        # Generate answer
        response = self._llm.chat(
            self._llm.create_messages(system, user),
            retry_on_error=True,
        )

        if not response.success:
            return f"I apologize, but I encountered an error generating the answer: {response.error}"

        return response.content.strip()
