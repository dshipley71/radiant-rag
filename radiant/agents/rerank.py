"""
Cross-encoder reranking agent for RAG pipeline.

Provides more accurate relevance scoring using cross-encoder models.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant.config import RerankConfig
    from radiant.llm.client import LocalNLPModels

logger = logging.getLogger(__name__)


class CrossEncoderRerankingAgent:
    """
    Reranks documents using a cross-encoder model.

    Provides more accurate relevance scoring than bi-encoder similarity.
    """

    def __init__(
        self,
        local: "LocalNLPModels",
        config: "RerankConfig",
    ) -> None:
        self._local = local
        self._config = config

    def run(
        self,
        query: str,
        docs: List[Tuple[Any, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Query text
            docs: Documents to rerank
            top_k: Maximum results

        Returns:
            Reranked documents
        """
        if not docs:
            return []

        k = top_k or self._config.top_k
        max_doc_chars = self._config.max_doc_chars

        # Determine candidate count
        num_candidates = max(
            k * self._config.candidate_multiplier,
            self._config.min_candidates,
        )
        candidates = docs[:num_candidates]

        # Prepare texts for reranking
        doc_texts = [
            doc.content[:max_doc_chars] if len(doc.content) > max_doc_chars else doc.content
            for doc, _ in candidates
        ]

        # Get reranking scores
        rerank_scores = self._local.rerank(query, doc_texts, top_k=k)

        # Map back to documents
        result = [
            (candidates[idx][0], score)
            for idx, score in rerank_scores
        ]

        return result
