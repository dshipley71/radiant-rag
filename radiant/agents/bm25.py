"""
BM25 retrieval agent for RAG pipeline.

Uses sparse keyword-based retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant.config import RetrievalConfig
    from radiant.storage.bm25_index import PersistentBM25Index

logger = logging.getLogger(__name__)


class BM25RetrievalAgent:
    """
    Sparse keyword-based retrieval using BM25.
    """

    def __init__(
        self,
        bm25_index: "PersistentBM25Index",
        config: "RetrievalConfig",
    ) -> None:
        self._index = bm25_index
        self._config = config

    def run(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Any, float]]:
        """
        Retrieve documents using BM25.

        Args:
            query: Query text
            top_k: Maximum results (defaults to config)

        Returns:
            List of (document, score) tuples
        """
        k = top_k or self._config.bm25_top_k
        return self._index.search(query, top_k=k)
