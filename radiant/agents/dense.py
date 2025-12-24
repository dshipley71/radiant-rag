"""
Dense retrieval agent for RAG pipeline.

Uses embedding-based vector similarity search.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant.config import RetrievalConfig
    from radiant.llm.client import LocalNLPModels
    from radiant.storage.redis_store import RedisVectorStore

logger = logging.getLogger(__name__)


class DenseRetrievalAgent:
    """
    Dense embedding-based retrieval using vector similarity search.
    """

    def __init__(
        self,
        store: "RedisVectorStore",
        local: "LocalNLPModels",
        config: "RetrievalConfig",
    ) -> None:
        self._store = store
        self._local = local
        self._config = config

    def run(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Any, float]]:
        """
        Retrieve documents by embedding similarity.

        Args:
            query: Query text
            top_k: Maximum results (defaults to config)

        Returns:
            List of (document, score) tuples
        """
        k = top_k or self._config.dense_top_k

        # Generate query embedding
        query_vec = self._local.embed_single(query)

        # Search vector store
        return self._store.retrieve_by_embedding(
            query_embedding=query_vec,
            top_k=k,
            min_similarity=self._config.min_similarity,
        )
