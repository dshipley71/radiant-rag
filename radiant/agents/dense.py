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
    from radiant.storage.base import BaseVectorStore

logger = logging.getLogger(__name__)


class DenseRetrievalAgent:
    """
    Dense embedding-based retrieval using vector similarity search.
    
    Supports searching leaves (child chunks), parents, or both based on
    the search_scope configuration.
    """

    def __init__(
        self,
        store: "BaseVectorStore",
        local: "LocalNLPModels",
        config: "RetrievalConfig",
    ) -> None:
        self._store = store
        self._local = local
        self._config = config

    def _get_doc_level_filter(self, search_scope: Optional[str] = None) -> Optional[str]:
        """
        Convert search_scope to doc_level_filter.
        
        Args:
            search_scope: Override for config search_scope
            
        Returns:
            doc_level_filter value or None for no filtering
        """
        scope = search_scope or self._config.search_scope
        
        if scope == "leaves":
            return "child"
        elif scope == "parents":
            return "parent"
        elif scope == "all":
            return None
        else:
            # Default to leaves for backward compatibility
            return "child"

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        search_scope: Optional[str] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Retrieve documents by embedding similarity.

        Args:
            query: Query text
            top_k: Maximum results (defaults to config)
            search_scope: Override search scope ("leaves", "parents", "all")

        Returns:
            List of (document, score) tuples
        """
        k = top_k or self._config.dense_top_k
        doc_level_filter = self._get_doc_level_filter(search_scope)

        # Generate query embedding
        query_vec = self._local.embed_single(query)

        # Search vector store
        return self._store.retrieve_by_embedding(
            query_embedding=query_vec,
            top_k=k,
            min_similarity=self._config.min_similarity,
            doc_level_filter=doc_level_filter,
        )
