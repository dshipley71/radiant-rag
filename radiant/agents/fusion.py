"""
Reciprocal Rank Fusion agent for RAG pipeline.

Combines results from multiple retrieval methods using RRF.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant.config import RetrievalConfig

logger = logging.getLogger(__name__)


class RRFAgent:
    """
    Reciprocal Rank Fusion agent for combining retrieval results.

    Merges results from multiple retrieval methods using the RRF formula.
    """

    def __init__(self, config: "RetrievalConfig") -> None:
        self._config = config

    def run(
        self,
        runs: List[List[Tuple[Any, float]]],
        top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Fuse multiple retrieval runs using RRF.

        Args:
            runs: List of retrieval results from different methods
            top_k: Maximum results (defaults to config)
            rrf_k: RRF constant (defaults to config)

        Returns:
            Fused results sorted by RRF score
        """
        k = top_k or self._config.fused_top_k
        rrf_constant = rrf_k or self._config.rrf_k

        scores: Dict[str, float] = {}
        doc_map: Dict[str, Any] = {}

        for run in runs:
            for rank, (doc, _score) in enumerate(run, start=1):
                doc_map[doc.doc_id] = doc
                scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + (1.0 / (rrf_constant + rank))

        fused = [(doc_map[doc_id], score) for doc_id, score in scores.items()]
        fused.sort(key=lambda x: x[1], reverse=True)

        return fused[:k]
