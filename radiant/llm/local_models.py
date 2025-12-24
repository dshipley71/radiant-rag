"""
Local NLP models for embedding and reranking.

Provides sentence-transformers based models for:
    - Text embedding (bi-encoder)
    - Cross-encoder reranking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

if TYPE_CHECKING:
    from radiant.config import LocalModelsConfig

logger = logging.getLogger(__name__)


def _resolve_device(pref: str) -> str:
    """
    Resolve device preference to actual device.

    Args:
        pref: Device preference ("auto", "cpu", "cuda")

    Returns:
        Resolved device string
    """
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class LocalNLPModels:
    """
    Local NLP models for embedding and reranking.

    Uses sentence-transformers for efficient local inference.
    """

    embedder: SentenceTransformer
    cross_encoder: CrossEncoder
    device: str
    embedding_dim: int

    @staticmethod
    def build(config: "LocalModelsConfig") -> "LocalNLPModels":
        """
        Build local NLP models from configuration.

        Args:
            config: Local models configuration

        Returns:
            LocalNLPModels instance
        """
        device = _resolve_device(config.device)
        logger.info(f"Loading local models on device: {device}")

        embedder = SentenceTransformer(config.embed_model_name, device=device)
        cross_encoder = CrossEncoder(config.cross_encoder_name, device=device)

        # Get actual embedding dimension
        embedding_dim = embedder.get_sentence_embedding_dimension()
        if embedding_dim != config.embedding_dimension:
            logger.warning(
                f"Config embedding_dimension ({config.embedding_dimension}) does not match "
                f"model dimension ({embedding_dim}). Using model dimension."
            )

        logger.info(
            f"Loaded embedder: {config.embed_model_name} (dim={embedding_dim}), "
            f"cross-encoder: {config.cross_encoder_name}"
        )

        return LocalNLPModels(
            embedder=embedder,
            cross_encoder=cross_encoder,
            device=device,
            embedding_dim=embedding_dim,
        )

    def embed(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            normalize: Whether to L2-normalize embeddings

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [emb.tolist() for emb in embeddings]

    def embed_single(self, text: str, normalize: bool = True) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            normalize: Whether to L2-normalize

        Returns:
            Embedding vector
        """
        result = self.embed([text], normalize=normalize)
        return result[0] if result else []

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[tuple[int, float]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Query text
            documents: List of document texts
            top_k: Optional limit on results

        Returns:
            List of (document_index, score) tuples, sorted by score descending
        """
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)

        # Create indexed scores and sort
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores
