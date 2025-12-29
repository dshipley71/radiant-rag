"""
Storage backend factory for Radiant Agentic RAG.

Provides a factory function to instantiate the appropriate storage backend
based on configuration. Supports Redis (default), Chroma, and PgVector.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

from radiant.config import AppConfig, ChromaConfig, PgVectorConfig, RedisConfig

if TYPE_CHECKING:
    from radiant.storage.base import BaseVectorStore

logger = logging.getLogger(__name__)


# Type alias for all supported store types
VectorStore = Union[
    "RedisVectorStore",
    "ChromaVectorStore", 
    "PgVectorStore",
]


def create_vector_store(config: AppConfig) -> "BaseVectorStore":
    """
    Create a vector store instance based on configuration.
    
    The backend is determined by config.storage.backend:
        - "redis" (default): Uses Redis with RediSearch for vector storage
        - "chroma": Uses ChromaDB for persistent vector storage
        - "pgvector": Uses PostgreSQL with pgvector extension
    
    Args:
        config: Application configuration
        
    Returns:
        An instance of the appropriate vector store class
        
    Raises:
        ValueError: If an unknown backend is specified
        ImportError: If required dependencies are not installed
    """
    backend = config.storage.backend.lower()
    
    if backend == "redis":
        from radiant.storage.redis_store import RedisVectorStore
        logger.info("Creating Redis vector store")
        return RedisVectorStore(config.redis)
    
    elif backend == "chroma":
        from radiant.storage.chroma_store import ChromaVectorStore
        logger.info("Creating Chroma vector store")
        return ChromaVectorStore(config.chroma)
    
    elif backend == "pgvector":
        from radiant.storage.pgvector_store import PgVectorStore
        logger.info("Creating PgVector store")
        return PgVectorStore(config.pgvector)
    
    else:
        raise ValueError(
            f"Unknown storage backend: '{backend}'. "
            f"Supported backends: redis, chroma, pgvector"
        )


def get_available_backends() -> dict:
    """
    Check which storage backends are available.
    
    Returns:
        Dictionary mapping backend name to availability status and import error (if any)
    """
    backends = {}
    
    # Check Redis
    try:
        import redis
        from redis.commands.search.field import VectorField
        backends["redis"] = {"available": True, "error": None}
    except ImportError as e:
        backends["redis"] = {"available": False, "error": str(e)}
    
    # Check Chroma
    try:
        import chromadb
        backends["chroma"] = {"available": True, "error": None}
    except ImportError as e:
        backends["chroma"] = {"available": False, "error": str(e)}
    
    # Check PgVector
    try:
        import psycopg2
        backends["pgvector"] = {"available": True, "error": None}
    except ImportError as e:
        backends["pgvector"] = {"available": False, "error": str(e)}
    
    return backends


def validate_backend_config(config: AppConfig) -> None:
    """
    Validate that the selected backend is properly configured.
    
    Args:
        config: Application configuration
        
    Raises:
        ValueError: If configuration is invalid or incomplete
    """
    backend = config.storage.backend.lower()
    
    if backend == "redis":
        if not config.redis.url:
            raise ValueError(
                "Redis URL not configured. Set redis.url in config or "
                "RADIANT_REDIS_URL environment variable."
            )
    
    elif backend == "chroma":
        if not config.chroma.persist_directory:
            raise ValueError(
                "Chroma persist_directory not configured. "
                "Set chroma.persist_directory in config."
            )
    
    elif backend == "pgvector":
        import os
        conn_str = config.pgvector.connection_string or os.environ.get("PG_CONN_STR")
        if not conn_str:
            raise ValueError(
                "PostgreSQL connection string not configured. "
                "Set pgvector.connection_string in config or PG_CONN_STR environment variable. "
                "Format: postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"
            )
    
    else:
        raise ValueError(
            f"Unknown storage backend: '{backend}'. "
            f"Supported backends: redis, chroma, pgvector"
        )


__all__ = [
    "create_vector_store",
    "get_available_backends",
    "validate_backend_config",
    "VectorStore",
]
