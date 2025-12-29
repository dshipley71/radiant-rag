"""
Storage backends package.

Provides:
    - BaseVectorStore: Abstract base class for vector storage
    - StoredDoc: Document data class
    - RedisVectorStore: Redis + Vector Search storage (default)
    - ChromaVectorStore: ChromaDB storage
    - PgVectorStore: PostgreSQL with pgvector storage
    - PersistentBM25Index: BM25 sparse index
    - create_vector_store: Factory function for storage backends
"""

from radiant.storage.base import BaseVectorStore, StoredDoc
from radiant.storage.factory import (
    create_vector_store,
    get_available_backends,
    validate_backend_config,
)

# Lazy imports for optional backends to avoid import errors
# when dependencies are not installed

def get_redis_store():
    """Get RedisVectorStore class (lazy import)."""
    from radiant.storage.redis_store import RedisVectorStore
    return RedisVectorStore

def get_chroma_store():
    """Get ChromaVectorStore class (lazy import)."""
    from radiant.storage.chroma_store import ChromaVectorStore
    return ChromaVectorStore

def get_pgvector_store():
    """Get PgVectorStore class (lazy import)."""
    from radiant.storage.pgvector_store import PgVectorStore
    return PgVectorStore

def get_bm25_index():
    """Get PersistentBM25Index class (lazy import)."""
    from radiant.storage.bm25_index import PersistentBM25Index
    return PersistentBM25Index

# For backward compatibility, try to import directly if possible
# This allows `from radiant.storage import RedisVectorStore` to work
# when redis is installed
try:
    from radiant.storage.redis_store import RedisVectorStore
except ImportError:
    RedisVectorStore = None

try:
    from radiant.storage.bm25_index import PersistentBM25Index
except ImportError:
    PersistentBM25Index = None

__all__ = [
    # Base classes
    "BaseVectorStore",
    "StoredDoc",
    # Storage implementations (lazy)
    "get_redis_store",
    "get_chroma_store",
    "get_pgvector_store",
    # BM25 index (lazy)
    "get_bm25_index",
    # Direct imports (may be None if deps missing)
    "RedisVectorStore",
    "PersistentBM25Index",
    # Factory functions
    "create_vector_store",
    "get_available_backends",
    "validate_backend_config",
]
