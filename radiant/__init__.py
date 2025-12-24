"""
Radiant Agentic RAG - A modular Retrieval-Augmented Generation system.

This package provides a complete RAG pipeline with:
    - Multiple retrieval strategies (dense, sparse, hybrid)
    - Agentic query processing (planning, decomposition, rewriting)
    - Hierarchical document storage with auto-merging
    - Real-time web search augmentation
    - Comprehensive reporting and visualization
"""

__version__ = "1.0.0"
__author__ = "Radiant RAG Team"

# Defer imports to avoid circular dependencies
# Use: from radiant.app import RadiantRAG
# Use: from radiant.config import load_config

__all__ = [
    "__version__",
]
