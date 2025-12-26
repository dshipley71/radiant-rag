"""
Diagnostic script to inspect indexed documents in Radiant RAG.

Usage:
    python inspect_index.py                    # List all sources
    python inspect_index.py --search "poker"   # Search for documents
    python inspect_index.py --source "dogs"    # Find docs from source containing "dogs"
    python inspect_index.py --doc-id "abc123"  # Get specific document
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for radiant imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add current directory to path
# sys.path.insert(0, str(Path(__file__).parent))

from radiant.config import load_config
from radiant.storage.redis_store import RedisVectorStore
from radiant.storage.bm25_index import PersistentBM25Index


def list_sources(store: RedisVectorStore, limit: int = 100):
    """List unique source files in the index."""
    doc_ids = store.list_doc_ids(limit=limit)
    sources = {}

    for doc_id in doc_ids:
        doc = store.get_doc(doc_id)
        if doc:
            source = doc.meta.get("source_path", "unknown")
            if source not in sources:
                sources[source] = {"count": 0, "sample_id": doc_id, "doc_level": set()}
            sources[source]["count"] += 1
            sources[source]["doc_level"].add(doc.meta.get("doc_level", "unknown"))

    print(f"\n{'='*60}")
    print(f"INDEXED SOURCES ({len(sources)} unique files)")
    print(f"{'='*60}\n")

    for source, info in sorted(sources.items()):
        levels = ", ".join(info["doc_level"])
        print(f"  📄 {source}")
        print(f"     Chunks: {info['count']}, Levels: {levels}")
        print()

    return sources


def search_content(store: RedisVectorStore, query: str, limit: int = 10):
    """Search document content for a string."""
    doc_ids = store.list_doc_ids(limit=10000)
    matches = []

    query_lower = query.lower()

    for doc_id in doc_ids:
        doc = store.get_doc(doc_id)
        if doc and query_lower in doc.content.lower():
            matches.append(doc)

    print(f"\n{'='*60}")
    print(f"CONTENT SEARCH: '{query}' ({len(matches)} matches)")
    print(f"{'='*60}\n")

    for doc in matches[:limit]:
        source = doc.meta.get("source_path", "unknown")
        level = doc.meta.get("doc_level", "unknown")

        # Find the matching snippet
        content_lower = doc.content.lower()
        idx = content_lower.find(query_lower)
        start = max(0, idx - 50)
        end = min(len(doc.content), idx + len(query) + 100)
        snippet = doc.content[start:end].replace("\n", " ")

        print(f"  📄 Source: {source}")
        print(f"     Doc ID: {doc.doc_id[:40]}...")
        print(f"     Level: {level}")
        print(f"     Snippet: ...{snippet}...")
        print()

    return matches


def find_by_source(store: RedisVectorStore, source_pattern: str, limit: int = 20):
    """Find documents from a specific source file."""
    doc_ids = store.list_doc_ids(limit=10000)
    matches = []

    pattern_lower = source_pattern.lower()

    for doc_id in doc_ids:
        doc = store.get_doc(doc_id)
        if doc:
            source = doc.meta.get("source_path", "").lower()
            if pattern_lower in source:
                matches.append(doc)

    print(f"\n{'='*60}")
    print(f"DOCS FROM SOURCE: '{source_pattern}' ({len(matches)} matches)")
    print(f"{'='*60}\n")

    for doc in matches[:limit]:
        source = doc.meta.get("source_path", "unknown")
        level = doc.meta.get("doc_level", "unknown")
        has_embedding = store.has_embedding(doc.doc_id)

        # Show content preview
        preview = doc.content[:300].replace("\n", " ")
        if len(doc.content) > 300:
            preview += "..."

        print(f"  📄 Source: {source}")
        print(f"     Doc ID: {doc.doc_id[:40]}...")
        print(f"     Level: {level}")
        print(f"     Has Embedding: {has_embedding}")
        print(f"     Content Length: {len(doc.content)} chars")
        print(f"     Preview: {preview}")
        print(f"     Metadata: {json.dumps({k: v for k, v in doc.meta.items() if k != 'cleaning'}, indent=2)}")
        print()

    return matches


def get_doc_by_id(store: RedisVectorStore, doc_id: str):
    """Get a specific document by ID."""
    doc = store.get_doc(doc_id)

    if not doc:
        print(f"\n❌ Document not found: {doc_id}")
        return None

    print(f"\n{'='*60}")
    print(f"DOCUMENT DETAILS")
    print(f"{'='*60}\n")

    print(f"  Doc ID: {doc.doc_id}")
    print(f"  Source: {doc.meta.get('source_path', 'unknown')}")
    print(f"  Level: {doc.meta.get('doc_level', 'unknown')}")
    print(f"  Has Embedding: {store.doc_has_embedding(doc.doc_id)}")
    print(f"  Content Length: {len(doc.content)} chars")
    print(f"\n  CONTENT:\n  {'-'*50}")
    print(f"  {doc.content}")
    print(f"\n  METADATA:\n  {'-'*50}")
    print(f"  {json.dumps(doc.meta, indent=2)}")

    return doc


def show_stats(store: RedisVectorStore, bm25: PersistentBM25Index):
    """Show index statistics."""
    doc_ids = store.list_doc_ids(limit=100000)

    # Count by level
    levels = {}
    for doc_id in doc_ids:
        doc = store.get_doc(doc_id)
        if doc:
            level = doc.meta.get("doc_level", "unknown")
            levels[level] = levels.get(level, 0) + 1

    print(f"\n{'='*60}")
    print(f"INDEX STATISTICS")
    print(f"{'='*60}\n")

    print(f"  Redis Documents: {len(doc_ids)}")
    for level, count in sorted(levels.items()):
        print(f"    - {level}: {count}")

    print(f"\n  BM25 Index: {len(bm25)} documents")

    bm25_stats = bm25.get_stats()
    print(f"    - Unique terms: {bm25_stats.get('unique_terms', 'N/A')}")
    print(f"    - Avg doc length: {bm25_stats.get('avg_doc_length', 'N/A'):.1f} tokens")

    # Vector index info
    vector_info = store.get_index_info()
    print(f"\n  Vector Index:")
    for key, value in vector_info.items():
        print(f"    - {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Inspect Radiant RAG index")
    parser.add_argument("--search", "-s", help="Search document content")
    parser.add_argument("--source", help="Find docs from source containing pattern")
    parser.add_argument("--doc-id", help="Get specific document by ID")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    parser.add_argument("--limit", type=int, default=20, help="Max results to show")
    parser.add_argument("--config", "-c", help="Config file path")

    args = parser.parse_args()

    # Load config and initialize stores
    config = load_config(args.config)
    store = RedisVectorStore(config.redis)
    bm25 = PersistentBM25Index(config.bm25, store)

    if not store.ping():
        print("❌ Cannot connect to Redis")
        return 1

    if args.search:
        search_content(store, args.search, args.limit)
    elif args.source:
        find_by_source(store, args.source, args.limit)
    elif args.doc_id:
        get_doc_by_id(store, args.doc_id)
    elif args.stats:
        show_stats(store, bm25)
    else:
        # Default: list sources and show stats
        list_sources(store, limit=10000)
        show_stats(store, bm25)

    return 0


if __name__ == "__main__":
    sys.exit(main())
