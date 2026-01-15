#!/usr/bin/env python3
"""
Calibrate int8 quantization ranges from existing embeddings.

This tool samples embeddings from the vector store and calculates
per-dimension min/max ranges needed for int8 quantization.

Usage:
    python tools/calibrate_int8_ranges.py --sample-size 100000 --output data/int8_ranges.npy
    python tools/calibrate_int8_ranges.py --backend redis --sample-size 50000
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from radiant.config import load_config
from radiant.storage.factory import get_vector_store

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sample_embeddings_redis(store, sample_size: int) -> np.ndarray:
    """Sample embeddings from Redis store."""
    logger.info("Sampling from Redis store...")
    
    doc_ids = store.list_doc_ids_with_embeddings(limit=min(sample_size, 500_000))
    
    if not doc_ids:
        raise ValueError("No documents with embeddings found in Redis")
    
    logger.info(f"Found {len(doc_ids)} documents with embeddings")
    
    embeddings = []
    for doc_id in tqdm(doc_ids[:sample_size], desc="Loading embeddings"):
        try:
            key = store._doc_key(doc_id)
            emb_bytes = store._r.hget(key, "embedding")
            if emb_bytes:
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                embeddings.append(emb)
        except Exception as e:
            logger.debug(f"Failed to load embedding for {doc_id}: {e}")
            continue
    
    if not embeddings:
        raise ValueError("No embeddings could be loaded from Redis")
    
    return np.array(embeddings)


def sample_embeddings_chroma(store, sample_size: int) -> np.ndarray:
    """Sample embeddings from Chroma store."""
    logger.info("Sampling from Chroma store...")
    
    doc_ids = store.list_doc_ids_with_embeddings(limit=min(sample_size, 500_000))
    
    if not doc_ids:
        raise ValueError("No documents with embeddings found in Chroma")
    
    logger.info(f"Found {len(doc_ids)} documents with embeddings")
    
    embeddings = []
    for doc_id in tqdm(doc_ids[:sample_size], desc="Loading embeddings"):
        try:
            result = store._collection.get(
                ids=[doc_id],
                include=["embeddings"],
            )
            if result["embeddings"] and result["embeddings"][0]:
                emb = np.array(result["embeddings"][0], dtype=np.float32)
                embeddings.append(emb)
        except Exception as e:
            logger.debug(f"Failed to load embedding for {doc_id}: {e}")
            continue
    
    if not embeddings:
        raise ValueError("No embeddings could be loaded from Chroma")
    
    return np.array(embeddings)


def sample_embeddings_pgvector(store, sample_size: int) -> np.ndarray:
    """Sample embeddings from PgVector store."""
    logger.info("Sampling from PgVector store...")
    
    doc_ids = store.list_doc_ids_with_embeddings(limit=min(sample_size, 500_000))
    
    if not doc_ids:
        raise ValueError("No documents with embeddings found in PgVector")
    
    logger.info(f"Found {len(doc_ids)} documents with embeddings")
    
    embeddings = []
    store._ensure_connection()
    
    with store._conn.cursor() as cur:
        for doc_id in tqdm(doc_ids[:sample_size], desc="Loading embeddings"):
            try:
                cur.execute(
                    f"SELECT embedding FROM {store._leaf_table} WHERE id = %s",
                    (doc_id,)
                )
                row = cur.fetchone()
                if row and row[0]:
                    # Parse pgvector format
                    emb_str = row[0]
                    if isinstance(emb_str, str):
                        emb_str = emb_str.strip('[]')
                        emb = np.array([float(x) for x in emb_str.split(',')], dtype=np.float32)
                        embeddings.append(emb)
            except Exception as e:
                logger.debug(f"Failed to load embedding for {doc_id}: {e}")
                continue
    
    if not embeddings:
        raise ValueError("No embeddings could be loaded from PgVector")
    
    return np.array(embeddings)


def sample_embeddings(store, sample_size: int, backend: str) -> np.ndarray:
    """Sample random embeddings from the store."""
    if backend == "redis":
        return sample_embeddings_redis(store, sample_size)
    elif backend == "chroma":
        return sample_embeddings_chroma(store, sample_size)
    elif backend == "pgvector":
        return sample_embeddings_pgvector(store, sample_size)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def calculate_ranges(embeddings: np.ndarray) -> np.ndarray:
    """Calculate per-dimension min/max ranges for int8 quantization."""
    logger.info(f"Calculating ranges for {embeddings.shape[1]} dimensions...")
    
    ranges = np.vstack([
        np.min(embeddings, axis=0),
        np.max(embeddings, axis=0)
    ])
    
    logger.info("Calibration statistics:")
    logger.info(f"  Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"  Sample size: {embeddings.shape[0]:,}")
    logger.info(f"  Global min: {ranges[0].min():.6f}")
    logger.info(f"  Global max: {ranges[1].max():.6f}")
    logger.info(f"  Mean range: {np.mean(ranges[1] - ranges[0]):.6f}")
    
    return ranges


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate int8 quantization ranges from existing embeddings"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Number of embeddings to sample (default: 100,000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/int8_ranges.npy",
        help="Output path for calibration ranges (default: data/int8_ranges.npy)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (optional)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["redis", "chroma", "pgvector"],
        help="Force specific backend (optional, auto-detected from config)"
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config) if args.config else load_config()
        logger.info("Loaded configuration")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Determine backend
    backend = args.backend if args.backend else config.storage.backend
    logger.info(f"Using backend: {backend}")
    
    # Get store
    logger.info("Connecting to vector store...")
    try:
        store = get_vector_store(config)
        if not store.ping():
            logger.error("Failed to connect to vector store")
            sys.exit(1)
        logger.info("Connected successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        sys.exit(1)
    
    # Check document count
    try:
        doc_count = store.count_documents()
        logger.info(f"Total documents in store: {doc_count:,}")
        
        if doc_count == 0:
            logger.error("No documents found in vector store. Please ingest documents first.")
            sys.exit(1)
        
        if doc_count < args.sample_size:
            logger.warning(
                f"Requested sample size ({args.sample_size:,}) exceeds "
                f"available documents ({doc_count:,}). Using all documents."
            )
            args.sample_size = doc_count
    except Exception as e:
        logger.warning(f"Could not count documents: {e}")
    
    # Sample embeddings
    try:
        embeddings = sample_embeddings(store, args.sample_size, backend)
        logger.info(f"Successfully loaded {len(embeddings):,} embeddings")
    except Exception as e:
        logger.error(f"Failed to sample embeddings: {e}")
        sys.exit(1)
    
    # Calculate ranges
    try:
        ranges = calculate_ranges(embeddings)
    except Exception as e:
        logger.error(f"Failed to calculate ranges: {e}")
        sys.exit(1)
    
    # Save ranges
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, ranges)
        logger.info(f"✓ Saved calibration ranges to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save ranges: {e}")
        sys.exit(1)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Calibration completed successfully!")
    logger.info("="*60)
    logger.info(f"Output file: {output_path}")
    logger.info(f"Embeddings sampled: {len(embeddings):,}")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")
    logger.info("\nNext steps:")
    logger.info("1. Update config.yaml to enable quantization:")
    logger.info(f"   {backend}:")
    logger.info("     quantization:")
    logger.info("       enabled: true")
    logger.info(f"       int8_ranges_file: \"{output_path}\"")
    logger.info("2. Restart the application to use quantized retrieval")
    logger.info("="*60)


if __name__ == "__main__":
    main()
