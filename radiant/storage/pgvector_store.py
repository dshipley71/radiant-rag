"""
PostgreSQL with pgvector extension for Radiant Agentic RAG.

Uses PostgreSQL with the pgvector extension for efficient vector similarity search
with support for both exact and approximate (HNSW) nearest neighbor search.

Requirements:
    - psycopg2-binary >= 2.9.0 (or psycopg2)
    - pgvector Python bindings (optional, for better vector handling)
    - PostgreSQL server with pgvector extension installed
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from radiant.config import PgVectorConfig
from radiant.storage.base import BaseVectorStore, StoredDoc
from radiant.storage.quantization import (
    quantize_embeddings,
    embedding_to_bytes as quant_embedding_to_bytes,
    bytes_to_embedding as quant_bytes_to_embedding,
    rescore_candidates,
    QUANTIZATION_AVAILABLE,
)

# Try to import psycopg2
PSYCOPG2_AVAILABLE = False
_PSYCOPG2_IMPORT_ERROR = None
try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import execute_values, Json, RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError as e:
    _PSYCOPG2_IMPORT_ERROR = str(e)
    psycopg2 = None
    sql = None

logger = logging.getLogger(__name__)

# Log import status
if _PSYCOPG2_IMPORT_ERROR:
    logger.debug(f"psycopg2 not available: {_PSYCOPG2_IMPORT_ERROR}")


class PgVectorStore(BaseVectorStore):
    """
    PostgreSQL with pgvector extension for document and vector storage.
    
    Storage schema:
        Leaf documents table (with embeddings):
            - id: VARCHAR PRIMARY KEY (document ID)
            - content: TEXT (document text)
            - embedding: VECTOR(dim) (vector embedding)
            - meta: JSONB (metadata)
            - doc_level: VARCHAR (parent/child)
            - parent_id: VARCHAR (parent document ID)
            - language_code: VARCHAR (language code)
            - has_embedding: BOOLEAN
            - created_at: TIMESTAMP
            - updated_at: TIMESTAMP
        
        Parent documents table (without embeddings):
            - id: VARCHAR PRIMARY KEY (document ID)
            - content: TEXT (document text)
            - meta: JSONB (metadata)
            - doc_level: VARCHAR (parent/child)
            - parent_id: VARCHAR (parent document ID)
            - language_code: VARCHAR (language code)
            - created_at: TIMESTAMP
            - updated_at: TIMESTAMP
    
    Supports:
        - Exact nearest neighbor search
        - HNSW approximate nearest neighbor search
        - Cosine similarity, inner product, L2 distance
        - Metadata filtering
    """

    def __init__(self, config: PgVectorConfig) -> None:
        """
        Initialize PgVector store.
        
        Args:
            config: PgVector configuration
            
        Raises:
            ImportError: If psycopg2 is not installed
            ValueError: If connection string is not provided
        """
        if not PSYCOPG2_AVAILABLE:
            raise ImportError(
                f"psycopg2 is not available. Install with: pip install psycopg2-binary\n"
                f"Original error: {_PSYCOPG2_IMPORT_ERROR}"
            )
        
        # Get connection string from config or environment
        conn_str = config.connection_string
        if not conn_str:
            conn_str = os.environ.get("PG_CONN_STR")
        
        if not conn_str:
            raise ValueError(
                "PostgreSQL connection string not provided. "
                "Set pgvector.connection_string in config or PG_CONN_STR environment variable. "
                "Format: postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"
            )
        
        self._config = config
        self._conn_str = conn_str
        self._max_chars = config.max_content_chars
        self._embedding_dim = config.embedding_dimension
        self._leaf_table = config.leaf_table_name
        self._parent_table = config.parent_table_name
        
        # Map vector functions to pgvector operators
        self._vector_ops = {
            "cosine_similarity": ("<=>", "1 - ({})"),  # Returns distance, convert to similarity
            "inner_product": ("<#>", "-({})"),  # Returns negative inner product
            "l2_distance": ("<->", "1 / (1 + ({}))"),  # Returns L2 distance, convert to similarity
        }
        
        # Get operator for configured function
        if config.vector_function not in self._vector_ops:
            logger.warning(
                f"Unknown vector function '{config.vector_function}', "
                f"defaulting to cosine_similarity"
            )
            self._vector_op, self._similarity_transform = self._vector_ops["cosine_similarity"]
        else:
            self._vector_op, self._similarity_transform = self._vector_ops[config.vector_function]
        
        # Quantization configuration
        self._quant_config = config.quantization
        self._binary_table = f"{self._leaf_table}_binary"
        self._int8_table = f"{self._leaf_table}_int8"
        
        # Load int8 calibration ranges
        self._int8_ranges: Optional[np.ndarray] = None
        if self._quant_config.enabled and self._quant_config.int8_ranges_file:
            try:
                self._int8_ranges = np.load(self._quant_config.int8_ranges_file)
                logger.info("Loaded int8 calibration ranges")
            except Exception as e:
                logger.warning(f"Failed to load int8 ranges: {e}")
        
        # Initialize connection
        self._conn: Optional[Any] = None
        self._connect()
        
        # Setup tables
        self._setup_tables()
        
        # Setup quantized tables if enabled
        if self._quant_config.enabled and QUANTIZATION_AVAILABLE:
            self._setup_quantized_tables()
        
        logger.info(
            f"Initialized PgVector store with tables "
            f"'{self._leaf_table}' and '{self._parent_table}'"
        )

    def _connect(self) -> None:
        """Establish database connection."""
        try:
            self._conn = psycopg2.connect(self._conn_str)
            self._conn.autocommit = True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def _ensure_connection(self) -> None:
        """Ensure database connection is active."""
        try:
            if self._conn is None or self._conn.closed:
                self._connect()
            else:
                # Test connection
                with self._conn.cursor() as cur:
                    cur.execute("SELECT 1")
        except Exception:
            self._connect()

    def _setup_tables(self) -> None:
        """Create tables and indexes if they don't exist."""
        self._ensure_connection()
        
        with self._conn.cursor() as cur:
            # Ensure pgvector extension is available
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Drop tables if recreate is enabled
            if self._config.recreate_table:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier(self._leaf_table)
                ))
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier(self._parent_table)
                ))
                logger.info("Dropped existing tables for recreation")
            
            # Create leaf documents table (with embeddings)
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    id VARCHAR(256) PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(%s),
                    meta JSONB DEFAULT '{}',
                    doc_level VARCHAR(32) DEFAULT 'child',
                    parent_id VARCHAR(256) DEFAULT '',
                    language_code VARCHAR(16) DEFAULT 'en',
                    has_embedding BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """).format(sql.Identifier(self._leaf_table)), (self._embedding_dim,))
            
            # Create parent documents table (without embeddings)
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    id VARCHAR(256) PRIMARY KEY,
                    content TEXT NOT NULL,
                    meta JSONB DEFAULT '{}',
                    doc_level VARCHAR(32) DEFAULT 'parent',
                    parent_id VARCHAR(256) DEFAULT '',
                    language_code VARCHAR(16) DEFAULT 'en',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """).format(sql.Identifier(self._parent_table)))
            
            # Create indexes for common queries
            cur.execute(sql.SQL("""
                CREATE INDEX IF NOT EXISTS {}_doc_level_idx ON {} (doc_level)
            """).format(
                sql.Identifier(f"{self._leaf_table}_doc_level_idx"),
                sql.Identifier(self._leaf_table)
            ))
            
            cur.execute(sql.SQL("""
                CREATE INDEX IF NOT EXISTS {}_parent_id_idx ON {} (parent_id)
            """).format(
                sql.Identifier(f"{self._leaf_table}_parent_id_idx"),
                sql.Identifier(self._leaf_table)
            ))
            
            cur.execute(sql.SQL("""
                CREATE INDEX IF NOT EXISTS {}_language_idx ON {} (language_code)
            """).format(
                sql.Identifier(f"{self._leaf_table}_language_idx"),
                sql.Identifier(self._leaf_table)
            ))
            
            # Create HNSW index if configured
            if self._config.search_strategy == "hnsw":
                self._create_hnsw_index(cur)
        
        self._conn.commit()

    def _create_hnsw_index(self, cur) -> None:
        """Create HNSW index for vector similarity search."""
        index_name = f"{self._leaf_table}_embedding_hnsw_idx"
        
        # Check if index exists
        cur.execute("""
            SELECT 1 FROM pg_indexes 
            WHERE indexname = %s
        """, (index_name,))
        
        if cur.fetchone():
            if self._config.hnsw_recreate_index_if_exists:
                cur.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(
                    sql.Identifier(index_name)
                ))
                logger.info(f"Dropped existing HNSW index '{index_name}'")
            else:
                logger.debug(f"HNSW index '{index_name}' already exists")
                return
        
        # Map distance function to index operator class
        ops_class_map = {
            "cosine_similarity": "vector_cosine_ops",
            "inner_product": "vector_ip_ops",
            "l2_distance": "vector_l2_ops",
        }
        ops_class = ops_class_map.get(
            self._config.vector_function, "vector_cosine_ops"
        )
        
        # Build HNSW index creation kwargs
        hnsw_kwargs = dict(self._config.hnsw_index_creation_kwargs)
        
        # Default HNSW parameters if not specified
        m = hnsw_kwargs.pop("m", 16)
        ef_construction = hnsw_kwargs.pop("ef_construction", 64)
        
        # Create HNSW index
        index_sql = sql.SQL("""
            CREATE INDEX {} ON {} 
            USING hnsw (embedding {})
            WITH (m = %s, ef_construction = %s)
        """).format(
            sql.Identifier(index_name),
            sql.Identifier(self._leaf_table),
            sql.SQL(ops_class),
        )
        
        try:
            cur.execute(index_sql, (m, ef_construction))
            logger.info(
                f"Created HNSW index '{index_name}' with m={m}, "
                f"ef_construction={ef_construction}"
            )
        except Exception as e:
            logger.warning(f"Failed to create HNSW index: {e}")

    def _setup_quantized_tables(self) -> None:
        """Create tables for binary and int8 embeddings."""
        self._ensure_connection()
        cursor = self._conn.cursor()
        
        try:
            # Binary embeddings table (stored as BYTEA)
            if self._quant_config.precision in ("binary", "both"):
                cursor.execute(sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {} (
                        id VARCHAR PRIMARY KEY,
                        binary_embedding BYTEA NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """).format(sql.Identifier(self._binary_table)))
                logger.info(f"Binary embeddings table '{self._binary_table}' ready")
            
            # Int8 embeddings table
            if self._quant_config.precision in ("int8", "both"):
                cursor.execute(sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {} (
                        id VARCHAR PRIMARY KEY,
                        int8_embedding BYTEA NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """).format(sql.Identifier(self._int8_table)))
                logger.info(f"Int8 embeddings table '{self._int8_table}' ready")
            
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            logger.warning(f"Failed to setup quantized tables: {e}")
        finally:
            cursor.close()

    def ping(self) -> bool:
        """Test PostgreSQL connection."""
        try:
            self._ensure_connection()
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL connection check failed: {e}")
            return False

    def make_doc_id(self, content: str, meta: Optional[Dict[str, Any]] = None) -> str:
        """Generate deterministic document ID from content and metadata."""
        return self._default_make_doc_id(content, meta)

    def _embedding_to_pgvector(self, embedding: List[float]) -> str:
        """Convert embedding list to pgvector string format."""
        return "[" + ",".join(str(x) for x in embedding) + "]"

    def upsert(
        self,
        doc_id: str,
        content: str,
        embedding: List[float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a document with embedding."""
        self._ensure_connection()
        
        # Truncate content if necessary
        truncated = False
        if len(content) > self._max_chars:
            content = content[: self._max_chars]
            truncated = True
        
        # Prepare metadata
        meta = dict(meta or {})
        if truncated:
            meta["truncated"] = True
        
        doc_level = str(meta.get("doc_level", "child"))
        parent_id = str(meta.get("parent_id", ""))
        language_code = str(meta.get("language_code", "en"))
        
        embedding_str = self._embedding_to_pgvector(embedding)
        
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""
                INSERT INTO {} (id, content, embedding, meta, doc_level, parent_id, language_code, has_embedding, updated_at)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s, TRUE, CURRENT_TIMESTAMP)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    meta = EXCLUDED.meta,
                    doc_level = EXCLUDED.doc_level,
                    parent_id = EXCLUDED.parent_id,
                    language_code = EXCLUDED.language_code,
                    has_embedding = TRUE,
                    updated_at = CURRENT_TIMESTAMP
            """).format(sql.Identifier(self._leaf_table)), (
                doc_id, content, embedding_str, Json(meta),
                doc_level, parent_id, language_code
            ))
            
            # Store quantized versions if enabled
            if self._quant_config.enabled and QUANTIZATION_AVAILABLE:
                try:
                    embedding_array = np.array([embedding], dtype=np.float32)
                    
                    # Binary
                    if self._quant_config.precision in ("binary", "both"):
                        try:
                            binary_emb = quantize_embeddings(embedding_array, precision="ubinary")[0]
                            cur.execute(sql.SQL("""
                                INSERT INTO {} (id, binary_embedding)
                                VALUES (%s, %s)
                                ON CONFLICT (id) DO UPDATE SET binary_embedding = EXCLUDED.binary_embedding
                            """).format(sql.Identifier(self._binary_table)),
                                (doc_id, psycopg2.Binary(quant_embedding_to_bytes(binary_emb)))
                            )
                        except Exception as e:
                            logger.debug(f"Failed to store binary embedding: {e}")
                    
                    # Int8
                    if self._quant_config.precision in ("int8", "both"):
                        try:
                            int8_emb = quantize_embeddings(
                                embedding_array,
                                precision="int8",
                                ranges=self._int8_ranges
                            )[0]
                            cur.execute(sql.SQL("""
                                INSERT INTO {} (id, int8_embedding)
                                VALUES (%s, %s)
                                ON CONFLICT (id) DO UPDATE SET int8_embedding = EXCLUDED.int8_embedding
                            """).format(sql.Identifier(self._int8_table)),
                                (doc_id, psycopg2.Binary(quant_embedding_to_bytes(int8_emb)))
                            )
                        except Exception as e:
                            logger.debug(f"Failed to store int8 embedding: {e}")
                except Exception as e:
                    logger.warning(f"Quantization storage failed for {doc_id}: {e}")
        
        logger.debug(f"Upserted document with embedding: {doc_id}")

    def upsert_doc_only(
        self,
        doc_id: str,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store document without embedding (for parent documents)."""
        self._ensure_connection()
        
        # Truncate content if necessary
        truncated = False
        if len(content) > self._max_chars:
            content = content[: self._max_chars]
            truncated = True
        
        # Prepare metadata
        meta = dict(meta or {})
        if truncated:
            meta["truncated"] = True
        
        doc_level = str(meta.get("doc_level", "parent"))
        parent_id = str(meta.get("parent_id", ""))
        language_code = str(meta.get("language_code", "en"))
        
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""
                INSERT INTO {} (id, content, meta, doc_level, parent_id, language_code, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    meta = EXCLUDED.meta,
                    doc_level = EXCLUDED.doc_level,
                    parent_id = EXCLUDED.parent_id,
                    language_code = EXCLUDED.language_code,
                    updated_at = CURRENT_TIMESTAMP
            """).format(sql.Identifier(self._parent_table)), (
                doc_id, content, Json(meta),
                doc_level, parent_id, language_code
            ))
        
        logger.debug(f"Upserted document (no embedding): {doc_id}")

    def upsert_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """Batch insert/update documents with embeddings."""
        if not documents:
            return 0
        
        self._ensure_connection()
        
        values = []
        for doc in documents:
            doc_id = doc["doc_id"]
            content = doc["content"]
            embedding = doc["embedding"]
            meta = dict(doc.get("meta") or {})
            
            # Truncate content if necessary
            if len(content) > self._max_chars:
                content = content[: self._max_chars]
                meta["truncated"] = True
            
            doc_level = str(meta.get("doc_level", "child"))
            parent_id = str(meta.get("parent_id", ""))
            language_code = str(meta.get("language_code", "en"))
            embedding_str = self._embedding_to_pgvector(embedding)
            
            values.append((
                doc_id, content, embedding_str, Json(meta),
                doc_level, parent_id, language_code
            ))
        
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                sql.SQL("""
                    INSERT INTO {} (id, content, embedding, meta, doc_level, parent_id, language_code, has_embedding, updated_at)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        meta = EXCLUDED.meta,
                        doc_level = EXCLUDED.doc_level,
                        parent_id = EXCLUDED.parent_id,
                        language_code = EXCLUDED.language_code,
                        has_embedding = TRUE,
                        updated_at = CURRENT_TIMESTAMP
                """).format(sql.Identifier(self._leaf_table)).as_string(self._conn),
                values,
                template="(%s, %s, %s::vector, %s, %s, %s, %s, TRUE, CURRENT_TIMESTAMP)",
            )
        
        logger.debug(f"Batch upserted {len(documents)} documents with embeddings")
        return len(documents)

    def upsert_doc_only_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """Batch store documents without embeddings."""
        if not documents:
            return 0
        
        self._ensure_connection()
        
        values = []
        for doc in documents:
            doc_id = doc["doc_id"]
            content = doc["content"]
            meta = dict(doc.get("meta") or {})
            
            # Truncate content if necessary
            if len(content) > self._max_chars:
                content = content[: self._max_chars]
                meta["truncated"] = True
            
            doc_level = str(meta.get("doc_level", "parent"))
            parent_id = str(meta.get("parent_id", ""))
            language_code = str(meta.get("language_code", "en"))
            
            values.append((
                doc_id, content, Json(meta),
                doc_level, parent_id, language_code
            ))
        
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                sql.SQL("""
                    INSERT INTO {} (id, content, meta, doc_level, parent_id, language_code, updated_at)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        meta = EXCLUDED.meta,
                        doc_level = EXCLUDED.doc_level,
                        parent_id = EXCLUDED.parent_id,
                        language_code = EXCLUDED.language_code,
                        updated_at = CURRENT_TIMESTAMP
                """).format(sql.Identifier(self._parent_table)).as_string(self._conn),
                values,
                template="(%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)",
            )
        
        logger.debug(f"Batch upserted {len(documents)} documents (no embedding)")
        return len(documents)

    def get_doc(self, doc_id: str) -> Optional[StoredDoc]:
        """Retrieve a document by ID."""
        self._ensure_connection()
        
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Try leaf table first
            cur.execute(sql.SQL("""
                SELECT id, content, meta FROM {} WHERE id = %s
            """).format(sql.Identifier(self._leaf_table)), (doc_id,))
            
            row = cur.fetchone()
            
            if not row:
                # Try parent table
                cur.execute(sql.SQL("""
                    SELECT id, content, meta FROM {} WHERE id = %s
                """).format(sql.Identifier(self._parent_table)), (doc_id,))
                row = cur.fetchone()
            
            if not row:
                return None
            
            content = row["content"] or ""
            meta = dict(row["meta"]) if row["meta"] else {}
            
            return StoredDoc(doc_id=doc_id, content=content, meta=meta)

    def has_embedding(self, doc_id: str) -> bool:
        """Check if document has an embedding stored."""
        self._ensure_connection()
        
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""
                SELECT has_embedding FROM {} WHERE id = %s
            """).format(sql.Identifier(self._leaf_table)), (doc_id,))
            
            row = cur.fetchone()
            return bool(row and row[0])

    def delete_doc(self, doc_id: str) -> bool:
        """Delete a document from both tables."""
        self._ensure_connection()
        
        deleted = False
        with self._conn.cursor() as cur:
            # Try leaf table
            cur.execute(sql.SQL("""
                DELETE FROM {} WHERE id = %s
            """).format(sql.Identifier(self._leaf_table)), (doc_id,))
            
            if cur.rowcount > 0:
                deleted = True
            
            # Try parent table
            cur.execute(sql.SQL("""
                DELETE FROM {} WHERE id = %s
            """).format(sql.Identifier(self._parent_table)), (doc_id,))
            
            if cur.rowcount > 0:
                deleted = True
        
        if deleted:
            logger.debug(f"Deleted document: {doc_id}")
        
        return deleted

    def retrieve_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int,
        min_similarity: float = 0.0,
        ef_runtime: Optional[int] = None,
        language_filter: Optional[str] = None,
        doc_level_filter: Optional[str] = None,
    ) -> List[Tuple[StoredDoc, float]]:
        """Retrieve documents by vector similarity."""
        self._ensure_connection()
        
        # Normalize doc_level_filter
        level_value = None
        if doc_level_filter:
            filter_lower = doc_level_filter.lower()
            if filter_lower in ("child", "leaves", "leaf"):
                level_value = "child"
            elif filter_lower in ("parent", "parents"):
                level_value = "parent"
            # "all" or None means no doc_level filter
        
        embedding_str = self._embedding_to_pgvector(query_embedding)
        
        # Build WHERE clause
        where_conditions = ["has_embedding = TRUE"]
        params: List[Any] = [embedding_str]
        
        if language_filter:
            where_conditions.append("language_code = %s")
            params.append(language_filter)
        
        if level_value:
            where_conditions.append("doc_level = %s")
            params.append(level_value)
        
        where_clause = " AND ".join(where_conditions)
        
        # Set HNSW ef_search for this query if specified
        ef_search = ef_runtime or self._config.hnsw_ef_search
        
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            if ef_search and self._config.search_strategy == "hnsw":
                cur.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
            
            # Build similarity expression
            distance_expr = f"embedding {self._vector_op} %s::vector"
            similarity_expr = self._similarity_transform.format(distance_expr)
            
            query = sql.SQL("""
                SELECT 
                    id, 
                    content, 
                    meta,
                    {} AS similarity
                FROM {}
                WHERE {}
                ORDER BY embedding {} %s::vector
                LIMIT %s
            """).format(
                sql.SQL(similarity_expr),
                sql.Identifier(self._leaf_table),
                sql.SQL(where_clause),
                sql.SQL(self._vector_op),
            )
            
            params.extend([embedding_str, top_k])
            
            try:
                cur.execute(query, params)
            except Exception as e:
                logger.error(f"PgVector query failed: {e}")
                return []
            
            docs: List[Tuple[StoredDoc, float]] = []
            
            for row in cur.fetchall():
                similarity = float(row["similarity"])
                
                if similarity < min_similarity:
                    continue
                
                content = row["content"] or ""
                meta = dict(row["meta"]) if row["meta"] else {}
                
                docs.append((
                    StoredDoc(doc_id=row["id"], content=content, meta=meta),
                    similarity,
                ))
            
            # Sort by similarity descending (should already be sorted)
            docs.sort(key=lambda x: x[1], reverse=True)
            return docs

    def retrieve_by_embedding_quantized(
        self,
        query_embedding: List[float],
        top_k: int,
        min_similarity: float = 0.0,
        rescore_multiplier: Optional[float] = None,
        use_rescoring: Optional[bool] = None,
        language_filter: Optional[str] = None,
        doc_level_filter: Optional[str] = None,
    ) -> List[Tuple[StoredDoc, float]]:
        """Retrieve using quantized embeddings with optional rescoring."""
        # Fall back if quantization not enabled
        if not self._quant_config.enabled or not QUANTIZATION_AVAILABLE:
            logger.debug("Quantization not enabled, using standard retrieval")
            return self.retrieve_by_embedding(
                query_embedding, top_k, min_similarity,
                language_filter=language_filter,
                doc_level_filter=doc_level_filter
            )
        
        rescore_mult = rescore_multiplier if rescore_multiplier is not None else self._quant_config.rescore_multiplier
        use_rescore = use_rescoring if use_rescoring is not None else self._quant_config.use_rescoring
        candidate_k = int(top_k * rescore_mult) if use_rescore else top_k
        
        # Stage 1: Get candidates using standard retrieval
        # (Binary HNSW search would require custom PostgreSQL extension)
        candidates = self.retrieve_by_embedding(
            query_embedding,
            candidate_k,
            min_similarity=0.0,
            language_filter=language_filter,
            doc_level_filter=doc_level_filter,
        )
        
        if not use_rescore or not candidates:
            return candidates[:top_k]
        
        # Stage 2: Rescore with int8 embeddings
        self._ensure_connection()
        query_vec = np.array(query_embedding, dtype=np.float32)
        candidate_embeddings: List[np.ndarray] = []
        candidate_docs: List[StoredDoc] = []
        
        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            for doc, _ in candidates:
                # Try int8 first
                if self._quant_config.precision in ("int8", "both"):
                    cur.execute(sql.SQL(
                        "SELECT int8_embedding FROM {} WHERE id = %s"
                    ).format(sql.Identifier(self._int8_table)), (doc.doc_id,))
                    row = cur.fetchone()
                    if row and row["int8_embedding"]:
                        emb = quant_bytes_to_embedding(
                            bytes(row["int8_embedding"]),
                            np.int8,
                            (self._embedding_dim,)
                        ).astype(np.float32)
                        candidate_embeddings.append(emb)
                        candidate_docs.append(doc)
                        continue
                
                # Fall back to float32
                cur.execute(sql.SQL(
                    "SELECT embedding FROM {} WHERE id = %s"
                ).format(sql.Identifier(self._leaf_table)), (doc.doc_id,))
                row = cur.fetchone()
                if row and row["embedding"]:
                    # Parse pgvector format
                    emb_str = row["embedding"]
                    if isinstance(emb_str, str):
                        emb_str = emb_str.strip('[]')
                        emb = np.array([float(x) for x in emb_str.split(',')], dtype=np.float32)
                        candidate_embeddings.append(emb)
                        candidate_docs.append(doc)
        
        if not candidate_embeddings:
            logger.warning("No embeddings loaded for rescoring")
            return candidates[:top_k]
        
        # Rescore
        rescored = rescore_candidates(query_vec, candidate_embeddings, [d.doc_id for d in candidate_docs])
        
        # Build results
        doc_map = {doc.doc_id: doc for doc in candidate_docs}
        results: List[Tuple[StoredDoc, float]] = []
        for doc_id, score in rescored[:top_k]:
            if score >= min_similarity and doc_id in doc_map:
                results.append((doc_map[doc_id], score))
        
        logger.debug(
            f"Quantized retrieval: {len(candidates)} candidates → "
            f"{len(rescored)} rescored → {len(results)} returned"
        )
        
        return results

    def list_doc_ids(self, pattern: str = "*", limit: int = 10_000) -> List[str]:
        """List document IDs from both tables."""
        self._ensure_connection()
        
        ids: List[str] = []
        
        with self._conn.cursor() as cur:
            # List from leaf table
            if pattern == "*":
                cur.execute(sql.SQL("""
                    SELECT id FROM {} LIMIT %s
                """).format(sql.Identifier(self._leaf_table)), (limit,))
            else:
                cur.execute(sql.SQL("""
                    SELECT id FROM {} WHERE id LIKE %s LIMIT %s
                """).format(sql.Identifier(self._leaf_table)), (
                    pattern.replace("*", "%"), limit
                ))
            
            ids.extend([row[0] for row in cur.fetchall()])
            
            remaining = limit - len(ids)
            if remaining > 0:
                # List from parent table
                if pattern == "*":
                    cur.execute(sql.SQL("""
                        SELECT id FROM {} LIMIT %s
                    """).format(sql.Identifier(self._parent_table)), (remaining,))
                else:
                    cur.execute(sql.SQL("""
                        SELECT id FROM {} WHERE id LIKE %s LIMIT %s
                    """).format(sql.Identifier(self._parent_table)), (
                        pattern.replace("*", "%"), remaining
                    ))
                
                ids.extend([row[0] for row in cur.fetchall()])
        
        return ids

    def list_doc_ids_with_embeddings(self, limit: int = 10_000) -> List[str]:
        """List document IDs that have embeddings stored."""
        self._ensure_connection()
        
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""
                SELECT id FROM {} WHERE has_embedding = TRUE LIMIT %s
            """).format(sql.Identifier(self._leaf_table)), (limit,))
            
            return [row[0] for row in cur.fetchall()]

    def get_index_info(self) -> Dict[str, Any]:
        """Get PgVector table and index statistics."""
        self._ensure_connection()
        
        info = {
            "backend": "pgvector",
            "leaf_table": self._leaf_table,
            "parent_table": self._parent_table,
            "embedding_dimension": self._embedding_dim,
            "vector_function": self._config.vector_function,
            "search_strategy": self._config.search_strategy,
        }
        
        with self._conn.cursor() as cur:
            # Count leaf documents
            cur.execute(sql.SQL("""
                SELECT COUNT(*) FROM {}
            """).format(sql.Identifier(self._leaf_table)))
            info["leaf_document_count"] = cur.fetchone()[0]
            
            # Count documents with embeddings
            cur.execute(sql.SQL("""
                SELECT COUNT(*) FROM {} WHERE has_embedding = TRUE
            """).format(sql.Identifier(self._leaf_table)))
            info["embedded_document_count"] = cur.fetchone()[0]
            
            # Count parent documents
            cur.execute(sql.SQL("""
                SELECT COUNT(*) FROM {}
            """).format(sql.Identifier(self._parent_table)))
            info["parent_document_count"] = cur.fetchone()[0]
            
            # Total document count
            info["document_count"] = info["leaf_document_count"] + info["parent_document_count"]
            
            # Check for HNSW index
            index_name = f"{self._leaf_table}_embedding_hnsw_idx"
            cur.execute("""
                SELECT indexdef FROM pg_indexes WHERE indexname = %s
            """, (index_name,))
            
            row = cur.fetchone()
            if row:
                info["hnsw_index"] = {
                    "name": index_name,
                    "exists": True,
                    "definition": row[0],
                }
            else:
                info["hnsw_index"] = {
                    "name": index_name,
                    "exists": False,
                }
        
        return info

    def drop_index(self, delete_documents: bool = False) -> bool:
        """Drop the HNSW index and optionally delete all documents."""
        self._ensure_connection()
        
        try:
            with self._conn.cursor() as cur:
                # Drop HNSW index
                index_name = f"{self._leaf_table}_embedding_hnsw_idx"
                cur.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(
                    sql.Identifier(index_name)
                ))
                logger.info(f"Dropped HNSW index '{index_name}'")
                
                if delete_documents:
                    # Truncate both tables
                    cur.execute(sql.SQL("TRUNCATE TABLE {} CASCADE").format(
                        sql.Identifier(self._leaf_table)
                    ))
                    cur.execute(sql.SQL("TRUNCATE TABLE {} CASCADE").format(
                        sql.Identifier(self._parent_table)
                    ))
                    logger.info("Deleted all documents from tables")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to drop index: {e}")
            return False

    def count_documents(self) -> int:
        """Count total documents stored in both tables."""
        self._ensure_connection()
        
        with self._conn.cursor() as cur:
            cur.execute(sql.SQL("""
                SELECT 
                    (SELECT COUNT(*) FROM {}) + 
                    (SELECT COUNT(*) FROM {})
            """).format(
                sql.Identifier(self._leaf_table),
                sql.Identifier(self._parent_table),
            ))
            
            return cur.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.debug("Closed PostgreSQL connection")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
