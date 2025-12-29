# Radiant Agentic RAG

A production-quality Agentic Retrieval-Augmented Generation (RAG) system with multi-agent architecture, hybrid search, and professional reporting.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Agent Pipeline](#agent-pipeline)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Query Pipeline](#query-pipeline)
- [GitHub Repository Ingestion](#github-repository-ingestion)
- [Code-Aware Chunking](#code-aware-chunking)
- [Multilingual Support](#multilingual-support)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

Radiant RAG is an enterprise-grade retrieval-augmented generation system that combines:

- **Multi-agent orchestration** for intelligent query processing
- **Hybrid search** combining dense embeddings and BM25 sparse retrieval
- **GitHub repository ingestion** with code-aware chunking
- **Multilingual support** with automatic language detection and translation
- **Professional reporting** in multiple formats

### Key Features

| Category | Features |
|----------|----------|
| **Retrieval** | Dense (HNSW), BM25, Hybrid (RRF fusion), Web Search |
| **Agents** | 15+ specialized agents for planning, retrieval, post-processing |
| **Ingestion** | Files, URLs, GitHub repos, with code-aware chunking |
| **Languages** | 176 languages detected, LLM-based translation |
| **Output** | Markdown, HTML, JSON, Text reports |
| **Interfaces** | CLI, TUI (Textual), Python API |

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RADIANT RAG SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   CLI/TUI   │    │  Python API │    │   Config    │    │   Reports   │   │
│  │  Interface  │    │   Access    │    │   (YAML)    │    │  Generator  │   │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘   │
│         │                  │                  │                  │          │
│         └──────────────────┼──────────────────┼──────────────────┘          │
│                            ▼                  ▼                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        RADIANT RAG APPLICATION                      │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │                      AGENTIC ORCHESTRATOR                     │  │    │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │  │    │
│  │  │  │Planning │→│  Query  │→│Retrieval│→│  Post-  │→│Generate │  │  │    │
│  │  │  │  Stage  │ │  Proc.  │ │  Stage  │ │Retrieval│ │  Stage  │  │  │    │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                            │                  │                             │
│         ┌──────────────────┼──────────────────┼──────────────────┐          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │    LLM      │    │   Redis     │    │    BM25     │    │   Local     │   │
│  │   Client    │    │Vector Store │    │    Index    │    │   Models    │   │
│  │  (Ollama)   │    │   (HNSW)    │    │ (Persistent)│    │(Embeddings) │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
                              USER QUERY
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            QUERY PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │  PLANNING   │  Analyze query complexity, select retrieval strategy       │
│  │    AGENT    │  Outputs: mode (hybrid/dense/bm25), decompose?, expand?    │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │   QUERY     │  Decompose complex queries into sub-queries                │
│  │DECOMPOSITION│  Example: "Compare X and Y" → ["What is X?", "What is Y?"] │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐  ┌─────────────┐                                           │
│  │   QUERY     │  │   QUERY     │  Rewrite for clarity, expand with         │
│  │  REWRITE    │→ │  EXPANSION  │  synonyms and related terms               │
│  └──────┬──────┘  └──────┬──────┘                                           │
│         │                │                                                  │
│         └───────┬────────┘                                                  │
│                 ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         RETRIEVAL STAGE                              │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │   │
│  │  │   DENSE     │    │    BM25     │    │ WEB SEARCH  │               │   │
│  │  │ RETRIEVAL   │    │ RETRIEVAL   │    │   (opt.)    │               │   │
│  │  │ (Embeddings)│    │  (Keywords) │    │             │               │   │
│  │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘               │   │
│  │         │                  │                  │                      │   │
│  │         └──────────────────┼──────────────────┘                      │   │
│  │                            ▼                                         │   │
│  │                     ┌─────────────┐                                  │   │
│  │                     │  RRF FUSION │  Reciprocal Rank Fusion          │   │
│  │                     └──────┬──────┘                                  │   │
│  └─────────────────────────────┼────────────────────────────────────────┘   │
│                                ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      POST-RETRIEVAL STAGE                            │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │   │
│  │  │ AUTO-MERGE  │ →  │  RERANKING  │ →  │  CONTEXT    │               │   │
│  │  │Hierarchical │    │ CrossEncoder│    │ EVALUATION  │               │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘               │   │
│  │         │                  │                  │                      │   │
│  │         ▼                  ▼                  ▼                      │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │   │
│  │  │SUMMARIZATION│    │  MULTI-HOP  │    │    FACT     │               │   │
│  │  │   (Long)    │    │  REASONING  │    │VERIFICATION │               │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                │                                            │
│                                ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        GENERATION STAGE                              │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │   │
│  │  │   ANSWER    │ →  │  CITATION   │ →  │   CRITIC    │               │   │
│  │  │  SYNTHESIS  │    │  TRACKING   │    │ EVALUATION  │               │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                │                                            │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 ▼
                          FINAL RESPONSE
                    (Answer + Citations + Score)
```

---

## Installation

### Prerequisites

- Python 3.10+
- Redis Stack (Redis + RediSearch module)
- CUDA-capable GPU (optional, for faster inference)

### Step 1: Install Redis Stack

```bash
# Using Docker (recommended)
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server

# Or install locally (Ubuntu)
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install redis-stack-server
```

### Step 2: Install Radiant RAG

```bash
# Clone or extract the package
git clone https://github.com/dshipley71/radiant-rag.git
cd radiant-rag

# Install as package (recommended)
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
# Required: LLM endpoint (Ollama or compatible)
export RADIANT_OLLAMA_OPENAI_BASE_URL="https://your-ollama-host/v1"
export RADIANT_OLLAMA_OPENAI_API_KEY="your-api-key"

# Optional: GitHub token for higher rate limits
export GITHUB_TOKEN="ghp_your_token_here"

# Optional: Redis connection (defaults to localhost:6379)
export RADIANT_REDIS_HOST="localhost"
export RADIANT_REDIS_PORT="6379"
```

---

## Quick Start

```bash
# 1. Ingest local documents
python -m radiant ingest ./documents/

# 2. Ingest from GitHub repository
python -m radiant ingest --url "https://github.com/owner/repo"

# 3. Query the system
python -m radiant query "What is the main topic of these documents?"

# 4. Interactive mode
python -m radiant interactive

# 5. Interactive TUI mode
python -m radiant interactive --tui
```

---

## CLI Reference

### Command Overview

```
python -m radiant <command> [options]

Commands:
  ingest       Ingest documents from files, directories, or URLs
  query        Query the RAG system with full pipeline
  search       Search documents (retrieval only, no LLM)
  interactive  Start interactive query mode
  stats        Display system statistics
  health       Check system health
  clear        Clear all indexed documents
  rebuild-bm25 Rebuild BM25 index from store
```

### ingest

Ingest documents from files, directories, or URLs.

```bash
python -m radiant ingest [paths...] [options]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--url URL` | `-u` | - | URL to ingest (repeatable) |
| `--flat` | - | false | Use flat storage (no hierarchy) |
| `--crawl-depth N` | - | config | Crawl depth for URLs |
| `--max-pages N` | - | config | Maximum pages to crawl |
| `--no-crawl` | - | false | Don't crawl, fetch single URL |
| `--auth USER:PASS` | - | - | Basic auth for URL ingestion |
| `--config PATH` | `-c` | config.yaml | Config file path |

**Examples:**

```bash
# Ingest local directory
python -m radiant ingest ./docs/

# Ingest GitHub repository (auto-detected)
python -m radiant ingest --url "https://github.com/owner/repo"

# Ingest website with crawling
python -m radiant ingest --url "https://docs.example.com" --crawl-depth 3

# Ingest multiple sources
python -m radiant ingest ./local/ --url "https://github.com/org/repo1" --url "https://github.com/org/repo2"

# Ingest with authentication
python -m radiant ingest --url "https://private.example.com" --auth "user:password"
```

### query

Query the RAG system with full agentic pipeline.

```bash
python -m radiant query "<question>" [options]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--mode MODE` | `-m` | hybrid | Retrieval mode: hybrid, dense, bm25 |
| `--conversation ID` | `-conv` | - | Continue conversation by ID |
| `--save PATH` | `-s` | - | Save report (.md, .html, .json, .txt) |
| `--compact` | - | false | Compact display format |
| `--simple` | - | false | Skip advanced agents (faster) |

**Examples:**

```bash
# Basic query
python -m radiant query "What is RAG?"

# Semantic search only
python -m radiant query "meaning of retrieval augmentation" --mode dense

# Keyword search only  
python -m radiant query "BM25 algorithm" --mode bm25

# Save report
python -m radiant query "Summarize the architecture" --save report.md

# Continue conversation
python -m radiant query "Tell me more about that" --conv abc123
```

### search

Search documents without LLM generation (retrieval only).

```bash
python -m radiant search "<query>" [options]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--mode MODE` | `-m` | hybrid | Retrieval mode |
| `--top-k N` | `-k` | 10 | Number of results |
| `--save PATH` | `-s` | - | Save results to file |

### clear

Clear all indexed documents.

```bash
python -m radiant clear [options]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--confirm` | false | Skip confirmation prompt |
| `--keep-bm25` | false | Keep BM25 index |

---

## Configuration

### Configuration File (config.yaml)

The system is configured via `config.yaml`. All settings can be overridden with environment variables prefixed with `RADIANT_`.

### Core Settings

```yaml
# LLM Configuration
llm:
  # Chat model for generation
  model: "gemma3:12b-cloud"
  
  # Temperature (0.0 = deterministic, 1.0 = creative)
  temperature: 0.3
  
  # Maximum tokens in response
  max_tokens: 4096

# Embedding Models
embeddings:
  # Sentence transformer model
  model: "sentence-transformers/all-MiniLM-L12-v2"
  
  # Embedding dimension (must match model)
  dimension: 384

# Reranking Model  
reranking:
  model: "cross-encoder/ms-marco-MiniLM-L12-v2"
  enabled: true
  top_k: 5
```

### Storage Backend Settings

Radiant RAG supports multiple vector storage backends. Choose the one that best fits your deployment needs:

| Backend | Use Case | Pros | Cons |
|---------|----------|------|------|
| **Redis** (default) | Production, low-latency | Fast, feature-rich, real-time | Requires Redis Stack |
| **Chroma** | Development, testing | Easy setup, embedded | Less scalable |
| **PgVector** | Enterprise, PostgreSQL shops | Mature, ACID, integrates with existing DB | More setup required |

```yaml
# Storage backend selection
storage:
  backend: redis  # Options: redis, chroma, pgvector

# Redis Configuration (default)
redis:
  url: "redis://localhost:6379/0"
  key_prefix: "radiant"
  vector_index:
    name: "radiant_vectors"
    hnsw_m: 16
    hnsw_ef_construction: 200
    distance_metric: "COSINE"

# Chroma Configuration (alternative)
chroma:
  persist_directory: "./data/chroma_db"
  collection_name: "radiant_docs"
  distance_fn: "cosine"

# PgVector Configuration (alternative)
pgvector:
  # Use PG_CONN_STR env var or set here
  connection_string: "postgresql://user:pass@localhost:5432/radiant"
  leaf_table_name: "haystack_leaves"
  parent_table_name: "haystack_parents"
  vector_function: "cosine_similarity"
  search_strategy: "hnsw"
```

To use Chroma, install the optional dependency:
```bash
pip install chromadb
```

To use PgVector, install PostgreSQL with the pgvector extension and the Python driver:
```bash
pip install psycopg2-binary
```

### Retrieval Settings

```yaml
retrieval:
  # Default retrieval mode
  default_mode: "hybrid"  # hybrid, dense, bm25
  
  # Number of documents to retrieve
  top_k: 10
  
  # Search scope for hierarchical storage
  # "leaves" - only search leaf chunks (default)
  # "parents" - only search parent documents (requires embed_parents: true)
  # "all" - search both leaves and parents (requires embed_parents: true)
  search_scope: "leaves"
  
  # Dense retrieval settings
  dense:
    # Number of candidates for HNSW
    ef_runtime: 200
  
  # BM25 settings
  bm25:
    k1: 1.5    # Term frequency saturation
    b: 0.75   # Length normalization
  
  # RRF fusion settings
  rrf:
    k: 60     # RRF constant (higher = more weight to lower ranks)
```

### Ingestion Settings

```yaml
ingestion:
  # Batch processing
  batch_enabled: true
  embedding_batch_size: 32  # Larger = faster, more memory
  redis_batch_size: 100
  
  # Chunking
  child_chunk_size: 512
  child_chunk_overlap: 50
  
  # Parent document settings (hierarchical mode)
  parent_chunk_size: 2048
  
  # Embed parent documents (enables parent retrieval)
  # When true, parent documents are embedded alongside child chunks
  # Required for search_scope: "parents" or "all"
  embed_parents: false
  
  # Progress display
  show_progress: true
```

### Web Crawler Settings

```yaml
web_crawler:
  # Maximum crawl depth (0 = seed URLs only)
  max_depth: 2
  
  # Maximum pages per crawl session
  max_pages: 100
  
  # Stay on same domain
  same_domain_only: true
  
  # Rate limiting (seconds between requests)
  delay: 0.5
  
  # Request timeout
  timeout: 30
  
  # URL patterns to exclude
  exclude_patterns:
    - ".*\\.(jpg|jpeg|png|gif|css|js)$"
    - ".*/login.*"
    - ".*/logout.*"
```

### GitHub Crawler Settings

```yaml
github_crawler:
  # Maximum files to fetch per repository
  max_files: 200
  
  # Rate limiting delay
  delay: 0.5
  
  # File extensions to include
  include_extensions:
    # Documentation
    - ".md"
    - ".txt"
    - ".rst"
    # Code
    - ".py"
    - ".js"
    - ".ts"
    - ".java"
    - ".go"
    - ".rs"
```

### Agent Settings

```yaml
# Planning Agent
planning:
  enabled: true
  
# Query Processing
query_decomposition:
  enabled: true
  max_sub_queries: 5

query_expansion:
  enabled: true
  max_expansions: 3

# Post-Retrieval
context_evaluation:
  enabled: true
  min_relevance_score: 0.3

summarization:
  enabled: true
  max_context_length: 8000

multihop_reasoning:
  enabled: true
  max_hops: 3

fact_verification:
  enabled: true
  min_confidence: 0.7

# Generation
citation:
  enabled: true
  style: "inline"  # inline, footnote, academic, enterprise

critic:
  enabled: true
  min_confidence: 0.6
```

### Environment Variable Overrides

All configuration values can be overridden with environment variables:

```bash
# Pattern: RADIANT_<SECTION>_<KEY>
export RADIANT_LLM_MODEL="llama3:8b"
export RADIANT_LLM_TEMPERATURE="0.5"
export RADIANT_RETRIEVAL_TOP_K="20"
export RADIANT_INGESTION_EMBEDDING_BATCH_SIZE="64"
```

---

## Agent Pipeline

### Agent Inventory

Radiant RAG includes 15+ specialized agents organized by pipeline stage:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AGENT INVENTORY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PLANNING STAGE                                                             │
│  ├── PlanningAgent          Analyze query, select retrieval strategy        │
│  └── StrategyMemoryAgent    Learn from successful retrieval patterns        │
│                                                                             │
│  QUERY PROCESSING STAGE                                                     │
│  ├── QueryDecompositionAgent  Break complex queries into sub-queries        │
│  ├── QueryRewriteAgent        Rewrite for clarity and precision             │
│  └── QueryExpansionAgent      Add synonyms and related terms                │
│                                                                             │
│  RETRIEVAL STAGE                                                            │
│  ├── DenseRetrievalAgent      Semantic search with embeddings               │
│  ├── BM25RetrievalAgent       Keyword search with BM25                      │
│  ├── WebSearchAgent           Real-time web search augmentation             │
│  └── RRFAgent                 Reciprocal Rank Fusion of results             │
│                                                                             │
│  POST-RETRIEVAL STAGE                                                       │
│  ├── HierarchicalAutoMergingAgent  Merge child chunks to parents            │
│  ├── CrossEncoderRerankingAgent    Rerank with cross-encoder model          │
│  ├── ContextEvaluationAgent        Score relevance of retrieved docs        │
│  ├── SummarizationAgent            Summarize long contexts                  │
│  ├── MultiHopReasoningAgent        Multi-step reasoning chains              │
│  └── FactVerificationAgent         Verify claims against context            │
│                                                                             │
│  GENERATION STAGE                                                           │
│  ├── AnswerSynthesisAgent          Generate final answer                    │
│  ├── CitationTrackingAgent         Add source citations                     │
│  └── CriticAgent                   Evaluate answer quality                  │
│                                                                             │
│  INGESTION AGENTS                                                           │
│  ├── IntelligentChunkingAgent      Semantic-aware chunking                  │
│  ├── LanguageDetectionAgent        Detect document language                 │
│  └── TranslationAgent              Translate to canonical language          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Execution Flow

```
Query: "Compare BM25 and dense retrieval for RAG systems"
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. PLANNING AGENT                                                           │
│    Input:  "Compare BM25 and dense retrieval for RAG systems"               │
│    Output: { mode: "hybrid", decompose: true, expand: true }                │
│    Reason: Comparison query needs both keyword (BM25) and semantic (dense)  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. QUERY DECOMPOSITION AGENT                                                │
│    Input:  "Compare BM25 and dense retrieval for RAG systems"               │
│    Output: [                                                                │
│      "What is BM25 retrieval?",                                             │
│      "What is dense retrieval with embeddings?",                            │
│      "How do BM25 and dense retrieval compare for RAG?"                     │
│    ]                                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. QUERY EXPANSION AGENT                                                    │
│    Input:  "What is BM25 retrieval?"                                        │
│    Output: "BM25 retrieval sparse lexical keyword term frequency TF-IDF"    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ 4a. DENSE RETRIEVAL AGENT     │ │ 4b. BM25 RETRIEVAL AGENT      │
│     Embedding similarity      │ │     Keyword matching          │
│     Returns: 10 documents     │ │     Returns: 10 documents     │
└───────────────────────────────┘ └───────────────────────────────┘
                        │                       │
                        └───────────┬───────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. RRF FUSION AGENT                                                         │
│    Combines dense + BM25 results with Reciprocal Rank Fusion                │
│    Output: 15 unique documents, ranked by combined score                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. CROSS-ENCODER RERANKING AGENT                                            │
│    Reranks with query-document cross-encoder                                │
│    Output: Top 5 documents with refined scores                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. CONTEXT EVALUATION AGENT                                                 │
│    Scores: [0.92, 0.87, 0.81, 0.76, 0.54]                                   │
│    Filters: Documents with score < 0.3 removed                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 8. ANSWER SYNTHESIS AGENT                                                   │
│    Generates comprehensive answer from context                              │
│    Output: "BM25 is a sparse retrieval method that uses term frequency..."  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 9. CITATION TRACKING AGENT                                                  │
│    Adds inline citations: "BM25 uses term frequency [1] and..."             │
│    Generates bibliography with source documents                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 10. CRITIC AGENT                                                            │
│     Evaluates: Completeness (0.85), Accuracy (0.90), Relevance (0.88)       │
│     Overall confidence: 0.87                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            FINAL RESPONSE
```

---

## Ingestion Pipeline

### Document Ingestion Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT SOURCES                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Local     │  │    URLs     │  │   GitHub    │  │   Images    │         │
│  │   Files     │  │  (Crawled)  │  │Repositories │  │   (VLM)     │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                │                │
│         └────────────────┼────────────────┼────────────────┘                │
│                          ▼                ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      DOCUMENT PROCESSOR                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │   Parse     │→ │  Language   │→ │ Translation │                   │   │
│  │  │  Document   │  │  Detection  │  │  (if needed)│                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                          │                                                  │
│                          ▼                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         CHUNKING                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │  HIERARCHICAL MODE (default)                                    │ │   │
│  │  │  ┌─────────────────────────────────────────────────────────┐    │ │   │
│  │  │  │  Parent Document (2048 tokens)                          │    │ │   │
│  │  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │    │ │   │
│  │  │  │  │ Child 1  │  │ Child 2  │  │ Child 3  │  │ Child 4  │ │    │ │   │
│  │  │  │  │ (512)    │  │ (512)    │  │ (512)    │  │ (512)    │ │    │ │   │
│  │  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │    │ │   │
│  │  │  └─────────────────────────────────────────────────────────┘    │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │  CODE-AWARE MODE (for source files)                             │ │   │
│  │  │  Chunks by: functions, classes, methods                         │ │   │
│  │  │  Preserves: imports context, docstrings                         │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                          │                                                  │
│                          ▼                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       EMBEDDING                                      │   │
│  │  ┌─────────────┐                                                     │   │
│  │  │  Sentence   │  Batch processing for efficiency                    │   │
│  │  │ Transformer │  Model: all-MiniLM-L12-v2 (384 dim)                 │   │
│  │  └─────────────┘                                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                          │                                                  │
│                          ▼                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        STORAGE                                       │   │
│  │  ┌─────────────────────────┐  ┌─────────────────────────┐            │   │
│  │  │  Redis Vector Store     │  │     BM25 Index          │            │   │
│  │  │  - HNSW index           │  │  - Tokenized documents  │            │   │
│  │  │  - Document metadata    │  │  - IDF values           │            │   │
│  │  │  - Parent references    │  │  - Persistent storage   │            │   │
│  │  └─────────────────────────┘  └─────────────────────────┘            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Supported File Types

| Type | Extensions | Processing |
|------|------------|------------|
| **Text** | `.txt`, `.md`, `.rst` | Direct text extraction |
| **Documents** | `.pdf`, `.docx`, `.doc` | Unstructured library |
| **Web** | `.html`, `.htm` | BeautifulSoup parsing |
| **Code** | `.py`, `.js`, `.java`, etc. | Code-aware chunking |
| **Data** | `.json`, `.yaml`, `.csv` | Structured extraction |
| **Images** | `.png`, `.jpg`, `.jpeg` | VLM captioning |

---

## Query Pipeline

### Retrieval Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **hybrid** | Dense + BM25 with RRF fusion | General queries (default) |
| **dense** | Semantic embedding similarity | Conceptual/meaning-based queries |
| **bm25** | Keyword/term matching | Exact term lookups, technical queries |

### Hybrid Retrieval Flow

```
                         Query: "machine learning optimization"
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        ┌───────────────────────┐               ┌───────────────────────┐
        │   DENSE RETRIEVAL     │               │   BM25 RETRIEVAL      │
        │   (Semantic)          │               │   (Lexical)           │
        ├───────────────────────┤               ├───────────────────────┤
        │ 1. Doc A (0.89)       │               │ 1. Doc C (12.4)       │
        │ 2. Doc B (0.85)       │               │ 2. Doc A (11.2)       │
        │ 3. Doc D (0.82)       │               │ 3. Doc E (10.8)       │
        │ 4. Doc C (0.78)       │               │ 4. Doc B (9.5)        │
        │ 5. Doc E (0.71)       │               │ 5. Doc F (8.2)        │
        └───────────────────────┘               └───────────────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                    ┌───────────────────────────────────────────┐
                    │         RRF FUSION (k=60)                 │
                    ├───────────────────────────────────────────┤
                    │  RRF_score(d) = Σ 1/(k + rank(d))         │
                    │                                           │
                    │  Doc A: 1/61 + 1/62 = 0.0326 (rank 1)     │
                    │  Doc C: 1/64 + 1/61 = 0.0320 (rank 2)     │
                    │  Doc B: 1/62 + 1/64 = 0.0318 (rank 3)     │
                    │  Doc E: 1/65 + 1/63 = 0.0312 (rank 4)     │
                    │  Doc D: 1/63 + 0    = 0.0159 (rank 5)     │
                    └───────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────────┐
                    │       CROSS-ENCODER RERANKING             │
                    ├───────────────────────────────────────────┤
                    │  Query-document relevance scoring         │
                    │  Final ranking: [Doc A, Doc B, Doc C...]  │
                    └───────────────────────────────────────────┘
```

---

## GitHub Repository Ingestion

### Automatic Detection

Radiant RAG automatically detects GitHub URLs and uses specialized processing:

```bash
# These all work the same way:
python -m radiant ingest --url "https://github.com/owner/repo"
python -m radiant ingest --url "https://github.com/owner/repo?tab=readme-ov-file"
python -m radiant ingest --url "https://github.com/owner/repo/tree/main"
```

### GitHub Crawler Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GITHUB CRAWLER PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: https://github.com/owner/repo                                       │
│                          │                                                  │
│                          ▼                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 1. URL PARSING                                                        │  │
│  │    Extract: owner="owner", repo="repo", branch="main"                 │  │
│  │    Strip: query strings (?tab=...), fragments (#...)                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 2. FETCH README                                                       │  │
│  │    URL: https://raw.githubusercontent.com/owner/repo/main/README.md   │  │
│  │    Extract: markdown links to other files                             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 3. LIST REPOSITORY FILES (GitHub API)                                 │  │
│  │    GET https://api.github.com/repos/owner/repo/contents/              │  │
│  │    Filter by extensions: .py, .js, .md, .go, .rs, etc.                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 4. FETCH FILE CONTENTS                                                │  │
│  │    Raw URL: https://raw.githubusercontent.com/owner/repo/main/{path}  │  │
│  │    Fetches clean text content (no HTML)                               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 5. CONTENT-AWARE CHUNKING                                             │  │
│  │    ┌─────────────────────────┐  ┌─────────────────────────┐           │  │
│  │    │  Markdown Files         │  │  Code Files             │           │  │
│  │    │  - Q&A pattern          │  │  - Functions/classes    │           │  │
│  │    │  - Header sections      │  │  - Imports context      │           │  │
│  │    │  - Paragraphs           │  │  - Docstrings           │           │  │
│  │    └─────────────────────────┘  └─────────────────────────┘           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│                    STORE IN INDEX                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Supported File Types

| Category | Extensions |
|----------|------------|
| **Documentation** | `.md`, `.txt`, `.rst`, `.mdx` |
| **Python** | `.py`, `.pyw`, `.pyx` |
| **JavaScript/TypeScript** | `.js`, `.jsx`, `.ts`, `.tsx`, `.mjs` |
| **Java/JVM** | `.java`, `.kt`, `.kts`, `.scala` |
| **Systems** | `.go`, `.rs`, `.c`, `.h`, `.cpp`, `.cc`, `.hpp`, `.cs` |
| **Scripting** | `.rb`, `.php`, `.swift`, `.r` |
| **Shell** | `.sh`, `.bash`, `.zsh` |
| **Config/Data** | `.yaml`, `.yml`, `.json`, `.toml`, `.sql` |

### Rate Limiting

| Authentication | Rate Limit |
|----------------|------------|
| No token | 60 requests/hour |
| With `GITHUB_TOKEN` | 5,000 requests/hour |

```bash
# Set GitHub token for higher rate limits
export GITHUB_TOKEN="ghp_your_token_here"
```

---

## Code-Aware Chunking

### How It Works

Instead of blindly splitting code by character count, Radiant RAG parses code structure:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CODE-AWARE CHUNKING                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: calculator.py                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  import math                                                          │  │
│  │  from typing import List                                              │  │
│  │                                                                       │  │
│  │  class Calculator:                                                    │  │
│  │      """A simple calculator."""                                       │  │
│  │                                                                       │  │
│  │      def add(self, a: float, b: float) -> float:                      │  │
│  │          """Add two numbers."""                                       │  │
│  │          return a + b                                                 │  │
│  │                                                                       │  │
│  │      def multiply(self, a: float, b: float) -> float:                 │  │
│  │          """Multiply two numbers."""                                  │  │
│  │          return a * b                                                 │  │
│  │                                                                       │  │
│  │  def helper(x: int) -> int:                                           │  │
│  │      """Helper function."""                                           │  │
│  │      return x * 2                                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ CODE PARSER (Python AST)                                              │  │
│  │ Extracts:                                                             │  │
│  │   - Imports block                                                     │  │
│  │   - Class: Calculator                                                 │  │
│  │   - Method: Calculator.add                                            │  │
│  │   - Method: Calculator.multiply                                       │  │
│  │   - Function: helper                                                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│  OUTPUT CHUNKS:                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Chunk 1: Class Calculator                                             │  │
│  │ ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │ │ File: calculator.py | class: Calculator | Language: python      │   │  │
│  │ │                                                                 │   │  │
│  │ │ Imports:                                                        │   │  │
│  │ │ import math                                                     │   │  │
│  │ │ from typing import List                                         │   │  │
│  │ │                                                                 │   │  │
│  │ │ Code:                                                           │   │  │
│  │ │ class Calculator:                                               │   │  │
│  │ │     """A simple calculator."""                                  │   │  │
│  │ │     ...                                                         │   │  │
│  │ └─────────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Chunk 2: Method Calculator.add                                        │  │
│  │ ┌─────────────────────────────────────────────────────────────────┐   │  │
│  │ │ File: calculator.py | method: Calculator.add | Language: python │   │  │
│  │ │ Imports: import math; from typing import List                   │   │  │
│  │ │ Code: def add(self, a: float, b: float) -> float: ...           │   │  │
│  │ └─────────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Language Support

| Language | Parsing Method | Block Types |
|----------|---------------|-------------|
| **Python** | AST (full) | classes, functions, methods, imports |
| **JavaScript/TypeScript** | Regex | classes, functions, arrow functions, imports |
| **Java** | Regex | classes, methods, imports |
| **Go** | Regex | types, functions, imports |
| **Rust** | Regex | structs, impl blocks, functions, use statements |
| **Others** | Fallback | Whole file as single chunk |

---

## Multilingual Support

### Language Detection

Radiant RAG automatically detects document languages using FastText:

```yaml
language_detection:
  enabled: true
  method: "fast"       # "fast" (FastText), "llm", "auto"
  min_confidence: 0.7
  use_llm_fallback: true
```

### Translation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MULTILINGUAL INGESTION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Document in French                                                  │
│  "L'apprentissage automatique est une branche de l'intelligence..."         │
│                          │                                                  │
│                          ▼                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ LANGUAGE DETECTION AGENT                                              │  │
│  │ Method: FastText (0.1ms)                                              │  │
│  │ Result: { language: "fr", confidence: 0.95 }                          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ TRANSLATION AGENT                                                     │  │
│  │ Method: LLM-based translation                                         │  │
│  │ Source: French → Target: English (canonical)                          │  │
│  │ Chunked at paragraph boundaries for long documents                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                          │                                                  │
│                          ▼                                                  │
│  OUTPUT: Indexed Document                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ content: "Machine learning is a branch of artificial intelligence..." │  │
│  │ metadata:                                                             │  │
│  │   language_code: "fr"                                                 │  │
│  │   language_name: "French"                                             │  │
│  │   was_translated: true                                                │  │
│  │   original_content: "L'apprentissage automatique est..."              │  │
│  │   translation_method: "llm"                                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Supported Languages

- **Detection**: 176 languages (FastText)
- **Translation**: 50+ language pairs (LLM-based)

---

## Advanced Features

### Fact Verification

Automatically verifies generated claims against source context:

```yaml
fact_verification:
  enabled: true
  min_claim_confidence: 0.6
  enable_correction: true
```

### Citation Tracking

Adds source references to answers:

```yaml
citation:
  enabled: true
  citation_style: "inline"  # inline, footnote, academic, enterprise
  include_excerpts: true
  generate_bibliography: true
```

### Strategy Memory

Learns from successful retrieval patterns:

```yaml
strategy_memory:
  enabled: true
  memory_file: "data/strategy_memory.json.gz"
  min_samples: 5
```

### Web Search Augmentation

Real-time web search for current information:

```yaml
web_search:
  enabled: false  # Enable when needed
  provider: "duckduckgo"
  max_results: 5
```

---

## API Reference

### Python API

```python
from radiant.app import RadiantRAG, create_app

# Create application
app = create_app("config.yaml")  # Or RadiantRAG()

# Ingest documents
app.ingest_paths(["./docs/"], hierarchical=True)

# Ingest URLs
app.ingest_urls(["https://github.com/owner/repo"])

# Query
result = app.query("What is RAG?", mode="hybrid")
print(result.answer)
print(result.confidence)

# Search only (no LLM)
results = app.search("BM25 algorithm", mode="hybrid", top_k=10)

# Interactive session
conversation_id = app.start_conversation()
result1 = app.query("What is RAG?", conversation_id=conversation_id)
result2 = app.query("Tell me more", conversation_id=conversation_id)

# Clear index
app.clear_index()

# Check health
health = app.check_health()
```

### Result Object

```python
@dataclass
class PipelineResult:
    answer: str                    # Generated answer
    sources: List[StoredDoc]       # Source documents
    confidence: float              # Critic score (0-1)
    metrics: Dict[str, Any]        # Performance metrics
    plan: Optional[Dict]           # Planning agent output
    sub_queries: List[str]         # Decomposed queries
    citations: List[Citation]      # Source citations
    verification: Optional[Dict]   # Fact verification result
```

---

## Troubleshooting

### Common Issues

**Redis connection failed**
```bash
# Check Redis is running
docker ps | grep redis-stack

# Start Redis if needed
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server
```

**No documents found after ingestion**
```bash
# Check index status
python -m radiant stats

# Clear and re-ingest
python -m radiant clear --confirm
python -m radiant ingest ./docs/
```

**GitHub rate limit exceeded**
```bash
# Set GitHub token
export GITHUB_TOKEN="ghp_your_token"
```

**Slow ingestion**
```yaml
# Increase batch sizes in config.yaml
ingestion:
  embedding_batch_size: 64  # Increase if GPU has memory
  redis_batch_size: 200
```

### Diagnostic Tools

```bash
# Check Redis connectivity and stats
python tools/check_redis.py

# Inspect index contents
python tools/inspect_index.py

# View system health
python -m radiant health

# View statistics
python -m radiant stats
```

---

## File Structure

```
radiant-rag/
├── config.yaml                 # Configuration file
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Package configuration
│
├── radiant/                    # Main package
│   ├── app.py                  # RadiantRAG application
│   ├── orchestrator.py         # Agent pipeline orchestration
│   ├── config.py               # Configuration loading
│   │
│   ├── agents/                 # Pipeline agents
│   │   ├── planning.py         # Query planning
│   │   ├── decomposition.py    # Query decomposition
│   │   ├── dense.py            # Dense retrieval
│   │   ├── bm25.py             # BM25 retrieval
│   │   ├── fusion.py           # RRF fusion
│   │   ├── rerank.py           # Cross-encoder reranking
│   │   ├── synthesis.py        # Answer generation
│   │   ├── citation.py         # Citation tracking
│   │   ├── fact_verification.py # Fact checking
│   │   └── ...                 # Other agents
│   │
│   ├── ingestion/              # Document processing
│   │   ├── processor.py        # Document processor
│   │   ├── github_crawler.py   # GitHub repository crawler
│   │   ├── web_crawler.py      # Web page crawler
│   │   ├── code_chunker.py     # Code-aware chunking
│   │   └── image_captioner.py  # VLM image captioning
│   │
│   ├── storage/                # Storage backends
│   │   ├── redis_store.py      # Redis vector store
│   │   └── bm25_index.py       # Persistent BM25 index
│   │
│   ├── llm/                    # LLM clients
│   │   ├── client.py           # LLM API client
│   │   └── local_models.py     # Local embedding/reranking
│   │
│   └── ui/                     # User interfaces
│       ├── display.py          # Console output
│       ├── tui.py              # Textual TUI
│       └── reports/            # Report generation
│
├── tools/                      # Diagnostic tools
│   ├── check_redis.py
│   └── inspect_index.py
│
├── docs/                       # Documentation
│   └── USER_MANUAL.md
│   
│
└── tests/                      # Test suite
    └── test_all.py
```

---

## License

MIT License

## Contributing

Contributions welcome! Please read the contributing guidelines first.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12 | Initial release |
| 1.1.0 | 2025-12 | Added GitHub crawler, code-aware chunking |
| 1.2.0 | 2025-12 | Added multilingual support, fact verification |
