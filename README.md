# Radiant Agentic RAG

A production-quality Agentic Retrieval-Augmented Generation (RAG) system with multi-agent architecture, hybrid search, and professional reporting.

## Features

### Core Capabilities
- **Multi-agent Pipeline**: Planning, query processing, retrieval, post-retrieval, and generation stages
- **Hybrid Retrieval**: Dense embedding search (HNSW) + BM25 sparse retrieval with RRF fusion
- **Web Search Agent**: Real-time web augmentation during queries for current information
- **Three Retrieval Modes**: Hybrid (default), Dense-only, BM25-only
- **Hierarchical Document Model**: Parent documents with auto-merging of child chunks
- **Cross-encoder Reranking**: Local model inference for accurate relevance scoring
- **Conversation History**: Multi-turn conversation support with Redis persistence
- **VLM Image Captioning**: Automatic image description using Qwen3-VL
- **Professional Reports**: Export results to Markdown, HTML, JSON, or Text formats
- **Critic Evaluation**: Quality assessment of generated answers

### Production Features
- **Batch Ingestion**: Configurable batch processing for embedding generation and Redis operations
- Redis Vector Search with HNSW indexing
- Persistent BM25 index with incremental updates
- Batch embedding and Redis pipeline for fast ingestion
- YAML configuration with environment variable overrides
- Comprehensive logging with third-party noise suppression

### User Interfaces
- **CLI Mode**: Full-featured command-line interface
- **Textual TUI**: Rich terminal UI with real-time pipeline visualization
- **Python API**: Programmatic access for integration

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as package (recommended)
pip install -e .

# Start Redis Stack
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server

# Set environment variables
export RADIANT_OLLAMA_OPENAI_BASE_URL="https://your-ollama-host/v1"
export RADIANT_OLLAMA_OPENAI_API_KEY="your-api-key"

# Ingest documents
python -m radiant ingest ./docs/

# Query
python -m radiant query "What is in my documents?"

# Interactive mode (CLI)
python -m radiant interactive

# Interactive mode (TUI - rich terminal interface)
python -m radiant interactive --tui
```

---

## CLI Reference

### Running the Application

The application can be run in three ways:

```bash
# As a Python module (recommended)
python -m radiant <command> [options]

# Via console script (after pip install -e .)
radiant <command> [options]

# Directly via app.py
python radiant/app.py <command> [options]
```

### Ingest Documents

Ingest files, directories, or URLs into the RAG system.

```bash
python -m radiant ingest <paths> [options]
python -m radiant ingest --url <url> [options]
```

**Arguments:**
- `paths` - One or more files or directories to ingest (optional if using --url)

**Options:**
| Option | Description |
|--------|-------------|
| `--flat` | Use flat storage instead of hierarchical parent/child |
| `--url`, `-u` | URL to ingest (can be specified multiple times) |
| `--crawl-depth N` | Crawl depth for URLs (0=no crawling, default from config) |
| `--max-pages N` | Maximum pages to crawl (default from config) |
| `--no-crawl` | Disable crawling, only fetch specified URLs |
| `--auth USER:PASS` | Basic auth credentials for URL ingestion |

**Examples:**

```bash
# Ingest a directory
python -m radiant ingest ./documents/

# Ingest multiple paths
python -m radiant ingest ./docs/ ./reports/ ./manual.pdf

# Ingest from a URL (with default crawling depth)
python -m radiant ingest --url https://docs.example.com/guide/

# Ingest multiple URLs
python -m radiant ingest --url https://example.com/page1 --url https://example.com/page2

# Mix local files and URLs
python -m radiant ingest ./local_docs/ --url https://example.com/remote_docs/

# Ingest URL without crawling (single page only)
python -m radiant ingest --url https://example.com/page.html --no-crawl

# Ingest URL with custom crawl depth
python -m radiant ingest --url https://docs.example.com/ --crawl-depth 3

# Ingest URL with authentication
python -m radiant ingest --url https://internal.example.com/docs/ --auth admin:secret123

# Flat storage (no parent/child hierarchy)
python -m radiant ingest ./docs/ --flat
```

#### URL Ingestion Configuration

URL crawling behavior is controlled in `config.yaml`:

```yaml
web_crawler:
  # Maximum crawl depth (0 = seed URLs only)
  max_depth: 2
  
  # Maximum pages to crawl per session
  max_pages: 100
  
  # Only crawl same domain as seed URLs
  same_domain_only: true
  
  # Delay between requests (rate limiting)
  delay: 0.5
  
  # URL patterns to exclude
  exclude_patterns:
    - ".*\\.(jpg|jpeg|png|gif|css|js)$"
    - ".*/login.*"
```

#### Batch Indexing Configuration

For large document collections, batch processing significantly improves ingestion performance. Batching is enabled by default and can be tuned in `config.yaml`:

```yaml
ingestion:
  # Enable batch processing (recommended for large corpora)
  batch_enabled: true
  
  # Batch size for embedding generation
  # Larger = faster but more memory. Recommended: 16-64 (GPU), 8-32 (CPU)
  embedding_batch_size: 32
  
  # Batch size for Redis pipeline operations
  # Larger = fewer network round trips. Recommended: 50-200
  redis_batch_size: 100
  
  # Default chunk sizes for hierarchical storage
  child_chunk_size: 512
  child_chunk_overlap: 50
  
  # Show progress during ingestion
  show_progress: true
```

**Performance Tips:**

| Corpus Size | embedding_batch_size | redis_batch_size | Expected Speedup |
|-------------|---------------------|------------------|------------------|
| Small (<100 docs) | 16 | 50 | 2-3x |
| Medium (100-1000 docs) | 32 | 100 | 5-10x |
| Large (1000+ docs) | 64 | 200 | 10-20x |

**Memory Considerations:**
- Larger `embedding_batch_size` requires more GPU/CPU memory
- Set `batch_enabled: false` for debugging or memory-constrained environments

---

### Query

Query the RAG system with full pipeline processing.

```bash
python -m radiant query "<question>" [options]
```

**Arguments:**
- `query` - Your question or query string

**Options:**
| Option | Description |
|--------|-------------|
| `--mode`, `-m` | Retrieval mode: `hybrid` (default), `dense`, `bm25` |
| `--conversation`, `-conv` | Conversation ID for multi-turn history |
| `--save`, `-s` | Save report to file (.md, .html, .json) |
| `--compact` | Use compact display format |
| `--simple` | Use simplified pipeline (faster, less features) |

**Examples:**

```bash
# Basic query
python -m radiant query "What is Article I of the Constitution?"

# Use semantic search only
python -m radiant query "meaning of liberty" --mode dense

# Use keyword search only
python -m radiant query "Article I Section 8" --mode bm25

# Save report as Markdown
python -m radiant query "Summarize the main points" --save report.md

# Save as HTML (styled, printable)
python -m radiant query "What are the key findings?" --save report.html

# Save as JSON (for programmatic use)
python -m radiant query "List all topics" --save results.json

# Compact output
python -m radiant query "Quick answer please" --compact

# Continue a conversation
python -m radiant query "Tell me more" --conversation abc123

# Simple/fast mode (skips planning, decomposition, critic)
python -m radiant query "Simple question" --simple
```

---

### Search (Retrieval Only)

Search documents without LLM generation - pure retrieval for quick lookups.

```bash
python -m radiant search "<query>" [options]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--mode`, `-m` | Retrieval mode: `hybrid` (default), `dense`, `bm25` |
| `--top-k`, `-k` | Number of results (default: 10) |
| `--save`, `-s` | Save results to file (.md, .json) |

**Examples:**

```bash
# Basic search
python -m radiant search "constitution amendments"

# BM25 keyword search
python -m radiant search "habeas corpus" --mode bm25

# Get more results
python -m radiant search "executive powers" --top-k 20

# Semantic search with saved results
python -m radiant search "freedom of speech" --mode dense --save search_results.md
```

---

### Interactive Mode

Start an interactive session with conversation history and live commands.

```bash
# Classic command-line mode
python -m radiant interactive

# Textual-based TUI (modern terminal interface)
python -m radiant interactive --tui
```

**Options:**
| Option | Description |
|--------|-------------|
| `--tui` | Use the Textual-based terminal UI |
| `--classic` | Use classic command-line mode (default) |

#### Classic Mode Commands

| Command | Description |
|---------|-------------|
| `quit` or `exit` | Exit interactive mode |
| `new` | Start a new conversation |
| `stats` | Show system statistics |
| `mode <MODE>` | Set retrieval mode (hybrid/dense/bm25) |
| `save <PATH>` | Save last result to markdown/json |
| `report <PATH>` | Save detailed text report |
| `search <QUERY>` | Search without LLM generation |

**Example Classic Session:**

```
━━━ Agentic RAG Interactive Mode ━━━

Commands:
  quit         Exit interactive mode
  new          Start new conversation
  stats        Show system statistics
  mode MODE    Set retrieval mode (hybrid/dense/bm25)
  save PATH    Save last result to file (markdown/json)
  report PATH  Save detailed text report
  search Q     Search without LLM generation

Conversation: 23cc5826...

[H]> What is Article I?
[... answer displayed ...]

[H]> What powers does it grant?
[... follow-up answer with context ...]

[H]> mode bm25
Retrieval mode: bm25

[B]> search executive branch
[... search results displayed ...]

[B]> save constitution_qa.html
✓ Saved to: /path/to/constitution_qa.html

[B]> report ./reports/detailed_report.txt
✓ Text report saved to: /path/to/reports/detailed_report.txt

[B]> new
New conversation: 8f2a1b3c...

[B]> quit
Goodbye!
```

#### Textual TUI Mode

The Textual TUI provides a rich terminal interface with:

- **Query Input**: Enter questions and see real-time results
- **Timeline Panel**: Visual execution trace of all pipeline steps
- **Answer Panel**: Final answer with inline citations
- **Tabbed Views**: Overview, Plan, Queries, Retrieval, Agents, Metrics, Logs

**TUI Keybindings:**
| Key | Action |
|-----|--------|
| `Ctrl+Q` | Quit the application |
| `Ctrl+N` | Start new conversation |
| `Ctrl+R` | Focus query input |
| `Ctrl+S` | Save report to file |
| `Escape` | Cancel current operation |

**Example TUI Layout:**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Agentic RAG Console                                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│  [ Ask a question about your knowledge base...                      ] [Run]  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────┐   ┌─────────────────────────────────────────┐   │
│  │ Run Timeline / Trace   │   │ Answer & Citations                      │   │
│  │─────────────────────────│   │─────────────────────────────────────────│   │
│  │ ● Planning         12ms│   │                                         │   │
│  │ ● Query Decomp     45ms│   │ The answer to your question is...      │   │
│  │ ● Dense Retrieval 210ms│   │                                         │   │
│  │ ● BM25 Retrieval   85ms│   │ Citations:                              │   │
│  │ ● Reranking       120ms│   │ [doc1] [doc2] [doc3]                    │   │
│  │ ● Generation      450ms│   │                                         │   │
│  └─────────────────────────┘   └─────────────────────────────────────────┘   │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  [ Overview ] [ Plan ] [ Queries ] [ Retrieval ] [ Agents ] [ Metrics ]      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### Statistics

Display index and system statistics.

```bash
python -m radiant stats
```

**Output includes:**
- Redis document count by level (parent/child)
- Vector index status and dimensions
- BM25 index statistics (documents, vocabulary size)
- Source file summary

---

### Health Check

Check system connectivity and component status.

```bash
python -m radiant health
```

**Checks:**
- Redis connection
- Vector index availability
- BM25 index status

---

### Rebuild BM25 Index

Rebuild the BM25 index from Redis store.

```bash
python -m radiant rebuild-bm25 [--limit N]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--limit` | Maximum documents to index (0 = all) |

**Examples:**

```bash
# Rebuild entire index
python -m radiant rebuild-bm25

# Rebuild with limit (for testing)
python -m radiant rebuild-bm25 --limit 1000
```

---

## Python API

### Basic Usage

```python
from radiant.app import RadiantRAG

# Initialize
app = RadiantRAG(config_path="config.yaml")

# Ingest documents
stats = app.ingest_documents(
    paths=["./docs/"],
    use_hierarchical=True,
    skip_vlm=False,
)
print(f"Ingested {stats['documents_stored']} documents")

# Query
result = app.query("What is machine learning?")
print(result.answer)

# Query with options
result = app.query(
    "Explain neural networks",
    retrieval_mode="dense",
    save_path="report.html",
)
```

### Search Without LLM

```python
# Pure retrieval - no LLM generation
results = app.search(
    query="neural networks",
    mode="hybrid",
    top_k=10,
)

for doc, score in results:
    print(f"{score:.3f}: {doc.content[:100]}...")
```

### Conversation History

```python
# Start conversation
conv_id = app.start_conversation()

# Multi-turn queries
result1 = app.query("What is Python?", conversation_id=conv_id)
result2 = app.query("How does it compare to Java?", conversation_id=conv_id)
result3 = app.query("Which should I learn first?", conversation_id=conv_id)
```

### Report Generation

```python
from radiant.ui.reports.report import QueryReport, save_report
from radiant.ui.reports.text import generate_text_report, save_text_report

# Generate report from result
report = QueryReport.from_pipeline_result(result, retrieval_mode="hybrid")

# Save in different formats
save_report(report, "report.md")      # Markdown
save_report(report, "report.html")    # Styled HTML
save_report(report, "report.json")    # JSON data

# Generate detailed text report (similar to enterprise run reports)
text_report = generate_text_report(result, retrieval_mode="hybrid")
save_text_report(result, "detailed_report.txt")
```

#### Text Report Format

The text report provides a comprehensive, enterprise-style run report with:

```
================================================================================
AGENTIC RAG RUN REPORT
================================================================================

Run ID        : rag-run-2025-01-15-10-30-45
Timestamp     : 2025-01-15 10:30:45
Environment   : production
User          : anonymous (role=user)
Workspace     : default

--------------------------------------------------------------------------------
1. USER QUERY
--------------------------------------------------------------------------------

Original Query
--------------
"How does the system handle document retrieval?"

...

--------------------------------------------------------------------------------
3. HIGH-LEVEL METRICS
--------------------------------------------------------------------------------

Status              : SUCCESS
Total Latency       : 1.25 s
Steps Executed      : 6
Documents Retrieved : 45 (pre-rerank)
Documents in Context: 5
Answer Confidence   : 0.87 (0–1)
Guardrails          : PASSED

...
================================================================================
END OF REPORT
================================================================================
```

---

## Configuration

### Environment Variables

```bash
# Required: LLM endpoint
export RADIANT_OLLAMA_OPENAI_BASE_URL="https://your-ollama-host/v1"
export RADIANT_OLLAMA_OPENAI_API_KEY="your-api-key"

# Optional: Override any config value
export RADIANT_REDIS_URL="redis://localhost:6379"
export RADIANT_LOCAL_MODELS_DEVICE="cuda"
```

### Configuration File (config.yaml)

```yaml
# LLM Settings
ollama:
  model: "gemma3:12b"
  temperature: 0.2
  max_tokens: 2048

# Local Models (embedding, reranking)
local_models:
  device: "auto"  # auto, cpu, cuda
  embedding_model: "sentence-transformers/all-MiniLM-L12-v2"
  cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L12-v2"

# Redis Settings
redis:
  url: "redis://localhost:6379"
  vector_index:
    name: "radiant_vectors"
    distance_metric: "COSINE"
    hnsw_m: 16
    hnsw_ef_construction: 200

# Retrieval Settings
retrieval:
  dense_top_k: 20
  bm25_top_k: 20
  rrf_k: 60

# VLM Image Captioning
vlm:
  enabled: true
  model: "Qwen/Qwen3-VL-8B-Instruct"
  max_new_tokens: 512

# Logging
logging:
  quiet_third_party: true  # Suppress noisy library logs
```

---

## Retrieval Modes Explained

| Mode | Description | Best For |
|------|-------------|----------|
| `hybrid` | Combines dense + BM25 with RRF fusion | General use, best overall accuracy |
| `dense` | Semantic embedding search only | Conceptual queries, paraphrasing |
| `bm25` | Keyword/term matching only | Exact terms, names, codes |

**Examples:**

```bash
# Hybrid (default) - combines both methods
python -m radiant query "What are the main themes discussed?"

# Dense - semantic similarity
python -m radiant query "freedom and liberty concepts" --mode dense

# BM25 - exact keyword matching
python -m radiant query "Section 8 Clause 3" --mode bm25
```

---

## Report Formats

### Markdown (.md)
Clean, readable format for documentation.

```bash
python -m radiant query "Summarize findings" --save report.md
```

### HTML (.html)
Styled, printable format with CSS.

```bash
python -m radiant query "Executive summary" --save report.html
```

### JSON (.json)
Structured data for programmatic use.

```bash
python -m radiant query "List all items" --save data.json
```

---

## Performance Tips

### Fast Ingestion

```bash
# Skip VLM captioning (images get placeholder text)
python -m radiant ingest ./docs/ --skip-vlm

# Skip images entirely
python -m radiant ingest ./docs/ --skip-images
```

### Ingestion Time Comparison

| Method | ~25 pages + 2 images |
|--------|---------------------|
| Full VLM captioning | ~10-20 minutes |
| `--skip-vlm` | ~30-60 seconds |
| `--skip-images` | ~20-40 seconds |

### Quick Queries

```bash
# Simple mode skips planning, decomposition, critic
python -m radiant query "Quick question" --simple

# Search only (no LLM generation)
python -m radiant search "keyword lookup"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│  Query Processing                                               │
│  ┌──────────┐ ┌─────────────┐ ┌──────────┐ ┌───────────────┐    │
│  │ Planning │→│ Decompose   │→│ Rewrite  │→│ Expansion     │    │
│  └──────────┘ └─────────────┘ └──────────┘ └───────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  Retrieval (configurable mode)                                  │
│  ┌──────────────────┐   ┌──────────────────┐   ┌────────────┐   │
│  │ Dense (HNSW)     │   │ BM25 (Sparse)    │   │ Web Search │   │
│  └────────┬─────────┘   └────────┬─────────┘   └─────┬──────┘   │
│           └──────────────────────┼───────────────────┘          │
│                            ┌─────┴─────┐                        │
│                            │ RRF Fusion│                        │
│                            └───────────┘                        │
├─────────────────────────────────────────────────────────────────┤
│  Post-Retrieval                                                 │
│  ┌───────────────┐     ┌────────────────────┐                   │
│  │ Auto-Merge    │────→│ Cross-Encoder      │                   │
│  │ (Hierarchical)│     │ Reranking          │                   │
│  └───────────────┘     └────────────────────┘                   │
├─────────────────────────────────────────────────────────────────┤
│  Generation                                                     │
│  ┌────────────────────┐     ┌─────────────────┐                 │
│  │ Answer Synthesis   │────→│ Critic Agent    │                 │
│  │ + History          │     │ (Quality Check) │                 │
│  └────────────────────┘     └─────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

### Web Search Agent (Optional)

When enabled, the **WebSearchAgent** provides real-time web augmentation:

- **Trigger Keywords**: Queries with "latest", "recent", "news", etc. trigger search
- **LLM URL Suggestion**: Uses LLM to suggest relevant URLs to fetch
- **Content Extraction**: Parses HTML, removes scripts/styles, extracts text
- **RRF Fusion**: Web results merge with indexed content via RRF

Enable in `config.yaml`:
```yaml
web_search:
  enabled: true    # Default: false
  max_pages: 3
  trigger_keywords:
    - "latest"
    - "recent"
    - "news"
```

---

## Agentic Enhancements

The system includes several agentic capabilities that enable self-improvement and adaptive behavior:

### Critic-Driven Retry Loop

When the CriticAgent detects quality issues, the system automatically retries with modified strategies:

```
Query → Plan → Retrieve → Generate → Critique ──┐
          ↑                                      │
          └────── Retry with modifications ←─────┘
                  (if confidence < threshold)
```

Configure in `config.yaml`:
```yaml
agentic:
  max_critic_retries: 2       # Maximum retry attempts
  rewrite_on_retry: true      # Rewrite query on retry
  expand_retrieval_on_retry: true  # Fetch more documents
```

### Confidence Thresholds

The system provides confidence scores and gracefully returns "I don't know" when confidence is low:

```yaml
agentic:
  confidence_threshold: 0.4   # Below this returns "I don't know"

critic:
  confidence_threshold: 0.4   # Minimum acceptable confidence
  min_retrieval_confidence: 0.3
```

### Dynamic Retrieval Mode Selection

The PlanningAgent automatically selects the optimal retrieval strategy:

- **hybrid**: Best for most queries (default)
- **dense**: Best for semantic/conceptual queries
- **bm25**: Best for specific terms, names, exact phrases

```yaml
agentic:
  dynamic_retrieval_mode: true  # Let planner choose mode
```

### Tool Integration

Built-in tools for enhanced capabilities:

- **Calculator**: Safe mathematical expression evaluation
- **Code Executor**: Sandboxed Python for data manipulation

```yaml
agentic:
  tools_enabled: true
```

### Strategy Memory

Tracks which retrieval strategies work best for different query patterns:

```yaml
agentic:
  strategy_memory_enabled: true
  strategy_memory_path: "./data/strategy_memory.json.gz"
```

The system learns over time which strategies work for:
- Factual queries ("what is...")
- Comparison queries ("compare X and Y")
- Procedural queries ("how to...")
- And more query patterns

---

## File Structure

```
radiant-rag/
├── pyproject.toml              # Python packaging configuration
├── config.yaml                 # Default configuration
├── README.md
├── requirements.txt
├── requirements-dev.txt
│
├── radiant/                    # Main package
│   ├── __init__.py             # Package init
│   ├── __main__.py             # Entry: python -m radiant
│   ├── app.py                  # RadiantRAG class + main()
│   ├── cli.py                  # CLI wrapper
│   ├── config.py               # Configuration loading
│   ├── orchestrator.py         # Agentic pipeline coordination
│   │
│   ├── agents/                 # Pipeline agents (one per file)
│   │   ├── __init__.py         # Package exports
│   │   ├── base.py             # AgentContext, new_agent_context
│   │   ├── planning.py         # PlanningAgent (dynamic mode selection)
│   │   ├── decomposition.py    # QueryDecompositionAgent
│   │   ├── rewrite.py          # QueryRewriteAgent
│   │   ├── expansion.py        # QueryExpansionAgent
│   │   ├── dense.py            # DenseRetrievalAgent
│   │   ├── bm25.py             # BM25RetrievalAgent
│   │   ├── web_search.py       # WebSearchAgent
│   │   ├── fusion.py           # RRFAgent (Reciprocal Rank Fusion)
│   │   ├── automerge.py        # HierarchicalAutoMergingAgent
│   │   ├── rerank.py           # CrossEncoderRerankingAgent
│   │   ├── synthesis.py        # AnswerSynthesisAgent
│   │   ├── critic.py           # CriticAgent (confidence scoring)
│   │   ├── tools.py            # Tool abstraction (calculator, code)
│   │   ├── strategy_memory.py  # Retrieval strategy learning
│   │   └── registry.py         # Agent registry
│   │
│   ├── storage/                # Storage backends
│   │   ├── __init__.py
│   │   ├── redis_store.py      # Redis + Vector Search
│   │   └── bm25_index.py       # Persistent BM25
│   │
│   ├── ingestion/              # Document processing
│   │   ├── __init__.py
│   │   ├── processor.py        # DocumentProcessor
│   │   ├── web_crawler.py      # URL crawling
│   │   └── image_captioner.py  # VLM captioning
│   │
│   ├── llm/                    # LLM clients
│   │   ├── __init__.py
│   │   ├── client.py           # LLMClient, LLMClients
│   │   └── local_models.py     # LocalNLPModels (embeddings, reranking)
│   │
│   ├── ui/                     # User interfaces
│   │   ├── __init__.py
│   │   ├── display.py          # Console utilities
│   │   ├── tui.py              # Textual TUI
│   │   └── reports/            # Report generation
│   │       ├── __init__.py
│   │       ├── report.py       # HTML, Markdown, JSON
│   │       └── text.py         # Text reports
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── metrics.py          # Performance tracking
│       └── conversation.py     # Conversation history
│
├── data/                       # Runtime data (created automatically)
│   ├── bm25_index.json.gz      # BM25 index
│   └── strategy_memory.json.gz # Strategy learning data
│
├── tools/                      # Diagnostic tools
│   ├── check_redis.py          # Redis diagnostics
│   └── inspect_index.py        # Index inspection
│
└── tests/                      # Test suite
    ├── __init__.py
    ├── test_all.py
    └── test_*/                 # Test subdirectories
```

---

## Troubleshooting

### Check Redis Status
```bash
python tools/check_redis.py
```

### Inspect Index
```bash
python tools/inspect_index.py
```

### Common Issues

**"Redis Search module not found"**
```bash
# Install Redis Stack (includes RediSearch)
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server
```

**"No documents found"**
```bash
# Check ingestion status
python -m radiant stats

# Re-ingest if needed
python -m radiant ingest ./docs/
```

**Slow image ingestion**
```bash
# Skip VLM captioning
python -m radiant ingest ./docs/ --skip-vlm
```

---

## Requirements

- Python 3.10+
- Redis Stack (Redis + RediSearch module)
- CUDA-capable GPU (optional, for faster inference)

### Key Dependencies
- `sentence-transformers` - Embedding and reranking
- `transformers>=4.57.0` - Qwen3-VL support
- `redis` - Redis client
- `rich` - Console output
- `unstructured` - Document parsing
- `qwen-vl-utils` - VLM image processing

---

## License

MIT License

## Contributing

Contributions welcome! Please read the contributing guidelines first.
