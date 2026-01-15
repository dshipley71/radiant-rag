# Radiant Agentic RAG
## User Manual

**Version 1.4**  
**January 2025**

---

## Document Information

| Field | Value |
|-------|-------|
| Document Title | Radiant Agentic RAG User Manual |
| Version | 1.4 |
| Release Date | January 2025 |
| Target Audience | Software Developers, Data Scientists, AI Engineers |

---

# Table of Contents

- [Part I: Introduction](#part-i-introduction)
- [Part II: Installation & Setup](#part-ii-installation--setup)
- [Part III: Core Concepts](#part-iii-core-concepts)
- [Part IV: Document Ingestion](#part-iv-document-ingestion)
- [Part V: Storage Backends](#part-v-storage-backends)
- [Part VI: Binary Quantization](#part-vi-binary-quantization)
- [Part VII: Agents Reference](#part-vii-agents-reference)
- [Part VIII: User Interfaces](#part-viii-user-interfaces)
- [Part IX: Metrics & Monitoring](#part-ix-metrics--monitoring)
- [Part X: Advanced Topics](#part-x-advanced-topics)
- [Part XI: API Reference](#part-xi-api-reference)
- [Part XII: Troubleshooting](#part-xii-troubleshooting)
- [Appendices](#appendices)

---

# Part I: Introduction

## Chapter 1: Overview

### 1.1 What is Radiant RAG?

Radiant RAG is an enterprise-grade, agentic Retrieval-Augmented Generation (RAG) system designed for building intelligent document question-answering applications. Unlike traditional RAG implementations that follow a fixed pipeline, Radiant RAG employs a modular, agent-based architecture where specialized agents collaborate to process queries, retrieve relevant information, and generate accurate, well-cited responses.

**Core Design Principles:**

1. **Agentic Architecture**: Each component is an autonomous agent with standardized interfaces (BaseAgent ABC)
2. **Dynamic Strategy Selection**: The system adapts its retrieval strategy based on query characteristics
3. **Self-Correction**: Built-in critic and verification agents detect and correct errors
4. **Enterprise Compliance**: Citation tracking and audit trails support regulatory requirements
5. **Multilingual Support**: Automatic language detection and translation for multilingual corpora
6. **Performance Optimization**: Binary quantization for 10-20x faster retrieval
7. **Flexible Storage**: Support for Redis, ChromaDB, and PostgreSQL pgvector

### 1.2 Key Features

| Category | Agents/Features | Purpose |
|----------|--------|---------|
| Query Processing | Planning, Decomposition, Rewrite, Expansion | Understand and optimize queries |
| Retrieval | Dense, BM25, Web Search | Fetch relevant documents |
| Storage | Redis, Chroma, PgVector | Flexible vector storage |
| Quantization | Binary, Int8 | 10-20x faster retrieval |
| Fusion | RRF | Combine results from multiple retrievers |
| Post-Retrieval | AutoMerge, Rerank, Context Eval, Summarization | Refine and validate context |
| Generation | Synthesis, Critic | Generate and evaluate answers |
| Verification | Fact Verification, Citation | Ensure accuracy and attribution |
| Multilingual | Language Detection, Translation | Support multiple languages |
| Tools | Calculator, Code Execution | Extended capabilities |
| Monitoring | Prometheus, OpenTelemetry | Metrics and tracing |

### 1.3 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.11 |
| RAM | 8 GB | 32+ GB |
| Storage Backend | Redis Stack 7.2+ | Latest |
| GPU | None | NVIDIA 8+ GB VRAM |

### 1.4 Architecture at a Glance

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                     PLANNING PHASE                           │
│  PlanningAgent → Decomposition → Rewrite → Expansion        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PHASE                           │
│  Dense Retrieval ─┬─► RRF Fusion                            │
│  BM25 Retrieval  ─┤                                         │
│  Web Search      ─┘                                         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                  POST-RETRIEVAL PHASE                        │
│  MultiHop → AutoMerge → Rerank → Context Eval → Summarize  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                   GENERATION PHASE                           │
│  Answer Synthesis ◄──► Critic Agent (Retry Loop)            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                  VERIFICATION PHASE                          │
│  Fact Verification → Citation Tracking                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Cited Response with Audit Trail
```

---

# Part II: Installation & Setup

## Chapter 2: Installation

### 2.1 Prerequisites

```bash
# Check Python version
python --version  # Should be 3.10+

# Create virtual environment
python -m venv venv
source venv/bin/activate
```

### 2.2 Storage Backend Setup

#### Redis Stack (Default)

```bash
# Docker (Recommended)
docker run -d \
  --name redis-stack \
  -p 6379:6379 \
  -p 8001:8001 \
  redis/redis-stack:latest

# Verify
docker exec redis-stack redis-cli ping
```

#### ChromaDB (Alternative)

```bash
pip install chromadb
```

#### PostgreSQL with pgvector (Alternative)

```bash
# Install PostgreSQL with pgvector extension
# Then install Python driver
pip install psycopg2-binary
```

### 2.3 Installing from Source

```bash
git clone https://github.com/your-org/radiant-rag.git
cd radiant-rag
pip install -e .
```

### 2.4 Verifying Installation

```bash
python -c "from radiant.app import RadiantRAG; print('OK')"
python -m radiant health
```

## Chapter 3: Configuration

### 3.1 Configuration File

```yaml
# config.yaml
ollama:
  openai_base_url: "http://localhost:11434/v1"
  openai_api_key: "ollama"
  chat_model: "qwen2.5:latest"

local_models:
  embed_model_name: "sentence-transformers/all-MiniLM-L12-v2"
  cross_encoder_name: "cross-encoder/ms-marco-MiniLM-L12-v2"
  device: "auto"
  embedding_dimension: 384

storage:
  backend: redis  # Options: redis, chroma, pgvector

redis:
  url: "redis://localhost:6379/0"
  key_prefix: "radiant"
```

### 3.2 Environment Variables

```bash
export RADIANT_OLLAMA_CHAT_MODEL="llama3:70b"
export RADIANT_REDIS_URL="redis://redis-server:6379/0"
export RADIANT_RETRIEVAL_DENSE_TOP_K="20"
```

---

# Part III: Core Concepts

## Chapter 4: Architecture Overview

### 4.1 Agentic RAG Paradigm

Traditional RAG: `Query → Embed → Retrieve → Generate → Response`

Agentic RAG:
```
Query → [Planning Agent decides strategy]
      → [Query Agents optimize query]
      → [Retrieval Agents fetch documents]
      → [Post-processing Agents refine context]
      → [Generation Agent creates answer]
      → [Verification Agents validate answer]
      → Verified, Cited Response
```

### 4.2 Pipeline Phases

1. **Planning**: Analyze query, select strategy
2. **Query Processing**: Decompose, rewrite, expand
3. **Retrieval**: Dense, sparse, web search
4. **Fusion**: Combine with RRF
5. **Post-Retrieval**: Rerank, merge, evaluate
6. **Generation**: Synthesize answer
7. **Verification**: Fact-check, cite sources

### 4.3 Agent Communication

All agents now use the standardized `AgentResult` wrapper:

```python
from radiant.agents import AgentResult, AgentStatus, AgentMetrics

@dataclass
class AgentResult(Generic[T]):
    data: T                           # The actual result data
    success: bool = True              # Execution status
    status: AgentStatus = AgentStatus.SUCCESS
    error: Optional[str] = None       # Error message if failed
    warnings: List[str] = []          # Warning messages
    metrics: Optional[AgentMetrics] = None  # Execution metrics

# Usage
result = agent.run(query="test")
if result.success:
    data = result.data
    print(f"Duration: {result.metrics.duration_ms}ms")
```

## Chapter 5: Data Model

### 5.1 Documents and Chunks

```python
@dataclass(frozen=True)
class IngestedChunk:
    content: str
    meta: Dict[str, Any]

@dataclass(frozen=True)
class StoredDoc:
    doc_id: str
    content: str
    meta: Dict[str, Any]
    score: Optional[float] = None
```

### 5.2 Metadata Schema

| Field | Type | Indexed | Description |
|-------|------|---------|-------------|
| doc_level | TAG | Yes | "parent" or "child" |
| parent_id | TAG | Yes | Parent document ID |
| language_code | TAG | Yes | ISO 639-1 code |
| source_path | str | No | Original file path |
| was_translated | bool | No | Translation flag |
| original_content | str | No | Original text |

---

# Part IV: Document Ingestion

## Chapter 6: Ingestion Pipeline

### 6.1 Supported Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| PDF | .pdf | Text and OCR |
| Word | .docx, .doc | Full support |
| Text | .txt | UTF-8 |
| Markdown | .md | Preserves structure |
| HTML | .html | Strips tags |
| Images | .png, .jpg | VLM captioning |
| Code | .py, .js, .ts, etc. | Code-aware chunking |

### 6.2 DocumentProcessor

```python
from radiant.ingestion import DocumentProcessor

processor = DocumentProcessor(
    cleaning_config=config.unstructured_cleaning,
    image_captioner=captioner,
)

chunks = processor.process_file("/path/to/document.pdf")
```

### 6.3 TranslatingDocumentProcessor

```python
from radiant.ingestion import TranslatingDocumentProcessor
from radiant.agents import LanguageDetectionAgent, TranslationAgent

processor = TranslatingDocumentProcessor(
    base_processor=base_processor,
    language_detection_agent=lang_detector,
    translation_agent=translator,
    canonical_language="en",
    translate_at_ingestion=True,
    preserve_original=True,
)

chunks = processor.process_file("/path/to/french_document.pdf")
```

### 6.4 Code-Aware Chunking

```python
from radiant.ingestion.code_chunker import CodeChunker

chunker = CodeChunker(
    max_chunk_size=2000,
    min_chunk_size=100,
    include_imports_context=True,
)

code_chunks = chunker.chunk_file(content, "main.py")
for chunk in code_chunks:
    print(f"Block: {chunk.block_type} - {chunk.block_name}")
    print(f"Lines: {chunk.start_line}-{chunk.end_line}")
```

---

# Part V: Storage Backends

## Chapter 7: Storage Backend Selection

### 7.1 Backend Comparison

| Backend | Use Case | Pros | Cons |
|---------|----------|------|------|
| **Redis** | Production, low-latency | Fast, feature-rich | Requires Redis Stack |
| **Chroma** | Development, testing | Easy setup, embedded | Less scalable |
| **PgVector** | Enterprise, PostgreSQL | Mature, ACID | More setup |

### 7.2 Redis Configuration

```yaml
storage:
  backend: redis

redis:
  url: "redis://localhost:6379/0"
  key_prefix: "radiant"
  doc_ns: "doc"
  embed_ns: "emb"
  vector_index:
    name: "radiant_vectors"
    hnsw_m: 16
    hnsw_ef_construction: 200
    hnsw_ef_runtime: 100
    distance_metric: "COSINE"
```

### 7.3 Chroma Configuration

```yaml
storage:
  backend: chroma

chroma:
  persist_directory: "./data/chroma_db"
  collection_name: "radiant_docs"
  distance_fn: "cosine"
  embedding_dimension: 384
```

### 7.4 PgVector Configuration

```yaml
storage:
  backend: pgvector

pgvector:
  connection_string: "postgresql://user:pass@localhost:5432/radiant"
  leaf_table_name: "haystack_leaves"
  parent_table_name: "haystack_parents"
  vector_function: "cosine_similarity"
  search_strategy: "hnsw"
```

### 7.5 Storage Factory

```python
from radiant.storage import create_vector_store, get_available_backends

# Check available backends
backends = get_available_backends()
print(backends)
# {'redis': {'available': True}, 'chroma': {'available': True}, 'pgvector': {'available': False}}

# Create store based on config
store = create_vector_store(config)
```

---

# Part VI: Binary Quantization

## Chapter 8: Quantization Overview

### 8.1 Performance Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Memory | 1,536 MB | 432 MB | **3.5x less** |
| Retrieval Speed | 50-100ms | 5-10ms | **10-20x faster** |
| Accuracy | 100% | 95-96% | **-4%** |

### 8.2 How It Works

```
Query (float32)
    ↓
[Stage 1] Binary Search
  • Quantize query to binary (1 bit per dimension)
  • Search with Hamming distance
  • Retrieve 4× candidate documents
  • Ultra-fast: 2 CPU cycles per comparison
    ↓
[Stage 2] Precision Rescoring
  • Load int8/float32 embeddings for candidates
  • Recalculate similarity scores
  • Return top-k results
  • High accuracy: 95-96% retention
    ↓
Final Results
```

### 8.3 Configuration

```yaml
redis:  # or chroma, or pgvector
  quantization:
    enabled: true
    precision: "both"  # "binary", "int8", or "both"
    rescore_multiplier: 4.0
    use_rescoring: true
    int8_ranges_file: "data/int8_ranges.npy"
```

### 8.4 Calibration

```bash
# Generate int8 calibration ranges
python tools/calibrate_int8_ranges.py \
    --sample-size 100000 \
    --output data/int8_ranges.npy \
    --config config.yaml
```

### 8.5 API Usage

```python
# Automatic (when quantization enabled)
results = store.retrieve_by_embedding(
    query_embedding=embedding,
    top_k=10
)

# Explicit quantized retrieval
results = store.retrieve_by_embedding_quantized(
    query_embedding=embedding,
    top_k=10,
    rescore_multiplier=4.0,
    use_rescoring=True
)
```

---

# Part VII: Agents Reference

## Chapter 9: Agent Hierarchy

### 9.1 BaseAgent Architecture

```
BaseAgent (Abstract)
├── LLMAgent (requires LLM client)
│   ├── PlanningAgent
│   ├── AnswerSynthesisAgent
│   ├── CriticAgent
│   ├── QueryDecompositionAgent
│   ├── QueryRewriteAgent
│   ├── QueryExpansionAgent
│   └── WebSearchAgent
│
├── RetrievalAgent (requires vector store)
│   └── DenseRetrievalAgent
│
└── BaseAgent (direct inheritance)
    ├── BM25RetrievalAgent
    ├── RRFAgent
    ├── HierarchicalAutoMergingAgent
    ├── CrossEncoderRerankingAgent
    └── MultiHopReasoningAgent
```

### 9.2 Agent Categories

```python
class AgentCategory(Enum):
    PLANNING = "planning"           # Query analysis
    QUERY_PROCESSING = "query_processing"  # Decomposition, rewrite
    RETRIEVAL = "retrieval"         # Dense, sparse, web
    POST_RETRIEVAL = "post_retrieval"      # Fusion, reranking
    GENERATION = "generation"       # Answer synthesis
    EVALUATION = "evaluation"       # Critic, verification
    TOOL = "tool"                   # Calculator, code
    UTILITY = "utility"             # General purpose
```

## Chapter 10: Creating Custom Agents

### 10.1 Agent Template

```python
from radiant.agents.base_agent import (
    BaseAgent, LLMAgent, RetrievalAgent,
    AgentCategory, AgentResult, AgentMetrics
)

class MyCustomAgent(LLMAgent):
    def __init__(self, llm, config, enabled=True):
        super().__init__(llm=llm, enabled=enabled)
        self._config = config
    
    @property
    def name(self) -> str:
        return "MyCustomAgent"
    
    @property
    def category(self) -> AgentCategory:
        return AgentCategory.UTILITY
    
    @property
    def description(self) -> str:
        return "Custom agent for specific task"
    
    def _execute(self, query: str, **kwargs) -> str:
        """Core execution logic."""
        result = self._chat(
            system="You are a helpful assistant.",
            user=query,
        )
        return result
    
    def _on_error(self, error, metrics, **kwargs):
        """Fallback behavior on error."""
        return "Default fallback response"
```

### 10.2 Agent Metrics

```python
def _after_execute(self, result, metrics, **kwargs):
    """Add custom metrics after execution."""
    metrics.items_processed = len(result)
    metrics.confidence = 0.85
    metrics.custom["cache_hit"] = True
    return result
```

## Chapter 11: Core Agents Reference

### 11.1 PlanningAgent

Analyzes queries and selects retrieval strategy.

```python
from radiant.agents import PlanningAgent

agent = PlanningAgent(llm=llm, config=config.agentic)
result = agent.run(query="Compare Python and JavaScript")

if result.success:
    plan = result.data
    print(f"Mode: {plan['retrieval_mode']}")
    print(f"Decompose: {plan['should_decompose']}")
    print(f"Expand: {plan['should_expand']}")
```

### 11.2 Retrieval Agents

```python
from radiant.agents import DenseRetrievalAgent, BM25RetrievalAgent, RRFAgent

# Dense retrieval
dense_agent = DenseRetrievalAgent(store, local_models, config.retrieval)
dense_result = dense_agent.run(query=query, top_k=10)

# BM25 retrieval
bm25_agent = BM25RetrievalAgent(bm25_index, config.retrieval)
bm25_result = bm25_agent.run(query=query, top_k=10)

# RRF fusion
rrf_agent = RRFAgent(config.retrieval)
fused_result = rrf_agent.run(
    runs=[dense_result.data, bm25_result.data],
    top_k=15
)
```

### 11.3 Post-Retrieval Agents

```python
from radiant.agents import (
    HierarchicalAutoMergingAgent,
    CrossEncoderRerankingAgent,
    ContextEvaluationAgent,
    SummarizationAgent,
    MultiHopReasoningAgent,
)

# Auto-merge child chunks to parents
automerge_agent = HierarchicalAutoMergingAgent(store, config.automerge)
merged = automerge_agent.run(candidates=fused_docs, top_k=10)

# Cross-encoder reranking
rerank_agent = CrossEncoderRerankingAgent(local_models, config.rerank)
reranked = rerank_agent.run(query=query, docs=merged.data, top_k=8)

# Context evaluation
context_agent = ContextEvaluationAgent(llm, config.context_evaluation)
evaluation = context_agent.run(query=query, documents=reranked.data)

# Multi-hop reasoning
multihop_agent = MultiHopReasoningAgent(llm, store, local_models, config.multihop)
if multihop_agent.requires_multihop(query):
    reasoning = multihop_agent.run(query=query, initial_context=reranked.data)
```

### 11.4 Generation Agents

```python
from radiant.agents import AnswerSynthesisAgent, CriticAgent

# Answer synthesis
synthesis_agent = AnswerSynthesisAgent(llm, config.synthesis)
answer = synthesis_agent.run(
    query=query,
    docs=context_docs,
    conversation_history=""
)

# Critic evaluation
critic_agent = CriticAgent(llm, config.critic)
evaluation = critic_agent.run(
    query=query,
    answer=answer.data,
    context_docs=context_docs,
    is_retry=False
)
```

### 11.5 Verification Agents

```python
from radiant.agents import FactVerificationAgent, CitationTrackingAgent

# Fact verification
fact_agent = FactVerificationAgent(llm, config.fact_verification)
verification = fact_agent.run(
    answer=answer,
    context_documents=context_docs
)

# Citation tracking
citation_agent = CitationTrackingAgent(llm, config.citation)
cited_answer = citation_agent.run(
    answer=answer,
    query=query,
    source_documents=context_docs
)
```

---

# Part VIII: User Interfaces

## Chapter 12: CLI Interface

### 12.1 Command Overview

```bash
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

### 12.2 Interactive Mode

```bash
# Standard interactive mode
python -m radiant interactive

# TUI mode with rich interface
python -m radiant interactive --tui
```

## Chapter 13: Python API

### 13.1 RadiantRAG Application

```python
from radiant.app import RadiantRAG, create_app

# Create application
app = create_app("config.yaml")

# Or with custom config
app = RadiantRAG(config=my_config)

# Ingest documents
stats = app.ingest_documents(
    paths=["./docs/"],
    use_hierarchical=True,
    child_chunk_size=512,
    child_chunk_overlap=50,
)

# Query with full pipeline
result = app.query(
    query="What is RAG?",
    retrieval_mode="hybrid",
    show_result=True,
)
print(result.answer)
print(f"Confidence: {result.confidence}")

# Search only (no LLM)
results = app.search("BM25 algorithm", mode="hybrid", top_k=10)

# Simple query (minimal pipeline)
answer = app.simple_query("What is RAG?", top_k=5)
```

---

# Part IX: Metrics & Monitoring

## Chapter 14: Metrics Collection

### 14.1 AgentMetrics

Every agent execution collects metrics:

```python
@dataclass
class AgentMetrics:
    agent_name: str
    agent_category: str
    run_id: str
    correlation_id: str
    
    # Timing
    start_time: float
    end_time: float
    duration_ms: float
    
    # Status
    status: AgentStatus
    error_message: Optional[str]
    
    # Counters
    items_processed: int
    items_returned: int
    llm_calls: int
    retrieval_calls: int
    
    # Quality
    confidence: float
    
    # Custom metrics
    custom: Dict[str, Any]
```

### 14.2 Prometheus Export

```python
from radiant.utils.metrics_export import PrometheusMetricsExporter

exporter = PrometheusMetricsExporter(
    namespace="radiant",
    subsystem="agent",
    enable_histograms=True,
)

# Register agents
exporter.register_agent(planning_agent)
exporter.register_agent(retrieval_agent)

# Record executions
result = agent.run(query="test")
exporter.record_execution(result)

# Get metrics for endpoint
metrics_output = exporter.get_metrics_output()
```

### 14.3 OpenTelemetry Export

```python
from radiant.utils.metrics_export import OpenTelemetryExporter

exporter = OpenTelemetryExporter(
    service_name="radiant-rag",
    endpoint="http://localhost:4317",
)

# Trace agent execution
with exporter.trace_agent(agent, query="test"):
    result = agent.run(query="test")
    exporter.record_result(result)
```

### 14.4 Unified Collector

```python
from radiant.utils.metrics_export import MetricsCollector

collector = MetricsCollector.create(
    prometheus_enabled=True,
    prometheus_namespace="radiant",
    otel_enabled=True,
    otel_service_name="radiant-rag",
    otel_endpoint="http://localhost:4317",
)

# Record all executions
result = agent.run(query="test")
collector.record(result)

# Get Prometheus output
print(collector.prometheus_output())
```

---

# Part X: Advanced Topics

## Chapter 15: Strategy Memory

### 15.1 Configuration

```yaml
agentic:
  strategy_memory_enabled: true
  strategy_memory_path: "./data/strategy_memory.json.gz"
```

### 15.2 Usage

```python
from radiant.agents.strategy_memory import RetrievalStrategyMemory

memory = RetrievalStrategyMemory(config.agentic)

# Record outcome
memory.record_outcome(
    query=query,
    strategy_used="hybrid",
    confidence=0.85,
    retrieval_quality=0.9,
)

# Get recommended strategy
recommendation = memory.recommend_strategy(query)
```

## Chapter 16: Citation Tracking

### 16.1 Configuration

```yaml
citation:
  enabled: true
  citation_style: "inline"  # inline, footnote, academic, enterprise
  generate_bibliography: true
  generate_audit_trail: true
```

### 16.2 Citation Styles

| Style | Format | Use Case |
|-------|--------|----------|
| inline | `[1]` | General |
| footnote | `^1` | Academic papers |
| academic | `(Author, Year)` | Research |
| enterprise | `[DOC-ID-123]` | Audit compliance |

## Chapter 17: Multi-Hop Reasoning

### 17.1 Configuration

```yaml
multihop:
  enabled: true
  max_hops: 3
  docs_per_hop: 5
  min_confidence_to_continue: 0.3
```

### 17.2 Query Detection

| Pattern | Example | Hops |
|---------|---------|------|
| Bridge questions | "Who founded the company that created Python?" | 2 |
| Comparison | "Is Python faster than the language Google uses?" | 2-3 |
| Temporal | "What happened after the event in 1991?" | 2 |
| Causal chain | "What caused the result of X?" | 2-3 |

---

# Part XI: API Reference

## Chapter 18: PipelineResult

```python
@dataclass
class PipelineResult:
    answer: str                    # Generated answer
    context: AgentContext          # Pipeline context
    metrics: RunMetrics            # Performance metrics
    success: bool = True
    error: Optional[str] = None
    
    # Agentic enhancements
    confidence: float = 0.0
    retrieval_mode_used: str = "hybrid"
    retry_count: int = 0
    tools_used: List[str] = []
    low_confidence: bool = False
    
    # Multi-hop reasoning
    multihop_used: bool = False
    multihop_hops: int = 0
    
    # Fact verification
    fact_verification_score: float = 1.0
    fact_verification_passed: bool = True
    
    # Citation tracking
    cited_answer: Optional[str] = None
    citations: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    audit_id: Optional[str] = None
```

## Chapter 19: RadiantRAG Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `query` | query, mode, conversation_id | PipelineResult | Full pipeline query |
| `query_raw` | query, mode | PipelineResult | Query without display |
| `simple_query` | query, top_k | str | Minimal pipeline |
| `search` | query, mode, top_k | List[Tuple] | Retrieval only |
| `ingest_documents` | paths, hierarchical | Dict | Ingest files |
| `ingest_urls` | urls | Dict | Ingest URLs |
| `ingest_github` | url | Dict | Ingest GitHub repo |
| `clear_index` | keep_bm25 | bool | Clear all documents |
| `check_health` | - | Dict | System health |
| `get_stats` | - | Dict | System statistics |

---

# Part XII: Troubleshooting

## Chapter 20: Common Issues

### 20.1 Connection Issues

**Redis connection failed**
```bash
# Check Redis is running
docker ps | grep redis-stack

# Start Redis
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server

# Test connection
python tools/check_redis.py
```

**PostgreSQL connection failed**
```bash
# Verify connection string
export PG_CONN_STR="postgresql://user:pass@localhost:5432/radiant"

# Test pgvector extension
psql -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
```

### 20.2 Retrieval Issues

**No documents found**
```bash
# Check index status
python -m radiant stats

# Inspect index
python tools/inspect_index.py

# Clear and re-ingest
python -m radiant clear --confirm
python -m radiant ingest ./docs/
```

**Quantization not working**
```bash
# Validate implementation
python tools/validate_quantization.py

# Check calibration file
python tools/calibrate_int8_ranges.py --sample-size 100000 --output data/int8_ranges.npy
```

### 20.3 Performance Issues

**Slow ingestion**
```yaml
# Increase batch sizes
ingestion:
  embedding_batch_size: 64
  redis_batch_size: 200
```

**High memory usage**
```yaml
# Enable quantization
redis:
  quantization:
    enabled: true
    precision: "both"
```

## Chapter 21: Diagnostic Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `check_redis.py` | Test Redis connection | `python tools/check_redis.py` |
| `inspect_index.py` | Inspect index contents | `python tools/inspect_index.py` |
| `validate_quantization.py` | Validate quantization | `python tools/validate_quantization.py` |
| `calibrate_int8_ranges.py` | Calibrate int8 | `python tools/calibrate_int8_ranges.py` |
| `validate_bugfix.py` | Validate fixes | `python tools/validate_bugfix.py` |

---

# Appendices

## Appendix A: Error Reference

| Code | Error | Cause | Solution |
|------|-------|-------|----------|
| E001 | Redis connection failed | Redis not running | Start Redis |
| E002 | LLM request timeout | Slow LLM response | Increase timeout |
| E003 | Embedding model load failed | Model not found | Check model name |
| E004 | Document parsing failed | Invalid file format | Check file |
| E005 | Index creation failed | Redis memory full | Add memory |
| E006 | Vector dimension mismatch | Wrong model | Match dimensions |
| E007 | Translation failed | LLM unavailable | Check endpoint |
| E008 | Language detection failed | Missing dependency | Install fast-langdetect |
| E009 | Tool execution failed | Invalid input | Check input |
| E010 | Conversation not found | Invalid ID | Verify ID |
| E011 | Configuration invalid | YAML syntax error | Validate YAML |
| E012 | Memory limit exceeded | Too many documents | Reduce batch |
| E013 | GPU out of memory | Model too large | Enable 4bit |
| E014 | File not found | Invalid path | Check path |
| E015 | Quantization failed | Missing calibration | Run calibration |

## Appendix B: Supported Languages

The LanguageDetectionAgent supports 176 languages including:

**Major Languages:** English (en), Chinese (zh), Spanish (es), French (fr), German (de), Japanese (ja), Korean (ko), Portuguese (pt), Russian (ru), Arabic (ar)

**European Languages:** Bulgarian (bg), Czech (cs), Danish (da), Greek (el), Finnish (fi), Hungarian (hu), Norwegian (no), Romanian (ro), Slovak (sk), Swedish (sv)

**Asian Languages:** Thai (th), Vietnamese (vi), Indonesian (id), Malay (ms), Tagalog (tl), Bengali (bn), Tamil (ta), Telugu (te), Hindi (hi)

## Appendix C: Complete Configuration Reference

```yaml
# =============================================================================
# RADIANT RAG COMPLETE CONFIGURATION
# =============================================================================

ollama:
  openai_base_url: "http://localhost:11434/v1"
  openai_api_key: "ollama"
  chat_model: "qwen2.5:latest"
  timeout: 90
  max_retries: 3
  retry_delay: 1.0

local_models:
  embed_model_name: "sentence-transformers/all-MiniLM-L12-v2"
  cross_encoder_name: "cross-encoder/ms-marco-MiniLM-L12-v2"
  device: "auto"
  embedding_dimension: 384

storage:
  backend: redis  # redis, chroma, pgvector

redis:
  url: "redis://localhost:6379/0"
  key_prefix: "radiant"
  quantization:
    enabled: false
    precision: "both"
    rescore_multiplier: 4.0
    int8_ranges_file: "data/int8_ranges.npy"

bm25:
  index_path: "./data/bm25_index"
  k1: 1.5
  b: 0.75

retrieval:
  dense_top_k: 10
  bm25_top_k: 10
  fused_top_k: 15
  rrf_k: 60
  search_scope: "leaves"

rerank:
  top_k: 8
  max_doc_chars: 3000

synthesis:
  max_context_docs: 8
  max_doc_chars: 4000

critic:
  enabled: true
  retry_on_issues: true
  max_retries: 2
  confidence_threshold: 0.4

agentic:
  dynamic_retrieval_mode: true
  tools_enabled: true
  strategy_memory_enabled: true
  strategy_memory_path: "./data/strategy_memory.json.gz"

multihop:
  enabled: true
  max_hops: 3
  docs_per_hop: 5

fact_verification:
  enabled: true
  min_support_confidence: 0.6
  max_claims_to_verify: 20

citation:
  enabled: true
  citation_style: "inline"
  generate_bibliography: true
  generate_audit_trail: true

language_detection:
  enabled: true
  method: "fast"
  min_confidence: 0.7

translation:
  enabled: true
  method: "llm"
  canonical_language: "en"
  translate_at_ingestion: true

logging:
  level: "INFO"
  colorize: true

metrics:
  enabled: true
  detailed_timing: true
```

---

*End of Radiant Agentic RAG User Manual*

**Version 1.4 | January 2025**

For support and updates, visit the project repository.
