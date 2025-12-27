# Radiant Agentic RAG
## User Manual

**Version 1.0**  
**December 2024**

---

## Document Information

| Field | Value |
|-------|-------|
| Document Title | Radiant Agentic RAG User Manual |
| Version | 1.0 |
| Release Date | December 2024 |
| Target Audience | Software Developers, Data Scientists, AI Engineers |

---

# Table of Contents

- [Part I: Introduction](#part-i-introduction)
- [Part II: Installation & Setup](#part-ii-installation--setup)
- [Part III: Core Concepts](#part-iii-core-concepts)
- [Part IV: Document Ingestion](#part-iv-document-ingestion)
- [Part V: Agents Reference](#part-v-agents-reference)
- [Part VI: Storage & Indexing](#part-vi-storage--indexing)
- [Part VII: User Interfaces](#part-vii-user-interfaces)
- [Part VIII: Advanced Topics](#part-viii-advanced-topics)
- [Part IX: API Reference](#part-ix-api-reference)
- [Part X: Troubleshooting](#part-x-troubleshooting)
- [Appendices](#appendices)

---

# Part I: Introduction

## Chapter 1: Overview

### 1.1 What is Radiant RAG?

Radiant RAG is an enterprise-grade, agentic Retrieval-Augmented Generation (RAG) system designed for building intelligent document question-answering applications. Unlike traditional RAG implementations that follow a fixed pipeline, Radiant RAG employs a modular, agent-based architecture where specialized agents collaborate to process queries, retrieve relevant information, and generate accurate, well-cited responses.

**Core Design Principles:**

1. **Agentic Architecture**: Each component is an autonomous agent with a specific responsibility
2. **Dynamic Strategy Selection**: The system adapts its retrieval strategy based on query characteristics
3. **Self-Correction**: Built-in critic and verification agents detect and correct errors
4. **Enterprise Compliance**: Citation tracking and audit trails support regulatory requirements
5. **Multilingual Support**: Automatic language detection and translation for multilingual corpora

### 1.2 Key Features

| Category | Agents | Purpose |
|----------|--------|---------|
| Query Processing | Planning, Decomposition, Rewrite, Expansion | Understand and optimize queries |
| Retrieval | Dense, BM25, Web Search | Fetch relevant documents |
| Fusion | RRF | Combine results from multiple retrievers |
| Post-Retrieval | AutoMerge, Rerank, Context Eval | Refine and validate context |
| Generation | Synthesis, Critic | Generate and evaluate answers |
| Verification | Fact Verification, Citation | Ensure accuracy and attribution |
| Multilingual | Language Detection, Translation | Support multiple languages |
| Tools | Calculator, Code Execution | Extended capabilities |

### 1.3 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.11 |
| RAM | 8 GB | 32+ GB |
| Redis Stack | 7.2+ | Latest |
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

### 2.2 Redis Stack

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

### 2.3 Installing from Source

```bash
git clone https://github.com/your-org/radiant-rag.git
cd radiant-rag
pip install -e .
```

### 2.4 Verifying Installation

```bash
python -c "from radiant.app import RadiantRAG; print('OK')"
python -m radiant --health-check
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

redis:
  url: "redis://localhost:6379/0"
  key_prefix: "radiant"
```

### 3.2 Environment Variables

```bash
export RADIANT_OLLAMA_CHAT_MODEL="llama3:70b"
export RADIANT_REDIS_URL="redis://redis-server:6379/0"
export RADIANT_RETRIEVAL_TOP_K="20"
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

```python
@dataclass
class AgentContext:
    query: str
    processed_queries: List[str]
    retrieval_mode: str
    documents: List[StoredDoc]
    context: str
    answer: str
    confidence: float
    citations: List[Citation]
    metrics: Dict[str, Any]
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

### 6.2 DocumentProcessor

```python
from radiant.ingestion import DocumentProcessor

processor = DocumentProcessor(
    cleaning_config=config.unstructured_cleaning,
    chunk_size=512,
    chunk_overlap=50,
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
    preserve_original=True,
)

# Process multilingual document
chunks = processor.process_file("/path/to/chinese_doc.pdf")
```

### 6.4 Batch Ingestion

```bash
python -m radiant ingest /path/to/docs/ --recursive --batch-size 10
```

---

# Part V: Agents Reference

## Chapter 8: PlanningAgent

### Purpose
Analyzes incoming queries and determines optimal retrieval strategy.

### Configuration
```yaml
agentic:
  use_planning: true
  min_confidence: 0.7
```

### Input/Output
- **Input**: Query string, conversation history
- **Output**: RetrievalPlan (mode, use_decomposition, confidence_threshold)

### Decision Logic
| Query Type | Retrieval Mode |
|------------|----------------|
| Conceptual/semantic | Dense |
| Exact terms/code | Sparse (BM25) |
| Technical/mixed | Hybrid |
| Current events | Web search |

### Usage Example
```
Query: "Compare Python and Java performance"
Plan:
  mode: hybrid
  use_decomposition: true
  reasoning: "Complex comparison requiring multiple perspectives"
```

---

## Chapter 9: QueryDecompositionAgent

### Purpose
Breaks complex queries into simpler sub-queries.

### Configuration
```yaml
query:
  max_decomposed_queries: 5
```

### Usage Example
```
Input: "What is Python and how does it compare to Java?"
Output:
  - "What is Python?"
  - "How does Python compare to Java?"
```

---

## Chapter 10: QueryRewriteAgent

### Purpose
Reformulates queries to improve retrieval effectiveness.

### Techniques
| Technique | Example |
|-----------|---------|
| Clarification | "it" → "the algorithm" |
| Contextualization | "more details" → "more details about neural networks" |

---

## Chapter 11: QueryExpansionAgent

### Purpose
Enriches queries with synonyms and related terms.

### Configuration
```yaml
query:
  max_expansions: 12
```

### Usage Example
```
Query: "ML model training"
Expansions: machine learning, neural network, deep learning, optimization
```

---

## Chapter 12: DenseRetrievalAgent

### Purpose
Semantic similarity search using vector embeddings.

### Configuration
```yaml
retrieval:
  top_k: 10
  min_similarity: 0.5

local_models:
  embed_model_name: "sentence-transformers/all-MiniLM-L12-v2"
```

### Recommended Models
| Model | Dimensions | Use Case |
|-------|------------|----------|
| all-MiniLM-L12-v2 | 384 | General, fast |
| all-mpnet-base-v2 | 768 | Higher quality |
| multilingual-e5-large | 1024 | Multilingual |

---

## Chapter 13: BM25RetrievalAgent

### Purpose
Keyword-based sparse retrieval using BM25 algorithm.

### Configuration
```yaml
bm25:
  k1: 1.5
  b: 0.75
  top_k: 10
```

### Best For
- Code identifiers
- Proper nouns
- Technical terms
- Exact phrases

---

## Chapter 14: WebSearchAgent

### Purpose
Real-time web searches for current information.

### Configuration
```yaml
web_search:
  enabled: true
  provider: "duckduckgo"
  max_results: 10
  trigger_keywords:
    - "latest"
    - "recent"
    - "today"
```

---

## Chapter 15: RRFAgent

### Purpose
Combines results from multiple retrievers using Reciprocal Rank Fusion.

### Formula
```
RRF_score(d) = Σ (weight_i / (k + rank_i(d)))
```

### Configuration
```yaml
pipeline:
  use_rrf: true
rrf_k: 60
```

---

## Chapter 16: HierarchicalAutoMergingAgent

### Purpose
Reconstructs parent documents from retrieved child chunks.

### Configuration
```yaml
automerge:
  enabled: true
  threshold: 2
  max_parent_size: 10000
```

---

## Chapter 17: CrossEncoderRerankingAgent

### Purpose
Neural reranking for precision using cross-encoder models.

### Configuration
```yaml
rerank:
  enabled: true
  top_k: 20
  model_name: "cross-encoder/ms-marco-MiniLM-L12-v2"
```

---

## Chapter 18: ContextEvaluationAgent

### Purpose
Assesses context quality before generation.

### Configuration
```yaml
context_evaluation:
  enabled: true
  min_relevance_score: 0.6
  min_coverage_score: 0.5
```

### Output
```python
@dataclass
class ContextEvaluation:
    relevance_score: float
    coverage_score: float
    coherence_score: float
    is_sufficient: bool
    missing_aspects: List[str]
```

---

## Chapter 19: SummarizationAgent

### Purpose
Compresses context to fit LLM token limits.

### Configuration
```yaml
summarization:
  enabled: true
  max_summary_length: 4000
  strategy: "abstractive"
```

### Strategies
| Strategy | Description |
|----------|-------------|
| extractive | Select key sentences |
| abstractive | Generate new summary |
| hierarchical | Multi-level compression |

---

## Chapter 20: IntelligentChunkingAgent

### Purpose
Semantic document chunking using LLM.

### Configuration
```yaml
chunking:
  enabled: true
  method: "llm"
  target_chunk_size: 1000
```

### Document Types
| Type | Detection | Strategy |
|------|-----------|----------|
| Code | Extension, syntax | Function boundaries |
| Prose | Paragraphs | Section boundaries |
| Markdown | Headers | Header-based |

---

## Chapter 21: AnswerSynthesisAgent

### Purpose
Generates answers from retrieved context.

### Configuration
```yaml
synthesis:
  max_tokens: 1024
  temperature: 0.7
  include_sources: true
```

---

## Chapter 22: CriticAgent

### Purpose
Evaluates answer quality and triggers regeneration if needed.

### Configuration
```yaml
critic:
  enabled: true
  min_confidence: 0.7
  max_retries: 2
```

### Output
```python
@dataclass
class CriticEvaluation:
    confidence: float
    accuracy_score: float
    completeness_score: float
    should_retry: bool
    feedback: str
```

---

## Chapter 23: FactVerificationAgent

### Purpose
Verifies each claim against source documents.

### Configuration
```yaml
fact_verification:
  enabled: true
  min_support_confidence: 0.6
  generate_corrections: true
```

### Verification Status
| Status | Description | Score |
|--------|-------------|-------|
| SUPPORTED | Fully backed | 1.0 |
| PARTIALLY_SUPPORTED | Some evidence | 0.7 |
| NOT_SUPPORTED | No evidence | 0.3 |
| CONTRADICTED | Evidence contradicts | 0.0 |

---

## Chapter 24: CitationTrackingAgent

### Purpose
Adds source attribution and audit trails.

### Configuration
```yaml
citation:
  enabled: true
  citation_style: "inline"
  generate_bibliography: true
  generate_audit_trail: true
```

### Citation Styles
| Style | Format |
|-------|--------|
| inline | [1], [2] |
| footnote | ^[1] |
| academic | (Author Year) |
| hyperlink | [text](url) |

---

## Chapter 25: MultiHopReasoningAgent

### Purpose
Handles complex queries requiring multiple reasoning steps.

### Configuration
```yaml
multihop:
  enabled: true
  max_hops: 3
  docs_per_hop: 5
```

### Example
```
Query: "Who is the CEO of the company that makes iPhone?"

Step 1: "Which company makes iPhone?" → Apple Inc.
Step 2: "Who is the CEO of Apple Inc.?" → Tim Cook

Final Answer: Tim Cook
```

---

## Chapter 26: RetrievalStrategyMemory

### Purpose
Learns optimal retrieval strategies from outcomes.

### Configuration
```yaml
agentic:
  use_strategy_memory: true
  strategy_memory_size: 1000
```

---

## Chapter 27: LanguageDetectionAgent

### Purpose
Identifies text language using FastText with LLM fallback.

### Configuration
```yaml
language_detection:
  enabled: true
  method: "fast"
  min_confidence: 0.7
  use_llm_fallback: true
```

### Output
```python
@dataclass
class LanguageDetection:
    language_code: str      # "en", "zh", "fr"
    language_name: str      # "English", "Chinese"
    confidence: float       # 0.0-1.0
    method: str            # "fast", "llm"
```

### Supported Languages
176 languages via FastText including: English, Chinese, Spanish, French, German, Japanese, Korean, Arabic, Hindi, Portuguese, Russian, and many more.

---

## Chapter 28: TranslationAgent

### Purpose
Translates documents to canonical language using LLM.

### Configuration
```yaml
translation:
  enabled: true
  method: "llm"
  canonical_language: "en"
  max_chars_per_llm_call: 4000
  preserve_original: true
```

### Output
```python
@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    was_translated: bool
```

---

## Chapter 29-31: Tool Agents

### ToolRegistry
Manages available tools (calculator, code execution).

### CalculatorTool
```python
result = calculator.execute("sqrt(144) + 15")
# output: "27.0"
```

### CodeExecutionTool
```python
code = """
import statistics
values = [23, 45, 67, 89]
print(f"Mean: {statistics.mean(values):.2f}")
"""
result = code_tool.execute(code)
```

---

# Part VI: Storage & Indexing

## Chapter 32: Redis Vector Store

### Index Schema
```python
schema = [
    TextField("content"),
    TagField("doc_level"),
    TagField("parent_id"),
    TagField("language_code"),
    VectorField("embedding", "HNSW", {...}),
]
```

### HNSW Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| M | 16 | Max connections per node |
| EF_CONSTRUCTION | 200 | Build-time search width |
| EF_RUNTIME | 100 | Query-time search width |

### Vector Search with Language Filter
```python
results = store.retrieve_by_embedding(
    query_embedding=query_vec,
    top_k=10,
    language_filter="zh",  # Only Chinese documents
)
```

---

# Part VII: User Interfaces

## Chapter 35: Command Line Interface

```bash
# Ingest documents
python -m radiant ingest /path/to/docs/ -r

# Query
python -m radiant query "What is machine learning?"

# Search
python -m radiant search "Python tutorial" --top-k 5

# Stats
python -m radiant stats

# Clear index
python -m radiant clear --confirm
```

## Chapter 36: Terminal User Interface

```bash
python -m radiant tui
```

| Key | Action |
|-----|--------|
| Tab | Switch panels |
| Enter | Submit query |
| ↑/↓ | Scroll results |
| Ctrl+C | Exit |

---

# Part VIII: Advanced Topics

## Chapter 38: Pipeline Customization

### Enabling/Disabling Agents
```yaml
pipeline:
  use_planning: true
  use_decomposition: true
  use_rewrite: true
  use_critic: true

context_evaluation:
  enabled: true

fact_verification:
  enabled: false
```

### Custom Agent Development
```python
class CustomAgent:
    def __init__(self, llm, config):
        self._llm = llm
        self._config = config
    
    def run(self, context: AgentContext) -> AgentContext:
        # Process and modify context
        return context
```

## Chapter 39: Performance Optimization

### Batch Processing
```python
app.ingest(paths, batch_size=50)
```

### GPU Acceleration
```yaml
local_models:
  device: "cuda"
vlm:
  load_in_4bit: true
```

## Chapter 41: Multilingual Deployment

### Translation Workflow
```
Document (Chinese)
    → Detect: "zh"
    → Translate to English
    → Embed English
    → Store: {content: English, original: Chinese}
```

### Language Filtering
```python
# Search only Chinese source documents
results = store.retrieve_by_embedding(
    query_embedding=query_vec,
    language_filter="zh",
)
```

---

# Part IX: API Reference

## Chapter 42: RadiantRAG Class

```python
from radiant.app import RadiantRAG

app = RadiantRAG(config_path="/path/to/config.yaml")

# Ingest
stats = app.ingest(paths=["/docs/"], recursive=True)

# Query
result = app.query("What is machine learning?")

# Search
docs = app.search("Python", top_k=10)

# Stats
stats = app.get_stats()
```

## Chapter 43: PipelineResult

```python
@dataclass
class PipelineResult:
    query: str
    answer: str
    confidence: float
    documents: List[StoredDoc]
    cited_answer: Optional[str]
    citations: List[Citation]
    fact_verification_score: Optional[float]
    total_latency_ms: float
    metrics: Dict[str, Any]
```

---

# Part X: Troubleshooting

## Chapter 46: Common Issues

| Issue | Solution |
|-------|----------|
| Redis connection refused | Check Redis is running: `docker ps` |
| LLM timeout | Increase `ollama.timeout` |
| CUDA out of memory | Enable `vlm.load_in_4bit: true` |
| fast-langdetect not available | `pip install fast-langdetect` |

## Chapter 47: Debugging

```yaml
logging:
  level: "DEBUG"
  file: "radiant.log"
```

```bash
grep "ERROR" radiant.log
grep "Agent" radiant.log
```

---

# Appendices

## Appendix A: Environment Variables

| Variable | Description |
|----------|-------------|
| RADIANT_OLLAMA_OPENAI_BASE_URL | LLM endpoint |
| RADIANT_OLLAMA_CHAT_MODEL | Model name |
| RADIANT_REDIS_URL | Redis connection |
| RADIANT_RETRIEVAL_TOP_K | Results count |
| RADIANT_PIPELINE_USE_CRITIC | Enable critic |
| RADIANT_TRANSLATION_CANONICAL_LANGUAGE | Target language |

## Appendix B: Language Codes

| Code | Language | Code | Language |
|------|----------|------|----------|
| en | English | ja | Japanese |
| zh | Chinese | ko | Korean |
| es | Spanish | ar | Arabic |
| fr | French | hi | Hindi |
| de | German | pt | Portuguese |
| it | Italian | ru | Russian |

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| Agent | Autonomous component performing specific task |
| Agentic RAG | RAG with autonomous decision-making agents |
| BM25 | Probabilistic keyword ranking function |
| Canonical Language | Target language for indexing |
| Cross-Encoder | Neural model for query-document scoring |
| Dense Retrieval | Semantic vector similarity search |
| Embedding | Dense vector representation of text |
| HNSW | Hierarchical Navigable Small World algorithm |
| Multi-Hop | Reasoning requiring multiple evidence steps |
| RRF | Reciprocal Rank Fusion |
| Sparse Retrieval | Keyword-based search |

---

*Radiant Agentic RAG User Manual v1.0 | December 2024*

---

# Extended Agent Documentation

## PlanningAgent - Detailed Reference

### 8.1 Purpose

The PlanningAgent is the first agent in the pipeline, responsible for analyzing incoming queries and determining the optimal retrieval and processing strategy. It acts as a router that directs queries to appropriate downstream agents based on query characteristics.

### 8.2 Configuration Parameters

```yaml
agentic:
  # Enable dynamic planning
  use_planning: true
  
  # Confidence threshold for accepting answers
  min_confidence: 0.7
  
  # Maximum retry attempts
  max_retries: 2

pipeline:
  use_planning: true
```

### 8.3 Input/Output Schema

**Input:**
- Query string
- Conversation history (optional)
- User preferences (optional)

**Output:**

```python
@dataclass
class RetrievalPlan:
    mode: str                    # "dense", "sparse", "hybrid", "web"
    use_decomposition: bool      # Whether to decompose query
    use_rewrite: bool            # Whether to rewrite query
    use_expansion: bool          # Whether to expand terms
    confidence_threshold: float  # Required confidence level
    retrieval_depth: int         # Number of documents to retrieve
    reasoning: str               # Explanation of decisions
```

### 8.4 Decision Logic

The PlanningAgent uses the following decision tree:

```
Query Analysis
├── Is it a current events query?
│   └── Yes → Web Search mode
├── Is it a complex multi-part query?
│   └── Yes → Enable decomposition
├── Does it contain specific technical terms?
│   └── Yes → Hybrid mode (dense + BM25)
├── Is it a semantic/conceptual query?
│   └── Yes → Dense mode
└── Is it a keyword/exact match query?
    └── Yes → Sparse (BM25) mode
```

### 8.5 Usage Examples

**Example 1: Simple Factual Query**

```
Query: "What is the capital of France?"
Plan:
  mode: dense
  use_decomposition: false
  use_rewrite: false
  reasoning: "Simple factual query, semantic search sufficient"
```

**Example 2: Complex Comparison Query**

```
Query: "Compare Python and Java performance, and explain when to use each"
Plan:
  mode: hybrid
  use_decomposition: true
  use_rewrite: true
  reasoning: "Complex comparison requiring multiple perspectives"
```

**Example 3: Current Events Query**

```
Query: "What happened in the stock market today?"
Plan:
  mode: web
  use_decomposition: false
  reasoning: "Real-time information required"
```

---

## FactVerificationAgent - Detailed Reference

### 23.1 Purpose

The FactVerificationAgent verifies each claim in a generated answer against the source documents. It identifies unsupported, contradicted, or unverifiable statements and can generate corrected versions.

### 23.2 Configuration Parameters

```yaml
fact_verification:
  # Enable fact verification
  enabled: true
  
  # Minimum support confidence
  min_support_confidence: 0.6
  
  # Maximum claims to verify
  max_claims_to_verify: 20
  
  # Generate corrected answers
  generate_corrections: true
  
  # Strict mode (fail on any contradiction)
  strict_mode: false
  
  # Minimum factuality score
  min_factuality_score: 0.5
  
  # Block answers that fail verification
  block_on_failure: false
```

### 23.3 Input/Output Schema

**Input:**
- Generated answer
- Source context

**Output:**

```python
@dataclass
class Claim:
    text: str                   # The claim statement
    source_sentence: str        # Original sentence

@dataclass
class ClaimVerification:
    claim: Claim
    status: VerificationStatus  # SUPPORTED, PARTIALLY_SUPPORTED, etc.
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]

class VerificationStatus(Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"

@dataclass
class FactVerificationResult:
    claims: List[ClaimVerification]
    overall_score: float        # Weighted factuality score
    is_factual: bool           # Passes threshold
    needs_correction: bool      # Has contradictions/unsupported
    corrected_answer: Optional[str]  # If corrections generated
```

### 23.4 Verification Process

1. **Claim Extraction**: Parse answer into individual verifiable claims
2. **Evidence Matching**: Find supporting/contradicting passages
3. **Status Assignment**: Classify each claim
4. **Score Calculation**: Compute weighted factuality score
5. **Correction Generation**: If enabled, produce corrected version

### 23.5 Usage Example

```python
result = fact_verification_agent.verify_answer(
    answer=generated_answer,
    context=source_documents,
)

print(f"Factuality Score: {result.overall_score:.2f}")
print(f"Is Factual: {result.is_factual}")

for cv in result.claims:
    status_icon = "✓" if cv.status == VerificationStatus.SUPPORTED else "✗"
    print(f"{status_icon} {cv.claim.text[:50]}... ({cv.status.value})")
```

---

## CitationTrackingAgent - Detailed Reference

### 24.1 Purpose

The CitationTrackingAgent adds source attribution to generated answers. It tracks which documents support each statement and formats citations according to configured styles.

### 24.2 Configuration Parameters

```yaml
citation:
  # Enable citation tracking
  enabled: true
  
  # Citation style: inline, footnote, academic, hyperlink, enterprise
  citation_style: "inline"
  
  # Minimum citation confidence
  min_citation_confidence: 0.5
  
  # Maximum citations per claim
  max_citations_per_claim: 3
  
  # Include excerpts from sources
  include_excerpts: true
  excerpt_max_length: 200
  
  # Generate bibliography
  generate_bibliography: true
  
  # Generate audit trail
  generate_audit_trail: true
```

### 24.3 Citation Styles

| Style | Format | Example |
|-------|--------|---------|
| inline | [n] | "Python is interpreted [1]." |
| footnote | ^[n] | "Python is interpreted^[1]." |
| academic | (Author Year) | "Python is interpreted (Van Rossum 1991)." |
| hyperlink | [text](url) | "[Python is interpreted](doc.pdf)." |
| enterprise | {ref:id} | "Python is interpreted {ref:DOC001}." |

### 24.4 Audit Trail

For enterprise compliance:

```python
audit = cited_answer.audit_log
# {
#     "audit_id": "a1b2c3d4e5f6",
#     "timestamp": "2024-12-15T10:30:00Z",
#     "query": "Original query",
#     "answer_hash": "sha256:...",
#     "sources": [
#         {"id": "doc1", "path": "/docs/file.pdf", "accessed_at": "..."},
#     ],
#     "citations": [...],
#     "verification_score": 0.95,
# }
```

### 24.5 Usage Example

```python
cited = citation_agent.create_cited_answer(
    answer=generated_answer,
    sources=retrieved_documents,
)

print(cited.cited_answer)
# Output: "Python is an interpreted language [1] that supports..."

print(cited.bibliography)
# Output: "[1] Python Reference. /docs/python.pdf"

report = citation_agent.generate_audit_report(cited)
```

---

## LanguageDetectionAgent - Detailed Reference

### 27.1 Purpose

The LanguageDetectionAgent identifies the language of text content. It uses fast-langdetect (FastText) for efficient detection with optional LLM fallback for ambiguous cases.

### 27.2 Configuration Parameters

```yaml
language_detection:
  # Enable language detection
  enabled: true
  
  # Detection method: "fast", "llm", "auto"
  method: "fast"
  
  # Minimum confidence threshold
  min_confidence: 0.7
  
  # Use LLM for low-confidence cases
  use_llm_fallback: true
  
  # Default language if detection fails
  fallback_language: "en"
```

### 27.3 Detection Methods

| Method | Engine | Speed | Languages | Best For |
|--------|--------|-------|-----------|----------|
| fast | FastText | ~0.1ms | 176 | High volume |
| llm | LLM | ~500ms | Unlimited | Ambiguous text |
| auto | Fast + LLM fallback | Varies | 176+ | Balanced |

### 27.4 Confidence Handling

```python
detection = agent.detect(text)

if detection.confidence >= 0.7:
    # High confidence, use result directly
    return detection
elif use_llm_fallback:
    # Try LLM for better accuracy
    return agent._detect_with_llm(text)
else:
    # Return with low confidence flag
    return detection
```

### 27.5 Usage Examples

```python
# Basic detection
detection = lang_agent.detect("这是中文文本")
print(f"Language: {detection.language_name}")  # Chinese
print(f"Code: {detection.language_code}")      # zh
print(f"Confidence: {detection.confidence:.2f}")  # 0.99

# Document-level detection
detection = lang_agent.detect_document(long_document_text)

# Batch detection
detections = lang_agent.detect_batch([text1, text2, text3])

# Detection with context hint
detection = lang_agent.detect_with_context(
    text=short_chunk,
    document_language="zh",
)
```

---

## TranslationAgent - Detailed Reference

### 28.1 Purpose

The TranslationAgent translates text between languages using LLM for high-quality translations. It supports automatic chunking for long documents and preserves formatting.

### 28.2 Configuration Parameters

```yaml
translation:
  # Enable translation
  enabled: true
  
  # Translation method
  method: "llm"
  
  # Target language for indexing
  canonical_language: "en"
  
  # Maximum chars per LLM call
  max_chars_per_llm_call: 4000
  
  # Translate at ingestion time
  translate_at_ingestion: true
  
  # Preserve original text
  preserve_original: true
```

### 28.3 Translation Workflow

```
Document (Any Language)
    │
    ▼
LanguageDetectionAgent
    │ detect source language
    ▼
Is canonical language?
    │
    ├── Yes → Skip translation
    │
    └── No → TranslationAgent
             │ translate to canonical
             ▼
        Store: {
          content: translated_text,
          meta: {
            language_code: source_lang,
            original_content: original_text,
            was_translated: true
          }
        }
```

### 28.4 Quality Preservation

The agent preserves:
- Paragraph structure
- Bullet points and lists
- Code blocks and technical terms
- Proper nouns
- Formatting markers

### 28.5 Usage Examples

```python
# Basic translation
result = translator.translate(
    text="Bonjour le monde",
    target_language="en",
    source_language="fr",
)
print(result.translated_text)  # "Hello world"

# Translate to canonical language
result = translator.translate_to_canonical(chinese_document)

# Check if translation needed
if translator.needs_translation(source_lang="fr"):
    result = translator.translate(text)
```

---

## MultiHopReasoningAgent - Detailed Reference

### 25.1 Purpose

The MultiHopReasoningAgent handles complex queries requiring multiple steps of reasoning and evidence gathering. It decomposes queries into reasoning chains and iteratively retrieves evidence.

### 25.2 Configuration Parameters

```yaml
multihop:
  # Enable multi-hop reasoning
  enabled: true
  
  # Maximum reasoning hops
  max_hops: 3
  
  # Documents to retrieve per hop
  docs_per_hop: 5
  
  # Minimum confidence to continue
  min_confidence_to_continue: 0.3
  
  # Enable entity extraction
  enable_entity_extraction: true
```

### 25.3 Reasoning Chain Example

Query: "Who is the CEO of the company that makes iPhone?"

```
Step 1:
  Sub-question: "Which company makes iPhone?"
  Evidence: "Apple Inc. designs and manufactures the iPhone..."
  Answer: Apple Inc.
  Entities extracted: [Apple Inc., iPhone]

Step 2:
  Sub-question: "Who is the CEO of Apple Inc.?"
  Evidence: "Tim Cook has served as CEO of Apple Inc. since..."
  Answer: Tim Cook
  
Final Answer: Tim Cook is the CEO of Apple Inc., the company that makes iPhone.
```

### 25.4 Query Detection

The agent identifies multi-hop queries:

| Pattern | Example | Hops |
|---------|---------|------|
| Bridge questions | "Who founded the company that created Python?" | 2 |
| Comparison | "Is Python faster than the language Google uses?" | 2-3 |
| Temporal | "What happened after the event in 1991?" | 2 |
| Causal chain | "What caused the result of X?" | 2-3 |

### 25.5 Usage Example

```python
if multihop_agent.requires_multihop(query):
    result = multihop_agent.run(query)
    
    print(f"Completed in {result.total_hops} hops")
    
    for step in result.reasoning_steps:
        print(f"Step {step.step_index}: {step.sub_question}")
        print(f"  Answer: {step.answer}")
        print(f"  Confidence: {step.confidence:.2f}")
    
    print(f"Final: {result.final_answer}")
```

---

# Complete Configuration Reference

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

bm25:
  k1: 1.5
  b: 0.75
  top_k: 10
  persist: true

retrieval:
  top_k: 10
  min_similarity: 0.5

rerank:
  enabled: true
  top_k: 20
  model_name: "cross-encoder/ms-marco-MiniLM-L12-v2"

automerge:
  enabled: true
  threshold: 2
  max_parent_size: 10000

synthesis:
  max_tokens: 1024
  temperature: 0.7
  include_sources: true

critic:
  enabled: true
  min_confidence: 0.7
  max_retries: 2

agentic:
  use_planning: true
  use_tools: true
  use_strategy_memory: true
  min_confidence: 0.7
  max_retries: 2

chunking:
  enabled: true
  method: "llm"
  target_chunk_size: 1000
  size_tolerance: 0.3

summarization:
  enabled: true
  max_summary_length: 4000
  strategy: "abstractive"

context_evaluation:
  enabled: true
  min_relevance_score: 0.6
  min_coverage_score: 0.5

multihop:
  enabled: true
  max_hops: 3
  docs_per_hop: 5
  min_confidence_to_continue: 0.3

fact_verification:
  enabled: true
  min_support_confidence: 0.6
  max_claims_to_verify: 20
  generate_corrections: true

citation:
  enabled: true
  citation_style: "inline"
  generate_bibliography: true
  generate_audit_trail: true

language_detection:
  enabled: true
  method: "fast"
  min_confidence: 0.7
  use_llm_fallback: true
  fallback_language: "en"

translation:
  enabled: true
  method: "llm"
  canonical_language: "en"
  max_chars_per_llm_call: 4000
  translate_at_ingestion: true
  preserve_original: true

query:
  max_decomposed_queries: 5
  max_expansions: 12

conversation:
  enabled: true
  max_turns: 50
  ttl: 86400

pipeline:
  use_planning: true
  use_decomposition: true
  use_rewrite: true
  use_expansion: true
  use_rrf: true
  use_automerge: true
  use_rerank: true
  use_critic: true

web_search:
  enabled: false
  provider: "duckduckgo"
  max_results: 10

vlm:
  enabled: true
  model_name: "Qwen/Qwen2-VL-2B-Instruct"
  device: "auto"
  load_in_4bit: false

logging:
  level: "INFO"
  file: ""

metrics:
  enabled: true
  detailed_timing: true
```

---

# Error Reference

| Code | Error | Cause | Solution |
|------|-------|-------|----------|
| E001 | Redis connection failed | Redis not running | Start Redis: `docker-compose up -d redis` |
| E002 | LLM request timeout | Slow LLM response | Increase `ollama.timeout` |
| E003 | Embedding model load failed | Model not found | Check model name, verify disk space |
| E004 | Document parsing failed | Invalid file format | Check file is not corrupted |
| E005 | Index creation failed | Redis memory full | Clear old data or add memory |
| E006 | Vector dimension mismatch | Wrong embedding model | Match `embedding_dimension` to model |
| E007 | Translation failed | LLM unavailable | Check LLM endpoint |
| E008 | Language detection failed | Missing dependency | `pip install fast-langdetect` |
| E009 | Tool execution failed | Invalid input | Check tool input format |
| E010 | Conversation not found | Invalid ID | Verify conversation exists |
| E011 | Configuration invalid | YAML syntax error | Validate YAML format |
| E012 | Memory limit exceeded | Too many documents | Reduce batch size |
| E013 | GPU out of memory | Model too large | Enable `load_in_4bit` |
| E014 | File not found | Invalid path | Check file exists |
| E015 | Permission denied | Access rights | Check permissions |

---

# Supported Languages (Complete List)

The LanguageDetectionAgent supports 176 languages:

**Major Languages:**
- English (en), Chinese (zh), Spanish (es), French (fr), German (de)
- Japanese (ja), Korean (ko), Portuguese (pt), Russian (ru), Arabic (ar)
- Hindi (hi), Italian (it), Dutch (nl), Polish (pl), Turkish (tr)

**European Languages:**
- Bulgarian (bg), Czech (cs), Danish (da), Greek (el), Finnish (fi)
- Hungarian (hu), Norwegian (no), Romanian (ro), Slovak (sk), Swedish (sv)
- Ukrainian (uk), Croatian (hr), Serbian (sr), Slovenian (sl), Estonian (et)
- Latvian (lv), Lithuanian (lt), Catalan (ca), Basque (eu), Galician (gl)

**Asian Languages:**
- Thai (th), Vietnamese (vi), Indonesian (id), Malay (ms), Tagalog (tl)
- Bengali (bn), Tamil (ta), Telugu (te), Kannada (kn), Malayalam (ml)
- Marathi (mr), Gujarati (gu), Punjabi (pa), Urdu (ur), Nepali (ne)

**Middle Eastern Languages:**
- Hebrew (he), Persian (fa), Kurdish (ku), Pashto (ps)

**African Languages:**
- Swahili (sw), Amharic (am), Hausa (ha), Yoruba (yo), Zulu (zu)

---

*End of Radiant Agentic RAG User Manual*

**Version 1.0 | December 2024**

For support and updates, visit the project repository.
