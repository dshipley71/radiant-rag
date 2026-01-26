# 🔆 Radiant RAG

**Agentic Retrieval-Augmented Generation Platform**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.5.0-orange.svg)](CHANGES_SUMMARY.md)

---

## Executive Summary

Radiant RAG is a production-grade, multi-agent retrieval-augmented generation system designed for enterprise deployments. It combines dense vector search, BM25 keyword retrieval, and optional web search through a sophisticated 8-phase processing pipeline with 20+ specialized agents.

### Key Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Agents** | 20+ | Specialized processing agents |
| **Pipeline Phases** | 8 | End-to-end query processing stages |
| **Storage Backends** | 3 | Redis, ChromaDB, PostgreSQL+pgvector |
| **Languages** | 176 | Supported via detection & translation |
| **Latency Reduction** | Up to 93% | Through parallel execution & caching |
| **Quantization Speedup** | 10-20× | Binary/Int8 vector quantization |

### Core Capabilities

- **Hybrid Search** — Combines dense embeddings (HNSW) + BM25 keywords + optional web search via RRF fusion
- **Multi-Hop Reasoning** — Decomposes complex queries into iterative retrieval steps
- **Automatic Fact Verification** — Validates generated answers against source documents
- **Citation Tracking** — Links claims to sources with full audit trail
- **Binary/Int8 Quantization** — 10-20× faster search with 3.5× memory reduction
- **Parallel Execution** — Concurrent retrieval and verification for 50%+ latency reduction
- **176-Language Support** — Automatic detection and translation
- **Hierarchical Chunking** — Parent-child document relationships for context preservation
- **Cross-Encoder Reranking** — High-precision relevance scoring
- **Critic-Driven QA** — Iterative refinement with confidence scoring
- **Strategy Memory** — Adaptive behavior based on past query outcomes
- **Prometheus/OpenTelemetry Metrics** — Production observability

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| **LLM & Embeddings** | Ollama, SentenceTransformers, CrossEncoder |
| **Storage** | Redis+RediSearch, ChromaDB, PostgreSQL+pgvector |
| **Document Processing** | Unstructured, AST Parsing, VLM Captioning |
| **Interfaces** | CLI, TUI (Textual), Python API |

---

### Phase Details

| Phase | Agent(s) | Function |
|-------|----------|----------|
| **① Plan** | PlanningAgent | Analyze query complexity, select retrieval strategy, identify tools |
| **② Tools** | Calculator, CodeExecutor | Execute computational tools if requested |
| **③ Query** | Decompose, Rewrite, Expand | Transform query for optimal retrieval |
| **④ Retrieve** | Dense, BM25, WebSearch | Parallel multi-modal document retrieval |
| **⑤ Post-Retrieve** | RRF, AutoMerge, Rerank, ContextEval | Fuse, merge, and rank results |
| **⑥ Generate** | MultiHop, Synthesis | Iterative reasoning and answer generation |
| **⑦ Critique** | CriticAgent | Evaluate answer quality, trigger retries |
| **⑧ Verify** | FactVerification, Citation | Validate claims and add source references |
