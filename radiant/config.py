"""
Configuration management for Radiant Agentic RAG.

Loads configuration from YAML file with environment variable overrides.
Environment variables use the pattern: RADIANT_<SECTION>_<KEY>
Example: RADIANT_OLLAMA_CHAT_MODEL overrides ollama.chat_model
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# Default config file locations (searched in order)
DEFAULT_CONFIG_PATHS = [
    Path("./config.yaml"),
    Path("./config.yml"),
    Path("./radiant.yaml"),
    Path("./radiant.yml"),
    Path.home() / ".radiant" / "config.yaml",
    Path("/etc/radiant/config.yaml"),
]


def _get_env_override(section: str, key: str) -> Optional[str]:
    """Get environment variable override for a config key."""
    env_key = f"RADIANT_{section.upper()}_{key.upper()}"
    return os.environ.get(env_key)


def _parse_bool(value: Union[str, bool]) -> bool:
    """Parse boolean from string or bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_int(value: Union[str, int], default: int) -> int:
    """Parse integer from string or int."""
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default


def _parse_float(value: Union[str, float], default: float) -> float:
    """Parse float from string or float."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


def _get_config_value(
    data: Dict[str, Any],
    section: str,
    key: str,
    default: Any,
    parser: Optional[callable] = None,
) -> Any:
    """Get config value with environment override support."""
    # Check environment override first
    env_value = _get_env_override(section, key)
    if env_value is not None:
        if parser:
            return parser(env_value) if parser != _parse_int and parser != _parse_float else parser(env_value, default)
        return env_value

    # Get from config data
    section_data = data.get(section, {})
    if isinstance(section_data, dict):
        value = section_data.get(key, default)
    else:
        value = default

    if parser and value is not None:
        if parser in (_parse_int, _parse_float):
            return parser(value, default)
        return parser(value)

    return value


def _get_nested_config_value(
    data: Dict[str, Any],
    section: str,
    subsection: str,
    key: str,
    default: Any,
    parser: Optional[callable] = None,
) -> Any:
    """Get nested config value with environment override support."""
    # Environment override: RADIANT_SECTION_SUBSECTION_KEY
    env_key = f"RADIANT_{section.upper()}_{subsection.upper()}_{key.upper()}"
    env_value = os.environ.get(env_key)
    if env_value is not None:
        if parser:
            if parser in (_parse_int, _parse_float):
                return parser(env_value, default)
            return parser(env_value)
        return env_value

    # Get from config data
    section_data = data.get(section, {})
    if isinstance(section_data, dict):
        subsection_data = section_data.get(subsection, {})
        if isinstance(subsection_data, dict):
            value = subsection_data.get(key, default)
        else:
            value = default
    else:
        value = default

    if parser and value is not None:
        if parser in (_parse_int, _parse_float):
            return parser(value, default)
        return parser(value)

    return value


@dataclass(frozen=True)
class OllamaConfig:
    """Ollama Cloud (OpenAI-compatible) configuration."""
    openai_base_url: str
    openai_api_key: str
    chat_model: str = "qwen2.5:latest"
    timeout: int = 90
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass(frozen=True)
class VLMCaptionerConfig:
    """Vision Language Model configuration for image captioning."""

    # Whether VLM captioning is enabled
    enabled: bool = True

    # HuggingFace model name
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"

    # Device: "auto", "cuda", "cpu", "mps"
    device: str = "auto"

    # Quantization for memory efficiency
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.2

    # Cache directory for model downloads (None = default HF cache)
    cache_dir: Optional[str] = None

    # Fallback to Ollama if HuggingFace not available
    ollama_fallback_url: str = "http://localhost:11434"
    ollama_fallback_model: str = "llava"


@dataclass(frozen=True)
class LocalModelsConfig:
    """Local HuggingFace/sentence-transformers configuration."""
    embed_model_name: str = "sentence-transformers/all-MiniLM-L12-v2"
    cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L12-v2"
    device: str = "auto"
    embedding_dimension: int = 384


@dataclass(frozen=True)
class VectorIndexConfig:
    """Redis vector index configuration."""
    name: str = "radiant_vectors"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_runtime: int = 100
    distance_metric: str = "COSINE"


@dataclass(frozen=True)
class RedisConfig:
    """Redis connection and storage configuration."""
    url: str = "redis://localhost:6379/0"
    key_prefix: str = "radiant"
    doc_ns: str = "doc"
    embed_ns: str = "emb"
    meta_ns: str = "meta"
    conversation_ns: str = "conv"
    max_content_chars: int = 200_000
    vector_index: VectorIndexConfig = field(default_factory=VectorIndexConfig)


@dataclass(frozen=True)
class BM25Config:
    """BM25 index configuration."""
    # Base path for index file (without extension)
    # Stored as .json.gz (compressed JSON) for security
    index_path: str = "./data/bm25_index"
    max_documents: int = 100_000
    auto_save_threshold: int = 100
    k1: float = 1.5
    b: float = 0.75


@dataclass(frozen=True)
class IngestionConfig:
    """Ingestion and batch processing configuration."""
    # Batch size for embedding generation (larger = faster but more memory)
    embedding_batch_size: int = 32
    # Batch size for Redis pipeline operations
    redis_batch_size: int = 100
    # Enable batch processing (recommended for large corpora)
    batch_enabled: bool = True
    # Child chunk size for hierarchical storage
    child_chunk_size: int = 512
    # Child chunk overlap for hierarchical storage
    child_chunk_overlap: int = 50
    # Show progress bar during ingestion
    show_progress: bool = True


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval configuration."""
    dense_top_k: int = 10
    bm25_top_k: int = 10
    fused_top_k: int = 15
    rrf_k: int = 60
    min_similarity: float = 0.0


@dataclass(frozen=True)
class RerankConfig:
    """Reranking configuration."""
    top_k: int = 8
    max_doc_chars: int = 3000
    candidate_multiplier: int = 4
    min_candidates: int = 16


@dataclass(frozen=True)
class AutoMergeConfig:
    """Hierarchical auto-merging configuration."""
    min_children_to_merge: int = 2
    max_parent_chars: int = 50_000


@dataclass(frozen=True)
class SynthesisConfig:
    """Answer synthesis configuration."""
    max_context_docs: int = 8
    max_doc_chars: int = 4000
    include_history: bool = True
    max_history_turns: int = 5


@dataclass(frozen=True)
class CriticConfig:
    """Critic agent configuration."""
    enabled: bool = True
    max_context_docs: int = 8
    max_doc_chars: int = 1200
    retry_on_issues: bool = True  # Enable critic-driven retry by default
    max_retries: int = 2
    # Confidence threshold - below this returns "I don't know"
    confidence_threshold: float = 0.4
    # Minimum score to consider retrieval successful
    min_retrieval_confidence: float = 0.3


@dataclass(frozen=True)
class AgenticConfig:
    """Agentic behavior configuration."""
    # Enable dynamic retrieval mode selection
    dynamic_retrieval_mode: bool = True
    
    # Enable tool usage (calculator, code execution)
    tools_enabled: bool = True
    
    # Enable strategy memory for adaptive retrieval
    strategy_memory_enabled: bool = True
    
    # Path to store strategy memory (relative to data dir)
    strategy_memory_path: str = "./data/strategy_memory.json.gz"
    
    # Maximum retry attempts when critic finds issues
    max_critic_retries: int = 2
    
    # Confidence threshold for "I don't know" response
    confidence_threshold: float = 0.4
    
    # Enable query rewriting on retry
    rewrite_on_retry: bool = True
    
    # Expand retrieval on retry (fetch more documents)
    expand_retrieval_on_retry: bool = True
    
    # Retrieval expansion factor on retry
    retry_expansion_factor: float = 1.5


@dataclass(frozen=True)
class ChunkingConfig:
    """Intelligent chunking configuration."""
    
    # Enable intelligent (LLM-based) chunking
    enabled: bool = True
    
    # Use LLM for chunking decisions (vs rule-based only)
    use_llm_chunking: bool = True
    
    # Document length threshold for LLM chunking (shorter docs use rule-based)
    llm_chunk_threshold: int = 3000
    
    # Chunk size parameters
    min_chunk_size: int = 200
    max_chunk_size: int = 1500
    target_chunk_size: int = 800
    overlap_size: int = 100


@dataclass(frozen=True)
class SummarizationConfig:
    """Summarization agent configuration."""
    
    # Enable summarization agent
    enabled: bool = True
    
    # Minimum document length to trigger summarization
    min_doc_length_for_summary: int = 2000
    
    # Target summary length
    target_summary_length: int = 500
    
    # Conversation compression settings
    conversation_compress_threshold: int = 6
    conversation_preserve_recent: int = 2
    
    # Document deduplication settings
    similarity_threshold: float = 0.85
    max_cluster_size: int = 3
    
    # Maximum total context characters (triggers compression if exceeded)
    max_total_context_chars: int = 8000


@dataclass(frozen=True)
class ContextEvaluationConfig:
    """Context evaluation agent configuration."""
    
    # Enable pre-generation context evaluation
    enabled: bool = True
    
    # Use LLM for detailed evaluation (vs heuristics only)
    use_llm_evaluation: bool = True
    
    # Minimum score to consider context sufficient (0-1)
    sufficiency_threshold: float = 0.5
    
    # Minimum number of relevant documents required
    min_relevant_docs: int = 1
    
    # Maximum docs to include in evaluation
    max_docs_to_evaluate: int = 8
    
    # Maximum characters per doc for evaluation
    max_doc_chars: int = 1000
    
    # Skip generation if context evaluation recommends abort
    abort_on_poor_context: bool = False


@dataclass(frozen=True)
class MultiHopConfig:
    """Multi-hop reasoning agent configuration."""
    
    # Enable multi-hop reasoning for complex queries
    enabled: bool = True
    
    # Maximum reasoning hops
    max_hops: int = 3
    
    # Documents to retrieve per hop
    docs_per_hop: int = 5
    
    # Minimum confidence to continue chain
    min_confidence_to_continue: float = 0.3
    
    # Enable entity extraction for follow-up queries
    enable_entity_extraction: bool = True
    
    # Force multi-hop for all queries (for testing)
    force_multihop: bool = False


@dataclass(frozen=True)
class FactVerificationConfig:
    """Fact verification agent configuration."""
    
    # Enable fact verification
    enabled: bool = True
    
    # Minimum confidence to consider a claim supported
    min_support_confidence: float = 0.6
    
    # Maximum claims to verify (for efficiency)
    max_claims_to_verify: int = 20
    
    # Generate corrected answers when issues found
    generate_corrections: bool = True
    
    # Strict mode: require explicit support (vs inference)
    strict_mode: bool = False
    
    # Minimum overall score to accept answer (0-1)
    min_factuality_score: float = 0.5
    
    # Block answers that fail verification
    block_on_failure: bool = False


@dataclass(frozen=True)
class CitationConfig:
    """Citation tracking agent configuration."""
    
    # Enable citation tracking
    enabled: bool = True
    
    # Citation style: inline, footnote, academic, hyperlink, enterprise
    citation_style: str = "inline"
    
    # Minimum confidence for citations
    min_citation_confidence: float = 0.5
    
    # Maximum citations per claim
    max_citations_per_claim: int = 3
    
    # Include supporting excerpts in citations
    include_excerpts: bool = True
    
    # Maximum excerpt length
    excerpt_max_length: int = 200
    
    # Generate bibliography/references section
    generate_bibliography: bool = True
    
    # Generate audit trail
    generate_audit_trail: bool = True


@dataclass(frozen=True)
class QueryConfig:
    """Query processing configuration."""
    max_decomposed_queries: int = 5
    max_expansions: int = 12
    cache_enabled: bool = False
    cache_ttl: int = 3600


@dataclass(frozen=True)
class ConversationConfig:
    """Conversation history configuration."""
    enabled: bool = True
    max_turns: int = 50
    ttl: int = 86400
    use_history_for_retrieval: bool = True
    history_turns_for_context: int = 3


@dataclass(frozen=True)
class ParsingConfig:
    """LLM response parsing configuration."""
    max_retries: int = 2
    retry_delay: float = 0.5
    strict_json: bool = False
    log_failures: bool = True


@dataclass(frozen=True)
class UnstructuredCleaningConfig:
    """Unstructured document cleaning configuration."""
    enabled: bool = True
    bullets: bool = False
    extra_whitespace: bool = True
    dashes: bool = False
    trailing_punctuation: bool = False
    lowercase: bool = False
    preview_enabled: bool = False
    preview_max_items: int = 12
    preview_max_chars: int = 800


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = ""
    json_logging: bool = False
    quiet_third_party: bool = True  # Suppress noisy third-party library logs


@dataclass(frozen=True)
class MetricsConfig:
    """Metrics and observability configuration."""
    enabled: bool = True
    detailed_timing: bool = True
    store_history: bool = False
    history_retention: int = 100


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline feature flags."""
    use_planning: bool = True
    use_decomposition: bool = True
    use_rewrite: bool = True
    use_expansion: bool = True
    use_rrf: bool = True
    use_automerge: bool = True
    use_rerank: bool = True
    use_critic: bool = True


@dataclass(frozen=True)
class WebCrawlerConfig:
    """Web crawler configuration for URL ingestion."""

    # Crawl depth (0 = seed URLs only, 1 = seed + direct links, etc.)
    max_depth: int = 2

    # Maximum total pages to crawl per session
    max_pages: int = 100

    # Only crawl pages from the same domain as seed URLs
    same_domain_only: bool = True

    # URL patterns (regex) - URLs must match at least one include pattern
    include_patterns: List[str] = field(default_factory=list)

    # URL patterns (regex) - URLs matching any exclude pattern are skipped
    exclude_patterns: List[str] = field(default_factory=list)

    # Request timeout in seconds
    timeout: int = 30

    # Delay between requests in seconds (rate limiting)
    delay: float = 0.5

    # User agent string for requests
    user_agent: str = "AgenticRAG-Crawler/1.0"

    # Basic authentication credentials (if required)
    basic_auth_user: str = ""
    basic_auth_password: str = ""

    # SSL certificate verification
    verify_ssl: bool = True

    # Temporary directory for downloaded files (None = system temp)
    temp_dir: Optional[str] = None

    # Whether to follow redirects
    follow_redirects: bool = True

    # Maximum file size to download (bytes, 0 = unlimited)
    max_file_size: int = 50_000_000  # 50 MB

    # Respect robots.txt (future enhancement)
    respect_robots_txt: bool = True


@dataclass(frozen=True)
class WebSearchConfig:
    """Real-time web search agent configuration."""

    # Enable/disable web search during queries
    enabled: bool = False

    # Maximum number of URLs to fetch per query
    max_results: int = 5

    # Maximum pages to fetch (may be less than max_results if pages fail)
    max_pages: int = 3

    # Request timeout in seconds
    timeout: int = 15

    # User agent for requests
    user_agent: str = "AgenticRAG-WebSearch/1.0"

    # Whether to include web results in the answer synthesis
    include_in_synthesis: bool = True

    # Minimum relevance score to include web result (0.0-1.0)
    min_relevance: float = 0.3

    # Search engine to use (for future expansion)
    # Currently only "direct" is supported (direct URL fetching based on query analysis)
    search_mode: str = "direct"

    # Whether to cache web search results
    cache_enabled: bool = True

    # Cache TTL in seconds (default 1 hour)
    cache_ttl: int = 3600

    # Keywords that trigger web search (if empty, rely on planner)
    trigger_keywords: List[str] = field(default_factory=lambda: [
        "latest", "recent", "current", "today", "news",
        "update", "new", "2024", "2025", "now",
    ])

    # Domains to prefer for searches (optional)
    preferred_domains: List[str] = field(default_factory=list)

    # Domains to block
    blocked_domains: List[str] = field(default_factory=lambda: [
        "facebook.com", "twitter.com", "instagram.com",
        "tiktok.com", "pinterest.com",
    ])


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""
    ollama: OllamaConfig
    local_models: LocalModelsConfig
    redis: RedisConfig
    bm25: BM25Config
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    automerge: AutoMergeConfig
    synthesis: SynthesisConfig
    critic: CriticConfig
    agentic: AgenticConfig
    chunking: ChunkingConfig
    summarization: SummarizationConfig
    context_evaluation: ContextEvaluationConfig
    multihop: MultiHopConfig
    fact_verification: FactVerificationConfig
    citation: CitationConfig
    query: QueryConfig
    conversation: ConversationConfig
    parsing: ParsingConfig
    unstructured_cleaning: UnstructuredCleaningConfig
    logging: LoggingConfig
    metrics: MetricsConfig
    pipeline: PipelineConfig
    vlm: VLMCaptionerConfig
    web_crawler: WebCrawlerConfig
    web_search: WebSearchConfig


def find_config_file(config_path: Optional[str] = None) -> Optional[Path]:
    """Find configuration file from explicit path or default locations."""
    if config_path:
        path = Path(config_path)
        if path.exists():
            return path
        logger.warning(f"Config file not found at specified path: {config_path}")
        return None

    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            logger.info(f"Found config file: {path}")
            return path

    return None


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.error(f"Failed to load config file {path}: {e}")
        return {}


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load application configuration.

    Priority (highest to lowest):
    1. Environment variables (RADIANT_<SECTION>_<KEY>)
    2. Config file values
    3. Default values

    Args:
        config_path: Optional explicit path to config file

    Returns:
        AppConfig instance

    Raises:
        ValueError: If required configuration is missing
    """
    # Load config file if available
    config_file = find_config_file(config_path)
    data = load_yaml_config(config_file) if config_file else {}

    # Build configuration with environment overrides

    # Ollama config (base_url and api_key are required)
    ollama_base_url = _get_config_value(data, "ollama", "openai_base_url", "")
    ollama_api_key = _get_config_value(data, "ollama", "openai_api_key", "")

    # Also check legacy environment variables
    if not ollama_base_url:
        ollama_base_url = os.environ.get("OLLAMA_OPENAI_BASE_URL", "")
    if not ollama_api_key:
        ollama_api_key = os.environ.get("OLLAMA_OPENAI_API_KEY", "")

    if not ollama_base_url:
        raise ValueError(
            "Missing Ollama base URL. Set RADIANT_OLLAMA_OPENAI_BASE_URL or "
            "OLLAMA_OPENAI_BASE_URL environment variable, or set ollama.openai_base_url in config file."
        )
    if not ollama_api_key:
        raise ValueError(
            "Missing Ollama API key. Set RADIANT_OLLAMA_OPENAI_API_KEY or "
            "OLLAMA_OPENAI_API_KEY environment variable, or set ollama.openai_api_key in config file."
        )

    ollama = OllamaConfig(
        openai_base_url=ollama_base_url.rstrip("/"),
        openai_api_key=ollama_api_key,
        chat_model=_get_config_value(data, "ollama", "chat_model", "qwen2.5:latest"),
        timeout=_get_config_value(data, "ollama", "timeout", 90, _parse_int),
        max_retries=_get_config_value(data, "ollama", "max_retries", 3, _parse_int),
        retry_delay=_get_config_value(data, "ollama", "retry_delay", 1.0, _parse_float),
    )

    vlm = VLMCaptionerConfig(
        enabled=_get_config_value(data, "vlm", "enabled", True, _parse_bool),
        model_name=_get_config_value(data, "vlm", "model_name", "Qwen/Qwen2-VL-2B-Instruct"),
        device=_get_config_value(data, "vlm", "device", "auto"),
        load_in_4bit=_get_config_value(data, "vlm", "load_in_4bit", False, _parse_bool),
        load_in_8bit=_get_config_value(data, "vlm", "load_in_8bit", False, _parse_bool),
        max_new_tokens=_get_config_value(data, "vlm", "max_new_tokens", 512, _parse_int),
        temperature=_get_config_value(data, "vlm", "temperature", 0.2, _parse_float),
        cache_dir=_get_config_value(data, "vlm", "cache_dir", None) or None,
        ollama_fallback_url=_get_config_value(data, "vlm", "ollama_fallback_url", "http://localhost:11434"),
        ollama_fallback_model=_get_config_value(data, "vlm", "ollama_fallback_model", "llava"),
    )

    local_models = LocalModelsConfig(
        embed_model_name=_get_config_value(data, "local_models", "embed_model_name", "sentence-transformers/all-MiniLM-L12-v2"),
        cross_encoder_name=_get_config_value(data, "local_models", "cross_encoder_name", "cross-encoder/ms-marco-MiniLM-L12-v2"),
        device=_get_config_value(data, "local_models", "device", "auto"),
        embedding_dimension=_get_config_value(data, "local_models", "embedding_dimension", 384, _parse_int),
    )

    vector_index = VectorIndexConfig(
        name=_get_nested_config_value(data, "redis", "vector_index", "name", "radiant_vectors"),
        hnsw_m=_get_nested_config_value(data, "redis", "vector_index", "hnsw_m", 16, _parse_int),
        hnsw_ef_construction=_get_nested_config_value(data, "redis", "vector_index", "hnsw_ef_construction", 200, _parse_int),
        hnsw_ef_runtime=_get_nested_config_value(data, "redis", "vector_index", "hnsw_ef_runtime", 100, _parse_int),
        distance_metric=_get_nested_config_value(data, "redis", "vector_index", "distance_metric", "COSINE"),
    )

    redis = RedisConfig(
        url=_get_config_value(data, "redis", "url", "redis://localhost:6379/0"),
        key_prefix=_get_config_value(data, "redis", "key_prefix", "radiant"),
        doc_ns=_get_config_value(data, "redis", "doc_ns", "doc"),
        embed_ns=_get_config_value(data, "redis", "embed_ns", "emb"),
        meta_ns=_get_config_value(data, "redis", "meta_ns", "meta"),
        conversation_ns=_get_config_value(data, "redis", "conversation_ns", "conv"),
        max_content_chars=_get_config_value(data, "redis", "max_content_chars", 200_000, _parse_int),
        vector_index=vector_index,
    )

    bm25 = BM25Config(
        index_path=_get_config_value(data, "bm25", "index_path", "./data/bm25_index.pkl"),
        max_documents=_get_config_value(data, "bm25", "max_documents", 100_000, _parse_int),
        auto_save_threshold=_get_config_value(data, "bm25", "auto_save_threshold", 100, _parse_int),
        k1=_get_config_value(data, "bm25", "k1", 1.5, _parse_float),
        b=_get_config_value(data, "bm25", "b", 0.75, _parse_float),
    )

    ingestion = IngestionConfig(
        embedding_batch_size=_get_config_value(data, "ingestion", "embedding_batch_size", 32, _parse_int),
        redis_batch_size=_get_config_value(data, "ingestion", "redis_batch_size", 100, _parse_int),
        batch_enabled=_get_config_value(data, "ingestion", "batch_enabled", True, _parse_bool),
        child_chunk_size=_get_config_value(data, "ingestion", "child_chunk_size", 512, _parse_int),
        child_chunk_overlap=_get_config_value(data, "ingestion", "child_chunk_overlap", 50, _parse_int),
        show_progress=_get_config_value(data, "ingestion", "show_progress", True, _parse_bool),
    )

    retrieval = RetrievalConfig(
        dense_top_k=_get_config_value(data, "retrieval", "dense_top_k", 10, _parse_int),
        bm25_top_k=_get_config_value(data, "retrieval", "bm25_top_k", 10, _parse_int),
        fused_top_k=_get_config_value(data, "retrieval", "fused_top_k", 15, _parse_int),
        rrf_k=_get_config_value(data, "retrieval", "rrf_k", 60, _parse_int),
        min_similarity=_get_config_value(data, "retrieval", "min_similarity", 0.0, _parse_float),
    )

    rerank = RerankConfig(
        top_k=_get_config_value(data, "rerank", "top_k", 8, _parse_int),
        max_doc_chars=_get_config_value(data, "rerank", "max_doc_chars", 3000, _parse_int),
        candidate_multiplier=_get_config_value(data, "rerank", "candidate_multiplier", 4, _parse_int),
        min_candidates=_get_config_value(data, "rerank", "min_candidates", 16, _parse_int),
    )

    automerge = AutoMergeConfig(
        min_children_to_merge=_get_config_value(data, "automerge", "min_children_to_merge", 2, _parse_int),
        max_parent_chars=_get_config_value(data, "automerge", "max_parent_chars", 50_000, _parse_int),
    )

    synthesis = SynthesisConfig(
        max_context_docs=_get_config_value(data, "synthesis", "max_context_docs", 8, _parse_int),
        max_doc_chars=_get_config_value(data, "synthesis", "max_doc_chars", 4000, _parse_int),
        include_history=_get_config_value(data, "synthesis", "include_history", True, _parse_bool),
        max_history_turns=_get_config_value(data, "synthesis", "max_history_turns", 5, _parse_int),
    )

    critic = CriticConfig(
        enabled=_get_config_value(data, "critic", "enabled", True, _parse_bool),
        max_context_docs=_get_config_value(data, "critic", "max_context_docs", 8, _parse_int),
        max_doc_chars=_get_config_value(data, "critic", "max_doc_chars", 1200, _parse_int),
        retry_on_issues=_get_config_value(data, "critic", "retry_on_issues", True, _parse_bool),
        max_retries=_get_config_value(data, "critic", "max_retries", 2, _parse_int),
        confidence_threshold=_get_config_value(data, "critic", "confidence_threshold", 0.4, _parse_float),
        min_retrieval_confidence=_get_config_value(data, "critic", "min_retrieval_confidence", 0.3, _parse_float),
    )

    agentic = AgenticConfig(
        dynamic_retrieval_mode=_get_config_value(data, "agentic", "dynamic_retrieval_mode", True, _parse_bool),
        tools_enabled=_get_config_value(data, "agentic", "tools_enabled", True, _parse_bool),
        strategy_memory_enabled=_get_config_value(data, "agentic", "strategy_memory_enabled", True, _parse_bool),
        strategy_memory_path=_get_config_value(data, "agentic", "strategy_memory_path", "./data/strategy_memory.json.gz"),
        max_critic_retries=_get_config_value(data, "agentic", "max_critic_retries", 2, _parse_int),
        confidence_threshold=_get_config_value(data, "agentic", "confidence_threshold", 0.4, _parse_float),
        rewrite_on_retry=_get_config_value(data, "agentic", "rewrite_on_retry", True, _parse_bool),
        expand_retrieval_on_retry=_get_config_value(data, "agentic", "expand_retrieval_on_retry", True, _parse_bool),
        retry_expansion_factor=_get_config_value(data, "agentic", "retry_expansion_factor", 1.5, _parse_float),
    )

    chunking = ChunkingConfig(
        enabled=_get_config_value(data, "chunking", "enabled", True, _parse_bool),
        use_llm_chunking=_get_config_value(data, "chunking", "use_llm_chunking", True, _parse_bool),
        llm_chunk_threshold=_get_config_value(data, "chunking", "llm_chunk_threshold", 3000, _parse_int),
        min_chunk_size=_get_config_value(data, "chunking", "min_chunk_size", 200, _parse_int),
        max_chunk_size=_get_config_value(data, "chunking", "max_chunk_size", 1500, _parse_int),
        target_chunk_size=_get_config_value(data, "chunking", "target_chunk_size", 800, _parse_int),
        overlap_size=_get_config_value(data, "chunking", "overlap_size", 100, _parse_int),
    )

    summarization = SummarizationConfig(
        enabled=_get_config_value(data, "summarization", "enabled", True, _parse_bool),
        min_doc_length_for_summary=_get_config_value(data, "summarization", "min_doc_length_for_summary", 2000, _parse_int),
        target_summary_length=_get_config_value(data, "summarization", "target_summary_length", 500, _parse_int),
        conversation_compress_threshold=_get_config_value(data, "summarization", "conversation_compress_threshold", 6, _parse_int),
        conversation_preserve_recent=_get_config_value(data, "summarization", "conversation_preserve_recent", 2, _parse_int),
        similarity_threshold=_get_config_value(data, "summarization", "similarity_threshold", 0.85, _parse_float),
        max_cluster_size=_get_config_value(data, "summarization", "max_cluster_size", 3, _parse_int),
        max_total_context_chars=_get_config_value(data, "summarization", "max_total_context_chars", 8000, _parse_int),
    )

    context_evaluation = ContextEvaluationConfig(
        enabled=_get_config_value(data, "context_evaluation", "enabled", True, _parse_bool),
        use_llm_evaluation=_get_config_value(data, "context_evaluation", "use_llm_evaluation", True, _parse_bool),
        sufficiency_threshold=_get_config_value(data, "context_evaluation", "sufficiency_threshold", 0.5, _parse_float),
        min_relevant_docs=_get_config_value(data, "context_evaluation", "min_relevant_docs", 1, _parse_int),
        max_docs_to_evaluate=_get_config_value(data, "context_evaluation", "max_docs_to_evaluate", 8, _parse_int),
        max_doc_chars=_get_config_value(data, "context_evaluation", "max_doc_chars", 1000, _parse_int),
        abort_on_poor_context=_get_config_value(data, "context_evaluation", "abort_on_poor_context", False, _parse_bool),
    )

    multihop = MultiHopConfig(
        enabled=_get_config_value(data, "multihop", "enabled", True, _parse_bool),
        max_hops=_get_config_value(data, "multihop", "max_hops", 3, _parse_int),
        docs_per_hop=_get_config_value(data, "multihop", "docs_per_hop", 5, _parse_int),
        min_confidence_to_continue=_get_config_value(data, "multihop", "min_confidence_to_continue", 0.3, _parse_float),
        enable_entity_extraction=_get_config_value(data, "multihop", "enable_entity_extraction", True, _parse_bool),
        force_multihop=_get_config_value(data, "multihop", "force_multihop", False, _parse_bool),
    )

    fact_verification = FactVerificationConfig(
        enabled=_get_config_value(data, "fact_verification", "enabled", True, _parse_bool),
        min_support_confidence=_get_config_value(data, "fact_verification", "min_support_confidence", 0.6, _parse_float),
        max_claims_to_verify=_get_config_value(data, "fact_verification", "max_claims_to_verify", 20, _parse_int),
        generate_corrections=_get_config_value(data, "fact_verification", "generate_corrections", True, _parse_bool),
        strict_mode=_get_config_value(data, "fact_verification", "strict_mode", False, _parse_bool),
        min_factuality_score=_get_config_value(data, "fact_verification", "min_factuality_score", 0.5, _parse_float),
        block_on_failure=_get_config_value(data, "fact_verification", "block_on_failure", False, _parse_bool),
    )

    citation = CitationConfig(
        enabled=_get_config_value(data, "citation", "enabled", True, _parse_bool),
        citation_style=_get_config_value(data, "citation", "citation_style", "inline"),
        min_citation_confidence=_get_config_value(data, "citation", "min_citation_confidence", 0.5, _parse_float),
        max_citations_per_claim=_get_config_value(data, "citation", "max_citations_per_claim", 3, _parse_int),
        include_excerpts=_get_config_value(data, "citation", "include_excerpts", True, _parse_bool),
        excerpt_max_length=_get_config_value(data, "citation", "excerpt_max_length", 200, _parse_int),
        generate_bibliography=_get_config_value(data, "citation", "generate_bibliography", True, _parse_bool),
        generate_audit_trail=_get_config_value(data, "citation", "generate_audit_trail", True, _parse_bool),
    )

    query = QueryConfig(
        max_decomposed_queries=_get_config_value(data, "query", "max_decomposed_queries", 5, _parse_int),
        max_expansions=_get_config_value(data, "query", "max_expansions", 12, _parse_int),
        cache_enabled=_get_config_value(data, "query", "cache_enabled", False, _parse_bool),
        cache_ttl=_get_config_value(data, "query", "cache_ttl", 3600, _parse_int),
    )

    conversation = ConversationConfig(
        enabled=_get_config_value(data, "conversation", "enabled", True, _parse_bool),
        max_turns=_get_config_value(data, "conversation", "max_turns", 50, _parse_int),
        ttl=_get_config_value(data, "conversation", "ttl", 86400, _parse_int),
        use_history_for_retrieval=_get_config_value(data, "conversation", "use_history_for_retrieval", True, _parse_bool),
        history_turns_for_context=_get_config_value(data, "conversation", "history_turns_for_context", 3, _parse_int),
    )

    parsing = ParsingConfig(
        max_retries=_get_config_value(data, "parsing", "max_retries", 2, _parse_int),
        retry_delay=_get_config_value(data, "parsing", "retry_delay", 0.5, _parse_float),
        strict_json=_get_config_value(data, "parsing", "strict_json", False, _parse_bool),
        log_failures=_get_config_value(data, "parsing", "log_failures", True, _parse_bool),
    )

    unstructured_cleaning = UnstructuredCleaningConfig(
        enabled=_get_config_value(data, "unstructured_cleaning", "enabled", True, _parse_bool),
        bullets=_get_config_value(data, "unstructured_cleaning", "bullets", False, _parse_bool),
        extra_whitespace=_get_config_value(data, "unstructured_cleaning", "extra_whitespace", True, _parse_bool),
        dashes=_get_config_value(data, "unstructured_cleaning", "dashes", False, _parse_bool),
        trailing_punctuation=_get_config_value(data, "unstructured_cleaning", "trailing_punctuation", False, _parse_bool),
        lowercase=_get_config_value(data, "unstructured_cleaning", "lowercase", False, _parse_bool),
        preview_enabled=_get_config_value(data, "unstructured_cleaning", "preview_enabled", False, _parse_bool),
        preview_max_items=_get_config_value(data, "unstructured_cleaning", "preview_max_items", 12, _parse_int),
        preview_max_chars=_get_config_value(data, "unstructured_cleaning", "preview_max_chars", 800, _parse_int),
    )

    logging_config = LoggingConfig(
        level=_get_config_value(data, "logging", "level", "INFO"),
        format=_get_config_value(data, "logging", "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        file=_get_config_value(data, "logging", "file", ""),
        json_logging=_get_config_value(data, "logging", "json_logging", False, _parse_bool),
    )

    metrics = MetricsConfig(
        enabled=_get_config_value(data, "metrics", "enabled", True, _parse_bool),
        detailed_timing=_get_config_value(data, "metrics", "detailed_timing", True, _parse_bool),
        store_history=_get_config_value(data, "metrics", "store_history", False, _parse_bool),
        history_retention=_get_config_value(data, "metrics", "history_retention", 100, _parse_int),
    )

    pipeline = PipelineConfig(
        use_planning=_get_config_value(data, "pipeline", "use_planning", True, _parse_bool),
        use_decomposition=_get_config_value(data, "pipeline", "use_decomposition", True, _parse_bool),
        use_rewrite=_get_config_value(data, "pipeline", "use_rewrite", True, _parse_bool),
        use_expansion=_get_config_value(data, "pipeline", "use_expansion", True, _parse_bool),
        use_rrf=_get_config_value(data, "pipeline", "use_rrf", True, _parse_bool),
        use_automerge=_get_config_value(data, "pipeline", "use_automerge", True, _parse_bool),
        use_rerank=_get_config_value(data, "pipeline", "use_rerank", True, _parse_bool),
        use_critic=_get_config_value(data, "pipeline", "use_critic", True, _parse_bool),
    )

    # Web crawler configuration
    web_crawler_include = _get_config_value(data, "web_crawler", "include_patterns", [])
    web_crawler_exclude = _get_config_value(data, "web_crawler", "exclude_patterns", [])
    
    # Ensure patterns are lists
    if isinstance(web_crawler_include, str):
        web_crawler_include = [web_crawler_include] if web_crawler_include else []
    if isinstance(web_crawler_exclude, str):
        web_crawler_exclude = [web_crawler_exclude] if web_crawler_exclude else []

    web_crawler = WebCrawlerConfig(
        max_depth=_get_config_value(data, "web_crawler", "max_depth", 2, _parse_int),
        max_pages=_get_config_value(data, "web_crawler", "max_pages", 100, _parse_int),
        same_domain_only=_get_config_value(data, "web_crawler", "same_domain_only", True, _parse_bool),
        include_patterns=web_crawler_include,
        exclude_patterns=web_crawler_exclude,
        timeout=_get_config_value(data, "web_crawler", "timeout", 30, _parse_int),
        delay=_get_config_value(data, "web_crawler", "delay", 0.5, _parse_float),
        user_agent=_get_config_value(data, "web_crawler", "user_agent", "AgenticRAG-Crawler/1.0"),
        basic_auth_user=_get_config_value(data, "web_crawler", "basic_auth_user", ""),
        basic_auth_password=_get_config_value(data, "web_crawler", "basic_auth_password", ""),
        verify_ssl=_get_config_value(data, "web_crawler", "verify_ssl", True, _parse_bool),
        temp_dir=_get_config_value(data, "web_crawler", "temp_dir", None) or None,
        follow_redirects=_get_config_value(data, "web_crawler", "follow_redirects", True, _parse_bool),
        max_file_size=_get_config_value(data, "web_crawler", "max_file_size", 50_000_000, _parse_int),
        respect_robots_txt=_get_config_value(data, "web_crawler", "respect_robots_txt", True, _parse_bool),
    )

    # Web search configuration (real-time during queries)
    web_search_triggers = _get_config_value(data, "web_search", "trigger_keywords", [])
    web_search_preferred = _get_config_value(data, "web_search", "preferred_domains", [])
    web_search_blocked = _get_config_value(data, "web_search", "blocked_domains", [])
    
    # Ensure lists
    if isinstance(web_search_triggers, str):
        web_search_triggers = [web_search_triggers] if web_search_triggers else []
    if isinstance(web_search_preferred, str):
        web_search_preferred = [web_search_preferred] if web_search_preferred else []
    if isinstance(web_search_blocked, str):
        web_search_blocked = [web_search_blocked] if web_search_blocked else []

    web_search = WebSearchConfig(
        enabled=_get_config_value(data, "web_search", "enabled", False, _parse_bool),
        max_results=_get_config_value(data, "web_search", "max_results", 5, _parse_int),
        max_pages=_get_config_value(data, "web_search", "max_pages", 3, _parse_int),
        timeout=_get_config_value(data, "web_search", "timeout", 15, _parse_int),
        user_agent=_get_config_value(data, "web_search", "user_agent", "AgenticRAG-WebSearch/1.0"),
        include_in_synthesis=_get_config_value(data, "web_search", "include_in_synthesis", True, _parse_bool),
        min_relevance=_get_config_value(data, "web_search", "min_relevance", 0.3, _parse_float),
        search_mode=_get_config_value(data, "web_search", "search_mode", "direct"),
        cache_enabled=_get_config_value(data, "web_search", "cache_enabled", True, _parse_bool),
        cache_ttl=_get_config_value(data, "web_search", "cache_ttl", 3600, _parse_int),
        trigger_keywords=web_search_triggers if web_search_triggers else [
            "latest", "recent", "current", "today", "news",
            "update", "new", "2024", "2025", "now",
        ],
        preferred_domains=web_search_preferred,
        blocked_domains=web_search_blocked if web_search_blocked else [
            "facebook.com", "twitter.com", "instagram.com",
            "tiktok.com", "pinterest.com",
        ],
    )

    return AppConfig(
        ollama=ollama,
        local_models=local_models,
        redis=redis,
        bm25=bm25,
        ingestion=ingestion,
        retrieval=retrieval,
        rerank=rerank,
        automerge=automerge,
        synthesis=synthesis,
        critic=critic,
        agentic=agentic,
        chunking=chunking,
        summarization=summarization,
        context_evaluation=context_evaluation,
        multihop=multihop,
        fact_verification=fact_verification,
        citation=citation,
        query=query,
        conversation=conversation,
        parsing=parsing,
        unstructured_cleaning=unstructured_cleaning,
        logging=logging_config,
        metrics=metrics,
        pipeline=pipeline,
        vlm=vlm,
        web_crawler=web_crawler,
        web_search=web_search,
    )


def setup_logging(config: LoggingConfig) -> None:
    """Configure application logging based on config."""
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(config.format))
    handlers.append(console_handler)

    # File handler if configured
    if config.file:
        file_path = Path(config.file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.file)
        file_handler.setFormatter(logging.Formatter(config.format))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    # Quiet noisy third-party loggers if enabled
    if config.quiet_third_party:
        # These libraries produce verbose or misleading log messages
        noisy_loggers = {
            # HuggingFace - suppress retry warnings (they're expected with slow connections)
            "huggingface_hub": logging.ERROR,
            "huggingface_hub.utils._http": logging.ERROR,

            # Transformers - reduce verbosity
            "transformers": logging.WARNING,
            "transformers.modeling_utils": logging.WARNING,

            # Accelerate - suppress memory allocation info
            "accelerate": logging.WARNING,
            "accelerate.utils.modeling": logging.WARNING,

            # Unstructured - suppress misleading "text extraction failed" (it's expected for OCR)
            "unstructured": logging.WARNING,

            # pikepdf - suppress initialization messages
            "pikepdf": logging.WARNING,
            "pikepdf._core": logging.WARNING,

            # Other common noisy libraries
            "urllib3": logging.WARNING,
            "httpx": logging.WARNING,
            "httpcore": logging.WARNING,
            "filelock": logging.WARNING,
            "PIL": logging.WARNING,
            "torch": logging.WARNING,
            "sentence_transformers": logging.WARNING,
        }

        for logger_name, level in noisy_loggers.items():
            logging.getLogger(logger_name).setLevel(level)


def config_to_dict(config: AppConfig) -> Dict[str, Any]:
    """Convert AppConfig to dictionary (for serialization/logging)."""
    from dataclasses import asdict

    result = asdict(config)
    # Redact sensitive values
    if "ollama" in result and "openai_api_key" in result["ollama"]:
        result["ollama"]["openai_api_key"] = "***REDACTED***"
    return result
