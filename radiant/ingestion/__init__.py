"""
Document ingestion package.

Provides:
    - DocumentProcessor: Main document processing class
    - IntelligentDocumentProcessor: Enhanced processor with LLM-based chunking
    - TranslatingDocumentProcessor: Processor with language detection and translation
    - IngestedChunk: Chunk data class
    - ChunkSplitter: Text chunking utility
    - WebCrawler: URL crawling
    - GitHubCrawler: GitHub repository crawling
    - ImageCaptioner: VLM image captioning
"""

from radiant.ingestion.processor import (
    DocumentProcessor,
    IntelligentDocumentProcessor,
    TranslatingDocumentProcessor,
    IngestedChunk,
    ChunkSplitter,
    CleaningOptions,
    CleaningPreview,
)
from radiant.ingestion.web_crawler import WebCrawler, CrawlResult
from radiant.ingestion.github_crawler import (
    GitHubCrawler,
    GitHubCrawlResult,
    GitHubFile,
    GitHubRepo,
    crawl_github_repo,
)
from radiant.ingestion.image_captioner import ImageCaptioner, VLMConfig, create_captioner

__all__ = [
    # Processor
    "DocumentProcessor",
    "IntelligentDocumentProcessor",
    "TranslatingDocumentProcessor",
    "IngestedChunk",
    "ChunkSplitter",
    "CleaningOptions",
    "CleaningPreview",
    # Web crawler
    "WebCrawler",
    "CrawlResult",
    # GitHub crawler
    "GitHubCrawler",
    "GitHubCrawlResult",
    "GitHubFile",
    "GitHubRepo",
    "crawl_github_repo",
    # Image captioner
    "ImageCaptioner",
    "VLMConfig",
    "create_captioner",
]
