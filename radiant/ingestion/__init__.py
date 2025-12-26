"""
Document ingestion package.

Provides:
    - DocumentProcessor: Main document processing class
    - IntelligentDocumentProcessor: Enhanced processor with LLM-based chunking
    - IngestedChunk: Chunk data class
    - ChunkSplitter: Text chunking utility
    - WebCrawler: URL crawling
    - ImageCaptioner: VLM image captioning
"""

from radiant.ingestion.processor import (
    DocumentProcessor,
    IntelligentDocumentProcessor,
    IngestedChunk,
    ChunkSplitter,
    CleaningOptions,
    CleaningPreview,
)
from radiant.ingestion.web_crawler import WebCrawler, CrawlResult
from radiant.ingestion.image_captioner import ImageCaptioner, VLMConfig, create_captioner

__all__ = [
    # Processor
    "DocumentProcessor",
    "IntelligentDocumentProcessor",
    "IngestedChunk",
    "ChunkSplitter",
    "CleaningOptions",
    "CleaningPreview",
    # Web crawler
    "WebCrawler",
    "CrawlResult",
    # Image captioner
    "ImageCaptioner",
    "VLMConfig",
    "create_captioner",
]
