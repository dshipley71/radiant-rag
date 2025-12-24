"""
Document ingestion package.

Provides:
    - DocumentProcessor: Main document processing class
    - IngestedChunk: Chunk data class
    - ChunkSplitter: Text chunking utility
    - WebCrawler: URL crawling
    - ImageCaptioner: VLM image captioning
"""

from radiant.ingestion.processor import (
    DocumentProcessor,
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
