"""
Document ingestion for Radiant Agentic RAG.

Provides document parsing, cleaning, chunking, and image captioning
using the Unstructured library and local VLM models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from radiant.config import UnstructuredCleaningConfig

if TYPE_CHECKING:
    from radiant.ingestion.image_captioner import ImageCaptioner

logger = logging.getLogger(__name__)

# Try to import unstructured - it's optional
try:
    from unstructured.cleaners.core import clean as unstructured_clean
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning(
        "Unstructured library not available. "
        "Install with: pip install unstructured"
    )

# Image extensions for captioning
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}


@dataclass(frozen=True)
class IngestedChunk:
    """A single chunk from a parsed document."""

    content: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class CleaningOptions:
    """Options for text cleaning."""

    enabled: bool = True
    bullets: bool = False
    extra_whitespace: bool = True
    dashes: bool = False
    trailing_punctuation: bool = False
    lowercase: bool = False

    @staticmethod
    def from_config(config: UnstructuredCleaningConfig) -> "CleaningOptions":
        """Create from configuration."""
        return CleaningOptions(
            enabled=config.enabled,
            bullets=config.bullets,
            extra_whitespace=config.extra_whitespace,
            dashes=config.dashes,
            trailing_punctuation=config.trailing_punctuation,
            lowercase=config.lowercase,
        )


@dataclass(frozen=True)
class CleaningPreview:
    """Preview sample of cleaning results."""

    source_path: str
    chunk_index: int
    before: str
    after: str
    cleaning_flags: Dict[str, bool]


def iter_input_files(paths: Sequence[str]) -> List[Path]:
    """
    Iterate over input paths, expanding directories.

    Args:
        paths: List of file or directory paths

    Returns:
        List of file paths
    """
    out: List[Path] = []

    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for child in pp.rglob("*"):
                if child.is_file() and not child.name.startswith("."):
                    out.append(child)
        elif pp.is_file():
            out.append(pp)
        else:
            logger.warning(f"Path does not exist: {p}")

    return out


def _apply_cleaning(text: str, opts: CleaningOptions) -> str:
    """
    Apply text cleaning using Unstructured library.

    Args:
        text: Raw text to clean
        opts: Cleaning options

    Returns:
        Cleaned text
    """
    if not opts.enabled:
        return text

    if not UNSTRUCTURED_AVAILABLE:
        # Basic fallback cleaning
        if opts.extra_whitespace:
            import re
            text = re.sub(r"\s+", " ", text)
        if opts.lowercase:
            text = text.lower()
        return text.strip()

    return unstructured_clean(
        text,
        bullets=opts.bullets,
        extra_whitespace=opts.extra_whitespace,
        dashes=opts.dashes,
        trailing_punctuation=opts.trailing_punctuation,
        lowercase=opts.lowercase,
    )


def parse_image_with_caption(
    file_path: str,
    captioner: Optional["ImageCaptioner"] = None,
    cleaning: Optional[CleaningOptions] = None,
) -> List[IngestedChunk]:
    """
    Parse an image file by generating a VLM caption.

    Args:
        file_path: Path to image file
        captioner: ImageCaptioner instance for generating captions
        cleaning: Text cleaning options

    Returns:
        List containing a single IngestedChunk with the image caption
    """
    path = Path(file_path)
    cleaning = cleaning or CleaningOptions()

    # Generate caption using VLM
    caption = None
    if captioner is not None:
        caption = captioner.caption_image(file_path)

    if caption:
        # Apply cleaning to caption
        content = _apply_cleaning(caption, cleaning)
        logger.info(f"Generated VLM caption for {path.name}: {content[:100]}...")
    else:
        # Fallback: just describe as an image
        content = f"[Image: {path.name}]"
        logger.warning(f"Could not generate caption for {path.name}, using placeholder")

    return [
        IngestedChunk(
            content=content,
            meta={
                "source_path": file_path,
                "element_type": "Image",
                "has_vlm_caption": caption is not None,
                "file_type": path.suffix.lower(),
            }
        )
    ]


def _is_image_file(file_path: str) -> bool:
    """Check if file is an image based on extension."""
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def parse_document(
    file_path: str,
    strategy: Optional[str] = None,
    cleaning: Optional[CleaningOptions] = None,
    preview_sink: Optional[List[CleaningPreview]] = None,
    preview_max_items: int = 12,
    preview_max_chars: int = 800,
) -> List[IngestedChunk]:
    """
    Parse a document into chunks.

    Uses Unstructured's partition function with configurable strategy.

    Args:
        file_path: Path to document file
        strategy: Partition strategy ("auto", "fast", "hi_res", "ocr_only")
                  Default is "auto" which selects based on file type
        cleaning: Text cleaning options
        preview_sink: Optional list to collect cleaning preview samples
        preview_max_items: Maximum preview samples to collect
        preview_max_chars: Maximum characters per preview sample

    Returns:
        List of IngestedChunk objects
    """
    if not UNSTRUCTURED_AVAILABLE:
        raise RuntimeError(
            "Unstructured library required for document parsing. "
            "Install with: pip install unstructured"
        )

    cleaning = cleaning or CleaningOptions()

    # Determine partition strategy based on file type
    partition_kwargs: Dict[str, Any] = {"filename": file_path}

    # Valid strategies: auto, fast, hi_res, ocr_only
    file_ext = Path(file_path).suffix.lower()

    if strategy and strategy in ("auto", "fast", "hi_res", "ocr_only"):
        partition_kwargs["strategy"] = strategy
    elif file_ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"):
        # For images, try OCR but fall back to auto if it fails
        partition_kwargs["strategy"] = "auto"
    elif file_ext == ".pdf":
        # For PDFs, try fast first (most compatible)
        partition_kwargs["strategy"] = "fast"
    # else: use default "auto" strategy

    try:
        elements = partition(**partition_kwargs)

        # If PDF returned no elements, try with ocr_only strategy
        if not elements and file_ext == ".pdf":
            logger.debug(f"PDF has no extractable text, using OCR for {file_path}")
            partition_kwargs["strategy"] = "ocr_only"
            elements = partition(**partition_kwargs)

    except Exception as e:
        # If strategy failed, try with "auto" as fallback
        if partition_kwargs.get("strategy") and partition_kwargs["strategy"] != "auto":
            logger.warning(f"Strategy '{partition_kwargs['strategy']}' failed for {file_path}, trying 'auto': {e}")
            partition_kwargs["strategy"] = "auto"
            try:
                elements = partition(**partition_kwargs)
            except Exception as e2:
                logger.error(f"Failed to parse {file_path} with auto strategy: {e2}")
                raise
        else:
            logger.error(f"Failed to parse {file_path}: {e}")
            raise

    chunks: List[IngestedChunk] = []

    for el in elements:
        text = getattr(el, "text", None)
        if not text:
            continue

        before = str(text).strip()
        after = _apply_cleaning(before, cleaning).strip()

        if not after:
            continue

        # Extract metadata
        meta: Dict[str, Any] = {"source_path": file_path}
        if hasattr(el, "metadata") and el.metadata is not None:
            if hasattr(el.metadata, "to_dict"):
                md = el.metadata.to_dict()
            else:
                md = dict(el.metadata) if isinstance(el.metadata, dict) else {}
            meta.update(md)

        # Record cleaning flags in metadata
        meta["cleaning"] = {
            "enabled": cleaning.enabled,
            "bullets": cleaning.bullets,
            "extra_whitespace": cleaning.extra_whitespace,
            "dashes": cleaning.dashes,
            "trailing_punctuation": cleaning.trailing_punctuation,
            "lowercase": cleaning.lowercase,
        }

        chunk_index = len(chunks)

        # Capture preview sample if requested
        if preview_sink is not None and len(preview_sink) < preview_max_items:
            preview_sink.append(
                CleaningPreview(
                    source_path=file_path,
                    chunk_index=chunk_index,
                    before=before[:preview_max_chars] + ("…" if len(before) > preview_max_chars else ""),
                    after=after[:preview_max_chars] + ("…" if len(after) > preview_max_chars else ""),
                    cleaning_flags=meta["cleaning"],
                )
            )

        chunks.append(IngestedChunk(content=after, meta=meta))

    logger.debug(f"Parsed {file_path}: {len(chunks)} chunks")
    return chunks


def parse_text_file(
    file_path: str,
    cleaning: Optional[CleaningOptions] = None,
) -> List[IngestedChunk]:
    """
    Simple text file parsing (without Unstructured).

    Args:
        file_path: Path to text file
        cleaning: Text cleaning options

    Returns:
        List containing single IngestedChunk
    """
    cleaning = cleaning or CleaningOptions()

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    cleaned = _apply_cleaning(content, cleaning).strip()

    if not cleaned:
        return []

    return [
        IngestedChunk(
            content=cleaned,
            meta={
                "source_path": file_path,
                "cleaning": {
                    "enabled": cleaning.enabled,
                    "bullets": cleaning.bullets,
                    "extra_whitespace": cleaning.extra_whitespace,
                    "dashes": cleaning.dashes,
                    "trailing_punctuation": cleaning.trailing_punctuation,
                    "lowercase": cleaning.lowercase,
                },
            },
        )
    ]


class ChunkSplitter:
    """
    Split large chunks into smaller pieces.

    Useful for ensuring chunks fit within embedding model limits
    or for creating child documents for hierarchical retrieval.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = " ",
    ) -> None:
        """
        Initialize chunk splitter.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            separator: Text separator for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of chunk strings
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        prev_start = -1

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at a separator
            if end < len(text):
                # Look for separator within the chunk
                sep_pos = text.rfind(self.separator, start, end)
                if sep_pos > start:
                    end = sep_pos + len(self.separator)

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Calculate next start with overlap
            next_start = end - self.chunk_overlap

            # Avoid infinite loop - ensure we make progress
            if next_start <= prev_start:
                next_start = end

            prev_start = start
            start = next_start

        return chunks

    def split_chunk(
        self,
        chunk: IngestedChunk,
    ) -> List[IngestedChunk]:
        """
        Split an IngestedChunk into smaller chunks.

        Args:
            chunk: Chunk to split

        Returns:
            List of smaller chunks
        """
        texts = self.split(chunk.content)

        return [
            IngestedChunk(
                content=text,
                meta={**chunk.meta, "split_index": i, "split_total": len(texts)},
            )
            for i, text in enumerate(texts)
        ]


class DocumentProcessor:
    """
    High-level document processing pipeline.

    Combines parsing, cleaning, splitting, and image captioning.
    """

    def __init__(
        self,
        cleaning_config: UnstructuredCleaningConfig,
        strategy: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        image_captioner: Optional["ImageCaptioner"] = None,
    ) -> None:
        """
        Initialize document processor.

        Args:
            cleaning_config: Cleaning configuration
            strategy: Partition strategy ("auto", "fast", "hi_res", "ocr_only")
            chunk_size: Target chunk size
            chunk_overlap: Chunk overlap
            image_captioner: Optional VLM captioner for image files
        """
        self._cleaning = CleaningOptions.from_config(cleaning_config)
        self._strategy = strategy
        self._splitter = ChunkSplitter(chunk_size, chunk_overlap)
        self._preview_config = cleaning_config
        self._captioner = image_captioner

    def process_file(
        self,
        file_path: str,
        split_large_chunks: bool = False,
        max_chunk_size: int = 2000,
        preview_sink: Optional[List[CleaningPreview]] = None,
    ) -> List[IngestedChunk]:
        """
        Process a single file.

        Args:
            file_path: Path to file
            split_large_chunks: Whether to split large chunks
            max_chunk_size: Maximum chunk size before splitting
            preview_sink: Optional list for cleaning previews

        Returns:
            List of processed chunks
        """
        # Determine parsing method
        path = Path(file_path)

        if path.suffix.lower() in (".txt", ".md", ".rst"):
            # Simple text files
            chunks = parse_text_file(file_path, self._cleaning)
        elif _is_image_file(file_path):
            # Image files - use VLM captioning
            chunks = parse_image_with_caption(
                file_path,
                captioner=self._captioner,
                cleaning=self._cleaning,
            )
        elif UNSTRUCTURED_AVAILABLE:
            # Use Unstructured for complex documents
            chunks = parse_document(
                file_path,
                strategy=self._strategy,
                cleaning=self._cleaning,
                preview_sink=preview_sink,
                preview_max_items=self._preview_config.preview_max_items,
                preview_max_chars=self._preview_config.preview_max_chars,
            )
        else:
            # Fallback to text parsing
            chunks = parse_text_file(file_path, self._cleaning)

        # Optionally split large chunks
        if split_large_chunks:
            result: List[IngestedChunk] = []
            for chunk in chunks:
                if len(chunk.content) > max_chunk_size:
                    result.extend(self._splitter.split_chunk(chunk))
                else:
                    result.append(chunk)
            return result

        return chunks

    def process_paths(
        self,
        paths: Sequence[str],
        split_large_chunks: bool = False,
        max_chunk_size: int = 2000,
    ) -> Dict[str, List[IngestedChunk]]:
        """
        Process multiple paths.

        Args:
            paths: File or directory paths
            split_large_chunks: Whether to split large chunks
            max_chunk_size: Maximum chunk size before splitting

        Returns:
            Dictionary mapping file paths to their chunks
        """
        files = iter_input_files(paths)
        results: Dict[str, List[IngestedChunk]] = {}
        preview_sink: List[CleaningPreview] = []

        for fp in files:
            file_path = str(fp)
            try:
                chunks = self.process_file(
                    file_path,
                    split_large_chunks=split_large_chunks,
                    max_chunk_size=max_chunk_size,
                    preview_sink=preview_sink if self._preview_config.preview_enabled else None,
                )
                results[file_path] = chunks
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results[file_path] = []

        # Log preview samples if enabled
        if self._preview_config.preview_enabled and preview_sink:
            logger.info(f"Cleaning preview ({len(preview_sink)} samples):")
            for sample in preview_sink[:5]:  # Log first 5
                logger.info(
                    f"  {sample.source_path}[{sample.chunk_index}]: "
                    f"'{sample.before[:50]}...' -> '{sample.after[:50]}...'"
                )

        return results
