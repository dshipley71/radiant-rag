"""
GDELT Article Download Agent.

Downloads news articles from the GDELT DOC 2.0 API and prepares
them for ingestion by the Radiant RAG application. Supports filtering
by keyword, date range, country, language, and domain, as well as
date-windowed pagination for collecting large result sets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from radiant.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    BaseAgent,
)
from radiant.agents.gdelt.client import GDELTClient
from radiant.agents.gdelt.config import GDELTConfig
from radiant.agents.gdelt.models import GDELTArticleSearchResult

logger = logging.getLogger(__name__)


class GDELTDownloadAgent(BaseAgent):
    """
    Agent that downloads news articles from the GDELT DOC 2.0 API.

    The agent searches GDELT for articles matching a query and
    optional filters, then persists the results to disk as JSON
    for subsequent ingestion into the Radiant RAG pipeline.

    Supports:
    - Keyword / phrase search
    - Date range filtering with automatic date-windowed pagination
    - Country, language, domain, and theme filters
    - Deduplication by article URL
    - JSON export for downstream ingestion
    """

    def __init__(
        self,
        config: GDELTConfig,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the GDELT download agent.

        Args:
            config: GDELT configuration.
            enabled: Whether the agent is enabled.
        """
        super().__init__(enabled=enabled)
        self._config = config
        self._client = GDELTClient(config)

    @property
    def name(self) -> str:
        return "GDELTDownloadAgent"

    @property
    def category(self) -> AgentCategory:
        return AgentCategory.RETRIEVAL

    @property
    def description(self) -> str:
        return (
            "Downloads news articles from the GDELT DOC 2.0 API "
            "for ingestion into the Radiant RAG pipeline"
        )

    def _execute(
        self,
        query: str,
        *,
        timespan: Optional[str] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        sourcecountry: Optional[str] = None,
        sourcelang: Optional[str] = None,
        domain: Optional[str] = None,
        theme: Optional[str] = None,
        max_records: Optional[int] = None,
        save_to_disk: bool = True,
        output_filename: Optional[str] = None,
        **kwargs: Any,
    ) -> GDELTArticleSearchResult:
        """
        Execute article download from GDELT.

        Args:
            query: Search keyword or phrase.
            timespan: Rolling window (e.g. ``"7d"``, ``"24h"``).
            start_datetime: Start time as ``YYYYMMDDHHmmSS``.
            end_datetime: End time as ``YYYYMMDDHHmmSS``.
            sourcecountry: FIPS 2-letter country code filter.
            sourcelang: Language filter (e.g. ``"English"``).
            domain: Domain filter (e.g. ``"nytimes.com"``).
            theme: GDELT GKG theme filter.
            max_records: Maximum articles to return (per window).
            save_to_disk: Whether to save results to JSON.
            output_filename: Custom output filename (without extension).

        Returns:
            ``GDELTArticleSearchResult`` with the collected articles.
        """
        self.logger.info(f"Starting GDELT article download for query: {query!r}")

        # Decide between windowed vs. simple search
        use_windowed = (
            self._config.enable_date_windowing
            and start_datetime is not None
            and end_datetime is not None
        )

        if use_windowed:
            articles = self._client.search_articles_windowed(
                query,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                sourcecountry=sourcecountry,
                sourcelang=sourcelang,
                domain=domain,
                theme=theme,
                max_total=max_records or self._config.max_total_articles,
            )
            date_range = f"{start_datetime} - {end_datetime}"
        else:
            articles = self._client.search_articles(
                query,
                timespan=timespan or self._config.default_timespan,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                sourcecountry=sourcecountry,
                sourcelang=sourcelang,
                domain=domain,
                theme=theme,
                max_records=max_records or self._config.max_records,
            )
            date_range = timespan or self._config.default_timespan

        result = GDELTArticleSearchResult(
            query=query,
            total_articles=len(articles),
            articles=articles,
            date_range=date_range,
        )

        self.logger.info(
            f"Downloaded {len(articles)} articles for query: {query!r}"
        )

        if save_to_disk and articles:
            self._save_results(result, output_filename)

        return result

    def _save_results(
        self,
        result: GDELTArticleSearchResult,
        output_filename: Optional[str] = None,
    ) -> str:
        """
        Save search results to a JSON file.

        Args:
            result: The search result to save.
            output_filename: Custom filename (without extension).

        Returns:
            Absolute path to the saved file.
        """
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_filename:
            # Sanitize to prevent path traversal
            filename = f"{Path(output_filename).name}.json"
        else:
            # Generate a descriptive filename
            safe_query = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in result.query[:50]
            ).strip("_")
            filename = f"gdelt_articles_{safe_query}_{len(result.articles)}.json"

        filepath = output_dir / filename
        # Ensure resolved path stays within output directory
        if not filepath.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError(f"Output path escapes output directory: {filepath}")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(result.articles)} articles to {filepath}")
        return str(filepath.resolve())

    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[GDELTArticleSearchResult]:
        """Return empty result on error."""
        self.logger.warning(f"GDELT download failed: {error}")
        query = kwargs.get("query", "")
        return GDELTArticleSearchResult(
            query=query,
            total_articles=0,
            articles=[],
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
