"""
HTTP client for the GDELT DOC 2.0 and GEO 2.0 APIs.

Handles request construction, rate limiting, retries with exponential
backoff, and response parsing for the various GDELT API modes.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from radiant.agents.gdelt.config import GDELTConfig
from radiant.agents.gdelt.models import (
    GDELTArticle,
    GDELTTimelinePoint,
)

logger = logging.getLogger(__name__)

# Valid modes for the DOC 2.0 API
DOC_MODES = {
    "ArtList",
    "TimelineVol",
    "TimelineVolRaw",
    "TimelineVolInfo",
    "TimelineLang",
    "TimelineSourceCountry",
    "TimelineTone",
    "ToneChart",
    "WordCloud",
}

# Timeline modes that return time-series data
TIMELINE_MODES = {
    "TimelineVol",
    "TimelineVolRaw",
    "TimelineLang",
    "TimelineSourceCountry",
    "TimelineTone",
}


class GDELTClient:
    """
    HTTP client for the GDELT DOC 2.0 and GEO 2.0 APIs.

    Provides methods for article search, timeline queries, and
    geographic data retrieval with built-in rate limiting and
    retry logic.
    """

    def __init__(self, config: GDELTConfig) -> None:
        """
        Initialize the GDELT API client.

        Args:
            config: GDELT configuration.
        """
        self._config = config
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": config.user_agent})
        self._last_request_time: float = 0.0

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def search_articles(
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
    ) -> List[GDELTArticle]:
        """
        Search for articles using the GDELT DOC 2.0 ArtList mode.

        Args:
            query: Search keyword or phrase.
            timespan: Rolling window (e.g. "7d", "24h", "3months").
            start_datetime: Start time as ``YYYYMMDDHHmmSS``.
            end_datetime: End time as ``YYYYMMDDHHmmSS``.
            sourcecountry: FIPS 2-letter country code filter.
            sourcelang: Language filter (e.g. "English").
            domain: Domain filter (e.g. "nytimes.com").
            theme: GDELT GKG theme filter.
            max_records: Maximum articles to return (cap 250).

        Returns:
            List of ``GDELTArticle`` objects.
        """
        full_query = self._build_query_string(
            query,
            sourcecountry=sourcecountry,
            sourcelang=sourcelang,
            domain=domain,
            theme=theme,
        )

        params = self._build_params(
            full_query,
            mode="ArtList",
            fmt="json",
            timespan=timespan,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            max_records=max_records,
        )

        data = self._request(self._config.doc_api_url, params)
        return self._parse_articles(data)

    def search_articles_windowed(
        self,
        query: str,
        *,
        start_datetime: str,
        end_datetime: str,
        window_hours: Optional[int] = None,
        max_total: Optional[int] = None,
        sourcecountry: Optional[str] = None,
        sourcelang: Optional[str] = None,
        domain: Optional[str] = None,
        theme: Optional[str] = None,
    ) -> List[GDELTArticle]:
        """
        Search for articles using date-windowed pagination.

        Because the GDELT API returns at most 250 articles per request,
        this method slices the date range into windows and collects
        articles across all windows, deduplicating by URL.

        Args:
            query: Search keyword or phrase.
            start_datetime: Start time as ``YYYYMMDDHHmmSS``.
            end_datetime: End time as ``YYYYMMDDHHmmSS``.
            window_hours: Size of each window in hours.
            max_total: Maximum total articles to collect.
            sourcecountry: FIPS 2-letter country code filter.
            sourcelang: Language filter.
            domain: Domain filter.
            theme: Theme filter.

        Returns:
            Deduplicated list of ``GDELTArticle`` objects.
        """
        window_hours = window_hours or self._config.date_window_hours
        max_total = max_total or self._config.max_total_articles

        windows = self._generate_time_windows(
            start_datetime, end_datetime, window_hours
        )

        seen_urls: set[str] = set()
        all_articles: List[GDELTArticle] = []

        for win_start, win_end in windows:
            if len(all_articles) >= max_total:
                break

            articles = self.search_articles(
                query,
                start_datetime=win_start,
                end_datetime=win_end,
                sourcecountry=sourcecountry,
                sourcelang=sourcelang,
                domain=domain,
                theme=theme,
                max_records=self._config.max_records,
            )

            for article in articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    all_articles.append(article)
                    if len(all_articles) >= max_total:
                        break

            logger.debug(
                "Window %s-%s: %d articles (total: %d)",
                win_start,
                win_end,
                len(articles),
                len(all_articles),
            )

        return all_articles

    def query_timeline(
        self,
        query: str,
        mode: str = "TimelineVol",
        *,
        timespan: Optional[str] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        sourcecountry: Optional[str] = None,
        sourcelang: Optional[str] = None,
        domain: Optional[str] = None,
        theme: Optional[str] = None,
        smoothing: Optional[int] = None,
    ) -> List[GDELTTimelinePoint]:
        """
        Query a GDELT timeline mode.

        Args:
            query: Search keyword or phrase.
            mode: Timeline mode (TimelineVol, TimelineVolRaw, TimelineTone,
                  TimelineSourceCountry, TimelineLang).
            timespan: Rolling window.
            start_datetime: Start time as ``YYYYMMDDHHmmSS``.
            end_datetime: End time as ``YYYYMMDDHHmmSS``.
            sourcecountry: Country code filter.
            sourcelang: Language filter.
            domain: Domain filter.
            theme: Theme filter.
            smoothing: Smoothing window size.

        Returns:
            List of ``GDELTTimelinePoint`` objects.
        """
        if mode not in TIMELINE_MODES:
            raise ValueError(
                f"Invalid timeline mode: {mode!r}. "
                f"Must be one of: {sorted(TIMELINE_MODES)}"
            )

        full_query = self._build_query_string(
            query,
            sourcecountry=sourcecountry,
            sourcelang=sourcelang,
            domain=domain,
            theme=theme,
        )

        params = self._build_params(
            full_query,
            mode=mode,
            fmt="csv",
            timespan=timespan,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            smoothing=smoothing,
        )

        data = self._request(self._config.doc_api_url, params, expect_json=False)
        return self._parse_timeline_csv(data, mode)

    def query_tone_chart(
        self,
        query: str,
        *,
        timespan: Optional[str] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the GDELT ToneChart mode.

        Returns histogram data of article sentiment distribution.

        Args:
            query: Search keyword or phrase.
            timespan: Rolling window.
            start_datetime: Start time.
            end_datetime: End time.

        Returns:
            List of dictionaries with tone bin data.
        """
        params = self._build_params(
            query,
            mode="ToneChart",
            fmt="json",
            timespan=timespan,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )

        data = self._request(self._config.doc_api_url, params)
        if isinstance(data, dict):
            return data.get("tonechart", [])
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_query_string(
        self,
        query: str,
        *,
        sourcecountry: Optional[str] = None,
        sourcelang: Optional[str] = None,
        domain: Optional[str] = None,
        theme: Optional[str] = None,
    ) -> str:
        """Build the composite query string with in-query operators."""
        parts = [query]
        if sourcecountry:
            parts.append(f"sourcecountry:{sourcecountry}")
        if sourcelang:
            parts.append(f"sourcelang:{sourcelang}")
        if domain:
            parts.append(f"domainis:{domain}")
        if theme:
            parts.append(f"theme:{theme}")
        return " ".join(parts)

    def _build_params(
        self,
        query: str,
        mode: str,
        fmt: str,
        *,
        timespan: Optional[str] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        max_records: Optional[int] = None,
        smoothing: Optional[int] = None,
    ) -> Dict[str, str]:
        """Build URL query parameters for the API request."""
        params: Dict[str, str] = {
            "query": query,
            "mode": mode,
            "format": fmt,
        }

        if max_records is not None:
            params["maxrecords"] = str(min(max_records, 250))

        if timespan:
            params["TIMESPAN"] = timespan
        if start_datetime:
            params["STARTDATETIME"] = start_datetime
        if end_datetime:
            params["ENDDATETIME"] = end_datetime
        if smoothing is not None:
            params["TIMELINESMOOTH"] = str(smoothing)

        return params

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.monotonic() - self._last_request_time
        remaining = self._config.request_delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _request(
        self,
        url: str,
        params: Dict[str, str],
        expect_json: bool = True,
    ) -> Any:
        """
        Execute an HTTP GET request with retry logic.

        Args:
            url: API endpoint URL.
            params: Query parameters.
            expect_json: Whether to parse response as JSON.

        Returns:
            Parsed JSON data or raw response text.

        Raises:
            requests.HTTPError: If all retries are exhausted.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self._config.max_retries + 1):
            self._rate_limit()

            try:
                self._last_request_time = time.monotonic()
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self._config.request_timeout,
                )

                if response.status_code == 429:
                    delay = self._config.retry_base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "GDELT rate limited (429). Retrying in %.1fs "
                        "(attempt %d/%d)",
                        delay,
                        attempt,
                        self._config.max_retries,
                    )
                    time.sleep(delay)
                    continue

                response.raise_for_status()

                if expect_json:
                    text = response.text.strip()
                    # GDELT sometimes returns BOM-prefixed text
                    if text.startswith("\ufeff"):
                        text = text[1:]
                    if not text:
                        return {}
                    return json.loads(text)
                return response.text

            except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self._config.max_retries:
                    delay = self._config.retry_base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "GDELT request failed: %s. Retrying in %.1fs "
                        "(attempt %d/%d)",
                        exc,
                        delay,
                        attempt,
                        self._config.max_retries,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "GDELT request failed after %d attempts: %s",
                        self._config.max_retries,
                        exc,
                    )

        if last_error is not None:
            raise last_error
        raise RuntimeError("GDELT request failed with no error captured")

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_articles(self, data: Any) -> List[GDELTArticle]:
        """Parse article list from JSON response."""
        if not isinstance(data, dict):
            return []

        articles_raw = data.get("articles", [])
        if not isinstance(articles_raw, list):
            return []

        articles: List[GDELTArticle] = []
        for item in articles_raw:
            if not isinstance(item, dict):
                continue
            try:
                articles.append(
                    GDELTArticle(
                        url=str(item.get("url", "")),
                        title=str(item.get("title", "")),
                        seendate=str(item.get("seendate", "")),
                        domain=str(item.get("domain", "")),
                        language=str(item.get("language", "")),
                        sourcecountry=str(item.get("sourcecountry", "")),
                        socialimage=str(item.get("socialimage", "")),
                        url_mobile=str(item.get("url_mobile", "")),
                    )
                )
            except (TypeError, ValueError) as exc:
                logger.debug("Skipping malformed article entry: %s", exc)
                continue

        return articles

    def _parse_timeline_csv(
        self,
        raw_csv: str,
        mode: str,
    ) -> List[GDELTTimelinePoint]:
        """
        Parse timeline CSV data into structured data points.

        GDELT timeline CSV format varies by mode:
        - TimelineVol/TimelineVolRaw: date, value
        - TimelineSourceCountry/TimelineLang: date, series1, series2, ...
        - TimelineTone: date, tone_value
        """
        if not raw_csv or not raw_csv.strip():
            return []

        # Strip BOM if present
        text = raw_csv.strip()
        if text.startswith("\ufeff"):
            text = text[1:]

        points: List[GDELTTimelinePoint] = []

        try:
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
        except csv.Error as exc:
            logger.warning("Failed to parse GDELT timeline CSV: %s", exc)
            return []

        if not rows:
            return []

        # Check for header row (first cell looks like a label, not a date)
        header: Optional[List[str]] = None
        first_row = rows[0]
        if first_row and not self._looks_like_date(first_row[0]):
            header = first_row
            rows = rows[1:]

        for row in rows:
            if len(row) < 2:
                continue

            date_str = row[0].strip()

            if mode in ("TimelineSourceCountry", "TimelineLang") and header:
                # Multi-series: each column after the date is a separate series
                for col_idx in range(1, len(row)):
                    series_name = (
                        header[col_idx].strip()
                        if header and col_idx < len(header)
                        else f"series_{col_idx}"
                    )
                    value = self._safe_float(row[col_idx])
                    if value is not None:
                        points.append(
                            GDELTTimelinePoint(
                                date=date_str,
                                value=value,
                                series=series_name,
                            )
                        )
            else:
                # Single-series modes
                value = self._safe_float(row[1])
                if value is not None:
                    series_label = "volume"
                    if mode == "TimelineTone":
                        series_label = "tone"
                    elif mode == "TimelineVolRaw":
                        series_label = "raw_count"

                    points.append(
                        GDELTTimelinePoint(
                            date=date_str,
                            value=value,
                            series=series_label,
                        )
                    )

        return points

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_date(value: str) -> bool:
        """Heuristic check if a string looks like a date/timestamp."""
        stripped = value.strip()
        if not stripped:
            return False
        # GDELT dates are typically YYYYMMDDTHHMMSS or similar patterns
        return stripped[0].isdigit() and len(stripped) >= 8

    @staticmethod
    def _safe_float(value: str) -> Optional[float]:
        """Safely parse a string to float, returning None on failure."""
        try:
            return float(value.strip())
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _generate_time_windows(
        start: str,
        end: str,
        window_hours: int,
    ) -> List[Tuple[str, str]]:
        """
        Generate time windows between start and end datetime strings.

        Args:
            start: Start datetime as ``YYYYMMDDHHmmSS``.
            end: End datetime as ``YYYYMMDDHHmmSS``.
            window_hours: Size of each window in hours.

        Returns:
            List of (window_start, window_end) tuples in ``YYYYMMDDHHmmSS``
            format.
        """
        fmt = "%Y%m%d%H%M%S"
        try:
            start_dt = datetime.strptime(start, fmt).replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            # Fallback: try shorter formats
            for short_fmt in ("%Y%m%d%H%M", "%Y%m%d"):
                try:
                    start_dt = datetime.strptime(start, short_fmt).replace(
                        tzinfo=timezone.utc
                    )
                    end_dt = datetime.strptime(end, short_fmt).replace(
                        tzinfo=timezone.utc
                    )
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(
                    f"Cannot parse datetime strings: start={start!r}, end={end!r}"
                )

        if start_dt >= end_dt:
            return [(start, end)]

        delta = timedelta(hours=window_hours)
        windows: List[Tuple[str, str]] = []
        current = start_dt

        while current < end_dt:
            win_end = min(current + delta, end_dt)
            windows.append(
                (current.strftime(fmt), win_end.strftime(fmt))
            )
            current = win_end

        return windows
