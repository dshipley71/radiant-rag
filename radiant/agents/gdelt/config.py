"""
Configuration for the GDELT news agents.

Provides dataclass-based configuration for GDELT API access,
article downloading, timeline analysis, and graph construction.
All settings can be overridden via environment variables using
the pattern RADIANT_GDELT_<KEY>.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class GDELTConfig:
    """
    Configuration for GDELT agents.

    Controls API access, rate limiting, output directories,
    and agent-specific parameters.
    """

    # --- API settings ---

    # GDELT DOC 2.0 API base URL
    doc_api_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"

    # GDELT GEO 2.0 API base URL
    geo_api_url: str = "https://api.gdeltproject.org/api/v2/geo/geo"

    # Default response format (json, csv, html)
    default_format: str = "json"

    # Default maximum records per request (GDELT max is 250)
    max_records: int = 250

    # Default timespan for queries (e.g., "7d", "24h", "3months")
    default_timespan: str = "7d"

    # --- Rate limiting ---

    # Delay between API requests in seconds
    request_delay: float = 1.0

    # Maximum retry attempts for failed requests
    max_retries: int = 3

    # Base delay for exponential backoff in seconds
    retry_base_delay: float = 2.0

    # Request timeout in seconds
    request_timeout: int = 30

    # --- Download agent settings ---

    # Output directory for downloaded article data
    output_dir: str = "./data/gdelt"

    # Enable date-windowing for collecting more than 250 articles
    enable_date_windowing: bool = True

    # Window size in hours for date-windowed collection
    date_window_hours: int = 12

    # Maximum total articles to collect across all windows
    max_total_articles: int = 1000

    # --- Timeline agent settings ---

    # Output directory for generated plots
    plot_output_dir: str = "./data/gdelt/plots"

    # Default smoothing window for timeline data
    default_smoothing: int = 5

    # Default plot dimensions (width, height) in inches
    plot_width: float = 14.0
    plot_height: float = 7.0

    # Default plot DPI
    plot_dpi: int = 150

    # --- Graph agent settings ---

    # Output directory for generated graph plots
    graph_output_dir: str = "./data/gdelt/graphs"

    # Maximum nodes in graph visualization
    graph_max_nodes: int = 200

    # Minimum edge weight threshold for inclusion
    graph_min_edge_weight: float = 1.0

    # Graph layout algorithm: "spring", "kamada_kawai", "circular", "shell"
    graph_layout: str = "spring"

    # Co-occurrence window in hours for building article connections
    co_occurrence_window_hours: int = 24

    # User agent string for HTTP requests
    user_agent: str = "RadiantRAG-GDELT/1.0"


def load_gdelt_config(data: Optional[Dict[str, Any]] = None) -> GDELTConfig:
    """
    Load GDELT configuration from a dictionary with environment
    variable overrides.

    Environment variables use the pattern RADIANT_GDELT_<KEY>.

    Args:
        data: Optional dictionary (typically from the ``gdelt`` section
              of the main ``config.yaml``).

    Returns:
        Populated ``GDELTConfig`` instance.
    """
    if data is None:
        data = {}

    gdelt_data = data.get("gdelt", data) if isinstance(data, dict) else {}

    def _val(key: str, default: Any) -> Any:
        env_key = f"RADIANT_GDELT_{key.upper()}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            # Coerce to the type of the default value
            if isinstance(default, bool):
                return env_value.strip().lower() in ("1", "true", "yes", "y", "on")
            if isinstance(default, int):
                try:
                    return int(env_value)
                except ValueError:
                    return default
            if isinstance(default, float):
                try:
                    return float(env_value)
                except ValueError:
                    return default
            return env_value
        return gdelt_data.get(key, default)

    return GDELTConfig(
        doc_api_url=_val("doc_api_url", GDELTConfig.doc_api_url),
        geo_api_url=_val("geo_api_url", GDELTConfig.geo_api_url),
        default_format=_val("default_format", GDELTConfig.default_format),
        max_records=_val("max_records", GDELTConfig.max_records),
        default_timespan=_val("default_timespan", GDELTConfig.default_timespan),
        request_delay=_val("request_delay", GDELTConfig.request_delay),
        max_retries=_val("max_retries", GDELTConfig.max_retries),
        retry_base_delay=_val("retry_base_delay", GDELTConfig.retry_base_delay),
        request_timeout=_val("request_timeout", GDELTConfig.request_timeout),
        output_dir=_val("output_dir", GDELTConfig.output_dir),
        enable_date_windowing=_val("enable_date_windowing", GDELTConfig.enable_date_windowing),
        date_window_hours=_val("date_window_hours", GDELTConfig.date_window_hours),
        max_total_articles=_val("max_total_articles", GDELTConfig.max_total_articles),
        plot_output_dir=_val("plot_output_dir", GDELTConfig.plot_output_dir),
        default_smoothing=_val("default_smoothing", GDELTConfig.default_smoothing),
        plot_width=_val("plot_width", GDELTConfig.plot_width),
        plot_height=_val("plot_height", GDELTConfig.plot_height),
        plot_dpi=_val("plot_dpi", GDELTConfig.plot_dpi),
        graph_output_dir=_val("graph_output_dir", GDELTConfig.graph_output_dir),
        graph_max_nodes=_val("graph_max_nodes", GDELTConfig.graph_max_nodes),
        graph_min_edge_weight=_val("graph_min_edge_weight", GDELTConfig.graph_min_edge_weight),
        graph_layout=_val("graph_layout", GDELTConfig.graph_layout),
        co_occurrence_window_hours=_val(
            "co_occurrence_window_hours", GDELTConfig.co_occurrence_window_hours
        ),
        user_agent=_val("user_agent", GDELTConfig.user_agent),
    )
