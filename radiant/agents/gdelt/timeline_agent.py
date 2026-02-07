"""
GDELT Timeline and Trend Plotting Agent.

Queries the GDELT DOC 2.0 API for timeline data (volume, tone,
source country breakdowns, language breakdowns) and generates
publication-quality plots for analysis.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from radiant.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    BaseAgent,
)
from radiant.agents.gdelt.client import GDELTClient, TIMELINE_MODES
from radiant.agents.gdelt.config import GDELTConfig
from radiant.agents.gdelt.models import GDELTTimelinePoint, GDELTTimelineResult

logger = logging.getLogger(__name__)


class GDELTTimelineAgent(BaseAgent):
    """
    Agent that generates timeline and trend plots from GDELT data.

    Supports the following GDELT timeline modes:

    - **TimelineVol**: Coverage volume as percentage of global coverage.
    - **TimelineVolRaw**: Raw article counts over time.
    - **TimelineTone**: Average sentiment tone over time.
    - **TimelineSourceCountry**: Volume broken down by source country.
    - **TimelineLang**: Volume broken down by article language.

    The agent queries the API, structures the data, and renders
    matplotlib plots saved to the configured output directory.
    """

    def __init__(
        self,
        config: GDELTConfig,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the GDELT timeline agent.

        Args:
            config: GDELT configuration.
            enabled: Whether the agent is enabled.
        """
        super().__init__(enabled=enabled)
        self._config = config
        self._client = GDELTClient(config)

    @property
    def name(self) -> str:
        return "GDELTTimelineAgent"

    @property
    def category(self) -> AgentCategory:
        return AgentCategory.UTILITY

    @property
    def description(self) -> str:
        return (
            "Generates timeline and trend plots from GDELT "
            "coverage data for analysis"
        )

    def _execute(
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
        save_plot: bool = True,
        output_filename: Optional[str] = None,
        **kwargs: Any,
    ) -> GDELTTimelineResult:
        """
        Execute a GDELT timeline query and generate a plot.

        Args:
            query: Search keyword or phrase.
            mode: Timeline mode (TimelineVol, TimelineVolRaw,
                  TimelineTone, TimelineSourceCountry, TimelineLang).
            timespan: Rolling window (e.g. ``"30d"``, ``"3months"``).
            start_datetime: Start time as ``YYYYMMDDHHmmSS``.
            end_datetime: End time as ``YYYYMMDDHHmmSS``.
            sourcecountry: FIPS country code filter.
            sourcelang: Language filter.
            domain: Domain filter.
            theme: Theme filter.
            smoothing: Smoothing window size.
            save_plot: Whether to save the plot to disk.
            output_filename: Custom output filename (without extension).

        Returns:
            ``GDELTTimelineResult`` containing the data points
            and the path to the generated plot (if saved).
        """
        if mode not in TIMELINE_MODES:
            raise ValueError(
                f"Invalid timeline mode: {mode!r}. "
                f"Must be one of: {sorted(TIMELINE_MODES)}"
            )

        effective_timespan = timespan or self._config.default_timespan
        effective_smoothing = smoothing if smoothing is not None else self._config.default_smoothing

        self.logger.info(
            f"Querying GDELT timeline: query={query!r}, "
            f"mode={mode}, timespan={effective_timespan}"
        )

        data_points = self._client.query_timeline(
            query,
            mode=mode,
            timespan=effective_timespan,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            sourcecountry=sourcecountry,
            sourcelang=sourcelang,
            domain=domain,
            theme=theme,
            smoothing=effective_smoothing,
        )

        self.logger.info(
            f"Retrieved {len(data_points)} data points for "
            f"query={query!r}, mode={mode}"
        )

        result = GDELTTimelineResult(
            query=query,
            mode=mode,
            timespan=effective_timespan,
            data_points=data_points,
        )

        if save_plot and data_points:
            plot_path = self._generate_plot(
                result, output_filename=output_filename
            )
            result.plot_path = plot_path

        return result

    def _generate_plot(
        self,
        result: GDELTTimelineResult,
        output_filename: Optional[str] = None,
    ) -> str:
        """
        Generate a matplotlib plot from timeline data.

        Args:
            result: Timeline result containing data points.
            output_filename: Custom filename (without extension).

        Returns:
            Absolute path to the saved plot image.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        output_dir = Path(self._config.plot_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group data by series
        series_data: Dict[str, List[GDELTTimelinePoint]] = defaultdict(list)
        for point in result.data_points:
            series_data[point.series].append(point)

        fig, ax = plt.subplots(
            figsize=(self._config.plot_width, self._config.plot_height)
        )

        is_multi_series = len(series_data) > 1

        for series_name, points in series_data.items():
            dates = self._parse_dates([p.date for p in points])
            values = [p.value for p in points]

            if not dates or not values:
                continue

            label = series_name if is_multi_series else result.query
            ax.plot(dates, values, label=label, linewidth=1.5)

        # Configure axes
        title = self._build_plot_title(result)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel(self._get_y_label(result.mode), fontsize=11)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=30, ha="right")

        ax.grid(True, alpha=0.3, linestyle="--")

        if is_multi_series:
            # Limit legend entries for readability
            handles, labels = ax.get_legend_handles_labels()
            if len(labels) > 15:
                # Show only the top N series by total value
                series_totals = {
                    name: sum(p.value for p in pts)
                    for name, pts in series_data.items()
                }
                top_series = sorted(
                    series_totals, key=lambda x: series_totals[x], reverse=True
                )[:15]
                filtered_handles = []
                filtered_labels = []
                for handle, label in zip(handles, labels):
                    if label in top_series:
                        filtered_handles.append(handle)
                        filtered_labels.append(label)
                ax.legend(
                    filtered_handles,
                    filtered_labels,
                    loc="upper left",
                    fontsize=8,
                    framealpha=0.8,
                )
            else:
                ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

        plt.tight_layout()

        # Save
        if output_filename:
            # Sanitize to prevent path traversal
            filename = f"{Path(output_filename).name}.png"
        else:
            safe_query = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in result.query[:40]
            ).strip("_")
            filename = f"gdelt_timeline_{result.mode}_{safe_query}.png"

        filepath = output_dir / filename
        if not filepath.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError(f"Output path escapes output directory: {filepath}")
        fig.savefig(filepath, dpi=self._config.plot_dpi, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"Saved timeline plot to {filepath}")
        return str(filepath.resolve())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_dates(date_strings: List[str]) -> List[datetime]:
        """
        Parse GDELT date strings into datetime objects.

        GDELT returns dates in various formats depending on
        the time resolution.
        """
        formats = [
            "%Y%m%dT%H%M%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y%m%d%H%M%S",
            "%Y%m%d",
        ]

        parsed: List[datetime] = []
        for ds in date_strings:
            ds = ds.strip()
            if not ds:
                continue
            for fmt in formats:
                try:
                    parsed.append(datetime.strptime(ds, fmt))
                    break
                except ValueError:
                    continue
            else:
                logger.debug("Could not parse date string: %r", ds)
        return parsed

    @staticmethod
    def _build_plot_title(result: GDELTTimelineResult) -> str:
        """Build a descriptive plot title."""
        mode_labels = {
            "TimelineVol": "Coverage Volume (% of global)",
            "TimelineVolRaw": "Article Count",
            "TimelineTone": "Average Sentiment Tone",
            "TimelineSourceCountry": "Coverage by Source Country",
            "TimelineLang": "Coverage by Language",
        }
        mode_label = mode_labels.get(result.mode, result.mode)
        return f"GDELT: {mode_label}\nQuery: \"{result.query}\""

    @staticmethod
    def _get_y_label(mode: str) -> str:
        """Get the Y-axis label for a given mode."""
        labels = {
            "TimelineVol": "% of Global Coverage",
            "TimelineVolRaw": "Article Count",
            "TimelineTone": "Tone Score",
            "TimelineSourceCountry": "Article Count",
            "TimelineLang": "Article Count",
        }
        return labels.get(mode, "Value")

    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[GDELTTimelineResult]:
        """Return empty result on error."""
        self.logger.warning(f"GDELT timeline query failed: {error}")
        return GDELTTimelineResult(
            query=kwargs.get("query", ""),
            mode=kwargs.get("mode", "TimelineVol"),
            timespan=kwargs.get("timespan", self._config.default_timespan),
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
