"""
Data models for the GDELT news agents.

Provides dataclasses representing GDELT articles, timeline data points,
graph nodes, and query results used throughout the GDELT agent subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GDELTArticle:
    """A single news article returned by the GDELT DOC 2.0 API."""

    url: str
    title: str
    seendate: str
    domain: str
    language: str
    sourcecountry: str
    socialimage: str = ""
    url_mobile: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "seendate": self.seendate,
            "domain": self.domain,
            "language": self.language,
            "sourcecountry": self.sourcecountry,
            "socialimage": self.socialimage,
            "url_mobile": self.url_mobile,
        }


@dataclass
class GDELTTimelinePoint:
    """A single data point in a GDELT timeline series."""

    date: str
    value: float
    series: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "date": self.date,
            "value": self.value,
            "series": self.series,
        }


@dataclass
class GDELTTimelineResult:
    """Result from a GDELT timeline query."""

    query: str
    mode: str
    timespan: str
    data_points: List[GDELTTimelinePoint] = field(default_factory=list)
    plot_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "mode": self.mode,
            "timespan": self.timespan,
            "data_points": [dp.to_dict() for dp in self.data_points],
            "plot_path": self.plot_path,
        }


@dataclass
class GDELTArticleSearchResult:
    """Result from a GDELT article search."""

    query: str
    total_articles: int
    articles: List[GDELTArticle] = field(default_factory=list)
    date_range: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "total_articles": self.total_articles,
            "articles": [a.to_dict() for a in self.articles],
            "date_range": self.date_range,
        }


@dataclass
class GDELTGraphNode:
    """A node in a GDELT-derived network graph."""

    node_id: str
    label: str
    node_type: str  # "article", "domain", "country", "theme"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type,
            "attributes": self.attributes,
        }


@dataclass
class GDELTGraphEdge:
    """An edge in a GDELT-derived network graph."""

    source: str
    target: str
    edge_type: str  # "shares_domain", "same_country", "co_occurrence", "temporal"
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "attributes": self.attributes,
        }


@dataclass
class GDELTGraphResult:
    """Result from a GDELT graph construction."""

    query: str
    nodes: List[GDELTGraphNode] = field(default_factory=list)
    edges: List[GDELTGraphEdge] = field(default_factory=list)
    plot_path: Optional[str] = None
    graph_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "plot_path": self.plot_path,
            "graph_metrics": self.graph_metrics,
        }
