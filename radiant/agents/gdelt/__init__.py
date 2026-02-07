"""
GDELT News Agents for the Radiant RAG pipeline.

Provides three specialized agents for working with the GDELT
(Global Database of Events, Language, and Tone) DOC 2.0 API:

- **GDELTDownloadAgent**: Downloads news articles for ingestion
  into the Radiant RAG pipeline.
- **GDELTTimelineAgent**: Generates timeline and trend plots
  for coverage analysis.
- **GDELTGraphAgent**: Constructs and visualizes network graphs
  from article data using NetworkX.
"""

from radiant.agents.gdelt.config import GDELTConfig, load_gdelt_config
from radiant.agents.gdelt.client import GDELTClient
from radiant.agents.gdelt.models import (
    GDELTArticle,
    GDELTArticleSearchResult,
    GDELTGraphEdge,
    GDELTGraphNode,
    GDELTGraphResult,
    GDELTTimelinePoint,
    GDELTTimelineResult,
)
from radiant.agents.gdelt.download_agent import GDELTDownloadAgent
from radiant.agents.gdelt.timeline_agent import GDELTTimelineAgent
from radiant.agents.gdelt.graph_agent import GDELTGraphAgent

__all__ = [
    # Configuration
    "GDELTConfig",
    "load_gdelt_config",
    # Client
    "GDELTClient",
    # Models
    "GDELTArticle",
    "GDELTArticleSearchResult",
    "GDELTGraphEdge",
    "GDELTGraphNode",
    "GDELTGraphResult",
    "GDELTTimelinePoint",
    "GDELTTimelineResult",
    # Agents
    "GDELTDownloadAgent",
    "GDELTTimelineAgent",
    "GDELTGraphAgent",
]
