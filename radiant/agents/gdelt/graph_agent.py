"""
GDELT Network Graph Agent.

Builds network graphs from GDELT article data, computes graph
metrics, and generates visual plots. Uses NetworkX as the in-memory
graph engine for construction, analysis, and layout computation.

Graph construction strategies:
- **Co-occurrence**: Articles sharing the same domain, country, or
  temporal window are connected.
- **Domain**: Domains are nodes; edges represent shared coverage of
  the same topic.
- **Country**: Countries are nodes; edges represent co-coverage.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from radiant.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    BaseAgent,
)
from radiant.agents.gdelt.client import GDELTClient
from radiant.agents.gdelt.config import GDELTConfig
from radiant.agents.gdelt.models import (
    GDELTArticle,
    GDELTGraphEdge,
    GDELTGraphNode,
    GDELTGraphResult,
)

logger = logging.getLogger(__name__)


class GDELTGraphAgent(BaseAgent):
    """
    Agent that constructs and visualizes network graphs from GDELT data.

    The agent downloads articles from GDELT, builds a NetworkX graph
    representing relationships between articles, domains, and countries,
    then computes graph metrics and renders a visual plot.

    Available graph types:

    - ``"co_occurrence"``: Articles as nodes, edges between articles
      sharing domain or published within a temporal window.
    - ``"domain"``: Domains as nodes, edges weighted by shared
      coverage volume.
    - ``"country"``: Countries as nodes, edges weighted by shared
      coverage topics.
    """

    def __init__(
        self,
        config: GDELTConfig,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the GDELT graph agent.

        Args:
            config: GDELT configuration.
            enabled: Whether the agent is enabled.
        """
        super().__init__(enabled=enabled)
        self._config = config
        self._client = GDELTClient(config)

    @property
    def name(self) -> str:
        return "GDELTGraphAgent"

    @property
    def category(self) -> AgentCategory:
        return AgentCategory.UTILITY

    @property
    def description(self) -> str:
        return (
            "Constructs and visualizes network graphs from GDELT "
            "article data using NetworkX"
        )

    def _execute(
        self,
        query: str,
        graph_type: str = "domain",
        *,
        timespan: Optional[str] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        sourcecountry: Optional[str] = None,
        sourcelang: Optional[str] = None,
        domain: Optional[str] = None,
        theme: Optional[str] = None,
        max_records: Optional[int] = None,
        save_plot: bool = True,
        save_graph_data: bool = True,
        output_filename: Optional[str] = None,
        articles: Optional[List[GDELTArticle]] = None,
        **kwargs: Any,
    ) -> GDELTGraphResult:
        """
        Build a network graph from GDELT article data.

        Args:
            query: Search keyword or phrase.
            graph_type: Type of graph to build. One of
                ``"co_occurrence"``, ``"domain"``, ``"country"``.
            timespan: Rolling window (e.g. ``"7d"``).
            start_datetime: Start time as ``YYYYMMDDHHmmSS``.
            end_datetime: End time as ``YYYYMMDDHHmmSS``.
            sourcecountry: FIPS country code filter.
            sourcelang: Language filter.
            domain: Domain filter.
            theme: Theme filter.
            max_records: Maximum articles to retrieve.
            save_plot: Whether to save the plot to disk.
            save_graph_data: Whether to save graph data as JSON.
            output_filename: Custom output filename (without extension).
            articles: Pre-fetched articles (skip API query if provided).

        Returns:
            ``GDELTGraphResult`` with nodes, edges, metrics,
            and path to the generated plot.
        """
        valid_types = ("co_occurrence", "domain", "country")
        if graph_type not in valid_types:
            raise ValueError(
                f"Invalid graph_type: {graph_type!r}. "
                f"Must be one of: {valid_types}"
            )

        # Fetch articles if not pre-supplied
        if articles is None:
            self.logger.info(
                f"Fetching articles for graph construction: "
                f"query={query!r}, graph_type={graph_type}"
            )
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

        if not articles:
            self.logger.warning(f"No articles found for query: {query!r}")
            return GDELTGraphResult(query=query)

        self.logger.info(
            f"Building {graph_type} graph from {len(articles)} articles"
        )

        # Build the NetworkX graph
        if graph_type == "co_occurrence":
            G = self._build_co_occurrence_graph(articles)
        elif graph_type == "domain":
            G = self._build_domain_graph(articles)
        else:
            G = self._build_country_graph(articles)

        # Extract nodes and edges for the result model
        nodes = self._extract_nodes(G)
        edges = self._extract_edges(G)
        metrics = self._compute_graph_metrics(G)

        result = GDELTGraphResult(
            query=query,
            nodes=nodes,
            edges=edges,
            graph_metrics=metrics,
        )

        if save_plot and nodes:
            plot_path = self._generate_plot(
                G, result, graph_type, output_filename
            )
            result.plot_path = plot_path

        if save_graph_data and nodes:
            self._save_graph_data(result, output_filename)

        return result

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_co_occurrence_graph(
        self,
        articles: List[GDELTArticle],
    ) -> nx.Graph:
        """
        Build a co-occurrence graph where articles are nodes.

        Edges are created between articles that:
        - Share the same domain
        - Were published within the configured temporal window
        """
        G = nx.Graph()
        window_hours = self._config.co_occurrence_window_hours

        # Add article nodes
        for i, article in enumerate(articles):
            node_id = f"article_{i}"
            G.add_node(
                node_id,
                label=article.title[:60],
                node_type="article",
                domain=article.domain,
                country=article.sourcecountry,
                language=article.language,
                seendate=article.seendate,
                url=article.url,
            )

        # Build edges based on shared domain
        domain_groups: Dict[str, List[str]] = defaultdict(list)
        for node_id, data in G.nodes(data=True):
            domain_groups[data["domain"]].append(node_id)

        for domain_val, node_ids in domain_groups.items():
            if len(node_ids) < 2:
                continue
            # Connect articles from the same domain (limit connections)
            max_edges = min(len(node_ids), 10)
            for i in range(max_edges):
                for j in range(i + 1, max_edges):
                    G.add_edge(
                        node_ids[i],
                        node_ids[j],
                        edge_type="shares_domain",
                        weight=1.0,
                        domain=domain_val,
                    )

        # Build edges based on temporal proximity
        dated_nodes = []
        for node_id, data in G.nodes(data=True):
            dt = self._parse_seendate(data.get("seendate", ""))
            if dt:
                dated_nodes.append((node_id, dt))

        dated_nodes.sort(key=lambda x: x[1])
        window_delta = timedelta(hours=window_hours)

        for i, (node_a, dt_a) in enumerate(dated_nodes):
            for j in range(i + 1, len(dated_nodes)):
                node_b, dt_b = dated_nodes[j]
                if dt_b - dt_a > window_delta:
                    break
                # Only add temporal edge if not already connected
                if not G.has_edge(node_a, node_b):
                    G.add_edge(
                        node_a,
                        node_b,
                        edge_type="temporal",
                        weight=0.5,
                    )

        # Prune to max nodes
        return self._prune_graph(G)

    def _build_domain_graph(
        self,
        articles: List[GDELTArticle],
    ) -> nx.Graph:
        """
        Build a domain-centric graph.

        Domains are nodes, edges connect domains that published
        articles about the same topic (as indicated by sharing the
        query keyword). Edge weight reflects the number of shared
        coverage instances.
        """
        G = nx.Graph()

        # Count articles per domain
        domain_counts: Counter = Counter()
        domain_countries: Dict[str, Counter] = defaultdict(Counter)
        domain_languages: Dict[str, Counter] = defaultdict(Counter)

        for article in articles:
            d = article.domain
            domain_counts[d] += 1
            if article.sourcecountry:
                domain_countries[d][article.sourcecountry] += 1
            if article.language:
                domain_languages[d][article.language] += 1

        # Add domain nodes
        for domain_val, count in domain_counts.items():
            top_country = ""
            if domain_countries[domain_val]:
                top_country = domain_countries[domain_val].most_common(1)[0][0]
            top_language = ""
            if domain_languages[domain_val]:
                top_language = domain_languages[domain_val].most_common(1)[0][0]

            G.add_node(
                domain_val,
                label=domain_val,
                node_type="domain",
                article_count=count,
                top_country=top_country,
                top_language=top_language,
            )

        # Connect domains that share the same source country
        country_domains: Dict[str, List[str]] = defaultdict(list)
        for domain_val, countries in domain_countries.items():
            for country in countries:
                country_domains[country].append(domain_val)

        for country, domains in country_domains.items():
            for i in range(len(domains)):
                for j in range(i + 1, len(domains)):
                    d1, d2 = domains[i], domains[j]
                    if G.has_edge(d1, d2):
                        G[d1][d2]["weight"] += 1.0
                    else:
                        G.add_edge(
                            d1,
                            d2,
                            edge_type="same_country",
                            weight=1.0,
                            country=country,
                        )

        # Filter edges below weight threshold
        edges_to_remove = [
            (u, v)
            for u, v, data in G.edges(data=True)
            if data.get("weight", 0) < self._config.graph_min_edge_weight
        ]
        G.remove_edges_from(edges_to_remove)

        # Remove isolated nodes
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)

        return self._prune_graph(G)

    def _build_country_graph(
        self,
        articles: List[GDELTArticle],
    ) -> nx.Graph:
        """
        Build a country-centric graph.

        Countries are nodes, edges connect countries whose media
        both covered the queried topic. Edge weight reflects the
        volume of shared coverage.
        """
        G = nx.Graph()

        # Count articles per country
        country_counts: Counter = Counter()
        country_domains: Dict[str, Set[str]] = defaultdict(set)

        for article in articles:
            c = article.sourcecountry
            if not c:
                continue
            country_counts[c] += 1
            country_domains[c].add(article.domain)

        # Add country nodes
        for country, count in country_counts.items():
            G.add_node(
                country,
                label=country,
                node_type="country",
                article_count=count,
                domain_count=len(country_domains[country]),
            )

        # Connect countries that share domains covering the topic
        countries = list(country_counts.keys())
        for i in range(len(countries)):
            for j in range(i + 1, len(countries)):
                c1, c2 = countries[i], countries[j]
                shared = country_domains[c1] & country_domains[c2]
                if shared:
                    G.add_edge(
                        c1,
                        c2,
                        edge_type="shared_coverage",
                        weight=float(len(shared)),
                        shared_domains=len(shared),
                    )

        return self._prune_graph(G)

    # ------------------------------------------------------------------
    # Graph analysis
    # ------------------------------------------------------------------

    def _compute_graph_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """Compute standard graph metrics."""
        if G.number_of_nodes() == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
            }

        metrics: Dict[str, Any] = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": round(nx.density(G), 4),
        }

        # Connected components
        if not nx.is_empty(G):
            components = list(nx.connected_components(G))
            metrics["num_components"] = len(components)
            metrics["largest_component_size"] = max(len(c) for c in components)

        # Degree statistics
        degrees = [d for _, d in G.degree()]
        if degrees:
            metrics["avg_degree"] = round(sum(degrees) / len(degrees), 2)
            metrics["max_degree"] = max(degrees)

        # Top nodes by degree centrality
        degree_cent = nx.degree_centrality(G)
        top_nodes = sorted(
            degree_cent.items(), key=lambda x: x[1], reverse=True
        )[:10]
        metrics["top_nodes_by_centrality"] = [
            {"node": n, "centrality": round(c, 4)} for n, c in top_nodes
        ]

        # Betweenness centrality for non-trivial graphs
        if G.number_of_nodes() <= 500:
            betweenness = nx.betweenness_centrality(G)
            top_between = sorted(
                betweenness.items(), key=lambda x: x[1], reverse=True
            )[:10]
            metrics["top_nodes_by_betweenness"] = [
                {"node": n, "betweenness": round(b, 4)}
                for n, b in top_between
            ]

        return metrics

    def _extract_nodes(self, G: nx.Graph) -> List[GDELTGraphNode]:
        """Extract nodes from NetworkX graph into model objects."""
        nodes: List[GDELTGraphNode] = []
        for node_id, data in G.nodes(data=True):
            attrs = {k: v for k, v in data.items() if k not in ("label", "node_type")}
            nodes.append(
                GDELTGraphNode(
                    node_id=str(node_id),
                    label=str(data.get("label", node_id)),
                    node_type=str(data.get("node_type", "unknown")),
                    attributes=attrs,
                )
            )
        return nodes

    def _extract_edges(self, G: nx.Graph) -> List[GDELTGraphEdge]:
        """Extract edges from NetworkX graph into model objects."""
        edges: List[GDELTGraphEdge] = []
        for u, v, data in G.edges(data=True):
            attrs = {
                k: v_val
                for k, v_val in data.items()
                if k not in ("edge_type", "weight")
            }
            edges.append(
                GDELTGraphEdge(
                    source=str(u),
                    target=str(v),
                    edge_type=str(data.get("edge_type", "related")),
                    weight=float(data.get("weight", 1.0)),
                    attributes=attrs,
                )
            )
        return edges

    def _prune_graph(self, G: nx.Graph) -> nx.Graph:
        """Prune graph to the configured maximum node count."""
        max_nodes = self._config.graph_max_nodes
        if G.number_of_nodes() <= max_nodes:
            return G

        # Keep nodes with highest degree
        degree_sorted = sorted(
            G.degree(), key=lambda x: x[1], reverse=True
        )
        keep_nodes = {n for n, _ in degree_sorted[:max_nodes]}
        remove_nodes = set(G.nodes()) - keep_nodes
        G.remove_nodes_from(remove_nodes)

        self.logger.debug(
            f"Pruned graph from {G.number_of_nodes() + len(remove_nodes)} "
            f"to {G.number_of_nodes()} nodes"
        )
        return G

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _generate_plot(
        self,
        G: nx.Graph,
        result: GDELTGraphResult,
        graph_type: str,
        output_filename: Optional[str] = None,
    ) -> str:
        """
        Generate a matplotlib plot of the graph.

        Args:
            G: NetworkX graph to plot.
            result: Graph result for metadata.
            graph_type: Type of graph for labeling.
            output_filename: Custom filename (without extension).

        Returns:
            Absolute path to the saved plot image.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(self._config.graph_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(
            figsize=(self._config.plot_width, self._config.plot_height)
        )

        if G.number_of_nodes() == 0:
            ax.text(
                0.5, 0.5,
                "No graph data available",
                ha="center", va="center",
                fontsize=14,
                transform=ax.transAxes,
            )
        else:
            # Compute layout
            layout = self._compute_layout(G)

            # Node sizes based on degree
            degrees = dict(G.degree())
            max_deg = max(degrees.values()) if degrees else 1
            node_sizes = [
                200 + (degrees.get(n, 0) / max_deg) * 800
                for n in G.nodes()
            ]

            # Node colors by type
            node_colors = self._get_node_colors(G)

            # Edge widths based on weight
            edge_weights = [
                data.get("weight", 1.0)
                for _, _, data in G.edges(data=True)
            ]
            max_weight = max(edge_weights) if edge_weights else 1.0
            edge_widths = [
                0.5 + (w / max_weight) * 2.5 for w in edge_weights
            ]

            # Draw
            nx.draw_networkx_edges(
                G,
                layout,
                ax=ax,
                alpha=0.3,
                width=edge_widths,
                edge_color="#888888",
            )
            nx.draw_networkx_nodes(
                G,
                layout,
                ax=ax,
                node_size=node_sizes,
                node_color=node_colors,
                alpha=0.8,
                edgecolors="#333333",
                linewidths=0.5,
            )

            # Labels for top nodes only
            top_count = min(20, G.number_of_nodes())
            top_nodes = sorted(
                degrees.items(), key=lambda x: x[1], reverse=True
            )[:top_count]
            labels = {}
            for node_id, _ in top_nodes:
                label = G.nodes[node_id].get("label", str(node_id))
                labels[node_id] = label[:25]

            nx.draw_networkx_labels(
                G,
                layout,
                labels,
                ax=ax,
                font_size=7,
                font_weight="bold",
            )

        # Title
        type_labels = {
            "co_occurrence": "Article Co-occurrence",
            "domain": "Domain Network",
            "country": "Country Coverage Network",
        }
        type_label = type_labels.get(graph_type, graph_type)
        ax.set_title(
            f"GDELT: {type_label}\nQuery: \"{result.query}\"  |  "
            f"Nodes: {G.number_of_nodes()}  |  "
            f"Edges: {G.number_of_edges()}",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.axis("off")
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
            filename = f"gdelt_graph_{graph_type}_{safe_query}.png"

        filepath = output_dir / filename
        if not filepath.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError(f"Output path escapes output directory: {filepath}")
        fig.savefig(filepath, dpi=self._config.plot_dpi, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"Saved graph plot to {filepath}")
        return str(filepath.resolve())

    def _compute_layout(self, G: nx.Graph) -> Dict[Any, Any]:
        """Compute node positions using the configured layout algorithm."""
        layout_name = self._config.graph_layout

        if layout_name == "kamada_kawai" and G.number_of_nodes() <= 200:
            return nx.kamada_kawai_layout(G)
        elif layout_name == "circular":
            return nx.circular_layout(G)
        elif layout_name == "shell":
            return nx.shell_layout(G)
        else:
            # Spring layout (default) with deterministic seed
            return nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    @staticmethod
    def _get_node_colors(G: nx.Graph) -> List[str]:
        """Assign colors based on node type."""
        color_map = {
            "article": "#4A90D9",
            "domain": "#E67E22",
            "country": "#27AE60",
            "theme": "#8E44AD",
            "unknown": "#95A5A6",
        }
        return [
            color_map.get(G.nodes[n].get("node_type", "unknown"), "#95A5A6")
            for n in G.nodes()
        ]

    def _save_graph_data(
        self,
        result: GDELTGraphResult,
        output_filename: Optional[str] = None,
    ) -> str:
        """Save graph data to JSON file."""
        output_dir = Path(self._config.graph_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_filename:
            # Sanitize to prevent path traversal
            filename = f"{Path(output_filename).name}_data.json"
        else:
            safe_query = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in result.query[:40]
            ).strip("_")
            filename = f"gdelt_graph_data_{safe_query}.json"

        filepath = output_dir / filename
        if not filepath.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError(f"Output path escapes output directory: {filepath}")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved graph data to {filepath}")
        return str(filepath.resolve())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_seendate(seendate: str) -> Optional[datetime]:
        """Parse a GDELT seendate string to datetime."""
        if not seendate:
            return None

        formats = [
            "%Y%m%dT%H%M%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y%m%d%H%M%S",
            "%Y%m%d",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(seendate.strip(), fmt)
            except ValueError:
                continue
        return None

    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[GDELTGraphResult]:
        """Return empty result on error."""
        self.logger.warning(f"GDELT graph construction failed: {error}")
        return GDELTGraphResult(query=kwargs.get("query", ""))

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
