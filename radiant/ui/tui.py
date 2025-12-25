"""
Textual-based Terminal User Interface for Agentic RAG.

Provides an interactive console interface for querying the RAG system
with real-time pipeline visualization and professional report display.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    LoadingIndicator,
    Markdown,
    RichLog,
    Rule,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from textual.message import Message

from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich import box

if TYPE_CHECKING:
    from radiant.app import RadiantRAG
    from radiant.orchestrator import PipelineResult
    from radiant.utils.metrics import RunMetrics, StepMetric
    from radiant.agents import AgentContext

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Widgets
# =============================================================================

class QueryInput(Horizontal):
    """Query input container with input field and run button."""
    
    DEFAULT_CSS = """
    QueryInput {
        height: 3;
        margin: 0 1;
        padding: 0;
    }
    
    QueryInput Input {
        width: 1fr;
        margin-right: 1;
    }
    
    QueryInput Button {
        width: 10;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Input(
            placeholder="Ask a question about your knowledge base...",
            id="query-input",
        )
        yield Button("Run", id="run-btn", variant="primary")


class TimelineStep(Static):
    """A single step in the execution timeline."""
    
    DEFAULT_CSS = """
    TimelineStep {
        height: auto;
        padding: 0 1;
        margin: 0;
    }
    
    TimelineStep.success {
        color: $success;
    }
    
    TimelineStep.failed {
        color: $error;
    }
    
    TimelineStep.running {
        color: $warning;
    }
    
    TimelineStep.pending {
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        step_name: str,
        status: str = "pending",
        duration_ms: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.step_name = step_name
        self.status = status
        self.duration_ms = duration_ms
        self._update_display()
    
    def _update_display(self) -> None:
        """Update the display based on status."""
        icons = {
            "success": "●",
            "failed": "✖",
            "running": "◐",
            "pending": "○",
        }
        icon = icons.get(self.status, "○")
        
        if self.duration_ms is not None:
            duration_str = f" ({self.duration_ms:.0f}ms)"
        else:
            duration_str = ""
        
        self.update(f"{icon} {self.step_name}{duration_str}")
        
        # Update CSS class
        self.remove_class("success", "failed", "running", "pending")
        self.add_class(self.status)
    
    def set_status(self, status: str, duration_ms: Optional[float] = None) -> None:
        """Update the step status."""
        self.status = status
        self.duration_ms = duration_ms
        self._update_display()


class RunTimeline(Vertical):
    """Timeline showing pipeline execution steps."""
    
    DEFAULT_CSS = """
    RunTimeline {
        width: 100%;
        height: auto;
        border: solid $primary;
        padding: 1;
    }
    
    RunTimeline .timeline-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._steps: Dict[str, TimelineStep] = {}
    
    def compose(self) -> ComposeResult:
        yield Static("Run Timeline / Trace", classes="timeline-title")
    
    def clear_steps(self) -> None:
        """Clear all steps."""
        for step in self._steps.values():
            step.remove()
        self._steps.clear()
    
    def add_step(self, step_name: str, status: str = "pending") -> None:
        """Add a step to the timeline."""
        if step_name not in self._steps:
            step_widget = TimelineStep(step_name, status)
            self._steps[step_name] = step_widget
            self.mount(step_widget)
    
    def update_step(
        self,
        step_name: str,
        status: str,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Update a step's status."""
        if step_name in self._steps:
            self._steps[step_name].set_status(status, duration_ms)
        else:
            self.add_step(step_name, status)
            if duration_ms is not None:
                self._steps[step_name].duration_ms = duration_ms
                self._steps[step_name]._update_display()


class AnswerPanel(Vertical):
    """Panel displaying the answer and citations."""
    
    DEFAULT_CSS = """
    AnswerPanel {
        width: 100%;
        height: 100%;
        border: solid $success;
        padding: 1;
    }
    
    AnswerPanel .answer-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    AnswerPanel .answer-content {
        height: auto;
        margin-bottom: 1;
    }
    
    AnswerPanel .citations-title {
        text-style: bold;
        margin-top: 1;
    }
    
    AnswerPanel .citations-list {
        height: auto;
        color: $text-muted;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._answer_widget: Optional[Static] = None
        self._citations_widget: Optional[Static] = None
    
    def compose(self) -> ComposeResult:
        yield Static("Answer & Citations", classes="answer-title")
        yield ScrollableContainer(
            Static("", id="answer-content", classes="answer-content"),
            Static("Citations", classes="citations-title"),
            Static("", id="citations-list", classes="citations-list"),
        )
    
    def set_answer(self, answer: str) -> None:
        """Set the answer text."""
        content = self.query_one("#answer-content", Static)
        content.update(answer)
    
    def set_citations(self, citations: List[str]) -> None:
        """Set the citations list."""
        citations_widget = self.query_one("#citations-list", Static)
        if citations:
            citations_text = "  ".join(f"[{c}]" for c in citations)
            citations_widget.update(citations_text)
        else:
            citations_widget.update("No citations")


class MetricsDisplay(Static):
    """Display for run metrics."""
    
    DEFAULT_CSS = """
    MetricsDisplay {
        height: auto;
        padding: 1;
    }
    """
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update with new metrics data."""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.2f}")
            else:
                table.add_row(key, str(value))
        
        self.update(table)


# =============================================================================
# Main TUI Application
# =============================================================================

class AgenticRAGApp(App):
    """
    Main Textual application for Agentic RAG.
    
    Provides an interactive interface for:
    - Querying the knowledge base
    - Viewing execution timeline
    - Browsing detailed results across tabs
    """
    
    TITLE = "Agentic RAG Console"
    SUB_TITLE = "Interactive Knowledge Base Query Interface"
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #main-container {
        height: 100%;
    }
    
    #top-section {
        height: 5;
        margin: 0 1;
    }
    
    #middle-section {
        height: 1fr;
        margin: 0 1;
    }
    
    #timeline-answer-container {
        height: 100%;
    }
    
    #timeline-panel {
        width: 35%;
        height: 100%;
        margin-right: 1;
    }
    
    #answer-panel {
        width: 65%;
        height: 100%;
    }
    
    #tabs-section {
        height: 50%;
        margin: 1;
    }
    
    .status-bar {
        height: 1;
        dock: bottom;
        background: $primary-background;
        color: $text;
        padding: 0 1;
    }
    
    #loading-indicator {
        display: none;
        dock: top;
        height: 1;
        background: $warning;
    }
    
    #loading-indicator.visible {
        display: block;
    }
    
    .log-panel {
        height: 100%;
        border: solid $primary-background;
    }
    
    .tab-content {
        padding: 1;
        height: 100%;
    }
    
    #overview-content, #plan-content, #queries-content, 
    #retrieval-content, #agents-content, #metrics-content, #logs-content {
        height: 100%;
        overflow-y: auto;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+n", "new_conversation", "New Conv"),
        Binding("ctrl+r", "focus_query", "Focus Query"),
        Binding("ctrl+s", "save_report", "Save Report"),
        Binding("escape", "cancel_query", "Cancel"),
    ]
    
    # Reactive state
    is_loading = reactive(False)
    current_mode = reactive("hybrid")
    conversation_id: Optional[str] = None
    last_result: Optional["PipelineResult"] = None
    
    def __init__(self, rag_app: "RadiantRAG", **kwargs):
        super().__init__(**kwargs)
        self._rag_app = rag_app
        self._log_buffer: List[str] = []
        self.conversation_id = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="main-container"):
            # Loading indicator
            yield LoadingIndicator(id="loading-indicator")
            
            # Query input section
            with Container(id="top-section"):
                yield QueryInput()
            
            # Middle section: Timeline + Answer
            with Horizontal(id="middle-section"):
                with Container(id="timeline-answer-container"):
                    with Horizontal():
                        with Container(id="timeline-panel"):
                            yield RunTimeline(id="timeline")
                        with Container(id="answer-panel"):
                            yield AnswerPanel(id="answer")
            
            # Tabs section
            with Container(id="tabs-section"):
                with TabbedContent():
                    with TabPane("Overview", id="tab-overview"):
                        yield ScrollableContainer(
                            Static(id="overview-content"),
                            classes="tab-content",
                        )
                    with TabPane("Plan", id="tab-plan"):
                        yield ScrollableContainer(
                            Static(id="plan-content"),
                            classes="tab-content",
                        )
                    with TabPane("Queries", id="tab-queries"):
                        yield ScrollableContainer(
                            Static(id="queries-content"),
                            classes="tab-content",
                        )
                    with TabPane("Retrieval", id="tab-retrieval"):
                        yield ScrollableContainer(
                            Static(id="retrieval-content"),
                            classes="tab-content",
                        )
                    with TabPane("Agents", id="tab-agents"):
                        yield ScrollableContainer(
                            Static(id="agents-content"),
                            classes="tab-content",
                        )
                    with TabPane("Metrics", id="tab-metrics"):
                        yield ScrollableContainer(
                            Static(id="metrics-content"),
                            classes="tab-content",
                        )
                    with TabPane("Logs", id="tab-logs"):
                        yield RichLog(id="logs-content", classes="log-panel")
            
            # Status bar
            yield Static(
                f"Mode: {self.current_mode} | Conversation: None",
                classes="status-bar",
                id="status-bar",
            )
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Handle app mount event."""
        # Start a new conversation
        self.conversation_id = self._rag_app.start_conversation()
        self._update_status_bar()
        
        # Focus the query input
        self.query_one("#query-input", Input).focus()
        
        # Log startup
        self._log("Agentic RAG Console initialized")
        self._log(f"Conversation started: {self.conversation_id[:8]}...")
    
    def _update_status_bar(self) -> None:
        """Update the status bar."""
        conv_display = self.conversation_id[:8] + "..." if self.conversation_id else "None"
        status = self.query_one("#status-bar", Static)
        status.update(f"Mode: {self.current_mode} | Conversation: {conv_display}")
    
    def _log(self, message: str) -> None:
        """Add a log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_widget = self.query_one("#logs-content", RichLog)
        log_widget.write(f"[dim]{timestamp}[/dim] {message}")
    
    def watch_is_loading(self, loading: bool) -> None:
        """React to loading state changes."""
        indicator = self.query_one("#loading-indicator", LoadingIndicator)
        if loading:
            indicator.add_class("visible")
        else:
            indicator.remove_class("visible")
    
    @on(Button.Pressed, "#run-btn")
    def on_run_pressed(self, event: Button.Pressed) -> None:
        """Handle run button press."""
        query_input = self.query_one("#query-input", Input)
        query = query_input.value.strip()
        
        if query:
            self._execute_query(query)
    
    @on(Input.Submitted, "#query-input")
    def on_query_submitted(self, event: Input.Submitted) -> None:
        """Handle query submission via Enter."""
        query = event.value.strip()
        
        if query:
            self._execute_query(query)
    
    @work(thread=True)
    def _execute_query(self, query: str) -> None:
        """Execute a query in a background thread."""
        self.is_loading = True
        self._log(f"Executing query: {query[:50]}...")
        
        # Clear previous results
        self.call_from_thread(self._clear_results)
        
        # Setup timeline with expected steps
        expected_steps = [
            "Planning",
            "Query Decomposition",
            "Query Expansion",
            "Dense Retrieval",
            "BM25 Retrieval",
            "RRF Fusion",
            "Auto-Merge",
            "Reranking",
            "Answer Synthesis",
            "Critic",
        ]
        
        for step in expected_steps:
            self.call_from_thread(self._add_timeline_step, step)
        
        try:
            # Execute the query
            result = self._rag_app.query_raw(
                query,
                conversation_id=self.conversation_id,
                retrieval_mode=self.current_mode,
            )
            
            self.last_result = result
            
            # Update UI with results
            self.call_from_thread(self._display_results, result)
            self._log("Query completed successfully")
            
        except Exception as e:
            self._log(f"Error: {str(e)}")
            self.call_from_thread(
                self._show_error,
                f"Query failed: {str(e)}"
            )
        finally:
            self.is_loading = False
    
    def _clear_results(self) -> None:
        """Clear previous results."""
        timeline = self.query_one("#timeline", RunTimeline)
        timeline.clear_steps()
        
        answer_panel = self.query_one("#answer", AnswerPanel)
        answer_panel.set_answer("")
        answer_panel.set_citations([])
    
    def _add_timeline_step(self, step_name: str) -> None:
        """Add a step to the timeline."""
        timeline = self.query_one("#timeline", RunTimeline)
        timeline.add_step(step_name, "pending")
    
    def _display_results(self, result: "PipelineResult") -> None:
        """Display query results."""
        ctx = result.context
        metrics = result.metrics
        
        # Update timeline with actual step results
        timeline = self.query_one("#timeline", RunTimeline)
        timeline.clear_steps()
        
        for step in metrics.steps:
            status = "success" if step.ok else "failed"
            timeline.add_step(step.name, status)
            timeline.update_step(step.name, status, step.latency_ms)
        
        # Update answer panel
        answer_panel = self.query_one("#answer", AnswerPanel)
        answer_panel.set_answer(result.answer)
        
        # Extract citation doc IDs
        citations = []
        for doc, _ in ctx.reranked[:5]:
            doc_id = doc.meta.get("doc_id", doc.doc_id[:8])
            citations.append(doc_id)
        answer_panel.set_citations(citations)
        
        # Update tabs
        self._update_overview_tab(result)
        self._update_plan_tab(ctx)
        self._update_queries_tab(ctx)
        self._update_retrieval_tab(ctx)
        self._update_agents_tab(metrics)
        self._update_metrics_tab(result)
    
    def _update_overview_tab(self, result: "PipelineResult") -> None:
        """Update the overview tab."""
        ctx = result.context
        metrics = result.metrics
        
        lines = [
            f"[bold]Run ID:[/bold] {ctx.run_id}",
            f"[bold]Status:[/bold] {'SUCCESS' if result.success else 'FAILED'}",
            f"[bold]Total Latency:[/bold] {metrics.total_latency_ms:.0f}ms" if metrics.total_latency_ms else "",
            f"[bold]Steps Executed:[/bold] {len(metrics.steps)}",
            f"[bold]Documents Retrieved:[/bold] {len(ctx.reranked)}",
            "",
            "[bold]Query:[/bold]",
            f"  {ctx.original_query}",
        ]
        
        if ctx.warnings:
            lines.append("")
            lines.append("[bold yellow]Warnings:[/bold yellow]")
            for warning in ctx.warnings:
                lines.append(f"  • {warning}")
        
        content = self.query_one("#overview-content", Static)
        content.update("\n".join(lines))
    
    def _update_plan_tab(self, ctx: "AgentContext") -> None:
        """Update the plan tab."""
        plan = ctx.plan or {}
        
        lines = ["[bold]Execution Plan[/bold]", ""]
        
        if plan:
            for key, value in plan.items():
                status = "✓" if value else "✗"
                lines.append(f"  {status} {key}: {value}")
        else:
            lines.append("  No plan available")
        
        content = self.query_one("#plan-content", Static)
        content.update("\n".join(lines))
    
    def _update_queries_tab(self, ctx: "AgentContext") -> None:
        """Update the queries tab."""
        lines = ["[bold]Query Processing[/bold]", ""]
        
        lines.append("[bold]Original Query:[/bold]")
        lines.append(f"  {ctx.original_query}")
        lines.append("")
        
        if ctx.decomposed_queries:
            lines.append("[bold]Decomposed Queries:[/bold]")
            for i, q in enumerate(ctx.decomposed_queries, 1):
                lines.append(f"  Q{i}: {q}")
            lines.append("")
        
        if ctx.expansions:
            lines.append("[bold]Query Expansions:[/bold]")
            for i, exp in enumerate(ctx.expansions, 1):
                lines.append(f"  {i}. {exp}")
            lines.append("")
        
        if ctx.rewrites:
            lines.append("[bold]Query Rewrites:[/bold]")
            for orig, rewritten in ctx.rewrites:
                lines.append(f"  Original: {orig}")
                lines.append(f"  Rewritten: {rewritten}")
                lines.append("")
        
        content = self.query_one("#queries-content", Static)
        content.update("\n".join(lines))
    
    def _update_retrieval_tab(self, ctx: "AgentContext") -> None:
        """Update the retrieval tab."""
        lines = ["[bold]Retrieval Summary[/bold]", ""]
        
        lines.append(f"[bold]Dense Retrieved:[/bold] {len(ctx.dense_retrieved)} docs")
        lines.append(f"[bold]BM25 Retrieved:[/bold] {len(ctx.bm25_retrieved)} docs")
        lines.append(f"[bold]After Fusion:[/bold] {len(ctx.fused)} docs")
        lines.append(f"[bold]After Rerank:[/bold] {len(ctx.reranked)} docs")
        lines.append("")
        
        if ctx.reranked:
            lines.append("[bold]Top Retrieved Documents:[/bold]")
            lines.append("")
            
            for i, (doc, score) in enumerate(ctx.reranked[:5], 1):
                source = doc.meta.get("source_path", "unknown")
                if "/" in source:
                    source = source.split("/")[-1]
                page = doc.meta.get("page_number", "-")
                
                lines.append(f"  [{i}] {source} (Page {page})")
                lines.append(f"      Score: {score:.4f}")
                snippet = doc.content[:100].replace("\n", " ")
                lines.append(f"      {snippet}...")
                lines.append("")
        
        content = self.query_one("#retrieval-content", Static)
        content.update("\n".join(lines))
    
    def _update_agents_tab(self, metrics: "RunMetrics") -> None:
        """Update the agents tab."""
        lines = ["[bold]Agent Execution Trace[/bold]", ""]
        
        for i, step in enumerate(metrics.steps, 1):
            status = "✓" if step.ok else "✗"
            duration = f"{step.latency_ms:.0f}ms" if step.latency_ms else "N/A"
            
            lines.append(f"[Step {i}] {step.name}")
            lines.append(f"  Status: {status}")
            lines.append(f"  Duration: {duration}")
            
            if step.extra:
                for key, value in step.extra.items():
                    if key not in ("inputs", "outputs"):
                        lines.append(f"  {key}: {value}")
            
            if step.error:
                lines.append(f"  [red]Error: {step.error}[/red]")
            
            lines.append("")
        
        content = self.query_one("#agents-content", Static)
        content.update("\n".join(lines))
    
    def _update_metrics_tab(self, result: "PipelineResult") -> None:
        """Update the metrics tab."""
        metrics = result.metrics
        
        lines = ["[bold]Run Metrics[/bold]", ""]
        
        lines.append(f"[bold]Run ID:[/bold] {metrics.run_id}")
        lines.append(f"[bold]Total Latency:[/bold] {metrics.total_latency_ms:.2f}ms" if metrics.total_latency_ms else "")
        lines.append(f"[bold]Success Rate:[/bold] {metrics.success_rate:.1%}")
        lines.append("")
        
        lines.append("[bold]Step Latencies:[/bold]")
        for step in metrics.steps:
            if step.latency_ms:
                bar_len = min(int(step.latency_ms / 50), 40)
                bar = "█" * bar_len
                lines.append(f"  {step.name:30} {bar} {step.latency_ms:.0f}ms")
        
        if metrics.degraded_features:
            lines.append("")
            lines.append("[bold yellow]Degraded Features:[/bold yellow]")
            for feature in metrics.degraded_features:
                lines.append(f"  • {feature}")
        
        if metrics.warnings:
            lines.append("")
            lines.append("[bold yellow]Warnings:[/bold yellow]")
            for warning in metrics.warnings:
                lines.append(f"  • {warning}")
        
        content = self.query_one("#metrics-content", Static)
        content.update("\n".join(lines))
    
    def _show_error(self, message: str) -> None:
        """Show an error message."""
        answer_panel = self.query_one("#answer", AnswerPanel)
        answer_panel.set_answer(f"[red]{message}[/red]")
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
    
    def action_new_conversation(self) -> None:
        """Start a new conversation."""
        self.conversation_id = self._rag_app.start_conversation()
        self._update_status_bar()
        self._log(f"New conversation: {self.conversation_id[:8]}...")
        self._clear_results()
    
    def action_focus_query(self) -> None:
        """Focus the query input."""
        self.query_one("#query-input", Input).focus()
    
    def action_save_report(self) -> None:
        """Save the current report."""
        if self.last_result:
            from text_report import generate_text_report
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"./reports/report_{timestamp}.txt"
            
            try:
                report = generate_text_report(
                    self.last_result,
                    retrieval_mode=self.current_mode,
                )
                
                import os
                os.makedirs("./reports", exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(report)
                
                self._log(f"Report saved to: {filepath}")
            except Exception as e:
                self._log(f"Failed to save report: {e}")
        else:
            self._log("No results to save")
    
    def action_cancel_query(self) -> None:
        """Cancel current operation."""
        if self.is_loading:
            self._log("Query cancellation requested (may take a moment)")


def run_tui(rag_app: "RadiantRAG") -> None:
    """
    Run the Textual TUI application.
    
    Args:
        rag_app: The initialized RadiantRAG application instance.
    """
    app = AgenticRAGApp(rag_app)
    app.run()
