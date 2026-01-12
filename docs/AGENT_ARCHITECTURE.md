# Radiant RAG Agent Architecture

This document describes the base agent architecture used throughout the Radiant RAG pipeline.

## Overview

All agents in the Radiant RAG system inherit from a hierarchy of base classes that provide:

- **Consistent Interface**: All agents have `name`, `category`, and `description` properties
- **Lifecycle Hooks**: `_before_execute()`, `_after_execute()`, `_on_error()` 
- **Automatic Metrics**: Duration, success/failure counts, custom metrics
- **Structured Logging**: Correlation IDs, context fields
- **Error Recovery**: Built-in fallback via `_on_error()`
- **Statistics Tracking**: `get_statistics()` for monitoring

## Base Class Hierarchy

```
BaseAgent (Abstract Base)
├── LLMAgent (For agents that use LLM)
│   ├── PlanningAgent
│   ├── AnswerSynthesisAgent
│   ├── CriticAgent
│   ├── QueryDecompositionAgent
│   ├── QueryRewriteAgent
│   ├── QueryExpansionAgent
│   ├── WebSearchAgent
│   └── MultiHopReasoningAgent
│
└── RetrievalAgent (For agents that perform retrieval)
    └── DenseRetrievalAgent

BaseAgent (Direct inheritance)
├── BM25RetrievalAgent
├── RRFAgent
├── HierarchicalAutoMergingAgent
├── CrossEncoderRerankingAgent
└── Other utility agents
```

## Core Classes

### AgentResult

Generic wrapper for agent execution results.

```python
from radiant.agents import AgentResult

@dataclass
class AgentResult(Generic[T]):
    data: T                              # The actual result data
    success: bool = True                 # Whether execution succeeded
    status: AgentStatus = SUCCESS        # Detailed status
    error: Optional[str] = None          # Error message if failed
    warnings: List[str]                  # Warning messages
    metrics: Optional[AgentMetrics]      # Execution metrics
```

### AgentMetrics

Metrics collected during agent execution.

```python
from radiant.agents import AgentMetrics

@dataclass
class AgentMetrics:
    agent_name: str
    agent_category: str
    run_id: str
    correlation_id: str
    start_time: float
    end_time: float
    duration_ms: float
    status: AgentStatus
    error_message: Optional[str]
    items_processed: int
    items_returned: int
    llm_calls: int
    retrieval_calls: int
    confidence: Optional[float]
    custom: Dict[str, Any]
```

### AgentCategory

Categories for classifying agents.

```python
from radiant.agents import AgentCategory

class AgentCategory(Enum):
    PLANNING = "planning"
    QUERY_PROCESSING = "query_processing"
    RETRIEVAL = "retrieval"
    POST_RETRIEVAL = "post_retrieval"
    GENERATION = "generation"
    EVALUATION = "evaluation"
    UTILITY = "utility"
```

### AgentStatus

Execution status values.

```python
from radiant.agents import AgentStatus

class AgentStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
```

## Creating a New Agent

### Step 1: Choose Base Class

- `BaseAgent`: General purpose agents
- `LLMAgent`: Agents that make LLM calls
- `RetrievalAgent`: Agents that retrieve documents

### Step 2: Implement Required Properties

```python
from radiant.agents import BaseAgent, AgentCategory

class MyCustomAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "MyCustomAgent"
    
    @property
    def category(self) -> AgentCategory:
        return AgentCategory.UTILITY
    
    @property
    def description(self) -> str:
        return "Description of what this agent does"
```

### Step 3: Implement `_execute()` Method

```python
def _execute(self, query: str, **kwargs) -> Any:
    """Core execution logic."""
    # Your implementation here
    return result
```

### Step 4: (Optional) Implement Lifecycle Hooks

```python
def _before_execute(self, **kwargs) -> None:
    """Called before _execute()."""
    pass

def _after_execute(self, result: Any, metrics: AgentMetrics, **kwargs) -> Any:
    """Called after successful _execute(). Can modify result."""
    return result

def _on_error(self, error: Exception, metrics: AgentMetrics, **kwargs) -> Optional[Any]:
    """Called on error. Return fallback value or None to propagate error."""
    return None  # Or return a fallback value
```

## Using Agents

### Standard Pattern (with AgentResult)

```python
from radiant.agents import PlanningAgent

agent = PlanningAgent(llm, config)

# Get full result with metrics
result = agent.run(query="user question")

if result.success:
    plan = result.data
    print(f"Duration: {result.metrics.duration_ms}ms")
else:
    print(f"Error: {result.error}")
```

### Backward-Compatible Pattern (raw data)

```python
# Use execute() for raw data (raises on failure)
plan = agent.execute(query="user question")
```

### With Correlation ID (for tracing)

```python
result = agent.run(
    correlation_id="request-123",
    query="user question"
)
```

## Metrics Export

### Prometheus Integration

```python
from radiant.utils.metrics_export import PrometheusMetricsExporter

exporter = PrometheusMetricsExporter(namespace="radiant")

# Record each execution
result = agent.run(query="test")
exporter.record_execution(result)

# Get metrics for /metrics endpoint
output = exporter.get_metrics_output()
```

### OpenTelemetry Integration

```python
from radiant.utils.metrics_export import OpenTelemetryExporter

exporter = OpenTelemetryExporter(
    service_name="radiant-rag",
    endpoint="http://localhost:4317"
)

# Create traces
with exporter.trace_agent(agent, query="test"):
    result = agent.run(query="test")
    exporter.record_result(result)
```

### Unified Metrics Collector

```python
from radiant.utils.metrics_export import MetricsCollector

collector = MetricsCollector.create(
    prometheus_enabled=True,
    otel_enabled=True,
    otel_endpoint="http://localhost:4317"
)

# Register agents
collector.register_agent(planning_agent)
collector.register_agent(synthesis_agent)

# Record executions
result = agent.run(query="test")
collector.record(result)

# Get Prometheus output
print(collector.prometheus_output())
```

## Agent Statistics

```python
stats = agent.get_statistics()

# Returns:
{
    "name": "PlanningAgent",
    "category": "planning",
    "enabled": True,
    "total_executions": 100,
    "total_successes": 95,
    "total_failures": 5,
    "success_rate": 0.95,
    "average_duration_ms": 123.45,
    "total_duration_ms": 12345.0
}
```

## Refactored Agents

The following agents have been fully migrated to the BaseAgent architecture:

### LLM-Based Agents (inherit from `LLMAgent`)

| Agent | Category | Description |
|-------|----------|-------------|
| `PlanningAgent` | PLANNING | Query analysis and execution planning |
| `AnswerSynthesisAgent` | GENERATION | Answer generation from context |
| `CriticAgent` | EVALUATION | Answer quality evaluation |
| `QueryDecompositionAgent` | QUERY_PROCESSING | Complex query breakdown |
| `QueryRewriteAgent` | QUERY_PROCESSING | Query transformation |
| `QueryExpansionAgent` | QUERY_PROCESSING | Synonym and term expansion |
| `WebSearchAgent` | RETRIEVAL | Web content augmentation |
| `MultiHopReasoningAgent` | EVALUATION | Multi-step reasoning chains |

### Retrieval-Based Agents (inherit from `RetrievalAgent`)

| Agent | Category | Description |
|-------|----------|-------------|
| `DenseRetrievalAgent` | RETRIEVAL | Vector similarity search |

### General Agents (inherit from `BaseAgent`)

| Agent | Category | Description |
|-------|----------|-------------|
| `BM25RetrievalAgent` | RETRIEVAL | Sparse keyword retrieval |
| `RRFAgent` | POST_RETRIEVAL | Reciprocal Rank Fusion |
| `HierarchicalAutoMergingAgent` | POST_RETRIEVAL | Child-to-parent chunk merging |
| `CrossEncoderRerankingAgent` | POST_RETRIEVAL | Cross-encoder reranking |

## Non-Refactored Agents

These agents have not been migrated due to complexity but remain fully functional:

| Agent | Reason |
|-------|--------|
| `SummarizationAgent` | Multiple public entry points |
| `ContextEvaluationAgent` | Dual evaluation modes |
| `FactVerificationAgent` | Multi-stage internal pipeline |
| `CitationTrackingAgent` | Complex output structure |
| `IntelligentChunkingAgent` | Ingestion-time agent |
| `LanguageDetectionAgent` | Utility agent |
| `TranslationAgent` | External service dependency |
| `RetrievalStrategyMemory` | Stateful learning component |

These agents continue to use their original interface and the orchestrator handles them appropriately.

## Error Handling Best Practices

### In Agent Implementation

```python
def _execute(self, **kwargs) -> Any:
    try:
        result = self._do_work(**kwargs)
        return result
    except SpecificError as e:
        # Handle known errors
        self._log.warning("Known error occurred", error=str(e))
        raise  # Re-raise to trigger _on_error

def _on_error(self, error: Exception, metrics: AgentMetrics, **kwargs) -> Optional[Any]:
    # Return fallback value
    if isinstance(error, SpecificError):
        return self._default_result()
    return None  # Propagate other errors
```

### In Orchestrator

```python
result = agent.run(query=query)

if result.success:
    data = result.data
else:
    # Use fallback or handle error
    data = default_value
    logger.warning(f"Agent failed: {result.error}")
```

## Migration Guide

### Updating Agent Callers

**Before:**
```python
result = agent.run(query)
if result.get("success"):
    data = result["data"]
```

**After (Full AgentResult):**
```python
result = agent.run(query=query)
if result.success:
    data = result.data
    print(f"Duration: {result.metrics.duration_ms}ms")
```

**After (Backward Compatible):**
```python
data = agent.execute(query=query)  # Raises on failure
```

### Updating Metrics Collection

```python
from radiant.utils.metrics_export import configure_metrics, record_agent_execution

# Configure once at startup
configure_metrics(prometheus_enabled=True)

# Record each execution
result = agent.run(query="test")
record_agent_execution(result)
```

## Testing Agents

```python
import pytest
from radiant.agents import AgentResult, AgentStatus

def test_my_agent_success():
    agent = MyAgent(config)
    
    result = agent.run(query="test")
    
    assert result.success is True
    assert result.status == AgentStatus.SUCCESS
    assert result.data is not None
    assert result.metrics.duration_ms > 0

def test_my_agent_failure_recovery():
    agent = MyAgent(config, should_fail=True)
    
    result = agent.run(query="test")
    
    assert result.success is False
    assert result.data == expected_fallback  # If _on_error returns fallback
```

## Configuration

### Enabling/Disabling Agents

```python
agent = MyAgent(config, enabled=False)
result = agent.run()  # Returns SKIPPED status
```

### Metrics Configuration

```yaml
# config.yaml
metrics:
  prometheus:
    enabled: true
    namespace: "radiant"
  opentelemetry:
    enabled: false
    endpoint: "http://localhost:4317"
```

## Performance Considerations

1. **Metrics Overhead**: Metrics collection adds ~0.1ms per call
2. **Logging**: Use appropriate log levels (DEBUG for verbose)
3. **Error Recovery**: `_on_error` should be fast (avoid heavy computation)
4. **Statistics**: Call `get_statistics()` periodically, not per-request

## Troubleshooting

### Agent Returns SKIPPED Status

Check if the agent is enabled:
```python
if not agent.enabled:
    agent._enabled = True  # Or configure properly
```

### Metrics Not Recording

Ensure metrics collector is configured:
```python
from radiant.utils.metrics_export import configure_metrics
configure_metrics(prometheus_enabled=True)
```

### Correlation ID Not Propagating

Pass correlation ID explicitly:
```python
result = agent.run(correlation_id=ctx.run_id, query=query)
```
