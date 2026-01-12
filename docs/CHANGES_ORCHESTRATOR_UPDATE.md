# Radiant RAG - Orchestrator and Metrics Update

## Summary

This update completes the BaseAgent architecture migration by:

1. **Updating the orchestrator** to use `AgentResult` wrappers for all refactored agents
2. **Adding metrics export** for Prometheus and OpenTelemetry
3. **Creating comprehensive documentation** for the base agent architecture
4. **Adding integration tests** for the base agent lifecycle
5. **Maintaining backward compatibility** for non-refactored agents

## Files Changed

### Core Updates

| File | Description |
|------|-------------|
| `radiant/orchestrator.py` | Updated all agent calls to handle `AgentResult`, added metrics collector |
| `radiant/agents/base_agent.py` | Added `execute()` helper method for backward compatibility |
| `radiant/utils/metrics_export.py` | **NEW** - Prometheus and OpenTelemetry exporters |

### Tests

| File | Description |
|------|-------------|
| `tests/test_base_agent_lifecycle.py` | **NEW** - 33 integration tests for agent lifecycle |

### Documentation

| File | Description |
|------|-------------|
| `docs/AGENT_ARCHITECTURE.md` | **NEW** - Comprehensive architecture documentation |

## Orchestrator Changes

### New Helper Function

```python
def _extract_agent_data(
    result: AgentResult[T],
    default: T,
    agent_name: str = "Agent",
    metrics_collector: Optional[MetricsCollector] = None,
) -> T:
    """Extract data from AgentResult with fallback and metrics recording."""
```

### Agent Call Pattern Changes

**Before:**
```python
ctx.plan = self._planning_agent.run(ctx.original_query)
```

**After:**
```python
result = self._planning_agent.run(
    correlation_id=ctx.run_id,
    query=ctx.original_query,
)
ctx.plan = _extract_agent_data(
    result,
    default=self._default_plan(),
    agent_name="PlanningAgent",
    metrics_collector=self._metrics_collector,
)
```

### Metrics Collector Integration

The orchestrator now accepts an optional `MetricsCollector`:

```python
orchestrator = RAGOrchestrator(
    config=config,
    llm=llm,
    local=local,
    store=store,
    bm25_index=bm25_index,
    metrics_collector=metrics_collector,  # NEW
)
```

## Metrics Export

### Prometheus Integration

```python
from radiant.utils.metrics_export import PrometheusMetricsExporter

exporter = PrometheusMetricsExporter(namespace="radiant")

# Record each agent execution
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

# Create traces and record metrics
with exporter.trace_agent(agent, query="test"):
    result = agent.run(query="test")
    exporter.record_result(result)
```

### Unified MetricsCollector

```python
from radiant.utils.metrics_export import MetricsCollector

collector = MetricsCollector.create(
    prometheus_enabled=True,
    otel_enabled=True,
    otel_endpoint="http://localhost:4317"
)

# Use with orchestrator
orchestrator = RAGOrchestrator(..., metrics_collector=collector)
```

## BaseAgent Changes

### New `execute()` Method

For backward compatibility, agents now have an `execute()` method that returns raw data:

```python
# New pattern (with AgentResult)
result = agent.run(query="test")
if result.success:
    data = result.data

# Backward compatible (raw data, raises on failure)
data = agent.execute(query="test")
```

## Non-Refactored Agents

The following agents **have not been migrated** but remain fully functional:

| Agent | Status |
|-------|--------|
| `SummarizationAgent` | Uses original interface |
| `ContextEvaluationAgent` | Uses original interface |
| `FactVerificationAgent` | Uses original interface |
| `CitationTrackingAgent` | Uses original interface |
| `IntelligentChunkingAgent` | Uses original interface |
| `LanguageDetectionAgent` | Uses original interface |
| `TranslationAgent` | Uses original interface |
| `RetrievalStrategyMemory` | Uses original interface |

The orchestrator handles these agents appropriately by calling their original methods.

## Test Results

```
======================== 30 passed, 3 skipped in 0.18s =========================
```

All core tests pass. The 3 skipped tests are for metrics export integration which require optional dependencies (redis, prometheus_client, opentelemetry).

## Migration Notes

### For Application Developers

1. **No changes required** if using the orchestrator - it handles all agent calls internally
2. **Optional**: Configure metrics collection for monitoring

### For Agent Developers

1. New agents should inherit from `BaseAgent`, `LLMAgent`, or `RetrievalAgent`
2. Use the `AgentResult` wrapper for consistent return values
3. See `docs/AGENT_ARCHITECTURE.md` for detailed examples

### For DevOps/SRE

1. Install optional dependencies for metrics:
   - `pip install prometheus_client` for Prometheus
   - `pip install opentelemetry-api opentelemetry-sdk` for OpenTelemetry

2. Configure metrics endpoint in your web framework:
   ```python
   @app.route("/metrics")
   def metrics():
       from radiant.utils.metrics_export import get_metrics_collector
       return get_metrics_collector().prometheus_output()
   ```

## Dependencies

### Optional (for metrics)

```
prometheus_client>=0.16.0
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp>=1.20.0
```
