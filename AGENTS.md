# AGENTS.md - Radiant RAG

A production-grade Retrieval-Augmented Generation system with agentic orchestration.

## Project Overview

Radiant RAG is a modular RAG pipeline with 20+ specialized agents coordinated by an orchestrator. The system uses a BaseAgent architecture with lifecycle hooks, automatic metrics collection, and structured logging.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Type checking
python -m mypy radiant/ --ignore-missing-imports

# Lint
python -m pyflakes radiant/
```

## Architecture Summary

```
radiant/
├── agents/           # 20+ pipeline agents (see radiant/agents/AGENTS.md)
├── orchestrator.py   # Pipeline coordinator
├── config.py         # Configuration dataclasses
├── ingestion/        # Document processing
├── storage/          # Vector stores (Redis, Chroma, PgVector)
├── llm/              # LLM client abstraction
├── ui/               # Streamlit web interface
└── utils/            # Metrics, conversation, helpers
```

## Key Patterns

### Agent Results
All refactored agents return `AgentResult` wrappers:
```python
result = agent.run(correlation_id=ctx.run_id, query=query)
if result.success:
    data = result.data
    print(f"Duration: {result.metrics.duration_ms}ms")
```

### Backward Compatibility
Use `execute()` for raw data (raises on failure):
```python
data = agent.execute(query=query)
```

### Parameter Names Matter
When calling `agent.run()`, parameter names MUST match the agent's `_execute()` signature:
```python
# Check the agent's _execute() method for correct parameter names
# Example: RRFAgent expects 'runs', not 'retrieval_lists'
result = self._rrf_agent.run(runs=retrieval_lists)  # ✓ Correct
result = self._rrf_agent.run(retrieval_lists=data)  # ✗ Wrong
```

## Testing Requirements

1. **Before committing**: Run `pytest tests/ -v`
2. **New agents**: Add tests to `tests/test_base_agent_lifecycle.py`
3. **Orchestrator changes**: Verify parameter names match agent signatures
4. **Syntax check**: `python -m py_compile radiant/orchestrator.py`

## Common Tasks

### Adding a New Agent
See `radiant/agents/AGENTS.md` for detailed instructions.

### Modifying the Orchestrator
1. Check the agent's `_execute()` signature in `radiant/agents/<agent>.py`
2. Use matching parameter names in `.run()` calls
3. Extract data using `_extract_agent_data()` helper
4. Always pass `correlation_id=ctx.run_id`

### Running Specific Tests
```bash
pytest tests/test_base_agent_lifecycle.py -v
pytest tests/test_agents/ -v
pytest -k "test_successful_execution" -v
```

## Code Style

- Type hints required for all public methods
- Docstrings for classes and public methods
- Use `Optional[]` for nullable parameters
- Prefer keyword arguments for clarity
- No unused imports (enforced by pyflakes)

## File Locations

| Purpose | Location |
|---------|----------|
| Agent base classes | `radiant/agents/base_agent.py` |
| Pipeline orchestrator | `radiant/orchestrator.py` |
| Configuration | `radiant/config.py` |
| Metrics export | `radiant/utils/metrics_export.py` |
| Agent tests | `tests/test_base_agent_lifecycle.py` |
| Architecture docs | `docs/AGENT_ARCHITECTURE.md` |

## Non-Refactored Agents

These agents use their original interface (not AgentResult):
- `SummarizationAgent` - Multiple entry points
- `ContextEvaluationAgent` - Dual evaluation modes  
- `FactVerificationAgent` - Multi-stage pipeline
- `CitationTrackingAgent` - Complex output structure
- `IntelligentChunkingAgent` - Ingestion-time agent
- `LanguageDetectionAgent` - Utility agent
- `TranslationAgent` - External service dependency
- `RetrievalStrategyMemory` - Stateful component

When modifying these, check their actual method signatures.

## Debugging Tips

1. **Agent parameter errors**: Check `_execute()` signature in agent file
2. **Import errors**: May need `pip install redis` or other optional deps
3. **Test failures**: Run with `-v --tb=long` for full traceback
4. **Metrics not recording**: Ensure `MetricsCollector` is configured
