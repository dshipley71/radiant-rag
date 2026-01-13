# Step-by-Step Example: How AGENTS.md Works

This document walks through a realistic scenario showing how AI coding assistants use AGENTS.md files to help developers work on Radiant RAG.

---

## Scenario

**Developer Goal**: Add a new `CachingAgent` that caches retrieval results to reduce latency.

**Tools**: VS Code with GitHub Copilot (or Cursor, Claude, Windsurf, etc.)

---

## Step 1: Developer Opens Project and Asks for Help

The developer opens the Radiant RAG project and types in the AI assistant:

```
"Create a new CachingAgent that caches dense retrieval results 
with a TTL of 5 minutes. It should integrate with the orchestrator."
```

---

## Step 2: AI Assistant Discovers AGENTS.md Files

Before generating any code, the AI assistant scans the project:

```
Scanning project structure...
├── AGENTS.md                     ✓ Found (root)
├── radiant/
│   └── agents/
│       └── AGENTS.md             ✓ Found (nested)
```

The AI reads both files to understand project conventions.

---

## Step 3: AI Extracts Key Information from Root AGENTS.md

From `/AGENTS.md`, the AI learns:

```markdown
## Key Patterns

### Agent Results
All refactored agents return `AgentResult` wrappers:
```python
result = agent.run(correlation_id=ctx.run_id, query=query)
if result.success:
    data = result.data
```

### Parameter Names Matter
When calling `agent.run()`, parameter names MUST match the agent's `_execute()` signature

## Testing Requirements
1. Before committing: Run `pytest tests/ -v`
2. New agents: Add tests to `tests/test_base_agent_lifecycle.py`
```

**AI now knows**:
- Must return `AgentResult` wrapper
- Parameter names are critical
- Must add tests before committing

---

## Step 4: AI Extracts Key Information from Nested AGENTS.md

From `/radiant/agents/AGENTS.md`, the AI learns:

```markdown
## Creating a New Agent

### Step 1: Choose Base Class
```python
from radiant.agents.base_agent import BaseAgent, AgentCategory
```

### Step 2: Implement Required Properties
```python
class MyAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "MyAgent"
    
    @property
    def category(self) -> AgentCategory:
        return AgentCategory.UTILITY
```

### Step 3: Implement _execute()
```python
def _execute(
    self,
    query: str,           # Use descriptive parameter names
    **kwargs: Any,
) -> YourReturnType:
```

## Agent Parameter Reference
| Agent | _execute() Parameters |
|-------|----------------------|
| `DenseRetrievalAgent` | `query: str, top_k: Optional[int]` |
```

**AI now knows**:
- Exact base class to inherit from
- Required properties (name, category, description)
- How to structure `_execute()` method
- Parameter conventions from similar agents

---

## Step 5: AI Generates the Agent Code

Based on AGENTS.md context, the AI generates:

```python
# radiant/agents/caching.py
"""
Caching agent for retrieval results.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from radiant.agents.base_agent import BaseAgent, AgentCategory, AgentMetrics


@dataclass
class CacheEntry:
    """A cached retrieval result with TTL."""
    data: List[Tuple[Any, float]]
    created_at: float
    ttl_seconds: float = 300.0  # 5 minutes default
    
    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class CachingAgent(BaseAgent):
    """
    Caches retrieval results to reduce latency on repeated queries.
    
    Wraps a retrieval agent and caches results with configurable TTL.
    """
    
    def __init__(
        self,
        wrapped_agent: BaseAgent,
        ttl_seconds: float = 300.0,
        max_cache_size: int = 1000,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the caching agent.
        
        Args:
            wrapped_agent: The retrieval agent to wrap
            ttl_seconds: Cache TTL in seconds (default 5 minutes)
            max_cache_size: Maximum cache entries
            enabled: Whether caching is enabled
        """
        super().__init__(enabled=enabled)
        self._wrapped_agent = wrapped_agent
        self._ttl_seconds = ttl_seconds
        self._max_cache_size = max_cache_size
        self._cache: Dict[str, CacheEntry] = {}
    
    @property
    def name(self) -> str:
        return "CachingAgent"
    
    @property
    def category(self) -> AgentCategory:
        return AgentCategory.RETRIEVAL
    
    @property
    def description(self) -> str:
        return "Caches retrieval results with TTL to reduce latency"
    
    def _cache_key(self, query: str, top_k: Optional[int] = None) -> str:
        """Generate a cache key from query parameters."""
        key_str = f"{query}:{top_k}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        bypass_cache: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        """
        Execute retrieval with caching.
        
        Args:
            query: Search query
            top_k: Maximum results
            bypass_cache: Skip cache lookup if True
            
        Returns:
            List of (document, score) tuples
        """
        cache_key = self._cache_key(query, top_k)
        
        # Check cache first (unless bypassed)
        if not bypass_cache and cache_key in self._cache:
            entry = self._cache[cache_key]
            if not entry.is_expired:
                self._log.debug("Cache hit", cache_key=cache_key[:8])
                return entry.data
            else:
                # Remove expired entry
                del self._cache[cache_key]
        
        # Cache miss - execute wrapped agent
        self._log.debug("Cache miss, executing wrapped agent", cache_key=cache_key[:8])
        
        result = self._wrapped_agent.run(
            correlation_id=self._log.correlation_id,
            query=query,
            top_k=top_k,
            **kwargs,
        )
        
        if result.success and result.data:
            # Store in cache
            self._cache[cache_key] = CacheEntry(
                data=result.data,
                created_at=time.time(),
                ttl_seconds=self._ttl_seconds,
            )
            
            # Evict old entries if cache is full
            self._evict_if_needed()
        
        return result.data if result.success else []
    
    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max size."""
        if len(self._cache) > self._max_cache_size:
            # Remove oldest 10%
            entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].created_at
            )
            for key, _ in entries[:len(entries) // 10]:
                del self._cache[key]
    
    def _after_execute(
        self,
        result: List[Tuple[Any, float]],
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        """Record cache statistics in metrics."""
        metrics.custom["cache_size"] = len(self._cache)
        metrics.items_returned = len(result)
        return result
    
    def clear_cache(self) -> int:
        """Clear all cached entries. Returns count of cleared entries."""
        count = len(self._cache)
        self._cache.clear()
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        expired = sum(1 for e in self._cache.values() if e.is_expired)
        return {
            "total_entries": len(self._cache),
            "expired_entries": expired,
            "active_entries": len(self._cache) - expired,
            "max_size": self._max_cache_size,
            "ttl_seconds": self._ttl_seconds,
        }
```

**Note how the AI**:
- Used `BaseAgent` as the base class (from AGENTS.md)
- Implemented required properties: `name`, `category`, `description`
- Used `_execute()` with proper signature
- Added `_after_execute()` for metrics (from AGENTS.md examples)
- Used `correlation_id` when calling wrapped agent (from AGENTS.md)
- Returned data in expected format

---

## Step 6: AI Generates Orchestrator Integration

The AI also generates the orchestrator changes:

```python
# In radiant/orchestrator.py __init__:

from radiant.agents.caching import CachingAgent

# After initializing dense retrieval:
self._dense_retrieval = DenseRetrievalAgent(store, local, config.retrieval)

# Wrap with caching if enabled
if config.caching.enabled:
    self._dense_retrieval = CachingAgent(
        wrapped_agent=self._dense_retrieval,
        ttl_seconds=config.caching.ttl_seconds,
        max_cache_size=config.caching.max_cache_size,
    )
```

**Note**: The AI knows to use the decorator pattern because AGENTS.md showed that agents are called via `.run()` with consistent signatures.

---

## Step 7: AI Generates Tests

Following AGENTS.md testing requirements:

```python
# tests/test_agents/test_caching.py

import pytest
import time
from unittest.mock import MagicMock

from radiant.agents.caching import CachingAgent, CacheEntry
from radiant.agents.base_agent import AgentResult, AgentStatus


class TestCachingAgent:
    """Tests for CachingAgent."""
    
    def test_cache_hit(self):
        """Test that cached results are returned."""
        # Create mock wrapped agent
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            data=[("doc1", 0.9), ("doc2", 0.8)],
            success=True,
            status=AgentStatus.SUCCESS,
        )
        
        agent = CachingAgent(wrapped_agent=mock_agent, ttl_seconds=300)
        
        # First call - cache miss
        result1 = agent.run(query="test query")
        assert result1.success
        assert len(result1.data) == 2
        assert mock_agent.run.call_count == 1
        
        # Second call - cache hit
        result2 = agent.run(query="test query")
        assert result2.success
        assert len(result2.data) == 2
        assert mock_agent.run.call_count == 1  # No additional call
    
    def test_cache_expiration(self):
        """Test that expired entries are not returned."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            data=[("doc1", 0.9)],
            success=True,
            status=AgentStatus.SUCCESS,
        )
        
        agent = CachingAgent(wrapped_agent=mock_agent, ttl_seconds=0.1)
        
        # First call
        agent.run(query="test")
        assert mock_agent.run.call_count == 1
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Second call - should miss due to expiration
        agent.run(query="test")
        assert mock_agent.run.call_count == 2
    
    def test_bypass_cache(self):
        """Test cache bypass flag."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            data=[("doc1", 0.9)],
            success=True,
            status=AgentStatus.SUCCESS,
        )
        
        agent = CachingAgent(wrapped_agent=mock_agent)
        
        # First call
        agent.run(query="test")
        
        # Second call with bypass
        agent.run(query="test", bypass_cache=True)
        
        assert mock_agent.run.call_count == 2
    
    def test_cache_stats(self):
        """Test cache statistics."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            data=[("doc1", 0.9)],
            success=True,
            status=AgentStatus.SUCCESS,
        )
        
        agent = CachingAgent(wrapped_agent=mock_agent, ttl_seconds=300)
        
        agent.run(query="query1")
        agent.run(query="query2")
        agent.run(query="query3")
        
        stats = agent.get_cache_stats()
        assert stats["total_entries"] == 3
        assert stats["active_entries"] == 3


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""
    
    def test_not_expired(self):
        entry = CacheEntry(data=[], created_at=time.time(), ttl_seconds=300)
        assert not entry.is_expired
    
    def test_expired(self):
        entry = CacheEntry(data=[], created_at=time.time() - 400, ttl_seconds=300)
        assert entry.is_expired
```

---

## Step 8: AI Provides Verification Commands

Finally, the AI tells the developer:

```
I've created the CachingAgent. To verify everything works:

1. Check syntax:
   python -m py_compile radiant/agents/caching.py

2. Run the new tests:
   pytest tests/test_agents/test_caching.py -v

3. Run all tests to ensure no regressions:
   pytest tests/ -v

4. Check for lint issues:
   python -m pyflakes radiant/agents/caching.py
```

These commands came directly from the root AGENTS.md file.

---

## Summary: What AGENTS.md Provided

| Information | Source | How It Helped |
|-------------|--------|---------------|
| Base class to use | agents/AGENTS.md | Used `BaseAgent` correctly |
| Required properties | agents/AGENTS.md | Implemented name, category, description |
| `_execute()` pattern | agents/AGENTS.md | Correct method signature |
| Parameter naming | agents/AGENTS.md | Used `query`, `top_k` consistently |
| `AgentResult` usage | Root AGENTS.md | Proper result handling |
| Correlation ID | Root AGENTS.md | Added tracing support |
| Testing requirements | Root AGENTS.md | Generated comprehensive tests |
| Verification commands | Root AGENTS.md | Provided exact commands to run |

---

## Without AGENTS.md

If AGENTS.md didn't exist, the AI might have:

- ❌ Used wrong base class or no base class
- ❌ Missed required properties
- ❌ Used wrong parameter names (causing runtime errors)
- ❌ Returned raw data instead of `AgentResult`
- ❌ Forgotten correlation ID for tracing
- ❌ Not known how to test or verify the code
- ❌ Generated code inconsistent with project style

**AGENTS.md turns a 30-minute back-and-forth into a single correct generation.**
