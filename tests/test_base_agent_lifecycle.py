"""
Integration tests for BaseAgent lifecycle.

Tests the complete lifecycle of agents including:
- Initialization and configuration
- Execution with success and failure paths
- Metrics collection
- Error handling and recovery
- Lifecycle hooks
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from radiant.agents.base_agent import (
    AgentCategory,
    AgentStatus,
    AgentMetrics,
    AgentResult,
    StructuredLogger,
    BaseAgent,
    LLMAgent,
    RetrievalAgent,
)


# =============================================================================
# Test Fixtures
# =============================================================================

class SimpleTestAgent(BaseAgent):
    """A simple test agent for basic lifecycle testing."""
    
    def __init__(
        self,
        return_value: Any = "test_result",
        should_fail: bool = False,
        fail_message: str = "Test failure",
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self._return_value = return_value
        self._should_fail = should_fail
        self._fail_message = fail_message
        self._execute_called = False
        self._before_called = False
        self._after_called = False
        self._on_error_called = False
    
    @property
    def name(self) -> str:
        return "SimpleTestAgent"
    
    @property
    def category(self) -> AgentCategory:
        return AgentCategory.UTILITY
    
    @property
    def description(self) -> str:
        return "A simple test agent"
    
    def _execute(self, **kwargs) -> Any:
        self._execute_called = True
        if self._should_fail:
            raise ValueError(self._fail_message)
        return self._return_value
    
    def _before_execute(self, **kwargs) -> None:
        self._before_called = True
    
    def _after_execute(self, result: Any, metrics: AgentMetrics, **kwargs) -> Any:
        self._after_called = True
        return result
    
    def _on_error(self, error: Exception, metrics: AgentMetrics, **kwargs) -> Optional[Any]:
        self._on_error_called = True
        return None


class LLMTestAgent(LLMAgent):
    """Test agent for LLM-based operations."""
    
    def __init__(self, llm: Optional[Any] = None, enabled: bool = True):
        super().__init__(llm=llm, enabled=enabled)
        self._execute_called = False
    
    @property
    def name(self) -> str:
        return "LLMTestAgent"
    
    @property
    def category(self) -> AgentCategory:
        return AgentCategory.QUERY_PROCESSING
    
    @property
    def description(self) -> str:
        return "A test LLM agent"
    
    def _execute(self, query: str = "", **kwargs) -> Dict[str, Any]:
        self._execute_called = True
        return {"processed_query": query.upper(), "success": True}


class RetrievalTestAgent(RetrievalAgent):
    """Test agent for retrieval operations."""
    
    def __init__(
        self,
        store: Optional[Any] = None,
        local_models: Optional[Any] = None,
        enabled: bool = True,
    ):
        super().__init__(store=store, local_models=local_models, enabled=enabled)
        self._execute_called = False
    
    @property
    def name(self) -> str:
        return "RetrievalTestAgent"
    
    @property
    def category(self) -> AgentCategory:
        return AgentCategory.RETRIEVAL
    
    @property
    def description(self) -> str:
        return "A test retrieval agent"
    
    def _execute(self, query: str = "", top_k: int = 5, **kwargs) -> List[tuple]:
        self._execute_called = True
        # Return mock documents
        return [(f"doc_{i}", 0.9 - i * 0.1) for i in range(min(top_k, 3))]


class FallbackTestAgent(BaseAgent):
    """Test agent with fallback on error."""
    
    def __init__(self, fallback_value: Any = "fallback"):
        super().__init__()
        self._fallback_value = fallback_value
    
    @property
    def name(self) -> str:
        return "FallbackTestAgent"
    
    @property
    def category(self) -> AgentCategory:
        return AgentCategory.UTILITY
    
    @property
    def description(self) -> str:
        return "Test agent with fallback"
    
    def _execute(self, **kwargs) -> Any:
        raise RuntimeError("Intentional failure")
    
    def _on_error(self, error: Exception, metrics: AgentMetrics, **kwargs) -> Optional[Any]:
        return self._fallback_value


# =============================================================================
# Test Classes
# =============================================================================

class TestBaseAgentLifecycle:
    """Tests for basic agent lifecycle."""
    
    def test_successful_execution(self):
        """Test successful agent execution."""
        agent = SimpleTestAgent(return_value="success")
        
        result = agent.run()
        
        assert result.success is True
        assert result.status == AgentStatus.SUCCESS
        assert result.data == "success"
        assert result.error is None
        assert agent._execute_called is True
        assert agent._before_called is True
        assert agent._after_called is True
        assert agent._on_error_called is False
    
    def test_failed_execution(self):
        """Test failed agent execution."""
        agent = SimpleTestAgent(should_fail=True, fail_message="Test error")
        
        result = agent.run()
        
        assert result.success is False
        assert result.status == AgentStatus.FAILED
        assert result.data is None
        assert "Test error" in result.error
        assert agent._execute_called is True
        assert agent._before_called is True
        assert agent._after_called is False
        assert agent._on_error_called is True
    
    def test_disabled_agent(self):
        """Test disabled agent returns skipped status."""
        agent = SimpleTestAgent(enabled=False)
        
        result = agent.run()
        
        assert result.success is True
        assert result.status == AgentStatus.SKIPPED
        assert result.data is None
        assert agent._execute_called is False
    
    def test_execute_helper_returns_raw_data(self):
        """Test execute() helper returns raw data."""
        agent = SimpleTestAgent(return_value={"key": "value"})
        
        data = agent.execute()
        
        assert data == {"key": "value"}
    
    def test_execute_helper_raises_on_failure(self):
        """Test execute() helper raises on failure."""
        agent = SimpleTestAgent(should_fail=True)
        
        with pytest.raises(RuntimeError) as exc_info:
            agent.execute()
        
        assert "SimpleTestAgent failed" in str(exc_info.value)
    
    def test_fallback_on_error(self):
        """Test _on_error fallback is used."""
        agent = FallbackTestAgent(fallback_value="fallback_result")
        
        result = agent.run()
        
        # When _on_error returns a value, success is True with PARTIAL status
        assert result.success is True
        assert result.status == AgentStatus.PARTIAL
        assert result.data == "fallback_result"  # Has fallback data
    
    def test_correlation_id_propagation(self):
        """Test correlation ID is propagated."""
        agent = SimpleTestAgent()
        
        result = agent.run(correlation_id="test-corr-123")
        
        assert result.metrics.correlation_id == "test-corr-123"
    
    def test_kwargs_passed_to_execute(self):
        """Test kwargs are passed through to _execute."""
        class KwargsTestAgent(BaseAgent):
            def __init__(self):
                super().__init__()
                self.received_kwargs = {}
            
            @property
            def name(self) -> str:
                return "KwargsTestAgent"
            
            @property
            def category(self) -> AgentCategory:
                return AgentCategory.UTILITY
            
            def _execute(self, **kwargs) -> Dict[str, Any]:
                self.received_kwargs = kwargs
                return kwargs
        
        agent = KwargsTestAgent()
        agent.run(query="test", extra_param=123)
        
        assert agent.received_kwargs["query"] == "test"
        assert agent.received_kwargs["extra_param"] == 123


class TestAgentMetrics:
    """Tests for metrics collection."""
    
    def test_metrics_collected_on_success(self):
        """Test metrics are collected on successful execution."""
        agent = SimpleTestAgent()
        
        result = agent.run()
        
        assert result.metrics is not None
        assert result.metrics.agent_name == "SimpleTestAgent"
        assert result.metrics.agent_category == "utility"
        assert result.metrics.status == AgentStatus.SUCCESS
        assert result.metrics.duration_ms > 0
        assert result.metrics.run_id is not None
    
    def test_metrics_collected_on_failure(self):
        """Test metrics are collected on failed execution."""
        agent = SimpleTestAgent(should_fail=True, fail_message="Expected error")
        
        result = agent.run()
        
        assert result.metrics is not None
        assert result.metrics.status == AgentStatus.FAILED
        assert result.metrics.error_message is not None
        assert "Expected error" in result.metrics.error_message
    
    def test_agent_statistics(self):
        """Test agent statistics tracking."""
        agent = SimpleTestAgent()
        
        # Run multiple times
        agent.run()
        agent.run()
        agent.run()
        
        stats = agent.get_statistics()
        
        assert stats["total_executions"] == 3
        assert stats["total_successes"] == 3
        assert stats["total_failures"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["avg_duration_ms"] > 0
    
    def test_mixed_success_failure_statistics(self):
        """Test statistics with mixed success/failure."""
        success_agent = SimpleTestAgent()
        failure_agent = SimpleTestAgent(should_fail=True)
        
        # Run success agent twice
        success_agent.run()
        success_agent.run()
        
        # Run failure agent once
        failure_agent.run()
        
        success_stats = success_agent.get_statistics()
        failure_stats = failure_agent.get_statistics()
        
        assert success_stats["success_rate"] == 1.0
        assert failure_stats["success_rate"] == 0.0
    
    def test_prometheus_labels(self):
        """Test Prometheus label generation."""
        agent = SimpleTestAgent()
        result = agent.run()
        
        labels = result.metrics.to_prometheus_labels()
        
        assert labels["agent_name"] == "SimpleTestAgent"
        assert labels["agent_category"] == "utility"
        assert labels["status"] == "success"
    
    def test_otel_attributes(self):
        """Test OpenTelemetry attribute generation."""
        agent = SimpleTestAgent()
        result = agent.run()
        
        attrs = result.metrics.to_otel_attributes()
        
        assert attrs["agent.name"] == "SimpleTestAgent"
        assert attrs["agent.category"] == "utility"
        assert attrs["agent.status"] == "success"
        assert "agent.duration_ms" in attrs
        assert "agent.run_id" in attrs


class TestLLMAgent:
    """Tests for LLM agent base class."""
    
    def test_llm_agent_execution(self):
        """Test LLM agent basic execution."""
        mock_llm = MagicMock()
        agent = LLMTestAgent(llm=mock_llm)
        
        result = agent.run(query="test query")
        
        assert result.success is True
        assert result.data["processed_query"] == "TEST QUERY"
        assert agent._execute_called is True
    
    def test_llm_agent_without_llm(self):
        """Test LLM agent requires LLM (raises if None)."""
        with pytest.raises(ValueError) as exc_info:
            LLMTestAgent(llm=None)
        
        assert "requires an LLM client" in str(exc_info.value)


class TestRetrievalAgent:
    """Tests for retrieval agent base class."""
    
    def test_retrieval_agent_execution(self):
        """Test retrieval agent basic execution."""
        mock_store = MagicMock()
        mock_local = MagicMock()
        agent = RetrievalTestAgent(store=mock_store, local_models=mock_local)
        
        result = agent.run(query="search query", top_k=3)
        
        assert result.success is True
        assert len(result.data) == 3
        assert result.data[0] == ("doc_0", 0.9)
    
    def test_retrieval_agent_without_dependencies(self):
        """Test retrieval agent requires store (raises if None)."""
        with pytest.raises(ValueError) as exc_info:
            RetrievalTestAgent(store=None, local_models=None)
        
        assert "requires a vector store" in str(exc_info.value)


class TestAgentResult:
    """Tests for AgentResult wrapper."""
    
    def test_result_to_dict(self):
        """Test AgentResult serialization."""
        result = AgentResult(
            data={"key": "value"},
            success=True,
            status=AgentStatus.SUCCESS,
        )
        
        d = result.to_dict()
        
        assert d["data"] == {"key": "value"}
        assert d["success"] is True
        assert d["status"] == "success"
    
    def test_result_add_warning(self):
        """Test warning addition."""
        result = AgentResult(data="test")
        
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")
        
        assert len(result.warnings) == 2
        assert result.status == AgentStatus.PARTIAL
    
    def test_result_with_dataclass_data(self):
        """Test result with dataclass data."""
        @dataclass
        class TestData:
            value: str
            
            def to_dict(self):
                return {"value": self.value}
        
        result = AgentResult(data=TestData(value="test"))
        d = result.to_dict()
        
        assert d["data"]["value"] == "test"


class TestStructuredLogger:
    """Tests for structured logging."""
    
    def test_logger_creation(self):
        """Test logger initialization."""
        logger = StructuredLogger("test_logger")
        
        assert logger.correlation_id is not None
        assert len(logger.correlation_id) == 8
    
    def test_correlation_id_setting(self):
        """Test correlation ID can be changed."""
        logger = StructuredLogger("test")
        
        logger.set_correlation_id("custom-id")
        
        assert logger.correlation_id == "custom-id"
    
    def test_context_addition(self):
        """Test context field addition."""
        logger = StructuredLogger("test")
        
        logger.add_context(user_id="123", session="abc")
        
        formatted = logger._format_message("Test message")
        assert "user_id=123" in formatted
        assert "session=abc" in formatted
    
    def test_context_clearing(self):
        """Test context can be cleared."""
        logger = StructuredLogger("test")
        logger.add_context(field="value")
        
        logger.clear_context()
        
        formatted = logger._format_message("Test")
        assert "field" not in formatted


class TestAgentCategory:
    """Tests for agent categories."""
    
    def test_all_categories_exist(self):
        """Test all expected categories are defined."""
        expected = [
            "PLANNING",
            "QUERY_PROCESSING",
            "RETRIEVAL",
            "POST_RETRIEVAL",
            "GENERATION",
            "EVALUATION",
            "UTILITY",
        ]
        
        for cat in expected:
            assert hasattr(AgentCategory, cat)
    
    def test_category_values(self):
        """Test category value strings."""
        assert AgentCategory.PLANNING.value == "planning"
        assert AgentCategory.RETRIEVAL.value == "retrieval"


class TestAgentStatus:
    """Tests for agent status enum."""
    
    def test_all_statuses_exist(self):
        """Test all expected statuses are defined."""
        expected = ["SUCCESS", "FAILED", "PARTIAL", "SKIPPED"]
        
        for status in expected:
            assert hasattr(AgentStatus, status)


# =============================================================================
# Integration Tests
# =============================================================================

class TestMetricsExportIntegration:
    """Integration tests for metrics export."""
    
    def test_prometheus_metrics_recording(self):
        """Test Prometheus metrics can be recorded."""
        try:
            from radiant.utils.metrics_export import PrometheusMetricsExporter
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Metrics export dependencies not available")
        
        exporter = PrometheusMetricsExporter(namespace="test")
        agent = SimpleTestAgent()
        
        result = agent.run()
        
        # Should not raise even if prometheus_client not installed
        exporter.record_execution(result)
    
    def test_otel_metrics_recording(self):
        """Test OpenTelemetry metrics can be recorded."""
        try:
            from radiant.utils.metrics_export import OpenTelemetryExporter
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Metrics export dependencies not available")
        
        exporter = OpenTelemetryExporter(service_name="test-service")
        agent = SimpleTestAgent()
        
        result = agent.run()
        
        # Should not raise even if opentelemetry not installed
        exporter.record_result(result)
    
    def test_metrics_collector(self):
        """Test unified metrics collector."""
        try:
            from radiant.utils.metrics_export import MetricsCollector
        except (ImportError, ModuleNotFoundError):
            pytest.skip("Metrics export dependencies not available")
        
        collector = MetricsCollector.create(
            prometheus_enabled=True,
            otel_enabled=False,
        )
        
        agent = SimpleTestAgent()
        collector.register_agent(agent)
        
        result = agent.run()
        collector.record(result)
        
        # Should work without errors
        output = collector.prometheus_output()
        assert isinstance(output, str)


class TestAgentChaining:
    """Tests for chaining multiple agents."""
    
    def test_sequential_execution(self):
        """Test agents can be executed sequentially."""
        agent1 = SimpleTestAgent(return_value="step1")
        agent2 = SimpleTestAgent(return_value="step2")
        
        result1 = agent1.run()
        result2 = agent2.run(input=result1.data)
        
        assert result1.success is True
        assert result2.success is True
        assert result1.data == "step1"
        assert result2.data == "step2"
    
    def test_correlation_id_across_agents(self):
        """Test correlation ID is consistent across agent chain."""
        agent1 = SimpleTestAgent()
        agent2 = SimpleTestAgent()
        
        corr_id = "chain-test-123"
        
        result1 = agent1.run(correlation_id=corr_id)
        result2 = agent2.run(correlation_id=corr_id)
        
        assert result1.metrics.correlation_id == corr_id
        assert result2.metrics.correlation_id == corr_id


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
