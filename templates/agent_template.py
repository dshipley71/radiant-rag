"""
================================================================================
AGENT TEMPLATE FOR RADIANT RAG ARCHITECTURE
================================================================================

This file provides a comprehensive template for integrating new agents into the
Radiant RAG system. It includes multiple agent patterns, configuration classes,
and step-by-step integration instructions.

================================================================================
INTEGRATION CHECKLIST
================================================================================

Follow these steps to integrate a new agent:

1. DEFINE YOUR AGENT CLASS
   - Copy the appropriate template pattern below (LLM-based, Retrieval-based, 
     Tool-based, or Complex agent)
   - Implement all required methods
   - Add proper error handling and logging

2. CREATE CONFIGURATION DATACLASS (radiant/config.py)
   Add a frozen dataclass for your agent's configuration:
   
   ```python
   @dataclass(frozen=True)
   class YourAgentConfig:
       '''Configuration for YourAgent.'''
       enabled: bool = True
       # Add your config parameters here
       param1: int = 10
       param2: float = 0.5
   ```

3. UPDATE AppConfig (radiant/config.py)
   Add your config to the AppConfig dataclass:
   
   ```python
   @dataclass(frozen=True)
   class AppConfig:
       # ... existing fields ...
       your_agent: YourAgentConfig = field(default_factory=YourAgentConfig)
   ```

4. UPDATE load_config() FUNCTION (radiant/config.py)
   Add parsing logic in the load_config() function:
   
   ```python
   your_agent = YourAgentConfig(
       enabled=_get_config_value(data, "your_agent", "enabled", True, _parse_bool),
       param1=_get_config_value(data, "your_agent", "param1", 10, _parse_int),
       param2=_get_config_value(data, "your_agent", "param2", 0.5, _parse_float),
   )
   ```

5. ADD YAML CONFIGURATION (config.yaml)
   Add a section for your agent in the config file:
   
   ```yaml
   your_agent:
     enabled: true
     param1: 10
     param2: 0.5
   ```

6. UPDATE AGENTS __init__.py (radiant/agents/__init__.py)
   Export your agent class:
   
   ```python
   from radiant.agents.your_agent import (
       YourAgent,
       YourAgentResult,  # if you have result dataclasses
   )
   
   __all__ = [
       # ... existing exports ...
       "YourAgent",
       "YourAgentResult",
   ]
   ```

7. INTEGRATE INTO ORCHESTRATOR (radiant/orchestrator.py)
   
   a) Import your agent at the top:
      ```python
      from radiant.agents import YourAgent, YourAgentResult
      ```
   
   b) Initialize in __init__:
      ```python
      self._your_agent: Optional[YourAgent] = None
      if config.your_agent.enabled:
          self._your_agent = YourAgent(
              llm=llm,
              # pass other dependencies
              param1=config.your_agent.param1,
          )
      ```
   
   c) Create a runner method:
      ```python
      def _run_your_agent(
          self,
          ctx: AgentContext,
          metrics: RunMetrics,
          # additional parameters
      ) -> Optional[YourAgentResult]:
          '''Execute your agent.'''
          if not self._your_agent:
              return None
          
          with metrics.track_step("YourAgent") as step:
              try:
                  result = self._your_agent.run(
                      query=ctx.original_query,
                      # pass other parameters
                  )
                  step.extra["key_metric"] = result.some_metric
                  return result
              except Exception as e:
                  logger.warning(f"YourAgent failed: {e}")
                  metrics.mark_degraded("your_agent", str(e))
                  return None
      ```
   
   d) Call from the main run() method at the appropriate pipeline phase

8. UPDATE AgentContext IF NEEDED (radiant/agents/base.py)
   If your agent produces data that should persist through the pipeline,
   add fields to AgentContext:
   
   ```python
   @dataclass
   class AgentContext:
       # ... existing fields ...
       your_agent_results: List[Any] = field(default_factory=list)
       your_agent_score: float = 0.0
   ```

9. UPDATE PipelineResult IF NEEDED (radiant/orchestrator.py)
   If your agent's results should be included in the final output:
   
   ```python
   @dataclass
   class PipelineResult:
       # ... existing fields ...
       your_agent_used: bool = False
       your_agent_result: Optional[Dict[str, Any]] = None
   ```

10. WRITE TESTS (tests/test_agents/)
    Create test file tests/test_agents/test_your_agent.py

================================================================================
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from radiant.config import AppConfig  # Your config class
    from radiant.llm.client import LLMClient, LocalNLPModels
    from radiant.storage.base import BaseVectorStore, StoredDoc
    from radiant.utils.conversation import ConversationManager

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASSES
# =============================================================================
# Define dataclasses for your agent's output. This provides type safety
# and makes it easy to serialize results.

@dataclass
class TemplateAgentResult:
    """
    Result from the template agent.
    
    Define all output fields your agent produces.
    """
    
    # Primary output
    output: str
    
    # Success/failure indicators
    success: bool
    error: Optional[str] = None
    
    # Metrics and scores
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    
    # Additional structured data
    items: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "output": self.output,
            "success": self.success,
            "error": self.error,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "items": self.items,
            "metadata": self.metadata,
        }


@dataclass
class ProcessingStep:
    """
    Represents a single step in multi-step processing.
    
    Useful for agents that perform iterative operations.
    """
    
    step_number: int
    description: str
    input_data: Any
    output_data: Any
    success: bool
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "description": self.description,
            "success": self.success,
            "duration_ms": self.duration_ms,
        }


# =============================================================================
# PATTERN 1: LLM-BASED AGENT
# =============================================================================
# Use this pattern when your agent primarily uses LLM for processing.
# Examples: PlanningAgent, QueryRewriteAgent, CriticAgent

class LLMBasedAgent:
    """
    Template for LLM-based agents.
    
    This pattern is used when the agent's primary function involves
    prompting an LLM and processing its response.
    
    Example use cases:
    - Query planning and decomposition
    - Answer synthesis and critique
    - Classification and categorization
    - Text transformation and rewriting
    """
    
    def __init__(
        self,
        llm: "LLMClient",
        # Add configuration parameters
        param1: int = 10,
        param2: float = 0.5,
        param3: str = "default",
    ) -> None:
        """
        Initialize the LLM-based agent.
        
        Args:
            llm: LLM client for chat completions
            param1: Description of parameter 1
            param2: Description of parameter 2
            param3: Description of parameter 3
        """
        self._llm = llm
        self._param1 = param1
        self._param2 = param2
        self._param3 = param3
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def run(
        self,
        query: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> TemplateAgentResult:
        """
        Execute the agent's main function.
        
        Args:
            query: User query or input text
            context: Optional additional context
            **kwargs: Additional arguments
            
        Returns:
            TemplateAgentResult with processing results
        """
        import time
        start_time = time.time()
        
        try:
            # Build the system prompt
            system = self._build_system_prompt(context)
            
            # Build the user prompt
            user = self._build_user_prompt(query, kwargs)
            
            # Call LLM for JSON response
            result, response = self._llm.chat_json(
                system=system,
                user=user,
                default=self._default_result(),
                expected_type=dict,
            )
            
            if not response.success:
                return TemplateAgentResult(
                    output="",
                    success=False,
                    error=response.error,
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
            
            # Process the LLM response
            processed = self._process_response(result, query)
            
            processing_time = (time.time() - start_time) * 1000
            
            return TemplateAgentResult(
                output=processed.get("output", ""),
                success=True,
                confidence=float(processed.get("confidence", 0.5)),
                items=processed.get("items", []),
                metadata=processed.get("metadata", {}),
                processing_time_ms=processing_time,
            )
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__} failed: {e}")
            return TemplateAgentResult(
                output="",
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    def _build_system_prompt(self, context: Optional[str] = None) -> str:
        """
        Build the system prompt for the LLM.
        
        Customize this method for your agent's specific instructions.
        """
        base_prompt = """You are a specialized agent in a RAG pipeline.
Your task is to [describe the task here].

Guidelines:
1. [First guideline]
2. [Second guideline]
3. [Third guideline]

Return a JSON object with the following structure:
{
    "output": "the main output",
    "confidence": 0.0-1.0,
    "items": [...],
    "metadata": {...}
}"""
        
        if context:
            base_prompt += f"\n\nAdditional context:\n{context}"
        
        return base_prompt
    
    def _build_user_prompt(
        self,
        query: str,
        kwargs: Dict[str, Any],
    ) -> str:
        """Build the user prompt for the LLM."""
        user_parts = [f"Query: {query}"]
        
        # Add any additional context from kwargs
        if kwargs.get("additional_info"):
            user_parts.append(f"Additional info: {kwargs['additional_info']}")
        
        user_parts.append("\nReturn JSON only.")
        
        return "\n".join(user_parts)
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default result for failed LLM calls."""
        return {
            "output": "",
            "confidence": 0.0,
            "items": [],
            "metadata": {},
        }
    
    def _process_response(
        self,
        result: Dict[str, Any],
        query: str,
    ) -> Dict[str, Any]:
        """
        Process and validate the LLM response.
        
        Override this method to add custom processing logic.
        """
        # Ensure required fields exist
        processed = dict(result)
        
        if "output" not in processed:
            processed["output"] = ""
        
        if "confidence" not in processed:
            processed["confidence"] = 0.5
        
        # Validate and clamp confidence
        processed["confidence"] = max(0.0, min(1.0, float(processed["confidence"])))
        
        return processed


# =============================================================================
# PATTERN 2: RETRIEVAL-BASED AGENT
# =============================================================================
# Use this pattern when your agent performs vector/keyword retrieval.
# Examples: DenseRetrievalAgent, BM25RetrievalAgent

class RetrievalBasedAgent:
    """
    Template for retrieval-based agents.
    
    This pattern is used when the agent retrieves documents from
    a vector store or search index.
    
    Example use cases:
    - Dense (embedding) retrieval
    - Sparse (BM25/keyword) retrieval
    - Hybrid retrieval
    - Specialized domain search
    """
    
    def __init__(
        self,
        store: "BaseVectorStore",
        local_models: "LocalNLPModels",
        top_k: int = 10,
        min_similarity: float = 0.0,
        search_scope: str = "leaves",
    ) -> None:
        """
        Initialize the retrieval-based agent.
        
        Args:
            store: Vector store for document retrieval
            local_models: Local NLP models (embedding, cross-encoder)
            top_k: Maximum number of documents to retrieve
            min_similarity: Minimum similarity threshold
            search_scope: Document scope ("leaves", "parents", "all")
        """
        self._store = store
        self._local = local_models
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._search_scope = search_scope
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Retrieve documents matching the query.
        
        Args:
            query: Search query
            top_k: Override for number of results
            filters: Optional filters for retrieval
            
        Returns:
            List of (document, score) tuples
        """
        k = top_k or self._top_k
        
        try:
            # Generate query embedding
            query_embedding = self._local.embed_single(query)
            
            # Apply any preprocessing to the embedding
            query_embedding = self._preprocess_embedding(query_embedding)
            
            # Determine doc level filter
            doc_level_filter = self._get_doc_level_filter()
            
            # Retrieve from vector store
            results = self._store.retrieve_by_embedding(
                query_embedding=query_embedding,
                top_k=k,
                min_similarity=self._min_similarity,
                doc_level_filter=doc_level_filter,
            )
            
            # Apply any post-processing
            results = self._postprocess_results(results, query, filters)
            
            logger.debug(
                f"{self.__class__.__name__} retrieved {len(results)} documents "
                f"for query: {query[:50]}..."
            )
            
            return results
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__} retrieval failed: {e}")
            return []
    
    def _get_doc_level_filter(self) -> Optional[str]:
        """Convert search scope to doc level filter."""
        if self._search_scope == "leaves":
            return "child"
        elif self._search_scope == "parents":
            return "parent"
        elif self._search_scope == "all":
            return None
        else:
            return "child"  # Default
    
    def _preprocess_embedding(
        self,
        embedding: List[float],
    ) -> List[float]:
        """
        Preprocess the query embedding.
        
        Override this method to add custom preprocessing.
        """
        return embedding
    
    def _postprocess_results(
        self,
        results: List[Tuple[Any, float]],
        query: str,
        filters: Optional[Dict[str, Any]],
    ) -> List[Tuple[Any, float]]:
        """
        Postprocess retrieval results.
        
        Override this method to add filtering, boosting, etc.
        """
        if not filters:
            return results
        
        # Example: Filter by metadata
        filtered = []
        for doc, score in results:
            # Check if document matches filters
            if self._matches_filters(doc, filters):
                filtered.append((doc, score))
        
        return filtered
    
    def _matches_filters(
        self,
        doc: Any,
        filters: Dict[str, Any],
    ) -> bool:
        """Check if a document matches the given filters."""
        # Implement your filtering logic here
        return True


# =============================================================================
# PATTERN 3: TOOL-BASED AGENT
# =============================================================================
# Use this pattern when creating a tool that can be invoked by the pipeline.
# Examples: CalculatorTool, CodeExecutionTool

class ToolType(Enum):
    """Types of tools available to agents."""
    RETRIEVAL = "retrieval"
    WEB_SEARCH = "web_search"
    CALCULATOR = "calculator"
    CODE_EXECUTION = "code_execution"
    CUSTOM = "custom"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: Any
    tool_name: str
    tool_type: ToolType
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": str(self.output)[:1000] if self.output else None,
            "tool_name": self.tool_name,
            "tool_type": self.tool_type.value,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Inherit from this class to create custom tools.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        tool_type: ToolType,
    ) -> None:
        """
        Initialize the tool.
        
        Args:
            name: Unique tool name
            description: Human-readable description for LLM context
            tool_type: Type of tool
        """
        self.name = name
        self.description = description
        self.tool_type = tool_type
        self._enabled = True
    
    @property
    def enabled(self) -> bool:
        """Check if tool is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the tool."""
        self._enabled = value
    
    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with given arguments.
        
        Must be implemented by subclasses.
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool metadata to dictionary for LLM context."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.tool_type.value,
            "enabled": self.enabled,
        }


class TemplateCustomTool(BaseTool):
    """
    Example custom tool implementation.
    
    Replace this with your actual tool logic.
    """
    
    def __init__(
        self,
        # Add tool-specific configuration
        setting1: str = "default",
        setting2: int = 100,
    ) -> None:
        """Initialize the custom tool."""
        super().__init__(
            name="custom_tool",
            description="A custom tool that does [describe functionality]",
            tool_type=ToolType.CUSTOM,
        )
        self._setting1 = setting1
        self._setting2 = setting2
    
    def execute(
        self,
        input_data: str,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute the custom tool.
        
        Args:
            input_data: Primary input to process
            **kwargs: Additional arguments
            
        Returns:
            ToolResult with execution output
        """
        try:
            # Validate input
            if not input_data:
                return ToolResult(
                    success=False,
                    output=None,
                    tool_name=self.name,
                    tool_type=self.tool_type,
                    error="Empty input provided",
                )
            
            # Process the input (implement your logic here)
            result = self._process(input_data, kwargs)
            
            return ToolResult(
                success=True,
                output=result,
                tool_name=self.name,
                tool_type=self.tool_type,
                metadata={
                    "input_length": len(input_data),
                    "setting1": self._setting1,
                },
            )
            
        except Exception as e:
            logger.warning(f"{self.name} error: {e}")
            return ToolResult(
                success=False,
                output=None,
                tool_name=self.name,
                tool_type=self.tool_type,
                error=str(e),
            )
    
    def _process(
        self,
        input_data: str,
        kwargs: Dict[str, Any],
    ) -> Any:
        """
        Internal processing logic.
        
        Replace with your actual implementation.
        """
        # Example: Simple echo
        return f"Processed: {input_data}"


# =============================================================================
# PATTERN 4: COMPLEX MULTI-STEP AGENT
# =============================================================================
# Use this pattern for agents that combine LLM and retrieval,
# or perform iterative multi-step processing.
# Examples: MultiHopReasoningAgent, FactVerificationAgent

class ComplexMultiStepAgent:
    """
    Template for complex multi-step agents.
    
    This pattern combines LLM reasoning with retrieval and
    performs iterative processing.
    
    Example use cases:
    - Multi-hop reasoning
    - Fact verification with evidence gathering
    - Iterative refinement
    - Chain-of-thought processing
    """
    
    def __init__(
        self,
        llm: "LLMClient",
        store: "BaseVectorStore",
        local_models: "LocalNLPModels",
        max_iterations: int = 3,
        confidence_threshold: float = 0.5,
        docs_per_iteration: int = 5,
    ) -> None:
        """
        Initialize the complex agent.
        
        Args:
            llm: LLM client for reasoning
            store: Vector store for retrieval
            local_models: Local models for embedding
            max_iterations: Maximum processing iterations
            confidence_threshold: Minimum confidence to continue
            docs_per_iteration: Documents to retrieve per iteration
        """
        self._llm = llm
        self._store = store
        self._local = local_models
        self._max_iterations = max_iterations
        self._confidence_threshold = confidence_threshold
        self._docs_per_iteration = docs_per_iteration
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def run(
        self,
        query: str,
        initial_context: Optional[List[Any]] = None,
        force_full_processing: bool = False,
    ) -> TemplateAgentResult:
        """
        Execute the multi-step agent.
        
        Args:
            query: User query
            initial_context: Optional initial retrieved documents
            force_full_processing: Force all iterations even if confident
            
        Returns:
            TemplateAgentResult with processing results
        """
        import time
        start_time = time.time()
        
        try:
            # Phase 1: Analyze if complex processing is needed
            needs_complex, reason = self._analyze_query(query)
            
            if not needs_complex and not force_full_processing:
                logger.debug(f"Query does not need complex processing: {reason}")
                return TemplateAgentResult(
                    output="",
                    success=True,
                    confidence=1.0,
                    metadata={"skipped": True, "reason": reason},
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
            
            # Phase 2: Execute iterative processing
            steps: List[ProcessingStep] = []
            accumulated_context: List[Any] = list(initial_context or [])
            accumulated_knowledge = ""
            
            for iteration in range(self._max_iterations):
                step_start = time.time()
                
                # Generate next query/action
                next_query = self._generate_next_query(
                    query,
                    accumulated_knowledge,
                    iteration,
                )
                
                # Retrieve relevant documents
                new_docs = self._retrieve(next_query)
                accumulated_context.extend(new_docs)
                
                # Extract information
                extracted, confidence = self._extract_information(
                    next_query,
                    new_docs,
                    accumulated_knowledge,
                )
                
                # Record step
                step = ProcessingStep(
                    step_number=iteration + 1,
                    description=next_query,
                    input_data={"docs_retrieved": len(new_docs)},
                    output_data=extracted,
                    success=True,
                    duration_ms=(time.time() - step_start) * 1000,
                )
                steps.append(step)
                
                # Accumulate knowledge
                if extracted:
                    accumulated_knowledge += f" {extracted}"
                
                # Check stopping conditions
                if confidence >= self._confidence_threshold:
                    logger.info(
                        f"Stopping at iteration {iteration + 1} with "
                        f"confidence {confidence:.2f}"
                    )
                    break
            
            # Phase 3: Synthesize final output
            final_output = self._synthesize_output(
                query,
                accumulated_knowledge,
                steps,
            )
            
            return TemplateAgentResult(
                output=final_output,
                success=True,
                confidence=confidence,
                items=[s.to_dict() for s in steps],
                metadata={
                    "iterations": len(steps),
                    "total_docs": len(accumulated_context),
                },
                processing_time_ms=(time.time() - start_time) * 1000,
            )
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__} failed: {e}")
            return TemplateAgentResult(
                output="",
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    def _analyze_query(self, query: str) -> Tuple[bool, str]:
        """
        Analyze if the query requires complex multi-step processing.
        
        Returns:
            Tuple of (needs_complex, reason)
        """
        # Implement your analysis logic
        # Example: Use pattern matching or LLM
        
        system = """Analyze if this query requires multi-step processing.

Return JSON:
{
    "needs_complex": true/false,
    "reason": "explanation"
}"""
        
        user = f"Query: {query}\n\nReturn JSON only."
        
        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={"needs_complex": False, "reason": "Simple query"},
            expected_type=dict,
        )
        
        return (
            bool(result.get("needs_complex", False)),
            str(result.get("reason", "Analysis result"))
        )
    
    def _generate_next_query(
        self,
        original_query: str,
        accumulated_knowledge: str,
        iteration: int,
    ) -> str:
        """Generate the next sub-query for retrieval."""
        if iteration == 0:
            return original_query
        
        system = """Generate a follow-up query to gather more information.

Return JSON:
{
    "next_query": "the follow-up query"
}"""
        
        user = f"""Original: {original_query}
Known so far: {accumulated_knowledge}
Iteration: {iteration + 1}

Return JSON only."""
        
        result, _ = self._llm.chat_json(
            system=system,
            user=user,
            default={"next_query": original_query},
            expected_type=dict,
        )
        
        return str(result.get("next_query", original_query))
    
    def _retrieve(self, query: str) -> List[Any]:
        """Retrieve documents for the current query."""
        try:
            query_embedding = self._local.embed_single(query)
            results = self._store.retrieve_by_embedding(
                query_embedding,
                top_k=self._docs_per_iteration,
            )
            return [doc for doc, score in results]
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return []
    
    def _extract_information(
        self,
        query: str,
        docs: List[Any],
        prior_knowledge: str,
    ) -> Tuple[str, float]:
        """
        Extract information from retrieved documents.
        
        Returns:
            Tuple of (extracted_text, confidence)
        """
        if not docs:
            return "", 0.0
        
        # Format context
        context_parts = []
        for i, doc in enumerate(docs[:5], start=1):
            content = getattr(doc, 'content', str(doc))[:1500]
            context_parts.append(f"[{i}] {content}")
        
        context = "\n\n".join(context_parts)
        
        system = """Extract relevant information from the context.

Return JSON:
{
    "extracted": "relevant information",
    "confidence": 0.0-1.0
}"""
        
        user = f"""Query: {query}
Prior knowledge: {prior_knowledge}

Context:
{context}

Return JSON only."""
        
        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={"extracted": "", "confidence": 0.0},
            expected_type=dict,
        )
        
        return (
            str(result.get("extracted", "")),
            float(result.get("confidence", 0.0))
        )
    
    def _synthesize_output(
        self,
        original_query: str,
        accumulated_knowledge: str,
        steps: List[ProcessingStep],
    ) -> str:
        """Synthesize the final output from accumulated knowledge."""
        system = """Synthesize a final answer from the accumulated knowledge.

Return a clear, concise answer."""
        
        user = f"""Original query: {original_query}

Accumulated knowledge: {accumulated_knowledge}

Processing steps: {len(steps)}

Synthesize the final answer."""
        
        response = self._llm.chat(
            self._llm.create_messages(system, user),
            retry_on_error=True,
        )
        
        if not response.success:
            return accumulated_knowledge
        
        return response.content.strip()


# =============================================================================
# PATTERN 5: REGISTRY-COMPATIBLE AGENT
# =============================================================================
# Use this pattern if you want to register your agent with the global registry.
# This enables dynamic discovery and invocation.

from radiant.agents.registry import (
    AgentRegistry,
    AgentMetadata,
    RegisteredAgent,
    get_global_registry,
    register_agent,
)


# Option A: Use decorator for function-based agents
@register_agent(
    name="TemplateFunction",
    description="A template function-based agent",
    category="template",
    tags=["template", "example"],
)
def template_function_agent(
    query: str,
    llm: "LLMClient",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    A simple function-based agent.
    
    This can be registered with the global registry using the decorator.
    """
    # Implementation
    return {
        "output": f"Processed: {query}",
        "success": True,
    }


# Option B: Register class-based agents programmatically
def register_template_agent(
    registry: AgentRegistry,
    llm: "LLMClient",
    config: Dict[str, Any],
) -> RegisteredAgent:
    """
    Register a template agent with the registry.
    
    Call this during application initialization.
    """
    # Create agent instance
    agent = LLMBasedAgent(
        llm=llm,
        param1=config.get("param1", 10),
        param2=config.get("param2", 0.5),
    )
    
    # Register with registry
    return registry.register_instance(
        instance=agent,
        name="TemplateAgent",
        description="A template class-based agent",
        category="template",
        method_name="run",  # The method to use as callable
        version="1.0.0",
        tags=["template", "example"],
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "output": {"type": "string"},
                "success": {"type": "boolean"},
                "confidence": {"type": "number"},
            },
        },
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate agent configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ["enabled"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    return True


def format_context_for_llm(
    docs: List[Any],
    max_docs: int = 10,
    max_chars_per_doc: int = 2000,
) -> str:
    """
    Format documents as context string for LLM.
    
    Args:
        docs: List of documents
        max_docs: Maximum documents to include
        max_chars_per_doc: Maximum characters per document
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, doc in enumerate(docs[:max_docs], start=1):
        # Get content from document
        if hasattr(doc, 'content'):
            content = doc.content
        else:
            content = str(doc)
        
        # Truncate if needed
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "..."
        
        # Get source if available
        source = ""
        if hasattr(doc, 'meta') and doc.meta:
            source = doc.meta.get("source_path", "")
            if source:
                source = f" (Source: {source})"
        
        context_parts.append(f"[DOC {i}]{source}\n{content}")
    
    return "\n\n".join(context_parts)


def extract_json_from_response(
    response: str,
    default: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract JSON from LLM response text.
    
    Handles markdown code blocks and common formatting issues.
    """
    import json
    
    text = response.strip()
    
    # Try to extract from markdown code block
    import re
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    
    # Try to find JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        text = match.group(0)
    
    # Clean common issues
    text = re.sub(r",\s*([}\]])", r"\1", text)  # Remove trailing commas
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example of how to use the template agents.
    
    Note: This requires actual LLM and storage instances to run.
    """
    print("Agent Template Module")
    print("=" * 60)
    print()
    print("This module provides templates for creating new agents.")
    print("See the module docstring for integration instructions.")
    print()
    print("Available patterns:")
    print("  1. LLMBasedAgent - For LLM-driven processing")
    print("  2. RetrievalBasedAgent - For vector/keyword search")
    print("  3. BaseTool - For custom tool implementations")
    print("  4. ComplexMultiStepAgent - For iterative processing")
    print("  5. Registry-compatible agents - For dynamic discovery")
