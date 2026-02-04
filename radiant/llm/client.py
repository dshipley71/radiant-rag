"""
LLM client with retry logic and robust response handling.

Provides:
    - JSONParser: Robust JSON extraction from LLM responses
    - LLMResponse: Structured response objects
    - LLMClient: Chat client with retries
    - LLMClients: Container for all LLM dependencies
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, TYPE_CHECKING

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from radiant.llm.local_models import LocalNLPModels

if TYPE_CHECKING:
    from radiant.config import LocalModelsConfig, OllamaConfig, ParsingConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that indicate a non-retryable client error.
# Retrying these wastes time since the request itself is invalid.
_NON_RETRYABLE_STATUS_CODES = (400, 401, 403, 404, 422)


def _is_non_retryable(error: Exception) -> bool:
    """Check if an error is a non-retryable client error (4xx)."""
    err_str = str(error)
    for code in _NON_RETRYABLE_STATUS_CODES:
        if f"Error code: {code}" in err_str or f"status_code: {code}" in err_str:
            return True
    # Also check for openai-specific exception attributes
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if isinstance(status, int) and 400 <= status < 500:
        return True
    return False


class JSONParser:
    """
    Robust JSON parser for LLM responses.

    Handles common issues:
        - Markdown code blocks
        - Leading/trailing text
        - Missing quotes on keys
        - Trailing commas
    """

    # Patterns for extracting JSON from responses
    JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
    JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*\}")
    JSON_ARRAY_PATTERN = re.compile(r"\[[\s\S]*\]")

    @classmethod
    def extract_json_string(cls, text: str) -> Optional[str]:
        """
        Extract JSON string from text that may contain markdown or other content.

        Args:
            text: Raw text potentially containing JSON

        Returns:
            Extracted JSON string, or None if not found
        """
        text = text.strip()

        # Try to extract from markdown code block first
        match = cls.JSON_BLOCK_PATTERN.search(text)
        if match:
            return match.group(1).strip()

        # Try to find raw JSON object
        match = cls.JSON_OBJECT_PATTERN.search(text)
        if match:
            return match.group(0)

        # Try to find raw JSON array
        match = cls.JSON_ARRAY_PATTERN.search(text)
        if match:
            return match.group(0)

        return None

    @classmethod
    def clean_json_string(cls, json_str: str) -> str:
        """
        Clean common JSON formatting issues.

        Args:
            json_str: Raw JSON string

        Returns:
            Cleaned JSON string
        """
        # Remove trailing commas before } or ]
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        # Remove comment-only lines (// style).
        # Only strip lines that begin with optional whitespace then //
        # to avoid corrupting URLs (https://...) inside JSON strings.
        json_str = re.sub(r"^\s*//[^\n]*$", "", json_str, flags=re.MULTILINE)

        return json_str

    @classmethod
    def repair_truncated_json(cls, json_str: str) -> Optional[str]:
        """
        Attempt to repair JSON truncated by LLM max_tokens limits.

        Walks the string tracking open/close brackets and quotes,
        then appends whatever closing characters are needed.

        Args:
            json_str: Potentially truncated JSON string

        Returns:
            Repaired JSON string, or None if repair is not feasible
        """
        if not json_str:
            return None

        # Strip trailing whitespace and incomplete trailing tokens
        # (e.g. a key name that was cut off mid-word)
        repaired = json_str.rstrip()

        # Track nesting state
        stack: list[str] = []  # expected closing chars
        in_string = False
        escape_next = False

        for ch in repaired:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                if in_string:
                    escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in ("}", "]"):
                if stack and stack[-1] == ch:
                    stack.pop()

        # If nothing to close, repair won't help
        if not stack and not in_string:
            return None

        # Close open string if needed
        if in_string:
            repaired += '"'

        # Remove a possible trailing comma before we close containers
        repaired = re.sub(r",\s*$", "", repaired)

        # Close all open containers in reverse order
        repaired += "".join(reversed(stack))

        return repaired

    @classmethod
    def parse(
        cls,
        text: str,
        default: Optional[T] = None,
        expected_type: Optional[Type[T]] = None,
    ) -> Union[Dict[str, Any], List[Any], T, None]:
        """
        Parse JSON from LLM response text.

        Args:
            text: Raw LLM response
            default: Default value if parsing fails
            expected_type: Expected return type (dict or list)

        Returns:
            Parsed JSON data, or default if parsing fails
        """
        if not text:
            return default

        # Extract JSON string
        json_str = cls.extract_json_string(text)
        if not json_str:
            # Try parsing the whole text
            json_str = text.strip()

        # Clean the JSON
        json_str = cls.clean_json_string(json_str)

        # Attempt parsing
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            # Try repairing truncated JSON (common with LLM max_tokens limits)
            repaired = cls.repair_truncated_json(json_str)
            if repaired:
                try:
                    result = json.loads(repaired)
                    logger.debug("Parsed JSON after truncation repair")
                except json.JSONDecodeError as e2:
                    logger.debug(f"JSON parse error after repair attempt: {e2}")
                    return default
            else:
                return default

        # Validate type if specified
        if expected_type is not None:
            if expected_type == dict and not isinstance(result, dict):
                logger.warning(f"Expected dict but got {type(result).__name__}")
                return default
            if expected_type == list and not isinstance(result, list):
                logger.warning(f"Expected list but got {type(result).__name__}")
                return default

        return result


@dataclass
class LLMResponse:
    """Structured LLM response with metadata."""

    content: str
    raw_response: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    retries: int = 0
    latency_ms: float = 0.0


class LLMClient:
    """
    LLM client with retry logic and robust response handling.

    Wraps Haystack's OpenAIChatGenerator with additional features:
        - Automatic retry on failures
        - JSON response parsing
        - Structured response objects
    """

    def __init__(
        self,
        ollama_config: "OllamaConfig",
        parsing_config: "ParsingConfig",
    ) -> None:
        """
        Initialize LLM client.

        Args:
            ollama_config: Ollama/LLM configuration
            parsing_config: Response parsing configuration
        """
        self._ollama_config = ollama_config
        self._parsing_config = parsing_config

        # Initialize Haystack generator
        self._generator = OpenAIChatGenerator(
            api_key=Secret.from_token(ollama_config.openai_api_key),
            model=ollama_config.chat_model,
            api_base_url=ollama_config.openai_base_url,
            timeout=ollama_config.timeout,
        )

        logger.info(f"Initialized LLM client: model={ollama_config.chat_model}")

    @staticmethod
    def create_messages(system: str, user: str) -> List[ChatMessage]:
        """
        Create chat messages for a request.

        Args:
            system: System prompt
            user: User message

        Returns:
            List of ChatMessage objects
        """
        return [
            ChatMessage.from_system(system),
            ChatMessage.from_user(user),
        ]

    def chat(
        self,
        messages: List[ChatMessage],
        retry_on_error: bool = True,
    ) -> LLMResponse:
        """
        Send chat request to LLM.

        Args:
            messages: Chat messages
            retry_on_error: Whether to retry on failures

        Returns:
            LLMResponse object
        """
        max_retries = self._ollama_config.max_retries if retry_on_error else 1
        retry_delay = self._ollama_config.retry_delay
        last_error = None

        for attempt in range(max_retries):
            start_time = time.time()

            try:
                result = self._generator.run(messages=messages)
                latency = (time.time() - start_time) * 1000

                replies = result.get("replies", [])
                if not replies:
                    raise ValueError("Empty response from LLM")

                # Handle ChatMessage objects from Haystack
                reply = replies[0]

                # Extract text content from ChatMessage
                if hasattr(reply, 'text'):
                    # Haystack ChatMessage has .text property
                    content = reply.text
                elif hasattr(reply, '_content') and reply._content:
                    # Fallback: access _content list of TextContent objects
                    text_parts = []
                    for item in reply._content:
                        if hasattr(item, 'text'):
                            text_parts.append(item.text)
                    content = "\n".join(text_parts)
                elif hasattr(reply, 'content'):
                    content = reply.content
                else:
                    content = str(reply)

                # Ensure content is a string
                if not isinstance(content, str):
                    content = str(content)

                return LLMResponse(
                    content=content,
                    raw_response=result,
                    success=True,
                    retries=attempt,
                    latency_ms=latency,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM request failed (attempt {attempt + 1}/{max_retries}): {e}")

                # Don't retry on non-retryable client errors (400, 401, etc.)
                if _is_non_retryable(e):
                    logger.debug("Non-retryable error, skipping remaining attempts")
                    break

                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

        return LLMResponse(
            content="",
            raw_response={},
            success=False,
            error=last_error,
            retries=max_retries,
            latency_ms=0.0,
        )

    def chat_json(
        self,
        system: str,
        user: str,
        default: Optional[T] = None,
        expected_type: Optional[Type[T]] = None,
        retry_on_parse_error: bool = True,
    ) -> tuple[Union[Dict[str, Any], List[Any], T, None], LLMResponse]:
        """
        Send chat request expecting JSON response.

        If parsing fails, can optionally retry with a clarification message.

        Args:
            system: System prompt
            user: User message
            default: Default value if parsing fails
            expected_type: Expected JSON type (dict or list)
            retry_on_parse_error: Whether to retry on JSON parse failures

        Returns:
            Tuple of (parsed_data, LLMResponse)
        """
        messages = self.create_messages(system, user)
        response = self.chat(messages)

        if not response.success:
            return default, response

        # Try to parse JSON
        parsed = JSONParser.parse(
            response.content,
            default=None,
            expected_type=expected_type,
        )

        if parsed is not None:
            return parsed, response

        # Parsing failed - optionally retry with clarification
        if retry_on_parse_error and self._parsing_config.max_retries > 0:
            logger.debug("JSON parse failed, retrying with clarification...")

            # Add clarification message
            clarification = (
                "Your previous response could not be parsed as valid JSON. "
                "Please respond with ONLY valid JSON, no markdown code blocks, "
                "no explanatory text, just the raw JSON data."
            )
            retry_messages = messages + [
                ChatMessage.from_assistant(response.content),
                ChatMessage.from_user(clarification),
            ]

            for retry_attempt in range(self._parsing_config.max_retries):
                time.sleep(self._parsing_config.retry_delay)

                retry_response = self.chat(retry_messages, retry_on_error=False)
                if not retry_response.success:
                    continue

                parsed = JSONParser.parse(
                    retry_response.content,
                    default=None,
                    expected_type=expected_type,
                )

                if parsed is not None:
                    retry_response.retries = response.retries + retry_attempt + 1
                    return parsed, retry_response

                # Update messages for next retry
                retry_messages = retry_messages + [
                    ChatMessage.from_assistant(retry_response.content),
                    ChatMessage.from_user("That still wasn't valid JSON. Please try again with just the JSON."),
                ]

        # Log failure if configured
        if self._parsing_config.log_failures:
            logger.warning(
                f"Failed to parse JSON from LLM response. "
                f"Raw content: {response.content[:500]}..."
            )

        return default, response


@dataclass(frozen=True)
class LLMClients:
    """
    Container for all LLM-related clients.

    Convenience class for passing around LLM dependencies.
    """

    chat: LLMClient
    local: LocalNLPModels

    @staticmethod
    def build(
        ollama_config: "OllamaConfig",
        local_config: "LocalModelsConfig",
        parsing_config: "ParsingConfig",
        embedding_cache_size: int = 10000,
    ) -> "LLMClients":
        """
        Build all LLM clients from configuration.

        Args:
            ollama_config: Ollama/LLM configuration
            local_config: Local models configuration
            parsing_config: Response parsing configuration
            embedding_cache_size: Size of embedding cache (default 10000)

        Returns:
            LLMClients instance
        """
        return LLMClients(
            chat=LLMClient(ollama_config, parsing_config),
            local=LocalNLPModels.build(local_config, cache_size=embedding_cache_size),
        )
