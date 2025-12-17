"""LLM client and utilities for Ollama integration.

This module provides async client wrappers for the Ollama API, including
support for chat completions, tool calling, streaming, and model management.
"""

from aria.llm.client import (
    OllamaClient,
    OllamaError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaAPIError,
)
from aria.llm.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseMetadata,
    ToolCall,
    ToolFunction,
    ToolDefinition,
    ModelInfo,
    ModelList,
    ModelCapabilities,
    DEFAULT_MODEL_CAPABILITIES,
)
from aria.llm.tools import (
    pydantic_to_json_schema,
    create_tool_definition,
    create_simple_tool,
    function_to_tool_definition,
    ToolRegistry,
)

__all__ = [
    # Client
    "OllamaClient",
    "OllamaError",
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "OllamaAPIError",
    # Models
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatResponseMetadata",
    "ToolCall",
    "ToolFunction",
    "ToolDefinition",
    "ModelInfo",
    "ModelList",
    "ModelCapabilities",
    "DEFAULT_MODEL_CAPABILITIES",
    # Tools
    "pydantic_to_json_schema",
    "create_tool_definition",
    "create_simple_tool",
    "function_to_tool_definition",
    "ToolRegistry",
]
