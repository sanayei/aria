"""Pydantic models for Ollama API requests and responses.

These models provide type-safe interfaces to the Ollama API, ensuring
proper validation and serialization of messages, tool calls, and responses.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Tool-related models
# =============================================================================


class ToolFunction(BaseModel):
    """Function specification for a tool call.

    Represents the function name and arguments when the LLM decides to call a tool.
    """

    name: str = Field(..., description="Name of the function to call")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the function",
    )


class ToolCall(BaseModel):
    """A tool call made by the LLM.

    Contains the function to call and an optional ID for tracking.
    """

    id: str | None = Field(default=None, description="Unique identifier for this tool call")
    function: ToolFunction = Field(..., description="Function to call")
    type: Literal["function"] = Field(default="function", description="Type of tool call")


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by the LLM.

    This is sent to the LLM to describe available tools.
    """

    type: Literal["function"] = Field(default="function", description="Type of tool")
    function: dict[str, Any] = Field(..., description="Function schema (JSON Schema format)")

    @classmethod
    def from_schema(cls, name: str, description: str, parameters: dict[str, Any]) -> "ToolDefinition":
        """Create a ToolDefinition from schema components.

        Args:
            name: Name of the function
            description: Human-readable description
            parameters: JSON Schema for the parameters

        Returns:
            ToolDefinition: The constructed tool definition
        """
        return cls(
            function={
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )


# =============================================================================
# Message models
# =============================================================================


class ChatMessage(BaseModel):
    """A message in a chat conversation.

    Supports text content, tool calls, and optional images.
    """

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ...,
        description="Role of the message sender",
    )
    content: str = Field(..., description="Text content of the message")
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool calls made by the assistant (only for assistant messages)",
    )
    images: list[str] = Field(
        default_factory=list,
        description="Base64-encoded images (for multimodal models)",
    )

    @field_validator("tool_calls", mode="before")
    @classmethod
    def validate_tool_calls(cls, v: Any) -> list[ToolCall]:
        """Ensure tool_calls is always a list."""
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]

    def model_dump_ollama(self) -> dict[str, Any]:
        """Dump the message in Ollama API format.

        Returns:
            dict: Message formatted for Ollama API
        """
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }

        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in self.tool_calls
            ]

        if self.images:
            result["images"] = self.images

        return result

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        """Create a system message.

        Args:
            content: System message content

        Returns:
            ChatMessage: System message
        """
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str, images: list[str] | None = None) -> "ChatMessage":
        """Create a user message.

        Args:
            content: User message content
            images: Optional base64-encoded images

        Returns:
            ChatMessage: User message
        """
        return cls(role="user", content=content, images=images or [])

    @classmethod
    def assistant(
        cls,
        content: str,
        tool_calls: list[ToolCall] | None = None,
    ) -> "ChatMessage":
        """Create an assistant message.

        Args:
            content: Assistant message content
            tool_calls: Optional tool calls made by the assistant

        Returns:
            ChatMessage: Assistant message
        """
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, content: str, tool_call_id: str | None = None) -> "ChatMessage":
        """Create a tool response message.

        Args:
            content: Tool response content
            tool_call_id: ID of the tool call this responds to

        Returns:
            ChatMessage: Tool message
        """
        return cls(role="tool", content=content)


# =============================================================================
# Request models
# =============================================================================


class ChatRequest(BaseModel):
    """Request to the Ollama chat API."""

    model: str = Field(..., description="Model name to use")
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    tools: list[ToolDefinition] = Field(
        default_factory=list,
        description="Available tools for the LLM",
    )
    stream: bool = Field(default=False, description="Enable streaming responses")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific options (temperature, etc.)",
    )

    def model_dump_ollama(self) -> dict[str, Any]:
        """Dump the request in Ollama API format.

        Returns:
            dict: Request formatted for Ollama API
        """
        result: dict[str, Any] = {
            "model": self.model,
            "messages": [msg.model_dump_ollama() for msg in self.messages],
            "stream": self.stream,
        }

        if self.tools:
            result["tools"] = [tool.model_dump() for tool in self.tools]

        if self.options:
            result["options"] = self.options

        return result


# =============================================================================
# Response models
# =============================================================================


class ChatResponseMetadata(BaseModel):
    """Metadata about the chat response."""

    total_duration: int | None = Field(
        default=None,
        description="Total duration in nanoseconds",
    )
    load_duration: int | None = Field(
        default=None,
        description="Model load duration in nanoseconds",
    )
    prompt_eval_count: int | None = Field(
        default=None,
        description="Number of tokens in the prompt",
    )
    prompt_eval_duration: int | None = Field(
        default=None,
        description="Prompt evaluation duration in nanoseconds",
    )
    eval_count: int | None = Field(
        default=None,
        description="Number of tokens in the response",
    )
    eval_duration: int | None = Field(
        default=None,
        description="Response generation duration in nanoseconds",
    )

    @property
    def total_duration_ms(self) -> float | None:
        """Get total duration in milliseconds."""
        return self.total_duration / 1_000_000 if self.total_duration else None

    @property
    def tokens_per_second(self) -> float | None:
        """Calculate tokens per second for generation."""
        if self.eval_count and self.eval_duration and self.eval_duration > 0:
            return (self.eval_count / self.eval_duration) * 1_000_000_000
        return None


class ChatResponse(BaseModel):
    """Response from the Ollama chat API."""

    model: str = Field(..., description="Model used for generation")
    message: ChatMessage = Field(..., description="Generated message")
    done: bool = Field(default=True, description="Whether generation is complete")
    created_at: datetime | None = Field(
        default=None,
        description="Timestamp of response creation",
    )

    # Metadata fields (present when done=True)
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None

    @property
    def metadata(self) -> ChatResponseMetadata:
        """Get response metadata.

        Returns:
            ChatResponseMetadata: Metadata about the response
        """
        return ChatResponseMetadata(
            total_duration=self.total_duration,
            load_duration=self.load_duration,
            prompt_eval_count=self.prompt_eval_count,
            prompt_eval_duration=self.prompt_eval_duration,
            eval_count=self.eval_count,
            eval_duration=self.eval_duration,
        )

    @property
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls.

        Returns:
            bool: True if the message has tool calls
        """
        return len(self.message.tool_calls) > 0


# =============================================================================
# Model information
# =============================================================================


class ModelInfo(BaseModel):
    """Information about an Ollama model."""

    name: str = Field(..., description="Model name")
    modified_at: datetime | None = Field(
        default=None,
        description="Last modification time",
    )
    size: int | None = Field(default=None, description="Model size in bytes")
    digest: str | None = Field(default=None, description="Model digest/hash")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model details",
    )

    @property
    def size_mb(self) -> float | None:
        """Get model size in megabytes.

        Returns:
            float | None: Size in MB, or None if size is not available
        """
        return self.size / (1024 * 1024) if self.size else None

    @property
    def size_gb(self) -> float | None:
        """Get model size in gigabytes.

        Returns:
            float | None: Size in GB, or None if size is not available
        """
        return self.size / (1024 * 1024 * 1024) if self.size else None


class ModelList(BaseModel):
    """List of available models."""

    models: list[ModelInfo] = Field(default_factory=list, description="Available models")


# =============================================================================
# Model capabilities configuration
# =============================================================================


class ModelCapabilities(BaseModel):
    """Configuration for model capabilities and limits."""

    name: str = Field(..., description="Model name or pattern")
    context_length: int = Field(
        default=8192,
        description="Maximum context length in tokens",
    )
    supports_tools: bool = Field(
        default=True,
        description="Whether the model supports tool/function calling",
    )
    supports_vision: bool = Field(
        default=False,
        description="Whether the model supports image inputs",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate (None = model default)",
    )
    recommended_temperature: float = Field(
        default=0.7,
        description="Recommended temperature for this model",
        ge=0.0,
        le=2.0,
    )


# Default model capabilities
DEFAULT_MODEL_CAPABILITIES = {
    "qwen3:30b": ModelCapabilities(
        name="qwen3:30b",
        context_length=32768,
        supports_tools=True,
        supports_vision=False,
        recommended_temperature=0.0,  # Lower temperature for better tool calling
    ),
    "qwen3:30b-a3b": ModelCapabilities(
        name="qwen3:30b-a3b",
        context_length=32768,
        supports_tools=True,
        supports_vision=False,
        recommended_temperature=0.0,  # Lower temperature for better tool calling
    ),
    "qwen2.5:32b": ModelCapabilities(
        name="qwen2.5:32b",
        context_length=32768,
        supports_tools=True,
        supports_vision=False,
        recommended_temperature=0.7,
    ),
    "llama3": ModelCapabilities(
        name="llama3",
        context_length=8192,
        supports_tools=True,
        supports_vision=False,
        recommended_temperature=0.7,
    ),
    "llama3.1": ModelCapabilities(
        name="llama3.1",
        context_length=131072,
        supports_tools=True,
        supports_vision=False,
        recommended_temperature=0.7,
    ),
    "mistral": ModelCapabilities(
        name="mistral",
        context_length=32768,
        supports_tools=True,
        supports_vision=False,
        recommended_temperature=0.7,
    ),
}
