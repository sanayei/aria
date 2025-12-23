"""Data models for conversation history and memory management.

This module defines the Pydantic models used for storing and managing
conversation history, sessions, messages, and tool calls in the SQLite database.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

    def __str__(self) -> str:
        """String representation of message role."""
        return self.value


class ToolCallStatus(str, Enum):
    """Status of a tool call execution."""

    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"
    DENIED = "denied"  # User denied approval

    def __str__(self) -> str:
        """String representation of tool call status."""
        return self.value


class Session(BaseModel):
    """A conversation session containing multiple messages.

    Sessions group related messages together and track session-level metadata
    like creation time, last update, and custom metadata.
    """

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique session identifier (UUID)")
    title: str = Field(..., description="Human-readable session title")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the session was created",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the session was last updated",
    )
    message_count: int = Field(
        default=0,
        description="Total number of messages in this session",
        ge=0,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional session metadata (tags, context, etc.)",
    )

    def __repr__(self) -> str:
        """String representation of session."""
        return f"Session(id={self.id[:8]}..., title='{self.title}', messages={self.message_count})"


class Message(BaseModel):
    """A single message in a conversation.

    Messages can be from the user, assistant, system, or tool results.
    Each message is associated with a session and ordered by timestamp.
    """

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique message identifier (UUID)")
    session_id: str = Field(..., description="ID of the session this message belongs to")
    role: MessageRole = Field(..., description="Who sent this message")
    content: str = Field(..., description="The message content")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the message was created",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional message metadata (tokens, model, etc.)",
    )

    def __repr__(self) -> str:
        """String representation of message."""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Message(role={self.role}, content='{preview}')"


class ToolCall(BaseModel):
    """A tool call made by the assistant during conversation.

    Tool calls represent the agent's use of tools, including the input
    parameters, output results, execution status, and performance metrics.
    """

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Unique tool call identifier (UUID)")
    message_id: str = Field(..., description="ID of the message that triggered this tool call")
    tool_name: str = Field(..., description="Name of the tool that was called")
    tool_input: dict[str, Any] = Field(
        ...,
        description="Input parameters passed to the tool",
    )
    tool_output: dict[str, Any] | None = Field(
        default=None,
        description="Output returned by the tool (None if pending/error)",
    )
    status: ToolCallStatus = Field(
        default=ToolCallStatus.PENDING,
        description="Execution status of the tool call",
    )
    duration_ms: int | None = Field(
        default=None,
        description="Execution time in milliseconds",
        ge=0,
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the tool was called",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if the tool call failed",
    )

    def __repr__(self) -> str:
        """String representation of tool call."""
        return (
            f"ToolCall(tool={self.tool_name}, status={self.status}, duration={self.duration_ms}ms)"
        )


class ConversationContext(BaseModel):
    """Helper model for passing conversation context to the agent.

    This model aggregates messages and tool calls from a session to provide
    the agent with the necessary context for continuing a conversation.
    """

    model_config = ConfigDict(strict=True)

    session: Session = Field(..., description="The current session")
    messages: list[Message] = Field(
        default_factory=list,
        description="Recent messages (limited by max_context_messages)",
    )
    tool_calls: dict[str, list[ToolCall]] = Field(
        default_factory=dict,
        description="Tool calls grouped by message_id",
    )
    total_messages: int = Field(
        default=0,
        description="Total messages in session (may be more than returned)",
        ge=0,
    )

    @property
    def is_truncated(self) -> bool:
        """Check if the context is truncated (not all messages included).

        Returns:
            bool: True if there are more messages than returned
        """
        return self.total_messages > len(self.messages)

    @property
    def message_count(self) -> int:
        """Get the number of messages in this context.

        Returns:
            int: Number of messages
        """
        return len(self.messages)

    def __repr__(self) -> str:
        """String representation of conversation context."""
        truncated = " (truncated)" if self.is_truncated else ""
        return f"ConversationContext(session={self.session.id[:8]}..., messages={len(self.messages)}/{self.total_messages}{truncated})"


class SessionSummary(BaseModel):
    """Summary information about a session for listing/browsing.

    Lightweight model for displaying session lists without loading
    all messages and tool calls.
    """

    model_config = ConfigDict(strict=True)

    id: str = Field(..., description="Session identifier")
    title: str = Field(..., description="Session title")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    message_count: int = Field(..., description="Number of messages", ge=0)
    tool_call_count: int = Field(
        default=0,
        description="Number of tool calls in this session",
        ge=0,
    )
    last_message_preview: str | None = Field(
        default=None,
        description="Preview of the last message (first 100 chars)",
    )

    def __repr__(self) -> str:
        """String representation of session summary."""
        return f"SessionSummary(id={self.id[:8]}..., title='{self.title}', messages={self.message_count})"
