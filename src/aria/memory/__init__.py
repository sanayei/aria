"""Memory and conversation history management for ARIA.

This module provides persistent storage for conversation history, sessions,
messages, and tool calls using SQLite.
"""

from aria.memory.context import context_to_chat_messages, messages_to_chat_messages
from aria.memory.conversation import ConversationStore
from aria.memory.exceptions import (
    DatabaseError,
    MemoryError,
    MessageNotFoundError,
    SessionNotFoundError,
    ToolCallNotFoundError,
)
from aria.memory.models import (
    ConversationContext,
    Message,
    MessageRole,
    Session,
    SessionSummary,
    ToolCall,
    ToolCallStatus,
)

__all__ = [
    # Store
    "ConversationStore",
    # Models
    "ConversationContext",
    "Message",
    "MessageRole",
    "Session",
    "SessionSummary",
    "ToolCall",
    "ToolCallStatus",
    # Exceptions
    "DatabaseError",
    "MemoryError",
    "MessageNotFoundError",
    "SessionNotFoundError",
    "ToolCallNotFoundError",
    # Context helpers
    "context_to_chat_messages",
    "messages_to_chat_messages",
]
