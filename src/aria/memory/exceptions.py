"""Custom exceptions for memory and conversation storage operations."""


class MemoryError(Exception):
    """Base exception for memory-related errors."""

    pass


class DatabaseError(MemoryError):
    """Exception raised when database operations fail."""

    pass


class SessionNotFoundError(MemoryError):
    """Exception raised when a session is not found."""

    def __init__(self, session_id: str):
        """Initialize with session ID.

        Args:
            session_id: The ID of the session that was not found
        """
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class MessageNotFoundError(MemoryError):
    """Exception raised when a message is not found."""

    def __init__(self, message_id: str):
        """Initialize with message ID.

        Args:
            message_id: The ID of the message that was not found
        """
        self.message_id = message_id
        super().__init__(f"Message not found: {message_id}")


class ToolCallNotFoundError(MemoryError):
    """Exception raised when a tool call is not found."""

    def __init__(self, tool_call_id: str):
        """Initialize with tool call ID.

        Args:
            tool_call_id: The ID of the tool call that was not found
        """
        self.tool_call_id = tool_call_id
        super().__init__(f"Tool call not found: {tool_call_id}")
