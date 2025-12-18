"""Async SQLite-based conversation storage for ARIA.

This module provides persistent storage for conversation history, sessions,
messages, and tool calls using an async SQLite database.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from aria.config import get_settings
from aria.logging import get_logger
from aria.memory.exceptions import (
    DatabaseError,
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

logger = get_logger("aria.memory.conversation")


class ConversationStore:
    """Async SQLite-based conversation storage.

    This class manages persistent storage of conversation history including
    sessions, messages, and tool calls. It provides async methods for CRUD
    operations and searching.

    Usage:
        store = ConversationStore()
        await store.initialize()

        session = await store.create_session("My Chat")
        await store.add_message(session.id, "user", "Hello!")
    """

    def __init__(self, db_path: Path | None = None, auto_index_callback: Any | None = None):
        """Initialize the conversation store.

        Args:
            db_path: Path to SQLite database file. If None, uses config default.
            auto_index_callback: Optional async callback for auto-indexing messages.
                                Should accept (session_id, message) and return None.
        """
        settings = get_settings()
        self.db_path = db_path or settings.conversation_db_path
        self._initialized = False
        self._auto_index_callback = auto_index_callback

        logger.debug("ConversationStore initialized", db_path=str(self.db_path))

    async def initialize(self) -> None:
        """Create database and tables if they don't exist.

        This must be called before using the store. It creates the database
        file and applies the schema from schema.sql.

        Raises:
            DatabaseError: If database initialization fails
        """
        if self._initialized:
            logger.debug("Database already initialized")
            return

        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Read schema file
            schema_path = Path(__file__).parent / "schema.sql"
            with open(schema_path, "r") as f:
                schema_sql = f.read()

            # Execute schema
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                # Enable foreign key constraints (required for cascade delete)
                await db.execute("PRAGMA foreign_keys = ON")
                await db.executescript(schema_sql)
                await db.commit()

            self._initialized = True
            logger.info("Database initialized successfully", db_path=str(self.db_path))

        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise DatabaseError(f"Failed to initialize database: {e}") from e

    async def _get_connection(self):
        """Create a database connection with foreign keys enabled.

        Returns:
            aiosqlite.Connection: Database connection with foreign keys enabled
        """
        db = await aiosqlite.connect(self.db_path)
        await db.execute("PRAGMA foreign_keys = ON")
        await db.commit()
        return db

    # ========================================================================
    # Session Management
    # ========================================================================

    async def create_session(self, title: str | None = None) -> Session:
        """Create a new conversation session.

        Args:
            title: Optional session title. Auto-generated if not provided.

        Returns:
            Session: The created session

        Raises:
            DatabaseError: If session creation fails
        """
        session_id = uuid.uuid4().hex
        now = datetime.now(UTC)
        title = title or f"Session {now.strftime('%Y-%m-%d %H:%M')}"

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                await db.execute(
                    """
                    INSERT INTO sessions (id, title, created_at, updated_at, message_count, metadata)
                    VALUES (?, ?, ?, ?, 0, '{}')
                    """,
                    (session_id, title, now.isoformat(), now.isoformat()),
                )
                await db.commit()

            session = Session(
                id=session_id,
                title=title,
                created_at=now,
                updated_at=now,
                message_count=0,
                metadata={},
            )

            logger.info("Session created", session_id=session_id, title=title)
            return session

        except Exception as e:
            logger.error("Failed to create session", error=str(e))
            raise DatabaseError(f"Failed to create session: {e}") from e

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            Session | None: The session if found, None otherwise
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM sessions WHERE id = ?", (session_id,)
                ) as cursor:
                    row = await cursor.fetchone()

                    if row is None:
                        return None

                    return self._row_to_session(row)

        except Exception as e:
            logger.error("Failed to get session", session_id=session_id, error=str(e))
            raise DatabaseError(f"Failed to get session: {e}") from e

    async def list_sessions(
        self, limit: int = 20, offset: int = 0
    ) -> list[SessionSummary]:
        """List sessions ordered by most recent activity.

        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip

        Returns:
            list[SessionSummary]: List of session summaries
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    """
                    SELECT * FROM session_summaries
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [self._row_to_session_summary(row) for row in rows]

        except Exception as e:
            logger.error("Failed to list sessions", error=str(e))
            raise DatabaseError(f"Failed to list sessions: {e}") from e

    async def update_session(self, session_id: str, title: str) -> Session:
        """Update a session's title.

        Args:
            session_id: The session ID to update
            title: New title for the session

        Returns:
            Session: The updated session

        Raises:
            SessionNotFoundError: If session doesn't exist
            DatabaseError: If update fails
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                cursor = await db.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?",
                    (title, session_id),
                )
                await db.commit()

                if cursor.rowcount == 0:
                    raise SessionNotFoundError(session_id)

            # Fetch and return updated session
            session = await self.get_session(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)

            logger.info("Session updated", session_id=session_id, title=title)
            return session

        except SessionNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update session", session_id=session_id, error=str(e))
            raise DatabaseError(f"Failed to update session: {e}") from e

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all associated messages and tool calls.

        Args:
            session_id: The session ID to delete

        Returns:
            bool: True if session was deleted, False if not found
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                cursor = await db.execute(
                    "DELETE FROM sessions WHERE id = ?", (session_id,)
                )
                await db.commit()

                deleted = cursor.rowcount > 0
                if deleted:
                    logger.info("Session deleted", session_id=session_id)
                else:
                    logger.warning("Session not found for deletion", session_id=session_id)

                return deleted

        except Exception as e:
            logger.error("Failed to delete session", session_id=session_id, error=str(e))
            raise DatabaseError(f"Failed to delete session: {e}") from e

    # ========================================================================
    # Message Operations
    # ========================================================================

    async def add_message(
        self, session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> Message:
        """Add a message to a session.

        Args:
            session_id: The session to add the message to
            role: Message role (user, assistant, system, tool)
            content: Message content
            metadata: Optional metadata dictionary

        Returns:
            Message: The created message

        Raises:
            SessionNotFoundError: If session doesn't exist
            DatabaseError: If message creation fails
        """
        # Verify session exists
        session = await self.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)

        message_id = uuid.uuid4().hex
        now = datetime.now(UTC)
        metadata_json = json.dumps(metadata or {})

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                await db.execute(
                    """
                    INSERT INTO messages (id, session_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (message_id, session_id, role, content, now.isoformat(), metadata_json),
                )
                await db.commit()

            message = Message(
                id=message_id,
                session_id=session_id,
                role=MessageRole(role),
                content=content,
                timestamp=now,
                metadata=metadata or {},
            )

            logger.debug("Message added", message_id=message_id, session_id=session_id, role=role)

            # Auto-index if callback is set
            if self._auto_index_callback:
                try:
                    await self._auto_index_callback(session_id, message)
                except Exception as e:
                    # Log but don't fail - indexing is non-critical
                    logger.warning(f"Auto-index failed for message {message_id}: {e}")

            return message

        except Exception as e:
            logger.error("Failed to add message", session_id=session_id, error=str(e))
            raise DatabaseError(f"Failed to add message: {e}") from e

    async def get_messages(
        self, session_id: str, limit: int | None = None
    ) -> list[Message]:
        """Get messages from a session.

        Args:
            session_id: The session ID
            limit: Maximum number of messages to return (most recent first).
                   If None, returns all messages.

        Returns:
            list[Message]: List of messages ordered by timestamp (oldest first)
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                db.row_factory = aiosqlite.Row

                if limit is not None:
                    # Get most recent N messages, then reverse to oldest-first
                    query = """
                        SELECT * FROM messages
                        WHERE session_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """
                    params = (session_id, limit)
                else:
                    query = """
                        SELECT * FROM messages
                        WHERE session_id = ?
                        ORDER BY timestamp ASC
                    """
                    params = (session_id,)

                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                    # If limited, reverse to get oldest-first order
                    if limit is not None:
                        rows = list(reversed(rows))

                    return [self._row_to_message(row) for row in rows]

        except Exception as e:
            logger.error("Failed to get messages", session_id=session_id, error=str(e))
            raise DatabaseError(f"Failed to get messages: {e}") from e

    async def get_context(
        self, session_id: str, max_messages: int = 50
    ) -> ConversationContext:
        """Get conversation context for a session.

        Args:
            session_id: The session ID
            max_messages: Maximum number of recent messages to include

        Returns:
            ConversationContext: Context with session, messages, and tool calls

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        # Get session
        session = await self.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)

        # Get recent messages
        messages = await self.get_messages(session_id, limit=max_messages)

        # Get tool calls for these messages
        tool_calls_by_message: dict[str, list[ToolCall]] = {}
        if messages:
            message_ids = [msg.id for msg in messages]
            tool_calls = await self._get_tool_calls_for_messages(message_ids)

            for tool_call in tool_calls:
                if tool_call.message_id not in tool_calls_by_message:
                    tool_calls_by_message[tool_call.message_id] = []
                tool_calls_by_message[tool_call.message_id].append(tool_call)

        context = ConversationContext(
            session=session,
            messages=messages,
            tool_calls=tool_calls_by_message,
            total_messages=session.message_count,
        )

        logger.debug(
            "Context retrieved",
            session_id=session_id,
            messages=len(messages),
            total=session.message_count,
        )
        return context

    # ========================================================================
    # Tool Call Tracking
    # ========================================================================

    async def add_tool_call(
        self,
        message_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: dict[str, Any] | None = None,
        status: str = "pending",
        error_message: str | None = None,
    ) -> ToolCall:
        """Add a tool call to a message.

        Args:
            message_id: The message ID
            tool_name: Name of the tool
            tool_input: Tool input parameters
            tool_output: Tool output (None if pending/error)
            status: Tool call status (pending, success, error, denied)
            error_message: Error message if status is error

        Returns:
            ToolCall: The created tool call

        Raises:
            MessageNotFoundError: If message doesn't exist
            DatabaseError: If tool call creation fails
        """
        tool_call_id = uuid.uuid4().hex
        now = datetime.now(UTC)
        input_json = json.dumps(tool_input)
        output_json = json.dumps(tool_output) if tool_output is not None else None

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                await db.execute(
                    """
                    INSERT INTO tool_calls
                    (id, message_id, tool_name, tool_input, tool_output, status, timestamp, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tool_call_id,
                        message_id,
                        tool_name,
                        input_json,
                        output_json,
                        status,
                        now.isoformat(),
                        error_message,
                    ),
                )
                await db.commit()

            tool_call = ToolCall(
                id=tool_call_id,
                message_id=message_id,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                status=ToolCallStatus(status),
                timestamp=now,
                error_message=error_message,
            )

            logger.debug(
                "Tool call added",
                tool_call_id=tool_call_id,
                message_id=message_id,
                tool_name=tool_name,
                status=status,
            )
            return tool_call

        except Exception as e:
            logger.error("Failed to add tool call", message_id=message_id, error=str(e))
            raise DatabaseError(f"Failed to add tool call: {e}") from e

    async def update_tool_call(
        self,
        tool_call_id: str,
        tool_output: dict[str, Any],
        status: str,
        duration_ms: int,
        error_message: str | None = None,
    ) -> ToolCall:
        """Update a tool call with results.

        Args:
            tool_call_id: The tool call ID
            tool_output: Tool output data
            status: Updated status (success, error)
            duration_ms: Execution duration in milliseconds
            error_message: Error message if status is error

        Returns:
            ToolCall: The updated tool call

        Raises:
            ToolCallNotFoundError: If tool call doesn't exist
            DatabaseError: If update fails
        """
        output_json = json.dumps(tool_output)

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                cursor = await db.execute(
                    """
                    UPDATE tool_calls
                    SET tool_output = ?, status = ?, duration_ms = ?, error_message = ?
                    WHERE id = ?
                    """,
                    (output_json, status, duration_ms, error_message, tool_call_id),
                )
                await db.commit()

                if cursor.rowcount == 0:
                    raise ToolCallNotFoundError(tool_call_id)

            # Fetch and return updated tool call
            tool_call = await self._get_tool_call(tool_call_id)
            if tool_call is None:
                raise ToolCallNotFoundError(tool_call_id)

            logger.debug(
                "Tool call updated",
                tool_call_id=tool_call_id,
                status=status,
                duration_ms=duration_ms,
            )
            return tool_call

        except ToolCallNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update tool call", tool_call_id=tool_call_id, error=str(e))
            raise DatabaseError(f"Failed to update tool call: {e}") from e

    # ========================================================================
    # Search & Queries
    # ========================================================================

    async def search_messages(
        self, query: str, session_id: str | None = None, limit: int = 50
    ) -> list[Message]:
        """Search messages by content.

        Args:
            query: Search query string
            session_id: Optional session ID to limit search to
            limit: Maximum number of results

        Returns:
            list[Message]: Matching messages ordered by timestamp (newest first)
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                db.row_factory = aiosqlite.Row

                if session_id:
                    sql = """
                        SELECT * FROM messages
                        WHERE session_id = ? AND content LIKE ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """
                    params = (session_id, f"%{query}%", limit)
                else:
                    sql = """
                        SELECT * FROM messages
                        WHERE content LIKE ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """
                    params = (f"%{query}%", limit)

                async with db.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()
                    return [self._row_to_message(row) for row in rows]

        except Exception as e:
            logger.error("Failed to search messages", query=query, error=str(e))
            raise DatabaseError(f"Failed to search messages: {e}") from e

    async def get_recent_sessions(self, days: int = 7) -> list[SessionSummary]:
        """Get sessions updated within the last N days.

        Args:
            days: Number of days to look back

        Returns:
            list[SessionSummary]: Recent sessions ordered by update time (newest first)
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    """
                    SELECT * FROM session_summaries
                    WHERE updated_at >= ?
                    ORDER BY updated_at DESC
                    """,
                    (cutoff.isoformat(),),
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [self._row_to_session_summary(row) for row in rows]

        except Exception as e:
            logger.error("Failed to get recent sessions", days=days, error=str(e))
            raise DatabaseError(f"Failed to get recent sessions: {e}") from e

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _row_to_session(self, row: aiosqlite.Row) -> Session:
        """Convert a database row to a Session model."""
        return Session(
            id=row["id"],
            title=row["title"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            message_count=row["message_count"],
            metadata=json.loads(row["metadata"]),
        )

    def _row_to_session_summary(self, row: aiosqlite.Row) -> SessionSummary:
        """Convert a database row to a SessionSummary model."""
        return SessionSummary(
            id=row["id"],
            title=row["title"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            message_count=row["message_count"],
            tool_call_count=row["tool_call_count"] or 0,
            last_message_preview=row["last_message_preview"],
        )

    def _row_to_message(self, row: aiosqlite.Row) -> Message:
        """Convert a database row to a Message model."""
        return Message(
            id=row["id"],
            session_id=row["session_id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            metadata=json.loads(row["metadata"]),
        )

    def _row_to_tool_call(self, row: aiosqlite.Row) -> ToolCall:
        """Convert a database row to a ToolCall model."""
        return ToolCall(
            id=row["id"],
            message_id=row["message_id"],
            tool_name=row["tool_name"],
            tool_input=json.loads(row["tool_input"]),
            tool_output=json.loads(row["tool_output"]) if row["tool_output"] else None,
            status=ToolCallStatus(row["status"]),
            duration_ms=row["duration_ms"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            error_message=row["error_message"],
        )

    async def _get_tool_call(self, tool_call_id: str) -> ToolCall | None:
        """Get a single tool call by ID."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM tool_calls WHERE id = ?", (tool_call_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    return self._row_to_tool_call(row) if row else None
        except Exception as e:
            logger.error("Failed to get tool call", tool_call_id=tool_call_id, error=str(e))
            raise DatabaseError(f"Failed to get tool call: {e}") from e

    async def _get_tool_calls_for_messages(
        self, message_ids: list[str]
    ) -> list[ToolCall]:
        """Get all tool calls for a list of message IDs."""
        if not message_ids:
            return []

        try:
            placeholders = ",".join("?" * len(message_ids))
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    f"""
                    SELECT * FROM tool_calls
                    WHERE message_id IN ({placeholders})
                    ORDER BY timestamp ASC
                    """,
                    message_ids,
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [self._row_to_tool_call(row) for row in rows]

        except Exception as e:
            logger.error("Failed to get tool calls", error=str(e))
            raise DatabaseError(f"Failed to get tool calls: {e}") from e

    async def close(self) -> None:
        """Close the database connection.

        Note: aiosqlite uses connection context managers, so this is
        mainly for API consistency.
        """
        logger.debug("ConversationStore closed")
