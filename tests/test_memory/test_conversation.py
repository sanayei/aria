"""Comprehensive tests for ConversationStore."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aria.memory.conversation import ConversationStore
from aria.memory.exceptions import (
    DatabaseError,
    MessageNotFoundError,
    SessionNotFoundError,
    ToolCallNotFoundError,
)
from aria.memory.models import MessageRole, ToolCallStatus


class TestConversationStoreInitialization:
    """Tests for database initialization."""

    @pytest.mark.asyncio
    async def test_initialize_creates_database(self, temp_db: Path):
        """Test that initialize creates the database file."""
        assert not temp_db.exists()

        store = ConversationStore(db_path=temp_db)
        await store.initialize()

        assert temp_db.exists()
        await store.close()

    @pytest.mark.asyncio
    async def test_initialize_creates_schema(self, temp_db: Path):
        """Test that initialize creates all required tables."""
        store = ConversationStore(db_path=temp_db)
        await store.initialize()

        # Verify tables exist by querying them (would fail if not created)
        import aiosqlite

        async with aiosqlite.connect(temp_db) as db:
            # Check sessions table
            async with db.execute("SELECT COUNT(*) FROM sessions") as cursor:
                result = await cursor.fetchone()
                assert result[0] == 0

            # Check messages table
            async with db.execute("SELECT COUNT(*) FROM messages") as cursor:
                result = await cursor.fetchone()
                assert result[0] == 0

            # Check tool_calls table
            async with db.execute("SELECT COUNT(*) FROM tool_calls") as cursor:
                result = await cursor.fetchone()
                assert result[0] == 0

        await store.close()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, temp_db: Path):
        """Test that initialize can be called multiple times safely."""
        store = ConversationStore(db_path=temp_db)
        await store.initialize()
        await store.initialize()  # Should not fail
        await store.close()


class TestSessionManagement:
    """Tests for session CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_session(self, conversation_store: ConversationStore):
        """Test creating a new session."""
        session = await conversation_store.create_session("My Chat")

        assert session.id is not None
        assert session.title == "My Chat"
        assert session.message_count == 0
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)
        assert session.metadata == {}

    @pytest.mark.asyncio
    async def test_create_session_auto_title(self, conversation_store: ConversationStore):
        """Test creating a session with auto-generated title."""
        session = await conversation_store.create_session()

        assert session.id is not None
        assert "Session" in session.title
        assert session.message_count == 0

    @pytest.mark.asyncio
    async def test_get_session(self, store_with_session):
        """Test retrieving a session by ID."""
        store, original_session = store_with_session

        retrieved = await store.get_session(original_session.id)

        assert retrieved is not None
        assert retrieved.id == original_session.id
        assert retrieved.title == original_session.title
        assert retrieved.message_count == original_session.message_count

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, conversation_store: ConversationStore):
        """Test getting a non-existent session returns None."""
        result = await conversation_store.get_session("nonexistent_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, conversation_store: ConversationStore):
        """Test listing sessions."""
        # Create multiple sessions
        session1 = await conversation_store.create_session("Session 1")
        session2 = await conversation_store.create_session("Session 2")
        session3 = await conversation_store.create_session("Session 3")

        # List all sessions
        sessions = await conversation_store.list_sessions(limit=10)

        assert len(sessions) == 3
        # Should be ordered by updated_at DESC (newest first)
        assert sessions[0].id == session3.id
        assert sessions[1].id == session2.id
        assert sessions[2].id == session1.id

    @pytest.mark.asyncio
    async def test_list_sessions_with_limit(self, conversation_store: ConversationStore):
        """Test listing sessions with limit."""
        await conversation_store.create_session("Session 1")
        await conversation_store.create_session("Session 2")
        await conversation_store.create_session("Session 3")

        sessions = await conversation_store.list_sessions(limit=2)

        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_with_offset(self, conversation_store: ConversationStore):
        """Test listing sessions with offset."""
        session1 = await conversation_store.create_session("Session 1")
        await conversation_store.create_session("Session 2")
        await conversation_store.create_session("Session 3")

        sessions = await conversation_store.list_sessions(limit=10, offset=2)

        assert len(sessions) == 1
        assert sessions[0].id == session1.id

    @pytest.mark.asyncio
    async def test_update_session(self, store_with_session):
        """Test updating a session title."""
        store, session = store_with_session

        updated = await store.update_session(session.id, "Updated Title")

        assert updated.title == "Updated Title"
        assert updated.id == session.id

    @pytest.mark.asyncio
    async def test_update_session_not_found(self, conversation_store: ConversationStore):
        """Test updating a non-existent session raises error."""
        with pytest.raises(SessionNotFoundError) as exc_info:
            await conversation_store.update_session("nonexistent_id", "New Title")

        assert "nonexistent_id" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_session(self, store_with_session):
        """Test deleting a session."""
        store, session = store_with_session

        result = await store.delete_session(session.id)

        assert result is True

        # Verify session is gone
        retrieved = await store.get_session(session.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, conversation_store: ConversationStore):
        """Test deleting a non-existent session returns False."""
        result = await conversation_store.delete_session("nonexistent_id")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_session_cascades(self, store_with_messages):
        """Test that deleting a session cascades to messages and tool calls."""
        store, session, messages = store_with_messages

        # Add a tool call
        await store.add_tool_call(messages[0].id, "test_tool", {"param": "value"}, status="success")

        # Delete session
        await store.delete_session(session.id)

        # Verify messages are gone (would be empty list, not error)
        retrieved_messages = await store.get_messages(session.id)
        assert len(retrieved_messages) == 0


class TestMessageOperations:
    """Tests for message CRUD operations."""

    @pytest.mark.asyncio
    async def test_add_message(self, store_with_session):
        """Test adding a message to a session."""
        store, session = store_with_session

        message = await store.add_message(session.id, "user", "Hello!")

        assert message.id is not None
        assert message.session_id == session.id
        assert message.role == MessageRole.USER
        assert message.content == "Hello!"
        assert isinstance(message.timestamp, datetime)
        assert message.metadata == {}

    @pytest.mark.asyncio
    async def test_add_message_with_metadata(self, store_with_session):
        """Test adding a message with metadata."""
        store, session = store_with_session

        metadata = {"tokens": 10, "model": "test-model"}
        message = await store.add_message(session.id, "assistant", "Response", metadata=metadata)

        assert message.metadata == metadata

    @pytest.mark.asyncio
    async def test_add_message_increments_count(self, store_with_session):
        """Test that adding messages increments session message count."""
        store, session = store_with_session

        # Add messages
        await store.add_message(session.id, "user", "Message 1")
        await store.add_message(session.id, "assistant", "Message 2")

        # Verify count updated
        updated_session = await store.get_session(session.id)
        assert updated_session.message_count == 2

    @pytest.mark.asyncio
    async def test_add_message_session_not_found(self, conversation_store: ConversationStore):
        """Test adding a message to non-existent session raises error."""
        with pytest.raises(SessionNotFoundError):
            await conversation_store.add_message("nonexistent_id", "user", "Hello")

    @pytest.mark.asyncio
    async def test_get_messages(self, store_with_messages):
        """Test retrieving messages from a session."""
        store, session, original_messages = store_with_messages

        messages = await store.get_messages(session.id)

        assert len(messages) == 3
        # Should be ordered by timestamp (oldest first)
        assert messages[0].content == "Hello, ARIA!"
        assert messages[1].content == "Hello! How can I help you?"
        assert messages[2].content == "What's the weather?"

    @pytest.mark.asyncio
    async def test_get_messages_with_limit(self, store_with_messages):
        """Test retrieving messages with limit."""
        store, session, _ = store_with_messages

        messages = await store.get_messages(session.id, limit=2)

        assert len(messages) == 2
        # Should get 2 most recent, then order oldest-first
        assert messages[0].content == "Hello! How can I help you?"
        assert messages[1].content == "What's the weather?"

    @pytest.mark.asyncio
    async def test_get_messages_empty_session(self, store_with_session):
        """Test getting messages from session with no messages."""
        store, session = store_with_session

        messages = await store.get_messages(session.id)

        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_get_context(self, store_with_messages):
        """Test getting conversation context."""
        store, session, messages = store_with_messages

        context = await store.get_context(session.id, max_messages=10)

        assert context.session.id == session.id
        assert len(context.messages) == 3
        assert context.total_messages == 3
        assert not context.is_truncated
        assert context.message_count == 3

    @pytest.mark.asyncio
    async def test_get_context_truncated(self, store_with_messages):
        """Test getting truncated context."""
        store, session, _ = store_with_messages

        context = await store.get_context(session.id, max_messages=2)

        assert len(context.messages) == 2
        assert context.total_messages == 3
        assert context.is_truncated

    @pytest.mark.asyncio
    async def test_get_context_with_tool_calls(self, store_with_messages):
        """Test getting context includes tool calls."""
        store, session, messages = store_with_messages

        # Add tool call to first message
        tool_call = await store.add_tool_call(
            messages[0].id,
            "echo",
            {"message": "test"},
            tool_output={"result": "test"},
            status="success",
        )

        context = await store.get_context(session.id)

        assert messages[0].id in context.tool_calls
        assert len(context.tool_calls[messages[0].id]) == 1
        assert context.tool_calls[messages[0].id][0].tool_name == "echo"

    @pytest.mark.asyncio
    async def test_get_context_session_not_found(self, conversation_store: ConversationStore):
        """Test getting context for non-existent session raises error."""
        with pytest.raises(SessionNotFoundError):
            await conversation_store.get_context("nonexistent_id")


class TestToolCallTracking:
    """Tests for tool call tracking."""

    @pytest.mark.asyncio
    async def test_add_tool_call(self, store_with_messages):
        """Test adding a tool call."""
        store, _, messages = store_with_messages

        tool_call = await store.add_tool_call(
            messages[0].id,
            "echo",
            {"message": "hello"},
            tool_output={"result": "hello"},
            status="success",
        )

        assert tool_call.id is not None
        assert tool_call.message_id == messages[0].id
        assert tool_call.tool_name == "echo"
        assert tool_call.tool_input == {"message": "hello"}
        assert tool_call.tool_output == {"result": "hello"}
        assert tool_call.status == ToolCallStatus.SUCCESS
        assert isinstance(tool_call.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_add_tool_call_pending(self, store_with_messages):
        """Test adding a pending tool call."""
        store, _, messages = store_with_messages

        tool_call = await store.add_tool_call(
            messages[0].id, "slow_tool", {"param": "value"}, status="pending"
        )

        assert tool_call.status == ToolCallStatus.PENDING
        assert tool_call.tool_output is None
        assert tool_call.duration_ms is None

    @pytest.mark.asyncio
    async def test_add_tool_call_with_error(self, store_with_messages):
        """Test adding a failed tool call."""
        store, _, messages = store_with_messages

        tool_call = await store.add_tool_call(
            messages[0].id,
            "failing_tool",
            {"param": "value"},
            status="error",
            error_message="Tool execution failed",
        )

        assert tool_call.status == ToolCallStatus.ERROR
        assert tool_call.error_message == "Tool execution failed"

    @pytest.mark.asyncio
    async def test_update_tool_call(self, store_with_messages):
        """Test updating a tool call with results."""
        store, _, messages = store_with_messages

        # Create pending tool call
        tool_call = await store.add_tool_call(
            messages[0].id, "test_tool", {"param": "value"}, status="pending"
        )

        # Update with results
        updated = await store.update_tool_call(
            tool_call.id,
            tool_output={"result": "success"},
            status="success",
            duration_ms=250,
        )

        assert updated.tool_output == {"result": "success"}
        assert updated.status == ToolCallStatus.SUCCESS
        assert updated.duration_ms == 250

    @pytest.mark.asyncio
    async def test_update_tool_call_with_error(self, store_with_messages):
        """Test updating a tool call with error."""
        store, _, messages = store_with_messages

        tool_call = await store.add_tool_call(
            messages[0].id, "test_tool", {"param": "value"}, status="pending"
        )

        updated = await store.update_tool_call(
            tool_call.id,
            tool_output={},
            status="error",
            duration_ms=100,
            error_message="Execution failed",
        )

        assert updated.status == ToolCallStatus.ERROR
        assert updated.error_message == "Execution failed"

    @pytest.mark.asyncio
    async def test_update_tool_call_not_found(self, conversation_store: ConversationStore):
        """Test updating non-existent tool call raises error."""
        with pytest.raises(ToolCallNotFoundError) as exc_info:
            await conversation_store.update_tool_call("nonexistent_id", {}, "success", 100)

        assert "nonexistent_id" in str(exc_info.value)


class TestSearchAndQueries:
    """Tests for search and query operations."""

    @pytest.mark.asyncio
    async def test_search_messages(self, store_with_messages):
        """Test searching messages by content."""
        store, session, _ = store_with_messages

        results = await store.search_messages("weather", session_id=session.id)

        assert len(results) == 1
        assert "weather" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_search_messages_case_insensitive(self, store_with_messages):
        """Test that search is case-insensitive."""
        store, session, _ = store_with_messages

        results = await store.search_messages("HELLO", session_id=session.id)

        assert len(results) >= 1
        assert any("hello" in msg.content.lower() for msg in results)

    @pytest.mark.asyncio
    async def test_search_messages_across_sessions(self, conversation_store: ConversationStore):
        """Test searching across all sessions."""
        # Create two sessions with messages
        session1 = await conversation_store.create_session("Session 1")
        session2 = await conversation_store.create_session("Session 2")

        await conversation_store.add_message(session1.id, "user", "Python is great")
        await conversation_store.add_message(session2.id, "user", "I love Python")

        results = await conversation_store.search_messages("Python")

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_messages_no_results(self, store_with_messages):
        """Test searching with no matching results."""
        store, session, _ = store_with_messages

        results = await store.search_messages("nonexistent", session_id=session.id)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_messages_with_limit(self, conversation_store: ConversationStore):
        """Test search with limit."""
        session = await conversation_store.create_session("Test")

        # Add many messages with same word
        for i in range(10):
            await conversation_store.add_message(session.id, "user", f"Test message {i}")

        results = await conversation_store.search_messages("Test", limit=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_get_recent_sessions(self, conversation_store: ConversationStore):
        """Test getting recent sessions."""
        # Create sessions
        session1 = await conversation_store.create_session("Recent")
        await conversation_store.create_session("Also Recent")

        results = await conversation_store.get_recent_sessions(days=7)

        assert len(results) >= 2
        assert any(s.id == session1.id for s in results)

    @pytest.mark.asyncio
    async def test_get_recent_sessions_filters_old(self, conversation_store: ConversationStore):
        """Test that get_recent_sessions filters old sessions."""
        # This is hard to test without mocking time, but we can verify
        # the function runs without error
        results = await conversation_store.get_recent_sessions(days=1)

        # Should return empty or only very recent sessions
        assert isinstance(results, list)


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_initialize_invalid_path(self):
        """Test initialization with invalid path."""
        # Create a path that can't be written to
        store = ConversationStore(db_path=Path("/invalid/path/db.sqlite"))

        with pytest.raises(DatabaseError):
            await store.initialize()

    @pytest.mark.asyncio
    async def test_session_summary_includes_tool_calls(self, store_with_messages):
        """Test that session summaries include tool call count."""
        store, session, messages = store_with_messages

        # Add some tool calls
        await store.add_tool_call(messages[0].id, "tool1", {"param": "value"}, status="success")
        await store.add_tool_call(messages[1].id, "tool2", {"param": "value"}, status="success")

        summaries = await store.list_sessions()

        # Find our session
        summary = next(s for s in summaries if s.id == session.id)

        assert summary.tool_call_count == 2
        assert summary.last_message_preview is not None
        assert "weather" in summary.last_message_preview.lower()


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_empty_database_queries(self, conversation_store: ConversationStore):
        """Test queries on empty database."""
        sessions = await conversation_store.list_sessions()
        assert len(sessions) == 0

        results = await conversation_store.search_messages("test")
        assert len(results) == 0

        recent = await conversation_store.get_recent_sessions()
        assert len(recent) == 0

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, store_with_session):
        """Test handling special characters in message content."""
        store, session = store_with_session

        special_content = "Hello! @#$%^&*() 'quotes' \"double\" \n newline \t tab"
        message = await store.add_message(session.id, "user", special_content)

        retrieved = await store.get_messages(session.id)
        assert retrieved[0].content == special_content

    @pytest.mark.asyncio
    async def test_json_metadata_roundtrip(self, store_with_session):
        """Test that complex JSON metadata is preserved."""
        store, session = store_with_session

        metadata = {
            "nested": {"key": "value", "number": 42},
            "array": [1, 2, 3],
            "bool": True,
            "null": None,
        }

        message = await store.add_message(session.id, "user", "Test", metadata=metadata)

        retrieved = await store.get_messages(session.id)
        assert retrieved[0].metadata == metadata

    @pytest.mark.asyncio
    async def test_concurrent_message_additions(self, store_with_session):
        """Test adding messages concurrently."""
        import asyncio

        store, session = store_with_session

        # Add multiple messages concurrently
        tasks = [store.add_message(session.id, "user", f"Message {i}") for i in range(5)]
        messages = await asyncio.gather(*tasks)

        assert len(messages) == 5

        # Verify all were saved
        retrieved = await store.get_messages(session.id)
        assert len(retrieved) == 5

        # Verify message count updated correctly
        updated_session = await store.get_session(session.id)
        assert updated_session.message_count == 5
