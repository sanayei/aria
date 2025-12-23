"""Integration tests for history CLI commands."""

from datetime import datetime, UTC, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aria.memory import ConversationStore


@pytest.fixture
async def temp_conversation_store_with_data(tmp_path: Path):
    """Create a temporary conversation store with test data."""
    db_path = tmp_path / "test_conversations.db"
    store = ConversationStore(db_path)
    await store.initialize()

    # Create a few test sessions
    session1 = await store.create_session("Test Session 1")
    await store.add_message(session1.id, "user", "Hello, this is the first message")
    await store.add_message(session1.id, "assistant", "Hello! How can I help you?")
    await store.add_message(session1.id, "user", "Can you help me organize my tax documents?")
    await store.add_message(session1.id, "assistant", "Sure, I can help with that!")

    session2 = await store.create_session("Email Draft Review")
    await store.add_message(session2.id, "user", "Review this email draft")
    await store.add_message(session2.id, "assistant", "I'll review it for you")

    session3 = await store.create_session("Project Planning")
    await store.add_message(session3.id, "user", "Let's plan the project")
    await store.add_message(session3.id, "assistant", "Great! Let's start")

    yield store, [session1, session2, session3]
    await store.close()


class TestHistoryCommands:
    """Tests for history CLI commands."""

    @pytest.mark.asyncio
    async def test_format_relative_time(self):
        """Test relative time formatting."""
        from aria.main import format_relative_time

        now = datetime.now(UTC)

        # Just now
        assert format_relative_time(now) == "Just now"

        # Minutes ago
        assert "minute" in format_relative_time(now - timedelta(minutes=5))

        # Hours ago
        assert "hour" in format_relative_time(now - timedelta(hours=2))

        # Yesterday
        assert format_relative_time(now - timedelta(days=1)) == "Yesterday"

        # Days ago
        assert "day" in format_relative_time(now - timedelta(days=3))

        # Weeks ago
        assert "week" in format_relative_time(now - timedelta(weeks=2))

        # Months ago
        assert "month" in format_relative_time(now - timedelta(days=60))

        # Years ago
        assert "year" in format_relative_time(now - timedelta(days=400))

    def test_truncate_text(self):
        """Test text truncation."""
        from aria.main import truncate_text

        # Short text should not be truncated
        short = "Hello, world!"
        assert truncate_text(short, 80) == short

        # Long text should be truncated
        long = "x" * 100
        truncated = truncate_text(long, 80)
        assert len(truncated) == 80
        assert truncated.endswith("...")

        # Text exactly at max length should not be truncated
        exact = "x" * 80
        assert truncate_text(exact, 80) == exact

    @pytest.mark.asyncio
    async def test_history_list_basic(self, temp_conversation_store_with_data):
        """Test basic history list functionality."""
        from aria.main import _history_list

        store, sessions = temp_conversation_store_with_data

        # Mock console and settings
        from unittest.mock import MagicMock, patch
        import io
        from contextlib import redirect_stdout

        # Capture output
        f = io.StringIO()

        with (
            patch("aria.main.get_console") as mock_get_console,
            patch("aria.main.get_settings") as mock_get_settings,
        ):
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.conversation_db_path = store.db_path
            mock_get_settings.return_value = mock_settings

            # Mock console
            mock_console = MagicMock()
            mock_get_console.return_value = mock_console

            # Run the command
            await _history_list(limit=10, show_all=False, no_color=True)

            # Verify console was called
            assert mock_console.console.print.called

    @pytest.mark.asyncio
    async def test_history_show(self, temp_conversation_store_with_data):
        """Test history show functionality."""
        from aria.main import _history_show
        from unittest.mock import MagicMock, patch

        store, sessions = temp_conversation_store_with_data
        session = sessions[0]

        with (
            patch("aria.main.get_console") as mock_get_console,
            patch("aria.main.get_settings") as mock_get_settings,
        ):
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.conversation_db_path = store.db_path
            mock_get_settings.return_value = mock_settings

            # Mock console
            mock_console = MagicMock()
            mock_get_console.return_value = mock_console

            # Run the command
            await _history_show(session_id=session.id, messages=None, no_color=True)

            # Verify console was called
            assert mock_console.console.print.called

    @pytest.mark.asyncio
    async def test_history_search(self, temp_conversation_store_with_data):
        """Test history search functionality."""
        from aria.main import _history_search
        from unittest.mock import MagicMock, patch

        store, sessions = temp_conversation_store_with_data

        with (
            patch("aria.main.get_console") as mock_get_console,
            patch("aria.main.get_settings") as mock_get_settings,
        ):
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.conversation_db_path = store.db_path
            mock_get_settings.return_value = mock_settings

            # Mock console
            mock_console = MagicMock()
            mock_get_console.return_value = mock_console

            # Run the command
            await _history_search(query="tax", session_id=None, limit=10, no_color=True)

            # Verify console was called
            assert mock_console.console.print.called

    @pytest.mark.asyncio
    async def test_history_delete(self, temp_conversation_store_with_data):
        """Test history delete functionality."""
        from aria.main import _history_delete
        from unittest.mock import MagicMock, patch

        store, sessions = temp_conversation_store_with_data
        session = sessions[2]  # Use the last session

        with (
            patch("aria.main.get_console") as mock_get_console,
            patch("aria.main.get_settings") as mock_get_settings,
        ):
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.conversation_db_path = store.db_path
            mock_get_settings.return_value = mock_settings

            # Mock console
            mock_console = MagicMock()
            mock_get_console.return_value = mock_console

            # Run the command with force=True to skip confirmation
            await _history_delete(session_id=session.id, force=True, no_color=True)

            # Verify session was deleted
            deleted_session = await store.get_session(session.id)
            assert deleted_session is None

    @pytest.mark.asyncio
    async def test_history_export(self, temp_conversation_store_with_data, tmp_path):
        """Test history export functionality."""
        from aria.main import _history_export
        from unittest.mock import MagicMock, patch

        store, sessions = temp_conversation_store_with_data
        session = sessions[0]

        output_file = tmp_path / "export.md"

        with (
            patch("aria.main.get_console") as mock_get_console,
            patch("aria.main.get_settings") as mock_get_settings,
        ):
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.conversation_db_path = store.db_path
            mock_get_settings.return_value = mock_settings

            # Mock console
            mock_console = MagicMock()
            mock_get_console.return_value = mock_console

            # Run the command
            await _history_export(session_id=session.id, output=str(output_file), no_color=True)

            # Verify file was created
            assert output_file.exists()

            # Verify content contains session info
            content = output_file.read_text()
            assert session.title in content
            assert session.id in content
            assert "Hello, this is the first message" in content

    @pytest.mark.asyncio
    async def test_history_list_empty(self, tmp_path):
        """Test history list with no sessions."""
        from aria.main import _history_list
        from unittest.mock import MagicMock, patch

        # Create empty store
        db_path = tmp_path / "empty.db"
        store = ConversationStore(db_path)
        await store.initialize()

        with (
            patch("aria.main.get_console") as mock_get_console,
            patch("aria.main.get_settings") as mock_get_settings,
        ):
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.conversation_db_path = store.db_path
            mock_get_settings.return_value = mock_settings

            # Mock console
            mock_console = MagicMock()
            mock_get_console.return_value = mock_console

            # Run the command
            await _history_list(limit=10, show_all=False, no_color=True)

            # Verify appropriate message was displayed
            assert mock_console.console.print.called

        await store.close()

    @pytest.mark.asyncio
    async def test_history_show_not_found(self, tmp_path):
        """Test history show with non-existent session."""
        from aria.main import _history_show
        from unittest.mock import MagicMock, patch

        # Create empty store
        db_path = tmp_path / "empty.db"
        store = ConversationStore(db_path)
        await store.initialize()

        with (
            patch("aria.main.get_console") as mock_get_console,
            patch("aria.main.get_settings") as mock_get_settings,
        ):
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.conversation_db_path = store.db_path
            mock_get_settings.return_value = mock_settings

            # Mock console
            mock_console = MagicMock()
            mock_get_console.return_value = mock_console

            # Run the command with invalid session_id - should exit
            with pytest.raises(SystemExit):
                await _history_show(session_id="invalid_id", messages=None, no_color=True)

        await store.close()

    @pytest.mark.asyncio
    async def test_history_search_no_results(self, temp_conversation_store_with_data):
        """Test history search with no results."""
        from aria.main import _history_search
        from unittest.mock import MagicMock, patch

        store, sessions = temp_conversation_store_with_data

        with (
            patch("aria.main.get_console") as mock_get_console,
            patch("aria.main.get_settings") as mock_get_settings,
        ):
            # Mock settings
            mock_settings = MagicMock()
            mock_settings.conversation_db_path = store.db_path
            mock_get_settings.return_value = mock_settings

            # Mock console
            mock_console = MagicMock()
            mock_get_console.return_value = mock_console

            # Run the command with query that won't match
            await _history_search(
                query="nonexistent_query_12345", session_id=None, limit=10, no_color=True
            )

            # Verify console was called (should show "no matches" message)
            assert mock_console.console.print.called
