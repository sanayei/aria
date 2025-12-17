"""Tests for email tools (ListEmailsTool, SearchEmailsTool, ReadEmailTool)."""

from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aria.tools.email.tools import (
    ListEmailsTool,
    ListEmailsParams,
    SearchEmailsTool,
    SearchEmailsParams,
    ReadEmailTool,
    ReadEmailParams,
)
from aria.tools.email.models import EmailSummary, EmailDetail, AttachmentInfo
from aria.tools.base import RiskLevel


@pytest.fixture
def mock_email_summary():
    """Create a mock email summary."""
    return EmailSummary(
        id="test-email-1",
        thread_id="thread-1",
        subject="Test Email",
        sender="test@example.com",
        snippet="This is a test email",
        date=datetime.now(UTC),
        labels=["INBOX", "UNREAD"],
        is_unread=True,
        has_attachments=False,
    )


@pytest.fixture
def mock_email_detail():
    """Create a mock email detail."""
    return EmailDetail(
        id="test-email-1",
        thread_id="thread-1",
        subject="Test Email",
        sender="test@example.com",
        to=["recipient@example.com"],
        cc=[],
        bcc=[],
        body_text="This is the email body",
        body_html="<p>This is the email body</p>",
        snippet="This is a test email",
        date=datetime.now(UTC),
        labels=["INBOX", "UNREAD"],
        is_unread=True,
        attachments=[],
    )


class TestListEmailsTool:
    """Tests for ListEmailsTool."""

    def test_init(self):
        """Test tool initialization."""
        tool = ListEmailsTool()
        assert tool.name == "list_emails"
        assert tool.risk_level == RiskLevel.LOW
        assert tool.parameters_schema == ListEmailsParams

    def test_input_validation(self):
        """Test input validation."""
        # Valid input
        input_data = {"folder": "INBOX", "max_results": 10, "unread_only": False}
        validated = ListEmailsParams(**input_data)
        assert validated.folder == "INBOX"
        assert validated.max_results == 10
        assert validated.unread_only is False

        # Default values
        validated = ListEmailsParams()
        assert validated.folder == "INBOX"
        assert validated.max_results == 10
        assert validated.unread_only is False

        # Invalid max_results (too high)
        with pytest.raises(Exception):  # Pydantic validation error
            ListEmailsParams(max_results=100)

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_email_summary):
        """Test successful email listing."""
        tool = ListEmailsTool()

        # Mock the Gmail client
        with patch.object(tool, '_get_gmail_client') as mock_get_client, \
             patch.object(tool, '_get_email_cache') as mock_get_cache:

            mock_client = AsyncMock()
            mock_client.list_messages = AsyncMock(return_value=[mock_email_summary])
            mock_get_client.return_value = mock_client

            mock_cache = AsyncMock()
            mock_cache.put_many = AsyncMock()
            mock_get_cache.return_value = mock_cache

            # Execute tool
            input_data = ListEmailsParams(folder="INBOX", max_results=10)
            result = await tool.execute(input_data)

            # Verify result
            assert result.success
            assert result.data["count"] == 1
            assert len(result.data["emails"]) == 1
            assert result.data["emails"][0]["subject"] == "Test Email"

    @pytest.mark.asyncio
    async def test_execute_no_emails(self):
        """Test listing when no emails found."""
        tool = ListEmailsTool()

        with patch.object(tool, '_get_gmail_client') as mock_get_client, \
             patch.object(tool, '_get_email_cache') as mock_get_cache:

            mock_client = AsyncMock()
            mock_client.list_messages = AsyncMock(return_value=[])
            mock_get_client.return_value = mock_client

            mock_cache = AsyncMock()
            mock_get_cache.return_value = mock_cache

            input_data = ListEmailsParams(folder="INBOX")
            result = await tool.execute(input_data)

            assert result.success
            assert result.data["count"] == 0
            assert len(result.data["emails"]) == 0

    @pytest.mark.asyncio
    async def test_execute_error(self):
        """Test error handling."""
        tool = ListEmailsTool()

        with patch.object(tool, '_get_gmail_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.list_messages = AsyncMock(side_effect=Exception("API Error"))
            mock_get_client.return_value = mock_client

            input_data = ListEmailsParams(folder="INBOX")
            result = await tool.execute(input_data)

            assert not result.success
            assert "Failed to list emails" in result.error


class TestSearchEmailsTool:
    """Tests for SearchEmailsTool."""

    def test_init(self):
        """Test tool initialization."""
        tool = SearchEmailsTool()
        assert tool.name == "search_emails"
        assert tool.risk_level == RiskLevel.LOW
        assert tool.parameters_schema == SearchEmailsParams

    def test_input_validation(self):
        """Test input validation."""
        # Valid input
        input_data = {"query": "from:test@example.com", "max_results": 10}
        validated = SearchEmailsParams(**input_data)
        assert validated.query == "from:test@example.com"
        assert validated.max_results == 10

        # Required field
        with pytest.raises(Exception):  # Missing query
            SearchEmailsParams(max_results=10)

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_email_summary):
        """Test successful email search."""
        tool = SearchEmailsTool()

        with patch.object(tool, '_get_gmail_client') as mock_get_client, \
             patch.object(tool, '_get_email_cache') as mock_get_cache:

            mock_client = AsyncMock()
            mock_client.search_messages = AsyncMock(return_value=[mock_email_summary])
            mock_get_client.return_value = mock_client

            mock_cache = AsyncMock()
            mock_get_cache.return_value = mock_cache

            input_data = SearchEmailsParams(query="from:test@example.com")
            result = await tool.execute(input_data)

            assert result.success
            assert result.data["count"] == 1
            assert result.data["query"] == "from:test@example.com"

    @pytest.mark.asyncio
    async def test_execute_no_results(self):
        """Test search with no results."""
        tool = SearchEmailsTool()

        with patch.object(tool, '_get_gmail_client') as mock_get_client, \
             patch.object(tool, '_get_email_cache') as mock_get_cache:

            mock_client = AsyncMock()
            mock_client.search_messages = AsyncMock(return_value=[])
            mock_get_client.return_value = mock_client

            mock_cache = AsyncMock()
            mock_get_cache.return_value = mock_cache

            input_data = SearchEmailsParams(query="subject:nonexistent")
            result = await tool.execute(input_data)

            assert result.success
            assert result.data["count"] == 0


class TestReadEmailTool:
    """Tests for ReadEmailTool."""

    def test_init(self):
        """Test tool initialization."""
        tool = ReadEmailTool()
        assert tool.name == "read_email"
        # Risk level is MEDIUM because it can modify state
        assert tool.risk_level == RiskLevel.MEDIUM
        assert tool.parameters_schema == ReadEmailParams

    def test_input_validation(self):
        """Test input validation."""
        # Valid input
        input_data = {"email_id": "test-email-1", "mark_as_read": False}
        validated = ReadEmailParams(**input_data)
        assert validated.email_id == "test-email-1"
        assert validated.mark_as_read is False

        # Required field
        with pytest.raises(Exception):  # Missing email_id
            ReadEmailParams(mark_as_read=True)

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_email_detail):
        """Test successful email reading."""
        tool = ReadEmailTool()

        with patch.object(tool, '_get_gmail_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get_message = AsyncMock(return_value=mock_email_detail)
            mock_get_client.return_value = mock_client

            input_data = ReadEmailParams(email_id="test-email-1", mark_as_read=False)
            result = await tool.execute(input_data)

            assert result.success
            assert result.data["id"] == "test-email-1"
            assert result.data["subject"] == "Test Email"
            assert "body" in result.data

    @pytest.mark.asyncio
    async def test_execute_mark_as_read(self, mock_email_detail):
        """Test reading and marking email as read."""
        tool = ReadEmailTool()

        with patch.object(tool, '_get_gmail_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get_message = AsyncMock(return_value=mock_email_detail)
            mock_client.mark_as_read = AsyncMock(return_value=True)
            mock_get_client.return_value = mock_client

            input_data = ReadEmailParams(email_id="test-email-1", mark_as_read=True)
            result = await tool.execute(input_data)

            assert result.success
            # Verify mark_as_read was called
            mock_client.mark_as_read.assert_called_once_with("test-email-1")

    @pytest.mark.asyncio
    async def test_execute_error(self):
        """Test error handling."""
        tool = ReadEmailTool()

        with patch.object(tool, '_get_gmail_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get_message = AsyncMock(side_effect=Exception("API Error"))
            mock_get_client.return_value = mock_client

            input_data = ReadEmailParams(email_id="test-email-1")
            result = await tool.execute(input_data)

            assert not result.success
            assert "Failed to read email" in result.error
