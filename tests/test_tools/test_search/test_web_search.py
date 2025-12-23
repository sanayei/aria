"""Tests for web search tools."""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from aria.tools.search.web import (
    WebSearchTool,
    WebSearchParams,
    FetchWebPageTool,
    FetchWebPageParams,
)
from aria.tools.search.models import SearchResult, WebPageContent
from aria.tools.base import RiskLevel


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    def test_init(self):
        """Test tool initialization."""
        tool = WebSearchTool()
        assert tool.name == "web_search"
        assert tool.risk_level == RiskLevel.LOW
        assert tool.parameters_schema == WebSearchParams

    def test_input_validation(self):
        """Test input validation."""
        # Valid input
        input_data = {"query": "python programming", "max_results": 5}
        validated = WebSearchParams(**input_data)
        assert validated.query == "python programming"
        assert validated.max_results == 5
        assert validated.region == "wt-wt"  # default
        assert validated.time_range is None  # default

        # Invalid max_results (too high)
        with pytest.raises(Exception):  # Pydantic validation error
            WebSearchParams(query="test", max_results=100)

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful web search."""
        tool = WebSearchTool()

        # Mock search results
        mock_results = [
            SearchResult(
                title="Test Result 1",
                url="https://example.com/1",
                snippet="This is a test snippet 1",
                source="example.com",
            ),
            SearchResult(
                title="Test Result 2",
                url="https://example.com/2",
                snippet="This is a test snippet 2",
                source="example.com",
            ),
        ]

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.search = AsyncMock(return_value=mock_results)
            mock_get_client.return_value = mock_client

            input_data = WebSearchParams(query="test query", max_results=5)
            result = await tool.execute(input_data)

            assert result.success
            assert result.data["count"] == 2
            assert result.data["query"] == "test query"
            assert len(result.data["results"]) == 2
            assert result.data["results"][0]["title"] == "Test Result 1"
            mock_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_no_results(self):
        """Test search with no results."""
        tool = WebSearchTool()

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.search = AsyncMock(return_value=[])
            mock_get_client.return_value = mock_client

            input_data = WebSearchParams(query="nonexistent query")
            result = await tool.execute(input_data)

            assert result.success
            assert result.data["count"] == 0
            assert len(result.data["results"]) == 0

    @pytest.mark.asyncio
    async def test_execute_error(self):
        """Test error handling."""
        tool = WebSearchTool()

        with patch.object(tool, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.search = AsyncMock(side_effect=Exception("Search failed"))
            mock_get_client.return_value = mock_client

            input_data = WebSearchParams(query="test")
            result = await tool.execute(input_data)

            assert not result.success
            assert "Search failed" in result.error

    def test_confirmation_message(self):
        """Test confirmation message generation."""
        tool = WebSearchTool()
        params = WebSearchParams(query="python", max_results=10, time_range="w")
        confirmation = tool.get_confirmation_message(params)
        assert "python" in confirmation
        assert "10 results" in confirmation
        assert "week" in confirmation


class TestFetchWebPageTool:
    """Tests for FetchWebPageTool."""

    def test_init(self):
        """Test tool initialization."""
        tool = FetchWebPageTool()
        assert tool.name == "fetch_webpage"
        assert tool.risk_level == RiskLevel.LOW
        assert tool.parameters_schema == FetchWebPageParams

    def test_input_validation(self):
        """Test input validation."""
        # Valid input
        input_data = {
            "url": "https://example.com",
            "extract_text": True,
            "max_length": 5000,
        }
        validated = FetchWebPageParams(**input_data)
        assert validated.url == "https://example.com"
        assert validated.extract_text is True
        assert validated.max_length == 5000

        # Required field
        with pytest.raises(Exception):  # Missing url
            FetchWebPageParams(extract_text=True)

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful webpage fetch."""
        tool = FetchWebPageTool()

        # Mock fetch result
        mock_content = WebPageContent(
            url="https://example.com",
            title="Example Page",
            text="This is the main content of the page.",
            length=40,
            truncated=False,
        )

        with patch.object(tool, "_get_fetcher") as mock_get_fetcher:
            mock_fetcher = MagicMock()
            mock_fetcher.fetch = AsyncMock(return_value=mock_content)
            mock_get_fetcher.return_value = mock_fetcher

            input_data = FetchWebPageParams(url="https://example.com")
            result = await tool.execute(input_data)

            assert result.success
            assert result.data["url"] == "https://example.com"
            assert result.data["title"] == "Example Page"
            assert result.data["text"] == "This is the main content of the page."
            assert result.data["length"] == 40
            assert result.data["truncated"] is False
            mock_fetcher.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_error(self):
        """Test error handling."""
        tool = FetchWebPageTool()

        with patch.object(tool, "_get_fetcher") as mock_get_fetcher:
            mock_fetcher = MagicMock()
            mock_fetcher.fetch = AsyncMock(side_effect=Exception("HTTP 404"))
            mock_get_fetcher.return_value = mock_fetcher

            input_data = FetchWebPageParams(url="https://example.com/notfound")
            result = await tool.execute(input_data)

            assert not result.success
            assert "HTTP 404" in result.error

    def test_confirmation_message(self):
        """Test confirmation message generation."""
        tool = FetchWebPageTool()
        params = FetchWebPageParams(url="https://example.com", extract_text=True)
        confirmation = tool.get_confirmation_message(params)
        assert "text content" in confirmation
        assert "https://example.com" in confirmation

        params2 = FetchWebPageParams(url="https://example.com", extract_text=False)
        confirmation2 = tool.get_confirmation_message(params2)
        assert "raw HTML" in confirmation2
