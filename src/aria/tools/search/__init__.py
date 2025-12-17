"""Web search tools for ARIA."""

from aria.tools.search.client import DuckDuckGoClient, SearchClientError
from aria.tools.search.fetcher import WebPageFetcher, FetcherError
from aria.tools.search.models import SearchResult, WebPageContent
from aria.tools.search.web import WebSearchTool, FetchWebPageTool

__all__ = [
    # Client
    "DuckDuckGoClient",
    "SearchClientError",
    # Fetcher
    "WebPageFetcher",
    "FetcherError",
    # Models
    "SearchResult",
    "WebPageContent",
    # Tools
    "WebSearchTool",
    "FetchWebPageTool",
]
