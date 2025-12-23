"""DuckDuckGo search client for ARIA."""

import asyncio
from typing import Any

from ddgs import DDGS

from aria.logging import get_logger
from aria.tools.search.models import SearchResult

logger = get_logger("aria.tools.search.client")


class SearchClientError(Exception):
    """Exception raised when search operations fail."""

    pass


class DuckDuckGoClient:
    """Async client for DuckDuckGo web search.

    This client wraps the ddgs library and provides
    a clean interface for searching the web.
    """

    def __init__(self):
        """Initialize the DuckDuckGo client."""
        pass

    async def search(
        self,
        query: str,
        max_results: int = 5,
        region: str = "wt-wt",
        time_range: str | None = None,
    ) -> list[SearchResult]:
        """Search the web using DuckDuckGo.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (1-20)
            region: Region code (e.g., 'wt-wt' for worldwide, 'us-en' for US)
            time_range: Time range filter ('d'=day, 'w'=week, 'm'=month, 'y'=year)

        Returns:
            list[SearchResult]: List of search results

        Raises:
            SearchClientError: If search fails
        """
        try:
            logger.info(f"Searching DuckDuckGo: '{query}' (max {max_results} results)")

            # Run sync search in thread pool
            raw_results = await asyncio.to_thread(
                self._sync_search,
                query=query,
                max_results=max_results,
                region=region,
                time_range=time_range,
            )

            # Parse results
            search_results = []
            for raw in raw_results:
                try:
                    # Extract domain from URL for source
                    url = raw.get("href", raw.get("link", ""))
                    source = self._extract_domain(url)

                    result = SearchResult(
                        title=raw.get("title", "No title"),
                        url=url,
                        snippet=raw.get("body", raw.get("snippet", "")),
                        source=source,
                    )
                    search_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to parse search result: {e}")
                    continue

            logger.info(f"Found {len(search_results)} results for '{query}'")
            return search_results

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            raise SearchClientError(f"Search failed: {e}") from e

    def _sync_search(
        self,
        query: str,
        max_results: int,
        region: str,
        time_range: str | None,
    ) -> list[dict[str, Any]]:
        """Synchronous search implementation.

        Args:
            query: Search query
            max_results: Maximum results to return
            region: Region code
            time_range: Time range filter

        Returns:
            list[dict]: Raw search results
        """
        with DDGS() as ddgs:
            # Build search parameters
            params: dict[str, Any] = {
                "region": region,
                "max_results": max_results,
            }

            if time_range:
                params["timelimit"] = time_range

            # Execute search - query is now a positional argument
            results = ddgs.text(query, **params)
            return list(results)

    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL.

        Args:
            url: Full URL

        Returns:
            str: Domain name (e.g., 'example.com')
        """
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path
            # Remove 'www.' prefix if present
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return "unknown"
