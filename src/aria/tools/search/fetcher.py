"""Web page fetcher and content extractor for ARIA."""

import asyncio
from typing import Any

import httpx
from bs4 import BeautifulSoup

from aria.logging import get_logger
from aria.tools.search.models import WebPageContent

logger = get_logger("aria.tools.search.fetcher")


class FetcherError(Exception):
    """Exception raised when fetching or parsing fails."""
    pass


class WebPageFetcher:
    """Async web page fetcher with content extraction.

    This fetcher downloads web pages and extracts the main text content,
    stripping out navigation, ads, and other non-content elements.
    """

    def __init__(self, timeout: int = 30, user_agent: str | None = None):
        """Initialize the fetcher.

        Args:
            timeout: Request timeout in seconds
            user_agent: Optional custom user agent string
        """
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (compatible; ARIA/1.0; +https://github.com/aria)"
        )

    async def fetch(
        self,
        url: str,
        extract_text: bool = True,
        max_length: int = 10000,
    ) -> WebPageContent:
        """Fetch and extract content from a web page.

        Args:
            url: URL to fetch
            extract_text: If True, extract and clean text content
            max_length: Maximum characters to return

        Returns:
            WebPageContent: Extracted page content

        Raises:
            FetcherError: If fetch or parsing fails
        """
        try:
            logger.info(f"Fetching web page: {url}")

            # Fetch page
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                html = response.text

            if not extract_text:
                # Return raw HTML (truncated)
                truncated = len(html) > max_length
                text = html[:max_length]
                return WebPageContent(
                    url=url,
                    title=None,
                    text=text,
                    length=len(text),
                    truncated=truncated,
                )

            # Parse HTML
            soup = BeautifulSoup(html, "lxml")

            # Extract title
            title = None
            if soup.title:
                title = soup.title.string

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

            # Extract text
            text = soup.get_text(separator="\n", strip=True)

            # Clean up text
            text = self._clean_text(text)

            # Truncate if needed
            truncated = len(text) > max_length
            if truncated:
                text = text[:max_length]

            logger.info(f"Extracted {len(text)} characters from {url}")

            return WebPageContent(
                url=url,
                title=title,
                text=text,
                length=len(text),
                truncated=truncated,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise FetcherError(f"HTTP {e.response.status_code}: {e}") from e
        except httpx.RequestError as e:
            logger.error(f"Request error fetching {url}: {e}")
            raise FetcherError(f"Request failed: {e}") from e
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise FetcherError(f"Fetch failed: {e}") from e

    def _clean_text(self, text: str) -> str:
        """Clean extracted text.

        Args:
            text: Raw text

        Returns:
            str: Cleaned text
        """
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split("\n")]
        # Remove empty lines
        lines = [line for line in lines if line]
        # Join with single newlines
        text = "\n".join(lines)

        return text
