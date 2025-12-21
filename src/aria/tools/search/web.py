"""Web search tools for ARIA agent."""

from pydantic import BaseModel, Field

from aria.logging import get_logger
from aria.tools.base import BaseTool, ToolResult, RiskLevel
from aria.tools.search.client import DuckDuckGoClient
from aria.tools.search.fetcher import WebPageFetcher

logger = get_logger("aria.tools.search.web")


class WebSearchParams(BaseModel):
    """Input for WebSearchTool."""
    query: str = Field(
        description="Search query string"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return"
    )
    region: str = Field(
        default="wt-wt",
        description="Region code (e.g., 'wt-wt' for worldwide, 'us-en' for US)"
    )
    time_range: str | None = Field(
        default=None,
        description="Time range filter: 'd' (day), 'w' (week), 'm' (month), 'y' (year)"
    )


class WebSearchTool(BaseTool[WebSearchParams]):
    """Search the web using DuckDuckGo.

    This tool searches the web and returns relevant results with titles,
    URLs, and snippets. No API key required.
    """

    name = "web_search"
    description = (
        "Search the web using DuckDuckGo. "
        "Returns relevant results with titles, URLs, and snippets. "
        "Use this ONLY when the user explicitly asks to search the web or look up information online. "
        "DO NOT use this to research email subjects or content already provided by other tools. "
        "Supports time range filtering and region selection."
    )
    risk_level = RiskLevel.LOW
    parameters_schema = WebSearchParams

    def __init__(self):
        """Initialize the web search tool."""
        super().__init__()
        self._client: DuckDuckGoClient | None = None

    def _get_client(self) -> DuckDuckGoClient:
        """Get or create DuckDuckGo client."""
        if self._client is None:
            self._client = DuckDuckGoClient()
        return self._client

    def get_confirmation_message(self, params: WebSearchParams) -> str:
        """Get confirmation message.

        Args:
            params: Validated parameters

        Returns:
            str: Confirmation message
        """
        msg = f"Search web for: '{params.query}' (max {params.max_results} results)"
        if params.time_range:
            time_labels = {"d": "day", "w": "week", "m": "month", "y": "year"}
            time_label = time_labels.get(params.time_range, params.time_range)
            msg += f" from last {time_label}"
        return msg

    async def execute(self, params: WebSearchParams) -> ToolResult:
        """Execute the web search tool.

        Args:
            params: Validated input parameters

        Returns:
            ToolResult: Search results
        """
        try:
            client = self._get_client()

            # Perform search
            results = await client.search(
                query=params.query,
                max_results=params.max_results,
                region=params.region,
                time_range=params.time_range,
            )

            if not results:
                return ToolResult.success_result(
                    data={
                        "query": params.query,
                        "count": 0,
                        "results": [],
                        "message": f"No results found for: {params.query}",
                    }
                )

            # Convert to dict for JSON serialization
            result_dicts = [
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "source": result.source,
                }
                for result in results
            ]

            return ToolResult.success_result(
                data={
                    "query": params.query,
                    "count": len(results),
                    "results": result_dicts,
                }
            )

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return ToolResult.error_result(error=f"Search failed: {e}")


class FetchWebPageParams(BaseModel):
    """Input for FetchWebPageTool."""
    url: str = Field(
        description="URL of the web page to fetch"
    )
    extract_text: bool = Field(
        default=True,
        description="If True, extract and clean text content; if False, return raw HTML"
    )
    max_length: int = Field(
        default=10000,
        ge=100,
        le=50000,
        description="Maximum characters to return"
    )


class FetchWebPageTool(BaseTool[FetchWebPageParams]):
    """Fetch and extract content from a web page.

    This tool downloads a web page and extracts the main text content,
    removing navigation, ads, and other non-content elements.
    """

    name = "fetch_webpage"
    description = (
        "Fetch and extract content from a web page. "
        "Downloads the page and extracts main text content, removing navigation and ads. "
        "Use this to read articles, documentation, or other web content. "
        "Optionally return raw HTML instead of cleaned text."
    )
    risk_level = RiskLevel.LOW
    parameters_schema = FetchWebPageParams

    def __init__(self):
        """Initialize the fetch webpage tool."""
        super().__init__()
        self._fetcher: WebPageFetcher | None = None

    def _get_fetcher(self) -> WebPageFetcher:
        """Get or create web page fetcher."""
        if self._fetcher is None:
            self._fetcher = WebPageFetcher()
        return self._fetcher

    def get_confirmation_message(self, params: FetchWebPageParams) -> str:
        """Get confirmation message.

        Args:
            params: Validated parameters

        Returns:
            str: Confirmation message
        """
        content_type = "text content" if params.extract_text else "raw HTML"
        return f"Fetch {content_type} from: {params.url}"

    async def execute(self, params: FetchWebPageParams) -> ToolResult:
        """Execute the fetch webpage tool.

        Args:
            params: Validated input parameters

        Returns:
            ToolResult: Page content
        """
        try:
            fetcher = self._get_fetcher()

            # Fetch page
            content = await fetcher.fetch(
                url=params.url,
                extract_text=params.extract_text,
                max_length=params.max_length,
            )

            return ToolResult.success_result(
                data={
                    "url": content.url,
                    "title": content.title,
                    "text": content.text,
                    "length": content.length,
                    "truncated": content.truncated,
                }
            )

        except Exception as e:
            logger.error(f"Failed to fetch webpage: {e}")
            return ToolResult.error_result(error=f"Fetch failed: {e}")
