"""Data models for web search results."""

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single web search result."""

    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the result")
    snippet: str = Field(description="Text snippet from the result")
    source: str = Field(description="Source domain (e.g., 'example.com')")

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.title}\n{self.url}\n{self.snippet[:100]}..."


class WebPageContent(BaseModel):
    """Content extracted from a web page."""

    url: str = Field(description="URL of the fetched page")
    title: str | None = Field(default=None, description="Page title")
    text: str = Field(description="Extracted text content")
    length: int = Field(description="Length of extracted text in characters")
    truncated: bool = Field(default=False, description="Whether content was truncated")
