"""Data models for document processing."""

from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field


class OCRResult(BaseModel):
    """Result from OCR extraction."""

    text: str = Field(description="Extracted text from the document")
    confidence: float = Field(
        description="Average confidence score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    page_count: int = Field(description="Number of pages processed", ge=1)


class ClassificationResult(BaseModel):
    """Result from LLM classification of a document."""

    person: str = Field(description="Family member this document belongs to")
    category: str = Field(description="Document category")
    document_date: date | None = Field(
        default=None,
        description="Date ON the document (not today)",
    )
    sender: str | None = Field(
        default=None,
        description="Organization/company that sent the document",
    )
    summary: str = Field(description="One-sentence summary of the document")


class ProcessedDocument(BaseModel):
    """Complete result of processing a scanned document."""

    source_path: Path = Field(description="Original file path")
    text_content: str = Field(description="Extracted OCR text")
    person: str = Field(description="Family member name")
    category: str = Field(description="Document category")
    document_date: date | None = Field(
        default=None,
        description="Date from the document",
    )
    sender: str | None = Field(
        default=None,
        description="Document sender/organization",
    )
    summary: str = Field(description="Document summary")
    confidence: float = Field(
        description="OCR confidence score",
        ge=0.0,
        le=1.0,
    )
    suggested_filename: str = Field(description="Generated filename")
    suggested_destination: Path = Field(description="Full destination path")
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata",
    )
