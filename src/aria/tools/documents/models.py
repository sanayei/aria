"""Data models for document processing."""

from datetime import date, datetime
from pathlib import Path
from typing import Any

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


# ============================================================================
# PDF-Specific Models
# ============================================================================


class PDFMetadata(BaseModel):
    """Metadata extracted from a PDF document."""

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: datetime | None = None
    modification_date: datetime | None = None
    format: str | None = None
    encrypted: bool = False


class PDFTable(BaseModel):
    """A table extracted from a PDF."""

    page_number: int
    table_number: int  # Table index on the page
    rows: int
    columns: int
    headers: list[str] | None = None
    data: list[list[str]]  # 2D array of cell values
    bbox: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1)


class PDFSearchMatch(BaseModel):
    """A search match within a PDF."""

    page_number: int
    text: str
    context_before: str
    context_after: str
    bbox: tuple[float, float, float, float] | None = None  # Location on page


class PDFPage(BaseModel):
    """Information about a single PDF page."""

    page_number: int
    text: str
    width: float
    height: float
    rotation: int = 0
    images_count: int = 0


class PDFDocument(BaseModel):
    """Complete PDF document information."""

    file_path: str
    metadata: PDFMetadata
    page_count: int
    pages: list[PDFPage] = Field(default_factory=list)
    full_text: str = ""
    tables: list[PDFTable] = Field(default_factory=list)
    file_size_bytes: int = 0
    is_encrypted: bool = False


class PDFComparison(BaseModel):
    """Comparison result between two PDFs."""

    file1: str
    file2: str
    pages_differ: list[int]
    text_similarity: float  # 0.0 to 1.0
    structural_differences: list[str]
    added_pages: list[int]
    removed_pages: list[int]
