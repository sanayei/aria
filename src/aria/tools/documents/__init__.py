"""Document processing tools for OCR and intelligent organization.

This module provides tools for:
- OCR text extraction from scanned PDFs and images
- LLM-based document classification
- Automated document organization by person and category
- Batch processing of document inboxes
- PDF analysis and extraction
"""

from aria.tools.documents.classifier import DocumentClassifier
from aria.tools.documents.models import (
    ClassificationResult,
    OCRResult,
    PDFComparison,
    PDFDocument,
    PDFMetadata,
    PDFPage,
    PDFSearchMatch,
    PDFTable,
    ProcessedDocument,
)
from aria.tools.documents.ocr import OCRParams, OCRTool
from aria.tools.documents.pdf_extractor import PDFError, PDFExtractor
from aria.tools.documents.process_document import (
    ProcessDocumentParams,
    ProcessDocumentTool,
)
from aria.tools.documents.process_inbox import (
    ProcessInboxParams,
    ProcessInboxTool,
)
from aria.tools.documents.processor import DocumentProcessor

__all__ = [
    # Models - OCR/Classification
    "OCRResult",
    "ClassificationResult",
    "ProcessedDocument",
    # Models - PDF
    "PDFMetadata",
    "PDFTable",
    "PDFSearchMatch",
    "PDFPage",
    "PDFDocument",
    "PDFComparison",
    # Core classes
    "DocumentClassifier",
    "DocumentProcessor",
    "PDFExtractor",
    "PDFError",
    # Tools
    "OCRTool",
    "ProcessDocumentTool",
    "ProcessInboxTool",
    # Parameters
    "OCRParams",
    "ProcessDocumentParams",
    "ProcessInboxParams",
]
