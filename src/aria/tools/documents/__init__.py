"""Document processing tools for OCR and intelligent organization.

This module provides tools for:
- OCR text extraction from scanned PDFs and images
- LLM-based document classification
- Automated document organization by person and category
- Batch processing of document inboxes
"""

from aria.tools.documents.models import (
    OCRResult,
    ClassificationResult,
    ProcessedDocument,
)
from aria.tools.documents.classifier import DocumentClassifier
from aria.tools.documents.processor import DocumentProcessor
from aria.tools.documents.ocr import OCRTool, OCRParams
from aria.tools.documents.process_document import (
    ProcessDocumentTool,
    ProcessDocumentParams,
)
from aria.tools.documents.process_inbox import (
    ProcessInboxTool,
    ProcessInboxParams,
)

__all__ = [
    # Models
    "OCRResult",
    "ClassificationResult",
    "ProcessedDocument",
    # Core classes
    "DocumentClassifier",
    "DocumentProcessor",
    # Tools
    "OCRTool",
    "ProcessDocumentTool",
    "ProcessInboxTool",
    # Parameters
    "OCRParams",
    "ProcessDocumentParams",
    "ProcessInboxParams",
]
