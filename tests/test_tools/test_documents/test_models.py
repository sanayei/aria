"""Tests for document processing models."""

import pytest
from datetime import date
from pathlib import Path

from aria.tools.documents.models import (
    OCRResult,
    ClassificationResult,
    ProcessedDocument,
)


class TestOCRResult:
    """Test OCRResult model."""

    def test_valid_ocr_result(self):
        """Test creating a valid OCR result."""
        result = OCRResult(
            text="Extracted text",
            confidence=0.95,
            page_count=2,
        )

        assert result.text == "Extracted text"
        assert result.confidence == 0.95
        assert result.page_count == 2

    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        OCRResult(text="test", confidence=0.5, page_count=1)
        OCRResult(text="test", confidence=0.0, page_count=1)
        OCRResult(text="test", confidence=1.0, page_count=1)

        # Invalid confidence
        with pytest.raises(ValueError):
            OCRResult(text="test", confidence=1.5, page_count=1)
        with pytest.raises(ValueError):
            OCRResult(text="test", confidence=-0.1, page_count=1)


class TestClassificationResult:
    """Test ClassificationResult model."""

    def test_valid_classification(self):
        """Test creating a valid classification."""
        result = ClassificationResult(
            person="Amir",
            category="medical",
            document_date=date(2025, 1, 15),
            sender="Kaiser Permanente",
            summary="Lab results showing normal values",
        )

        assert result.person == "Amir"
        assert result.category == "medical"
        assert result.document_date == date(2025, 1, 15)
        assert result.sender == "Kaiser Permanente"

    def test_optional_fields(self):
        """Test classification with optional fields as None."""
        result = ClassificationResult(
            person="unknown",
            category="other",
            document_date=None,
            sender=None,
            summary="Document with unclear details",
        )

        assert result.document_date is None
        assert result.sender is None


class TestProcessedDocument:
    """Test ProcessedDocument model."""

    def test_valid_processed_document(self):
        """Test creating a valid processed document."""
        doc = ProcessedDocument(
            source_path=Path("/tmp/scan.pdf"),
            text_content="Extracted text from OCR",
            person="Amir",
            category="medical",
            document_date=date(2025, 1, 15),
            sender="Kaiser Permanente",
            summary="Medical lab results",
            confidence=0.92,
            suggested_filename="2025-01-15_kaiser_permanente_medical.pdf",
            suggested_destination=Path("/home/user/documents/amir/medical/2025-01-15_kaiser_permanente_medical.pdf"),
            metadata={"page_count": 3},
        )

        assert doc.person == "Amir"
        assert doc.category == "medical"
        assert doc.confidence == 0.92
        assert doc.metadata["page_count"] == 3

    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        doc = ProcessedDocument(
            source_path=Path("/tmp/scan.pdf"),
            text_content="Text",
            person="Amir",
            category="other",
            summary="Summary",
            confidence=0.8,
            suggested_filename="file.pdf",
            suggested_destination=Path("/dest/file.pdf"),
        )

        assert doc.metadata == {}
