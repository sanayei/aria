"""Tests for document processor."""

import pytest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from aria.config import Settings
from aria.tools.documents.processor import DocumentProcessor
from aria.tools.documents.models import ClassificationResult


class TestDocumentProcessor:
    """Test DocumentProcessor."""

    def test_slugify(self):
        """Test text slugification."""
        processor = self._create_processor()

        assert processor._slugify("Kaiser Permanente") == "kaiser_permanente"
        assert processor._slugify("PG&E") == "pg_and_e"
        assert processor._slugify("Bank of America") == "bank_of_america"
        assert processor._slugify("IRS") == "irs"
        assert processor._slugify("Lincoln Elementary School") == "lincoln_elementary_school"
        assert processor._slugify("AT&T") == "at_and_t"

    def test_slugify_special_characters(self):
        """Test slugifying text with special characters."""
        processor = self._create_processor()

        assert processor._slugify("O'Reilly") == "o_reilly"
        assert processor._slugify("St. Mary's Hospital") == "st_mary_s_hospital"
        assert processor._slugify("Company (USA)") == "company_usa"

    def test_generate_filename_with_date(self):
        """Test filename generation with document date."""
        processor = self._create_processor()

        classification = ClassificationResult(
            person="Amir",
            category="medical",
            document_date=date(2025, 1, 15),
            sender="Kaiser Permanente",
            summary="Lab results",
        )

        filename = processor.generate_filename(classification, "scan001.pdf")

        assert filename == "2025-01-15_kaiser_permanente_medical.pdf"

    def test_generate_filename_without_date(self):
        """Test filename generation without document date (uses today)."""
        processor = self._create_processor()
        today = date.today()

        classification = ClassificationResult(
            person="Amir",
            category="other",
            document_date=None,
            sender="Unknown Sender",
            summary="Document",
        )

        filename = processor.generate_filename(classification, "scan.pdf")

        # Should use today's date
        assert filename.startswith(today.isoformat())
        assert "unknown_sender" in filename
        assert filename.endswith("_other.pdf")

    def test_generate_filename_no_sender(self):
        """Test filename generation with no sender."""
        processor = self._create_processor()

        classification = ClassificationResult(
            person="Maral",
            category="education",
            document_date=date(2025, 1, 10),
            sender=None,
            summary="Report card",
        )

        filename = processor.generate_filename(classification, "file.pdf")

        assert filename == "2025-01-10_unknown_education.pdf"

    def test_get_destination_path(self):
        """Test destination path construction."""
        processor = self._create_processor()

        path = processor.get_destination_path(
            person="Amir",
            category="medical",
            filename="2025-01-15_kaiser_permanente_medical.pdf",
        )

        expected = (
            processor.settings.documents_output_dir
            / "amir"
            / "medical"
            / "2025-01-15_kaiser_permanente_medical.pdf"
        )

        assert path == expected

    def test_get_destination_path_case_insensitive(self):
        """Test destination path uses lowercase person name."""
        processor = self._create_processor()

        path = processor.get_destination_path(
            person="AMIR",
            category="financial",
            filename="file.pdf",
        )

        # Person name should be lowercase in path
        assert "amir" in str(path)
        assert "AMIR" not in str(path)

    def _create_processor(self):
        """Create a processor with mock dependencies."""
        settings = Settings()
        ocr_tool = Mock()
        classifier = Mock()

        return DocumentProcessor(
            ocr_tool=ocr_tool,
            classifier=classifier,
            settings=settings,
        )
