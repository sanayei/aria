"""Tests for document classifier."""

import pytest
from datetime import date
from unittest.mock import AsyncMock, Mock

from aria.config import Settings
from aria.llm import ChatMessage, ChatResponse
from aria.tools.documents.classifier import DocumentClassifier


class TestDocumentClassifier:
    """Test DocumentClassifier."""

    @pytest.mark.asyncio
    async def test_classify_valid_json(self):
        """Test classification with valid JSON response."""
        client = Mock()
        settings = Settings()

        # Mock LLM response with valid JSON
        mock_response = ChatResponse(
            message=ChatMessage(
                role="assistant",
                content='{"person": "Amir", "category": "medical", "document_date": "2025-01-15", "sender": "Kaiser Permanente", "summary": "Lab results from annual checkup"}',
            ),
            model="test-model",
            created_at="2025-01-01T00:00:00Z",
            done=True,
        )
        client.chat = AsyncMock(return_value=mock_response)

        classifier = DocumentClassifier(client, settings)
        result = await classifier.classify("Sample OCR text")

        assert result.person == "Amir"
        assert result.category == "medical"
        assert result.document_date == date(2025, 1, 15)
        assert result.sender == "Kaiser Permanente"
        assert "Lab results" in result.summary

    @pytest.mark.asyncio
    async def test_classify_json_with_null_date(self):
        """Test classification with null document_date."""
        client = Mock()
        settings = Settings()

        mock_response = ChatResponse(
            message=ChatMessage(
                role="assistant",
                content='{"person": "Munira", "category": "other", "document_date": null, "sender": "Unknown", "summary": "Document with unclear date"}',
            ),
            model="test-model",
            created_at="2025-01-01T00:00:00Z",
            done=True,
        )
        client.chat = AsyncMock(return_value=mock_response)

        classifier = DocumentClassifier(client, settings)
        result = await classifier.classify("Sample OCR text")

        assert result.person == "Munira"
        assert result.document_date is None
        assert result.sender == "Unknown"

    @pytest.mark.asyncio
    async def test_classify_json_in_code_block(self):
        """Test classification when LLM wraps JSON in markdown code block."""
        client = Mock()
        settings = Settings()

        # LLM sometimes returns JSON wrapped in ```json ... ```
        mock_response = ChatResponse(
            message=ChatMessage(
                role="assistant",
                content='```json\n{"person": "Gazelle", "category": "education", "document_date": "2025-01-10", "sender": "School District", "summary": "Report card"}\n```',
            ),
            model="test-model",
            created_at="2025-01-01T00:00:00Z",
            done=True,
        )
        client.chat = AsyncMock(return_value=mock_response)

        classifier = DocumentClassifier(client, settings)
        result = await classifier.classify("Sample OCR text")

        assert result.person == "Gazelle"
        assert result.category == "education"
        assert result.sender == "School District"

    @pytest.mark.asyncio
    async def test_classify_invalid_json(self):
        """Test classification with invalid JSON response."""
        client = Mock()
        settings = Settings()

        mock_response = ChatResponse(
            message=ChatMessage(
                role="assistant",
                content="This is not valid JSON",
            ),
            model="test-model",
            created_at="2025-01-01T00:00:00Z",
            done=True,
        )
        client.chat = AsyncMock(return_value=mock_response)

        classifier = DocumentClassifier(client, settings)

        with pytest.raises(RuntimeError, match="classification failed"):
            await classifier.classify("Sample OCR text")

    @pytest.mark.asyncio
    async def test_classify_missing_required_fields(self):
        """Test classification with missing required fields."""
        client = Mock()
        settings = Settings()

        # Missing 'summary' field
        mock_response = ChatResponse(
            message=ChatMessage(
                role="assistant",
                content='{"person": "Amir", "category": "medical"}',
            ),
            model="test-model",
            created_at="2025-01-01T00:00:00Z",
            done=True,
        )
        client.chat = AsyncMock(return_value=mock_response)

        classifier = DocumentClassifier(client, settings)

        with pytest.raises(RuntimeError, match="classification failed"):
            await classifier.classify("Sample OCR text")

    @pytest.mark.asyncio
    async def test_classify_invalid_date_format(self):
        """Test classification with invalid date format (should set to None)."""
        client = Mock()
        settings = Settings()

        mock_response = ChatResponse(
            message=ChatMessage(
                role="assistant",
                content='{"person": "Amir", "category": "medical", "document_date": "invalid-date", "sender": "Clinic", "summary": "Summary"}',
            ),
            model="test-model",
            created_at="2025-01-01T00:00:00Z",
            done=True,
        )
        client.chat = AsyncMock(return_value=mock_response)

        classifier = DocumentClassifier(client, settings)
        result = await classifier.classify("Sample OCR text")

        # Invalid date should be set to None
        assert result.document_date is None
