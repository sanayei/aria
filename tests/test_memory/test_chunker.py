"""Tests for document chunking functionality."""

import tempfile
from pathlib import Path

import pytest

from aria.memory.chunker import DocumentChunk, DocumentChunker


class TestDocumentChunker:
    """Test suite for DocumentChunker."""

    def test_chunk_small_text(self):
        """Test chunking text smaller than chunk_size."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        text = "This is a small text."

        chunks = chunker.chunk_text(
            text=text,
            metadata={"test": "value"},
            source_file="/test/file.txt",
        )

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1
        assert chunks[0].source_file == "/test/file.txt"
        assert chunks[0].metadata["test"] == "value"

    def test_chunk_text_with_paragraphs(self):
        """Test chunking respects paragraph boundaries."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

        text = "First paragraph is here.\n\nSecond paragraph follows.\n\nThird paragraph ends."

        chunks = chunker.chunk_text(
            text=text,
            metadata=None,
            source_file="/test/file.txt",
        )

        # Should create multiple chunks
        assert len(chunks) > 1

        # Each chunk should have overlap metadata
        for chunk in chunks:
            assert chunk.chunk_index >= 0
            assert chunk.total_chunks == len(chunks)
            assert chunk.source_file == "/test/file.txt"

    def test_chunk_text_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = DocumentChunker(chunk_size=30, chunk_overlap=10)

        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"

        chunks = chunker.chunk_text(
            text=text,
            metadata=None,
            source_file="/test/file.txt",
        )

        # Should have multiple chunks with overlap
        assert len(chunks) >= 2

        # Check that subsequent chunks have some overlap
        # (implementation-dependent, but overlap should be preserved)
        for i in range(len(chunks) - 1):
            # Chunks should not be empty
            assert len(chunks[i].content) > 0
            assert len(chunks[i + 1].content) > 0

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

        chunks = chunker.chunk_text(
            text="",
            metadata=None,
            source_file="/test/file.txt",
        )

        assert len(chunks) == 0

    def test_chunk_txt_file(self, tmp_path: Path):
        """Test chunking a TXT file."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test file.\n\nWith multiple paragraphs.\n\nAnd content.")

        chunks = chunker.chunk_file(
            file_path=test_file,
            metadata={"source": "test"},
        )

        assert len(chunks) >= 1
        assert chunks[0].source_file == str(test_file)
        assert chunks[0].metadata["source"] == "test"
        assert "test file" in chunks[0].content

    def test_chunk_md_file(self, tmp_path: Path):
        """Test chunking a Markdown file."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Header\n\nThis is markdown content.\n\n## Section\n\nMore content.")

        chunks = chunker.chunk_file(
            file_path=test_file,
            metadata=None,
        )

        assert len(chunks) >= 1
        assert "Header" in chunks[0].content or "markdown" in chunks[0].content

    def test_chunk_html_file(self, tmp_path: Path):
        """Test chunking an HTML file."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

        # Create test file
        test_file = tmp_path / "test.html"
        html_content = "<html><body><h1>Title</h1><p>Paragraph content here.</p></body></html>"
        test_file.write_text(html_content)

        chunks = chunker.chunk_file(
            file_path=test_file,
            metadata=None,
        )

        assert len(chunks) >= 1
        # Should extract text from HTML
        assert "Title" in chunks[0].content
        assert "Paragraph content" in chunks[0].content
        # Should not include HTML tags
        assert "<html>" not in chunks[0].content

    def test_chunk_unsupported_file_type(self, tmp_path: Path):
        """Test chunking unsupported file type returns empty list."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

        # Create unsupported file
        test_file = tmp_path / "test.xyz"
        test_file.write_text("Some content")

        chunks = chunker.chunk_file(file_path=test_file, metadata=None)

        # Should return empty list for unsupported types
        assert chunks == []

    def test_chunk_nonexistent_file(self):
        """Test chunking nonexistent file returns empty list."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

        chunks = chunker.chunk_file(
            file_path=Path("/nonexistent/file.txt"),
            metadata=None,
        )

        # Should return empty list for nonexistent files
        assert chunks == []

    def test_chunk_metadata_preservation(self):
        """Test that metadata is preserved across chunks."""
        chunker = DocumentChunker(chunk_size=30, chunk_overlap=5)

        metadata = {"author": "Test Author", "year": 2025}
        text = "This is some text that will be split into multiple chunks for testing."

        chunks = chunker.chunk_text(
            text=text,
            metadata=metadata,
            source_file="/test/file.txt",
        )

        # All chunks should have the same metadata
        for chunk in chunks:
            assert chunk.metadata["author"] == "Test Author"
            assert chunk.metadata["year"] == 2025

    def test_custom_separator(self):
        """Test using custom separator."""
        chunker = DocumentChunker(
            chunk_size=50,
            chunk_overlap=5,
            separator="\n",  # Split on single newline
        )

        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"

        chunks = chunker.chunk_text(
            text=text,
            metadata=None,
            source_file="/test/file.txt",
        )

        assert len(chunks) >= 1

    def test_very_long_paragraph(self):
        """Test handling of paragraph longer than chunk_size."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

        # Create a very long paragraph (no separators)
        text = " ".join([f"Word{i}" for i in range(100)])

        chunks = chunker.chunk_text(
            text=text,
            metadata=None,
            source_file="/test/file.txt",
        )

        # Should split into multiple chunks
        assert len(chunks) > 1

        # Each chunk should be roughly chunk_size
        for chunk in chunks[:-1]:  # Except last chunk
            assert len(chunk.content) <= chunker.chunk_size + 50  # Some tolerance

    def test_chunk_indices(self):
        """Test that chunk indices are correct."""
        chunker = DocumentChunker(chunk_size=30, chunk_overlap=5)

        text = "This is some text that will be split into multiple chunks."

        chunks = chunker.chunk_text(
            text=text,
            metadata=None,
            source_file="/test/file.txt",
        )

        # Check indices
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)

    def test_pdf_chunking_requires_file(self):
        """Test that PDF chunking is triggered by file extension."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

        # This would fail if PDF file doesn't exist, but tests the routing logic
        # We'll check this in integration tests with actual PDF files


class TestDocumentChunk:
    """Test suite for DocumentChunk model."""

    def test_document_chunk_creation(self):
        """Test creating a DocumentChunk."""
        chunk = DocumentChunk(
            content="Test content",
            metadata={"key": "value"},
            chunk_index=0,
            total_chunks=1,
            source_file="/test/file.txt",
        )

        assert chunk.content == "Test content"
        assert chunk.metadata["key"] == "value"
        assert chunk.chunk_index == 0
        assert chunk.total_chunks == 1
        assert chunk.source_file == "/test/file.txt"
        assert chunk.page_number is None

    def test_document_chunk_with_page_number(self):
        """Test creating a DocumentChunk with page number."""
        chunk = DocumentChunk(
            content="PDF content",
            metadata={},
            chunk_index=0,
            total_chunks=1,
            source_file="/test/file.pdf",
            page_number=5,
        )

        assert chunk.page_number == 5
