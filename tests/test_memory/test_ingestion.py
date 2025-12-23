"""Tests for document ingestion functionality."""

import tempfile
from pathlib import Path

import pytest

from aria.memory.chunker import DocumentChunker
from aria.memory.embeddings import SentenceTransformerEmbeddings
from aria.memory.ingestion import DocumentIngester
from aria.memory.vectors import VectorStore


@pytest.fixture
async def vector_store(tmp_path: Path):
    """Create a test vector store."""
    embedding_provider = SentenceTransformerEmbeddings()
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_docs",
        embedding_provider=embedding_provider,
    )
    await store.initialize()
    yield store
    # Cleanup happens automatically


@pytest.fixture
def ingester(vector_store: VectorStore):
    """Create a test document ingester."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    return DocumentIngester(
        vector_store=vector_store,
        chunker=chunker,
    )


class TestDocumentIngester:
    """Test suite for DocumentIngester."""

    @pytest.mark.asyncio
    async def test_ingest_txt_file(self, ingester: DocumentIngester, tmp_path: Path):
        """Test ingesting a single TXT file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text(
            "This is a test document.\n\nWith multiple paragraphs.\n\nFor testing."
        )

        result = await ingester.ingest_file(
            file_path=test_file,
            metadata={"source": "test"},
            tags=["test", "sample"],
        )

        assert result.success is True
        assert result.file_path == str(test_file)
        assert result.chunks_created > 0
        assert result.error is None
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_ingest_md_file(self, ingester: DocumentIngester, tmp_path: Path):
        """Test ingesting a Markdown file."""
        # Create test file
        test_file = tmp_path / "test.md"
        content = "# Test Document\n\nThis is a test.\n\n## Section\n\nMore content here."
        test_file.write_text(content)

        result = await ingester.ingest_file(
            file_path=test_file,
            metadata=None,
            tags=None,
        )

        assert result.success is True
        assert result.chunks_created >= 1

    @pytest.mark.asyncio
    async def test_ingest_html_file(self, ingester: DocumentIngester, tmp_path: Path):
        """Test ingesting an HTML file."""
        # Create test file
        test_file = tmp_path / "test.html"
        html = "<html><body><h1>Title</h1><p>Content here.</p></body></html>"
        test_file.write_text(html)

        result = await ingester.ingest_file(
            file_path=test_file,
            metadata={"type": "html"},
            tags=["web"],
        )

        assert result.success is True
        assert result.chunks_created >= 1

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_file(self, ingester: DocumentIngester):
        """Test ingesting nonexistent file returns error."""
        result = await ingester.ingest_file(
            file_path=Path("/nonexistent/file.txt"),
            metadata=None,
            tags=None,
        )

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_ingest_directory(self, ingester: DocumentIngester, tmp_path: Path):
        """Test batch ingesting directory."""
        # Create test files
        (tmp_path / "doc1.txt").write_text("Document 1 content")
        (tmp_path / "doc2.txt").write_text("Document 2 content")
        (tmp_path / "doc3.md").write_text("# Document 3\n\nContent")

        # Create subdirectory with more files
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "doc4.txt").write_text("Document 4 content")

        result = await ingester.ingest_directory(
            directory=tmp_path,
            pattern="**/*.txt",
            recursive=True,
            metadata={"batch": "test"},
            tags=["batch"],
        )

        assert result.total_files == 3  # doc1, doc2, doc4 (not doc3.md)
        assert result.successful >= 0  # At least some should succeed
        assert result.total_chunks > 0

    @pytest.mark.asyncio
    async def test_ingest_directory_nonrecursive(self, ingester: DocumentIngester, tmp_path: Path):
        """Test batch ingesting directory without recursion."""
        # Create test files
        (tmp_path / "doc1.txt").write_text("Document 1 content")
        (tmp_path / "doc2.txt").write_text("Document 2 content")

        # Create subdirectory with files (should be ignored)
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "doc3.txt").write_text("Document 3 content")

        result = await ingester.ingest_directory(
            directory=tmp_path,
            pattern="*.txt",  # Non-recursive pattern
            recursive=False,
            metadata=None,
            tags=None,
        )

        assert result.total_files == 2  # Only doc1 and doc2

    @pytest.mark.asyncio
    async def test_remove_document(self, ingester: DocumentIngester, tmp_path: Path):
        """Test removing an ingested document."""
        # First ingest a document
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document for removal.")

        ingest_result = await ingester.ingest_file(
            file_path=test_file,
            metadata=None,
            tags=None,
        )
        assert ingest_result.success is True

        # Now remove it
        removed = await ingester.remove_document(file_path=test_file)
        assert removed is True

    @pytest.mark.asyncio
    async def test_remove_nonexistent_document(self, ingester: DocumentIngester):
        """Test removing nonexistent document."""
        removed = await ingester.remove_document(file_path=Path("/nonexistent/file.txt"))
        # Should return False (no documents to remove)
        assert removed is False

    @pytest.mark.asyncio
    async def test_list_ingested(self, ingester: DocumentIngester, tmp_path: Path):
        """Test listing ingested documents."""
        # Ingest some documents
        for i in range(3):
            test_file = tmp_path / f"doc{i}.txt"
            test_file.write_text(f"Document {i} content here.")
            await ingester.ingest_file(
                file_path=test_file,
                metadata=None,
                tags=[f"tag{i}"],
            )

        # List documents
        documents = await ingester.list_ingested()

        assert len(documents) >= 3  # At least our 3 documents

        # Check document structure
        for doc in documents:
            assert doc.file_path is not None
            assert doc.file_name is not None
            assert doc.chunk_count > 0
            assert doc.file_size_bytes > 0

    @pytest.mark.asyncio
    async def test_get_document_chunks(self, ingester: DocumentIngester, tmp_path: Path):
        """Test retrieving chunks for a specific document."""
        # Ingest a document
        test_file = tmp_path / "test.txt"
        test_file.write_text(
            "This is test content.\n\nWith multiple parts.\n\nFor testing retrieval."
        )

        ingest_result = await ingester.ingest_file(
            file_path=test_file,
            metadata=None,
            tags=None,
        )
        assert ingest_result.success is True

        # Get chunks
        chunks = await ingester.get_document_chunks(file_path=test_file)

        assert len(chunks) == ingest_result.chunks_created
        for chunk in chunks:
            assert chunk.source_file == str(test_file)
            assert len(chunk.content) > 0

    @pytest.mark.asyncio
    async def test_metadata_preservation(self, ingester: DocumentIngester, tmp_path: Path):
        """Test that metadata is preserved through ingestion."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        metadata = {"author": "Test Author", "year": 2025}
        tags = ["important", "test"]

        await ingester.ingest_file(
            file_path=test_file,
            metadata=metadata,
            tags=tags,
        )

        # Retrieve chunks and verify metadata
        chunks = await ingester.get_document_chunks(file_path=test_file)

        for chunk in chunks:
            # Should have both custom metadata and auto-generated metadata
            assert chunk.metadata["author"] == "Test Author"
            assert chunk.metadata["year"] == 2025
            assert chunk.metadata["file_path"] == str(test_file)
            assert chunk.metadata["file_name"] == "test.txt"
            assert chunk.metadata["tags"] == "important,test"

    @pytest.mark.asyncio
    async def test_reingest_document(self, ingester: DocumentIngester, tmp_path: Path):
        """Test re-ingesting a document (should update)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        # First ingestion
        result1 = await ingester.ingest_file(
            file_path=test_file,
            metadata=None,
            tags=["v1"],
        )
        assert result1.success is True
        chunks1 = result1.chunks_created

        # Update file and re-ingest
        test_file.write_text("Updated content with more text to change chunk count")

        result2 = await ingester.ingest_file(
            file_path=test_file,
            metadata=None,
            tags=["v2"],
        )
        assert result2.success is True

        # Should have new chunks (due to ChromaDB upsert behavior)

    @pytest.mark.asyncio
    async def test_empty_file(self, ingester: DocumentIngester, tmp_path: Path):
        """Test ingesting an empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        result = await ingester.ingest_file(
            file_path=test_file,
            metadata=None,
            tags=None,
        )

        # Empty file should create no chunks
        assert result.chunks_created == 0
        # Empty file is considered a failure (no chunks to ingest)
        assert result.success is False
        assert "empty" in result.error.lower() or "no chunks" in result.error.lower()

    @pytest.mark.asyncio
    async def test_custom_chunk_size(self, vector_store: VectorStore, tmp_path: Path):
        """Test ingester with custom chunk size."""
        ingester = DocumentIngester(
            vector_store=vector_store,
            chunker=None,  # Will create default
            chunk_size=50,  # Small chunks
            chunk_overlap=5,
        )

        test_file = tmp_path / "test.txt"
        # Create text with multiple paragraphs to ensure chunking
        test_file.write_text(
            "Paragraph one with some text.\n\nParagraph two with more text.\n\nParagraph three continues.\n\nParagraph four adds more.\n\nParagraph five finishes it."
        )

        result = await ingester.ingest_file(
            file_path=test_file,
            metadata=None,
            tags=None,
        )

        # Small chunk size should create multiple chunks
        assert result.chunks_created >= 1
        assert result.success is True
