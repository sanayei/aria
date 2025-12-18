"""Tests for ChromaDB vector store."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.memory.embeddings import EmbeddingProvider
from aria.memory.vectors import (
    CollectionStats,
    Document,
    SearchResult,
    VectorStore,
    VectorStoreError,
)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._call_count = 0

    async def embed_text(self, text: str) -> list[float]:
        """Generate a mock embedding based on text hash."""
        self._call_count += 1
        # Generate deterministic embedding based on text
        hash_val = hash(text)
        return [float((hash_val + i) % 100) / 100.0 for i in range(self._dimension)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for batch."""
        return [await self.embed_text(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "mock-embeddings"


@pytest.fixture
async def temp_vector_store():
    """Create a temporary vector store for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_dir = Path(tmpdir) / "chroma"
        embedding_provider = MockEmbeddingProvider(dimension=384)

        store = VectorStore(
            persist_directory=persist_dir,
            embedding_provider=embedding_provider,
            collection_name="test_collection",
        )

        await store.initialize()

        yield store

        await store.close()


class TestVectorStoreInitialization:
    """Tests for VectorStore initialization."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test VectorStore initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir) / "chroma"
            embedding_provider = MockEmbeddingProvider()

            store = VectorStore(
                persist_directory=persist_dir,
                embedding_provider=embedding_provider,
                collection_name="test",
            )

            await store.initialize()

            # Check that persist directory was created
            assert persist_dir.exists()

            # Get stats to verify collection exists
            stats = await store.get_stats()
            assert stats.name == "test"
            assert stats.count == 0
            assert stats.dimension == 384

            await store.close()

    @pytest.mark.asyncio
    async def test_initialization_creates_directory(self):
        """Test that initialization creates persist directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir) / "nested" / "chroma"
            embedding_provider = MockEmbeddingProvider()

            store = VectorStore(
                persist_directory=persist_dir,
                embedding_provider=embedding_provider,
            )

            await store.initialize()

            assert persist_dir.exists()

            await store.close()

    @pytest.mark.asyncio
    async def test_operations_without_initialization(self):
        """Test that operations fail before initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir) / "chroma"
            embedding_provider = MockEmbeddingProvider()

            store = VectorStore(
                persist_directory=persist_dir,
                embedding_provider=embedding_provider,
            )

            # Don't initialize, try to add document
            with pytest.raises(VectorStoreError, match="not initialized"):
                await store.add_document("test", {})


class TestDocumentOperations:
    """Tests for document CRUD operations."""

    @pytest.mark.asyncio
    async def test_add_document(self, temp_vector_store):
        """Test adding a single document."""
        doc_id = await temp_vector_store.add_document(
            content="This is a test document.",
            metadata={"source": "test"},
        )

        assert doc_id is not None
        assert isinstance(doc_id, str)

        # Verify document was added
        stats = await temp_vector_store.get_stats()
        assert stats.count == 1

    @pytest.mark.asyncio
    async def test_add_document_with_custom_id(self, temp_vector_store):
        """Test adding a document with custom ID."""
        custom_id = "custom-doc-123"
        doc_id = await temp_vector_store.add_document(
            content="Test document",
            metadata={},
            doc_id=custom_id,
        )

        assert doc_id == custom_id

    @pytest.mark.asyncio
    async def test_add_documents_batch(self, temp_vector_store):
        """Test adding multiple documents in batch."""
        documents = [
            Document(content="Document 1", metadata={"index": 1}),
            Document(content="Document 2", metadata={"index": 2}),
            Document(content="Document 3", metadata={"index": 3}),
        ]

        doc_ids = await temp_vector_store.add_documents(documents)

        assert len(doc_ids) == 3
        assert all(isinstance(doc_id, str) for doc_id in doc_ids)

        # Verify all documents were added
        stats = await temp_vector_store.get_stats()
        assert stats.count == 3

    @pytest.mark.asyncio
    async def test_add_documents_empty_batch(self, temp_vector_store):
        """Test adding empty batch."""
        doc_ids = await temp_vector_store.add_documents([])
        assert doc_ids == []

    @pytest.mark.asyncio
    async def test_add_documents_large_batch(self, temp_vector_store):
        """Test adding large batch with batching."""
        # Create 250 documents (should be split into 3 batches of 100)
        documents = [
            Document(content=f"Document {i}", metadata={"index": i})
            for i in range(250)
        ]

        doc_ids = await temp_vector_store.add_documents(documents, batch_size=100)

        assert len(doc_ids) == 250

        stats = await temp_vector_store.get_stats()
        assert stats.count == 250

    @pytest.mark.asyncio
    async def test_get_document(self, temp_vector_store):
        """Test retrieving a document by ID."""
        # Add document
        doc_id = await temp_vector_store.add_document(
            content="Test content",
            metadata={"key": "value"},
        )

        # Retrieve document
        doc = await temp_vector_store.get_document(doc_id)

        assert doc is not None
        assert doc.doc_id == doc_id
        assert doc.content == "Test content"
        assert doc.metadata["key"] == "value"

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, temp_vector_store):
        """Test retrieving a document that doesn't exist."""
        doc = await temp_vector_store.get_document("nonexistent-id")
        assert doc is None

    @pytest.mark.asyncio
    async def test_delete_document(self, temp_vector_store):
        """Test deleting a document."""
        # Add document
        doc_id = await temp_vector_store.add_document("Test content", {})

        # Verify it exists
        stats = await temp_vector_store.get_stats()
        assert stats.count == 1

        # Delete document
        result = await temp_vector_store.delete_document(doc_id)
        assert result is True

        # Verify it's gone
        stats = await temp_vector_store.get_stats()
        assert stats.count == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, temp_vector_store):
        """Test deleting a document that doesn't exist."""
        result = await temp_vector_store.delete_document("nonexistent-id")
        assert result is False


class TestSearch:
    """Tests for semantic search operations."""

    @pytest.mark.asyncio
    async def test_search_with_results(self, temp_vector_store):
        """Test searching with results."""
        # Add some documents
        await temp_vector_store.add_documents(
            [
                Document(content="Python programming language", metadata={"topic": "python"}),
                Document(content="JavaScript web development", metadata={"topic": "javascript"}),
                Document(content="Python data science", metadata={"topic": "python"}),
            ]
        )

        # Search for Python-related content
        results = await temp_vector_store.search("Python programming", limit=2)

        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(0.0 <= r.score <= 1.0 for r in results)

        # Results should be ordered by relevance
        if len(results) > 1:
            assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_search_no_results(self, temp_vector_store):
        """Test searching in empty collection."""
        results = await temp_vector_store.search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_limit(self, temp_vector_store):
        """Test search respects limit."""
        # Add 10 documents
        documents = [
            Document(content=f"Document number {i}", metadata={"index": i})
            for i in range(10)
        ]
        await temp_vector_store.add_documents(documents)

        # Search with limit
        results = await temp_vector_store.search("Document", limit=5)

        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_search_by_embedding(self, temp_vector_store):
        """Test searching with pre-computed embedding."""
        # Add documents
        await temp_vector_store.add_document("Test content", {})

        # Generate embedding
        embedding = await temp_vector_store._embedding_provider.embed_text("Test query")

        # Search by embedding
        results = await temp_vector_store.search_by_embedding(embedding, limit=10)

        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_result_structure(self, temp_vector_store):
        """Test search result contains all expected fields."""
        # Add document
        await temp_vector_store.add_document(
            content="Test content",
            metadata={"key": "value"},
            doc_id="test-id",
        )

        # Search
        results = await temp_vector_store.search("Test")

        assert len(results) > 0

        result = results[0]
        assert result.doc_id == "test-id"
        assert result.content == "Test content"
        assert result.metadata["key"] == "value"
        assert isinstance(result.score, float)
        assert isinstance(result.distance, float)


class TestCollectionManagement:
    """Tests for collection management operations."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, temp_vector_store):
        """Test getting stats for empty collection."""
        stats = await temp_vector_store.get_stats()

        assert isinstance(stats, CollectionStats)
        assert stats.name == "test_collection"
        assert stats.count == 0
        assert stats.dimension == 384

    @pytest.mark.asyncio
    async def test_get_stats_with_documents(self, temp_vector_store):
        """Test getting stats with documents."""
        # Add documents
        await temp_vector_store.add_documents(
            [
                Document(content="Doc 1"),
                Document(content="Doc 2"),
                Document(content="Doc 3"),
            ]
        )

        stats = await temp_vector_store.get_stats()

        assert stats.count == 3
        assert stats.dimension == 384

    @pytest.mark.asyncio
    async def test_clear_collection(self, temp_vector_store):
        """Test clearing all documents."""
        # Add documents
        await temp_vector_store.add_documents(
            [
                Document(content="Doc 1"),
                Document(content="Doc 2"),
            ]
        )

        # Verify documents exist
        stats = await temp_vector_store.get_stats()
        assert stats.count == 2

        # Clear collection
        await temp_vector_store.clear()

        # Verify collection is empty
        stats = await temp_vector_store.get_stats()
        assert stats.count == 0


class TestDocumentModel:
    """Tests for Document model."""

    def test_document_creation(self):
        """Test creating a Document."""
        doc = Document(
            content="Test content",
            metadata={"key": "value"},
            doc_id="test-id",
        )

        assert doc.content == "Test content"
        assert doc.metadata == {"key": "value"}
        assert doc.doc_id == "test-id"

    def test_document_with_defaults(self):
        """Test Document with default values."""
        doc = Document(content="Test")

        assert doc.content == "Test"
        assert doc.metadata == {}
        assert doc.doc_id is None

    def test_document_with_id_no_id(self):
        """Test with_id generates ID if missing."""
        doc = Document(content="Test")
        doc_with_id = doc.with_id()

        assert doc_with_id.doc_id is not None
        assert isinstance(doc_with_id.doc_id, str)
        # Original should be unchanged
        assert doc.doc_id is None

    def test_document_with_id_existing_id(self):
        """Test with_id preserves existing ID."""
        doc = Document(content="Test", doc_id="existing-id")
        doc_with_id = doc.with_id()

        assert doc_with_id.doc_id == "existing-id"


class TestPersistence:
    """Tests for data persistence."""

    @pytest.mark.asyncio
    async def test_persistence_across_sessions(self):
        """Test that data persists across VectorStore instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = Path(tmpdir) / "chroma"
            embedding_provider = MockEmbeddingProvider()

            # Create first store and add documents
            store1 = VectorStore(
                persist_directory=persist_dir,
                embedding_provider=embedding_provider,
                collection_name="persist_test",
            )
            await store1.initialize()

            doc_id = await store1.add_document(
                content="Persistent document",
                metadata={"test": "value"},
            )

            await store1.close()

            # Create second store with same persist directory
            store2 = VectorStore(
                persist_directory=persist_dir,
                embedding_provider=embedding_provider,
                collection_name="persist_test",
            )
            await store2.initialize()

            # Verify document is still there
            stats = await store2.get_stats()
            assert stats.count == 1

            doc = await store2.get_document(doc_id)
            assert doc is not None
            assert doc.content == "Persistent document"
            assert doc.metadata["test"] == "value"

            await store2.close()
