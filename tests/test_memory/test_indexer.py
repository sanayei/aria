"""Tests for conversation indexing."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aria.memory.conversation import ConversationStore
from aria.memory.embeddings import EmbeddingProvider
from aria.memory.indexer import ConversationIndexer, IndexingStats
from aria.memory.models import MessageRole
from aria.memory.vectors import VectorStore


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    async def embed_text(self, text: str) -> list[float]:
        """Generate deterministic embedding based on text hash."""
        hash_val = hash(text)
        return [float((hash_val + i) % 100) / 100.0 for i in range(self._dimension)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate batch embeddings."""
        return [await self.embed_text(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "mock-embeddings"


@pytest.fixture
async def indexer_setup():
    """Create indexer with stores for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create stores
        conv_db_path = tmppath / "conversations.db"
        chroma_path = tmppath / "chroma"

        conv_store = ConversationStore(db_path=conv_db_path)
        await conv_store.initialize()

        embedding_provider = MockEmbeddingProvider()
        vector_store = VectorStore(
            persist_directory=chroma_path,
            embedding_provider=embedding_provider,
            collection_name="test_conversations",
        )
        await vector_store.initialize()

        indexer = ConversationIndexer(
            vector_store=vector_store,
            conversation_store=conv_store,
        )

        yield {
            "indexer": indexer,
            "conv_store": conv_store,
            "vector_store": vector_store,
        }

        await vector_store.close()


class TestConversationIndexer:
    """Tests for ConversationIndexer."""

    @pytest.mark.asyncio
    async def test_initialization(self, indexer_setup):
        """Test indexer initialization."""
        indexer = indexer_setup["indexer"]
        assert indexer is not None

    @pytest.mark.asyncio
    async def test_index_message_user(self, indexer_setup):
        """Test indexing a user message."""
        indexer = indexer_setup["indexer"]
        conv_store = indexer_setup["conv_store"]

        # Create session and message
        session = await conv_store.create_session("Test Session")
        message = await conv_store.add_message(
            session_id=session.id,
            role="user",
            content="This is a test message with enough content to be indexed",
        )

        # Index message
        doc_id = await indexer.index_message(session.id, message)

        assert doc_id is not None
        assert doc_id.startswith("conv_")

        # Verify it's in vector store
        stats = await indexer.get_stats()
        assert stats["total_messages_indexed"] == 1

    @pytest.mark.asyncio
    async def test_index_message_skip_short(self, indexer_setup):
        """Test that short messages are skipped."""
        indexer = indexer_setup["indexer"]
        conv_store = indexer_setup["conv_store"]

        session = await conv_store.create_session("Test")
        message = await conv_store.add_message(
            session_id=session.id,
            role="user",
            content="Hi",  # Too short
        )

        doc_id = await indexer.index_message(session.id, message)

        assert doc_id is None

        stats = await indexer.get_stats()
        assert stats["total_messages_indexed"] == 0

    @pytest.mark.asyncio
    async def test_index_message_skip_tool(self, indexer_setup):
        """Test that tool messages are skipped."""
        indexer = indexer_setup["indexer"]
        conv_store = indexer_setup["conv_store"]

        session = await conv_store.create_session("Test")
        message = await conv_store.add_message(
            session_id=session.id,
            role="tool",
            content="Tool result with enough content to normally be indexed",
        )

        doc_id = await indexer.index_message(session.id, message)

        assert doc_id is None

    @pytest.mark.asyncio
    async def test_index_session(self, indexer_setup):
        """Test indexing an entire session."""
        indexer = indexer_setup["indexer"]
        conv_store = indexer_setup["conv_store"]

        # Create session with multiple messages
        session = await conv_store.create_session("Multi-Message Session")

        await conv_store.add_message(
            session_id=session.id,
            role="user",
            content="First user message with sufficient content",
        )
        await conv_store.add_message(
            session_id=session.id,
            role="assistant",
            content="Assistant response with enough content to be indexed",
        )
        await conv_store.add_message(
            session_id=session.id,
            role="user",
            content="Hi",  # Too short
        )
        await conv_store.add_message(
            session_id=session.id,
            role="tool",
            content="Tool result that should be skipped",
        )

        # Index session
        count = await indexer.index_session(session.id)

        # Should index 2 messages (skip short and tool)
        assert count == 2

        stats = await indexer.get_stats()
        assert stats["total_messages_indexed"] == 2

    @pytest.mark.asyncio
    async def test_index_all_sessions(self, indexer_setup):
        """Test indexing all sessions."""
        indexer = indexer_setup["indexer"]
        conv_store = indexer_setup["conv_store"]

        # Create multiple sessions
        session1 = await conv_store.create_session("Session 1")
        await conv_store.add_message(
            session_id=session1.id,
            role="user",
            content="Message in session 1 with enough content",
        )

        session2 = await conv_store.create_session("Session 2")
        await conv_store.add_message(
            session_id=session2.id,
            role="user",
            content="Message in session 2 with enough content",
        )

        # Index all
        stats = await indexer.index_all_sessions()

        assert stats.sessions_processed == 2
        assert stats.messages_indexed == 2
        assert stats.errors == 0
        assert stats.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_index_all_sessions_rebuild(self, indexer_setup):
        """Test rebuilding index clears first."""
        indexer = indexer_setup["indexer"]
        conv_store = indexer_setup["conv_store"]

        # Create and index session
        session = await conv_store.create_session("Test")
        await conv_store.add_message(
            session_id=session.id,
            role="user",
            content="First message with enough content",
        )

        await indexer.index_all_sessions()

        # Add another message
        await conv_store.add_message(
            session_id=session.id,
            role="user",
            content="Second message with enough content",
        )

        # Rebuild index
        stats = await indexer.index_all_sessions(rebuild=True)

        # Should index both messages (rebuild cleared old one)
        assert stats.messages_indexed == 2


class TestConversationSearch:
    """Tests for conversation search."""

    @pytest.mark.asyncio
    async def test_search_conversations(self, indexer_setup):
        """Test searching conversations."""
        indexer = indexer_setup["indexer"]
        conv_store = indexer_setup["conv_store"]

        # Create messages
        session = await conv_store.create_session("Python Tutorial")
        await conv_store.add_message(
            session_id=session.id,
            role="user",
            content="How do I use Python async and await keywords?",
        )
        await conv_store.add_message(
            session_id=session.id,
            role="assistant",
            content="Python async/await is used for concurrent programming",
        )

        # Index
        await indexer.index_session(session.id)

        # Search
        results = await indexer.search_conversations(
            query="Python async programming",
            limit=10,
        )

        assert len(results) > 0
        assert results[0].session_id == session.id
        assert "async" in results[0].message_content.lower()

    @pytest.mark.asyncio
    async def test_search_with_session_filter(self, indexer_setup):
        """Test searching with session filter."""
        indexer = indexer_setup["indexer"]
        conv_store = indexer_setup["conv_store"]

        # Create two sessions
        session1 = await conv_store.create_session("Session 1")
        await conv_store.add_message(
            session_id=session1.id,
            role="user",
            content="Message about Python programming basics",
        )

        session2 = await conv_store.create_session("Session 2")
        await conv_store.add_message(
            session_id=session2.id,
            role="user",
            content="Message about Python programming advanced",
        )

        await indexer.index_all_sessions()

        # Search with session filter
        results = await indexer.search_conversations(
            query="Python",
            session_id=session1.id,
        )

        assert len(results) == 1
        assert results[0].session_id == session1.id

    @pytest.mark.asyncio
    async def test_search_with_role_filter(self, indexer_setup):
        """Test searching with role filter."""
        indexer = indexer_setup["indexer"]
        conv_store = indexer_setup["conv_store"]

        session = await conv_store.create_session("Test")
        await conv_store.add_message(
            session_id=session.id,
            role="user",
            content="User question about Python programming",
        )
        await conv_store.add_message(
            session_id=session.id,
            role="assistant",
            content="Assistant answer about Python programming",
        )

        await indexer.index_session(session.id)

        # Search for user messages only
        results = await indexer.search_conversations(
            query="Python",
            role="user",
        )

        assert len(results) >= 1
        assert all(r.message_role == "user" for r in results)

    @pytest.mark.asyncio
    async def test_get_relevant_context(self, indexer_setup):
        """Test getting relevant context excluding current session."""
        indexer = indexer_setup["indexer"]
        conv_store = indexer_setup["conv_store"]

        # Create old session with relevant info
        old_session = await conv_store.create_session("Past Discussion")
        await conv_store.add_message(
            session_id=old_session.id,
            role="user",
            content="Previously discussed Python async patterns and best practices",
        )

        # Create current session
        current_session = await conv_store.create_session("Current")
        await conv_store.add_message(
            session_id=current_session.id,
            role="user",
            content="This is the current session about Python async",
        )

        await indexer.index_all_sessions()

        # Get context excluding current session
        results = await indexer.get_relevant_context(
            query="Python async",
            current_session_id=current_session.id,
            max_results=5,
        )

        # Should get old session, not current
        assert len(results) >= 1
        assert all(r.session_id != current_session.id for r in results)
        assert any(r.session_id == old_session.id for r in results)

    @pytest.mark.asyncio
    async def test_get_relevant_context_empty(self, indexer_setup):
        """Test getting context when only current session exists."""
        indexer = indexer_setup["indexer"]
        conv_store = indexer_setup["conv_store"]

        session = await conv_store.create_session("Only Session")
        await conv_store.add_message(
            session_id=session.id,
            role="user",
            content="This is the only session message",
        )

        await indexer.index_session(session.id)

        # Get context - should be empty since we exclude current session
        results = await indexer.get_relevant_context(
            query="session",
            current_session_id=session.id,
        )

        assert len(results) == 0


class TestIndexManagement:
    """Tests for index management operations."""

    @pytest.mark.asyncio
    async def test_get_stats(self, indexer_setup):
        """Test getting indexer stats."""
        indexer = indexer_setup["indexer"]
        conv_store = indexer_setup["conv_store"]

        # Initially empty
        stats = await indexer.get_stats()
        assert stats["total_messages_indexed"] == 0

        # Add and index messages
        session = await conv_store.create_session("Test")
        await conv_store.add_message(
            session_id=session.id,
            role="user",
            content="Message 1 with enough content to index",
        )
        await conv_store.add_message(
            session_id=session.id,
            role="user",
            content="Message 2 with enough content to index",
        )

        await indexer.index_session(session.id)

        # Check stats
        stats = await indexer.get_stats()
        assert stats["total_messages_indexed"] == 2

    @pytest.mark.asyncio
    async def test_delete_session_index(self, indexer_setup):
        """Test deleting session from index."""
        indexer = indexer_setup["indexer"]
        conv_store = indexer_setup["conv_store"]

        # Create and index session
        session = await conv_store.create_session("To Delete")
        await conv_store.add_message(
            session_id=session.id,
            role="user",
            content="Message that will be deleted from index",
        )

        await indexer.index_session(session.id)

        stats = await indexer.get_stats()
        assert stats["total_messages_indexed"] == 1

        # Delete from index
        deleted = await indexer.delete_session_index(session.id)
        assert deleted == 1

        stats = await indexer.get_stats()
        assert stats["total_messages_indexed"] == 0


class TestAutoIndexing:
    """Tests for auto-indexing hook."""

    @pytest.mark.asyncio
    async def test_auto_index_on_add_message(self):
        """Test that messages are auto-indexed when added."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create stores
            conv_db_path = tmppath / "conversations.db"
            chroma_path = tmppath / "chroma"

            embedding_provider = MockEmbeddingProvider()
            vector_store = VectorStore(
                persist_directory=chroma_path,
                embedding_provider=embedding_provider,
            )
            await vector_store.initialize()

            indexer = ConversationIndexer(
                vector_store=vector_store,
                conversation_store=None,  # Will set after creating
            )

            # Create conversation store with auto-index callback
            conv_store = ConversationStore(
                db_path=conv_db_path,
                auto_index_callback=indexer.index_message,
            )
            await conv_store.initialize()

            # Update indexer's conversation store reference
            indexer._conversation_store = conv_store

            # Add message - should auto-index
            session = await conv_store.create_session("Auto Test")
            await conv_store.add_message(
                session_id=session.id,
                role="user",
                content="This message should be automatically indexed",
            )

            # Check that it was indexed
            stats = await indexer.get_stats()
            assert stats["total_messages_indexed"] == 1

            await vector_store.close()
