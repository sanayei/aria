"""Memory and conversation history management for ARIA.

This module provides persistent storage for conversation history, sessions,
messages, and tool calls using SQLite, as well as semantic search via ChromaDB.
"""

from aria.memory.chunker import DocumentChunk, DocumentChunker
from aria.memory.context import context_to_chat_messages, messages_to_chat_messages
from aria.memory.conversation import ConversationStore
from aria.memory.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    OllamaEmbeddings,
    SentenceTransformerEmbeddings,
    get_embedding_provider,
)
from aria.memory.exceptions import (
    DatabaseError,
    MemoryError,
    MessageNotFoundError,
    SessionNotFoundError,
    ToolCallNotFoundError,
)
from aria.memory.indexer import (
    ConversationIndexer,
    ConversationSearchResult,
    IndexingStats,
)
from aria.memory.ingestion import (
    BatchIngestionResult,
    DocumentIngester,
    IngestedDocument,
    IngestionResult,
)
from aria.memory.models import (
    ConversationContext,
    Message,
    MessageRole,
    Session,
    SessionSummary,
    ToolCall,
    ToolCallStatus,
)
from aria.memory.vectors import (
    CollectionStats,
    Document,
    SearchResult,
    VectorStore,
    VectorStoreError,
)

__all__ = [
    # Store
    "ConversationStore",
    "VectorStore",
    "ConversationIndexer",
    "DocumentIngester",
    "DocumentChunker",
    # Models
    "ConversationContext",
    "Message",
    "MessageRole",
    "Session",
    "SessionSummary",
    "ToolCall",
    "ToolCallStatus",
    "Document",
    "SearchResult",
    "CollectionStats",
    "ConversationSearchResult",
    "IndexingStats",
    "DocumentChunk",
    "IngestionResult",
    "BatchIngestionResult",
    "IngestedDocument",
    # Exceptions
    "DatabaseError",
    "MemoryError",
    "MessageNotFoundError",
    "SessionNotFoundError",
    "ToolCallNotFoundError",
    "VectorStoreError",
    "EmbeddingError",
    # Context helpers
    "context_to_chat_messages",
    "messages_to_chat_messages",
    # Embeddings
    "EmbeddingProvider",
    "OllamaEmbeddings",
    "SentenceTransformerEmbeddings",
    "get_embedding_provider",
]
