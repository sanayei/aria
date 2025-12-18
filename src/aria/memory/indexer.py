"""Conversation indexing for semantic search."""

import asyncio
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from aria.logging import get_logger
from aria.memory.conversation import ConversationStore
from aria.memory.models import Message, MessageRole
from aria.memory.vectors import Document, VectorStore

logger = get_logger("aria.memory.indexer")


class ConversationSearchResult(BaseModel):
    """Result from conversation search."""

    session_id: str
    session_title: str | None
    message_id: str
    message_role: str
    message_content: str
    timestamp: datetime
    score: float


class IndexingStats(BaseModel):
    """Statistics from indexing operation."""

    sessions_processed: int
    messages_indexed: int
    messages_skipped: int
    errors: int
    duration_seconds: float


class ConversationIndexer:
    """Index conversations in vector store for semantic search.

    This class bridges ConversationStore (SQLite) and VectorStore (ChromaDB)
    to enable semantic search over conversation history.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        conversation_store: ConversationStore,
        min_message_length: int = 10,
    ):
        """Initialize conversation indexer.

        Args:
            vector_store: ChromaDB vector store
            conversation_store: SQLite conversation store
            min_message_length: Minimum message length to index
        """
        self._vector_store = vector_store
        self._conversation_store = conversation_store
        self._min_length = min_message_length

        logger.info("Initialized ConversationIndexer")

    def _should_index_message(self, message: Message) -> bool:
        """Determine if a message should be indexed.

        Args:
            message: Message to check

        Returns:
            bool: True if message should be indexed
        """
        # Skip tool calls and system messages
        if message.role in [MessageRole.TOOL, MessageRole.SYSTEM]:
            return False

        # Skip very short messages
        if len(message.content) < self._min_length:
            return False

        # Skip messages that are just tool results
        if message.content.startswith("[Tool"):
            return False

        return True

    def _create_document_id(self, session_id: str, message_id: str) -> str:
        """Create a unique document ID for a message.

        Args:
            session_id: Session ID
            message_id: Message ID

        Returns:
            str: Document ID
        """
        return f"conv_{session_id}_{message_id}"

    async def index_message(
        self,
        session_id: str,
        message: Message,
    ) -> str | None:
        """Index a single message in the vector store.

        Args:
            session_id: Session ID
            message: Message to index

        Returns:
            str | None: Document ID if indexed, None if skipped
        """
        try:
            # Check if message should be indexed
            if not self._should_index_message(message):
                logger.debug(f"Skipping message {message.id} (too short or wrong type)")
                return None

            # Get session for title
            session = await self._conversation_store.get_session(session_id)
            session_title = session.title if session else None

            # Create document ID
            doc_id = self._create_document_id(session_id, str(message.id))

            # Prepare metadata
            metadata = {
                "session_id": session_id,
                "session_title": session_title or "Untitled",
                "message_id": str(message.id),
                "role": message.role.value,
                "timestamp": message.timestamp.isoformat(),
            }

            # Add to vector store
            await self._vector_store.add_document(
                content=message.content,
                metadata=metadata,
                doc_id=doc_id,
            )

            logger.debug(f"Indexed message {message.id} from session {session_id}")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to index message {message.id}: {e}")
            return None

    async def index_session(self, session_id: str) -> int:
        """Index all messages in a session.

        Args:
            session_id: Session ID to index

        Returns:
            int: Number of messages indexed
        """
        try:
            logger.info(f"Indexing session {session_id}")

            # Get all messages in session
            messages = await self._conversation_store.get_messages(
                session_id=session_id,
                limit=1000,  # Reasonable limit
            )

            # Index messages in batch
            indexed_count = 0
            documents: list[Document] = []

            # Get session for title
            session = await self._conversation_store.get_session(session_id)
            session_title = session.title if session else None

            for message in messages:
                if not self._should_index_message(message):
                    continue

                doc_id = self._create_document_id(session_id, str(message.id))

                metadata = {
                    "session_id": session_id,
                    "session_title": session_title or "Untitled",
                    "message_id": str(message.id),
                    "role": message.role.value,
                    "timestamp": message.timestamp.isoformat(),
                }

                documents.append(
                    Document(
                        doc_id=doc_id,
                        content=message.content,
                        metadata=metadata,
                    )
                )

            # Batch add to vector store
            if documents:
                await self._vector_store.add_documents(documents)
                indexed_count = len(documents)

            logger.info(f"Indexed {indexed_count} messages from session {session_id}")
            return indexed_count

        except Exception as e:
            logger.error(f"Failed to index session {session_id}: {e}")
            return 0

    async def index_all_sessions(
        self,
        rebuild: bool = False,
    ) -> IndexingStats:
        """Index all sessions in the conversation store.

        Args:
            rebuild: If True, clear vector store before indexing

        Returns:
            IndexingStats: Statistics about the indexing operation
        """
        start_time = datetime.now()
        stats = IndexingStats(
            sessions_processed=0,
            messages_indexed=0,
            messages_skipped=0,
            errors=0,
            duration_seconds=0.0,
        )

        try:
            logger.info("Starting full conversation indexing")

            # Clear vector store if rebuild requested
            if rebuild:
                logger.warning("Rebuilding index - clearing vector store")
                await self._vector_store.clear()

            # Get all sessions
            sessions = await self._conversation_store.list_sessions(limit=1000)

            # Index each session
            for session in sessions:
                try:
                    count = await self.index_session(session.id)
                    stats.sessions_processed += 1
                    stats.messages_indexed += count
                except Exception as e:
                    logger.error(f"Error indexing session {session.id}: {e}")
                    stats.errors += 1

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            stats.duration_seconds = duration

            logger.info(
                f"Indexing complete: {stats.messages_indexed} messages "
                f"from {stats.sessions_processed} sessions in {duration:.1f}s"
            )

            return stats

        except Exception as e:
            logger.error(f"Failed to index all sessions: {e}")
            stats.errors += 1
            return stats

    async def search_conversations(
        self,
        query: str,
        limit: int = 10,
        session_id: str | None = None,
        role: str | None = None,
        days_back: int | None = None,
    ) -> list[ConversationSearchResult]:
        """Search across conversations semantically.

        Args:
            query: Search query
            limit: Maximum number of results
            session_id: Optional session ID filter
            role: Optional role filter (user/assistant)
            days_back: Optional filter to recent N days

        Returns:
            list[ConversationSearchResult]: Search results
        """
        try:
            # Build metadata filter
            metadata_filter: dict[str, Any] = {}

            if session_id:
                metadata_filter["session_id"] = session_id

            if role:
                metadata_filter["role"] = role

            # Note: days_back filtering done post-search due to ChromaDB string timestamp limitations

            # Search vector store
            results = await self._vector_store.search(
                query=query,
                limit=limit,
                filter=metadata_filter if metadata_filter else None,
            )

            # Convert to ConversationSearchResult
            search_results: list[ConversationSearchResult] = []

            # Calculate cutoff date if filtering by days_back
            cutoff_date = None
            if days_back:
                from datetime import UTC
                cutoff_date = datetime.now(UTC) - timedelta(days=days_back)

            for result in results:
                # Parse metadata
                meta = result.metadata

                timestamp = datetime.fromisoformat(meta.get("timestamp", datetime.now().isoformat()))

                # Filter by days_back if specified
                if cutoff_date and timestamp < cutoff_date:
                    continue

                search_results.append(
                    ConversationSearchResult(
                        session_id=meta.get("session_id", "unknown"),
                        session_title=meta.get("session_title"),
                        message_id=meta.get("message_id", "unknown"),
                        message_role=meta.get("role", "unknown"),
                        message_content=result.content,
                        timestamp=timestamp,
                        score=result.score,
                    )
                )

            logger.info(f"Found {len(search_results)} results for query: '{query[:50]}'")
            return search_results

        except Exception as e:
            logger.error(f"Conversation search failed: {e}")
            return []

    async def get_relevant_context(
        self,
        query: str,
        current_session_id: str,
        max_results: int = 5,
    ) -> list[ConversationSearchResult]:
        """Get relevant past context for the agent.

        Excludes messages from the current session.

        Args:
            query: Query to find relevant context
            current_session_id: Current session ID to exclude
            max_results: Maximum results to return

        Returns:
            list[ConversationSearchResult]: Relevant context from past conversations
        """
        try:
            # Search all conversations
            all_results = await self.search_conversations(
                query=query,
                limit=max_results * 2,  # Get more to filter
                days_back=90,  # Last 90 days only
            )

            # Filter out current session
            filtered_results = [
                r for r in all_results
                if r.session_id != current_session_id
            ]

            # Limit to max_results
            relevant_context = filtered_results[:max_results]

            logger.debug(
                f"Found {len(relevant_context)} relevant context messages "
                f"for query: '{query[:50]}'"
            )

            return relevant_context

        except Exception as e:
            logger.error(f"Failed to get relevant context: {e}")
            return []

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about indexed conversations.

        Returns:
            dict[str, Any]: Statistics dictionary
        """
        try:
            vector_stats = await self._vector_store.get_stats()

            return {
                "total_messages_indexed": vector_stats.count,
                "collection_name": vector_stats.name,
                "embedding_dimension": vector_stats.dimension,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_messages_indexed": 0,
                "error": str(e),
            }

    async def delete_session_index(self, session_id: str) -> int:
        """Delete all indexed messages from a session.

        Args:
            session_id: Session ID to delete

        Returns:
            int: Number of documents deleted
        """
        try:
            # Get all messages from session
            messages = await self._conversation_store.get_messages(
                session_id=session_id,
                limit=1000,
            )

            deleted_count = 0

            for message in messages:
                doc_id = self._create_document_id(session_id, str(message.id))
                success = await self._vector_store.delete_document(doc_id)
                if success:
                    deleted_count += 1

            logger.info(f"Deleted {deleted_count} indexed messages from session {session_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete session index: {e}")
            return 0
