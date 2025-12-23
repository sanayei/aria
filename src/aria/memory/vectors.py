"""ChromaDB-based vector storage for semantic search."""

import asyncio
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel, Field

from aria.logging import get_logger
from aria.memory.embeddings import EmbeddingProvider

logger = get_logger("aria.memory.vectors")


class VectorStoreError(Exception):
    """Exception raised when vector store operations fail."""

    pass


class Document(BaseModel):
    """Document to be stored in vector store."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    doc_id: str | None = None

    def with_id(self) -> "Document":
        """Return a copy with a generated ID if missing.

        Returns:
            Document: Document with ID set
        """
        if self.doc_id is None:
            return self.model_copy(update={"doc_id": str(uuid.uuid4())})
        return self


class SearchResult(BaseModel):
    """Result from semantic search."""

    doc_id: str
    content: str
    metadata: dict[str, Any]
    score: float  # Similarity score (0-1, higher is better)
    distance: float  # Raw distance from ChromaDB


class CollectionStats(BaseModel):
    """Statistics about a ChromaDB collection."""

    name: str
    count: int
    dimension: int | None = None


class VectorStore:
    """ChromaDB-based vector storage for semantic search.

    This class provides a clean interface for storing and searching
    documents using semantic similarity via embeddings.
    """

    def __init__(
        self,
        persist_directory: Path,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "aria_default",
    ):
        """Initialize the vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_provider: Provider for generating embeddings
            collection_name: Name of the ChromaDB collection
        """
        self._persist_dir = Path(persist_directory).expanduser()
        self._embedding_provider = embedding_provider
        self._collection_name = collection_name
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

        logger.info(
            f"Initialized VectorStore: collection='{collection_name}', "
            f"persist_dir={self._persist_dir}"
        )

    async def initialize(self) -> None:
        """Initialize ChromaDB and create/get collection.

        This must be called before using the vector store.

        Raises:
            VectorStoreError: If initialization fails
        """
        try:
            # Create persist directory if it doesn't exist
            self._persist_dir.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client with persistence
            logger.info(f"Initializing ChromaDB at {self._persist_dir}")

            # Run in thread pool as ChromaDB is synchronous
            self._client = await asyncio.to_thread(
                chromadb.PersistentClient,
                path=str(self._persist_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Get or create collection
            self._collection = await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )

            logger.info(
                f"ChromaDB collection '{self._collection_name}' ready "
                f"(count: {await self._get_count()})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise VectorStoreError(f"Initialization failed: {e}") from e

    async def _ensure_initialized(self) -> None:
        """Ensure the vector store is initialized.

        Raises:
            VectorStoreError: If not initialized
        """
        if self._client is None or self._collection is None:
            raise VectorStoreError("VectorStore not initialized. Call initialize() first.")

    async def _get_count(self) -> int:
        """Get the number of documents in the collection.

        Returns:
            int: Number of documents
        """
        if self._collection is None:
            return 0
        return await asyncio.to_thread(self._collection.count)

    async def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Add a document to the vector store.

        Args:
            content: Document content (text)
            metadata: Optional metadata dictionary
            doc_id: Optional document ID (generated if None)

        Returns:
            str: Document ID

        Raises:
            VectorStoreError: If operation fails
        """
        await self._ensure_initialized()

        try:
            # Generate ID if not provided
            if doc_id is None:
                doc_id = str(uuid.uuid4())

            # Prepare metadata (ChromaDB requires non-empty dict)
            meta = metadata or {"_empty": True}

            # Generate embedding
            logger.debug(f"Generating embedding for document '{doc_id}'")
            embedding = await self._embedding_provider.embed_text(content)

            # Add to ChromaDB
            await asyncio.to_thread(
                self._collection.add,  # type: ignore
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[meta],
            )

            logger.debug(f"Added document '{doc_id}' to collection")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise VectorStoreError(f"Failed to add document: {e}") from e

    async def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 100,
    ) -> list[str]:
        """Batch add documents to the vector store.

        Args:
            documents: List of documents to add
            batch_size: Number of documents to process per batch

        Returns:
            list[str]: List of document IDs

        Raises:
            VectorStoreError: If operation fails
        """
        await self._ensure_initialized()

        if not documents:
            return []

        try:
            all_doc_ids: list[str] = []

            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]

                # Ensure all docs have IDs
                batch_with_ids = [doc.with_id() for doc in batch]

                # Extract data
                doc_ids = [doc.doc_id for doc in batch_with_ids]  # type: ignore
                contents = [doc.content for doc in batch_with_ids]
                # ChromaDB requires non-empty metadata dicts
                metadatas = [doc.metadata or {"_empty": True} for doc in batch_with_ids]

                # Generate embeddings in batch
                logger.debug(f"Generating embeddings for {len(batch)} documents")
                embeddings = await self._embedding_provider.embed_batch(contents)

                # Add to ChromaDB
                await asyncio.to_thread(
                    self._collection.add,  # type: ignore
                    ids=doc_ids,
                    embeddings=embeddings,
                    documents=contents,
                    metadatas=metadatas,
                )

                all_doc_ids.extend(doc_ids)
                logger.debug(f"Added batch of {len(batch)} documents")

            logger.info(f"Added {len(all_doc_ids)} documents to collection")
            return all_doc_ids

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}") from e

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID.

        Args:
            doc_id: Document ID to delete

        Returns:
            bool: True if deleted, False if not found

        Raises:
            VectorStoreError: If operation fails
        """
        await self._ensure_initialized()

        try:
            # Check if document exists
            result = await asyncio.to_thread(
                self._collection.get,  # type: ignore
                ids=[doc_id],
            )

            if not result["ids"]:
                logger.debug(f"Document '{doc_id}' not found")
                return False

            # Delete document
            await asyncio.to_thread(
                self._collection.delete,  # type: ignore
                ids=[doc_id],
            )

            logger.debug(f"Deleted document '{doc_id}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise VectorStoreError(f"Failed to delete document: {e}") from e

    async def search(
        self,
        query: str,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Semantic search over documents.

        Args:
            query: Search query text
            limit: Maximum number of results
            filter: Optional metadata filter (ChromaDB where clause)

        Returns:
            list[SearchResult]: Search results ordered by relevance

        Raises:
            VectorStoreError: If search fails
        """
        await self._ensure_initialized()

        try:
            # Generate query embedding
            logger.debug(f"Searching for: '{query[:100]}...'")
            query_embedding = await self._embedding_provider.embed_text(query)

            # Search using embedding
            return await self.search_by_embedding(
                embedding=query_embedding,
                limit=limit,
                filter=filter,
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Search failed: {e}") from e

    async def search_by_embedding(
        self,
        embedding: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search using pre-computed embedding.

        Args:
            embedding: Query embedding vector
            limit: Maximum number of results
            filter: Optional metadata filter (ChromaDB where clause)

        Returns:
            list[SearchResult]: Search results ordered by relevance

        Raises:
            VectorStoreError: If search fails
        """
        await self._ensure_initialized()

        try:
            # Query ChromaDB
            query_params: dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": limit,
                "include": ["documents", "metadatas", "distances"],
            }

            if filter:
                query_params["where"] = filter

            results = await asyncio.to_thread(
                self._collection.query,  # type: ignore
                **query_params,
            )

            # Parse results
            search_results: list[SearchResult] = []

            if results["ids"] and results["ids"][0]:
                for idx in range(len(results["ids"][0])):
                    doc_id = results["ids"][0][idx]
                    content = results["documents"][0][idx]
                    metadata = results["metadatas"][0][idx] or {}
                    distance = results["distances"][0][idx]

                    # Convert distance to similarity score (cosine similarity)
                    # ChromaDB returns cosine distance (1 - cosine_similarity)
                    # So similarity = 1 - distance
                    score = max(0.0, 1.0 - distance)

                    search_results.append(
                        SearchResult(
                            doc_id=doc_id,
                            content=content,
                            metadata=metadata,
                            score=score,
                            distance=distance,
                        )
                    )

            logger.debug(f"Found {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            raise VectorStoreError(f"Embedding search failed: {e}") from e

    async def get_document(self, doc_id: str) -> Document | None:
        """Get a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document | None: Document if found, None otherwise

        Raises:
            VectorStoreError: If operation fails
        """
        await self._ensure_initialized()

        try:
            result = await asyncio.to_thread(
                self._collection.get,  # type: ignore
                ids=[doc_id],
                include=["documents", "metadatas"],
            )

            if not result["ids"]:
                return None

            return Document(
                doc_id=result["ids"][0],
                content=result["documents"][0],
                metadata=result["metadatas"][0] or {},
            )

        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            raise VectorStoreError(f"Failed to get document: {e}") from e

    async def get_stats(self) -> CollectionStats:
        """Get collection statistics.

        Returns:
            CollectionStats: Collection statistics

        Raises:
            VectorStoreError: If operation fails
        """
        await self._ensure_initialized()

        try:
            count = await self._get_count()

            return CollectionStats(
                name=self._collection_name,
                count=count,
                dimension=self._embedding_provider.dimension,
            )

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise VectorStoreError(f"Failed to get stats: {e}") from e

    async def clear(self) -> None:
        """Clear all documents from collection.

        Warning: This is irreversible!

        Raises:
            VectorStoreError: If operation fails
        """
        await self._ensure_initialized()

        try:
            # Delete and recreate collection
            logger.warning(f"Clearing collection '{self._collection_name}'")

            await asyncio.to_thread(
                self._client.delete_collection,  # type: ignore
                name=self._collection_name,
            )

            self._collection = await asyncio.to_thread(
                self._client.get_or_create_collection,  # type: ignore
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(f"Cleared collection '{self._collection_name}'")

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise VectorStoreError(f"Failed to clear collection: {e}") from e

    async def close(self) -> None:
        """Close the ChromaDB client and release resources."""
        if self._client:
            logger.info("Closing ChromaDB client")
            self._client = None
            self._collection = None
