"""Document ingestion pipeline for semantic search."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aria.logging import get_logger
from aria.memory.chunker import DocumentChunk, DocumentChunker
from aria.memory.vectors import Document, VectorStore

logger = get_logger("aria.memory.ingestion")


class IngestionResult(BaseModel):
    """Result from ingesting a single file."""

    file_path: str
    success: bool
    chunks_created: int
    error: str | None = None
    processing_time_ms: int


class BatchIngestionResult(BaseModel):
    """Result from batch ingestion."""

    total_files: int
    successful: int
    failed: int
    total_chunks: int
    results: list[IngestionResult]


class IngestedDocument(BaseModel):
    """Information about an ingested document."""

    file_path: str
    file_name: str
    file_type: str
    chunk_count: int
    ingested_at: datetime
    file_size_bytes: int
    tags: list[str] = Field(default_factory=list)


class DocumentIngester:
    """Ingest documents into vector store for semantic search.

    This class handles converting various document types into searchable chunks
    and storing them in the vector store.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        chunker: DocumentChunker | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """Initialize document ingester.

        Args:
            vector_store: Vector store to ingest into
            chunker: Optional custom chunker, creates default if None
            chunk_size: Default chunk size for new chunker
            chunk_overlap: Default overlap for new chunker
        """
        self._vector_store = vector_store
        self._chunker = chunker or DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        logger.info("Initialized DocumentIngester")

    def _create_doc_id(self, file_path: Path, chunk_index: int) -> str:
        """Create unique document ID for a chunk.

        Args:
            file_path: Source file path
            chunk_index: Chunk index

        Returns:
            str: Unique document ID
        """
        # Use file path and chunk index for ID
        # This allows us to re-ingest and update documents
        file_hash = str(file_path).replace("/", "_").replace("\\", "_")
        return f"doc_{file_hash}_{chunk_index}"

    async def ingest_text(
        self,
        text: str,
        file_path: Path | str,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> IngestionResult:
        """Ingest pre-extracted text into the vector store.

        This is useful for documents where text has already been extracted
        (e.g., via OCR) and you want to skip the extraction step.

        Args:
            text: Pre-extracted text content
            file_path: Path to associate with the text (for tracking)
            metadata: Optional metadata to attach to chunks
            tags: Optional tags for categorization

        Returns:
            IngestionResult: Result of ingestion
        """
        start_time = datetime.now()
        file_path = Path(file_path)

        try:
            logger.info(f"Ingesting pre-extracted text for: {file_path}")

            # Prepare metadata
            meta = metadata or {}
            meta["file_path"] = str(file_path)
            meta["file_name"] = file_path.name
            meta["file_type"] = file_path.suffix.lower()
            meta["ingested_at"] = datetime.now().isoformat()

            if tags:
                meta["tags"] = ",".join(tags)

            # Chunk the text directly
            chunks = self._chunker.chunk_text(
                text=text,
                metadata=meta,
                source_file=str(file_path),
            )

            if not chunks:
                return IngestionResult(
                    file_path=str(file_path),
                    success=False,
                    chunks_created=0,
                    error="No chunks created (empty text)",
                    processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                )

            # Convert chunks to Documents for vector store
            documents = []
            for chunk in chunks:
                doc_id = self._create_doc_id(file_path, chunk.chunk_index)
                documents.append(
                    Document(
                        doc_id=doc_id,
                        content=chunk.content,
                        metadata=chunk.metadata,
                    )
                )

            # Ingest into vector store
            await self._vector_store.add_documents(documents, batch_size=100)

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            logger.info(f"Ingested {len(chunks)} chunks from pre-extracted text")

            return IngestionResult(
                file_path=str(file_path),
                success=True,
                chunks_created=len(chunks),
                error=None,
                processing_time_ms=duration_ms,
            )

        except Exception as e:
            logger.error(f"Failed to ingest text for {file_path}: {e}")
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return IngestionResult(
                file_path=str(file_path),
                success=False,
                chunks_created=0,
                error=str(e),
                processing_time_ms=duration_ms,
            )

    async def ingest_file(
        self,
        file_path: Path | str,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> IngestionResult:
        """Ingest a single file into the vector store.

        Args:
            file_path: Path to file to ingest
            metadata: Optional metadata to attach to chunks
            tags: Optional tags for categorization

        Returns:
            IngestionResult: Result of ingestion
        """
        start_time = datetime.now()
        file_path = Path(file_path)

        try:
            if not file_path.exists():
                return IngestionResult(
                    file_path=str(file_path),
                    success=False,
                    chunks_created=0,
                    error=f"File not found: {file_path}",
                    processing_time_ms=0,
                )

            logger.info(f"Ingesting file: {file_path}")

            # Prepare metadata
            meta = metadata or {}
            meta["file_path"] = str(file_path)
            meta["file_name"] = file_path.name
            meta["file_type"] = file_path.suffix.lower()
            meta["file_size"] = file_path.stat().st_size
            meta["ingested_at"] = datetime.now().isoformat()

            if tags:
                meta["tags"] = ",".join(tags)

            # Chunk the file
            chunks = self._chunker.chunk_file(file_path, metadata=meta)

            if not chunks:
                return IngestionResult(
                    file_path=str(file_path),
                    success=False,
                    chunks_created=0,
                    error="No chunks created (unsupported file type or empty file)",
                    processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                )

            # Convert chunks to Documents for vector store
            documents = []
            for chunk in chunks:
                doc_id = self._create_doc_id(file_path, chunk.chunk_index)
                documents.append(
                    Document(
                        doc_id=doc_id,
                        content=chunk.content,
                        metadata=chunk.metadata,
                    )
                )

            # Ingest into vector store
            await self._vector_store.add_documents(documents, batch_size=100)

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            logger.info(f"Ingested {len(chunks)} chunks from {file_path.name}")

            return IngestionResult(
                file_path=str(file_path),
                success=True,
                chunks_created=len(chunks),
                error=None,
                processing_time_ms=duration_ms,
            )

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return IngestionResult(
                file_path=str(file_path),
                success=False,
                chunks_created=0,
                error=str(e),
                processing_time_ms=duration_ms,
            )

    async def ingest_directory(
        self,
        directory: Path | str,
        pattern: str = "**/*.pdf",
        recursive: bool = True,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> BatchIngestionResult:
        """Ingest all matching files in a directory.

        Args:
            directory: Directory to scan
            pattern: Glob pattern for file matching
            recursive: Whether to recurse into subdirectories
            metadata: Optional metadata for all files
            tags: Optional tags for all files

        Returns:
            BatchIngestionResult: Batch ingestion results
        """
        directory = Path(directory)

        if not directory.exists() or not directory.is_dir():
            logger.error(f"Directory not found: {directory}")
            return BatchIngestionResult(
                total_files=0,
                successful=0,
                failed=0,
                total_chunks=0,
                results=[],
            )

        logger.info(f"Ingesting directory: {directory} with pattern '{pattern}'")

        # Find matching files
        if recursive:
            files = list(directory.glob(pattern))
        else:
            # Non-recursive: use pattern without **
            simple_pattern = pattern.replace("**/", "")
            files = list(directory.glob(simple_pattern))

        logger.info(f"Found {len(files)} files matching pattern")

        # Ingest each file
        results = []
        for file_path in files:
            if file_path.is_file():
                result = await self.ingest_file(file_path, metadata, tags)
                results.append(result)

        # Calculate totals
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        total_chunks = sum(r.chunks_created for r in results)

        logger.info(
            f"Batch ingestion complete: {successful} successful, "
            f"{failed} failed, {total_chunks} total chunks"
        )

        return BatchIngestionResult(
            total_files=len(results),
            successful=successful,
            failed=failed,
            total_chunks=total_chunks,
            results=results,
        )

    async def remove_document(self, file_path: Path | str) -> bool:
        """Remove a document from the index.

        Args:
            file_path: Path to document to remove

        Returns:
            bool: True if any chunks were removed
        """
        file_path = Path(file_path)

        try:
            # We need to find and delete all chunks for this document
            # Since we don't have a direct way to query by file_path metadata,
            # we'll need to search for it

            logger.info(f"Removing document: {file_path}")

            # Search for chunks from this document
            results = await self._vector_store.search(
                query=str(file_path),  # Use file path as query
                limit=1000,  # Get many results
                filter={"file_path": str(file_path)},
            )

            deleted_count = 0

            for result in results:
                # Verify this is from our file
                if result.metadata.get("file_path") == str(file_path):
                    success = await self._vector_store.delete_document(result.doc_id)
                    if success:
                        deleted_count += 1

            logger.info(f"Removed {deleted_count} chunks for {file_path.name}")
            return deleted_count > 0

        except Exception as e:
            logger.error(f"Failed to remove document {file_path}: {e}")
            return False

    async def list_ingested(self) -> list[IngestedDocument]:
        """List all ingested documents.

        Returns:
            list[IngestedDocument]: List of ingested documents
        """
        try:
            # Get stats from vector store
            stats = await self._vector_store.get_stats()

            # We need to group chunks by file_path
            # This is expensive - we'd search and group
            # For now, return basic info

            # Note: This is a limitation - we should store document-level
            # metadata separately for efficient listing
            # For now, we'll search and deduplicate

            results = await self._vector_store.search(
                query="document",  # Generic query to get results
                limit=1000,
            )

            # Group by file_path
            docs_map: dict[str, IngestedDocument] = {}

            for result in results:
                file_path = result.metadata.get("file_path")
                if not file_path:
                    continue

                if file_path not in docs_map:
                    docs_map[file_path] = IngestedDocument(
                        file_path=file_path,
                        file_name=result.metadata.get("file_name", Path(file_path).name),
                        file_type=result.metadata.get("file_type", "unknown"),
                        chunk_count=1,
                        ingested_at=datetime.fromisoformat(
                            result.metadata.get("ingested_at", datetime.now().isoformat())
                        ),
                        file_size_bytes=result.metadata.get("file_size", 0),
                        tags=result.metadata.get("tags", "").split(",")
                        if result.metadata.get("tags")
                        else [],
                    )
                else:
                    docs_map[file_path].chunk_count += 1

            return list(docs_map.values())

        except Exception as e:
            logger.error(f"Failed to list ingested documents: {e}")
            return []

    async def get_document_chunks(self, file_path: Path | str) -> list[DocumentChunk]:
        """Get all chunks for a specific document.

        Args:
            file_path: Path to document

        Returns:
            list[DocumentChunk]: List of chunks
        """
        file_path = Path(file_path)

        try:
            # Search for chunks from this document
            results = await self._vector_store.search(
                query=str(file_path),
                limit=1000,
                filter={"file_path": str(file_path)},
            )

            chunks = []
            for idx, result in enumerate(results):
                if result.metadata.get("file_path") == str(file_path):
                    chunks.append(
                        DocumentChunk(
                            content=result.content,
                            metadata=result.metadata,
                            chunk_index=idx,
                            total_chunks=len(results),
                            source_file=str(file_path),
                            page_number=result.metadata.get("page_number"),
                        )
                    )

            logger.info(f"Found {len(chunks)} chunks for {file_path.name}")
            return chunks

        except Exception as e:
            logger.error(f"Failed to get chunks for {file_path}: {e}")
            return []
