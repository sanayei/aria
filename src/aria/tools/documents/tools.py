"""Document management tools for ingestion and Q&A."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aria.llm.client import OllamaClient
from aria.memory.ingestion import (
    BatchIngestionResult,
    DocumentIngester,
    IngestedDocument,
    IngestionResult,
)
from aria.memory.vectors import SearchResult, VectorStore
from aria.tools.base import BaseTool, RiskLevel, ToolResult
from aria.tools.documents.qa import DocumentQA, QAResponse

logger = logging.getLogger(__name__)


# ============================================================================
# Ingestion Tools
# ============================================================================


class IngestDocumentParams(BaseModel):
    """Parameters for ingesting a single document."""

    file_path: str = Field(..., description="Path to the document to ingest")
    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags to associate with the document",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional additional metadata",
    )


class IngestDocumentTool(BaseTool[IngestDocumentParams]):
    """Tool for ingesting a document into the vector store for semantic search."""

    def __init__(self, ingester: DocumentIngester):
        """Initialize the tool.

        Args:
            ingester: DocumentIngester instance
        """
        self._ingester = ingester

    @property
    def name(self) -> str:
        return "ingest_document"

    @property
    def description(self) -> str:
        return (
            "Ingest a document (PDF, TXT, MD, HTML) into the vector store "
            "for semantic search and Q&A. The document will be chunked "
            "and embedded for later retrieval."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW  # Read-only, just adds to vector store

    async def execute(self, params: IngestDocumentParams) -> ToolResult:
        """Execute document ingestion."""
        try:
            file_path = Path(params.file_path).expanduser().resolve()

            if not file_path.exists():
                return ToolResult.error_result(f"File not found: {file_path}")

            result = await self._ingester.ingest_file(
                file_path=file_path,
                metadata=params.metadata if params.metadata else None,
                tags=params.tags if params.tags else None,
            )

            if result.success:
                return ToolResult.success_result(
                    data={
                        "file_path": result.file_path,
                        "chunks_created": result.chunks_created,
                        "processing_time_ms": result.processing_time_ms,
                    },
                    message=f"Successfully ingested {file_path.name} ({result.chunks_created} chunks)",
                )
            else:
                return ToolResult.error_result(
                    error=result.error or "Unknown ingestion error"
                )

        except Exception as e:
            logger.error(f"Error ingesting document: {e}", exc_info=True)
            return ToolResult.error_result(f"Ingestion failed: {str(e)}")


class IngestDirectoryParams(BaseModel):
    """Parameters for ingesting a directory of documents."""

    directory: str = Field(..., description="Path to the directory to ingest")
    pattern: str = Field(
        default="**/*.pdf",
        description="Glob pattern for files to ingest (default: **/*.pdf)",
    )
    recursive: bool = Field(
        default=True,
        description="Whether to search recursively",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags to associate with all documents",
    )


class IngestDirectoryTool(BaseTool[IngestDirectoryParams]):
    """Tool for batch ingesting documents from a directory."""

    def __init__(self, ingester: DocumentIngester):
        """Initialize the tool.

        Args:
            ingester: DocumentIngester instance
        """
        self._ingester = ingester

    @property
    def name(self) -> str:
        return "ingest_directory"

    @property
    def description(self) -> str:
        return (
            "Batch ingest documents from a directory using a glob pattern. "
            "Useful for adding multiple documents at once."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    async def execute(self, params: IngestDirectoryParams) -> ToolResult:
        """Execute directory ingestion."""
        try:
            directory = Path(params.directory).expanduser().resolve()

            if not directory.exists():
                return ToolResult.error_result(f"Directory not found: {directory}")

            if not directory.is_dir():
                return ToolResult.error_result(f"Not a directory: {directory}")

            result = await self._ingester.ingest_directory(
                directory=directory,
                pattern=params.pattern,
                recursive=params.recursive,
                metadata=None,
                tags=params.tags if params.tags else None,
            )

            return ToolResult.success_result(
                data={
                    "total_files": result.total_files,
                    "successful": result.successful,
                    "failed": result.failed,
                    "total_chunks": result.total_chunks,
                },
                message=(
                    f"Ingested {result.successful}/{result.total_files} documents "
                    f"({result.total_chunks} chunks total)"
                ),
            )

        except Exception as e:
            logger.error(f"Error ingesting directory: {e}", exc_info=True)
            return ToolResult.error_result(f"Directory ingestion failed: {str(e)}")


class SearchDocumentsParams(BaseModel):
    """Parameters for searching ingested documents."""

    query: str = Field(..., description="The search query")
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return",
    )
    file_type: str | None = Field(
        default=None,
        description="Filter by file type (e.g., '.pdf', '.txt')",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Filter by tags",
    )


class SearchDocumentsTool(BaseTool[SearchDocumentsParams]):
    """Tool for semantic search across ingested documents."""

    def __init__(self, vector_store: VectorStore):
        """Initialize the tool.

        Args:
            vector_store: VectorStore instance
        """
        self._vector_store = vector_store

    @property
    def name(self) -> str:
        return "search_documents"

    @property
    def description(self) -> str:
        return (
            "Search through ingested documents using semantic search. "
            "Returns relevant document chunks with metadata."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    async def execute(self, params: SearchDocumentsParams) -> ToolResult:
        """Execute document search."""
        try:
            # Build metadata filter
            metadata_filter = {}
            if params.file_type:
                metadata_filter["file_type"] = params.file_type
            if params.tags:
                # Tags are stored as comma-separated string
                metadata_filter["tags"] = ",".join(params.tags)

            results = await self._vector_store.search(
                query=params.query,
                limit=params.max_results,
                metadata_filter=metadata_filter if metadata_filter else None,
            )

            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "content": result.content,
                        "score": result.score,
                        "file_name": result.metadata.get("file_name", "unknown"),
                        "file_path": result.metadata.get("file_path", "unknown"),
                        "page_number": result.metadata.get("page_number"),
                    }
                )

            return ToolResult.success_result(
                data={"results": formatted_results, "count": len(formatted_results)},
                message=f"Found {len(formatted_results)} relevant chunks",
            )

        except Exception as e:
            logger.error(f"Error searching documents: {e}", exc_info=True)
            return ToolResult.error_result(f"Search failed: {str(e)}")


class ListIngestedDocumentsParams(BaseModel):
    """Parameters for listing ingested documents."""

    # No parameters needed - lists all documents


class ListIngestedDocumentsTool(BaseTool[ListIngestedDocumentsParams]):
    """Tool for listing all ingested documents."""

    def __init__(self, ingester: DocumentIngester):
        """Initialize the tool.

        Args:
            ingester: DocumentIngester instance
        """
        self._ingester = ingester

    @property
    def name(self) -> str:
        return "list_ingested_documents"

    @property
    def description(self) -> str:
        return "List all documents that have been ingested into the vector store."

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    async def execute(self, params: ListIngestedDocumentsParams) -> ToolResult:
        """Execute document listing."""
        try:
            documents = await self._ingester.list_ingested()

            formatted_docs = []
            for doc in documents:
                formatted_docs.append(
                    {
                        "file_name": doc.file_name,
                        "file_path": doc.file_path,
                        "file_type": doc.file_type,
                        "chunk_count": doc.chunk_count,
                        "ingested_at": doc.ingested_at.isoformat(),
                        "file_size_bytes": doc.file_size_bytes,
                        "tags": doc.tags,
                    }
                )

            return ToolResult.success_result(
                data={"documents": formatted_docs, "count": len(formatted_docs)},
                message=f"Found {len(formatted_docs)} ingested documents",
            )

        except Exception as e:
            logger.error(f"Error listing documents: {e}", exc_info=True)
            return ToolResult.error_result(f"Listing failed: {str(e)}")


# ============================================================================
# Q&A Tools
# ============================================================================


class AskDocumentsParams(BaseModel):
    """Parameters for asking questions about documents."""

    question: str = Field(..., description="The question to answer")
    file_path: str | None = Field(
        default=None,
        description="Optional: restrict to a specific document",
    )
    file_type: str | None = Field(
        default=None,
        description="Optional: filter by file type (e.g., '.pdf')",
    )
    max_chunks: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of document chunks to use",
    )


class AskDocumentsTool(BaseTool[AskDocumentsParams]):
    """Tool for asking questions about ingested documents using RAG."""

    def __init__(self, qa_system: DocumentQA):
        """Initialize the tool.

        Args:
            qa_system: DocumentQA instance
        """
        self._qa_system = qa_system

    @property
    def name(self) -> str:
        return "ask_documents"

    @property
    def description(self) -> str:
        return (
            "Ask questions about ingested documents. Uses RAG (Retrieval Augmented Generation) "
            "to search relevant document chunks and generate accurate answers with source citations."
        )

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    async def execute(self, params: AskDocumentsParams) -> ToolResult:
        """Execute document Q&A."""
        try:
            # Build document filter
            document_filter = {}
            if params.file_path:
                document_filter["file_path"] = params.file_path
            if params.file_type:
                document_filter["file_type"] = params.file_type

            # Ask the question
            response = await self._qa_system.ask(
                question=params.question,
                document_filter=document_filter if document_filter else None,
                include_sources=True,
                max_chunks=params.max_chunks,
            )

            # Format sources
            sources = []
            for source in response.sources:
                sources.append(
                    {
                        "file_name": source.file_name,
                        "page_number": source.page_number,
                        "relevance_score": source.relevance_score,
                        "excerpt": source.chunk_text[:200] + "..."
                        if len(source.chunk_text) > 200
                        else source.chunk_text,
                    }
                )

            return ToolResult.success_result(
                data={
                    "question": response.question,
                    "answer": response.answer,
                    "confidence": response.confidence,
                    "chunks_used": response.chunks_used,
                    "sources": sources,
                },
                message=f"Answer generated with {response.confidence:.1%} confidence using {response.chunks_used} chunks",
            )

        except Exception as e:
            logger.error(f"Error answering question: {e}", exc_info=True)
            return ToolResult.error_result(f"Q&A failed: {str(e)}")
