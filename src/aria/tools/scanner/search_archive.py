"""Archive search tools for finding and retrieving processed documents."""

from pathlib import Path

from pydantic import BaseModel, Field

from aria.logging import get_logger
from aria.memory.archive import ArchiveIndex
from aria.memory.vectors import VectorStore
from aria.tools import BaseTool, RiskLevel, ToolResult

logger = get_logger("aria.tools.scanner.search")


# =============================================================================
# Search Archived Documents Tool
# =============================================================================


class SearchArchivedDocumentsParams(BaseModel):
    """Parameters for searching archived documents."""

    query: str = Field(description="Semantic search query (searches document content)")
    person: str | None = Field(default=None, description="Filter by person name")
    category: str | None = Field(default=None, description="Filter by category")
    year: int | None = Field(default=None, description="Filter by year")
    tags: list[str] = Field(default_factory=list, description="Filter by tags (match any)")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results to return")


class SearchArchivedDocumentsTool(BaseTool[SearchArchivedDocumentsParams]):
    """Search archived documents using semantic search and metadata filters.

    This tool combines:
    1. ChromaDB semantic search for content matching
    2. Archive index metadata filtering
    3. Returns documents with file paths and relevant excerpts

    Use this to find documents by content (e.g., "Social Security benefits",
    "medical bills from 2024").
    """

    name = "search_archived_documents"
    description = (
        "Search archived documents by content and metadata. "
        "Performs semantic search across document text and filters by "
        "person, category, year, and tags. Returns matching documents "
        "with file paths and relevant excerpts."
    )
    risk_level = RiskLevel.LOW
    parameters_schema = SearchArchivedDocumentsParams

    def __init__(self, vector_store: VectorStore, archive_index: ArchiveIndex):
        """Initialize the search tool.

        Args:
            vector_store: Vector store for semantic search
            archive_index: Archive index for metadata
        """
        super().__init__()
        self._vector_store = vector_store
        self._archive_index = archive_index

    def get_confirmation_message(self, params: SearchArchivedDocumentsParams) -> str:
        """Get confirmation message (not needed for read-only search)."""
        return f"Search for: {params.query}"

    async def execute(self, params: SearchArchivedDocumentsParams) -> ToolResult:
        """Execute archive search."""
        try:
            # Build metadata filter for ChromaDB (only non-date filters)
            # Note: ChromaDB comparison operators only work with numeric values,
            # so we'll do year filtering post-search in Python
            filter_conditions = []

            if params.person:
                filter_conditions.append({"person": params.person})

            if params.category:
                filter_conditions.append({"category": params.category})

            # Combine all filter conditions with $and
            metadata_filter = None
            if len(filter_conditions) == 1:
                metadata_filter = filter_conditions[0]
            elif len(filter_conditions) > 1:
                metadata_filter = {"$and": filter_conditions}

            # Perform semantic search
            search_results = await self._vector_store.search(
                query=params.query,
                limit=params.max_results * 3,  # Get extra to account for post-filtering
                filter=metadata_filter,
            )

            # Deduplicate by archived_path and build result list
            seen_paths = set()
            results = []

            for result in search_results:
                archived_path = result.metadata.get("archived_path")
                if not archived_path or archived_path in seen_paths:
                    continue

                seen_paths.add(archived_path)

                # Year filtering (post-search since ChromaDB doesn't support date comparison)
                if params.year:
                    document_date = result.metadata.get("document_date")
                    if document_date:
                        # Extract year from YYYY-MM-DD format
                        try:
                            doc_year = int(document_date.split("-")[0])
                            if doc_year != params.year:
                                continue
                        except (ValueError, IndexError):
                            # Skip documents with invalid date format
                            continue
                    else:
                        # Skip documents without a date if year filter is specified
                        continue

                # Tag filtering (post-search since ChromaDB doesn't support it well)
                if params.tags:
                    doc_tags = result.metadata.get("tags", "").split(",")
                    if not any(tag in doc_tags for tag in params.tags):
                        continue
                results.append(
                    {
                        "archived_path": archived_path,
                        "person": result.metadata.get("person"),
                        "category": result.metadata.get("category"),
                        "document_date": result.metadata.get("document_date"),
                        "sender": result.metadata.get("sender"),
                        "summary": result.metadata.get("summary"),
                        "tags": result.metadata.get("tags", "").split(",")
                        if result.metadata.get("tags")
                        else [],
                        "relevance_score": result.score,
                        "excerpt": result.content[:300] + "..."
                        if len(result.content) > 300
                        else result.content,
                    }
                )

                if len(results) >= params.max_results:
                    break

            return ToolResult.success_result(
                data={
                    "query": params.query,
                    "filters": {
                        "person": params.person,
                        "category": params.category,
                        "year": params.year,
                        "tags": params.tags,
                    },
                    "count": len(results),
                    "results": results,
                }
            )

        except Exception as e:
            logger.error(f"Archive search failed: {e}", exc_info=True)
            return ToolResult.error_result(f"Search failed: {str(e)}")


# =============================================================================
# List Archived Documents Tool
# =============================================================================


class ListArchivedDocumentsParams(BaseModel):
    """Parameters for listing archived documents."""

    person: str | None = Field(default=None, description="Filter by person name")
    category: str | None = Field(default=None, description="Filter by category")
    year: int | None = Field(default=None, description="Filter by year")
    sender: str | None = Field(default=None, description="Filter by sender (partial match)")
    tag: str | None = Field(default=None, description="Filter by tag (partial match)")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum results to return")


class ListArchivedDocumentsTool(BaseTool[ListArchivedDocumentsParams]):
    """List archived documents by metadata filters (no semantic search).

    Use this to browse documents by person, category, year, etc.
    without performing semantic search.

    Examples:
    - List all documents for "John Smith"
    - List all medical documents from 2024
    - List all IRS letters
    """

    name = "list_archived_documents"
    description = (
        "List archived documents filtered by metadata (person, category, "
        "year, sender, tags). Does NOT perform semantic search - "
        "use search_archived_documents for content search."
    )
    risk_level = RiskLevel.LOW
    parameters_schema = ListArchivedDocumentsParams

    def __init__(self, archive_index: ArchiveIndex):
        """Initialize the list tool.

        Args:
            archive_index: Archive index for metadata queries
        """
        super().__init__()
        self._archive_index = archive_index

    def get_confirmation_message(self, params: ListArchivedDocumentsParams) -> str:
        """Get confirmation message (not needed for read-only list)."""
        filters = []
        if params.person:
            filters.append(f"person={params.person}")
        if params.category:
            filters.append(f"category={params.category}")
        return f"List documents ({', '.join(filters) if filters else 'all'})"

    async def execute(self, params: ListArchivedDocumentsParams) -> ToolResult:
        """Execute document listing."""
        try:
            # Query archive index
            documents = await self._archive_index.search(
                person=params.person,
                category=params.category,
                year=params.year,
                sender=params.sender,
                tag=params.tag,
                limit=params.limit,
            )

            # Format results
            results = [
                {
                    "id": doc.id,
                    "archived_path": doc.archived_path,
                    "person": doc.person,
                    "category": doc.category,
                    "document_date": doc.document_date,
                    "sender": doc.sender,
                    "summary": doc.summary,
                    "tags": doc.tags,
                    "processed_at": doc.processed_at,
                    "file_size_bytes": doc.file_size_bytes,
                }
                for doc in documents
            ]

            return ToolResult.success_result(
                data={
                    "filters": {
                        "person": params.person,
                        "category": params.category,
                        "year": params.year,
                        "sender": params.sender,
                        "tag": params.tag,
                    },
                    "count": len(results),
                    "results": results,
                }
            )

        except Exception as e:
            logger.error(f"List documents failed: {e}", exc_info=True)
            return ToolResult.error_result(f"Listing failed: {str(e)}")


# =============================================================================
# Get Archived Document Tool
# =============================================================================


class GetArchivedDocumentParams(BaseModel):
    """Parameters for getting a specific archived document."""

    document_id: str = Field(description="Document ID from archive index")


class GetArchivedDocumentTool(BaseTool[GetArchivedDocumentParams]):
    """Retrieve detailed information about a specific archived document.

    Returns complete metadata, file paths, and all chunks from ChromaDB.
    """

    name = "get_archived_document"
    description = (
        "Get detailed information about a specific archived document by ID. "
        "Returns complete metadata, file paths, and document chunks."
    )
    risk_level = RiskLevel.LOW
    parameters_schema = GetArchivedDocumentParams

    def __init__(self, archive_index: ArchiveIndex, vector_store: VectorStore):
        """Initialize the get document tool.

        Args:
            archive_index: Archive index for document lookup
            vector_store: Vector store for chunk retrieval
        """
        super().__init__()
        self._archive_index = archive_index
        self._vector_store = vector_store

    def get_confirmation_message(self, params: GetArchivedDocumentParams) -> str:
        """Get confirmation message (not needed for read-only get)."""
        return f"Get document: {params.document_id}"

    async def execute(self, params: GetArchivedDocumentParams) -> ToolResult:
        """Execute document retrieval."""
        try:
            # Get document from archive index
            document = await self._archive_index.get_document(params.document_id)

            if not document:
                return ToolResult.error_result(f"Document not found: {params.document_id}")

            # Check if file still exists
            file_exists = Path(document.archived_path).exists()

            # Format result
            result = {
                "id": document.id,
                "original_filename": document.original_filename,
                "original_path": document.original_path,
                "archived_path": document.archived_path,
                "file_exists": file_exists,
                "person": document.person,
                "category": document.category,
                "document_date": document.document_date,
                "sender": document.sender,
                "summary": document.summary,
                "tags": document.tags,
                "ocr_confidence": document.ocr_confidence,
                "processed_at": document.processed_at,
                "file_size_bytes": document.file_size_bytes,
                "chroma_chunks": len(document.chroma_doc_ids),
            }

            return ToolResult.success_result(data=result)

        except Exception as e:
            logger.error(f"Get document failed: {e}", exc_info=True)
            return ToolResult.error_result(f"Retrieval failed: {str(e)}")


# =============================================================================
# Archive Statistics Tool
# =============================================================================


class ArchiveStatisticsParams(BaseModel):
    """Parameters for getting archive statistics."""

    # No parameters needed


class ArchiveStatisticsTool(BaseTool[ArchiveStatisticsParams]):
    """Get statistics about the document archive.

    Returns counts by person, category, year, total size, etc.
    """

    name = "archive_statistics"
    description = (
        "Get statistics about the document archive including "
        "counts by person, category, year, total documents, and total size."
    )
    risk_level = RiskLevel.LOW
    parameters_schema = ArchiveStatisticsParams

    def __init__(self, archive_index: ArchiveIndex):
        """Initialize the statistics tool.

        Args:
            archive_index: Archive index for statistics queries
        """
        super().__init__()
        self._archive_index = archive_index

    def get_confirmation_message(self, params: ArchiveStatisticsParams) -> str:
        """Get confirmation message (not needed for read-only stats)."""
        return "Get archive statistics"

    async def execute(self, params: ArchiveStatisticsParams) -> ToolResult:
        """Execute statistics query."""
        try:
            stats = await self._archive_index.get_statistics()

            return ToolResult.success_result(
                data={
                    "total_documents": stats.total_documents,
                    "documents_by_person": stats.documents_by_person,
                    "documents_by_category": stats.documents_by_category,
                    "documents_by_year": stats.documents_by_year,
                    "total_size_bytes": stats.total_size_bytes,
                    "total_size_mb": round(stats.total_size_bytes / (1024 * 1024), 2),
                    "earliest_document": stats.earliest_document,
                    "latest_document": stats.latest_document,
                }
            )

        except Exception as e:
            logger.error(f"Get statistics failed: {e}", exc_info=True)
            return ToolResult.error_result(f"Statistics query failed: {str(e)}")
