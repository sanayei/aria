"""Document API endpoints."""

import io
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import FileResponse, StreamingResponse
from pdf2image import convert_from_path
from PIL import Image
from pydantic import BaseModel, Field

from aria.config import get_settings
from aria.logging import get_logger
from aria.memory import ArchiveIndex, VectorStore
from aria.memory.embeddings import OllamaEmbeddings
from aria.tools.scanner import SearchArchivedDocumentsParams, SearchArchivedDocumentsTool
from aria.web.auth import build_document_filters, check_document_access, get_current_active_user
from aria.web.dependencies import get_archive_index, get_vector_store
from aria.web.models import User

logger = get_logger("aria.web.api.documents")

router = APIRouter(prefix="/api/documents", tags=["documents"])


# =============================================================================
# Request/Response Models
# =============================================================================


class DocumentUpdateRequest(BaseModel):
    """Request model for updating document metadata."""

    person: str | None = Field(default=None, description="Update person")
    category: str | None = Field(default=None, description="Update category")
    summary: str | None = Field(default=None, description="Update summary")
    tags: list[str] | None = Field(default=None, description="Update tags")
    sender: str | None = Field(default=None, description="Update sender")


@router.get("/search")
async def search_documents(
    query: str = Query(..., description="Search query"),
    person: str | None = Query(None, description="Filter by person"),
    category: str | None = Query(None, description="Filter by category"),
    year: int | None = Query(None, description="Filter by year"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
    vector_store: Annotated[VectorStore, Depends(get_vector_store)] = None,
    archive_index: Annotated[ArchiveIndex, Depends(get_archive_index)] = None,
):
    """Search documents with semantic search.

    Args:
        query: Search query
        person: Filter by person
        category: Filter by category
        year: Filter by year
        limit: Maximum results
        current_user: Current authenticated user
        vector_store: Vector store (injected)
        archive_index: Archive index (injected)

    Returns:
        Search results filtered by user permissions
    """
    try:
        # Create search tool
        search_tool = SearchArchivedDocumentsTool(
            vector_store=vector_store,
            archive_index=archive_index,
        )

        # Build filters based on user permissions
        additional_filters = {}
        if person:
            additional_filters["person"] = person
        if category:
            additional_filters["category"] = category

        # Create search params
        # Note: The tool will apply ChromaDB filters, then we'll post-filter results
        params = SearchArchivedDocumentsParams(
            query=query,
            person=person if person and (current_user.permissions.allowed_persons is None or person in current_user.permissions.allowed_persons) else None,
            category=category if category and (current_user.permissions.allowed_categories is None or category in current_user.permissions.allowed_categories) else None,
            year=year,
            tags=[],
            max_results=limit,
        )

        # Execute search
        result = await search_tool.execute(params)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        # Post-filter results by user permissions and add document IDs
        filtered_results = []
        for doc in result.data.get("results", []):
            doc_person = doc.get("person")
            doc_category = doc.get("category")

            if check_document_access(current_user, doc_person, doc_category):
                # Look up document ID from archive index using archived_path
                archived_path = doc.get("archived_path")
                if archived_path:
                    # Get document from archive to retrieve its ID
                    archive_doc = await archive_index.get_document_by_path(archived_path)
                    if archive_doc:
                        doc["id"] = archive_doc.id

                filtered_results.append(doc)

        return {
            "query": query,
            "count": len(filtered_results),
            "results": filtered_results,
        }

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_documents(
    person: str | None = Query(None, description="Filter by person"),
    category: str | None = Query(None, description="Filter by category"),
    year: int | None = Query(None, description="Filter by year"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
    archive_index: Annotated[ArchiveIndex, Depends(get_archive_index)] = None,
):
    """List documents with metadata filters.

    Args:
        person: Filter by person
        category: Filter by category
        year: Filter by year
        limit: Maximum results
        offset: Pagination offset
        current_user: Current authenticated user
        archive_index: Archive index (injected)

    Returns:
        List of documents filtered by user permissions
    """
    try:
        # Verify user has access to requested filters
        if person and current_user.permissions.allowed_persons is not None:
            if person not in current_user.permissions.allowed_persons:
                return {"count": 0, "results": []}

        if category and current_user.permissions.allowed_categories is not None:
            if category not in current_user.permissions.allowed_categories:
                return {"count": 0, "results": []}

        # Query archive index
        documents = await archive_index.search(
            person=person,
            category=category,
            year=year,
            limit=limit,
        )

        # Post-filter by user permissions
        filtered_docs = []
        for doc in documents:
            if check_document_access(current_user, doc.person, doc.category):
                filtered_docs.append(
                    {
                        "id": doc.id,
                        "archived_path": doc.archived_path,
                        "person": doc.person,
                        "category": doc.category,
                        "document_date": doc.document_date,
                        "sender": doc.sender,
                        "summary": doc.summary,
                        "tags": doc.tags,
                        "ocr_confidence": doc.ocr_confidence,
                        "processed_at": doc.processed_at,
                        "extraction_model": doc.extraction_model,
                        "last_edited_at": doc.last_edited_at,
                        "last_edited_by": doc.last_edited_by,
                    }
                )

        return {"count": len(filtered_docs), "results": filtered_docs}

    except Exception as e:
        logger.error(f"List documents failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
    archive_index: Annotated[ArchiveIndex, Depends(get_archive_index)] = None,
):
    """Get document details.

    Args:
        document_id: Document ID
        current_user: Current authenticated user
        archive_index: Archive index (injected)

    Returns:
        Document details

    Raises:
        HTTPException: If document not found or access denied
    """
    try:
        # Get document from archive
        document = await archive_index.get_document(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Check access
        if not check_document_access(current_user, document.person, document.category):
            raise HTTPException(status_code=403, detail="Access denied")

        return {
            "id": document.id,
            "archived_path": document.archived_path,
            "original_filename": document.original_filename,
            "person": document.person,
            "category": document.category,
            "document_date": document.document_date,
            "sender": document.sender,
            "summary": document.summary,
            "tags": document.tags,
            "ocr_confidence": document.ocr_confidence,
            "processed_at": document.processed_at,
            "file_size_bytes": document.file_size_bytes,
            "extraction_model": document.extraction_model,
            "last_edited_at": document.last_edited_at,
            "last_edited_by": document.last_edited_by,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/pdf")
async def download_pdf(
    document_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
    archive_index: Annotated[ArchiveIndex, Depends(get_archive_index)] = None,
):
    """Download or view PDF file.

    Args:
        document_id: Document ID
        current_user: Current authenticated user
        archive_index: Archive index (injected)

    Returns:
        PDF file

    Raises:
        HTTPException: If document not found, access denied, or file not found
    """
    try:
        # Check download permission
        if not current_user.permissions.can_download:
            raise HTTPException(status_code=403, detail="Download not permitted")

        # Get document from archive
        document = await archive_index.get_document(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Check access
        if not check_document_access(current_user, document.person, document.category):
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if file exists
        file_path = Path(document.archived_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found")

        logger.info(
            f"User {current_user.username} downloading document: {document.id}"
        )

        # Return file
        return FileResponse(
            path=str(file_path),
            media_type="application/pdf",
            filename=file_path.name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download PDF failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/filters/persons")
async def get_persons(
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
):
    """Get list of persons user can access.

    Args:
        current_user: Current authenticated user

    Returns:
        List of person names
    """
    settings = get_settings()

    # If user has access to all, return all family members (lowercase to match DB)
    if current_user.permissions.allowed_persons is None:
        return {"persons": [name.lower() for name in settings.family_members]}

    # Return user's allowed persons (lowercase to match DB)
    return {"persons": [name.lower() for name in current_user.permissions.allowed_persons]}


@router.get("/filters/categories")
async def get_categories(
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
):
    """Get list of categories user can access.

    Args:
        current_user: Current authenticated user

    Returns:
        List of category names
    """
    settings = get_settings()

    # If user has access to all, return all categories
    if current_user.permissions.allowed_categories is None:
        return {"categories": settings.document_categories}

    # Return user's allowed categories
    return {"categories": current_user.permissions.allowed_categories}


@router.get("/filters/years")
async def get_years(archive_index: Annotated[ArchiveIndex, Depends(get_archive_index)] = None):
    """Get list of available years.

    Args:
        archive_index: Archive index (injected)

    Returns:
        List of years
    """
    try:
        stats = await archive_index.get_statistics()
        years = list(stats.documents_by_year.keys())
        years.sort(reverse=True)  # Most recent first

        return {"years": years}

    except Exception as e:
        logger.error(f"Get years failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/filters/tags")
async def get_tags(
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
    archive_index: Annotated[ArchiveIndex, Depends(get_archive_index)] = None,
):
    """Get list of all unique tags.

    Args:
        current_user: Current authenticated user
        archive_index: Archive index (injected)

    Returns:
        List of unique tags with permission info
    """
    try:
        tags = await archive_index.get_all_tags()
        tags.sort()  # Alphabetical order

        return {
            "tags": tags,
            "can_create_tags": current_user.permissions.can_create_tags,
        }

    except Exception as e:
        logger.error(f"Get tags failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/thumbnail")
async def get_thumbnail(
    document_id: str,
    size: int = Query(default=200, ge=50, le=800, description="Thumbnail size"),
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
    archive_index: Annotated[ArchiveIndex, Depends(get_archive_index)] = None,
):
    """Generate and return PDF thumbnail (first page).

    Args:
        document_id: Document ID
        size: Thumbnail width in pixels (height scaled proportionally)
        current_user: Current authenticated user
        archive_index: Archive index (injected)

    Returns:
        JPEG image of first page
    """
    try:
        # Get document from archive
        document = await archive_index.get_document(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Check access
        if not check_document_access(current_user, document.person, document.category):
            raise HTTPException(status_code=403, detail="Access denied")

        # Check if file exists
        file_path = Path(document.archived_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found")

        # Generate thumbnail (first page only)
        images = convert_from_path(
            str(file_path),
            first_page=1,
            last_page=1,
            size=(size, None),  # Width, height=None for proportional scaling
        )

        if not images:
            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")

        # Convert to JPEG and return
        img_io = io.BytesIO()
        images[0].save(img_io, format="JPEG", quality=85)
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Thumbnail generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{document_id}")
async def update_document(
    document_id: str,
    update_data: DocumentUpdateRequest,
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
    archive_index: Annotated[ArchiveIndex, Depends(get_archive_index)] = None,
    vector_store: Annotated[VectorStore, Depends(get_vector_store)] = None,
):
    """Update document metadata.

    Args:
        document_id: Document ID
        update_data: Fields to update
        current_user: Current authenticated user
        archive_index: Archive index (injected)
        vector_store: Vector store (injected)

    Returns:
        Updated document
    """
    try:
        # Get document from archive
        document = await archive_index.get_document(document_id)

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Check access
        if not check_document_access(current_user, document.person, document.category):
            raise HTTPException(status_code=403, detail="Access denied")

        # Check tag creation permission if new tags are being added
        if update_data.tags is not None and not current_user.permissions.can_create_tags:
            # Get existing tags to check if user is adding new ones
            all_existing_tags = await archive_index.get_all_tags()
            new_tags = [tag for tag in update_data.tags if tag not in all_existing_tags]

            if new_tags:
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: Cannot create new tags. New tags: {', '.join(new_tags)}"
                )

        # Build update dict
        updates = {}
        if update_data.person is not None:
            updates["person"] = update_data.person.lower()
        if update_data.category is not None:
            updates["category"] = update_data.category
        if update_data.summary is not None:
            updates["summary"] = update_data.summary
        if update_data.tags is not None:
            updates["tags"] = update_data.tags
        if update_data.sender is not None:
            updates["sender"] = update_data.sender

        if not updates:
            return {"message": "No updates provided"}

        # Add edit tracking
        from datetime import datetime
        updates["last_edited_at"] = datetime.utcnow().isoformat()
        updates["last_edited_by"] = current_user.username

        # Update in archive index
        await archive_index.update_document(document_id, updates)

        # Update in vector store metadata
        if document.chroma_doc_ids:
            for chroma_id in document.chroma_doc_ids:
                await vector_store.update_metadata(chroma_id, updates)

        # Get updated document
        updated_doc = await archive_index.get_document(document_id)

        logger.info(
            f"User {current_user.username} updated document {document_id}: {list(updates.keys())}"
        )

        return {
            "message": "Document updated successfully",
            "document": {
                "id": updated_doc.id,
                "archived_path": updated_doc.archived_path,
                "person": updated_doc.person,
                "category": updated_doc.category,
                "summary": updated_doc.summary,
                "tags": updated_doc.tags,
                "sender": updated_doc.sender,
                "document_date": updated_doc.document_date,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update document failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
