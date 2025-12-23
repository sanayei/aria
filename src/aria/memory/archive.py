"""Archive index for tracking processed and organized documents.

This module provides SQLite-based tracking of documents that have been processed,
organized, and indexed into the vector store. It enables efficient searching and
retrieval of archived documents by metadata.
"""

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aria.logging import get_logger

logger = get_logger("aria.memory.archive")


class ArchiveDocument(BaseModel):
    """Model representing an archived document."""

    id: str = Field(description="Unique document ID")
    original_filename: str = Field(description="Original filename before processing")
    original_path: str = Field(description="Original file path")
    archived_path: str = Field(description="Path in archive directory")
    person: str = Field(description="Person this document belongs to")
    category: str = Field(description="Document category")
    document_date: str | None = Field(default=None, description="Date on the document (YYYY-MM-DD)")
    sender: str | None = Field(default=None, description="Document sender/organization")
    summary: str = Field(description="Document summary")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    ocr_confidence: float = Field(ge=0.0, le=1.0, description="OCR confidence score")
    processed_at: str = Field(description="Processing timestamp (ISO format)")
    file_size_bytes: int = Field(description="File size in bytes")
    chroma_doc_ids: list[str] = Field(
        default_factory=list, description="ChromaDB document IDs for chunks"
    )


class ArchiveStatistics(BaseModel):
    """Statistics about the archive."""

    total_documents: int
    documents_by_person: dict[str, int]
    documents_by_category: dict[str, int]
    documents_by_year: dict[str, int]
    total_size_bytes: int
    earliest_document: str | None
    latest_document: str | None


class ArchiveIndex:
    """SQLite-based index for archived documents.

    This class manages a SQLite database that tracks all processed and archived
    documents, enabling efficient searching and retrieval by metadata.
    """

    def __init__(self, db_path: Path):
        """Initialize archive index.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = db_path
        self._initialized = False
        logger.info(f"Initialized ArchiveIndex at {db_path}")

    async def initialize(self) -> None:
        """Initialize the database and create tables if needed."""
        if self._initialized:
            return

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create tables
        await asyncio.to_thread(self._create_tables)
        self._initialized = True
        logger.info(f"Archive database initialized at {self._db_path}")

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.cursor()

            # Create archived_documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS archived_documents (
                    id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    archived_path TEXT NOT NULL,
                    person TEXT,
                    category TEXT,
                    document_date TEXT,
                    sender TEXT,
                    summary TEXT,
                    tags TEXT,
                    ocr_confidence REAL,
                    processed_at TEXT NOT NULL,
                    file_size_bytes INTEGER,
                    chroma_doc_ids TEXT
                )
            """)

            # Create indexes for common queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_person ON archived_documents(person)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_category ON archived_documents(category)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_date ON archived_documents(document_date)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_at ON archived_documents(processed_at)"
            )

            conn.commit()
            logger.debug("Archive database tables created successfully")
        finally:
            conn.close()

    async def add_document(self, document: ArchiveDocument) -> None:
        """Add a document to the archive index.

        Args:
            document: Document to add

        Raises:
            RuntimeError: If database operation fails
        """
        if not self._initialized:
            await self.initialize()

        def _insert():
            conn = sqlite3.connect(self._db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO archived_documents
                    (id, original_filename, original_path, archived_path, person,
                     category, document_date, sender, summary, tags,
                     ocr_confidence, processed_at, file_size_bytes, chroma_doc_ids)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.original_filename,
                        document.original_path,
                        document.archived_path,
                        document.person,
                        document.category,
                        document.document_date,
                        document.sender,
                        document.summary,
                        json.dumps(document.tags),
                        document.ocr_confidence,
                        document.processed_at,
                        document.file_size_bytes,
                        json.dumps(document.chroma_doc_ids),
                    ),
                )
                conn.commit()
                logger.debug(f"Added document '{document.id}' to archive index")
            finally:
                conn.close()

        await asyncio.to_thread(_insert)

    async def get_document(self, doc_id: str) -> ArchiveDocument | None:
        """Retrieve a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            ArchiveDocument if found, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        def _query():
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM archived_documents WHERE id = ?", (doc_id,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_document(row)
                return None
            finally:
                conn.close()

        return await asyncio.to_thread(_query)

    async def search(
        self,
        person: str | None = None,
        category: str | None = None,
        year: int | None = None,
        sender: str | None = None,
        tag: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ArchiveDocument]:
        """Search documents by metadata filters.

        Args:
            person: Filter by person
            category: Filter by category
            year: Filter by year (extracted from document_date)
            sender: Filter by sender (partial match)
            tag: Filter by tag (partial match in tags JSON)
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            List of matching documents
        """
        if not self._initialized:
            await self.initialize()

        def _query():
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()

                # Build query dynamically based on filters
                conditions = []
                params = []

                if person:
                    conditions.append("person = ?")
                    params.append(person)

                if category:
                    conditions.append("category = ?")
                    params.append(category)

                if year:
                    conditions.append("document_date LIKE ?")
                    params.append(f"{year}%")

                if sender:
                    conditions.append("sender LIKE ?")
                    params.append(f"%{sender}%")

                if tag:
                    conditions.append("tags LIKE ?")
                    params.append(f"%{tag}%")

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                query = f"""
                    SELECT * FROM archived_documents
                    WHERE {where_clause}
                    ORDER BY document_date DESC, processed_at DESC
                    LIMIT ? OFFSET ?
                """
                params.extend([limit, offset])

                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [self._row_to_document(row) for row in rows]
            finally:
                conn.close()

        return await asyncio.to_thread(_query)

    async def list_by_person(self, person: str) -> list[ArchiveDocument]:
        """List all documents for a specific person.

        Args:
            person: Person name

        Returns:
            List of documents
        """
        return await self.search(person=person)

    async def list_by_category(self, category: str) -> list[ArchiveDocument]:
        """List all documents in a category.

        Args:
            category: Category name

        Returns:
            List of documents
        """
        return await self.search(category=category)

    async def list_by_date_range(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> list[ArchiveDocument]:
        """List documents within a date range.

        Args:
            start_date: Start date (YYYY-MM-DD) inclusive
            end_date: End date (YYYY-MM-DD) inclusive

        Returns:
            List of documents
        """
        if not self._initialized:
            await self.initialize()

        def _query():
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()

                conditions = []
                params = []

                if start_date:
                    conditions.append("document_date >= ?")
                    params.append(start_date)

                if end_date:
                    conditions.append("document_date <= ?")
                    params.append(end_date)

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                query = f"""
                    SELECT * FROM archived_documents
                    WHERE {where_clause}
                    ORDER BY document_date DESC
                """

                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [self._row_to_document(row) for row in rows]
            finally:
                conn.close()

        return await asyncio.to_thread(_query)

    async def get_statistics(self) -> ArchiveStatistics:
        """Get statistics about the archive.

        Returns:
            Archive statistics
        """
        if not self._initialized:
            await self.initialize()

        def _query():
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()

                # Total documents
                cursor.execute("SELECT COUNT(*) as count FROM archived_documents")
                total = cursor.fetchone()["count"]

                # By person
                cursor.execute("""
                    SELECT person, COUNT(*) as count
                    FROM archived_documents
                    GROUP BY person
                """)
                by_person = {row["person"]: row["count"] for row in cursor.fetchall()}

                # By category
                cursor.execute("""
                    SELECT category, COUNT(*) as count
                    FROM archived_documents
                    GROUP BY category
                """)
                by_category = {row["category"]: row["count"] for row in cursor.fetchall()}

                # By year (extract from document_date)
                cursor.execute("""
                    SELECT substr(document_date, 1, 4) as year, COUNT(*) as count
                    FROM archived_documents
                    WHERE document_date IS NOT NULL
                    GROUP BY year
                """)
                by_year = {row["year"]: row["count"] for row in cursor.fetchall()}

                # Total size
                cursor.execute("SELECT SUM(file_size_bytes) as total FROM archived_documents")
                total_size = cursor.fetchone()["total"] or 0

                # Earliest and latest documents
                cursor.execute("""
                    SELECT MIN(document_date) as earliest, MAX(document_date) as latest
                    FROM archived_documents
                    WHERE document_date IS NOT NULL
                """)
                row = cursor.fetchone()
                earliest = row["earliest"]
                latest = row["latest"]

                return ArchiveStatistics(
                    total_documents=total,
                    documents_by_person=by_person,
                    documents_by_category=by_category,
                    documents_by_year=by_year,
                    total_size_bytes=total_size,
                    earliest_document=earliest,
                    latest_document=latest,
                )
            finally:
                conn.close()

        return await asyncio.to_thread(_query)

    def _row_to_document(self, row: sqlite3.Row) -> ArchiveDocument:
        """Convert database row to ArchiveDocument.

        Args:
            row: SQLite row

        Returns:
            ArchiveDocument instance
        """
        return ArchiveDocument(
            id=row["id"],
            original_filename=row["original_filename"],
            original_path=row["original_path"],
            archived_path=row["archived_path"],
            person=row["person"],
            category=row["category"],
            document_date=row["document_date"],
            sender=row["sender"],
            summary=row["summary"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            ocr_confidence=row["ocr_confidence"],
            processed_at=row["processed_at"],
            file_size_bytes=row["file_size_bytes"],
            chroma_doc_ids=json.loads(row["chroma_doc_ids"]) if row["chroma_doc_ids"] else [],
        )
