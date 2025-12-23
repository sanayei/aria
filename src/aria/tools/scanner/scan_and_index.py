"""Integrated document scanning and indexing tool.

This module provides a unified tool that scans a directory for documents,
processes them with OCR and classification, organizes them into an archive,
and indexes them for semantic search.
"""

import asyncio
import shutil
import signal
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from aria.config import Settings
from aria.logging import get_logger
from aria.memory.archive import ArchiveDocument, ArchiveIndex
from aria.memory.ingestion import DocumentIngester
from aria.tools import BaseTool, RiskLevel, ToolResult
from aria.tools.documents.models import ProcessedDocument
from aria.tools.documents.processor import DocumentProcessor

logger = get_logger("aria.tools.scanner")


class ScanAndIndexParams(BaseModel):
    """Parameters for scanning and indexing documents."""

    source_directory: str | None = Field(
        default=None,
        description="Directory to scan (defaults to scan_directory from settings)",
    )
    file_pattern: str = Field(
        default="*.pdf",
        description="Glob pattern for files to process (default: *.pdf)",
    )
    preview_only: bool = Field(
        default=True,
        description="If True, show what would happen without actually processing",
    )
    move_originals: bool = Field(
        default=True,
        description="Move original files to processed directory after successful indexing",
    )
    max_files: int | None = Field(
        default=None,
        description="Maximum number of files to process (for testing)",
    )


class ScanAndIndexTool(BaseTool[ScanAndIndexParams]):
    """Scan directory, process documents, and index them for semantic search.

    This tool provides the complete document processing pipeline:
    1. Scans source directory for PDF files
    2. Processes each document:
       - Runs OCR to extract text
       - Uses LLM to classify (person, category, date, sender, summary, tags)
       - Generates intelligent filename
    3. Organizes the file:
       - Copies to archive directory with structured path
    4. Indexes for search:
       - Ingests into ChromaDB with metadata
       - Adds to archive index database
    5. Moves original:
       - Moves original file to processed directory (after confirmation)

    Risk Level: MEDIUM - Copies files to archive, modifies ChromaDB, moves originals
    """

    name = "scan_and_index"
    description = (
        "Scan a directory for documents, process them with OCR and classification, "
        "organize into archive, and index for semantic search. "
        "Complete end-to-end document processing pipeline."
    )
    risk_level = RiskLevel.MEDIUM
    parameters_schema = ScanAndIndexParams

    def __init__(
        self,
        processor: DocumentProcessor,
        ingester: DocumentIngester,
        archive_index: ArchiveIndex,
        settings: Settings,
    ):
        """Initialize the scan and index tool.

        Args:
            processor: Document processor for OCR and classification
            ingester: Document ingester for ChromaDB
            archive_index: Archive index for tracking
            settings: ARIA settings
        """
        super().__init__()
        self.processor = processor
        self.ingester = ingester
        self.archive_index = archive_index
        self.settings = settings
        self._shutdown_event = asyncio.Event()
        self._current_task: asyncio.Task | None = None

    def get_confirmation_message(self, params: ScanAndIndexParams) -> str:
        """Get confirmation message."""
        source = params.source_directory or self.settings.scan_directory
        if params.preview_only:
            return f"Preview documents in: {source}"
        else:
            action = "process, organize, index"
            if params.move_originals:
                action += ", and move originals"
            return f"Scan {source} - {action} all {params.file_pattern} files"

    async def execute(self, params: ScanAndIndexParams) -> ToolResult:
        """Execute the scan and index operation."""
        try:
            # Determine source directory
            source_dir = (
                Path(params.source_directory or self.settings.scan_directory).expanduser().resolve()
            )

            if not source_dir.exists():
                return ToolResult.error_result(f"Source directory does not exist: {source_dir}")

            if not source_dir.is_dir():
                return ToolResult.error_result(f"Source path is not a directory: {source_dir}")

            # Find files matching pattern
            pdf_files = list(source_dir.glob(params.file_pattern))

            # Filter out hidden macOS metadata files (._*)
            pdf_files = [f for f in pdf_files if not f.name.startswith("._")]

            if params.max_files:
                pdf_files = pdf_files[: params.max_files]

            if not pdf_files:
                return ToolResult.success_result(
                    data={
                        "source_dir": str(source_dir),
                        "pattern": params.file_pattern,
                        "found_count": 0,
                        "processed_count": 0,
                        "message": f"No files matching '{params.file_pattern}' found",
                    }
                )

            logger.info(
                f"Found {len(pdf_files)} files in {source_dir} (pattern: {params.file_pattern})"
            )

            # Process documents (parallel or sequential based on settings)
            processed_docs: list[ProcessedDocument] = []
            failed: list[dict] = []

            if self.settings.enable_parallel_processing and len(pdf_files) > 1:
                # Parallel processing mode
                processed_docs, failed = await self._process_documents_parallel(pdf_files)
            else:
                # Sequential processing mode
                for idx, pdf_file in enumerate(pdf_files, 1):
                    # Check for shutdown/cancellation
                    if self._shutdown_event.is_set():
                        logger.info(
                            "Shutdown requested, stopping processing",
                            processed=idx - 1,
                            remaining=len(pdf_files) - idx + 1,
                        )
                        break

                    try:
                        logger.info(f"Processing {pdf_file.name} ({idx}/{len(pdf_files)})...")

                        # Create cancellable task
                        self._current_task = asyncio.create_task(
                            self.processor.process_document(pdf_file)
                        )

                        # Wait for completion or cancellation
                        doc = await self._current_task
                        processed_docs.append(doc)
                        logger.info(
                            f"Processed {pdf_file.name}: "
                            f"{doc.person}/{doc.category} - {doc.summary[:50]}..."
                        )

                    except asyncio.CancelledError:
                        logger.info(f"Processing cancelled for {pdf_file.name}")
                        break

                    except Exception as e:
                        logger.error(f"Failed to process {pdf_file.name}: {e}")
                        failed.append({"file": str(pdf_file), "error": str(e)})

            # If preview only, return summary without modifying anything
            if params.preview_only:
                person_counts: dict[str, int] = defaultdict(int)
                category_counts: dict[str, int] = defaultdict(int)

                for doc in processed_docs:
                    person_counts[doc.person] += 1
                    category_counts[doc.category] += 1

                doc_list = [
                    {
                        "source": str(doc.source_path),
                        "person": doc.person,
                        "category": doc.category,
                        "date": doc.document_date.isoformat() if doc.document_date else None,
                        "sender": doc.sender,
                        "summary": doc.summary,
                        "tags": doc.tags,
                        "destination": str(doc.suggested_destination),
                        "confidence": doc.confidence,
                    }
                    for doc in processed_docs
                ]

                return ToolResult.success_result(
                    data={
                        "source_dir": str(source_dir),
                        "found_count": len(pdf_files),
                        "processed_count": len(processed_docs),
                        "failed_count": len(failed),
                        "preview_only": True,
                        "person_counts": dict(person_counts),
                        "category_counts": dict(category_counts),
                        "documents": doc_list,
                        "failures": failed,
                    }
                )

            # NOT preview mode - actually process files
            organized_count = 0
            indexed_count = 0
            moved_count = 0

            for doc in processed_docs:
                try:
                    # Step 1: Copy to archive directory
                    archive_path = self._get_archive_path(doc)
                    archive_path.parent.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(str(doc.source_path), str(archive_path))
                    logger.info(f"Copied to archive: {archive_path}")
                    organized_count += 1

                    # Step 2: Ingest into ChromaDB
                    # Build metadata, filtering out None values (ChromaDB doesn't support them)
                    metadata = {
                        "original_filename": doc.source_path.name,
                        "original_path": str(doc.source_path),
                        "archived_path": str(archive_path),
                        "person": doc.person,
                        "category": doc.category,
                        "summary": doc.summary,
                        "tags": ",".join(doc.tags),
                        "processed_at": datetime.now().isoformat(),
                        "file_size_bytes": doc.source_path.stat().st_size,
                    }

                    # Add optional fields only if they have values
                    if doc.document_date:
                        metadata["document_date"] = doc.document_date.isoformat()
                    if doc.sender:
                        metadata["sender"] = doc.sender

                    # Use ingest_text with pre-extracted OCR text instead of re-extracting
                    ingest_result = await self.ingester.ingest_text(
                        text=doc.text_content,
                        file_path=archive_path,
                        metadata=metadata,
                        tags=doc.tags,
                    )

                    if ingest_result.success:
                        logger.info(f"Ingested to ChromaDB: {ingest_result.chunks_created} chunks")
                        indexed_count += 1

                        # Step 3: Add to archive index
                        doc_id = str(uuid.uuid4())
                        # Get the ChromaDB document IDs for tracking
                        # (they're created by ingester with specific pattern)
                        chroma_doc_ids = [
                            f"doc_{str(archive_path).replace('/', '_').replace('\\', '_')}_{i}"
                            for i in range(ingest_result.chunks_created)
                        ]

                        archive_doc = ArchiveDocument(
                            id=doc_id,
                            original_filename=doc.source_path.name,
                            original_path=str(doc.source_path),
                            archived_path=str(archive_path),
                            person=doc.person,
                            category=doc.category,
                            document_date=doc.document_date.isoformat()
                            if doc.document_date
                            else None,
                            sender=doc.sender,
                            summary=doc.summary,
                            tags=doc.tags,
                            ocr_confidence=doc.confidence,
                            processed_at=datetime.now().isoformat(),
                            file_size_bytes=doc.source_path.stat().st_size,
                            chroma_doc_ids=chroma_doc_ids,
                        )

                        await self.archive_index.add_document(archive_doc)
                        logger.info(f"Added to archive index: {doc_id}")

                        # Step 4: Move original to processed directory (if requested)
                        if params.move_originals:
                            processed_dir = self.settings.processed_originals_directory
                            processed_dir.mkdir(parents=True, exist_ok=True)

                            processed_path = processed_dir / doc.source_path.name

                            # Handle filename collisions
                            if processed_path.exists():
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                processed_path = (
                                    processed_dir / f"{timestamp}_{doc.source_path.name}"
                                )

                            shutil.move(str(doc.source_path), str(processed_path))
                            logger.info(f"Moved original to: {processed_path}")
                            moved_count += 1

                    else:
                        logger.error(f"Failed to ingest: {ingest_result.error}")
                        failed.append(
                            {
                                "file": str(doc.source_path),
                                "error": f"Ingestion failed: {ingest_result.error}",
                            }
                        )

                except Exception as e:
                    logger.error(f"Failed to index {doc.source_path.name}: {e}")
                    failed.append(
                        {
                            "file": str(doc.source_path),
                            "error": f"Indexing failed: {str(e)}",
                        }
                    )

            # Generate summary statistics
            person_counts: dict[str, int] = defaultdict(int)
            category_counts: dict[str, int] = defaultdict(int)

            for doc in processed_docs:
                person_counts[doc.person] += 1
                category_counts[doc.category] += 1

            doc_list = [
                {
                    "source": str(doc.source_path),
                    "person": doc.person,
                    "category": doc.category,
                    "date": doc.document_date.isoformat() if doc.document_date else None,
                    "sender": doc.sender,
                    "summary": doc.summary,
                    "tags": doc.tags,
                    "destination": str(self._get_archive_path(doc)),  # Use helper method
                    "confidence": doc.confidence,
                }
                for doc in processed_docs
            ]

            return ToolResult.success_result(
                data={
                    "source_dir": str(source_dir),
                    "found_count": len(pdf_files),
                    "processed_count": len(processed_docs),
                    "organized_count": organized_count,
                    "indexed_count": indexed_count,
                    "moved_count": moved_count,
                    "failed_count": len(failed),
                    "preview_only": False,
                    "person_counts": dict(person_counts),
                    "category_counts": dict(category_counts),
                    "documents": doc_list,
                    "failures": failed,
                }
            )

        except Exception as e:
            logger.error(f"Scan and index failed: {e}", exc_info=True)
            return ToolResult.error_result(f"Operation failed: {str(e)}")

    def _get_archive_path(self, doc: ProcessedDocument) -> Path:
        """Get the archive path for a processed document.

        Creates path: archive_directory/{Year}/{Person}/{Category}/{filename}

        Args:
            doc: Processed document

        Returns:
            Full archive path
        """
        # Get year from document date or use current year
        year = str(doc.document_date.year) if doc.document_date else str(datetime.now().year)

        # Build path: Archive/{Year}/{Person}/{Category}/{filename}
        archive_path = (
            self.settings.archive_directory
            / year
            / doc.person
            / doc.category
            / doc.suggested_filename
        )

        return archive_path

    async def _process_documents_parallel(
        self, pdf_files: list[Path]
    ) -> tuple[list[ProcessedDocument], list[dict]]:
        """Process multiple documents in parallel batches.

        Strategy:
        - Process documents in batches for better throughput
        - LLM classification is sequential (single GPU), but OCR can be parallel
        - Use semaphore to limit concurrent OCR operations

        Args:
            pdf_files: List of PDF files to process

        Returns:
            Tuple of (processed_docs, failed_docs)
        """
        processed_docs: list[ProcessedDocument] = []
        failed: list[dict] = []

        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.settings.max_parallel_ocr)

        async def process_with_semaphore(
            pdf_file: Path, idx: int
        ) -> tuple[ProcessedDocument | None, dict | None]:
            """Process a single document with semaphore control."""
            async with semaphore:
                # Check for shutdown
                if self._shutdown_event.is_set():
                    return None, None

                try:
                    logger.info(f"Processing {pdf_file.name} ({idx + 1}/{len(pdf_files)})...")

                    # Process document
                    doc = await self.processor.process_document(pdf_file)

                    logger.info(
                        f"Processed {pdf_file.name}: "
                        f"{doc.person}/{doc.category} - {doc.summary[:50]}..."
                    )

                    return doc, None

                except Exception as e:
                    logger.error(f"Failed to process {pdf_file.name}: {e}")
                    return None, {"file": str(pdf_file), "error": str(e)}

        # Create tasks for all documents
        tasks = [
            asyncio.create_task(process_with_semaphore(pdf_file, idx))
            for idx, pdf_file in enumerate(pdf_files)
        ]

        # Store tasks for cancellation
        self._current_task = None  # Track batch instead

        try:
            # Process all tasks concurrently (with semaphore limiting parallelism)
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Separate successful and failed results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
                    continue

                doc, error = result

                if doc is not None:
                    processed_docs.append(doc)
                elif error is not None:
                    failed.append(error)

        except asyncio.CancelledError:
            logger.info("Parallel processing cancelled, cancelling all tasks...")
            # Cancel all pending tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for all to finish
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        logger.info(
            f"Parallel processing complete: processed={len(processed_docs)}, failed={len(failed)}"
        )

        return processed_docs, failed
