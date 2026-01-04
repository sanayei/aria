"""Deduplication tool for checking and moving already-processed documents.

This module provides functionality to check if files in the source directory
have already been processed and exist in the archive. Duplicates are moved
to the processed_originals_directory to avoid re-processing.
"""

import asyncio
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from aria.config import Settings
from aria.logging import get_logger
from aria.memory.archive import ArchiveIndex
from aria.tools import BaseTool, RiskLevel, ToolResult

logger = get_logger("aria.tools.scanner.deduplicate")


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read

    Returns:
        Hexadecimal SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


class DeduplicateParams(BaseModel):
    """Parameters for deduplication."""

    source_directory: str | None = Field(
        default=None,
        description="Directory to check for duplicates (defaults to scan_directory from settings)",
    )
    file_pattern: str = Field(
        default="*.pdf",
        description="Glob pattern for files to check (default: *.pdf)",
    )
    preview_only: bool = Field(
        default=True,
        description="If True, show what would happen without actually moving files",
    )
    check_by: str = Field(
        default="filename",
        description="Deduplication method: 'filename' or 'hash' (filename is faster)",
    )


class DeduplicateTool(BaseTool[DeduplicateParams]):
    """Check source directory for already-processed files and delete duplicates.

    This tool helps prevent re-processing of documents by:
    1. Scanning source directory for files
    2. Checking if they already exist in the archive (by filename or hash)
    3. Deleting duplicates from source directory (since they're already safely archived)

    Risk Level: MEDIUM - Deletes duplicate files from source directory
    """

    name = "deduplicate"
    description = (
        "Check source directory for files that have already been processed "
        "and delete duplicates to avoid re-processing. "
        "Can check by filename (fast) or file hash (thorough)."
    )
    risk_level = RiskLevel.MEDIUM
    parameters_schema = DeduplicateParams

    def __init__(self, archive_index: ArchiveIndex, settings: Settings):
        """Initialize the deduplicate tool.

        Args:
            archive_index: Archive index for checking processed files
            settings: ARIA settings
        """
        super().__init__()
        self.archive_index = archive_index
        self.settings = settings

    def get_confirmation_message(self, params: DeduplicateParams) -> str:
        """Get confirmation message."""
        source = params.source_directory or self.settings.scan_directory
        if params.preview_only:
            return f"Preview duplicates in: {source}"
        else:
            return f"Delete duplicate {params.file_pattern} files from {source}"

    async def execute(self, params: DeduplicateParams) -> ToolResult:
        """Execute deduplication."""
        try:
            # Determine source directory
            source_dir = (
                Path(params.source_directory or self.settings.scan_directory)
                .expanduser()
                .resolve()
            )

            if not source_dir.exists():
                return ToolResult.error_result(f"Source directory does not exist: {source_dir}")

            if not source_dir.is_dir():
                return ToolResult.error_result(f"Source path is not a directory: {source_dir}")

            # Find files matching pattern
            files = list(source_dir.glob(params.file_pattern))

            # Filter out hidden macOS metadata files (._*)
            files = [f for f in files if not f.name.startswith("._")]

            if not files:
                return ToolResult.success_result(
                    data={
                        "source_dir": str(source_dir),
                        "pattern": params.file_pattern,
                        "found_count": 0,
                        "duplicate_count": 0,
                        "moved_count": 0,
                        "message": f"No files matching '{params.file_pattern}' found",
                    }
                )

            logger.info(
                f"Found {len(files)} files in {source_dir} (pattern: {params.file_pattern})"
            )

            # Get all archived documents for comparison
            all_archived_docs = await self.archive_index.search(limit=100000)

            # Build lookup sets for fast checking
            if params.check_by == "filename":
                # Check by original filename
                archived_filenames = {doc.original_filename for doc in all_archived_docs}
                logger.info(f"Checking against {len(archived_filenames)} archived filenames")
            else:
                # Check by file hash (need to compute hashes for archived files)
                # Note: This is slower and requires archived files to still exist
                archived_hashes = {}
                for doc in all_archived_docs:
                    archived_path = Path(doc.archived_path)
                    if archived_path.exists():
                        try:
                            file_hash = await asyncio.to_thread(
                                compute_file_hash, archived_path
                            )
                            archived_hashes[file_hash] = doc
                        except Exception as e:
                            logger.warning(
                                f"Failed to compute hash for {archived_path}: {e}"
                            )
                logger.info(f"Checking against {len(archived_hashes)} archived file hashes")

            # Check each file for duplicates
            duplicates: list[dict] = []
            for file_path in files:
                is_duplicate = False
                matched_doc = None

                if params.check_by == "filename":
                    # Simple filename matching
                    if file_path.name in archived_filenames:
                        is_duplicate = True
                        # Find the document for metadata
                        for doc in all_archived_docs:
                            if doc.original_filename == file_path.name:
                                matched_doc = doc
                                break
                else:
                    # Hash-based matching
                    try:
                        file_hash = await asyncio.to_thread(compute_file_hash, file_path)
                        if file_hash in archived_hashes:
                            is_duplicate = True
                            matched_doc = archived_hashes[file_hash]
                    except Exception as e:
                        logger.error(f"Failed to compute hash for {file_path}: {e}")
                        continue

                if is_duplicate:
                    duplicate_info = {
                        "source_file": str(file_path),
                        "filename": file_path.name,
                        "size_bytes": file_path.stat().st_size,
                    }

                    if matched_doc:
                        duplicate_info.update({
                            "archived_path": matched_doc.archived_path,
                            "person": matched_doc.person,
                            "category": matched_doc.category,
                            "processed_at": matched_doc.processed_at,
                        })

                    duplicates.append(duplicate_info)

            logger.info(
                f"Found {len(duplicates)} duplicates out of {len(files)} files "
                f"(method: {params.check_by})"
            )

            # If preview only, return without moving
            if params.preview_only:
                return ToolResult.success_result(
                    data={
                        "source_dir": str(source_dir),
                        "pattern": params.file_pattern,
                        "check_by": params.check_by,
                        "found_count": len(files),
                        "duplicate_count": len(duplicates),
                        "moved_count": 0,
                        "preview_only": True,
                        "duplicates": duplicates,
                    }
                )

            # NOT preview mode - actually delete duplicates (they're already in archive)
            deleted_count = 0
            failed_deletes: list[dict] = []

            for dup in duplicates:
                source_file = Path(dup["source_file"])

                try:
                    # Simply delete the duplicate file (it's already archived)
                    source_file.unlink()

                    logger.info(f"Deleted duplicate: {source_file.name}")
                    deleted_count += 1

                    # Mark as deleted
                    dup["deleted"] = True

                except Exception as e:
                    logger.error(f"Failed to delete {source_file}: {e}")
                    failed_deletes.append({
                        "file": str(source_file),
                        "error": str(e),
                    })
                    dup["deleted"] = False

            return ToolResult.success_result(
                data={
                    "source_dir": str(source_dir),
                    "pattern": params.file_pattern,
                    "check_by": params.check_by,
                    "found_count": len(files),
                    "duplicate_count": len(duplicates),
                    "deleted_count": deleted_count,
                    "failed_count": len(failed_deletes),
                    "preview_only": False,
                    "duplicates": duplicates,
                    "failures": failed_deletes,
                }
            )

        except Exception as e:
            logger.error(f"Deduplication failed: {e}", exc_info=True)
            return ToolResult.error_result(f"Operation failed: {str(e)}")
