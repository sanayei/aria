"""Document scanning and indexing tools."""

from aria.tools.scanner.deduplicate import DeduplicateParams, DeduplicateTool
from aria.tools.scanner.scan_and_index import ScanAndIndexParams, ScanAndIndexTool
from aria.tools.scanner.search_archive import (
    ArchiveStatisticsParams,
    ArchiveStatisticsTool,
    GetArchivedDocumentParams,
    GetArchivedDocumentTool,
    ListArchivedDocumentsParams,
    ListArchivedDocumentsTool,
    SearchArchivedDocumentsParams,
    SearchArchivedDocumentsTool,
)

__all__ = [
    # Scanning and indexing
    "ScanAndIndexTool",
    "ScanAndIndexParams",
    # Deduplication
    "DeduplicateTool",
    "DeduplicateParams",
    # Archive search
    "SearchArchivedDocumentsTool",
    "SearchArchivedDocumentsParams",
    "ListArchivedDocumentsTool",
    "ListArchivedDocumentsParams",
    "GetArchivedDocumentTool",
    "GetArchivedDocumentParams",
    "ArchiveStatisticsTool",
    "ArchiveStatisticsParams",
]
