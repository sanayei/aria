"""Process inbox tool for batch processing scanned documents."""

import shutil
from pathlib import Path
from collections import defaultdict

from pydantic import BaseModel, Field

from aria.config import Settings
from aria.tools import BaseTool, ToolResult, RiskLevel
from aria.tools.documents.processor import DocumentProcessor
from aria.tools.documents.models import ProcessedDocument


class ProcessInboxParams(BaseModel):
    """Parameters for processing inbox of documents."""

    preview_only: bool = Field(
        default=True,
        description="If True, only show what would happen without moving files",
    )


class ProcessInboxTool(BaseTool[ProcessInboxParams]):
    """Process all PDF files in the scanner inbox.

    This tool:
    1. Scans the documents_source_dir for *.pdf files
    2. Processes each document (OCR + classification)
    3. If preview_only=True, shows summary of what would happen
    4. If preview_only=False, processes and moves all files

    Returns a summary with counts per person/category.
    """

    name = "process_inbox"
    description = "Process and organize all documents in the scanner inbox"
    risk_level = RiskLevel.MEDIUM
    parameters_schema = ProcessInboxParams

    def __init__(self, processor: DocumentProcessor, settings: Settings):
        """Initialize with document processor and settings.

        Args:
            processor: DocumentProcessor instance
            settings: ARIA settings (for source directory)
        """
        super().__init__()
        self.processor = processor
        self.settings = settings

    async def execute(self, params: ProcessInboxParams) -> ToolResult:
        """Process inbox of documents."""
        try:
            source_dir = self.settings.documents_source_dir

            if not source_dir.exists():
                return ToolResult.error_result(
                    f"Source directory does not exist: {source_dir}"
                )

            if not source_dir.is_dir():
                return ToolResult.error_result(
                    f"Source path is not a directory: {source_dir}"
                )

            # Find all PDF files
            pdf_files = list(source_dir.glob("*.pdf"))

            if not pdf_files:
                return ToolResult.success_result(
                    data={
                        "source_dir": str(source_dir),
                        "found_count": 0,
                        "processed_count": 0,
                        "message": "No PDF files found in inbox",
                    }
                )

            # Process each document
            processed_docs: list[ProcessedDocument] = []
            failed: list[dict] = []

            for pdf_file in pdf_files:
                try:
                    doc = await self.processor.process_document(pdf_file)
                    processed_docs.append(doc)
                except Exception as e:
                    failed.append(
                        {
                            "file": str(pdf_file),
                            "error": str(e),
                        }
                    )

            # If not preview_only, move the files
            organized_count = 0
            if not params.preview_only:
                for doc in processed_docs:
                    try:
                        # Create destination directory
                        doc.suggested_destination.parent.mkdir(
                            parents=True, exist_ok=True
                        )

                        # Move the file
                        shutil.move(
                            str(doc.source_path),
                            str(doc.suggested_destination),
                        )
                        organized_count += 1

                    except Exception as e:
                        failed.append(
                            {
                                "file": str(doc.source_path),
                                "error": f"Move failed: {e}",
                            }
                        )

            # Generate summary statistics
            person_counts: dict[str, int] = defaultdict(int)
            category_counts: dict[str, int] = defaultdict(int)

            for doc in processed_docs:
                person_counts[doc.person] += 1
                category_counts[doc.category] += 1

            # Build document list for response
            doc_list = [
                {
                    "source": str(doc.source_path),
                    "person": doc.person,
                    "category": doc.category,
                    "date": doc.document_date.isoformat() if doc.document_date else None,
                    "sender": doc.sender,
                    "summary": doc.summary,
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
                    "organized_count": organized_count if not params.preview_only else 0,
                    "failed_count": len(failed),
                    "preview_only": params.preview_only,
                    "person_counts": dict(person_counts),
                    "category_counts": dict(category_counts),
                    "documents": doc_list,
                    "failures": failed,
                }
            )

        except Exception as e:
            return ToolResult.error_result(f"Failed to process inbox: {e}")

    def get_confirmation_message(self, params: ProcessInboxParams) -> str:
        """Get confirmation message for user approval."""
        if params.preview_only:
            return f"Preview documents in inbox: {self.settings.documents_source_dir}"
        else:
            return f"Process and organize ALL documents in: {self.settings.documents_source_dir}"
