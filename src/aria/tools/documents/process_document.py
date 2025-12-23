"""Process document tool for scanning and organizing individual documents."""

import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from aria.tools import BaseTool, ToolResult, RiskLevel
from aria.tools.documents.processor import DocumentProcessor


class ProcessDocumentParams(BaseModel):
    """Parameters for processing a single document."""

    source_path: str = Field(description="Path to the PDF or image file")
    auto_organize: bool = Field(
        default=False,
        description="If True, automatically move the file to the destination",
    )


class ProcessDocumentTool(BaseTool[ProcessDocumentParams]):
    """Process a single scanned document.

    This tool:
    1. Runs OCR to extract text
    2. Uses LLM to classify the document
    3. Generates a suggested filename and destination
    4. Optionally moves the file if auto_organize=True

    If auto_organize=False, only shows what would happen (preview mode).
    If auto_organize=True, actually moves the file.
    """

    name = "process_document"
    description = "Process and optionally organize a scanned document"
    risk_level = RiskLevel.MEDIUM
    parameters_schema = ProcessDocumentParams

    def __init__(self, processor: DocumentProcessor):
        """Initialize with document processor.

        Args:
            processor: DocumentProcessor instance
        """
        super().__init__()
        self.processor = processor

    async def execute(self, params: ProcessDocumentParams) -> ToolResult:
        """Process a document."""
        try:
            source_path = Path(params.source_path).expanduser().resolve()

            if not source_path.exists():
                return ToolResult.error_result(f"File does not exist: {source_path}")

            if not source_path.is_file():
                return ToolResult.error_result(f"Path is not a file: {source_path}")

            # Process the document (OCR + classification)
            try:
                doc = await self.processor.process_document(source_path)
            except Exception as e:
                return ToolResult.error_result(f"Processing failed: {e}")

            # Build result data
            result_data = {
                "source_path": str(doc.source_path),
                "person": doc.person,
                "category": doc.category,
                "document_date": doc.document_date.isoformat() if doc.document_date else None,
                "sender": doc.sender,
                "summary": doc.summary,
                "tags": doc.tags,
                "confidence": doc.confidence,
                "suggested_filename": doc.suggested_filename,
                "suggested_destination": str(doc.suggested_destination),
                "metadata": doc.metadata,
                "organized": False,
            }

            # If auto_organize, move the file
            if params.auto_organize:
                try:
                    # Create destination directory
                    doc.suggested_destination.parent.mkdir(parents=True, exist_ok=True)

                    # Move the file
                    shutil.move(str(source_path), str(doc.suggested_destination))

                    result_data["organized"] = True
                    result_data["final_location"] = str(doc.suggested_destination)

                except Exception as e:
                    return ToolResult.error_result(
                        f"Document processed but move failed: {e}. "
                        f"Suggested destination was: {doc.suggested_destination}"
                    )

            return ToolResult.success_result(data=result_data)

        except Exception as e:
            return ToolResult.error_result(f"Failed to process document: {e}")

    def get_confirmation_message(self, params: ProcessDocumentParams) -> str:
        """Get confirmation message for user approval."""
        if params.auto_organize:
            return f"Process and organize: {params.source_path}"
        else:
            return f"Process (preview only): {params.source_path}"
