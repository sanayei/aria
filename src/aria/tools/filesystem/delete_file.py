"""Delete file tool for filesystem operations."""

from pathlib import Path

from pydantic import BaseModel, Field

from aria.tools import BaseTool, ToolResult, RiskLevel


class DeleteFileParams(BaseModel):
    """Parameters for deleting a file."""

    path: str = Field(description="File path to delete")
    confirm_name: str = Field(
        description="Confirmation: must match the filename to proceed (safety check)",
    )


class DeleteFileTool(BaseTool[DeleteFileParams]):
    """Delete a file (HIGH RISK).

    This tool permanently deletes a file. For safety, it requires the user
    to confirm the filename. This prevents accidental deletions.

    Classified as HIGH risk because deletion is irreversible.
    """

    name = "delete_file"
    description = "Permanently delete a file (requires confirmation)"
    risk_level = RiskLevel.HIGH
    parameters_schema = DeleteFileParams

    async def execute(self, params: DeleteFileParams) -> ToolResult:
        """Delete a file."""
        try:
            path = Path(params.path).expanduser().resolve()

            # Validation
            if not path.exists():
                return ToolResult.error_result(f"File does not exist: {path}")

            if not path.is_file():
                return ToolResult.error_result(f"Path is not a file: {path}")

            # Safety check: confirm_name must match filename
            if path.name != params.confirm_name:
                return ToolResult.error_result(
                    f"Confirmation failed: expected '{path.name}', got '{params.confirm_name}'"
                )

            # Get file info before deletion
            size_bytes = path.stat().st_size
            file_name = path.name

            # Delete the file
            path.unlink()

            return ToolResult.success_result(
                data={
                    "deleted": str(path),
                    "name": file_name,
                    "size_bytes": size_bytes,
                }
            )

        except PermissionError:
            return ToolResult.error_result(f"Permission denied: {params.path}")
        except Exception as e:
            return ToolResult.error_result(f"Failed to delete file: {e}")

    def get_confirmation_message(self, params: DeleteFileParams) -> str:
        """Get confirmation message for user approval."""
        path = Path(params.path).expanduser().resolve()
        try:
            size_bytes = path.stat().st_size
            size_kb = size_bytes / 1024
            if size_kb < 1024:
                size_str = f"{size_kb:.1f} KB"
            else:
                size_str = f"{size_kb / 1024:.1f} MB"
            return f"DELETE file: {path} ({size_str})"
        except Exception:
            return f"DELETE file: {path}"
