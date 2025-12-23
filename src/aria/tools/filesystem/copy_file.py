"""Copy file tool for filesystem operations."""

import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from aria.tools import BaseTool, ToolResult, RiskLevel


class CopyFileParams(BaseModel):
    """Parameters for copying a file."""

    source: str = Field(description="Source file path")
    destination: str = Field(description="Destination file path")
    overwrite: bool = Field(
        default=False,
        description="Overwrite destination if it exists",
    )


class CopyFileTool(BaseTool[CopyFileParams]):
    """Copy a file to a new location.

    This tool creates a copy of a file, preserving the original.
    It can create destination directories if needed.
    """

    name = "copy_file"
    description = "Copy a file to a new location, creating directories if needed"
    risk_level = RiskLevel.LOW
    parameters_schema = CopyFileParams

    async def execute(self, params: CopyFileParams) -> ToolResult:
        """Copy a file."""
        try:
            source = Path(params.source).expanduser().resolve()
            destination = Path(params.destination).expanduser().resolve()

            # Validation
            if not source.exists():
                return ToolResult.error_result(f"Source file does not exist: {source}")

            if not source.is_file():
                return ToolResult.error_result(f"Source is not a file: {source}")

            if destination.exists() and not params.overwrite:
                return ToolResult.error_result(
                    f"Destination already exists: {destination}. Set overwrite=true to replace it."
                )

            # Create destination directory if needed
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Get source file size
            size_bytes = source.stat().st_size

            # Copy the file
            shutil.copy2(str(source), str(destination))

            return ToolResult.success_result(
                data={
                    "source": str(source),
                    "destination": str(destination),
                    "size_bytes": size_bytes,
                    "overwritten": params.overwrite and destination.exists(),
                }
            )

        except PermissionError:
            return ToolResult.error_result(
                f"Permission denied when copying {params.source} to {params.destination}"
            )
        except Exception as e:
            return ToolResult.error_result(f"Failed to copy file: {e}")

    def get_confirmation_message(self, params: CopyFileParams) -> str:
        """Get confirmation message (not needed for low-risk copy)."""
        msg = f"Copy {params.source} to {params.destination}"
        if params.overwrite:
            msg += " (will overwrite if exists)"
        return msg
