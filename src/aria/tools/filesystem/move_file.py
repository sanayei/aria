"""Move file tool for filesystem operations."""

import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from aria.tools import BaseTool, ToolResult, RiskLevel


class MoveFileParams(BaseModel):
    """Parameters for moving a file."""

    source: str = Field(description="Source file path")
    destination: str = Field(description="Destination file path")
    overwrite: bool = Field(
        default=False,
        description="Overwrite destination if it exists",
    )


class MoveFileTool(BaseTool[MoveFileParams]):
    """Move or rename a file.

    This tool moves files between locations or renames them. It can create
    destination directories if needed. Classified as MEDIUM risk because
    it modifies the filesystem.
    """

    name = "move_file"
    description = "Move or rename a file, creating directories if needed"
    risk_level = RiskLevel.MEDIUM
    parameters_schema = MoveFileParams

    async def execute(self, params: MoveFileParams) -> ToolResult:
        """Move a file."""
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

            # Get source file size before moving
            size_bytes = source.stat().st_size

            # Move the file
            shutil.move(str(source), str(destination))

            return ToolResult.success_result(
                data={
                    "source": str(source),
                    "destination": str(destination),
                    "size_bytes": size_bytes,
                    "overwritten": destination.exists() and params.overwrite,
                }
            )

        except PermissionError:
            return ToolResult.error_result(
                f"Permission denied when moving {params.source} to {params.destination}"
            )
        except Exception as e:
            return ToolResult.error_result(f"Failed to move file: {e}")

    def get_confirmation_message(self, params: MoveFileParams) -> str:
        """Get confirmation message for user approval."""
        msg = f"Move {params.source} to {params.destination}"
        if params.overwrite:
            msg += " (will overwrite if exists)"
        return msg
