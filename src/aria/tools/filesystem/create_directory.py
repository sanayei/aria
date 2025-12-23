"""Create directory tool for filesystem operations."""

from pathlib import Path

from pydantic import BaseModel, Field

from aria.tools import BaseTool, ToolResult, RiskLevel


class CreateDirectoryParams(BaseModel):
    """Parameters for creating a directory."""

    path: str = Field(description="Directory path to create")


class CreateDirectoryTool(BaseTool[CreateDirectoryParams]):
    """Create a directory (and parent directories if needed).

    This tool creates directories using mkdir with parents=True,
    so it will create any missing parent directories in the path.
    """

    name = "create_directory"
    description = "Create a directory and any missing parent directories"
    risk_level = RiskLevel.LOW
    parameters_schema = CreateDirectoryParams

    async def execute(self, params: CreateDirectoryParams) -> ToolResult:
        """Create a directory."""
        try:
            path = Path(params.path).expanduser().resolve()

            # Check if it already exists
            if path.exists():
                if path.is_dir():
                    return ToolResult.success_result(
                        data={
                            "path": str(path),
                            "created": False,
                            "message": "Directory already exists",
                        }
                    )
                else:
                    return ToolResult.error_result(f"Path exists but is not a directory: {path}")

            # Create the directory
            path.mkdir(parents=True, exist_ok=True)

            return ToolResult.success_result(
                data={
                    "path": str(path),
                    "created": True,
                    "message": "Directory created successfully",
                }
            )

        except PermissionError:
            return ToolResult.error_result(f"Permission denied: {params.path}")
        except Exception as e:
            return ToolResult.error_result(f"Failed to create directory: {e}")

    def get_confirmation_message(self, params: CreateDirectoryParams) -> str:
        """Get confirmation message (not needed for low-risk create)."""
        return f"Create directory: {params.path}"
