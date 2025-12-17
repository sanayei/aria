"""Read file tool for filesystem operations."""

from pathlib import Path

from pydantic import BaseModel, Field

from aria.tools import BaseTool, ToolResult, RiskLevel


class ReadFileParams(BaseModel):
    """Parameters for reading a file."""

    path: str = Field(description="File path to read")
    encoding: str = Field(
        default="utf-8",
        description="File encoding (default: utf-8)",
    )


class ReadFileTool(BaseTool[ReadFileParams]):
    """Read text file contents.

    This tool allows the agent to read file contents for analysis,
    search, or processing.
    """

    name = "read_file"
    description = "Read the contents of a text file"
    risk_level = RiskLevel.LOW
    parameters_schema = ReadFileParams

    async def execute(self, params: ReadFileParams) -> ToolResult:
        """Read file contents."""
        try:
            path = Path(params.path).expanduser().resolve()

            if not path.exists():
                return ToolResult.error_result(f"File does not exist: {path}")

            if not path.is_file():
                return ToolResult.error_result(f"Path is not a file: {path}")

            # Read file contents
            try:
                content = path.read_text(encoding=params.encoding)
            except UnicodeDecodeError:
                return ToolResult.error_result(
                    f"Failed to decode file with encoding '{params.encoding}'. "
                    "File may be binary or use a different encoding."
                )

            # Get file metadata
            stat = path.stat()
            size_kb = stat.st_size / 1024

            return ToolResult.success_result(
                data={
                    "path": str(path),
                    "content": content,
                    "size_bytes": stat.st_size,
                    "size_kb": round(size_kb, 2),
                    "lines": len(content.splitlines()),
                }
            )

        except PermissionError:
            return ToolResult.error_result(f"Permission denied: {params.path}")
        except Exception as e:
            return ToolResult.error_result(f"Failed to read file: {e}")

    def get_confirmation_message(self, params: ReadFileParams) -> str:
        """Get confirmation message (not needed for low-risk read)."""
        return f"Read file: {params.path}"
