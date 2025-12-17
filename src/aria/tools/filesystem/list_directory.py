"""List directory tool for filesystem operations."""

import fnmatch
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from aria.tools import BaseTool, ToolResult, RiskLevel


class FileInfo(BaseModel):
    """Information about a file or directory."""

    name: str
    path: str
    size: int  # Bytes, 0 for directories
    modified: datetime
    is_dir: bool
    is_file: bool


class ListDirectoryParams(BaseModel):
    """Parameters for listing directory contents."""

    path: str = Field(description="Directory path to list")
    pattern: str | None = Field(
        default=None,
        description="Optional glob pattern to filter files (e.g., '*.pdf', '*.txt')",
    )
    recursive: bool = Field(
        default=False,
        description="Recursively list subdirectories",
    )


class ListDirectoryTool(BaseTool[ListDirectoryParams]):
    """List files and directories in a path.

    This tool allows the agent to explore the filesystem and find files
    matching specific patterns.
    """

    name = "list_directory"
    description = "List files and directories, optionally filtering by pattern"
    risk_level = RiskLevel.LOW
    parameters_schema = ListDirectoryParams

    async def execute(self, params: ListDirectoryParams) -> ToolResult:
        """List directory contents."""
        try:
            path = Path(params.path).expanduser().resolve()

            if not path.exists():
                return ToolResult.error_result(f"Path does not exist: {path}")

            if not path.is_dir():
                return ToolResult.error_result(f"Path is not a directory: {path}")

            files: list[FileInfo] = []

            if params.recursive:
                # Recursive listing
                pattern = params.pattern or "*"
                for item in path.rglob(pattern):
                    try:
                        stat = item.stat()
                        files.append(
                            FileInfo(
                                name=item.name,
                                path=str(item),
                                size=stat.st_size if item.is_file() else 0,
                                modified=datetime.fromtimestamp(stat.st_mtime),
                                is_dir=item.is_dir(),
                                is_file=item.is_file(),
                            )
                        )
                    except (OSError, PermissionError):
                        # Skip files we can't access
                        continue
            else:
                # Non-recursive listing
                for item in path.iterdir():
                    try:
                        # Apply pattern filter if specified
                        if params.pattern and not fnmatch.fnmatch(item.name, params.pattern):
                            continue

                        stat = item.stat()
                        files.append(
                            FileInfo(
                                name=item.name,
                                path=str(item),
                                size=stat.st_size if item.is_file() else 0,
                                modified=datetime.fromtimestamp(stat.st_mtime),
                                is_dir=item.is_dir(),
                                is_file=item.is_file(),
                            )
                        )
                    except (OSError, PermissionError):
                        # Skip files we can't access
                        continue

            # Sort: directories first, then by name
            files.sort(key=lambda f: (not f.is_dir, f.name.lower()))

            return ToolResult.success_result(
                data={
                    "path": str(path),
                    "count": len(files),
                    "files": [f.model_dump() for f in files],
                }
            )

        except Exception as e:
            return ToolResult.error_result(f"Failed to list directory: {e}")

    def get_confirmation_message(self, params: ListDirectoryParams) -> str:
        """Get confirmation message (not needed for low-risk read)."""
        return f"List directory: {params.path}"
