"""File analysis tool for comprehensive file metadata extraction."""

import mimetypes
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from aria.tools import BaseTool, ToolResult, RiskLevel


# File category mappings
FILE_CATEGORIES = {
    "document": {
        "pdf", "doc", "docx", "txt", "md", "rtf", "odt", "tex", "wpd",
        "pages", "epub", "mobi", "azw"
    },
    "spreadsheet": {
        "xls", "xlsx", "csv", "ods", "tsv", "numbers"
    },
    "presentation": {
        "ppt", "pptx", "odp", "key"
    },
    "image": {
        "jpg", "jpeg", "png", "gif", "bmp", "webp", "svg", "tiff", "tif",
        "ico", "heic", "heif", "raw", "cr2", "nef", "psd"
    },
    "video": {
        "mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "m4v", "mpg",
        "mpeg", "3gp", "ogv"
    },
    "audio": {
        "mp3", "wav", "flac", "aac", "ogg", "m4a", "wma", "opus", "ape",
        "alac", "aiff"
    },
    "code": {
        "py", "js", "ts", "tsx", "jsx", "java", "c", "cpp", "h", "hpp",
        "go", "rs", "rb", "php", "html", "css", "scss", "sass", "json",
        "yaml", "yml", "xml", "sql", "sh", "bash", "zsh", "ps1", "r",
        "swift", "kt", "scala", "pl", "lua", "vim", "el"
    },
    "archive": {
        "zip", "tar", "gz", "bz2", "xz", "rar", "7z", "tgz", "tbz2",
        "iso", "dmg", "pkg"
    },
    "executable": {
        "exe", "msi", "app", "bat", "cmd", "com", "bin", "deb", "rpm",
        "apk", "jar"
    },
}

# Text file extensions (files that can be safely read as text)
TEXT_EXTENSIONS = {
    "txt", "md", "rst", "log", "csv", "json", "xml", "yaml", "yml",
    "toml", "ini", "cfg", "conf", "py", "js", "ts", "java", "c", "cpp",
    "h", "go", "rs", "rb", "php", "html", "css", "sh", "bash", "sql",
    "tex", "r", "scala", "kt", "swift", "pl", "lua", "vim", "el",
}


def format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string (e.g., "2.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def get_file_category(extension: str) -> str:
    """Determine file category from extension.

    Args:
        extension: File extension (lowercase, without dot)

    Returns:
        Category name
    """
    for category, extensions in FILE_CATEGORIES.items():
        if extension in extensions:
            return category
    return "other"


class AnalyzeFileParams(BaseModel):
    """Parameters for analyzing a file."""

    path: str = Field(description="Path to the file to analyze")


class AnalyzeFileTool(BaseTool[AnalyzeFileParams]):
    """Analyze a file and return detailed metadata.

    This tool provides comprehensive information about a file including:
    - Basic metadata (name, size, timestamps)
    - MIME type detection
    - Automatic category classification
    - Text file detection
    """

    name = "analyze_file"
    description = "Analyze a file and return detailed metadata including size, type, and category"
    risk_level = RiskLevel.LOW
    parameters_schema = AnalyzeFileParams

    async def execute(self, params: AnalyzeFileParams) -> ToolResult:
        """Analyze file and return metadata."""
        try:
            path = Path(params.path).expanduser().resolve()

            # Check if file exists
            if not path.exists():
                return ToolResult.error_result(f"File does not exist: {path}")

            if not path.is_file():
                return ToolResult.error_result(f"Path is not a file: {path}")

            # Get file stats
            stat = path.stat()

            # Extract file information
            name = path.name
            extension = path.suffix.lower().lstrip(".") if path.suffix else ""

            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type is None:
                mime_type = "application/octet-stream"

            # Determine category
            category = get_file_category(extension)

            # Check if file is hidden (Unix-style)
            is_hidden = name.startswith(".")

            # Check if file appears to be text-based
            is_text = extension in TEXT_EXTENSIONS or mime_type.startswith("text/")

            # Format timestamps
            created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()
            modified_at = datetime.fromtimestamp(stat.st_mtime).isoformat()

            # Build result
            result_data = {
                "name": name,
                "extension": extension if extension else None,
                "size_bytes": stat.st_size,
                "size_human": format_size(stat.st_size),
                "created_at": created_at,
                "modified_at": modified_at,
                "mime_type": mime_type,
                "category": category,
                "is_hidden": is_hidden,
                "is_text": is_text,
                "path": str(path),
            }

            return ToolResult.success_result(data=result_data)

        except PermissionError:
            return ToolResult.error_result(f"Permission denied: {params.path}")
        except Exception as e:
            return ToolResult.error_result(f"Failed to analyze file: {e}")

    def get_confirmation_message(self, params: AnalyzeFileParams) -> str:
        """Get confirmation message (not needed for low-risk analysis)."""
        return f"Analyze file: {params.path}"
