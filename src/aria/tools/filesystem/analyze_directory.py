"""Directory analysis tool for batch file metadata extraction and statistics."""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from pydantic import BaseModel, Field

from aria.tools import BaseTool, ToolResult, RiskLevel
from aria.tools.filesystem.analyze_file import (
    get_file_category,
    format_size,
)


class AnalyzeDirectoryParams(BaseModel):
    """Parameters for analyzing a directory."""

    path: str = Field(description="Directory path to analyze")
    recursive: bool = Field(
        default=False,
        description="Include subdirectories in analysis",
    )
    include_hidden: bool = Field(
        default=False,
        description="Include hidden files (starting with .)",
    )
    max_files: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum number of files to analyze (safety limit)",
    )


class AnalyzeDirectoryTool(BaseTool[AnalyzeDirectoryParams]):
    """Analyze all files in a directory and provide categorized summary.

    This tool scans a directory and provides:
    - Total file count and size
    - Breakdown by category (document, image, code, etc.)
    - Top extensions
    - Largest, oldest, and newest files
    - Potential duplicates (same name and size)
    """

    name = "analyze_directory"
    description = "Analyze all files in a directory and return categorized summary statistics"
    risk_level = RiskLevel.LOW
    parameters_schema = AnalyzeDirectoryParams

    async def execute(self, params: AnalyzeDirectoryParams) -> ToolResult:
        """Analyze directory and return comprehensive statistics."""
        try:
            path = Path(params.path).expanduser().resolve()

            # Validate directory
            if not path.exists():
                return ToolResult.error_result(f"Directory does not exist: {path}")

            if not path.is_dir():
                return ToolResult.error_result(f"Path is not a directory: {path}")

            # Initialize statistics
            total_files = 0
            total_size = 0
            categories: Dict[str, Dict[str, Any]] = defaultdict(
                lambda: {"count": 0, "size_bytes": 0, "extensions": set()}
            )
            by_extension: Dict[str, Dict[str, Any]] = defaultdict(
                lambda: {"count": 0, "size_bytes": 0}
            )
            all_files: List[Dict[str, Any]] = []
            duplicates_map: Dict[tuple, List[str]] = defaultdict(list)
            permission_errors = 0

            # Get file iterator
            if params.recursive:
                file_iterator = path.rglob("*")
            else:
                file_iterator = path.glob("*")

            # Process files
            for file_path in file_iterator:
                # Check max files limit
                if total_files >= params.max_files:
                    break

                # Skip non-files
                if not file_path.is_file():
                    continue

                # Skip hidden files if not included
                if not params.include_hidden and file_path.name.startswith("."):
                    continue

                try:
                    # Get file stats
                    stat = file_path.stat()
                    extension = file_path.suffix.lower().lstrip(".") if file_path.suffix else ""
                    category = get_file_category(extension)

                    # Update totals
                    total_files += 1
                    total_size += stat.st_size

                    # Update category stats
                    categories[category]["count"] += 1
                    categories[category]["size_bytes"] += stat.st_size
                    if extension:
                        categories[category]["extensions"].add(extension)

                    # Update extension stats
                    ext_key = f".{extension}" if extension else "(no extension)"
                    by_extension[ext_key]["count"] += 1
                    by_extension[ext_key]["size_bytes"] += stat.st_size

                    # Store file info for sorting/ranking
                    file_info = {
                        "path": str(file_path.relative_to(path)),
                        "name": file_path.name,
                        "size_bytes": stat.st_size,
                        "size_human": format_size(stat.st_size),
                        "category": category,
                        "extension": extension if extension else None,
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "modified_timestamp": stat.st_mtime,
                    }
                    all_files.append(file_info)

                    # Track potential duplicates (same name + size)
                    dup_key = (file_path.name, stat.st_size)
                    duplicates_map[dup_key].append(str(file_path.relative_to(path)))

                except PermissionError:
                    permission_errors += 1
                    continue
                except Exception:
                    # Skip files we can't process
                    continue

            # Sort and get top files
            all_files_by_size = sorted(all_files, key=lambda x: x["size_bytes"], reverse=True)
            all_files_by_date = sorted(all_files, key=lambda x: x["modified_timestamp"])

            largest_files = [
                {k: v for k, v in f.items() if k != "modified_timestamp"}
                for f in all_files_by_size[:10]
            ]

            oldest_files = [
                {k: v for k, v in f.items() if k != "modified_timestamp"}
                for f in all_files_by_date[:10]
            ]

            newest_files = [
                {k: v for k, v in f.items() if k != "modified_timestamp"}
                for f in reversed(all_files_by_date[-10:])
            ]

            # Get top extensions
            top_extensions = dict(
                sorted(
                    by_extension.items(),
                    key=lambda x: x[1]["count"],
                    reverse=True,
                )[:20]
            )

            # Find duplicates (files appearing in multiple locations)
            potential_duplicates = [
                {
                    "name": name,
                    "size_bytes": size,
                    "size_human": format_size(size),
                    "locations": locations,
                    "count": len(locations),
                }
                for (name, size), locations in duplicates_map.items()
                if len(locations) > 1
            ]
            # Sort by number of duplicates
            potential_duplicates.sort(key=lambda x: x["count"], reverse=True)
            potential_duplicates = potential_duplicates[:20]  # Top 20 duplicates

            # Convert category extensions sets to lists
            categories_output = {}
            for cat, data in categories.items():
                categories_output[cat] = {
                    "count": data["count"],
                    "size_bytes": data["size_bytes"],
                    "size_human": format_size(data["size_bytes"]),
                    "extensions": sorted(data["extensions"]),
                }

            # Build result
            result_data = {
                "path": str(path),
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_human": format_size(total_size),
                "categories": categories_output,
                "by_extension": top_extensions,
                "largest_files": largest_files,
                "oldest_files": oldest_files,
                "newest_files": newest_files,
                "potential_duplicates": potential_duplicates,
                "permission_errors": permission_errors,
                "reached_max_files": total_files >= params.max_files,
            }

            return ToolResult.success_result(data=result_data)

        except PermissionError:
            return ToolResult.error_result(f"Permission denied: {params.path}")
        except Exception as e:
            return ToolResult.error_result(f"Failed to analyze directory: {e}")

    def get_confirmation_message(self, params: AnalyzeDirectoryParams) -> str:
        """Get confirmation message (not needed for low-risk analysis)."""
        mode = "recursively" if params.recursive else "non-recursively"
        return f"Analyze directory {mode}: {params.path} (max {params.max_files} files)"
