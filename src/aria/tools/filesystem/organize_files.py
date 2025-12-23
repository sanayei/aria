"""Smart file organizer tool for organizing files into categorized subfolders."""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Dict, List, Any

from pydantic import BaseModel, Field

from aria.tools import BaseTool, ToolResult, RiskLevel
from aria.tools.filesystem.analyze_file import get_file_category

logger = logging.getLogger(__name__)

# Import log manager for saving operation logs
try:
    from aria.tools.organization.log_manager import save_organization_log

    _LOG_MANAGER_AVAILABLE = True
except ImportError:
    _LOG_MANAGER_AVAILABLE = False
    logger.warning("Organization log manager not available")


# Category to folder name mapping
CATEGORY_FOLDERS = {
    "document": "Documents",
    "spreadsheet": "Spreadsheets",
    "presentation": "Presentations",
    "image": "Images",
    "video": "Videos",
    "audio": "Audio",
    "code": "Code",
    "archive": "Archives",
    "executable": "Programs",
    "other": "Other",
}


class OrganizeFilesParams(BaseModel):
    """Parameters for organizing files in a directory."""

    source_path: str = Field(description="Directory containing files to organize")
    destination_path: str | None = Field(
        default=None,
        description="Where to create organized structure (default: same as source)",
    )
    organization_scheme: Literal["category", "date", "extension", "date_category"] = Field(
        default="category",
        description=(
            "How to organize files: "
            "'category' (by type), 'date' (by modified date), "
            "'extension' (by file extension), 'date_category' (combined)"
        ),
    )
    dry_run: bool = Field(
        default=True,
        description="If True, only return what would be done without moving files",
    )
    recursive: bool = Field(
        default=False,
        description="Process subdirectories",
    )
    skip_hidden: bool = Field(
        default=True,
        description="Skip hidden files (starting with .)",
    )
    conflict_resolution: Literal["skip", "rename", "overwrite"] = Field(
        default="rename",
        description=(
            "What to do if destination exists: "
            "'skip' (keep original), 'rename' (add number suffix), "
            "'overwrite' (replace - requires explicit approval)"
        ),
    )


class FileOperation(BaseModel):
    """Represents a single file operation."""

    action: Literal["move", "skip"]
    source: str
    destination: str
    category: str
    reason: str | None = None
    status: Literal["pending", "completed", "skipped", "error"] = "pending"
    error_message: str | None = None


class OrganizeFilesTool(BaseTool[OrganizeFilesParams]):
    """Organize files in a directory by moving them into categorized subfolders.

    This tool helps clean up messy directories by automatically organizing
    files into a structured hierarchy based on file types, dates, or extensions.

    Safety features:
    - Defaults to dry_run mode (no actual moves)
    - Requires approval for actual file moves
    - Handles conflicts intelligently
    - Provides detailed operation reports
    """

    name = "organize_files"
    description = (
        "Organize files in a directory by moving them into categorized subfolders. "
        "Supports organization by category, date, extension, or combined schemes. "
        "Always runs in dry-run mode by default for safety."
    )
    risk_level = RiskLevel.MEDIUM
    parameters_schema = OrganizeFilesParams

    async def execute(self, params: OrganizeFilesParams) -> ToolResult:
        """Organize files according to the specified scheme."""
        try:
            source_path = Path(params.source_path).expanduser().resolve()

            # Validate source directory
            if not source_path.exists():
                return ToolResult.error_result(f"Source directory does not exist: {source_path}")

            if not source_path.is_dir():
                return ToolResult.error_result(f"Source path is not a directory: {source_path}")

            # Determine destination path
            if params.destination_path:
                dest_path = Path(params.destination_path).expanduser().resolve()
            else:
                dest_path = source_path

            # Validate overwrite permission
            if params.conflict_resolution == "overwrite" and not params.dry_run:
                return ToolResult.error_result(
                    "Overwrite mode requires HIGH risk approval. "
                    "Please use 'skip' or 'rename' conflict resolution instead."
                )

            # Collect files to organize
            operations: List[FileOperation] = []
            file_iterator = source_path.rglob("*") if params.recursive else source_path.glob("*")

            for file_path in file_iterator:
                # Skip non-files
                if not file_path.is_file():
                    continue

                # Skip hidden files if requested
                if params.skip_hidden and file_path.name.startswith("."):
                    continue

                # Skip files already in organized subfolders (avoid re-organizing)
                try:
                    relative = file_path.relative_to(source_path)
                    # If file is in a subdirectory with a known category folder name
                    if len(relative.parts) > 1 and relative.parts[0] in CATEGORY_FOLDERS.values():
                        continue
                except ValueError:
                    pass

                # Determine destination for this file
                destination = self._determine_destination(
                    file_path,
                    dest_path,
                    params.organization_scheme,
                )

                # Check if already in correct location
                if file_path == destination:
                    operations.append(
                        FileOperation(
                            action="skip",
                            source=str(file_path),
                            destination=str(destination),
                            category=self._get_file_category(file_path),
                            reason="Already in correct location",
                            status="skipped",
                        )
                    )
                    continue

                # Handle conflicts
                if destination.exists() and file_path != destination:
                    if params.conflict_resolution == "skip":
                        operations.append(
                            FileOperation(
                                action="skip",
                                source=str(file_path),
                                destination=str(destination),
                                category=self._get_file_category(file_path),
                                reason="Destination exists (conflict)",
                                status="skipped",
                            )
                        )
                        continue
                    elif params.conflict_resolution == "rename":
                        destination = self._find_unique_destination(destination)

                # Create operation
                operations.append(
                    FileOperation(
                        action="move",
                        source=str(file_path),
                        destination=str(destination),
                        category=self._get_file_category(file_path),
                        status="pending",
                    )
                )

            # If dry run, return planned operations
            if params.dry_run:
                summary = self._generate_summary(operations, dest_path)
                return ToolResult.success_result(
                    data={
                        "source_path": str(source_path),
                        "destination_path": str(dest_path),
                        "scheme": params.organization_scheme,
                        "dry_run": True,
                        "operations": [op.model_dump() for op in operations],
                        "summary": summary,
                    }
                )

            # Execute actual moves
            executed_operations = await self._execute_moves(operations)
            summary = self._generate_summary(executed_operations, dest_path)

            # Save operation log for undo capability
            log_path = None
            if _LOG_MANAGER_AVAILABLE:
                try:
                    log_path = save_organization_log(
                        source_path=str(source_path),
                        destination_path=str(dest_path),
                        scheme=params.organization_scheme,
                        operations=[op.model_dump() for op in executed_operations],
                    )
                    logger.info(f"Saved organization log: {log_path}")
                except Exception as e:
                    logger.warning(f"Failed to save organization log: {e}")

            result_data = {
                "source_path": str(source_path),
                "destination_path": str(dest_path),
                "scheme": params.organization_scheme,
                "dry_run": False,
                "operations": [op.model_dump() for op in executed_operations],
                "summary": summary,
            }

            # Add log path to result if available
            if log_path:
                result_data["log_file"] = str(log_path)

            return ToolResult.success_result(data=result_data)

        except PermissionError as e:
            return ToolResult.error_result(f"Permission denied: {e}")
        except Exception as e:
            logger.exception("Error organizing files")
            return ToolResult.error_result(f"Failed to organize files: {e}")

    def _determine_destination(
        self,
        file_path: Path,
        dest_base: Path,
        scheme: str,
    ) -> Path:
        """Determine the destination path for a file based on the organization scheme."""
        if scheme == "category":
            category = self._get_file_category(file_path)
            folder = CATEGORY_FOLDERS.get(category, "Other")
            return dest_base / folder / file_path.name

        elif scheme == "date":
            stat = file_path.stat()
            modified_date = datetime.fromtimestamp(stat.st_mtime)
            year_month = modified_date.strftime("%Y/%m")
            return dest_base / year_month / file_path.name

        elif scheme == "extension":
            ext = file_path.suffix.lower().lstrip(".") if file_path.suffix else "no_extension"
            return dest_base / ext / file_path.name

        elif scheme == "date_category":
            stat = file_path.stat()
            modified_date = datetime.fromtimestamp(stat.st_mtime)
            year_month = modified_date.strftime("%Y/%m")
            category = self._get_file_category(file_path)
            folder = CATEGORY_FOLDERS.get(category, "Other")
            return dest_base / year_month / folder / file_path.name

        else:
            # Default to category
            category = self._get_file_category(file_path)
            folder = CATEGORY_FOLDERS.get(category, "Other")
            return dest_base / folder / file_path.name

    def _get_file_category(self, file_path: Path) -> str:
        """Get the category for a file."""
        extension = file_path.suffix.lower().lstrip(".") if file_path.suffix else ""
        return get_file_category(extension)

    def _find_unique_destination(self, destination: Path) -> Path:
        """Find a unique destination by adding number suffixes."""
        if not destination.exists():
            return destination

        base_name = destination.stem
        extension = destination.suffix
        parent = destination.parent
        counter = 1

        while True:
            new_name = f"{base_name}_{counter}{extension}"
            new_dest = parent / new_name
            if not new_dest.exists():
                return new_dest
            counter += 1

            # Safety limit
            if counter > 10000:
                raise ValueError("Could not find unique destination filename")

    async def _execute_moves(self, operations: List[FileOperation]) -> List[FileOperation]:
        """Execute the actual file move operations."""
        executed = []

        for operation in operations:
            if operation.action != "move":
                executed.append(operation)
                continue

            try:
                source = Path(operation.source)
                destination = Path(operation.destination)

                # Create destination directory if needed
                destination.parent.mkdir(parents=True, exist_ok=True)

                # Move the file
                source.rename(destination)

                # Update operation status
                operation.status = "completed"
                logger.info(f"Moved {source} -> {destination}")

            except Exception as e:
                operation.status = "error"
                operation.error_message = str(e)
                logger.error(f"Failed to move {operation.source}: {e}")

            executed.append(operation)

        return executed

    def _generate_summary(
        self,
        operations: List[FileOperation],
        dest_path: Path,
    ) -> Dict[str, Any]:
        """Generate a summary of operations."""
        total_files = len(operations)
        to_move = sum(1 for op in operations if op.action == "move" and op.status == "pending")
        to_skip = sum(1 for op in operations if op.action == "skip")
        completed = sum(1 for op in operations if op.status == "completed")
        errors = sum(1 for op in operations if op.status == "error")

        # Count by destination folder
        by_destination: Dict[str, int] = defaultdict(int)
        for op in operations:
            if op.action == "move":
                dest = Path(op.destination)
                try:
                    relative = dest.relative_to(dest_path)
                    # Get the top-level folder
                    if len(relative.parts) > 1:
                        folder = relative.parts[0]
                        by_destination[f"{folder}/"] += 1
                    else:
                        by_destination["(root)/"] += 1
                except ValueError:
                    by_destination["(unknown)/"] += 1

        summary = {
            "total_files": total_files,
            "to_move": to_move,
            "to_skip": to_skip,
            "by_destination": dict(by_destination),
        }

        # Add execution stats if moves were performed
        if completed > 0 or errors > 0:
            summary["completed"] = completed
            summary["errors"] = errors

        return summary

    def get_confirmation_message(self, params: OrganizeFilesParams) -> str:
        """Get confirmation message for file organization."""
        if params.dry_run:
            return f"Plan organization of files in: {params.source_path} (dry run - no changes)"

        dest = params.destination_path or params.source_path
        scheme_desc = {
            "category": "by file category",
            "date": "by date",
            "extension": "by extension",
            "date_category": "by date and category",
        }

        return (
            f"Organize files in {params.source_path} "
            f"{scheme_desc.get(params.organization_scheme, 'by category')} "
            f"to {dest} "
            f"(conflict: {params.conflict_resolution})"
        )
