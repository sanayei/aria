"""Tool for undoing file organization operations."""

import logging
from pathlib import Path
from typing import List, Dict, Any

from pydantic import BaseModel, Field

from aria.tools import BaseTool, ToolResult, RiskLevel
from aria.tools.organization.log_manager import (
    load_organization_log,
    get_most_recent_log,
)

logger = logging.getLogger(__name__)


class UndoOrganizationParams(BaseModel):
    """Parameters for undoing organization operations."""

    log_file: str | None = Field(
        default=None,
        description="Specific log file to undo. If not provided, use most recent.",
    )
    dry_run: bool = Field(
        default=True,
        description="If True, preview what would be undone without moving files",
    )


class UndoOrganizationTool(BaseTool[UndoOrganizationParams]):
    """Undo a file organization operation.

    This tool reverses a previous file organization by moving files
    back to their original locations. Requires the organization log
    to determine what operations to reverse.

    Safety features:
    - Defaults to dry_run mode
    - Only undoes if destination file still exists
    - Only undoes if source location is available
    - Reports what can/cannot be undone
    """

    name = "undo_organization"
    description = (
        "Undo a previous file organization operation by moving files back "
        "to their original locations. Requires organization log."
    )
    risk_level = RiskLevel.MEDIUM
    parameters_schema = UndoOrganizationParams

    async def execute(self, params: UndoOrganizationParams) -> ToolResult:
        """Undo organization operation."""
        try:
            # Load the log
            if params.log_file:
                log_path = Path(params.log_file).expanduser().resolve()
                if not log_path.exists():
                    return ToolResult.error_result(f"Log file not found: {log_path}")
                log = load_organization_log(log_path)
            else:
                # Use most recent log
                log = get_most_recent_log()
                if log is None:
                    return ToolResult.error_result(
                        "No organization logs found. Cannot undo."
                    )

            # Filter operations that were actually completed
            completed_ops = [
                op for op in log.operations
                if op.get("status") == "completed" and op.get("action") == "move"
            ]

            if not completed_ops:
                return ToolResult.success_result(data={
                    "dry_run": params.dry_run,
                    "can_undo": [],
                    "cannot_undo": [],
                    "summary": {
                        "can_undo": 0,
                        "cannot_undo": 0,
                        "undone": 0,
                        "failed": 0,
                    },
                    "message": "No completed move operations to undo",
                })

            # Check which operations can be undone
            can_undo: List[Dict[str, Any]] = []
            cannot_undo: List[Dict[str, Any]] = []

            for op in completed_ops:
                source = Path(op["source"])
                destination = Path(op["destination"])

                # Check if destination still exists
                if not destination.exists():
                    cannot_undo.append({
                        "source": str(source),
                        "destination": str(destination),
                        "reason": "Destination file no longer exists",
                        "status": "cannot_undo",
                    })
                    continue

                # Check if source location is available
                if source.exists():
                    cannot_undo.append({
                        "source": str(source),
                        "destination": str(destination),
                        "reason": "Source location already occupied",
                        "status": "cannot_undo",
                    })
                    continue

                # Can undo this operation
                can_undo.append({
                    "source": str(source),
                    "destination": str(destination),
                    "status": "pending",
                })

            # If dry run, return what would be done
            if params.dry_run:
                summary = {
                    "can_undo": len(can_undo),
                    "cannot_undo": len(cannot_undo),
                }

                return ToolResult.success_result(data={
                    "dry_run": True,
                    "log_timestamp": log.timestamp,
                    "original_scheme": log.scheme,
                    "original_source": log.source_path,
                    "can_undo": can_undo,
                    "cannot_undo": cannot_undo,
                    "summary": summary,
                })

            # Execute undo operations
            undone_ops = []
            failed_ops = []

            for undo_op in can_undo:
                try:
                    source = Path(undo_op["source"])
                    destination = Path(undo_op["destination"])

                    # Create source parent directory if needed
                    source.parent.mkdir(parents=True, exist_ok=True)

                    # Move file back
                    destination.rename(source)

                    undo_op["status"] = "undone"
                    undone_ops.append(undo_op)
                    logger.info(f"Undone: {destination} -> {source}")

                except Exception as e:
                    undo_op["status"] = "error"
                    undo_op["error_message"] = str(e)
                    failed_ops.append(undo_op)
                    logger.error(f"Failed to undo {destination} -> {source}: {e}")

            # Clean up empty directories
            self._cleanup_empty_directories(Path(log.destination_path))

            summary = {
                "can_undo": len(can_undo),
                "cannot_undo": len(cannot_undo),
                "undone": len(undone_ops),
                "failed": len(failed_ops),
            }

            return ToolResult.success_result(data={
                "dry_run": False,
                "log_timestamp": log.timestamp,
                "original_scheme": log.scheme,
                "original_source": log.source_path,
                "undone": undone_ops,
                "failed": failed_ops,
                "cannot_undo": cannot_undo,
                "summary": summary,
            })

        except ValueError as e:
            return ToolResult.error_result(str(e))
        except Exception as e:
            logger.exception("Error undoing organization")
            return ToolResult.error_result(f"Failed to undo organization: {e}")

    def _cleanup_empty_directories(self, base_path: Path) -> None:
        """Remove empty directories after undoing organization.

        Args:
            base_path: Base path to search for empty directories
        """
        try:
            # Traverse directories bottom-up to remove empty ones
            for dirpath, dirnames, filenames in sorted(
                base_path.walk(), key=lambda x: len(x[0].parts), reverse=True
            ):
                # Skip base directory
                if dirpath == base_path:
                    continue

                # Check if directory is empty
                try:
                    if not any(dirpath.iterdir()):
                        dirpath.rmdir()
                        logger.info(f"Removed empty directory: {dirpath}")
                except OSError:
                    # Directory not empty or permission error
                    pass

        except Exception as e:
            logger.warning(f"Failed to cleanup empty directories: {e}")

    def get_confirmation_message(self, params: UndoOrganizationParams) -> str:
        """Get confirmation message."""
        if params.dry_run:
            if params.log_file:
                return f"Preview undo of organization from log: {params.log_file}"
            else:
                return "Preview undo of most recent organization (dry run)"
        else:
            if params.log_file:
                return f"Undo organization from log: {params.log_file}"
            else:
                return "Undo most recent organization operation"
