"""Tool for listing organization operation logs."""

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from aria.tools import BaseTool, ToolResult, RiskLevel
from aria.tools.organization.log_manager import list_organization_logs

logger = logging.getLogger(__name__)


class ListOrganizationLogsParams(BaseModel):
    """Parameters for listing organization logs."""

    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of logs to return (newest first)",
    )


class ListOrganizationLogsTool(BaseTool[ListOrganizationLogsParams]):
    """List all organization operation logs.

    This tool shows a history of file organization operations,
    allowing users to review past operations and select logs to undo.
    """

    name = "list_organization_logs"
    description = (
        "List organization operation history logs. "
        "Shows timestamp, source folder, file count, and scheme used for each operation."
    )
    risk_level = RiskLevel.LOW
    parameters_schema = ListOrganizationLogsParams

    async def execute(self, params: ListOrganizationLogsParams) -> ToolResult:
        """List organization logs."""
        try:
            logs = list_organization_logs()

            if not logs:
                return ToolResult.success_result(
                    data={
                        "logs": [],
                        "total_logs": 0,
                        "message": "No organization logs found",
                    }
                )

            # Limit results
            logs = logs[: params.limit]

            # Format log data for output
            log_data = []
            for log in logs:
                # Get log file path
                timestamp_safe = log.timestamp.replace(":", "-").replace(".", "-")
                log_filename = f"{timestamp_safe}_organize.json"
                log_path = Path.home() / ".aria" / "organization_logs" / log_filename

                log_data.append(
                    {
                        "timestamp": log.timestamp,
                        "log_file": str(log_path),
                        "source_path": log.source_path,
                        "destination_path": log.destination_path,
                        "scheme": log.scheme,
                        "total_files": log.total_operations,
                        "completed": log.completed_operations,
                        "failed": log.failed_operations,
                    }
                )

            return ToolResult.success_result(
                data={
                    "logs": log_data,
                    "total_logs": len(log_data),
                    "showing": len(log_data),
                }
            )

        except Exception as e:
            logger.exception("Error listing organization logs")
            return ToolResult.error_result(f"Failed to list logs: {e}")

    def get_confirmation_message(self, params: ListOrganizationLogsParams) -> str:
        """Get confirmation message."""
        return f"List organization logs (limit: {params.limit})"
