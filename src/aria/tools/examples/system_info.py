"""System information tool - Returns basic system info.

This tool provides information about the current system, including
time, platform, and basic system details.
"""

import platform
import sys
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from aria.tools.base import BaseTool, ToolResult, RiskLevel


class SystemInfoParams(BaseModel):
    """Parameters for the system info tool."""

    info_type: Literal["time", "platform", "python", "all"] = Field(
        default="all",
        description="Type of system information to retrieve",
    )


class SystemInfoTool(BaseTool[SystemInfoParams]):
    """System information tool.

    Returns basic system information like current time, platform details,
    and Python version. This is a read-only tool with no side effects.
    """

    name = "system_info"
    description = "Get system information (time, platform, Python version)"
    risk_level = RiskLevel.LOW
    parameters_schema = SystemInfoParams

    async def execute(self, params: SystemInfoParams) -> ToolResult:
        """Execute the system info operation.

        Args:
            params: Validated parameters

        Returns:
            ToolResult: Result containing system information
        """
        info = {}

        if params.info_type in ("time", "all"):
            now = datetime.now(timezone.utc)
            info["time"] = {
                "utc": now.isoformat(),
                "local": datetime.now().isoformat(),
                "timestamp": now.timestamp(),
            }

        if params.info_type in ("platform", "all"):
            info["platform"] = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "node": platform.node(),
            }

        if params.info_type in ("python", "all"):
            info["python"] = {
                "version": sys.version,
                "version_info": {
                    "major": sys.version_info.major,
                    "minor": sys.version_info.minor,
                    "micro": sys.version_info.micro,
                },
                "implementation": platform.python_implementation(),
                "compiler": platform.python_compiler(),
            }

        return ToolResult.success_result(
            data=info,
            info_type=params.info_type,
        )

    def get_confirmation_message(self, params: SystemInfoParams) -> str:
        """Get confirmation message.

        Args:
            params: Validated parameters

        Returns:
            str: Confirmation message
        """
        if params.info_type == "all":
            return "Get all system information"
        else:
            return f"Get system {params.info_type} information"
