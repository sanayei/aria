"""Echo tool - Simple tool for testing.

This tool simply echoes back the input message. It's useful for testing
the tool execution framework without side effects.
"""

from pydantic import BaseModel, Field

from aria.tools.base import BaseTool, ToolResult, RiskLevel


class EchoParams(BaseModel):
    """Parameters for the echo tool."""

    message: str = Field(..., description="Message to echo back")
    uppercase: bool = Field(
        default=False,
        description="Whether to convert the message to uppercase",
    )
    repeat: int = Field(
        default=1,
        description="Number of times to repeat the message",
        ge=1,
        le=10,
    )


class EchoTool(BaseTool[EchoParams]):
    """Echo tool that returns the input message.

    This is a simple, safe tool with no side effects, perfect for
    testing the tool execution framework.
    """

    name = "echo"
    description = "Echo back a message, optionally in uppercase and repeated"
    risk_level = RiskLevel.LOW
    parameters_schema = EchoParams

    async def execute(self, params: EchoParams) -> ToolResult:
        """Execute the echo operation.

        Args:
            params: Validated echo parameters

        Returns:
            ToolResult: Result containing the echoed message
        """
        message = params.message

        # Apply transformations
        if params.uppercase:
            message = message.upper()

        if params.repeat > 1:
            message = " ".join([message] * params.repeat)

        return ToolResult.success_result(
            data=message,
            original_message=params.message,
            transformations={
                "uppercase": params.uppercase,
                "repeat": params.repeat,
            },
        )

    def get_confirmation_message(self, params: EchoParams) -> str:
        """Get confirmation message.

        Echo tool is LOW risk and doesn't require confirmation,
        but we implement this for completeness.

        Args:
            params: Validated parameters

        Returns:
            str: Confirmation message
        """
        return f"Echo the message: '{params.message}'"
