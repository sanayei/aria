"""Tool execution with approval handling for the ARIA agent.

This module provides the ToolExecutor class which handles executing tools
with the appropriate approval flows based on risk level.
"""

import time
from typing import Any

from aria.tools.base import BaseTool, ToolResult
from aria.tools.registry import ToolRegistry
from aria.approval import ApprovalHandler, ActionClassifier
from aria.ui.console import ARIAConsole
from aria.logging import get_logger, AsyncTimer

logger = get_logger("aria.agent.executor")


class ToolExecutionError(Exception):
    """Exception raised when tool execution fails."""

    pass


class ToolExecutor:
    """Executes tools with risk-based approval handling.

    The executor manages the complete lifecycle of tool execution:
    1. Tool lookup and validation
    2. Parameter validation
    3. Risk assessment and approval (if needed)
    4. Execution
    5. Result formatting and display
    """

    def __init__(
        self,
        registry: ToolRegistry,
        approval_handler: ApprovalHandler,
        console: ARIAConsole,
    ):
        """Initialize the tool executor.

        Args:
            registry: Tool registry for looking up tools
            approval_handler: Handler for user approvals
            console: Console for displaying execution progress
        """
        self.registry = registry
        self.approval_handler = approval_handler
        self.console = console
        self.classifier = ActionClassifier()

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool with approval handling.

        Steps:
        1. Get tool from registry
        2. Validate parameters
        3. Check if approval needed
        4. Request approval if needed
        5. Execute tool
        6. Display results

        Args:
            tool_name: Name of the tool to execute
            arguments: Raw arguments for the tool

        Returns:
            ToolResult: Result of the execution

        Raises:
            ToolExecutionError: If tool execution fails
        """
        exec_start = time.perf_counter()
        logger.info(
            "ToolExecutor.execute() started",
            tool_name=tool_name,
            arg_count=len(arguments),
        )

        # Step 1: Get tool from registry
        tool = self.registry.get(tool_name)
        if tool is None:
            error_msg = f"Tool '{tool_name}' not found in registry"
            logger.error(error_msg, available_tools=self.registry.get_tool_names())
            self.console.error(error_msg)
            return ToolResult.error_result(error=error_msg)

        try:
            # Step 2: Validate parameters
            try:
                logger.debug(f"Validating parameters for tool: {tool_name}")
                params = tool.validate_params(arguments)
                logger.debug(
                    f"Parameters validated successfully",
                    tool_name=tool_name,
                    param_keys=list(params.keys()) if isinstance(params, dict) else "object",
                )
            except Exception as e:
                error_msg = f"Invalid parameters for tool '{tool_name}': {e}"
                logger.error(error_msg, exception=str(e))
                self.console.error(error_msg)
                return ToolResult.error_result(error=error_msg)

            # Display tool call
            self.console.tool_call(tool_name, arguments)

            # Step 3 & 4: Check approval and request if needed
            approval_start = time.perf_counter()
            if self.classifier.should_require_double_confirmation(tool, params):
                # CRITICAL - double confirmation
                logger.info(
                    f"Requesting double confirmation for tool: {tool_name}",
                    risk_level="CRITICAL",
                )
                approval_result = await self.approval_handler.request_double_confirmation(
                    tool, params
                )
            elif self.classifier.should_require_confirmation(tool, params):
                # MEDIUM/HIGH - single confirmation
                logger.info(
                    f"Requesting confirmation for tool: {tool_name}",
                    risk_level=tool.risk_level.value,
                )
                approval_result = await self.approval_handler.request_approval(
                    tool, params
                )
            else:
                # LOW - auto-approve
                logger.debug(
                    f"Auto-approving low-risk tool: {tool_name}",
                    risk_level=tool.risk_level.value,
                )
                from aria.approval.handler import ApprovalResult
                approval_result = ApprovalResult.approve()

            approval_elapsed = time.perf_counter() - approval_start
            logger.debug(
                f"Approval check complete",
                tool_name=tool_name,
                approved=approval_result.approved,
                approval_time_s=f"{approval_elapsed:.3f}",
            )

            # Check if approved
            if not approval_result.approved:
                denial_msg = f"User denied tool execution: {approval_result.reason}"
                logger.info(f"Tool '{tool_name}' denied: {approval_result.reason}")
                self.console.warning(denial_msg)
                return ToolResult.error_result(error=denial_msg)

            # Use modified params if provided
            final_params = approval_result.modified_params or params

            # Step 5: Execute the tool
            logger.info(f"Executing tool: {tool_name}")
            async with AsyncTimer(f"Tool.run({tool_name})", logger):
                result = await tool.run(arguments, track_time=True)

            # Step 6: Display results
            if result.success:
                data_preview = str(result.data)[:200] if result.data else ""
                logger.info(
                    f"Tool '{tool_name}' executed successfully",
                    data_length=len(str(result.data)) if result.data else 0,
                    data_preview=data_preview,
                )
                self.console.tool_result(tool_name, str(result.data)[:500], error=False)
            else:
                logger.error(
                    f"Tool '{tool_name}' execution failed",
                    error=result.error,
                )
                self.console.tool_result(tool_name, result.error or "Unknown error", error=True)

            exec_elapsed = time.perf_counter() - exec_start
            logger.info(
                "ToolExecutor.execute() complete",
                tool_name=tool_name,
                success=result.success,
                total_time_s=f"{exec_elapsed:.3f}",
            )

            return result

        except Exception as e:
            exec_elapsed = time.perf_counter() - exec_start
            error_msg = f"Unexpected error executing tool '{tool_name}': {e}"
            logger.exception(
                error_msg,
                tool_name=tool_name,
                total_time_s=f"{exec_elapsed:.3f}",
            )
            self.console.error(error_msg, exception=e)
            return ToolResult.error_result(error=error_msg)

    async def execute_multiple(
        self,
        tool_calls: list[tuple[str, dict[str, Any]]],
    ) -> list[ToolResult]:
        """Execute multiple tools in sequence.

        Args:
            tool_calls: List of (tool_name, arguments) tuples

        Returns:
            list[ToolResult]: Results from each tool execution
        """
        results = []

        for tool_name, arguments in tool_calls:
            result = await self.execute(tool_name, arguments)
            results.append(result)

            # Stop if a tool fails critically
            if not result.success and result.error and "denied" in result.error.lower():
                logger.warning(
                    f"Stopping tool execution chain due to denial of '{tool_name}'"
                )
                break

        return results

    def get_execution_summary(self, results: list[ToolResult]) -> str:
        """Generate a summary of multiple tool executions.

        Args:
            results: List of tool results

        Returns:
            str: Human-readable summary
        """
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful

        summary_lines = [
            f"Executed {total} tool(s):",
            f"  ✓ {successful} succeeded",
        ]

        if failed > 0:
            summary_lines.append(f"  ✗ {failed} failed")

        return "\n".join(summary_lines)
