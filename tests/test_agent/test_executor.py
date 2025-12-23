"""Tests for tool executor."""

import pytest
from pydantic import BaseModel

from aria.agent.executor import ToolExecutor
from aria.tools import ToolRegistry, BaseTool, ToolResult, RiskLevel
from aria.approval import ApprovalHandler
from aria.ui.console import ARIAConsole


class SimpleParams(BaseModel):
    """Simple params for testing."""

    value: str


class SimpleTool(BaseTool):
    """Simple test tool."""

    name = "simple"
    description = "A simple test tool"
    risk_level = RiskLevel.LOW
    parameters_schema = SimpleParams

    async def execute(self, params: SimpleParams) -> ToolResult:
        return ToolResult.success_result(data=f"Got: {params.value}")

    def get_confirmation_message(self, params: SimpleParams) -> str:
        return f"Run with {params.value}"


class TestToolExecutor:
    """Test ToolExecutor class."""

    @pytest.mark.asyncio
    async def test_execute_low_risk_tool(self):
        """Test executing a low risk tool (no approval needed)."""
        # Setup
        registry = ToolRegistry()
        registry.register(SimpleTool())
        console = ARIAConsole()
        approval_handler = ApprovalHandler(console)
        executor = ToolExecutor(registry, approval_handler, console)

        # Execute
        result = await executor.execute("simple", {"value": "test"})

        # Verify
        assert result.success is True
        assert "Got: test" in result.data

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist."""
        # Setup
        registry = ToolRegistry()
        console = ARIAConsole()
        approval_handler = ApprovalHandler(console)
        executor = ToolExecutor(registry, approval_handler, console)

        # Execute
        result = await executor.execute("nonexistent", {})

        # Verify
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_params(self):
        """Test executing with invalid parameters."""
        # Setup
        registry = ToolRegistry()
        registry.register(SimpleTool())
        console = ARIAConsole()
        approval_handler = ApprovalHandler(console)
        executor = ToolExecutor(registry, approval_handler, console)

        # Execute with missing required param
        result = await executor.execute("simple", {})

        # Verify
        assert result.success is False
        assert "invalid" in result.error.lower() or "required" in result.error.lower()
