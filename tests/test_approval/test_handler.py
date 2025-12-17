"""Tests for approval handler."""

import pytest
from pydantic import BaseModel

from aria.approval.handler import ApprovalResult, ApprovalHandler
from aria.approval.classifier import ActionClassifier
from aria.tools.base import BaseTool, ToolResult, RiskLevel
from aria.ui.console import ARIAConsole


class DummyParams(BaseModel):
    """Dummy parameters for testing."""

    value: str


class TestTool(BaseTool):
    """Test tool for approval handler tests."""

    name = "test_tool"
    description = "A test tool"
    risk_level = RiskLevel.MEDIUM
    parameters_schema = DummyParams

    async def execute(self, params):
        return ToolResult.success_result(data=params.value)

    def get_confirmation_message(self, params):
        return f"Execute test with {params.value}"


class TestApprovalResult:
    """Test ApprovalResult class."""

    def test_approve_result(self):
        """Test creating an approved result."""
        result = ApprovalResult.approve()

        assert result.approved is True
        assert result.modified_params is None
        assert result.reason is None

    def test_approve_with_modified_params(self):
        """Test approving with modified parameters."""
        params = DummyParams(value="modified")
        result = ApprovalResult.approve(modified_params=params)

        assert result.approved is True
        assert result.modified_params == params

    def test_deny_result(self):
        """Test creating a denied result."""
        result = ApprovalResult.deny("User declined")

        assert result.approved is False
        assert result.reason == "User declined"

    def test_deny_default_reason(self):
        """Test deny with default reason."""
        result = ApprovalResult.deny()

        assert result.approved is False
        assert result.reason == "User denied"


class TestApprovalHandler:
    """Test ApprovalHandler class."""

    def test_initialization(self):
        """Test handler initialization."""
        console = ARIAConsole()
        handler = ApprovalHandler(console)

        assert handler.console == console
        assert isinstance(handler.classifier, ActionClassifier)

    def test_initialization_with_classifier(self):
        """Test handler initialization with custom classifier."""
        console = ARIAConsole()
        classifier = ActionClassifier()
        handler = ApprovalHandler(console, classifier)

        assert handler.console == console
        assert handler.classifier == classifier

    def test_format_approval_prompt(self):
        """Test formatting approval prompt."""
        console = ARIAConsole()
        handler = ApprovalHandler(console)
        tool = TestTool()
        params = DummyParams(value="test")

        panel = handler.format_approval_prompt(
            tool=tool,
            params=params,
            risk_level=RiskLevel.MEDIUM,
            risk_factors=["Test factor 1", "Test factor 2"],
        )

        # Check that panel is created (we can't easily test rich Panel content)
        assert panel is not None
        assert panel.title is not None

    def test_format_approval_prompt_critical(self):
        """Test formatting approval prompt for critical action."""
        console = ARIAConsole()
        handler = ApprovalHandler(console)
        tool = TestTool()
        tool.risk_level = RiskLevel.CRITICAL
        params = DummyParams(value="test")

        panel = handler.format_approval_prompt(
            tool=tool,
            params=params,
            risk_level=RiskLevel.CRITICAL,
            risk_factors=["Dangerous operation"],
            critical_warning=True,
        )

        assert panel is not None

    def test_format_approval_prompt_without_factors(self):
        """Test formatting approval prompt without risk factors."""
        console = ARIAConsole()
        handler = ApprovalHandler(console)
        tool = TestTool()
        params = DummyParams(value="test")

        panel = handler.format_approval_prompt(
            tool=tool,
            params=params,
            risk_level=RiskLevel.LOW,
            risk_factors=None,
        )

        assert panel is not None

    def test_format_tool_summary(self):
        """Test formatting tool summary."""
        console = ARIAConsole()
        handler = ApprovalHandler(console)
        tool = TestTool()
        params = DummyParams(value="test value")

        summary = handler.format_tool_summary(tool, params)

        assert "test_tool" in summary
        assert "test value" in summary
        assert "medium" in summary.lower()

    def test_format_tool_summary_long_params(self):
        """Test formatting tool summary with long parameters."""
        console = ARIAConsole()
        handler = ApprovalHandler(console)
        tool = TestTool()
        long_value = "x" * 200
        params = DummyParams(value=long_value)

        summary = handler.format_tool_summary(tool, params)

        # Long values should be truncated
        assert "..." in summary
        assert len(summary) < len(long_value) + 100  # Some overhead for formatting
