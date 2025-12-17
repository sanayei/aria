"""Tests for action classifier."""

import pytest
from pydantic import BaseModel, Field

from aria.approval.classifier import ActionClassifier
from aria.tools.base import BaseTool, ToolResult, RiskLevel


class DummyParams(BaseModel):
    """Dummy parameters for testing."""

    value: str
    count: int = 1


class LowRiskTool(BaseTool):
    """Low risk test tool."""

    name = "low_risk"
    description = "A low risk tool"
    risk_level = RiskLevel.LOW
    parameters_schema = DummyParams

    async def execute(self, params):
        return ToolResult.success_result(data=params.value)

    def get_confirmation_message(self, params):
        return f"Execute with {params.value}"


class MediumRiskTool(BaseTool):
    """Medium risk test tool."""

    name = "medium_risk"
    description = "A medium risk tool"
    risk_level = RiskLevel.MEDIUM
    parameters_schema = DummyParams

    async def execute(self, params):
        return ToolResult.success_result(data=params.value)

    def get_confirmation_message(self, params):
        return f"Execute with {params.value}"


class HighRiskTool(BaseTool):
    """High risk test tool."""

    name = "high_risk"
    description = "A high risk tool"
    risk_level = RiskLevel.HIGH
    parameters_schema = DummyParams

    async def execute(self, params):
        return ToolResult.success_result(data=params.value)

    def get_confirmation_message(self, params):
        return f"Execute with {params.value}"


class CriticalRiskTool(BaseTool):
    """Critical risk test tool."""

    name = "critical_risk"
    description = "A critical risk tool"
    risk_level = RiskLevel.CRITICAL
    parameters_schema = DummyParams

    async def execute(self, params):
        return ToolResult.success_result(data=params.value)

    def get_confirmation_message(self, params):
        return f"Execute with {params.value}"


class TestActionClassifier:
    """Test ActionClassifier class."""

    def test_classify_low_risk(self):
        """Test classifying low risk tool."""
        classifier = ActionClassifier()
        tool = LowRiskTool()
        params = DummyParams(value="test")

        risk = classifier.classify_tool(tool, params)

        assert risk == RiskLevel.LOW

    def test_classify_medium_risk(self):
        """Test classifying medium risk tool."""
        classifier = ActionClassifier()
        tool = MediumRiskTool()
        params = DummyParams(value="test")

        risk = classifier.classify_tool(tool, params)

        assert risk == RiskLevel.MEDIUM

    def test_classify_high_risk(self):
        """Test classifying high risk tool."""
        classifier = ActionClassifier()
        tool = HighRiskTool()
        params = DummyParams(value="test")

        risk = classifier.classify_tool(tool, params)

        assert risk == RiskLevel.HIGH

    def test_classify_critical_risk(self):
        """Test classifying critical risk tool."""
        classifier = ActionClassifier()
        tool = CriticalRiskTool()
        params = DummyParams(value="test")

        risk = classifier.classify_tool(tool, params)

        assert risk == RiskLevel.CRITICAL

    def test_get_risk_factors_low(self):
        """Test getting risk factors for low risk tool."""
        classifier = ActionClassifier()
        tool = LowRiskTool()
        params = DummyParams(value="test")

        factors = classifier.get_risk_factors(tool, params)

        assert isinstance(factors, list)
        assert len(factors) > 0
        assert any("read-only" in f.lower() for f in factors)

    def test_get_risk_factors_critical(self):
        """Test getting risk factors for critical risk tool."""
        classifier = ActionClassifier()
        tool = CriticalRiskTool()
        params = DummyParams(value="test")

        factors = classifier.get_risk_factors(tool, params)

        assert isinstance(factors, list)
        assert len(factors) > 0
        assert any("dangerous" in f.lower() for f in factors)

    def test_should_require_confirmation_low(self):
        """Test that low risk doesn't require confirmation."""
        classifier = ActionClassifier()
        tool = LowRiskTool()
        params = DummyParams(value="test")

        requires = classifier.should_require_confirmation(tool, params)

        assert requires is False

    def test_should_require_confirmation_medium(self):
        """Test that medium risk requires confirmation."""
        classifier = ActionClassifier()
        tool = MediumRiskTool()
        params = DummyParams(value="test")

        requires = classifier.should_require_confirmation(tool, params)

        assert requires is True

    def test_should_require_double_confirmation_low(self):
        """Test that low risk doesn't require double confirmation."""
        classifier = ActionClassifier()
        tool = LowRiskTool()
        params = DummyParams(value="test")

        requires = classifier.should_require_double_confirmation(tool, params)

        assert requires is False

    def test_should_require_double_confirmation_critical(self):
        """Test that critical risk requires double confirmation."""
        classifier = ActionClassifier()
        tool = CriticalRiskTool()
        params = DummyParams(value="test")

        requires = classifier.should_require_double_confirmation(tool, params)

        assert requires is True

    def test_analyze_bulk_parameters(self):
        """Test parameter analysis detects bulk operations."""

        class BulkParams(BaseModel):
            value: str
            count: int

        class BulkTool(BaseTool):
            name = "bulk"
            description = "Bulk operation"
            risk_level = RiskLevel.LOW
            parameters_schema = BulkParams

            async def execute(self, params):
                return ToolResult.success_result(data=params.value)

            def get_confirmation_message(self, params):
                return f"Bulk operation on {params.count} items"

        classifier = ActionClassifier()
        tool = BulkTool()
        params = BulkParams(value="test", count=150)

        factors = classifier.get_risk_factors(tool, params)

        assert any("batch" in f.lower() or "bulk" in f.lower() for f in factors)

    def test_analyze_deletion_parameters(self):
        """Test parameter analysis detects deletion operations."""

        class DeleteParams(BaseModel):
            file_to_delete: str

        class DeleteTool(BaseTool):
            name = "delete"
            description = "Delete a file"
            risk_level = RiskLevel.HIGH
            parameters_schema = DeleteParams

            async def execute(self, params):
                return ToolResult.success_result(data="deleted")

            def get_confirmation_message(self, params):
                return f"Delete {params.file_to_delete}"

        classifier = ActionClassifier()
        tool = DeleteTool()
        params = DeleteParams(file_to_delete="/path/to/file.txt")

        factors = classifier.get_risk_factors(tool, params)

        assert any("deletion" in f.lower() or "cannot be undone" in f.lower() for f in factors)

    def test_get_risk_color(self):
        """Test getting risk colors."""
        classifier = ActionClassifier()

        assert classifier.get_risk_color(RiskLevel.LOW) == "green"
        assert classifier.get_risk_color(RiskLevel.MEDIUM) == "yellow"
        assert classifier.get_risk_color(RiskLevel.HIGH) == "red"
        assert "red" in classifier.get_risk_color(RiskLevel.CRITICAL)

    def test_get_risk_emoji(self):
        """Test getting risk emojis."""
        classifier = ActionClassifier()

        assert classifier.get_risk_emoji(RiskLevel.LOW) == "✓"
        assert classifier.get_risk_emoji(RiskLevel.MEDIUM) == "⚠️"
        assert classifier.get_risk_emoji(RiskLevel.HIGH) == "⚠️"
        assert classifier.get_risk_emoji(RiskLevel.CRITICAL) == "🚨"
