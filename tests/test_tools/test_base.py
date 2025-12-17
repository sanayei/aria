"""Tests for tool base classes."""

import pytest
from pydantic import BaseModel, Field, ValidationError

from aria.tools.base import (
    RiskLevel,
    ToolResult,
    BaseTool,
    create_params_schema,
)


class TestRiskLevel:
    """Test RiskLevel enum."""

    def test_risk_levels(self):
        """Test all risk levels exist."""
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MEDIUM == "medium"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.CRITICAL == "critical"

    def test_requires_confirmation(self):
        """Test requires_confirmation property."""
        assert not RiskLevel.LOW.requires_confirmation
        assert RiskLevel.MEDIUM.requires_confirmation
        assert RiskLevel.HIGH.requires_confirmation
        assert RiskLevel.CRITICAL.requires_confirmation

    def test_requires_double_confirmation(self):
        """Test requires_double_confirmation property."""
        assert not RiskLevel.LOW.requires_double_confirmation
        assert not RiskLevel.MEDIUM.requires_double_confirmation
        assert not RiskLevel.HIGH.requires_double_confirmation
        assert RiskLevel.CRITICAL.requires_double_confirmation

    def test_string_representation(self):
        """Test string conversion."""
        assert str(RiskLevel.LOW) == "low"
        assert str(RiskLevel.MEDIUM) == "medium"


class TestToolResult:
    """Test ToolResult model."""

    def test_success_result(self):
        """Test creating a success result."""
        result = ToolResult.success_result(
            data="test data",
            extra_info="metadata",
        )

        assert result.success is True
        assert result.data == "test data"
        assert result.error is None
        assert result.metadata["extra_info"] == "metadata"

    def test_error_result(self):
        """Test creating an error result."""
        result = ToolResult.error_result(
            error="Something went wrong",
            error_code=500,
        )

        assert result.success is False
        assert result.data is None
        assert result.error == "Something went wrong"
        assert result.metadata["error_code"] == 500

    def test_string_representation(self):
        """Test string conversion."""
        success = ToolResult.success_result(data="test")
        assert "Success" in str(success)

        error = ToolResult.error_result(error="failed")
        assert "Error" in str(error)


class TestBaseTool:
    """Test BaseTool abstract class."""

    def test_missing_attributes(self):
        """Test that tools without required attributes raise errors."""

        class IncompleteTool(BaseTool):
            # Missing required attributes, but has abstract methods
            async def execute(self, params):
                pass

            def get_confirmation_message(self, params):
                return "test"

        with pytest.raises(ValueError, match="must define"):
            IncompleteTool()

    def test_invalid_parameters_schema(self):
        """Test that invalid parameters_schema raises error."""

        class BadTool(BaseTool):
            name = "bad"
            description = "Bad tool"
            risk_level = RiskLevel.LOW
            parameters_schema = str  # Not a BaseModel!

            async def execute(self, params):
                pass

            def get_confirmation_message(self, params):
                return "test"

        with pytest.raises(ValueError, match="Pydantic BaseModel"):
            BadTool()

    def test_valid_tool_initialization(self):
        """Test creating a valid tool."""

        class TestParams(BaseModel):
            value: str

        class ValidTool(BaseTool):
            name = "valid"
            description = "Valid tool"
            risk_level = RiskLevel.LOW
            parameters_schema = TestParams

            async def execute(self, params):
                return ToolResult.success_result(data=params.value)

            def get_confirmation_message(self, params):
                return f"Execute with {params.value}"

        tool = ValidTool()
        assert tool.name == "valid"
        assert tool.risk_level == RiskLevel.LOW
        assert not tool.requires_confirmation

    def test_requires_confirmation_property(self):
        """Test requires_confirmation property."""

        class TestParams(BaseModel):
            value: str

        class LowRiskTool(BaseTool):
            name = "low"
            description = "Low risk"
            risk_level = RiskLevel.LOW
            parameters_schema = TestParams

            async def execute(self, params):
                pass

            def get_confirmation_message(self, params):
                return "test"

        class HighRiskTool(BaseTool):
            name = "high"
            description = "High risk"
            risk_level = RiskLevel.HIGH
            parameters_schema = TestParams

            async def execute(self, params):
                pass

            def get_confirmation_message(self, params):
                return "test"

        low_tool = LowRiskTool()
        high_tool = HighRiskTool()

        assert not low_tool.requires_confirmation
        assert high_tool.requires_confirmation

    def test_validate_params_success(self):
        """Test successful parameter validation."""

        class TestParams(BaseModel):
            value: str
            count: int = 1

        class TestTool(BaseTool):
            name = "test"
            description = "Test"
            risk_level = RiskLevel.LOW
            parameters_schema = TestParams

            async def execute(self, params):
                pass

            def get_confirmation_message(self, params):
                return "test"

        tool = TestTool()
        params = tool.validate_params({"value": "test", "count": 5})

        assert params.value == "test"
        assert params.count == 5

    def test_validate_params_failure(self):
        """Test parameter validation failure."""

        class TestParams(BaseModel):
            value: str

        class TestTool(BaseTool):
            name = "test"
            description = "Test"
            risk_level = RiskLevel.LOW
            parameters_schema = TestParams

            async def execute(self, params):
                pass

            def get_confirmation_message(self, params):
                return "test"

        tool = TestTool()

        with pytest.raises(ValidationError):
            tool.validate_params({"wrong_field": "value"})

    def test_to_ollama_tool(self):
        """Test converting to Ollama tool definition."""

        class TestParams(BaseModel):
            message: str = Field(..., description="Test message")

        class TestTool(BaseTool):
            name = "test_tool"
            description = "A test tool"
            risk_level = RiskLevel.LOW
            parameters_schema = TestParams

            async def execute(self, params):
                pass

            def get_confirmation_message(self, params):
                return "test"

        tool = TestTool()
        ollama_def = tool.to_ollama_tool()

        assert ollama_def.type == "function"
        assert ollama_def.function["name"] == "test_tool"
        assert ollama_def.function["description"] == "A test tool"
        assert "parameters" in ollama_def.function

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Test successful tool run."""

        class TestParams(BaseModel):
            value: str

        class TestTool(BaseTool):
            name = "test"
            description = "Test"
            risk_level = RiskLevel.LOW
            parameters_schema = TestParams

            async def execute(self, params):
                return ToolResult.success_result(data=f"processed: {params.value}")

            def get_confirmation_message(self, params):
                return "test"

        tool = TestTool()
        result = await tool.run({"value": "test"})

        assert result.success is True
        assert result.data == "processed: test"
        assert "execution_time" in result.metadata

    @pytest.mark.asyncio
    async def test_run_validation_error(self):
        """Test tool run with invalid parameters."""

        class TestParams(BaseModel):
            value: str

        class TestTool(BaseTool):
            name = "test"
            description = "Test"
            risk_level = RiskLevel.LOW
            parameters_schema = TestParams

            async def execute(self, params):
                return ToolResult.success_result(data="success")

            def get_confirmation_message(self, params):
                return "test"

        tool = TestTool()
        result = await tool.run({"wrong_field": "value"})

        assert result.success is False
        assert "validation" in result.error.lower()

    @pytest.mark.asyncio
    async def test_run_execution_error(self):
        """Test tool run with execution error."""

        class TestParams(BaseModel):
            value: str

        class TestTool(BaseTool):
            name = "test"
            description = "Test"
            risk_level = RiskLevel.LOW
            parameters_schema = TestParams

            async def execute(self, params):
                raise RuntimeError("Execution failed")

            def get_confirmation_message(self, params):
                return "test"

        tool = TestTool()
        result = await tool.run({"value": "test"})

        assert result.success is False
        assert "RuntimeError" in result.error

    def test_repr(self):
        """Test string representation."""

        class TestParams(BaseModel):
            value: str

        class TestTool(BaseTool):
            name = "test_tool"
            description = "Test"
            risk_level = RiskLevel.LOW
            parameters_schema = TestParams

            async def execute(self, params):
                pass

            def get_confirmation_message(self, params):
                return "test"

        tool = TestTool()
        repr_str = repr(tool)

        assert "TestTool" in repr_str
        assert "test_tool" in repr_str
        assert "low" in repr_str


class TestCreateParamsSchema:
    """Test create_params_schema helper."""

    def test_create_simple_schema(self):
        """Test creating a simple schema."""
        ParamsSchema = create_params_schema(
            "TestParams",
            {"value": (str, Field(..., description="Test value"))},
        )

        assert issubclass(ParamsSchema, BaseModel)
        instance = ParamsSchema(value="test")
        assert instance.value == "test"
