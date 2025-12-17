"""Tests for example tools."""

import pytest
import platform

from aria.tools.examples.echo import EchoTool, EchoParams
from aria.tools.examples.system_info import SystemInfoTool, SystemInfoParams
from aria.tools.base import RiskLevel


class TestEchoTool:
    """Test EchoTool."""

    def test_tool_properties(self):
        """Test tool properties are set correctly."""
        tool = EchoTool()

        assert tool.name == "echo"
        assert tool.description is not None
        assert tool.risk_level == RiskLevel.LOW
        assert not tool.requires_confirmation

    @pytest.mark.asyncio
    async def test_simple_echo(self):
        """Test simple echo without transformations."""
        tool = EchoTool()
        params = EchoParams(message="Hello, World!")

        result = await tool.execute(params)

        assert result.success is True
        assert result.data == "Hello, World!"
        assert result.metadata["original_message"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_echo_uppercase(self):
        """Test echo with uppercase transformation."""
        tool = EchoTool()
        params = EchoParams(message="hello", uppercase=True)

        result = await tool.execute(params)

        assert result.success is True
        assert result.data == "HELLO"

    @pytest.mark.asyncio
    async def test_echo_repeat(self):
        """Test echo with repetition."""
        tool = EchoTool()
        params = EchoParams(message="test", repeat=3)

        result = await tool.execute(params)

        assert result.success is True
        assert result.data == "test test test"

    @pytest.mark.asyncio
    async def test_echo_uppercase_and_repeat(self):
        """Test echo with both transformations."""
        tool = EchoTool()
        params = EchoParams(message="hi", uppercase=True, repeat=2)

        result = await tool.execute(params)

        assert result.success is True
        assert result.data == "HI HI"

    def test_params_validation(self):
        """Test parameter validation."""
        tool = EchoTool()

        # Valid params
        params = tool.validate_params({"message": "test"})
        assert params.message == "test"

        # Invalid repeat value (too high)
        with pytest.raises(Exception):  # ValidationError
            tool.validate_params({"message": "test", "repeat": 100})

    @pytest.mark.asyncio
    async def test_run_method(self):
        """Test run method with raw parameters."""
        tool = EchoTool()

        result = await tool.run({"message": "test", "uppercase": True})

        assert result.success is True
        assert result.data == "TEST"
        assert "execution_time" in result.metadata

    def test_confirmation_message(self):
        """Test get_confirmation_message."""
        tool = EchoTool()
        params = EchoParams(message="test")

        msg = tool.get_confirmation_message(params)

        assert "test" in msg

    def test_to_ollama_tool(self):
        """Test conversion to Ollama tool definition."""
        tool = EchoTool()
        ollama_def = tool.to_ollama_tool()

        assert ollama_def.function["name"] == "echo"
        assert "message" in ollama_def.function["parameters"]["properties"]


class TestSystemInfoTool:
    """Test SystemInfoTool."""

    def test_tool_properties(self):
        """Test tool properties are set correctly."""
        tool = SystemInfoTool()

        assert tool.name == "system_info"
        assert tool.description is not None
        assert tool.risk_level == RiskLevel.LOW
        assert not tool.requires_confirmation

    @pytest.mark.asyncio
    async def test_get_time_info(self):
        """Test getting time information."""
        tool = SystemInfoTool()
        params = SystemInfoParams(info_type="time")

        result = await tool.execute(params)

        assert result.success is True
        assert "time" in result.data
        assert "utc" in result.data["time"]
        assert "local" in result.data["time"]
        assert "timestamp" in result.data["time"]

    @pytest.mark.asyncio
    async def test_get_platform_info(self):
        """Test getting platform information."""
        tool = SystemInfoTool()
        params = SystemInfoParams(info_type="platform")

        result = await tool.execute(params)

        assert result.success is True
        assert "platform" in result.data
        assert "system" in result.data["platform"]
        assert result.data["platform"]["system"] == platform.system()

    @pytest.mark.asyncio
    async def test_get_python_info(self):
        """Test getting Python information."""
        tool = SystemInfoTool()
        params = SystemInfoParams(info_type="python")

        result = await tool.execute(params)

        assert result.success is True
        assert "python" in result.data
        assert "version" in result.data["python"]
        assert "version_info" in result.data["python"]

    @pytest.mark.asyncio
    async def test_get_all_info(self):
        """Test getting all information."""
        tool = SystemInfoTool()
        params = SystemInfoParams(info_type="all")

        result = await tool.execute(params)

        assert result.success is True
        assert "time" in result.data
        assert "platform" in result.data
        assert "python" in result.data

    def test_params_validation(self):
        """Test parameter validation."""
        tool = SystemInfoTool()

        # Valid params
        params = tool.validate_params({"info_type": "time"})
        assert params.info_type == "time"

        # Invalid info_type
        with pytest.raises(Exception):  # ValidationError
            tool.validate_params({"info_type": "invalid"})

    @pytest.mark.asyncio
    async def test_run_method(self):
        """Test run method with raw parameters."""
        tool = SystemInfoTool()

        result = await tool.run({"info_type": "time"})

        assert result.success is True
        assert "time" in result.data
        assert "execution_time" in result.metadata

    def test_confirmation_message(self):
        """Test get_confirmation_message."""
        tool = SystemInfoTool()
        params = SystemInfoParams(info_type="platform")

        msg = tool.get_confirmation_message(params)

        assert "platform" in msg

    def test_to_ollama_tool(self):
        """Test conversion to Ollama tool definition."""
        tool = SystemInfoTool()
        ollama_def = tool.to_ollama_tool()

        assert ollama_def.function["name"] == "system_info"
        assert "info_type" in ollama_def.function["parameters"]["properties"]
