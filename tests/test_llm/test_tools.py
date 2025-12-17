"""Tests for LLM tool utilities."""

import pytest
from pydantic import BaseModel, Field

from aria.llm.tools import (
    pydantic_to_json_schema,
    create_tool_definition,
    create_simple_tool,
    function_to_tool_definition,
    ToolRegistry,
)


class WeatherInput(BaseModel):
    """Test input model for weather tool."""

    location: str = Field(..., description="City name")
    units: str = Field(default="celsius", description="Temperature units")


class TestPydanticToJsonSchema:
    """Test Pydantic to JSON Schema conversion."""

    def test_basic_conversion(self):
        """Test converting a Pydantic model to JSON Schema."""
        schema = pydantic_to_json_schema(WeatherInput)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "location" in schema["properties"]
        assert "units" in schema["properties"]
        assert "required" in schema
        assert "location" in schema["required"]

    def test_exclude_fields(self):
        """Test excluding fields from schema."""
        schema = pydantic_to_json_schema(WeatherInput, exclude_fields={"units"})

        assert "location" in schema["properties"]
        assert "units" not in schema["properties"]


class TestCreateToolDefinition:
    """Test tool definition creation."""

    def test_from_pydantic_model(self):
        """Test creating tool definition from Pydantic model."""
        tool_def = create_tool_definition(
            name="get_weather",
            description="Get weather for a location",
            parameters_model=WeatherInput,
        )

        assert tool_def.type == "function"
        assert tool_def.function["name"] == "get_weather"
        assert tool_def.function["description"] == "Get weather for a location"
        assert "parameters" in tool_def.function
        assert tool_def.function["parameters"]["type"] == "object"

    def test_from_schema_dict(self):
        """Test creating tool definition from schema dict."""
        schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
        }

        tool_def = create_tool_definition(
            name="get_weather",
            description="Get weather",
            parameters_schema=schema,
        )

        assert tool_def.function["name"] == "get_weather"
        assert tool_def.function["parameters"] == schema

    def test_no_parameters(self):
        """Test creating tool with no parameters."""
        tool_def = create_tool_definition(
            name="no_params",
            description="A tool with no parameters",
        )

        assert tool_def.function["name"] == "no_params"
        assert tool_def.function["parameters"]["type"] == "object"
        assert tool_def.function["parameters"]["properties"] == {}


class TestCreateSimpleTool:
    """Test simple tool creation."""

    def test_basic_tool(self):
        """Test creating a basic tool."""
        tool = create_simple_tool(
            name="get_weather",
            description="Get weather",
            properties={
                "location": {
                    "type": "string",
                    "description": "City name",
                },
            },
            required=["location"],
        )

        assert tool.function["name"] == "get_weather"
        assert tool.function["description"] == "Get weather"
        assert "location" in tool.function["parameters"]["properties"]
        assert tool.function["parameters"]["required"] == ["location"]

    def test_tool_with_enum(self):
        """Test creating a tool with enum parameter."""
        tool = create_simple_tool(
            name="set_mode",
            description="Set mode",
            properties={
                "mode": {
                    "type": "string",
                    "enum": ["light", "dark"],
                    "description": "UI mode",
                },
            },
        )

        mode_prop = tool.function["parameters"]["properties"]["mode"]
        assert mode_prop["type"] == "string"
        assert mode_prop["enum"] == ["light", "dark"]


class TestFunctionToToolDefinition:
    """Test function to tool definition conversion."""

    def test_from_function(self):
        """Test creating tool definition from function."""

        def get_weather(location: str, units: str = "celsius") -> str:
            """Get the current weather for a location."""
            return f"Weather in {location}: sunny, 22{units[0].upper()}"

        tool_def = function_to_tool_definition(get_weather, WeatherInput)

        assert tool_def.function["name"] == "get_weather"
        assert "weather" in tool_def.function["description"].lower()

    def test_custom_description(self):
        """Test overriding function description."""

        def my_func():
            """Original docstring."""
            pass

        tool_def = function_to_tool_definition(
            my_func,
            description="Custom description",
        )

        assert tool_def.function["description"] == "Custom description"


class TestToolRegistry:
    """Test ToolRegistry class."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()

        tool = create_simple_tool(
            name="test_tool",
            description="A test tool",
            properties={},
        )

        registry.register(tool)

        assert len(registry) == 1
        assert "test_tool" in registry
        assert registry.get("test_tool") == tool

    def test_register_from_function(self):
        """Test registering a tool from a function."""
        registry = ToolRegistry()

        def my_tool(arg: str) -> str:
            """My tool description."""
            return arg

        class MyToolInput(BaseModel):
            arg: str

        registry.register_from_function(my_tool, MyToolInput)

        assert len(registry) == 1
        assert "my_tool" in registry

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()

        tool = create_simple_tool(
            name="test_tool",
            description="Test",
            properties={},
        )

        registry.register(tool)
        assert len(registry) == 1

        registry.unregister("test_tool")
        assert len(registry) == 0
        assert "test_tool" not in registry

    def test_get_all_tools(self):
        """Test getting all tools."""
        registry = ToolRegistry()

        tool1 = create_simple_tool(name="tool1", description="Tool 1", properties={})
        tool2 = create_simple_tool(name="tool2", description="Tool 2", properties={})

        registry.register(tool1)
        registry.register(tool2)

        all_tools = registry.get_all()
        assert len(all_tools) == 2

    def test_get_names(self):
        """Test getting tool names."""
        registry = ToolRegistry()

        tool1 = create_simple_tool(name="tool1", description="Tool 1", properties={})
        tool2 = create_simple_tool(name="tool2", description="Tool 2", properties={})

        registry.register(tool1)
        registry.register(tool2)

        names = registry.get_names()
        assert set(names) == {"tool1", "tool2"}

    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = ToolRegistry()

        tool = create_simple_tool(name="test", description="Test", properties={})
        registry.register(tool)

        assert len(registry) == 1

        registry.clear()
        assert len(registry) == 0

    def test_repr(self):
        """Test string representation."""
        registry = ToolRegistry()

        tool = create_simple_tool(name="test", description="Test", properties={})
        registry.register(tool)

        repr_str = repr(registry)
        assert "ToolRegistry" in repr_str
        assert "1 tools" in repr_str
        assert "test" in repr_str
