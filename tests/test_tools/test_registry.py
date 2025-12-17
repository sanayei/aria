"""Tests for tool registry."""

import pytest
from pydantic import BaseModel

from aria.tools.base import BaseTool, ToolResult, RiskLevel
from aria.tools.registry import ToolRegistry, get_registry, reset_registry


class DummyParams(BaseModel):
    """Dummy parameters for test tools."""

    value: str


class DummyTool(BaseTool):
    """Dummy tool for testing."""

    name = "dummy"
    description = "A dummy tool"
    risk_level = RiskLevel.LOW
    parameters_schema = DummyParams

    async def execute(self, params):
        return ToolResult.success_result(data=params.value)

    def get_confirmation_message(self, params):
        return f"Execute with {params.value}"


class AnotherDummyTool(BaseTool):
    """Another dummy tool for testing."""

    name = "another_dummy"
    description = "Another dummy tool"
    risk_level = RiskLevel.MEDIUM
    parameters_schema = DummyParams

    async def execute(self, params):
        return ToolResult.success_result(data=params.value)

    def get_confirmation_message(self, params):
        return f"Execute with {params.value}"


class TestToolRegistry:
    """Test ToolRegistry class."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = DummyTool()

        registry.register(tool)

        assert registry.has_tool("dummy")
        assert registry.count() == 1

    def test_register_duplicate(self):
        """Test that registering duplicate tool raises error."""
        registry = ToolRegistry()
        tool1 = DummyTool()
        tool2 = DummyTool()

        registry.register(tool1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool2)

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        tool = DummyTool()

        registry.register(tool)
        assert registry.has_tool("dummy")

        registry.unregister("dummy")
        assert not registry.has_tool("dummy")

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent tool raises error."""
        registry = ToolRegistry()

        with pytest.raises(KeyError):
            registry.unregister("nonexistent")

    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        tool = DummyTool()
        registry.register(tool)

        retrieved = registry.get("dummy")
        assert retrieved is not None
        assert retrieved.name == "dummy"

    def test_get_nonexistent(self):
        """Test getting nonexistent tool returns None."""
        registry = ToolRegistry()

        assert registry.get("nonexistent") is None

    def test_list_tools(self):
        """Test listing all tools."""
        registry = ToolRegistry()
        tool1 = DummyTool()
        tool2 = AnotherDummyTool()

        registry.register(tool1)
        registry.register(tool2)

        tools = registry.list_tools()
        assert len(tools) == 2
        assert any(t.name == "dummy" for t in tools)
        assert any(t.name == "another_dummy" for t in tools)

    def test_get_tool_names(self):
        """Test getting tool names."""
        registry = ToolRegistry()
        tool1 = DummyTool()
        tool2 = AnotherDummyTool()

        registry.register(tool1)
        registry.register(tool2)

        names = registry.get_tool_names()
        assert set(names) == {"dummy", "another_dummy"}

    def test_get_tools_for_ollama(self):
        """Test getting tools in Ollama format."""
        registry = ToolRegistry()
        tool = DummyTool()
        registry.register(tool)

        ollama_tools = registry.get_tools_for_ollama()

        assert len(ollama_tools) == 1
        assert ollama_tools[0].function["name"] == "dummy"

    def test_get_by_risk_level(self):
        """Test filtering tools by risk level."""
        registry = ToolRegistry()
        tool1 = DummyTool()  # LOW
        tool2 = AnotherDummyTool()  # MEDIUM

        registry.register(tool1)
        registry.register(tool2)

        low_tools = registry.get_by_risk_level(RiskLevel.LOW)
        medium_tools = registry.get_by_risk_level(RiskLevel.MEDIUM)

        assert len(low_tools) == 1
        assert low_tools[0].name == "dummy"
        assert len(medium_tools) == 1
        assert medium_tools[0].name == "another_dummy"

    def test_get_safe_tools(self):
        """Test getting safe (LOW risk) tools."""
        registry = ToolRegistry()
        tool1 = DummyTool()  # LOW
        tool2 = AnotherDummyTool()  # MEDIUM

        registry.register(tool1)
        registry.register(tool2)

        safe_tools = registry.get_safe_tools()

        assert len(safe_tools) == 1
        assert safe_tools[0].name == "dummy"

    def test_get_tools_requiring_confirmation(self):
        """Test getting tools that require confirmation."""
        registry = ToolRegistry()
        tool1 = DummyTool()  # LOW - no confirmation
        tool2 = AnotherDummyTool()  # MEDIUM - confirmation

        registry.register(tool1)
        registry.register(tool2)

        conf_tools = registry.get_tools_requiring_confirmation()

        assert len(conf_tools) == 1
        assert conf_tools[0].name == "another_dummy"

    def test_clear(self):
        """Test clearing the registry."""
        registry = ToolRegistry()
        tool1 = DummyTool()
        tool2 = AnotherDummyTool()

        registry.register(tool1)
        registry.register(tool2)
        assert registry.count() == 2

        registry.clear()
        assert registry.count() == 0

    def test_decorator_registration(self):
        """Test decorator-based registration."""
        registry = ToolRegistry()

        @registry.tool
        class DecoratedTool(BaseTool):
            name = "decorated"
            description = "Decorated tool"
            risk_level = RiskLevel.LOW
            parameters_schema = DummyParams

            async def execute(self, params):
                return ToolResult.success_result(data=params.value)

            def get_confirmation_message(self, params):
                return "test"

        assert registry.has_tool("decorated")
        assert registry.count() == 1

    def test_has_tool(self):
        """Test has_tool method."""
        registry = ToolRegistry()
        tool = DummyTool()

        assert not registry.has_tool("dummy")
        registry.register(tool)
        assert registry.has_tool("dummy")

    def test_count(self):
        """Test count method."""
        registry = ToolRegistry()
        assert registry.count() == 0

        registry.register(DummyTool())
        assert registry.count() == 1

        registry.register(AnotherDummyTool())
        assert registry.count() == 2

    def test_get_tools_by_category(self):
        """Test getting tools organized by category."""
        registry = ToolRegistry()
        tool1 = DummyTool()  # LOW
        tool2 = AnotherDummyTool()  # MEDIUM

        registry.register(tool1)
        registry.register(tool2)

        by_category = registry.get_tools_by_category()

        assert len(by_category[RiskLevel.LOW]) == 1
        assert len(by_category[RiskLevel.MEDIUM]) == 1
        assert len(by_category[RiskLevel.HIGH]) == 0
        assert len(by_category[RiskLevel.CRITICAL]) == 0

    def test_len(self):
        """Test __len__ method."""
        registry = ToolRegistry()
        assert len(registry) == 0

        registry.register(DummyTool())
        assert len(registry) == 1

    def test_contains(self):
        """Test __contains__ method."""
        registry = ToolRegistry()
        tool = DummyTool()

        assert "dummy" not in registry
        registry.register(tool)
        assert "dummy" in registry

    def test_repr(self):
        """Test string representation."""
        registry = ToolRegistry()
        registry.register(DummyTool())

        repr_str = repr(registry)
        assert "ToolRegistry" in repr_str
        assert "1 tools" in repr_str
        assert "dummy" in repr_str

    def test_iter(self):
        """Test iterating over tools."""
        registry = ToolRegistry()
        tool1 = DummyTool()
        tool2 = AnotherDummyTool()

        registry.register(tool1)
        registry.register(tool2)

        tools = list(registry)
        assert len(tools) == 2


class TestGlobalRegistry:
    """Test global registry functions."""

    def test_get_registry_singleton(self):
        """Test that get_registry returns singleton."""
        reset_registry()  # Clear first

        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_reset_registry(self):
        """Test resetting the global registry."""
        registry1 = get_registry()
        registry1.register(DummyTool())

        reset_registry()

        registry2 = get_registry()
        assert registry2 is not registry1
        assert registry2.count() == 0
