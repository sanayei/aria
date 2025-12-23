"""Tool registry for managing and organizing tools.

This module provides the ToolRegistry class which manages all available
tools, provides lookup functionality, and supports decorator-based
registration.
"""

import logging
from typing import Type, TypeVar, Callable

from aria.tools.base import BaseTool, RiskLevel
from aria.llm.models import ToolDefinition

logger = logging.getLogger(__name__)

# Type variable for tool classes
T = TypeVar("T", bound=BaseTool)


class ToolRegistry:
    """Registry for managing ARIA tools.

    The registry maintains a collection of available tools and provides
    methods for registration, lookup, and filtering. It supports both
    manual registration and decorator-based registration.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(MyTool())
        >>>
        >>> # Or using decorator
        >>> @registry.tool
        >>> class MyTool(BaseTool):
        ...     name = "my_tool"
        ...     ...
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                f"Use unregister() first or choose a different name."
            )

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} (risk: {tool.risk_level.value})")

    def unregister(self, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: Name of the tool to unregister

        Raises:
            KeyError: If the tool is not registered
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")

        del self._tools[name]
        logger.info(f"Unregistered tool: {name}")

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name.

        Args:
            name: Name of the tool

        Returns:
            BaseTool | None: The tool instance, or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Get a list of all registered tools.

        Returns:
            list[BaseTool]: List of all tool instances
        """
        return list(self._tools.values())

    def get_tool_names(self) -> list[str]:
        """Get names of all registered tools.

        Returns:
            list[str]: List of tool names
        """
        return list(self._tools.keys())

    def get_tools_for_ollama(self) -> list[ToolDefinition]:
        """Get all tools in Ollama ToolDefinition format.

        Returns:
            list[ToolDefinition]: List of tool definitions for Ollama
        """
        return [tool.to_ollama_tool() for tool in self._tools.values()]

    def get_by_risk_level(self, level: RiskLevel) -> list[BaseTool]:
        """Get all tools with a specific risk level.

        Args:
            level: Risk level to filter by

        Returns:
            list[BaseTool]: List of tools with the specified risk level
        """
        return [tool for tool in self._tools.values() if tool.risk_level == level]

    def get_safe_tools(self) -> list[BaseTool]:
        """Get all low-risk tools that can be auto-executed.

        Returns:
            list[BaseTool]: List of LOW risk tools
        """
        return self.get_by_risk_level(RiskLevel.LOW)

    def get_tools_requiring_confirmation(self) -> list[BaseTool]:
        """Get all tools that require user confirmation.

        Returns:
            list[BaseTool]: List of tools requiring confirmation
        """
        return [tool for tool in self._tools.values() if tool.requires_confirmation]

    def clear(self) -> None:
        """Remove all tools from the registry."""
        count = len(self._tools)
        self._tools.clear()
        logger.info(f"Cleared {count} tools from registry")

    def tool(self, tool_class: Type[T]) -> Type[T]:
        """Decorator for registering tool classes.

        This decorator instantiates and registers the tool class.

        Args:
            tool_class: Tool class to register

        Returns:
            Type[T]: The same tool class (for chaining)

        Example:
            >>> registry = ToolRegistry()
            >>>
            >>> @registry.tool
            >>> class MyTool(BaseTool):
            ...     name = "my_tool"
            ...     description = "Does something"
            ...     risk_level = RiskLevel.LOW
            ...     parameters_schema = MyParams
            ...     ...
        """
        # Instantiate and register the tool
        tool_instance = tool_class()
        self.register(tool_instance)
        return tool_class

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Tool name

        Returns:
            bool: True if the tool is registered
        """
        return name in self._tools

    def count(self) -> int:
        """Get the number of registered tools.

        Returns:
            int: Number of registered tools
        """
        return len(self._tools)

    def get_tools_by_category(self) -> dict[RiskLevel, list[BaseTool]]:
        """Get tools organized by risk level.

        Returns:
            dict[RiskLevel, list[BaseTool]]: Tools grouped by risk level
        """
        result: dict[RiskLevel, list[BaseTool]] = {
            RiskLevel.LOW: [],
            RiskLevel.MEDIUM: [],
            RiskLevel.HIGH: [],
            RiskLevel.CRITICAL: [],
        }

        for tool in self._tools.values():
            result[tool.risk_level].append(tool)

        return result

    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"<ToolRegistry: {len(self._tools)} tools ({', '.join(self._tools.keys())})>"

    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.values())


# Global default registry
_default_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global default tool registry.

    Returns:
        ToolRegistry: The global registry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _default_registry
    _default_registry = None
