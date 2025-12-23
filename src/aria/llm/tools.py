"""Utilities for converting Pydantic models to Ollama tool schemas.

This module provides functions to convert Pydantic models into the JSON Schema
format expected by Ollama for tool/function calling.
"""

import inspect
import logging
from typing import Any, Callable, Type, get_type_hints

from pydantic import BaseModel

from aria.llm.models import ToolDefinition

logger = logging.getLogger(__name__)


def pydantic_to_json_schema(
    model: Type[BaseModel],
    exclude_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Convert a Pydantic model to JSON Schema format.

    Args:
        model: Pydantic model class
        exclude_fields: Optional set of field names to exclude

    Returns:
        dict: JSON Schema representation of the model
    """
    schema = model.model_json_schema()

    # Remove title and other metadata that Ollama doesn't need
    if "title" in schema:
        del schema["title"]

    # Exclude specified fields
    if exclude_fields and "properties" in schema:
        for field in exclude_fields:
            schema["properties"].pop(field, None)

        # Update required fields
        if "required" in schema:
            schema["required"] = [f for f in schema["required"] if f not in exclude_fields]

    return schema


def create_tool_definition(
    name: str,
    description: str,
    parameters_model: Type[BaseModel] | None = None,
    parameters_schema: dict[str, Any] | None = None,
) -> ToolDefinition:
    """Create a tool definition for Ollama.

    Args:
        name: Name of the tool/function
        description: Human-readable description of what the tool does
        parameters_model: Pydantic model defining the parameters (optional)
        parameters_schema: Pre-built JSON schema for parameters (optional)

    Returns:
        ToolDefinition: Tool definition ready for Ollama

    Raises:
        ValueError: If neither parameters_model nor parameters_schema is provided
    """
    if parameters_model is not None:
        parameters = pydantic_to_json_schema(parameters_model)
    elif parameters_schema is not None:
        parameters = parameters_schema
    else:
        # No parameters - empty object schema
        parameters = {"type": "object", "properties": {}}

    return ToolDefinition.from_schema(
        name=name,
        description=description,
        parameters=parameters,
    )


def function_to_tool_definition(
    func: Callable,
    parameters_model: Type[BaseModel] | None = None,
    description: str | None = None,
) -> ToolDefinition:
    """Convert a Python function to a tool definition.

    Args:
        func: The function to convert
        parameters_model: Pydantic model for the function's parameters
        description: Override description (uses docstring if not provided)

    Returns:
        ToolDefinition: Tool definition for the function

    Example:
        >>> class WeatherInput(BaseModel):
        ...     location: str
        ...     units: str = "celsius"
        ...
        >>> def get_weather(location: str, units: str = "celsius") -> str:
        ...     '''Get the current weather for a location.'''
        ...     ...
        ...
        >>> tool_def = function_to_tool_definition(get_weather, WeatherInput)
    """
    # Get function name
    func_name = func.__name__

    # Get description from docstring or parameter
    func_description = description
    if func_description is None:
        func_description = inspect.getdoc(func) or f"Call the {func_name} function"

    # Create the tool definition
    return create_tool_definition(
        name=func_name,
        description=func_description,
        parameters_model=parameters_model,
    )


def create_simple_tool(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str] | None = None,
) -> ToolDefinition:
    """Create a simple tool definition without Pydantic models.

    Args:
        name: Tool name
        description: Tool description
        properties: Dictionary of parameter names to JSON Schema property definitions
        required: List of required parameter names

    Returns:
        ToolDefinition: Tool definition ready for Ollama

    Example:
        >>> tool = create_simple_tool(
        ...     name="get_weather",
        ...     description="Get weather for a location",
        ...     properties={
        ...         "location": {
        ...             "type": "string",
        ...             "description": "City name"
        ...         },
        ...         "units": {
        ...             "type": "string",
        ...             "enum": ["celsius", "fahrenheit"],
        ...             "description": "Temperature units"
        ...         }
        ...     },
        ...     required=["location"]
        ... )
    """
    parameters = {
        "type": "object",
        "properties": properties,
    }

    if required:
        parameters["required"] = required

    return ToolDefinition.from_schema(
        name=name,
        description=description,
        parameters=parameters,
    )


class ToolRegistry:
    """Registry for managing tool definitions.

    This class helps organize and manage multiple tools, making it easy
    to pass them to the LLM.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        tool: ToolDefinition,
        name: str | None = None,
    ) -> None:
        """Register a tool definition.

        Args:
            tool: Tool definition to register
            name: Optional name (uses tool's name if not provided)
        """
        tool_name = name or tool.function["name"]
        self._tools[tool_name] = tool
        logger.debug(f"Registered tool: {tool_name}")

    def register_from_function(
        self,
        func: Callable,
        parameters_model: Type[BaseModel] | None = None,
        description: str | None = None,
    ) -> None:
        """Register a tool from a Python function.

        Args:
            func: Function to register as a tool
            parameters_model: Pydantic model for parameters
            description: Optional description override
        """
        tool_def = function_to_tool_definition(func, parameters_model, description)
        self.register(tool_def)

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: Name of the tool to remove
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name.

        Args:
            name: Tool name

        Returns:
            ToolDefinition | None: The tool definition, or None if not found
        """
        return self._tools.get(name)

    def get_all(self) -> list[ToolDefinition]:
        """Get all registered tools.

        Returns:
            list[ToolDefinition]: List of all tool definitions
        """
        return list(self._tools.values())

    def get_names(self) -> list[str]:
        """Get names of all registered tools.

        Returns:
            list[str]: List of tool names
        """
        return list(self._tools.keys())

    def clear(self) -> None:
        """Remove all tools from the registry."""
        self._tools.clear()
        logger.debug("Cleared all tools from registry")

    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"ToolRegistry({len(self._tools)} tools: {', '.join(self.get_names())})"
