"""Tool system for ARIA.

This module provides the foundation for building and managing tools that
the agent can use to interact with the world.
"""

from aria.tools.base import (
    RiskLevel,
    ToolResult,
    BaseTool,
    create_params_schema,
)
from aria.tools.registry import (
    ToolRegistry,
    get_registry,
    reset_registry,
)

# Create and export the default global registry
default_registry = get_registry()

__all__ = [
    # Base classes
    "RiskLevel",
    "ToolResult",
    "BaseTool",
    "create_params_schema",
    # Registry
    "ToolRegistry",
    "get_registry",
    "reset_registry",
    "default_registry",
]
