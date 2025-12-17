"""Base infrastructure for ARIA tools.

This module provides the foundational classes for building tools that the
agent can use, including risk classification, result handling, and schema
validation.
"""

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, TypeVar, Generic

from pydantic import BaseModel, Field, ValidationError

from aria.llm.models import ToolDefinition
from aria.llm.tools import create_tool_definition, pydantic_to_json_schema


# Type variable for tool parameters
TParams = TypeVar("TParams", bound=BaseModel)


class RiskLevel(str, Enum):
    """Risk level classification for tool operations.

    Tools are classified by their potential impact, determining whether
    they require user approval before execution.
    """

    LOW = "low"  # Read-only, no side effects (auto-execute)
    MEDIUM = "medium"  # Reversible side effects (confirm)
    HIGH = "high"  # Irreversible/sensitive operations (explicit approval)
    CRITICAL = "critical"  # Dangerous/bulk operations (double confirm)

    @property
    def requires_confirmation(self) -> bool:
        """Check if this risk level requires user confirmation.

        Returns:
            bool: True if confirmation is required
        """
        return self in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)

    @property
    def requires_double_confirmation(self) -> bool:
        """Check if this risk level requires double confirmation.

        Returns:
            bool: True if double confirmation is required
        """
        return self == RiskLevel.CRITICAL

    def __str__(self) -> str:
        """String representation of risk level."""
        return self.value


class ToolResult(BaseModel):
    """Result from tool execution.

    Contains the execution result, success status, error information,
    and metadata about the execution.
    """

    success: bool = Field(..., description="Whether the tool executed successfully")
    data: Any = Field(default=None, description="The actual result data")
    error: str | None = Field(default=None, description="Error message if failed")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata (time, tokens, etc.)",
    )

    @classmethod
    def success_result(cls, data: Any, **metadata: Any) -> "ToolResult":
        """Create a successful result.

        Args:
            data: The result data
            **metadata: Additional metadata

        Returns:
            ToolResult: Successful tool result
        """
        return cls(success=True, data=data, error=None, metadata=metadata)

    @classmethod
    def error_result(cls, error: str, **metadata: Any) -> "ToolResult":
        """Create an error result.

        Args:
            error: Error message
            **metadata: Additional metadata

        Returns:
            ToolResult: Error tool result
        """
        return cls(success=False, data=None, error=error, metadata=metadata)

    def __str__(self) -> str:
        """String representation of the result."""
        if self.success:
            return f"Success: {self.data}"
        else:
            return f"Error: {self.error}"


class BaseTool(ABC, Generic[TParams]):
    """Abstract base class for all ARIA tools.

    Tools are the primary way the agent interacts with the world. Each tool
    defines its parameters, risk level, and execution logic.

    Subclasses must implement:
    - execute() - The actual tool logic
    - get_confirmation_message() - Human-readable description for approval

    Type Parameters:
        TParams: Pydantic model defining the tool's parameters
    """

    # Class attributes (must be set in subclasses)
    name: str
    description: str
    risk_level: RiskLevel
    parameters_schema: type[BaseModel]

    def __init__(self):
        """Initialize the tool.

        Validates that required class attributes are set.
        """
        # Validate required attributes
        required_attrs = ["name", "description", "risk_level", "parameters_schema"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise ValueError(
                    f"Tool must define '{attr}' class attribute. "
                    f"Subclass {self.__class__.__name__} is missing it."
                )

        # Validate parameters_schema is a Pydantic model
        if not issubclass(self.parameters_schema, BaseModel):
            raise ValueError(
                f"parameters_schema must be a Pydantic BaseModel subclass, "
                f"got {type(self.parameters_schema)}"
            )

    @property
    def requires_confirmation(self) -> bool:
        """Check if this tool requires user confirmation.

        Returns:
            bool: True if confirmation is required based on risk level
        """
        return self.risk_level.requires_confirmation

    @property
    def requires_double_confirmation(self) -> bool:
        """Check if this tool requires double confirmation.

        Returns:
            bool: True if double confirmation is required
        """
        return self.risk_level.requires_double_confirmation

    @abstractmethod
    async def execute(self, params: TParams) -> ToolResult:
        """Execute the tool with validated parameters.

        Args:
            params: Validated parameters conforming to parameters_schema

        Returns:
            ToolResult: Result of the execution

        Raises:
            Exception: Tool-specific exceptions during execution
        """
        pass

    @abstractmethod
    def get_confirmation_message(self, params: TParams) -> str:
        """Get a human-readable description of what will happen.

        This message is shown to the user when confirmation is required.

        Args:
            params: Validated parameters

        Returns:
            str: Human-readable description of the operation
        """
        pass

    def validate_params(self, raw_params: dict[str, Any]) -> BaseModel:
        """Parse and validate raw parameters.

        Args:
            raw_params: Raw parameter dictionary

        Returns:
            BaseModel: Validated parameters

        Raises:
            ValidationError: If parameters are invalid
        """
        return self.parameters_schema(**raw_params)

    def to_ollama_tool(self) -> ToolDefinition:
        """Convert this tool to Ollama ToolDefinition format.

        Returns:
            ToolDefinition: Tool definition for Ollama API
        """
        return create_tool_definition(
            name=self.name,
            description=self.description,
            parameters_model=self.parameters_schema,
        )

    async def run(
        self,
        raw_params: dict[str, Any],
        track_time: bool = True,
    ) -> ToolResult:
        """Run the tool with parameter validation and error handling.

        This is the main entry point for tool execution. It handles:
        - Parameter validation
        - Execution timing
        - Error catching
        - Result wrapping

        Args:
            raw_params: Raw parameter dictionary
            track_time: Whether to track execution time

        Returns:
            ToolResult: Execution result
        """
        start_time = time.time() if track_time else None

        try:
            # Validate parameters
            validated_params = self.validate_params(raw_params)

            # Execute the tool
            result = await self.execute(validated_params)

            # Add execution time if tracking
            if track_time and start_time is not None:
                execution_time = time.time() - start_time
                result.metadata["execution_time"] = execution_time

            return result

        except ValidationError as e:
            # Parameter validation failed
            error_msg = f"Parameter validation failed: {e}"
            return ToolResult.error_result(
                error=error_msg,
                execution_time=time.time() - start_time if start_time else None,
            )

        except Exception as e:
            # Tool execution failed
            error_msg = f"Tool execution failed: {type(e).__name__}: {e}"
            return ToolResult.error_result(
                error=error_msg,
                execution_time=time.time() - start_time if start_time else None,
            )

    def __repr__(self) -> str:
        """String representation of the tool."""
        return (
            f"<{self.__class__.__name__} "
            f"name='{self.name}' "
            f"risk={self.risk_level.value}>"
        )


# Helper function for creating simple parameter schemas
def create_params_schema(
    name: str,
    fields: dict[str, tuple[type, Any]],
) -> type[BaseModel]:
    """Create a Pydantic model for tool parameters dynamically.

    Args:
        name: Name for the schema class
        fields: Dictionary of field_name: (type, default/Field())

    Returns:
        type[BaseModel]: Pydantic model class

    Example:
        >>> ParamsSchema = create_params_schema(
        ...     "EchoParams",
        ...     {"message": (str, Field(..., description="Message to echo"))}
        ... )
    """
    from pydantic import create_model

    return create_model(name, **fields)  # type: ignore
