"""Risk classification for tool actions.

This module provides the ActionClassifier which assesses the risk level
of tool executions and provides human-readable explanations.
"""

import logging
from typing import Any

from pydantic import BaseModel

from aria.tools.base import BaseTool, RiskLevel

logger = logging.getLogger(__name__)


class ActionClassifier:
    """Classifier for assessing risk levels of tool actions.

    The classifier determines whether a tool action requires user approval
    and provides human-readable explanations for the risk assessment.
    """

    def __init__(self):
        """Initialize the action classifier."""
        # Risk factor templates for common patterns
        self._risk_factor_templates = {
            RiskLevel.LOW: [
                "Read-only operation with no side effects",
                "Safe to execute automatically",
            ],
            RiskLevel.MEDIUM: [
                "Modifies data or system state",
                "Action can be undone or reversed",
                "Requires user confirmation",
            ],
            RiskLevel.HIGH: [
                "Irreversible operation",
                "Affects external systems or users",
                "Cannot be easily undone",
            ],
            RiskLevel.CRITICAL: [
                "Dangerous operation with significant impact",
                "Bulk or mass operation",
                "Potential for data loss or corruption",
                "Requires double confirmation",
            ],
        }

    def classify_tool(
        self,
        tool: BaseTool,
        params: BaseModel,
    ) -> RiskLevel:
        """Classify the risk level of a tool execution.

        Currently returns the tool's declared risk level, but this method
        allows for future dynamic classification based on parameters.

        Args:
            tool: Tool to classify
            params: Parameters for the tool execution

        Returns:
            RiskLevel: Assessed risk level
        """
        # For now, use the tool's declared risk level
        # In the future, this could be enhanced to:
        # - Increase risk for bulk operations
        # - Increase risk for external/production targets
        # - Decrease risk for dry-run modes
        # - Consider parameter values (e.g., large file sizes)

        base_risk = tool.risk_level

        # Log the classification
        logger.debug(
            f"Classified tool '{tool.name}' as {base_risk.value} risk "
            f"with params: {params.model_dump()}"
        )

        return base_risk

    def get_risk_factors(
        self,
        tool: BaseTool,
        params: BaseModel,
    ) -> list[str]:
        """Get human-readable risk factors for a tool execution.

        Returns a list of reasons explaining why the action has its risk level.

        Args:
            tool: Tool being executed
            params: Parameters for execution

        Returns:
            list[str]: List of risk factor descriptions
        """
        risk_level = self.classify_tool(tool, params)
        factors = []

        # Get general risk factors for this level
        if risk_level in self._risk_factor_templates:
            factors.extend(self._risk_factor_templates[risk_level])

        # Add tool-specific factors if available
        # Tools can optionally provide a get_risk_factors method
        if hasattr(tool, "get_risk_factors"):
            try:
                tool_factors = tool.get_risk_factors(params)
                if tool_factors:
                    factors.extend(tool_factors)
            except Exception as e:
                logger.warning(f"Failed to get risk factors from tool {tool.name}: {e}")

        # Add parameter-based factors
        param_factors = self._analyze_parameters(tool, params)
        if param_factors:
            factors.extend(param_factors)

        return factors

    def should_require_confirmation(
        self,
        tool: BaseTool,
        params: BaseModel,
    ) -> bool:
        """Check if a tool execution requires user confirmation.

        Args:
            tool: Tool being executed
            params: Parameters for execution

        Returns:
            bool: True if confirmation is required
        """
        risk_level = self.classify_tool(tool, params)
        return risk_level.requires_confirmation

    def should_require_double_confirmation(
        self,
        tool: BaseTool,
        params: BaseModel,
    ) -> bool:
        """Check if a tool execution requires double confirmation.

        Args:
            tool: Tool being executed
            params: Parameters for execution

        Returns:
            bool: True if double confirmation is required
        """
        risk_level = self.classify_tool(tool, params)
        return risk_level.requires_double_confirmation

    def _analyze_parameters(
        self,
        tool: BaseTool,
        params: BaseModel,
    ) -> list[str]:
        """Analyze parameters to identify additional risk factors.

        Args:
            tool: Tool being executed
            params: Parameters to analyze

        Returns:
            list[str]: Additional risk factors based on parameters
        """
        factors = []
        param_dict = params.model_dump()

        # Check for bulk operations
        bulk_indicators = ["count", "limit", "max", "all", "batch_size"]
        for key, value in param_dict.items():
            if key in bulk_indicators and isinstance(value, int):
                if value > 100:
                    factors.append(f"Large batch operation ({key}={value} items)")
                elif value > 10:
                    factors.append(f"Batch operation ({key}={value} items)")

        # Check for external targets
        external_indicators = ["email", "url", "host", "domain", "recipient"]
        for key, value in param_dict.items():
            if any(ind in key.lower() for ind in external_indicators):
                if isinstance(value, str) and value:
                    factors.append(
                        f"Targets external system ({key}={value[:50]}...)"
                        if len(str(value)) > 50
                        else f"Targets external system ({key}={value})"
                    )

        # Check for file operations
        file_indicators = ["path", "file", "directory", "folder"]
        for key, value in param_dict.items():
            if any(ind in key.lower() for ind in file_indicators):
                if isinstance(value, str) and value:
                    factors.append(f"File operation: {value}")

        # Check for deletion operations
        delete_indicators = ["delete", "remove", "purge", "destroy"]
        for key in param_dict.keys():
            if any(ind in key.lower() for ind in delete_indicators):
                factors.append("Deletion operation - cannot be undone")
                break

        return factors

    def get_risk_color(self, risk_level: RiskLevel) -> str:
        """Get the color code for a risk level.

        Args:
            risk_level: Risk level to get color for

        Returns:
            str: Rich color code
        """
        return {
            RiskLevel.LOW: "green",
            RiskLevel.MEDIUM: "yellow",
            RiskLevel.HIGH: "red",
            RiskLevel.CRITICAL: "red bold",
        }[risk_level]

    def get_risk_emoji(self, risk_level: RiskLevel) -> str:
        """Get an emoji representing the risk level.

        Args:
            risk_level: Risk level to get emoji for

        Returns:
            str: Emoji character
        """
        return {
            RiskLevel.LOW: "✓",
            RiskLevel.MEDIUM: "⚠️",
            RiskLevel.HIGH: "⚠️",
            RiskLevel.CRITICAL: "🚨",
        }[risk_level]
