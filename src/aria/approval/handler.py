"""Approval handler for user confirmation of risky tool executions.

This module provides the ApprovalHandler which manages user approval flows
for tools based on their risk level.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field
from rich.panel import Panel
from rich.text import Text

from aria.tools.base import BaseTool, RiskLevel
from aria.ui.console import ARIAConsole
from aria.ui.prompts import confirm, prompt
from aria.approval.classifier import ActionClassifier

logger = logging.getLogger(__name__)


class ApprovalResult(BaseModel):
    """Result of an approval request.

    Contains whether the action was approved, any modified parameters,
    and the reason if denied.
    """

    approved: bool = Field(..., description="Whether the action was approved")
    modified_params: BaseModel | None = Field(
        default=None,
        description="Modified parameters (if user changed them)",
    )
    reason: str | None = Field(
        default=None,
        description="Reason for denial (if not approved)",
    )

    @classmethod
    def approve(
        cls,
        modified_params: BaseModel | None = None,
    ) -> "ApprovalResult":
        """Create an approved result.

        Args:
            modified_params: Optional modified parameters

        Returns:
            ApprovalResult: Approved result
        """
        return cls(approved=True, modified_params=modified_params)

    @classmethod
    def deny(cls, reason: str = "User denied") -> "ApprovalResult":
        """Create a denied result.

        Args:
            reason: Reason for denial

        Returns:
            ApprovalResult: Denied result
        """
        return cls(approved=False, reason=reason)


class ApprovalHandler:
    """Handler for managing user approval flows.

    This class handles the presentation of approval requests to users
    and collects their responses, with appropriate flows for different
    risk levels.
    """

    def __init__(
        self,
        console: ARIAConsole,
        classifier: ActionClassifier | None = None,
    ):
        """Initialize the approval handler.

        Args:
            console: Console for user interaction
            classifier: Action classifier (creates default if None)
        """
        self.console = console
        self.classifier = classifier or ActionClassifier()

    async def request_approval(
        self,
        tool: BaseTool,
        params: BaseModel,
        risk_factors: list[str] | None = None,
    ) -> ApprovalResult:
        """Request user approval for a tool execution.

        Displays information about the action and prompts for confirmation.

        Args:
            tool: Tool to execute
            params: Parameters for execution
            risk_factors: Optional list of risk factors (auto-generated if None)

        Returns:
            ApprovalResult: Approval decision
        """
        # Get risk assessment
        risk_level = self.classifier.classify_tool(tool, params)

        # Get risk factors if not provided
        if risk_factors is None:
            risk_factors = self.classifier.get_risk_factors(tool, params)

        # Format and display the approval prompt
        panel = self.format_approval_prompt(tool, params, risk_level, risk_factors)
        self.console.console.print(panel)

        # Get user decision
        approved = confirm(
            "Approve this action?",
            default=False,
            console=self.console.console,
        )

        if approved:
            logger.info(f"User approved tool execution: {tool.name}")
            return ApprovalResult.approve()
        else:
            logger.info(f"User denied tool execution: {tool.name}")
            return ApprovalResult.deny("User declined to approve")

    async def request_double_confirmation(
        self,
        tool: BaseTool,
        params: BaseModel,
        risk_factors: list[str] | None = None,
    ) -> ApprovalResult:
        """Request double confirmation for CRITICAL actions.

        Shows detailed information and requires two confirmations.

        Args:
            tool: Tool to execute
            params: Parameters for execution
            risk_factors: Optional list of risk factors (auto-generated if None)

        Returns:
            ApprovalResult: Approval decision
        """
        # Get risk assessment
        risk_level = self.classifier.classify_tool(tool, params)

        # Get risk factors if not provided
        if risk_factors is None:
            risk_factors = self.classifier.get_risk_factors(tool, params)

        # Format and display the approval prompt with extra warning
        panel = self.format_approval_prompt(
            tool,
            params,
            risk_level,
            risk_factors,
            critical_warning=True,
        )
        self.console.console.print(panel)

        # First confirmation
        first_approved = confirm(
            "⚠️  This is a CRITICAL operation. Do you want to proceed?",
            default=False,
            console=self.console.console,
        )

        if not first_approved:
            logger.info(f"User denied critical tool execution: {tool.name} (first confirmation)")
            return ApprovalResult.deny("User declined at first confirmation")

        # Second confirmation - require typing CONFIRM
        self.console.console.print(
            "\n[red bold]⚠️  CRITICAL ACTION REQUIRES CONFIRMATION[/red bold]"
        )
        self.console.console.print(
            "[yellow]Type 'CONFIRM' exactly to proceed with this action:[/yellow]"
        )

        confirmation_text = prompt(
            "Enter confirmation",
            default="",
            console=self.console.console,
        )

        if confirmation_text == "CONFIRM":
            logger.info(f"User approved critical tool execution: {tool.name} (double confirmed)")
            return ApprovalResult.approve()
        else:
            logger.info(
                f"User denied critical tool execution: {tool.name} "
                f"(incorrect confirmation: '{confirmation_text}')"
            )
            return ApprovalResult.deny(
                f"Incorrect confirmation (got '{confirmation_text}', expected 'CONFIRM')"
            )

    def format_approval_prompt(
        self,
        tool: BaseTool,
        params: BaseModel,
        risk_level: RiskLevel,
        risk_factors: list[str] | None = None,
        critical_warning: bool = False,
    ) -> Panel:
        """Format an approval prompt as a Rich Panel.

        Args:
            tool: Tool to execute
            params: Parameters for execution
            risk_level: Risk level of the action
            risk_factors: List of risk factors
            critical_warning: Whether to show critical warning

        Returns:
            Panel: Formatted approval prompt
        """
        # Get risk styling
        risk_color = self.classifier.get_risk_color(risk_level)
        risk_emoji = self.classifier.get_risk_emoji(risk_level)

        # Build the content
        content = Text()

        # Tool name
        content.append("🔧 Tool: ", style="bold")
        content.append(f"{tool.name}\n", style="cyan bold")

        # Action description
        action_description = tool.get_confirmation_message(params)
        content.append("📋 Action: ", style="bold")
        content.append(f"{action_description}\n\n", style="white")

        # Risk level
        content.append(f"{risk_emoji} Risk Level: ", style="bold")
        content.append(f"{risk_level.value.upper()}\n\n", style=risk_color)

        # Critical warning
        if critical_warning:
            content.append(
                "⚠️  CRITICAL OPERATION - REQUIRES DOUBLE CONFIRMATION\n\n",
                style="red bold",
            )

        # Action details (what will happen)
        if risk_factors:
            content.append("This action will:\n", style="bold")
            for factor in risk_factors:
                content.append(f"  • {factor}\n", style="dim")

        # Create panel
        border_style = "red bold" if risk_level == RiskLevel.CRITICAL else risk_color

        return Panel(
            content,
            title=f"[bold]{risk_emoji} Approval Required[/bold]",
            border_style=border_style,
            padding=(1, 2),
        )

    def format_tool_summary(
        self,
        tool: BaseTool,
        params: BaseModel,
    ) -> str:
        """Format a summary of tool and parameters.

        Args:
            tool: Tool being executed
            params: Parameters for execution

        Returns:
            str: Formatted summary
        """
        lines = [
            f"Tool: {tool.name}",
            f"Description: {tool.description}",
            f"Risk Level: {tool.risk_level.value}",
            "",
            "Parameters:",
        ]

        # Add parameters
        for key, value in params.model_dump().items():
            # Truncate long values
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            lines.append(f"  {key}: {value_str}")

        return "\n".join(lines)
