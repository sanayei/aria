"""Task planning and analysis for the ARIA agent.

This module provides simple task analysis to help the agent understand
user requests and plan appropriate responses.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskAnalysis(BaseModel):
    """Analysis of a user's task/request.

    Provides insights into what the user wants and how complex it might be.
    """

    requires_tools: bool = Field(
        ...,
        description="Whether this task likely requires tool use",
    )
    complexity: Literal["simple", "moderate", "complex"] = Field(
        ...,
        description="Estimated complexity of the task",
    )
    suggested_approach: str = Field(
        ...,
        description="Suggested approach for handling this task",
    )
    confidence: float = Field(
        default=0.5,
        description="Confidence in this analysis (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )


class TaskPlanner:
    """Simple task planner using heuristics.

    This planner uses pattern matching and keywords to analyze tasks.
    In the future, this could be enhanced with LLM-based analysis.
    """

    def __init__(self):
        """Initialize the task planner."""
        # Keywords that suggest tool usage
        self._tool_keywords = {
            # Time/date related
            "time", "date", "clock", "when", "today", "now", "current",
            # System info
            "system", "platform", "version", "os", "computer",
            # File operations
            "file", "directory", "folder", "path", "read", "write", "create", "delete",
            # Email
            "email", "send", "message", "mail",
            # Web/search
            "search", "google", "web", "internet", "url", "website",
            # Actions
            "do", "execute", "run", "perform", "make", "create",
        }

        # Keywords for conversational/informational requests
        self._conversational_keywords = {
            "what", "how", "why", "explain", "tell me", "describe",
            "can you", "could you", "would you", "help me understand",
        }

    def analyze_request(self, user_message: str) -> TaskAnalysis:
        """Analyze a user request to determine approach.

        Uses simple heuristics to assess:
        - Whether tools are likely needed
        - Task complexity
        - Suggested approach

        Args:
            user_message: The user's message/request

        Returns:
            TaskAnalysis: Analysis of the task
        """
        message_lower = user_message.lower()

        # Check for tool usage patterns
        requires_tools = self._likely_needs_tools(message_lower)

        # Assess complexity
        complexity = self._assess_complexity(message_lower)

        # Generate approach suggestion
        suggested_approach = self._suggest_approach(
            message_lower,
            requires_tools,
            complexity,
        )

        # Calculate confidence (simple heuristic)
        confidence = self._calculate_confidence(message_lower, requires_tools)

        analysis = TaskAnalysis(
            requires_tools=requires_tools,
            complexity=complexity,
            suggested_approach=suggested_approach,
            confidence=confidence,
        )

        logger.debug(
            f"Task analysis: requires_tools={requires_tools}, "
            f"complexity={complexity}, confidence={confidence:.2f}"
        )

        return analysis

    def _likely_needs_tools(self, message: str) -> bool:
        """Check if message likely needs tools.

        Args:
            message: Lowercase user message

        Returns:
            bool: True if tools likely needed
        """
        # Check for tool-related keywords
        tool_keyword_count = sum(
            1 for keyword in self._tool_keywords if keyword in message
        )

        # Strong indicators that tools are needed
        strong_indicators = [
            "what time",
            "current time",
            "what's the time",
            "system info",
            "platform",
            "send email",
            "create file",
            "delete file",
            "search for",
        ]

        has_strong_indicator = any(
            indicator in message for indicator in strong_indicators
        )

        # Check for action verbs
        action_verbs = ["send", "create", "delete", "run", "execute", "make", "do"]
        has_action = any(verb in message for verb in action_verbs)

        # Decide based on signals
        if has_strong_indicator:
            return True

        if tool_keyword_count >= 2 and has_action:
            return True

        if tool_keyword_count >= 3:
            return True

        return False

    def _assess_complexity(self, message: str) -> Literal["simple", "moderate", "complex"]:
        """Assess task complexity.

        Args:
            message: Lowercase user message

        Returns:
            Literal["simple", "moderate", "complex"]: Complexity level
        """
        # Simple heuristics based on message structure

        # Count words
        word_count = len(message.split())

        # Check for multi-step indicators
        multi_step_indicators = [
            "and then",
            "after that",
            "first",
            "second",
            "then",
            "finally",
            "multiple",
            "several",
            "all",
        ]

        has_multi_step = any(
            indicator in message for indicator in multi_step_indicators
        )

        # Check for conditional logic
        conditional_indicators = ["if", "unless", "when", "only if", "depending on"]
        has_conditional = any(
            indicator in message for indicator in conditional_indicators
        )

        # Determine complexity
        if has_conditional or (has_multi_step and word_count > 30):
            return "complex"
        elif has_multi_step or word_count > 20:
            return "moderate"
        else:
            return "simple"

    def _suggest_approach(
        self,
        message: str,
        requires_tools: bool,
        complexity: Literal["simple", "moderate", "complex"],
    ) -> str:
        """Suggest an approach for handling the task.

        Args:
            message: Lowercase user message
            requires_tools: Whether tools are likely needed
            complexity: Task complexity

        Returns:
            str: Suggested approach
        """
        if not requires_tools:
            if "explain" in message or "what is" in message or "how does" in message:
                return "Provide a direct, informative answer"
            else:
                return "Respond conversationally without tools"

        if complexity == "simple":
            return "Use a single tool to retrieve/perform the requested information/action"
        elif complexity == "moderate":
            return "Use multiple tools in sequence to complete the task"
        else:
            return "Break down into steps, use multiple tools, and validate results"

    def _calculate_confidence(self, message: str, requires_tools: bool) -> float:
        """Calculate confidence in the analysis.

        Args:
            message: Lowercase user message
            requires_tools: Whether tools are determined to be needed

        Returns:
            float: Confidence score (0.0-1.0)
        """
        # Start with base confidence
        confidence = 0.5

        # Increase confidence for clear patterns
        clear_patterns = [
            "what time",
            "current time",
            "system info",
            "send email",
            "create file",
        ]

        if any(pattern in message for pattern in clear_patterns):
            confidence += 0.3

        # Increase confidence for question marks (clear intent)
        if "?" in message:
            confidence += 0.1

        # Decrease confidence for ambiguous messages
        ambiguous_words = ["maybe", "perhaps", "might", "could", "possibly"]
        if any(word in message for word in ambiguous_words):
            confidence -= 0.2

        # Decrease confidence for very short or very long messages
        word_count = len(message.split())
        if word_count < 3 or word_count > 100:
            confidence -= 0.1

        # Clamp to valid range
        return max(0.0, min(1.0, confidence))
