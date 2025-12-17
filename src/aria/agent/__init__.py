"""Agent system for ARIA implementing the ReAct pattern.

This module provides the agent core that orchestrates LLM calls,
tool execution, and the Thought -> Action -> Observation loop.
"""

from aria.agent.core import Agent, AgentError, ToolCallRecord
from aria.agent.executor import ToolExecutor, ToolExecutionError
from aria.agent.planner import TaskPlanner, TaskAnalysis
from aria.agent.prompts import (
    SYSTEM_PROMPT,
    format_tool_result,
    format_tool_error,
    format_available_tools,
)

__all__ = [
    # Core
    "Agent",
    "AgentError",
    "ToolCallRecord",
    # Executor
    "ToolExecutor",
    "ToolExecutionError",
    # Planner
    "TaskPlanner",
    "TaskAnalysis",
    # Prompts
    "SYSTEM_PROMPT",
    "format_tool_result",
    "format_tool_error",
    "format_available_tools",
]
