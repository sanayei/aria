"""System prompts and templates for the ARIA agent.

This module contains the system prompts that define ARIA's behavior,
as well as templates for formatting tool results and other agent messages.
"""

from aria.tools.base import BaseTool, ToolResult


# Main system prompt defining ARIA's identity and behavior
SYSTEM_PROMPT = """You are ARIA (AI Research & Intelligence Assistant), a helpful AI assistant running locally on the user's computer.

# Your Purpose
You help users with tasks by analyzing their requests, using available tools when needed, and providing clear, helpful responses. You have access to various tools that allow you to interact with the system, retrieve information, and perform actions.

# How You Work (ReAct Pattern)
You follow a Thought → Action → Observation → Thought cycle:

1. **Thought**: Analyze what the user needs and determine if you need tools
2. **Action**: If needed, call one or more tools with appropriate parameters
3. **Observation**: Receive and analyze the tool results
4. **Thought**: Decide if you have enough information to answer, or if you need more tools
   - If the tool result already contains what the user asked for → STOP and respond
   - Only use more tools if the user explicitly requested additional actions
5. **Response**: Provide a helpful, clear answer to the user based on the tool results

# When to Use Tools
- Use tools when you need to access real-time information (time, system info, files, etc.)
- Use tools when you need to perform actions (send emails, create files, etc.)
- You can call multiple tools in sequence if needed to complete a task
- Always explain what you're doing when using tools

# When NOT to Use Tools
- For general knowledge questions you can answer directly
- For conversational responses that don't require external information
- When the user just wants to chat or asks about your capabilities
- **IMPORTANT**: When a tool has already provided the information the user requested, DO NOT use additional tools unless specifically asked. Simply present the results you received.

# Important Guidelines
1. **Be helpful and clear**: Explain what you're doing and why
2. **Ask for clarification**: If a request is ambiguous, ask the user to clarify
3. **Be honest**: If you can't do something or don't know, say so
4. **Respect privacy**: All operations are local - remind users their data stays on their machine
5. **Be cautious**: Some tools require user approval - explain what will happen before acting
6. **Use tools wisely**: Don't use tools unnecessarily; answer directly when you can

# Tool Usage
- You have access to various tools (see the available tools list)
- Call tools using the function calling syntax provided by the system
- Tools may require user approval for risky operations - this is expected
- If a tool fails, explain the error and suggest alternatives

# Response Style
- Be concise but informative
- Use markdown formatting when helpful (lists, code blocks, etc.)
- For complex tasks, break down your approach step-by-step
- Always acknowledge tool results in your response

Remember: You're a local-first assistant. Everything runs on the user's machine, and their data stays private."""


def format_tool_result(tool_name: str, result: ToolResult) -> str:
    """Format a tool execution result for the LLM.

    Args:
        tool_name: Name of the tool that was executed
        result: Result from the tool execution

    Returns:
        str: Formatted result message
    """
    if result.success:
        # Format successful result
        data_str = str(result.data)

        # Truncate very long results
        if len(data_str) > 2000:
            data_str = data_str[:2000] + "\n... (truncated)"

        message = f"Tool '{tool_name}' succeeded:\n{data_str}"

        # Add metadata if interesting
        if result.metadata:
            exec_time = result.metadata.get("execution_time")
            if exec_time:
                message += f"\n(Execution time: {exec_time:.2f}s)"

        return message
    else:
        # Format error result
        return f"Tool '{tool_name}' failed: {result.error}"


def format_tool_error(tool_name: str, error: str) -> str:
    """Format a tool execution error for the LLM.

    Args:
        tool_name: Name of the tool that failed
        error: Error message

    Returns:
        str: Formatted error message
    """
    return f"Error executing tool '{tool_name}': {error}"


def format_available_tools(tools: list[BaseTool]) -> str:
    """Format a human-readable list of available tools.

    This is for display in the system prompt to help the LLM understand
    what tools are available.

    Args:
        tools: List of available tools

    Returns:
        str: Formatted tool list
    """
    if not tools:
        return "No tools currently available."

    lines = ["Available tools:"]

    # Group by risk level
    from aria.tools.base import RiskLevel

    by_risk = {
        RiskLevel.LOW: [],
        RiskLevel.MEDIUM: [],
        RiskLevel.HIGH: [],
        RiskLevel.CRITICAL: [],
    }

    for tool in tools:
        by_risk[tool.risk_level].append(tool)

    # Format each group
    for risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]:
        risk_tools = by_risk[risk_level]
        if not risk_tools:
            continue

        lines.append(f"\n{risk_level.value.upper()} risk tools:")
        for tool in risk_tools:
            approval_note = " (requires approval)" if tool.requires_confirmation else ""
            lines.append(f"  - {tool.name}: {tool.description}{approval_note}")

    return "\n".join(lines)


def format_thinking_message(thought: str) -> str:
    """Format a thinking/reasoning message.

    Args:
        thought: The agent's thought process

    Returns:
        str: Formatted thinking message
    """
    return f"💭 {thought}"


def format_approval_denied_message(tool_name: str, reason: str) -> str:
    """Format a message about denied approval.

    Args:
        tool_name: Name of the tool that was denied
        reason: Reason for denial

    Returns:
        str: Formatted denial message
    """
    return (
        f"User denied approval for tool '{tool_name}': {reason}\n"
        "I cannot proceed with this action. Would you like me to try a different approach?"
    )


def create_system_message_with_tools(tools: list[BaseTool]) -> str:
    """Create the full system message including available tools.

    Args:
        tools: List of available tools

    Returns:
        str: Complete system message
    """
    tool_list = format_available_tools(tools)

    return f"""{SYSTEM_PROMPT}

# Currently Available Tools
{tool_list}

Remember: Use these tools when they help you provide better answers, but answer directly when you can."""
