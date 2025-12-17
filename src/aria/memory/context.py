"""Conversation context formatting for LLM integration.

This module provides utilities for formatting conversation history from the
database into a format suitable for LLM context, with proper truncation and
tool call summarization.
"""

from aria.llm.models import ChatMessage
from aria.memory.models import ConversationContext, Message, MessageRole, ToolCall


def messages_to_chat_messages(
    messages: list[Message],
    tool_calls: dict[str, list[ToolCall]],
) -> list[ChatMessage]:
    """Convert database messages to LLM chat messages.

    Args:
        messages: List of messages from database
        tool_calls: Tool calls grouped by message_id

    Returns:
        list[ChatMessage]: Formatted messages for LLM
    """
    chat_messages: list[ChatMessage] = []

    for message in messages:
        # Convert message based on role
        if message.role == MessageRole.USER:
            chat_messages.append(ChatMessage.user(message.content))

        elif message.role == MessageRole.ASSISTANT:
            # Check if this message has associated tool calls
            if message.id in tool_calls:
                # Create assistant message with tool calls
                # Note: The actual tool_calls structure depends on how they were stored
                # For now, we'll just add the text content
                # TODO: Properly reconstruct tool_calls from database
                chat_messages.append(ChatMessage.assistant(message.content))

                # Add tool results as separate messages
                for tool_call in tool_calls[message.id]:
                    if tool_call.tool_output is not None:
                        # Format tool result
                        result_text = _format_tool_call_summary(tool_call)
                        chat_messages.append(ChatMessage.tool(result_text))
            else:
                chat_messages.append(ChatMessage.assistant(message.content))

        elif message.role == MessageRole.SYSTEM:
            chat_messages.append(ChatMessage.system(message.content))

        elif message.role == MessageRole.TOOL:
            chat_messages.append(ChatMessage.tool(message.content))

    return chat_messages


def context_to_chat_messages(context: ConversationContext) -> list[ChatMessage]:
    """Convert a ConversationContext to LLM chat messages.

    Args:
        context: Conversation context from database

    Returns:
        list[ChatMessage]: Formatted messages for LLM
    """
    return messages_to_chat_messages(context.messages, context.tool_calls)


def _format_tool_call_summary(tool_call: ToolCall) -> str:
    """Format a tool call into a summary string.

    Args:
        tool_call: Tool call from database

    Returns:
        str: Formatted summary for LLM context
    """
    parts = [f"Tool: {tool_call.tool_name}"]

    if tool_call.status.value == "success":
        parts.append("Status: ✓ Success")
        if tool_call.tool_output:
            # Truncate output if very long
            output_str = str(tool_call.tool_output)
            if len(output_str) > 500:
                output_str = output_str[:500] + "... (truncated)"
            parts.append(f"Result: {output_str}")
    elif tool_call.status.value == "error":
        parts.append("Status: ✗ Error")
        if tool_call.error_message:
            parts.append(f"Error: {tool_call.error_message}")
    elif tool_call.status.value == "denied":
        parts.append("Status: ⊘ Denied")
        if tool_call.error_message:
            parts.append(f"Reason: {tool_call.error_message}")
    else:
        parts.append(f"Status: {tool_call.status.value}")

    if tool_call.duration_ms:
        parts.append(f"Duration: {tool_call.duration_ms}ms")

    return "\n".join(parts)


def truncate_messages_for_context(
    messages: list[ChatMessage], max_messages: int
) -> list[ChatMessage]:
    """Truncate messages to fit within context window.

    Keeps the most recent N messages, ensuring we don't exceed the limit.
    System messages are always kept if present at the start.

    Args:
        messages: List of chat messages
        max_messages: Maximum number of messages to keep

    Returns:
        list[ChatMessage]: Truncated message list
    """
    if len(messages) <= max_messages:
        return messages

    # Check if first message is a system message
    has_system_message = messages and messages[0].role == "system"

    if has_system_message:
        # Keep system message + most recent (max_messages - 1) messages
        system_message = messages[0]
        recent_messages = messages[-(max_messages - 1) :]
        return [system_message] + recent_messages
    else:
        # Just keep most recent max_messages
        return messages[-max_messages:]


def get_conversation_summary(context: ConversationContext) -> str:
    """Get a brief summary of the conversation context.

    Args:
        context: Conversation context

    Returns:
        str: Human-readable summary
    """
    parts = []

    parts.append(f"Session: {context.session.title}")
    parts.append(
        f"Messages: {len(context.messages)} of {context.total_messages} total"
    )

    if context.is_truncated:
        parts.append("(Context truncated - showing recent messages only)")

    # Count tool calls
    total_tool_calls = sum(len(calls) for calls in context.tool_calls.values())
    if total_tool_calls > 0:
        parts.append(f"Tool calls: {total_tool_calls}")

    return " | ".join(parts)
