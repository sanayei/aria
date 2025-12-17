"""Core agent implementation using the ReAct pattern.

This module provides the main Agent class that orchestrates the
Thought → Action → Observation loop using LLM and tools.
"""

import logging
import time
from typing import AsyncIterator, NamedTuple, Any

from aria.llm import OllamaClient, ChatMessage
from aria.tools.registry import ToolRegistry
from aria.tools.base import ToolResult
from aria.ui.console import ARIAConsole
from aria.approval import ApprovalHandler
from aria.agent.executor import ToolExecutor
from aria.agent.prompts import (
    SYSTEM_PROMPT,
    create_system_message_with_tools,
    format_tool_result,
    format_approval_denied_message,
)
from aria.logging import get_logger, AsyncTimer
from aria.memory import ConversationStore, MessageRole, ToolCallStatus
from aria.memory.context import context_to_chat_messages

logger = get_logger("aria.agent.core")


class AgentError(Exception):
    """Exception raised when agent execution fails."""

    pass


class ToolCallRecord(NamedTuple):
    """Record of a tool call execution.

    This is used to track tool calls during agent execution so they
    can be persisted to the conversation history database.
    """
    tool_name: str
    tool_input: dict[str, Any]
    result: ToolResult
    execution_time_ms: float | None = None


class Agent:
    """ARIA agent implementing the ReAct pattern.

    The agent follows the Thought → Action → Observation → Thought cycle:
    1. Receives user input and analyzes it
    2. Calls LLM to decide on actions (tool use or direct response)
    3. Executes tools if needed (with approval handling)
    4. Observes tool results and incorporates them
    5. Decides if more actions are needed or provides final response

    The agent handles:
    - Tool execution with risk-based approval
    - Multi-step reasoning and tool chaining
    - Streaming responses
    - Error recovery
    - Safety limits (max iterations)
    """

    def __init__(
        self,
        client: OllamaClient,
        registry: ToolRegistry,
        console: ARIAConsole,
        approval_handler: ApprovalHandler | None = None,
        max_iterations: int = 10,
        conversation_store: ConversationStore | None = None,
    ):
        """Initialize the agent.

        Args:
            client: Ollama client for LLM calls
            registry: Tool registry for available tools
            console: Console for user interaction
            approval_handler: Handler for tool approvals (creates default if None)
            max_iterations: Maximum number of ReAct loop iterations
            conversation_store: Optional conversation store for history persistence
        """
        self.client = client
        self.registry = registry
        self.console = console
        self.max_iterations = max_iterations
        self.conversation_store = conversation_store
        self.current_session_id: str | None = None

        # Create approval handler if not provided
        if approval_handler is None:
            from aria.approval import ApprovalHandler
            approval_handler = ApprovalHandler(console)

        self.approval_handler = approval_handler

        # Create tool executor
        self.executor = ToolExecutor(
            registry=registry,
            approval_handler=approval_handler,
            console=console,
        )

    async def start_session(self, session_id: str | None = None, title: str | None = None) -> str:
        """Start a new session or resume an existing one.

        Args:
            session_id: Optional session ID to resume. If None, creates new session.
            title: Optional title for new session

        Returns:
            str: The session ID

        Raises:
            AgentError: If conversation store is not configured
        """
        if self.conversation_store is None:
            raise AgentError("Cannot start session: conversation store not configured")

        if session_id:
            # Resume existing session
            session = await self.conversation_store.get_session(session_id)
            if session is None:
                raise AgentError(f"Session not found: {session_id}")
            self.current_session_id = session_id
            logger.info("Session resumed", session_id=session_id, title=session.title)
        else:
            # Create new session
            session = await self.conversation_store.create_session(title)
            self.current_session_id = session.id
            logger.info("New session created", session_id=session.id, title=session.title)

        return self.current_session_id

    async def chat(self, user_message: str) -> str:
        """Process a message with automatic history persistence.

        This is a convenience method that:
        1. Loads conversation history from the current session
        2. Saves the user message
        3. Runs the agent
        4. Saves the assistant response and tool calls
        5. Returns the response

        Args:
            user_message: The user's input message

        Returns:
            str: The agent's response

        Raises:
            AgentError: If no session is active or conversation store not configured
        """
        if self.conversation_store is None:
            # Fall back to run() without persistence
            logger.warning("No conversation store - running without persistence")
            response, _ = await self.run(user_message)
            return response

        if self.current_session_id is None:
            raise AgentError("No active session. Call start_session() first.")

        # Load conversation context from database
        context = await self.conversation_store.get_context(
            self.current_session_id,
            max_messages=self.client.settings.max_context_messages,
        )

        # Convert to chat messages for LLM
        conversation_history = context_to_chat_messages(context)

        # Save user message to database
        user_msg = await self.conversation_store.add_message(
            self.current_session_id,
            role="user",
            content=user_message,
        )

        # Run the agent
        response, tool_calls = await self.run(user_message, conversation_history)

        # Save assistant response to database
        assistant_msg = await self.conversation_store.add_message(
            self.current_session_id,
            role="assistant",
            content=response,
        )

        # Save tool calls to database
        for tool_call_record in tool_calls:
            # Determine status based on result
            if tool_call_record.result.success:
                status = "success"
                error_message = None
                tool_output = {"data": tool_call_record.result.data}
            else:
                # Check if it was denied or just an error
                if tool_call_record.result.error and "denied" in tool_call_record.result.error.lower():
                    status = "denied"
                else:
                    status = "error"
                error_message = tool_call_record.result.error
                tool_output = None

            await self.conversation_store.add_tool_call(
                message_id=assistant_msg.id,
                tool_name=tool_call_record.tool_name,
                tool_input=tool_call_record.tool_input,
                tool_output=tool_output,
                status=status,
                error_message=error_message,
            )

        logger.debug(
            "Messages and tool calls saved to session",
            session_id=self.current_session_id,
            tool_calls_count=len(tool_calls),
        )

        return response

    async def run(
        self,
        user_message: str,
        conversation_history: list[ChatMessage] | None = None,
    ) -> tuple[str, list[ToolCallRecord]]:
        """Run the agent on a user message.

        Executes the ReAct loop:
        1. Build context with system prompt and history
        2. Call LLM with available tools
        3. If tool calls in response:
           - Execute tools (with approval)
           - Add results to context
           - Call LLM again with results
           - Repeat until no more tool calls or max iterations
        4. Return final text response and tool call records

        Args:
            user_message: The user's input message
            conversation_history: Previous conversation messages (optional)

        Returns:
            tuple[str, list[ToolCallRecord]]: The agent's final response and tool call records

        Raises:
            AgentError: If agent execution fails
        """
        # Start timing the entire run
        run_start = time.perf_counter()
        logger.info(
            "Agent.run() started",
            user_message_preview=user_message[:100],
            history_length=len(conversation_history) if conversation_history else 0,
        )

        # Initialize conversation
        if conversation_history is None:
            conversation_history = []

        # Build initial messages
        messages = self._build_messages(user_message, conversation_history)

        # Get available tools
        tools = self.registry.get_tools_for_ollama()
        logger.debug(
            "Tools prepared",
            tool_count=len(tools),
            tool_names=[t.function.get("name", "unknown") if isinstance(t.function, dict) else t.function.name for t in tools],
        )

        # Track tool calls for conversation history
        tool_call_records: list[ToolCallRecord] = []

        # ReAct loop
        iteration = 0
        final_response = ""

        while iteration < self.max_iterations:
            iteration += 1
            iteration_start = time.perf_counter()
            logger.info(
                f"ReAct iteration {iteration}/{self.max_iterations} starting",
                message_count=len(messages),
            )

            try:
                # Call LLM with tools
                with self.console.thinking(f"Thinking... (iteration {iteration})"):
                    async with AsyncTimer(
                        f"LLM call (iteration {iteration})", logger
                    ) as llm_timer:
                        response = await self.client.chat_with_tools(
                            messages=messages,
                            tools=tools,
                            temperature=self.client.settings.ollama_temperature,
                        )

                # Check for tool calls
                if response.has_tool_calls:
                    tool_names_called = [
                        tc.function.name for tc in response.message.tool_calls
                    ]
                    logger.info(
                        "Tool calls requested",
                        count=len(response.message.tool_calls),
                        tools=tool_names_called,
                    )

                    # Add assistant message with tool calls
                    messages.append(response.message)

                    # Execute each tool
                    for tool_call in response.message.tool_calls:
                        tool_name = tool_call.function.name
                        arguments = tool_call.function.arguments

                        logger.debug(
                            f"Executing tool: {tool_name}",
                            arguments=arguments,
                        )

                        # Execute tool with timing
                        tool_exec_start = time.perf_counter()
                        async with AsyncTimer(f"Tool execution: {tool_name}", logger):
                            result = await self.executor.execute(tool_name, arguments)
                        tool_exec_elapsed = time.perf_counter() - tool_exec_start

                        # Record tool call for conversation history
                        tool_call_records.append(ToolCallRecord(
                            tool_name=tool_name,
                            tool_input=arguments,
                            result=result,
                            execution_time_ms=tool_exec_elapsed * 1000,
                        ))

                        logger.info(
                            f"Tool {tool_name} complete",
                            success=result.success,
                            has_error=bool(result.error),
                        )

                        # Format result message
                        result_text = format_tool_result(tool_name, result)

                        # If tool was denied, add that info
                        if not result.success and result.error and "denied" in result.error.lower():
                            result_text = format_approval_denied_message(
                                tool_name,
                                result.error,
                            )

                        # Add tool result to messages
                        tool_message = ChatMessage.tool(result_text)
                        messages.append(tool_message)

                    # Log iteration completion
                    iteration_elapsed = time.perf_counter() - iteration_start
                    logger.debug(
                        f"Iteration {iteration} complete (continuing)",
                        iteration_time_s=f"{iteration_elapsed:.3f}",
                    )
                    # Continue loop to get LLM's response to tool results
                    continue

                else:
                    # No tool calls - we have a final response
                    final_response = response.message.content
                    iteration_elapsed = time.perf_counter() - iteration_start
                    logger.info(
                        "Final response received",
                        iteration=iteration,
                        iteration_time_s=f"{iteration_elapsed:.3f}",
                    )
                    break

            except Exception as e:
                error_msg = f"Error in agent iteration {iteration}: {e}"
                logger.exception(error_msg)
                self.console.error(error_msg, exception=e)
                raise AgentError(error_msg) from e

        # Check if we hit max iterations
        if iteration >= self.max_iterations and not final_response:
            warning_msg = (
                f"Reached maximum iterations ({self.max_iterations}). "
                "Requesting final response from LLM."
            )
            logger.warning(warning_msg)
            self.console.warning(warning_msg)

            # Try to get a final response
            try:
                messages.append(
                    ChatMessage.system(
                        "You have used many tools. Please provide a final response "
                        "to the user based on the information gathered."
                    )
                )

                with self.console.thinking("Generating final response..."):
                    response = await self.client.chat(
                        messages=messages,
                        temperature=self.client.settings.ollama_temperature,
                    )

                final_response = response.message.content

            except Exception as e:
                error_msg = f"Failed to get final response: {e}"
                logger.error(error_msg)
                final_response = (
                    "I apologize, but I encountered an issue completing this task. "
                    "I may have tried to use too many tools. Could you please rephrase "
                    "your request or break it into smaller parts?"
                )

        # Log total run time
        run_elapsed = time.perf_counter() - run_start
        logger.info(
            "Agent.run() complete",
            total_iterations=iteration,
            total_time_s=f"{run_elapsed:.3f}",
            response_length=len(final_response),
            tool_calls_count=len(tool_call_records),
        )

        return final_response, tool_call_records

    async def stream_response(
        self,
        user_message: str,
        conversation_history: list[ChatMessage] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the agent's response in real-time.

        Note: This is a simpler version that doesn't support tool use yet.
        For tool-using responses, use run() instead.

        Args:
            user_message: The user's input message
            conversation_history: Previous conversation messages (optional)

        Yields:
            str: Chunks of the response
        """
        if conversation_history is None:
            conversation_history = []

        messages = self._build_messages(user_message, conversation_history)

        # For now, stream without tools (tools require non-streaming for tool calls)
        async for chunk in self.client.stream_chat(
            messages=messages,
            temperature=self.client.settings.ollama_temperature,
        ):
            if chunk.message.content:
                yield chunk.message.content

    def _build_messages(
        self,
        user_message: str,
        conversation_history: list[ChatMessage],
    ) -> list[ChatMessage]:
        """Build the message list for LLM call.

        Args:
            user_message: Current user message
            conversation_history: Previous messages

        Returns:
            list[ChatMessage]: Complete message list with system prompt
        """
        messages = []

        # Add system message with tool information
        tools = self.registry.list_tools()
        system_message_text = create_system_message_with_tools(tools)
        messages.append(ChatMessage.system(system_message_text))

        # Add conversation history (if any)
        messages.extend(conversation_history)

        # Add current user message
        messages.append(ChatMessage.user(user_message))

        return messages
