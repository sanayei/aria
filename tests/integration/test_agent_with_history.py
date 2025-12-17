"""Integration tests for agent with conversation history."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aria.agent.core import Agent
from aria.llm import OllamaClient, ChatMessage
from aria.llm.models import ChatResponse
from aria.memory import ConversationStore
from aria.tools.examples import EchoTool
from aria.tools.registry import ToolRegistry
from aria.ui.console import ARIAConsole


@pytest.fixture
async def temp_conversation_store(tmp_path: Path):
    """Create a temporary conversation store for testing."""
    db_path = tmp_path / "test_conversations.db"
    store = ConversationStore(db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def mock_console():
    """Create a mock console."""
    console = ARIAConsole(no_color=True, verbose=False)
    return console


@pytest.fixture
def mock_registry():
    """Create a registry with test tools."""
    registry = ToolRegistry()
    registry.register(EchoTool())
    return registry


class TestAgentWithHistory:
    """Tests for agent with conversation history integration."""

    @pytest.mark.asyncio
    async def test_start_new_session(
        self, temp_conversation_store: ConversationStore, mock_console, mock_registry
    ):
        """Test starting a new session."""
        # Create mock client
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.settings = AsyncMock()
        mock_client.settings.max_context_messages = 50

        # Create agent with conversation store
        agent = Agent(
            client=mock_client,
            registry=mock_registry,
            console=mock_console,
            conversation_store=temp_conversation_store,
        )

        # Start new session
        session_id = await agent.start_session(title="Test Session")

        assert session_id is not None
        assert agent.current_session_id == session_id

        # Verify session was created in database
        session = await temp_conversation_store.get_session(session_id)
        assert session is not None
        assert session.title == "Test Session"
        assert session.message_count == 0

    @pytest.mark.asyncio
    async def test_resume_existing_session(
        self, temp_conversation_store: ConversationStore, mock_console, mock_registry
    ):
        """Test resuming an existing session."""
        # Create a session first
        session = await temp_conversation_store.create_session("Existing Session")
        await temp_conversation_store.add_message(
            session.id, "user", "Previous message"
        )

        # Create agent
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.settings = AsyncMock()
        mock_client.settings.max_context_messages = 50

        agent = Agent(
            client=mock_client,
            registry=mock_registry,
            console=mock_console,
            conversation_store=temp_conversation_store,
        )

        # Resume session
        resumed_session_id = await agent.start_session(session_id=session.id)

        assert resumed_session_id == session.id
        assert agent.current_session_id == session.id

        # Verify session exists
        resumed_session = await temp_conversation_store.get_session(resumed_session_id)
        assert resumed_session.message_count == 1

    @pytest.mark.asyncio
    async def test_chat_saves_messages(
        self, temp_conversation_store: ConversationStore, mock_console, mock_registry
    ):
        """Test that chat() saves messages to database."""
        # Create mock client that returns a response
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.settings = AsyncMock()
        mock_client.settings.max_context_messages = 50
        mock_client.settings.ollama_temperature = 0.7
        mock_client.settings.agent_max_iterations = 10

        # Mock the chat_with_tools response
        mock_response = ChatResponse(
            message=ChatMessage.assistant("Hello! How can I help you?"),
            model="test-model",
            has_tool_calls=False,
        )
        mock_client.chat_with_tools = AsyncMock(return_value=mock_response)

        # Create agent
        agent = Agent(
            client=mock_client,
            registry=mock_registry,
            console=mock_console,
            conversation_store=temp_conversation_store,
        )

        # Start session
        session_id = await agent.start_session()

        # Send a message using chat()
        response = await agent.chat("Hello ARIA")

        assert response == "Hello! How can I help you?"

        # Verify messages were saved
        messages = await temp_conversation_store.get_messages(session_id)
        assert len(messages) == 2
        assert messages[0].content == "Hello ARIA"
        assert messages[0].role.value == "user"
        assert messages[1].content == "Hello! How can I help you?"
        assert messages[1].role.value == "assistant"

        # Verify session message count updated
        session = await temp_conversation_store.get_session(session_id)
        assert session.message_count == 2

    @pytest.mark.asyncio
    async def test_chat_loads_conversation_history(
        self, temp_conversation_store: ConversationStore, mock_console, mock_registry
    ):
        """Test that chat() loads previous conversation history."""
        # Create session with some history
        session = await temp_conversation_store.create_session("Test")
        await temp_conversation_store.add_message(session.id, "user", "First message")
        await temp_conversation_store.add_message(
            session.id, "assistant", "First response"
        )

        # Create mock client
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.settings = AsyncMock()
        mock_client.settings.max_context_messages = 50
        mock_client.settings.ollama_temperature = 0.7

        mock_response = ChatResponse(
            message=ChatMessage.assistant("Second response"),
            model="test-model",
            has_tool_calls=False,
        )
        mock_client.chat_with_tools = AsyncMock(return_value=mock_response)

        # Create agent and resume session
        agent = Agent(
            client=mock_client,
            registry=mock_registry,
            console=mock_console,
            conversation_store=temp_conversation_store,
        )
        await agent.start_session(session_id=session.id)

        # Send another message
        response = await agent.chat("Second message")

        # Verify the call to chat_with_tools received the history
        # The mock should have been called with messages including history
        call_args = mock_client.chat_with_tools.call_args
        messages = call_args.kwargs["messages"]

        # Should have: system prompt + 2 history messages + current user message
        # (Note: The exact count depends on implementation details)
        assert len(messages) >= 4  # At least system + 2 history + current

    @pytest.mark.asyncio
    async def test_chat_without_conversation_store(
        self, mock_console, mock_registry
    ):
        """Test that chat() falls back to run() without conversation store."""
        # Create mock client
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.settings = AsyncMock()
        mock_client.settings.ollama_temperature = 0.7

        mock_response = ChatResponse(
            message=ChatMessage.assistant("Response without history"),
            model="test-model",
            has_tool_calls=False,
        )
        mock_client.chat_with_tools = AsyncMock(return_value=mock_response)

        # Create agent WITHOUT conversation store
        agent = Agent(
            client=mock_client,
            registry=mock_registry,
            console=mock_console,
            conversation_store=None,  # No store
        )

        # chat() should work but not persist anything
        response = await agent.chat("Hello")

        assert response == "Response without history"
        # No database operations should have occurred

    @pytest.mark.asyncio
    async def test_chat_raises_without_active_session(
        self, temp_conversation_store: ConversationStore, mock_console, mock_registry
    ):
        """Test that chat() raises error if no session is active."""
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.settings = AsyncMock()
        mock_client.settings.max_context_messages = 50

        agent = Agent(
            client=mock_client,
            registry=mock_registry,
            console=mock_console,
            conversation_store=temp_conversation_store,
        )

        # Don't start a session - should raise error
        with pytest.raises(Exception, match="No active session"):
            await agent.chat("Hello")

    @pytest.mark.asyncio
    async def test_chat_saves_tool_calls(
        self, temp_conversation_store: ConversationStore, mock_console, mock_registry
    ):
        """Test that chat() saves tool calls to database."""
        from aria.llm.models import ToolCall as LLMToolCall, ToolFunction

        # Create mock client
        mock_client = AsyncMock(spec=OllamaClient)
        mock_client.settings = AsyncMock()
        mock_client.settings.max_context_messages = 50
        mock_client.settings.ollama_temperature = 0.7
        mock_client.settings.agent_max_iterations = 10

        # First response with tool call
        tool_call = LLMToolCall(
            id="test-call-1",
            type="function",
            function=ToolFunction(
                name="echo",
                arguments={"message": "test message"}
            )
        )
        tool_response = ChatResponse(
            message=ChatMessage.assistant("", tool_calls=[tool_call]),
            model="test-model",
            has_tool_calls=True,
        )

        # Final response after tool execution
        final_response = ChatResponse(
            message=ChatMessage.assistant("Tool executed successfully!"),
            model="test-model",
            has_tool_calls=False,
        )

        # Mock chat_with_tools to return tool call first, then final response
        mock_client.chat_with_tools = AsyncMock(side_effect=[tool_response, final_response])

        # Create agent
        agent = Agent(
            client=mock_client,
            registry=mock_registry,
            console=mock_console,
            conversation_store=temp_conversation_store,
        )

        # Start session
        session_id = await agent.start_session()

        # Send a message that triggers a tool call
        response = await agent.chat("Use the echo tool")

        # Verify messages were saved
        messages = await temp_conversation_store.get_messages(session_id)
        assert len(messages) == 2  # User message + assistant response

        # Get the assistant message
        assistant_msg = messages[1]
        assert assistant_msg.role.value == "assistant"

        # Verify tool calls were saved
        context = await temp_conversation_store.get_context(session_id)
        assert assistant_msg.id in context.tool_calls

        tool_calls = context.tool_calls[assistant_msg.id]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "echo"
        assert tool_calls[0].tool_input == {"message": "test message"}
        assert tool_calls[0].status.value == "success"
        assert tool_calls[0].tool_output is not None


class TestConversationContext:
    """Tests for conversation context formatting."""

    @pytest.mark.asyncio
    async def test_context_to_chat_messages(
        self, temp_conversation_store: ConversationStore
    ):
        """Test converting conversation context to chat messages."""
        from aria.memory.context import context_to_chat_messages

        # Create session with messages
        session = await temp_conversation_store.create_session("Test")
        await temp_conversation_store.add_message(session.id, "user", "Hello")
        await temp_conversation_store.add_message(session.id, "assistant", "Hi there")

        # Get context
        context = await temp_conversation_store.get_context(session.id)

        # Convert to chat messages
        chat_messages = context_to_chat_messages(context)

        assert len(chat_messages) == 2
        assert chat_messages[0].role == "user"
        assert chat_messages[0].content == "Hello"
        assert chat_messages[1].role == "assistant"
        assert chat_messages[1].content == "Hi there"
