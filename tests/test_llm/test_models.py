"""Tests for LLM models."""

import pytest
from datetime import datetime

from aria.llm.models import (
    ChatMessage,
    ToolCall,
    ToolFunction,
    ToolDefinition,
    ChatResponse,
    ModelInfo,
    ModelCapabilities,
)


class TestChatMessage:
    """Test ChatMessage model."""

    def test_system_message(self):
        """Test creating a system message."""
        msg = ChatMessage.system("You are a helpful assistant")

        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant"
        assert msg.tool_calls == []
        assert msg.images == []

    def test_user_message(self):
        """Test creating a user message."""
        msg = ChatMessage.user("Hello!")

        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.tool_calls == []

    def test_user_message_with_images(self):
        """Test user message with images."""
        msg = ChatMessage.user("What's in this image?", images=["base64data"])

        assert msg.role == "user"
        assert msg.content == "What's in this image?"
        assert msg.images == ["base64data"]

    def test_assistant_message(self):
        """Test creating an assistant message."""
        msg = ChatMessage.assistant("I can help you with that")

        assert msg.role == "assistant"
        assert msg.content == "I can help you with that"
        assert msg.tool_calls == []

    def test_assistant_message_with_tool_calls(self):
        """Test assistant message with tool calls."""
        tool_call = ToolCall(
            function=ToolFunction(
                name="get_weather",
                arguments={"location": "Tokyo"},
            )
        )
        msg = ChatMessage.assistant("Let me check the weather", tool_calls=[tool_call])

        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_weather"

    def test_tool_message(self):
        """Test creating a tool message."""
        msg = ChatMessage.tool("Weather: 22°C, sunny")

        assert msg.role == "tool"
        assert msg.content == "Weather: 22°C, sunny"

    def test_model_dump_ollama(self):
        """Test dumping message in Ollama format."""
        msg = ChatMessage.user("Hello")
        dumped = msg.model_dump_ollama()

        assert dumped["role"] == "user"
        assert dumped["content"] == "Hello"
        assert "tool_calls" not in dumped  # Empty tool_calls not included

    def test_model_dump_ollama_with_tool_calls(self):
        """Test dumping message with tool calls in Ollama format."""
        tool_call = ToolCall(
            function=ToolFunction(
                name="get_weather",
                arguments={"location": "Tokyo"},
            )
        )
        msg = ChatMessage.assistant("Checking", tool_calls=[tool_call])
        dumped = msg.model_dump_ollama()

        assert dumped["role"] == "assistant"
        assert "tool_calls" in dumped
        assert len(dumped["tool_calls"]) == 1
        assert dumped["tool_calls"][0]["function"]["name"] == "get_weather"


class TestToolDefinition:
    """Test ToolDefinition model."""

    def test_from_schema(self):
        """Test creating a tool definition from schema components."""
        tool_def = ToolDefinition.from_schema(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        )

        assert tool_def.type == "function"
        assert tool_def.function["name"] == "get_weather"
        assert tool_def.function["description"] == "Get weather for a location"
        assert "parameters" in tool_def.function


class TestChatResponse:
    """Test ChatResponse model."""

    def test_basic_response(self):
        """Test basic chat response."""
        response = ChatResponse(
            model="qwen3:30b-a3b",
            message=ChatMessage.assistant("Hello!"),
            done=True,
        )

        assert response.model == "qwen3:30b-a3b"
        assert response.message.content == "Hello!"
        assert response.done is True

    def test_has_tool_calls(self):
        """Test has_tool_calls property."""
        # Response without tool calls
        response1 = ChatResponse(
            model="test",
            message=ChatMessage.assistant("Hello"),
        )
        assert response1.has_tool_calls is False

        # Response with tool calls
        tool_call = ToolCall(function=ToolFunction(name="test", arguments={}))
        response2 = ChatResponse(
            model="test",
            message=ChatMessage.assistant("Calling tool", tool_calls=[tool_call]),
        )
        assert response2.has_tool_calls is True

    def test_metadata_property(self):
        """Test metadata property."""
        response = ChatResponse(
            model="test",
            message=ChatMessage.assistant("Test"),
            eval_count=100,
            eval_duration=1_000_000_000,  # 1 second in nanoseconds
        )

        metadata = response.metadata
        assert metadata.eval_count == 100
        assert metadata.tokens_per_second is not None
        assert metadata.tokens_per_second == pytest.approx(100.0, rel=0.01)


class TestModelInfo:
    """Test ModelInfo model."""

    def test_size_properties(self):
        """Test size conversion properties."""
        model = ModelInfo(
            name="test",
            size=1024 * 1024 * 1024 * 2,  # 2 GB
        )

        assert model.size_mb == pytest.approx(2048.0, rel=0.01)
        assert model.size_gb == pytest.approx(2.0, rel=0.01)

    def test_no_size(self):
        """Test model info without size."""
        model = ModelInfo(name="test")

        assert model.size_mb is None
        assert model.size_gb is None


class TestModelCapabilities:
    """Test ModelCapabilities model."""

    def test_default_capabilities(self):
        """Test default model capabilities."""
        caps = ModelCapabilities(name="test")

        assert caps.name == "test"
        assert caps.context_length == 8192
        assert caps.supports_tools is True
        assert caps.supports_vision is False
        assert caps.max_tokens is None

    def test_custom_capabilities(self):
        """Test custom model capabilities."""
        caps = ModelCapabilities(
            name="custom",
            context_length=32768,
            supports_tools=True,
            supports_vision=True,
            recommended_temperature=0.5,
        )

        assert caps.context_length == 32768
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.recommended_temperature == 0.5
