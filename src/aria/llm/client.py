"""Async client for Ollama API.

This module provides an async wrapper around the Ollama API with support for
chat completions, tool calling, streaming, and model management.
"""

import asyncio
import logging
from typing import AsyncIterator, Any
from contextlib import asynccontextmanager

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from aria.config import Settings, get_settings
from aria.llm.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ModelInfo,
    ModelList,
    ToolDefinition,
    ModelCapabilities,
    DEFAULT_MODEL_CAPABILITIES,
)
from aria.logging import get_logger, AsyncTimer

logger = get_logger("aria.llm.client")


# =============================================================================
# Exceptions
# =============================================================================


class OllamaError(Exception):
    """Base exception for Ollama client errors."""

    pass


class OllamaConnectionError(OllamaError):
    """Raised when connection to Ollama fails."""

    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested model is not available."""

    pass


class OllamaAPIError(OllamaError):
    """Raised when the Ollama API returns an error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# =============================================================================
# Ollama Client
# =============================================================================


class OllamaClient:
    """Async client for interacting with Ollama API.

    This client handles all communication with the Ollama server, including
    chat completions, tool calling, streaming responses, and model management.

    Attributes:
        base_url: Base URL of the Ollama server
        model: Default model name to use
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        settings: Settings | None = None,
    ):
        """Initialize the Ollama client.

        Args:
            base_url: Ollama server URL (defaults to settings)
            model: Default model name (defaults to settings)
            timeout: Request timeout in seconds (defaults to settings)
            settings: Settings instance (uses global if not provided)
        """
        self.settings = settings or get_settings()
        self.base_url = (base_url or self.settings.ollama_base_url).rstrip("/")
        self.model = model or self.settings.ollama_model
        self.timeout = timeout or self.settings.ollama_timeout
        self.max_retries = 3

        self._client: httpx.AsyncClient | None = None
        self._model_capabilities: dict[str, ModelCapabilities] = DEFAULT_MODEL_CAPABILITIES.copy()

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[httpx.AsyncClient]:
        """Get or create an async HTTP client.

        Yields:
            httpx.AsyncClient: The HTTP client instance
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )

        try:
            yield self._client
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Error: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaConnectionError(
                f"Request to Ollama timed out after {self.timeout}s. "
                f"Consider increasing OLLAMA_TIMEOUT. Error: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise OllamaModelNotFoundError(
                    f"Model '{self.model}' not found. "
                    f"Run 'ollama pull {self.model}' to download it."
                ) from e
            else:
                error_detail = e.response.text
                raise OllamaAPIError(
                    f"Ollama API error: {error_detail}",
                    status_code=e.response.status_code,
                ) from e

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "OllamaClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    # =========================================================================
    # Health and connectivity
    # =========================================================================

    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def health_check(self) -> bool:
        """Check if Ollama is running and accessible.

        Returns:
            bool: True if Ollama is healthy

        Raises:
            OllamaConnectionError: If connection fails
        """
        try:
            async with self._get_client() as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                logger.info("Ollama health check passed")
                return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            raise

    # =========================================================================
    # Model management
    # =========================================================================

    async def list_models(self) -> list[ModelInfo]:
        """List all available models.

        Returns:
            list[ModelInfo]: List of available models

        Raises:
            OllamaConnectionError: If connection fails
            OllamaAPIError: If the API returns an error
        """
        async with self._get_client() as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            data = response.json()
            model_list = ModelList(**data)

            logger.info(f"Found {len(model_list.models)} available models")
            return model_list.models

    async def model_exists(self, model_name: str) -> bool:
        """Check if a specific model is available.

        Args:
            model_name: Name of the model to check

        Returns:
            bool: True if the model exists
        """
        try:
            models = await self.list_models()
            return any(m.name == model_name for m in models)
        except Exception as e:
            logger.warning(f"Failed to check if model exists: {e}")
            return False

    def get_model_capabilities(self, model_name: str | None = None) -> ModelCapabilities:
        """Get capabilities for a model.

        Args:
            model_name: Name of the model (uses default if not provided)

        Returns:
            ModelCapabilities: Model capabilities configuration
        """
        model_name = model_name or self.model

        # Check for exact match first
        if model_name in self._model_capabilities:
            return self._model_capabilities[model_name]

        # Check for partial matches (e.g., "qwen2.5" in "qwen2.5:32b")
        for known_model, capabilities in self._model_capabilities.items():
            if known_model in model_name or model_name in known_model:
                return capabilities

        # Return default capabilities
        logger.warning(f"Unknown model '{model_name}', using default capabilities")
        return ModelCapabilities(
            name=model_name,
            context_length=8192,
            supports_tools=True,
            supports_vision=False,
        )

    # =========================================================================
    # Chat completions
    # =========================================================================

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException,)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **options: Any,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: List of chat messages
            model: Model to use (defaults to client default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **options: Additional model options

        Returns:
            ChatResponse: The model's response

        Raises:
            OllamaConnectionError: If connection fails
            OllamaModelNotFoundError: If model not found
            OllamaAPIError: If the API returns an error
        """
        model = model or self.model

        # Build options dict
        model_options = options.copy()
        if temperature is not None:
            model_options["temperature"] = temperature
        if max_tokens is not None:
            model_options["num_predict"] = max_tokens

        # Create request
        request = ChatRequest(
            model=model,
            messages=messages,
            stream=False,
            options=model_options,
        )

        # Calculate approximate prompt size
        prompt_size = sum(len(m.content or "") for m in messages)
        logger.debug(
            "chat() called",
            model=model,
            message_count=len(messages),
            prompt_chars=prompt_size,
        )

        async with AsyncTimer(f"LLM chat request ({model})", logger) as timer:
            async with self._get_client() as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=request.model_dump_ollama(),
                )
                response.raise_for_status()

                data = response.json()
                chat_response = ChatResponse(**data)

        logger.debug(
            "chat() complete",
            elapsed_s=f"{timer.elapsed:.3f}",
            response_chars=len(chat_response.message.content or ""),
            tokens_per_s=f"{chat_response.metadata.tokens_per_second:.1f}"
            if chat_response.metadata.tokens_per_second
            else "unknown",
        )

        return chat_response

    async def chat_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        model: str | None = None,
        temperature: float | None = None,
        **options: Any,
    ) -> ChatResponse:
        """Send a chat completion request with tool support.

        Args:
            messages: List of chat messages
            tools: Available tools for the LLM
            model: Model to use (defaults to client default)
            temperature: Sampling temperature
            **options: Additional model options

        Returns:
            ChatResponse: The model's response (may include tool calls)

        Raises:
            OllamaConnectionError: If connection fails
            OllamaModelNotFoundError: If model not found
            OllamaAPIError: If the API returns an error
        """
        model = model or self.model

        # Check if model supports tools
        capabilities = self.get_model_capabilities(model)
        if not capabilities.supports_tools:
            logger.warning(f"Model {model} may not support tool calling")

        # Build options dict
        model_options = options.copy()
        if temperature is not None:
            model_options["temperature"] = temperature
        else:
            # Use recommended temperature if not specified
            model_options["temperature"] = capabilities.recommended_temperature

        # Add Qwen3-specific options for better tool calling
        if "qwen3" in model.lower():
            # Qwen3 models benefit from these settings for tool calling
            if "think" not in model_options:
                model_options["think"] = False  # Disable thinking mode for faster tool calls
            if "num_predict" not in model_options:
                model_options["num_predict"] = -2  # Allow full generation
            logger.debug(f"Applied Qwen3-specific options: think={model_options.get('think')}")

        # Create request
        request = ChatRequest(
            model=model,
            messages=messages,
            tools=tools,
            stream=False,
            options=model_options,
        )

        # Calculate sizes for logging
        prompt_size = sum(len(m.content or "") for m in messages)
        tools_json_size = len(str([t.model_dump() for t in tools]))
        tool_names = [t.function.get("name", "unknown") if isinstance(t.function, dict) else t.function.name for t in tools]

        logger.debug(
            "chat_with_tools() called",
            model=model,
            message_count=len(messages),
            tool_count=len(tools),
            tool_names=tool_names,
            prompt_chars=prompt_size,
            tools_schema_chars=tools_json_size,
        )

        # Dump full request for debugging
        request_json = request.model_dump_ollama()
        logger.debug(
            "Full Ollama request",
            request_json=request_json,
        )

        async with AsyncTimer(f"LLM chat_with_tools request ({model})", logger) as timer:
            async with self._get_client() as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=request_json,
                )
                response.raise_for_status()

                data = response.json()

                # Log full response for debugging
                logger.debug(
                    "Full Ollama response",
                    response_data=data,
                )

                chat_response = ChatResponse(**data)

        tool_call_info = []
        if chat_response.has_tool_calls:
            tool_call_info = [tc.function.name for tc in chat_response.message.tool_calls]

        logger.debug(
            "chat_with_tools() complete",
            elapsed_s=f"{timer.elapsed:.3f}",
            has_tool_calls=chat_response.has_tool_calls,
            tool_calls=tool_call_info,
            response_chars=len(chat_response.message.content or ""),
        )

        return chat_response

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        tools: list[ToolDefinition] | None = None,
        **options: Any,
    ) -> AsyncIterator[ChatResponse]:
        """Stream a chat completion request.

        Args:
            messages: List of chat messages
            model: Model to use (defaults to client default)
            temperature: Sampling temperature
            tools: Optional tools for the LLM
            **options: Additional model options

        Yields:
            ChatResponse: Chunks of the model's response

        Raises:
            OllamaConnectionError: If connection fails
            OllamaModelNotFoundError: If model not found
            OllamaAPIError: If the API returns an error
        """
        model = model or self.model

        # Build options dict
        model_options = options.copy()
        if temperature is not None:
            model_options["temperature"] = temperature

        # Create request
        request = ChatRequest(
            model=model,
            messages=messages,
            tools=tools or [],
            stream=True,
            options=model_options,
        )

        logger.debug(f"Starting streaming chat with {model}")

        async with self._get_client() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=request.model_dump_ollama(),
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        import json
                        data = json.loads(line)
                        chunk = ChatResponse(**data)
                        yield chunk

                        if chunk.done:
                            logger.debug("Stream completed")
                            break
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse streaming response: {e}")
                        continue

    # =========================================================================
    # Convenience methods
    # =========================================================================

    async def simple_chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a simple chat request and return just the text response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model to use (defaults to client default)
            temperature: Sampling temperature

        Returns:
            str: The model's text response
        """
        messages = []
        if system_prompt:
            messages.append(ChatMessage.system(system_prompt))
        messages.append(ChatMessage.user(prompt))

        response = await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
        )

        return response.message.content

    async def stream_simple_chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream a simple chat request yielding text chunks.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model to use (defaults to client default)
            temperature: Sampling temperature

        Yields:
            str: Text chunks from the model's response
        """
        messages = []
        if system_prompt:
            messages.append(ChatMessage.system(system_prompt))
        messages.append(ChatMessage.user(prompt))

        async for chunk in self.stream_chat(
            messages=messages,
            model=model,
            temperature=temperature,
        ):
            if chunk.message.content:
                yield chunk.message.content
