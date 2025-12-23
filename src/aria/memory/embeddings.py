"""Embedding providers for semantic search."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import httpx

from aria.config import get_settings
from aria.logging import get_logger

logger = get_logger("aria.memory.embeddings")


class EmbeddingError(Exception):
    """Exception raised when embedding generation fails."""

    pass


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for single text.

        Args:
            text: Text to embed

        Returns:
            list[float]: Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            list[list[float]]: List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension.

        Returns:
            int: Dimension of embedding vectors
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name.

        Returns:
            str: Name of the embedding model
        """
        pass


class OllamaEmbeddings(EmbeddingProvider):
    """Use Ollama's embedding API.

    Supports models like:
    - nomic-embed-text (768 dimensions)
    - mxbai-embed-large (1024 dimensions)
    - snowflake-arctic-embed (1024 dimensions)
    """

    def __init__(self, model: str = "nomic-embed-text", host: str | None = None):
        """Initialize Ollama embeddings provider.

        Args:
            model: Ollama embedding model name
            host: Ollama API host (defaults to settings.ollama_host)
        """
        self._model = model
        self._host = host or get_settings().ollama_host
        self._dimension: int | None = None
        logger.info(f"Initialized OllamaEmbeddings with model '{model}' at {self._host}")

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for single text using Ollama.

        Args:
            text: Text to embed

        Returns:
            list[float]: Embedding vector

        Raises:
            EmbeddingError: If API call fails
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._host}/api/embed",
                    json={"model": self._model, "input": text},
                )
                response.raise_for_status()
                data = response.json()

                embedding = data.get("embeddings")
                if not embedding:
                    raise EmbeddingError("No embedding in Ollama response")

                # Ollama returns list of embeddings, take the first one
                if isinstance(embedding, list) and len(embedding) > 0:
                    embedding = embedding[0]

                # Cache dimension on first call
                if self._dimension is None:
                    self._dimension = len(embedding)

                return embedding

        except httpx.HTTPError as e:
            logger.error(f"Ollama embedding API error: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in embed_text: {e}")
            raise EmbeddingError(f"Embedding failed: {e}") from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Note: Ollama doesn't have a native batch API, so we make parallel requests.

        Args:
            texts: List of texts to embed

        Returns:
            list[list[float]]: List of embedding vectors

        Raises:
            EmbeddingError: If any embedding fails
        """
        if not texts:
            return []

        try:
            # Make parallel requests for efficiency
            tasks = [self.embed_text(text) for text in texts]
            embeddings = await asyncio.gather(*tasks)
            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise EmbeddingError(f"Batch embedding failed: {e}") from e

    @property
    def dimension(self) -> int:
        """Return embedding dimension.

        Returns:
            int: Dimension of embedding vectors (768 for nomic-embed-text)

        Raises:
            EmbeddingError: If dimension not yet determined
        """
        if self._dimension is None:
            # Common dimensions for Ollama models
            known_dimensions = {
                "nomic-embed-text": 768,
                "mxbai-embed-large": 1024,
                "snowflake-arctic-embed": 1024,
                "all-minilm": 384,
            }

            for key, dim in known_dimensions.items():
                if key in self._model.lower():
                    self._dimension = dim
                    break

            if self._dimension is None:
                raise EmbeddingError(
                    f"Unknown dimension for model '{self._model}'. "
                    "Generate at least one embedding to determine dimension."
                )

        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name.

        Returns:
            str: Name of the embedding model
        """
        return self._model


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Use sentence-transformers library (runs locally).

    Popular models:
    - all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
    - all-mpnet-base-v2 (768 dimensions, slower, better quality)
    - multi-qa-mpnet-base-dot-v1 (768 dimensions, optimized for Q&A)
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """Initialize sentence-transformers provider.

        Args:
            model: Sentence transformer model name from HuggingFace
        """
        self._model_name = model
        self._model: Any = None
        self._dimension: int | None = None
        logger.info(f"Initialized SentenceTransformerEmbeddings with model '{model}'")

    def _load_model(self) -> None:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading sentence transformer model '{self._model_name}'...")
                self._model = SentenceTransformer(self._model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded model '{self._model_name}' with dimension {self._dimension}")

            except ImportError as e:
                raise EmbeddingError(
                    "sentence-transformers not installed. "
                    "Install with: uv add sentence-transformers"
                ) from e
            except Exception as e:
                logger.error(f"Failed to load sentence transformer model: {e}")
                raise EmbeddingError(f"Failed to load model '{self._model_name}': {e}") from e

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for single text.

        Args:
            text: Text to embed

        Returns:
            list[float]: Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            self._load_model()

            # Run in thread pool to avoid blocking
            embedding = await asyncio.to_thread(self._model.encode, text)

            # Convert numpy array to list
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise EmbeddingError(f"Embedding failed: {e}") from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            list[list[float]]: List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        try:
            self._load_model()

            # Use batch encoding for efficiency
            embeddings = await asyncio.to_thread(
                self._model.encode,
                texts,
                batch_size=32,
                show_progress_bar=False,
            )

            # Convert numpy arrays to lists
            return [emb.tolist() for emb in embeddings]

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise EmbeddingError(f"Batch embedding failed: {e}") from e

    @property
    def dimension(self) -> int:
        """Return embedding dimension.

        Returns:
            int: Dimension of embedding vectors
        """
        if self._dimension is None:
            self._load_model()

        return self._dimension  # type: ignore

    @property
    def model_name(self) -> str:
        """Return the model name.

        Returns:
            str: Name of the embedding model
        """
        return self._model_name


def get_embedding_provider(provider: str = "ollama", model: str | None = None) -> EmbeddingProvider:
    """Factory function to get an embedding provider.

    Args:
        provider: Provider name ("ollama" or "sentence-transformers")
        model: Model name (uses default if None)

    Returns:
        EmbeddingProvider: Configured embedding provider

    Raises:
        ValueError: If provider is unknown
    """
    if provider == "ollama":
        model = model or "nomic-embed-text"
        return OllamaEmbeddings(model=model)
    elif provider == "sentence-transformers":
        model = model or "all-MiniLM-L6-v2"
        return SentenceTransformerEmbeddings(model=model)
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. Choose 'ollama' or 'sentence-transformers'."
        )
