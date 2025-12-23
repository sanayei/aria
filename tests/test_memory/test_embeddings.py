"""Tests for embedding providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aria.memory.embeddings import (
    EmbeddingError,
    OllamaEmbeddings,
    SentenceTransformerEmbeddings,
    get_embedding_provider,
)


class TestOllamaEmbeddings:
    """Tests for OllamaEmbeddings provider."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test OllamaEmbeddings initialization."""
        embedder = OllamaEmbeddings(model="nomic-embed-text")
        assert embedder.model_name == "nomic-embed-text"

    @pytest.mark.asyncio
    async def test_initialization_with_host(self):
        """Test OllamaEmbeddings initialization with custom host."""
        embedder = OllamaEmbeddings(model="nomic-embed-text", host="http://custom:11434")
        assert embedder._host == "http://custom:11434"

    @pytest.mark.asyncio
    async def test_embed_text_success(self):
        """Test successful text embedding."""
        embedder = OllamaEmbeddings(model="nomic-embed-text")

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            embedding = await embedder.embed_text("test text")

            assert embedding == [0.1, 0.2, 0.3]
            assert embedder.dimension == 3

    @pytest.mark.asyncio
    async def test_embed_text_no_embedding_in_response(self):
        """Test handling of missing embedding in response."""
        embedder = OllamaEmbeddings(model="nomic-embed-text")

        # Mock response with no embedding
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(EmbeddingError, match="No embedding in Ollama response"):
                await embedder.embed_text("test")

    @pytest.mark.asyncio
    async def test_embed_text_http_error(self):
        """Test handling of HTTP errors."""
        embedder = OllamaEmbeddings(model="nomic-embed-text")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.HTTPError("Connection failed")
            )

            with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
                await embedder.embed_text("test")

    @pytest.mark.asyncio
    async def test_embed_batch_success(self):
        """Test batch embedding."""
        embedder = OllamaEmbeddings(model="nomic-embed-text")

        # Mock responses for each text
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            embeddings = await embedder.embed_batch(["text1", "text2", "text3"])

            assert len(embeddings) == 3
            assert all(emb == [0.1, 0.2, 0.3] for emb in embeddings)

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        """Test batch embedding with empty list."""
        embedder = OllamaEmbeddings(model="nomic-embed-text")
        embeddings = await embedder.embed_batch([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_dimension_known_model(self):
        """Test dimension property with known model."""
        embedder = OllamaEmbeddings(model="nomic-embed-text")
        assert embedder.dimension == 768

    @pytest.mark.asyncio
    async def test_dimension_unknown_model(self):
        """Test dimension property with unknown model."""
        embedder = OllamaEmbeddings(model="unknown-model")

        with pytest.raises(EmbeddingError, match="Unknown dimension"):
            _ = embedder.dimension

    @pytest.mark.asyncio
    async def test_dimension_after_embedding(self):
        """Test dimension is cached after first embedding."""
        embedder = OllamaEmbeddings(model="test-model")

        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 512}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            await embedder.embed_text("test")

            # Now dimension should be known
            assert embedder.dimension == 512


class TestSentenceTransformerEmbeddings:
    """Tests for SentenceTransformerEmbeddings provider."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test SentenceTransformerEmbeddings initialization."""
        embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
        assert embedder.model_name == "all-MiniLM-L6-v2"

    @pytest.mark.asyncio
    async def test_embed_text_success(self):
        """Test successful text embedding."""
        embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")

        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=MagicMock(return_value=[0.1, 0.2, 0.3]))
        mock_model.get_sentence_embedding_dimension.return_value = 3

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embedding = await embedder.embed_text("test text")

            assert embedding == [0.1, 0.2, 0.3]
            assert embedder.dimension == 3

    @pytest.mark.asyncio
    async def test_embed_text_import_error(self):
        """Test handling of missing sentence-transformers package."""
        embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")

        with patch(
            "sentence_transformers.SentenceTransformer",
            side_effect=ImportError("No module named 'sentence_transformers'"),
        ):
            with pytest.raises(EmbeddingError, match="sentence-transformers not installed"):
                await embedder.embed_text("test")

    @pytest.mark.asyncio
    async def test_embed_batch_success(self):
        """Test batch embedding."""
        embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")

        # Mock the model
        mock_model = MagicMock()
        mock_embeddings = [
            MagicMock(tolist=MagicMock(return_value=[0.1, 0.2, 0.3])),
            MagicMock(tolist=MagicMock(return_value=[0.4, 0.5, 0.6])),
        ]
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 3

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            embeddings = await embedder.embed_batch(["text1", "text2"])

            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        """Test batch embedding with empty list."""
        embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
        embeddings = await embedder.embed_batch([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_model_lazy_loading(self):
        """Test that model is loaded lazily."""
        embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")

        # Model should not be loaded yet
        assert embedder._model is None

        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=MagicMock(return_value=[0.1, 0.2, 0.3]))
        mock_model.get_sentence_embedding_dimension.return_value = 3

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            await embedder.embed_text("test")

            # Now model should be loaded
            assert embedder._model is not None


class TestGetEmbeddingProvider:
    """Tests for get_embedding_provider factory function."""

    def test_get_ollama_provider(self):
        """Test getting Ollama provider."""
        provider = get_embedding_provider("ollama")
        assert isinstance(provider, OllamaEmbeddings)
        assert provider.model_name == "nomic-embed-text"

    def test_get_ollama_provider_custom_model(self):
        """Test getting Ollama provider with custom model."""
        provider = get_embedding_provider("ollama", model="custom-model")
        assert isinstance(provider, OllamaEmbeddings)
        assert provider.model_name == "custom-model"

    def test_get_sentence_transformers_provider(self):
        """Test getting sentence-transformers provider."""
        provider = get_embedding_provider("sentence-transformers")
        assert isinstance(provider, SentenceTransformerEmbeddings)
        assert provider.model_name == "all-MiniLM-L6-v2"

    def test_get_sentence_transformers_provider_custom_model(self):
        """Test getting sentence-transformers provider with custom model."""
        provider = get_embedding_provider("sentence-transformers", model="custom-model")
        assert isinstance(provider, SentenceTransformerEmbeddings)
        assert provider.model_name == "custom-model"

    def test_unknown_provider(self):
        """Test error on unknown provider."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_provider("unknown")
