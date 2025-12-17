"""Tests for configuration management."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from aria.config import Settings, get_settings, reload_settings


class TestSettings:
    """Test the Settings class."""

    def test_default_settings(self, temp_data_dir, monkeypatch):
        """Test that default settings are created correctly."""
        # Clear any env vars and disable .env loading to test actual Field defaults
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        monkeypatch.delenv("ARIA_DEFAULT_MODEL", raising=False)
        monkeypatch.setattr("aria.config.Settings.model_config", {
            **Settings.model_config,
            "env_file": None,
        })

        settings = Settings(aria_data_dir=temp_data_dir)

        assert settings.ollama_host == "http://localhost:11434"
        assert settings.ollama_model == "qwen3:30b-a3b"
        assert settings.aria_log_level == "INFO"
        assert settings.tool_auto_approve_low_risk is True
        assert settings.tool_require_confirmation is True

    def test_custom_settings(self, temp_data_dir):
        """Test custom settings override defaults."""
        settings = Settings(
            aria_data_dir=temp_data_dir,
            ollama_host="http://localhost:8080",
            ollama_model="llama2",
            aria_log_level="DEBUG",
        )

        assert settings.ollama_host == "http://localhost:8080"
        assert settings.ollama_model == "llama2"
        assert settings.aria_log_level == "DEBUG"

    def test_path_expansion(self, temp_data_dir):
        """Test that paths are properly expanded."""
        settings = Settings(aria_data_dir=temp_data_dir)

        # Paths should be absolute
        assert settings.aria_data_dir.is_absolute()
        assert settings.db_path.is_absolute()
        assert settings.chroma_path.is_absolute()

    def test_default_db_path(self, temp_data_dir):
        """Test that default DB path is generated correctly."""
        settings = Settings(aria_data_dir=temp_data_dir)

        expected_db_path = temp_data_dir / "cache" / "aria.db"
        assert settings.db_path == expected_db_path

    def test_default_chroma_path(self, temp_data_dir):
        """Test that default ChromaDB path is generated correctly."""
        settings = Settings(aria_data_dir=temp_data_dir)

        expected_chroma_path = temp_data_dir / "chroma"
        assert settings.chroma_path == expected_chroma_path

    def test_custom_paths(self, temp_data_dir):
        """Test that custom paths override defaults."""
        custom_db = temp_data_dir / "custom.db"
        custom_chroma = temp_data_dir / "custom_chroma"

        settings = Settings(
            aria_data_dir=temp_data_dir,
            db_path=custom_db,
            chroma_path=custom_chroma,
        )

        assert settings.db_path == custom_db.resolve()
        assert settings.chroma_path == custom_chroma.resolve()

    def test_ensure_directories(self, temp_data_dir):
        """Test that ensure_directories creates all required directories."""
        settings = Settings(aria_data_dir=temp_data_dir)
        settings.ensure_directories()

        # Check that all directories exist
        assert settings.aria_data_dir.exists()
        assert (settings.aria_data_dir / "cache").exists()
        assert (settings.aria_data_dir / "logs").exists()
        assert settings.chroma_path.exists()

    def test_log_file_path(self, temp_data_dir):
        """Test log file path property."""
        settings = Settings(aria_data_dir=temp_data_dir)

        expected_log = temp_data_dir / "logs" / "aria.log"
        assert settings.log_file_path == expected_log

    def test_ollama_base_url(self, temp_data_dir):
        """Test Ollama base URL property strips trailing slashes."""
        settings = Settings(
            aria_data_dir=temp_data_dir,
            ollama_host="http://localhost:11434/",
        )

        assert settings.ollama_base_url == "http://localhost:11434"

    def test_validation_errors(self, temp_data_dir):
        """Test that invalid settings raise validation errors."""
        # Invalid log level
        with pytest.raises(ValidationError):
            Settings(aria_data_dir=temp_data_dir, aria_log_level="INVALID")

        # Invalid timeout (too low)
        with pytest.raises(ValidationError):
            Settings(aria_data_dir=temp_data_dir, ollama_timeout=5)

        # Invalid temperature (too high)
        with pytest.raises(ValidationError):
            Settings(aria_data_dir=temp_data_dir, ollama_temperature=3.0)

    def test_model_dump_safe(self, temp_data_dir):
        """Test safe model dump for logging."""
        settings = Settings(aria_data_dir=temp_data_dir)
        safe_dump = settings.model_dump_safe()

        assert "ollama_host" in safe_dump
        assert "ollama_model" in safe_dump
        assert "log_level" in safe_dump
        assert "data_dir" in safe_dump


class TestGlobalSettings:
    """Test global settings functions."""

    def test_get_settings_singleton(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_reload_settings(self):
        """Test that reload_settings creates a new instance."""
        settings1 = get_settings()
        settings2 = reload_settings()

        # Should be different instances (reloaded)
        assert settings1 is not settings2
