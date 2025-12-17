"""Configuration management for ARIA using Pydantic settings.

This module handles all configuration for the ARIA assistant, loading from
environment variables and .env files with sensible defaults.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main configuration settings for ARIA.

    Settings are loaded from environment variables and .env files.
    Environment variables take precedence over .env file values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama Configuration
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL (set to your Windows host IP for WSL)",
    )
    ollama_model: str = Field(
        default="qwen3:30b-a3b",
        description="Primary LLM model for reasoning and tool calling",
    )
    ollama_timeout: int = Field(
        default=120,
        description="Timeout for Ollama API calls in seconds",
        ge=10,
        le=600,
    )
    ollama_temperature: float = Field(
        default=0.7,
        description="Temperature for LLM generation",
        ge=0.0,
        le=2.0,
    )

    # Application Settings
    aria_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level for the application",
    )
    aria_log_file: Path | None = Field(
        default=None,
        description="Optional file path to write logs (defaults to console only)",
    )
    aria_debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with detailed LLM interactions and timing",
    )
    aria_data_dir: Path = Field(
        default=Path("./data"),
        description="Base directory for all local data storage",
    )
    aria_max_history: int = Field(
        default=50,
        description="Maximum number of conversation turns to keep in context",
        ge=1,
        le=1000,
    )

    # Database Configuration
    db_path: Path | None = Field(
        default=None,
        description="Path to SQLite database (auto-generated in data_dir if not set)",
    )

    # Conversation Memory Settings
    conversation_db_path: Path | None = Field(
        default=None,
        description="Path to conversation history database (auto-generated in data_dir if not set)",
    )
    max_context_messages: int = Field(
        default=50,
        description="Maximum number of messages to include in conversation context",
        ge=1,
        le=500,
    )
    auto_save_conversations: bool = Field(
        default=True,
        description="Automatically save conversations to database",
    )

    # Vector Store Configuration
    chroma_path: Path | None = Field(
        default=None,
        description="Path to ChromaDB storage (auto-generated in data_dir if not set)",
    )
    chroma_collection: str = Field(
        default="aria_knowledge",
        description="ChromaDB collection name for knowledge base",
    )

    # Email Configuration
    gmail_credentials_dir: Path = Field(
        default=Path.home() / ".aria" / "credentials",
        description="Directory for Gmail OAuth credentials and tokens",
    )
    gmail_enabled: bool = Field(
        default=False,
        description="Enable Gmail integration (set to True after running setup_gmail.py)",
    )
    gmail_cache_ttl: int = Field(
        default=300,
        description="Gmail metadata cache TTL in seconds",
        ge=0,
    )

    # Tool Execution Settings
    tool_auto_approve_low_risk: bool = Field(
        default=True,
        description="Automatically approve low-risk tool executions",
    )
    tool_require_confirmation: bool = Field(
        default=True,
        description="Require user confirmation for medium+ risk tools",
    )
    tool_timeout: int = Field(
        default=60,
        description="Default timeout for tool execution in seconds",
        ge=5,
        le=600,
    )

    # Agent Settings
    agent_max_iterations: int = Field(
        default=20,
        description="Maximum number of agent loop iterations",
        ge=1,
        le=100,
    )
    agent_verbose: bool = Field(
        default=False,
        description="Enable verbose agent logging",
    )

    # Document Processing Settings
    documents_source_dir: Path = Field(
        default=Path("/mnt/readyshare"),
        description="Source directory for scanned documents (network share)",
    )
    documents_output_dir: Path = Field(
        default=Path.home() / "documents",
        description="Output directory for organized documents",
    )
    family_members: list[str] = Field(
        default=["Amir", "Munira", "Maral", "Gazelle"],
        description="Family members for document classification",
    )
    document_categories: list[str] = Field(
        default=[
            "medical",
            "financial",
            "tax",
            "insurance",
            "correspondence",
            "legal",
            "education",
            "utilities",
            "other",
        ],
        description="Document categories for classification",
    )

    @field_validator("aria_data_dir", "aria_log_file", "db_path", "conversation_db_path", "chroma_path", "gmail_credentials_dir", "documents_source_dir", "documents_output_dir", mode="before")
    @classmethod
    def expand_paths(cls, v: str | Path | None) -> Path | None:
        """Expand relative paths to absolute paths."""
        if v is None:
            return None
        path = Path(v)
        return path.expanduser().resolve()

    @field_validator("db_path", mode="after")
    @classmethod
    def set_default_db_path(cls, v: Path | None, info) -> Path:
        """Set default database path if not specified."""
        if v is None:
            data_dir = info.data.get("aria_data_dir", Path("./data"))
            return data_dir / "cache" / "aria.db"
        return v

    @field_validator("chroma_path", mode="after")
    @classmethod
    def set_default_chroma_path(cls, v: Path | None, info) -> Path:
        """Set default ChromaDB path if not specified."""
        if v is None:
            data_dir = info.data.get("aria_data_dir", Path("./data"))
            return data_dir / "chroma"
        return v

    @field_validator("conversation_db_path", mode="after")
    @classmethod
    def set_default_conversation_db_path(cls, v: Path | None, info) -> Path:
        """Set default conversation database path if not specified."""
        if v is None:
            data_dir = info.data.get("aria_data_dir", Path("./data"))
            return data_dir / "cache" / "conversations.db"
        return v

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.aria_data_dir,
            self.aria_data_dir / "cache",
            self.aria_data_dir / "logs",
            self.aria_data_dir / "chroma",
            self.db_path.parent,
            self.conversation_db_path.parent,
            self.chroma_path,
            self.gmail_credentials_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def log_file_path(self) -> Path:
        """Get the path to the main log file."""
        return self.aria_data_dir / "logs" / "aria.log"

    @property
    def ollama_base_url(self) -> str:
        """Get the base URL for Ollama API (without /api suffix)."""
        return self.ollama_host.rstrip("/")

    def model_dump_safe(self) -> dict[str, str]:
        """Dump settings as a dictionary with safe string representations.

        Useful for logging configuration without exposing sensitive data.
        """
        return {
            "ollama_host": self.ollama_host,
            "ollama_model": self.ollama_model,
            "log_level": self.aria_log_level,
            "data_dir": str(self.aria_data_dir),
            "db_path": str(self.db_path),
            "conversation_db_path": str(self.conversation_db_path),
            "chroma_path": str(self.chroma_path),
            "auto_approve_low_risk": self.tool_auto_approve_low_risk,
            "auto_save_conversations": self.auto_save_conversations,
            "max_context_messages": self.max_context_messages,
            "documents_source_dir": str(self.documents_source_dir),
            "documents_output_dir": str(self.documents_output_dir),
        }


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance.

    Creates and caches the settings on first call.
    Ensures all required directories exist.

    Returns:
        Settings: The global settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment/files.

    Useful for testing or when configuration changes at runtime.

    Returns:
        Settings: The newly loaded settings instance
    """
    global _settings
    _settings = Settings()
    _settings.ensure_directories()
    return _settings
