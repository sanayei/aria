"""Pytest configuration and fixtures for ARIA tests."""

import pytest
from pathlib import Path
import tempfile
import shutil

from aria.config import Settings


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_settings(temp_data_dir):
    """Create test settings with isolated data directory."""
    return Settings(
        aria_data_dir=temp_data_dir,
        ollama_host="http://localhost:11434",
        ollama_model="qwen3:30b-a3b",
        aria_log_level="DEBUG",
    )
