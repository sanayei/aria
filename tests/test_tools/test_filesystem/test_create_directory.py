"""Tests for create directory tool."""

import pytest
from pathlib import Path

from aria.tools.filesystem import CreateDirectoryTool, CreateDirectoryParams


class TestCreateDirectoryTool:
    """Test CreateDirectoryTool."""

    @pytest.mark.asyncio
    async def test_create_directory_success(self, tmp_path):
        """Test successful directory creation."""
        dir_path = tmp_path / "newdir"

        tool = CreateDirectoryTool()
        params = CreateDirectoryParams(path=str(dir_path))

        result = await tool.execute(params)

        assert result.success is True
        assert dir_path.exists()
        assert dir_path.is_dir()
        assert result.data["created"] is True

    @pytest.mark.asyncio
    async def test_create_nested_directories(self, tmp_path):
        """Test creating nested directories (parents=True)."""
        dir_path = tmp_path / "level1" / "level2" / "level3"

        tool = CreateDirectoryTool()
        params = CreateDirectoryParams(path=str(dir_path))

        result = await tool.execute(params)

        assert result.success is True
        assert dir_path.exists()
        assert (tmp_path / "level1").exists()
        assert (tmp_path / "level1" / "level2").exists()

    @pytest.mark.asyncio
    async def test_create_existing_directory(self, tmp_path):
        """Test creating a directory that already exists."""
        dir_path = tmp_path / "existing"
        dir_path.mkdir()

        tool = CreateDirectoryTool()
        params = CreateDirectoryParams(path=str(dir_path))

        result = await tool.execute(params)

        assert result.success is True
        assert result.data["created"] is False
        assert "already exists" in result.data["message"]

    @pytest.mark.asyncio
    async def test_create_when_file_exists(self, tmp_path):
        """Test creating directory when a file with same name exists."""
        file_path = tmp_path / "conflict"
        file_path.write_text("content")

        tool = CreateDirectoryTool()
        params = CreateDirectoryParams(path=str(file_path))

        result = await tool.execute(params)

        assert result.success is False
        assert "not a directory" in result.error.lower()
