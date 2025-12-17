"""Tests for list directory tool."""

import pytest
from pathlib import Path

from aria.tools.filesystem import ListDirectoryTool, ListDirectoryParams


class TestListDirectoryTool:
    """Test ListDirectoryTool."""

    @pytest.mark.asyncio
    async def test_list_directory_basic(self, tmp_path):
        """Test basic directory listing."""
        # Create test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.pdf").write_text("content2")
        (tmp_path / "subdir").mkdir()

        tool = ListDirectoryTool()
        params = ListDirectoryParams(path=str(tmp_path))

        result = await tool.execute(params)

        assert result.success is True
        assert result.data["count"] == 3
        assert len(result.data["files"]) == 3

        # Check file names are present
        names = [f["name"] for f in result.data["files"]]
        assert "file1.txt" in names
        assert "file2.pdf" in names
        assert "subdir" in names

    @pytest.mark.asyncio
    async def test_list_directory_with_pattern(self, tmp_path):
        """Test directory listing with glob pattern."""
        # Create test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.pdf").write_text("content2")
        (tmp_path / "file3.txt").write_text("content3")

        tool = ListDirectoryTool()
        params = ListDirectoryParams(path=str(tmp_path), pattern="*.txt")

        result = await tool.execute(params)

        assert result.success is True
        assert result.data["count"] == 2

        names = [f["name"] for f in result.data["files"]]
        assert "file1.txt" in names
        assert "file3.txt" in names
        assert "file2.pdf" not in names

    @pytest.mark.asyncio
    async def test_list_directory_recursive(self, tmp_path):
        """Test recursive directory listing."""
        # Create nested structure
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file2.txt").write_text("content2")
        (tmp_path / "subdir" / "deep").mkdir()
        (tmp_path / "subdir" / "deep" / "file3.txt").write_text("content3")

        tool = ListDirectoryTool()
        params = ListDirectoryParams(
            path=str(tmp_path),
            pattern="*.txt",
            recursive=True,
        )

        result = await tool.execute(params)

        assert result.success is True
        assert result.data["count"] == 3

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self):
        """Test listing a directory that doesn't exist."""
        tool = ListDirectoryTool()
        params = ListDirectoryParams(path="/nonexistent/path")

        result = await tool.execute(params)

        assert result.success is False
        assert "does not exist" in result.error.lower()

    @pytest.mark.asyncio
    async def test_list_file_instead_of_directory(self, tmp_path):
        """Test listing a file instead of a directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        tool = ListDirectoryTool()
        params = ListDirectoryParams(path=str(file_path))

        result = await tool.execute(params)

        assert result.success is False
        assert "not a directory" in result.error.lower()
