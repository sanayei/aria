"""Tests for file operation tools (copy, move, delete, read)."""

import pytest
from pathlib import Path

from aria.tools.filesystem import (
    CopyFileTool,
    CopyFileParams,
    MoveFileTool,
    MoveFileParams,
    DeleteFileTool,
    DeleteFileParams,
    ReadFileTool,
    ReadFileParams,
)


class TestCopyFileTool:
    """Test CopyFileTool."""

    @pytest.mark.asyncio
    async def test_copy_file_success(self, tmp_path):
        """Test successful file copy."""
        source = tmp_path / "source.txt"
        source.write_text("test content")
        dest = tmp_path / "dest.txt"

        tool = CopyFileTool()
        params = CopyFileParams(source=str(source), destination=str(dest))

        result = await tool.execute(params)

        assert result.success is True
        assert source.exists()  # Original still exists
        assert dest.exists()
        assert dest.read_text() == "test content"

    @pytest.mark.asyncio
    async def test_copy_with_directory_creation(self, tmp_path):
        """Test copying to a directory that doesn't exist yet."""
        source = tmp_path / "source.txt"
        source.write_text("test content")
        dest = tmp_path / "newdir" / "subdir" / "dest.txt"

        tool = CopyFileTool()
        params = CopyFileParams(source=str(source), destination=str(dest))

        result = await tool.execute(params)

        assert result.success is True
        assert dest.exists()
        assert dest.parent.exists()

    @pytest.mark.asyncio
    async def test_copy_without_overwrite(self, tmp_path):
        """Test copy fails when destination exists and overwrite=False."""
        source = tmp_path / "source.txt"
        source.write_text("source content")
        dest = tmp_path / "dest.txt"
        dest.write_text("existing content")

        tool = CopyFileTool()
        params = CopyFileParams(source=str(source), destination=str(dest))

        result = await tool.execute(params)

        assert result.success is False
        assert "already exists" in result.error.lower()

    @pytest.mark.asyncio
    async def test_copy_with_overwrite(self, tmp_path):
        """Test copy succeeds when overwrite=True."""
        source = tmp_path / "source.txt"
        source.write_text("new content")
        dest = tmp_path / "dest.txt"
        dest.write_text("old content")

        tool = CopyFileTool()
        params = CopyFileParams(
            source=str(source),
            destination=str(dest),
            overwrite=True,
        )

        result = await tool.execute(params)

        assert result.success is True
        assert dest.read_text() == "new content"


class TestMoveFileTool:
    """Test MoveFileTool."""

    @pytest.mark.asyncio
    async def test_move_file_success(self, tmp_path):
        """Test successful file move."""
        source = tmp_path / "source.txt"
        source.write_text("test content")
        dest = tmp_path / "dest.txt"

        tool = MoveFileTool()
        params = MoveFileParams(source=str(source), destination=str(dest))

        result = await tool.execute(params)

        assert result.success is True
        assert not source.exists()  # Original moved
        assert dest.exists()
        assert dest.read_text() == "test content"

    @pytest.mark.asyncio
    async def test_move_with_directory_creation(self, tmp_path):
        """Test moving to a directory that doesn't exist yet."""
        source = tmp_path / "source.txt"
        source.write_text("test content")
        dest = tmp_path / "newdir" / "dest.txt"

        tool = MoveFileTool()
        params = MoveFileParams(source=str(source), destination=str(dest))

        result = await tool.execute(params)

        assert result.success is True
        assert dest.exists()

    @pytest.mark.asyncio
    async def test_move_nonexistent_file(self, tmp_path):
        """Test moving a file that doesn't exist."""
        source = tmp_path / "nonexistent.txt"
        dest = tmp_path / "dest.txt"

        tool = MoveFileTool()
        params = MoveFileParams(source=str(source), destination=str(dest))

        result = await tool.execute(params)

        assert result.success is False
        assert "does not exist" in result.error.lower()


class TestDeleteFileTool:
    """Test DeleteFileTool."""

    @pytest.mark.asyncio
    async def test_delete_file_success(self, tmp_path):
        """Test successful file deletion."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        tool = DeleteFileTool()
        params = DeleteFileParams(
            path=str(file_path),
            confirm_name="test.txt",
        )

        result = await tool.execute(params)

        assert result.success is True
        assert not file_path.exists()
        assert result.data["deleted"] == str(file_path)

    @pytest.mark.asyncio
    async def test_delete_with_wrong_confirmation(self, tmp_path):
        """Test deletion fails with incorrect filename confirmation."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        tool = DeleteFileTool()
        params = DeleteFileParams(
            path=str(file_path),
            confirm_name="wrong.txt",
        )

        result = await tool.execute(params)

        assert result.success is False
        assert "confirmation failed" in result.error.lower()
        assert file_path.exists()  # File not deleted

    @pytest.mark.asyncio
    async def test_delete_nonexistent_file(self, tmp_path):
        """Test deleting a file that doesn't exist."""
        file_path = tmp_path / "nonexistent.txt"

        tool = DeleteFileTool()
        params = DeleteFileParams(
            path=str(file_path),
            confirm_name="nonexistent.txt",
        )

        result = await tool.execute(params)

        assert result.success is False
        assert "does not exist" in result.error.lower()


class TestReadFileTool:
    """Test ReadFileTool."""

    @pytest.mark.asyncio
    async def test_read_file_success(self, tmp_path):
        """Test successful file read."""
        file_path = tmp_path / "test.txt"
        content = "Hello, world!\nLine 2\nLine 3"
        file_path.write_text(content)

        tool = ReadFileTool()
        params = ReadFileParams(path=str(file_path))

        result = await tool.execute(params)

        assert result.success is True
        assert result.data["content"] == content
        assert result.data["lines"] == 3

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_path):
        """Test reading a file that doesn't exist."""
        file_path = tmp_path / "nonexistent.txt"

        tool = ReadFileTool()
        params = ReadFileParams(path=str(file_path))

        result = await tool.execute(params)

        assert result.success is False
        assert "does not exist" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_with_encoding(self, tmp_path):
        """Test reading with specific encoding."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Test content", encoding="utf-8")

        tool = ReadFileTool()
        params = ReadFileParams(path=str(file_path), encoding="utf-8")

        result = await tool.execute(params)

        assert result.success is True
        assert result.data["content"] == "Test content"
