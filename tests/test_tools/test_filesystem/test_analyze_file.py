"""Tests for file analysis tool."""

import pytest
from pathlib import Path
from datetime import datetime

from aria.tools.filesystem import AnalyzeFileTool, AnalyzeFileParams


class TestAnalyzeFileTool:
    """Test AnalyzeFileTool."""

    @pytest.mark.asyncio
    async def test_analyze_text_file(self, tmp_path):
        """Test analyzing a text file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, World!")

        tool = AnalyzeFileTool()
        params = AnalyzeFileParams(path=str(file_path))

        result = await tool.execute(params)

        assert result.success is True
        data = result.data

        assert data["name"] == "test.txt"
        assert data["extension"] == "txt"
        assert data["size_bytes"] == 13
        assert data["size_human"] == "13.0 B"
        assert data["mime_type"] == "text/plain"
        assert data["category"] == "document"
        assert data["is_hidden"] is False
        assert data["is_text"] is True
        assert data["path"] == str(file_path)

        # Check timestamps are valid ISO format
        datetime.fromisoformat(data["created_at"])
        datetime.fromisoformat(data["modified_at"])

    @pytest.mark.asyncio
    async def test_analyze_python_file(self, tmp_path):
        """Test analyzing a Python source file."""
        file_path = tmp_path / "script.py"
        file_path.write_text("print('Hello')")

        tool = AnalyzeFileTool()
        params = AnalyzeFileParams(path=str(file_path))

        result = await tool.execute(params)

        assert result.success is True
        data = result.data

        assert data["extension"] == "py"
        assert data["category"] == "code"
        assert data["is_text"] is True

    @pytest.mark.asyncio
    async def test_analyze_image_file(self, tmp_path):
        """Test analyzing an image file."""
        file_path = tmp_path / "photo.jpg"
        # Create a minimal JPEG file (just header bytes for testing)
        file_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

        tool = AnalyzeFileTool()
        params = AnalyzeFileParams(path=str(file_path))

        result = await tool.execute(params)

        assert result.success is True
        data = result.data

        assert data["extension"] == "jpg"
        assert data["mime_type"] == "image/jpeg"
        assert data["category"] == "image"
        assert data["is_text"] is False

    @pytest.mark.asyncio
    async def test_analyze_pdf_file(self, tmp_path):
        """Test analyzing a PDF file."""
        file_path = tmp_path / "document.pdf"
        file_path.write_bytes(b"%PDF-1.4")

        tool = AnalyzeFileTool()
        params = AnalyzeFileParams(path=str(file_path))

        result = await tool.execute(params)

        assert result.success is True
        data = result.data

        assert data["extension"] == "pdf"
        assert data["mime_type"] == "application/pdf"
        assert data["category"] == "document"
        assert data["is_text"] is False

    @pytest.mark.asyncio
    async def test_analyze_archive_file(self, tmp_path):
        """Test analyzing an archive file."""
        file_path = tmp_path / "archive.zip"
        # ZIP file magic number
        file_path.write_bytes(b"PK\x03\x04")

        tool = AnalyzeFileTool()
        params = AnalyzeFileParams(path=str(file_path))

        result = await tool.execute(params)

        assert result.success is True
        data = result.data

        assert data["extension"] == "zip"
        assert data["category"] == "archive"

    @pytest.mark.asyncio
    async def test_analyze_hidden_file(self, tmp_path):
        """Test analyzing a hidden file (Unix-style)."""
        file_path = tmp_path / ".hidden"
        file_path.write_text("secret")

        tool = AnalyzeFileTool()
        params = AnalyzeFileParams(path=str(file_path))

        result = await tool.execute(params)

        assert result.success is True
        assert result.data["is_hidden"] is True

    @pytest.mark.asyncio
    async def test_analyze_file_without_extension(self, tmp_path):
        """Test analyzing a file with no extension."""
        file_path = tmp_path / "README"
        file_path.write_text("Documentation")

        tool = AnalyzeFileTool()
        params = AnalyzeFileParams(path=str(file_path))

        result = await tool.execute(params)

        assert result.success is True
        assert result.data["extension"] is None
        assert result.data["category"] == "other"

    @pytest.mark.asyncio
    async def test_analyze_large_file(self, tmp_path):
        """Test human-readable size formatting for large file."""
        file_path = tmp_path / "large.bin"
        # Create a file larger than 1MB
        file_path.write_bytes(b"0" * (2 * 1024 * 1024))  # 2 MB

        tool = AnalyzeFileTool()
        params = AnalyzeFileParams(path=str(file_path))

        result = await tool.execute(params)

        assert result.success is True
        assert result.data["size_bytes"] == 2 * 1024 * 1024
        assert "MB" in result.data["size_human"]

    @pytest.mark.asyncio
    async def test_analyze_nonexistent_file(self, tmp_path):
        """Test analyzing a file that doesn't exist."""
        file_path = tmp_path / "nonexistent.txt"

        tool = AnalyzeFileTool()
        params = AnalyzeFileParams(path=str(file_path))

        result = await tool.execute(params)

        assert result.success is False
        assert "does not exist" in result.error.lower()

    @pytest.mark.asyncio
    async def test_analyze_directory_instead_of_file(self, tmp_path):
        """Test analyzing a directory (should fail)."""
        dir_path = tmp_path / "testdir"
        dir_path.mkdir()

        tool = AnalyzeFileTool()
        params = AnalyzeFileParams(path=str(dir_path))

        result = await tool.execute(params)

        assert result.success is False
        assert "not a file" in result.error.lower()

    @pytest.mark.asyncio
    async def test_analyze_various_categories(self, tmp_path):
        """Test category detection for various file types."""
        test_cases = [
            ("spreadsheet.xlsx", "spreadsheet"),
            ("presentation.pptx", "presentation"),
            ("video.mp4", "video"),
            ("audio.mp3", "audio"),
            ("script.js", "code"),
            ("data.json", "code"),
            ("config.yaml", "code"),
            ("program.exe", "executable"),
            ("unknown.xyz", "other"),
        ]

        for filename, expected_category in test_cases:
            file_path = tmp_path / filename
            file_path.write_bytes(b"test")

            tool = AnalyzeFileTool()
            params = AnalyzeFileParams(path=str(file_path))

            result = await tool.execute(params)

            assert result.success is True, f"Failed for {filename}"
            assert result.data["category"] == expected_category, (
                f"Wrong category for {filename}: expected {expected_category}, got {result.data['category']}"
            )

    @pytest.mark.asyncio
    async def test_analyze_with_expanduser(self, tmp_path, monkeypatch):
        """Test that ~ is expanded correctly."""
        # Create a file in tmp_path
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        # Mock home directory to tmp_path
        monkeypatch.setenv("HOME", str(tmp_path))

        tool = AnalyzeFileTool()
        # Use relative path from home
        params = AnalyzeFileParams(path="~/test.txt")

        result = await tool.execute(params)

        assert result.success is True
        assert result.data["name"] == "test.txt"

    @pytest.mark.asyncio
    async def test_confirmation_message(self, tmp_path):
        """Test the confirmation message format."""
        file_path = tmp_path / "test.txt"

        tool = AnalyzeFileTool()
        params = AnalyzeFileParams(path=str(file_path))

        message = tool.get_confirmation_message(params)

        assert "Analyze file" in message
        assert str(file_path) in message
