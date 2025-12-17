"""Tests for analyze_directory tool."""

import tempfile
from pathlib import Path

import pytest

from aria.tools.filesystem.analyze_directory import (
    AnalyzeDirectoryTool,
    AnalyzeDirectoryParams,
)
from aria.tools import ToolResult


@pytest.fixture
def sample_directory(tmp_path: Path):
    """Create a sample directory structure for testing.

    Structure:
    tmp_path/
        ├── file1.txt (100 bytes, "text")
        ├── file2.txt (200 bytes, "text")
        ├── image.png (300 bytes, "image")
        ├── script.py (140 bytes, "code")
        ├── .hidden (50 bytes, hidden)
        ├── large_file.dat (5000 bytes)
        └── subdir/
            ├── nested.txt (100 bytes)
            └── nested.jpg (400 bytes)
    """
    # Create files
    (tmp_path / "file1.txt").write_text("x" * 100)
    (tmp_path / "file2.txt").write_text("y" * 200)
    (tmp_path / "image.png").write_bytes(b"PNG" * 100)
    (tmp_path / "script.py").write_text("print('hello')" * 10)
    (tmp_path / ".hidden").write_text("secret" * 10)
    (tmp_path / "large_file.dat").write_bytes(b"data" * 1250)

    # Create subdirectory with files
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("n" * 100)
    (subdir / "nested.jpg").write_bytes(b"JPG" * 133 + b"J")

    return tmp_path


class TestAnalyzeDirectoryTool:
    """Test AnalyzeDirectoryTool class."""

    @pytest.mark.asyncio
    async def test_analyze_simple_directory(self, sample_directory):
        """Test analyzing a simple directory (non-recursive)."""
        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(
            path=str(sample_directory),
            recursive=False,
            include_hidden=False,
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data is not None

        # Check basic stats (excludes hidden and subdirectory files)
        assert result.data["total_files"] == 5  # file1, file2, image, script, large_file
        assert result.data["total_size_bytes"] == 5740  # 100+200+300+140+5000
        assert result.data["path"] == str(sample_directory)
        assert not result.data["reached_max_files"]

        # Check categories exist
        assert "categories" in result.data
        categories = result.data["categories"]

        # Should have document, code, image, and other categories
        assert "document" in categories
        assert categories["document"]["count"] == 2  # file1.txt, file2.txt

        assert "code" in categories
        assert categories["code"]["count"] == 1  # script.py

        assert "image" in categories
        assert categories["image"]["count"] == 1  # image.png

        assert "other" in categories
        assert categories["other"]["count"] == 1  # large_file.dat

    @pytest.mark.asyncio
    async def test_analyze_recursive(self, sample_directory):
        """Test recursive directory analysis."""
        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(
            path=str(sample_directory),
            recursive=True,
            include_hidden=False,
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data is not None

        # Should include files from subdirectory (7 total, excluding hidden)
        assert result.data["total_files"] == 7

        # Check that nested files are included
        categories = result.data["categories"]
        assert categories["document"]["count"] == 3  # file1, file2, nested.txt
        assert categories["image"]["count"] == 2  # image.png, nested.jpg

    @pytest.mark.asyncio
    async def test_analyze_include_hidden(self, sample_directory):
        """Test including hidden files."""
        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(
            path=str(sample_directory),
            recursive=False,
            include_hidden=True,
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data is not None

        # Should include .hidden file
        assert result.data["total_files"] == 6

        # Find .hidden in largest/oldest/newest files
        all_files_data = (
            result.data["largest_files"] +
            result.data["oldest_files"] +
            result.data["newest_files"]
        )
        file_names = [f["name"] for f in all_files_data]
        assert ".hidden" in file_names

    @pytest.mark.asyncio
    async def test_analyze_max_files_limit(self, sample_directory):
        """Test max_files limit."""
        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(
            path=str(sample_directory),
            recursive=True,
            include_hidden=True,
            max_files=3,  # Limit to 3 files
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data is not None

        # Should stop at 3 files
        assert result.data["total_files"] == 3
        assert result.data["reached_max_files"] is True

    @pytest.mark.asyncio
    async def test_analyze_by_extension(self, sample_directory):
        """Test extension breakdown."""
        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(
            path=str(sample_directory),
            recursive=True,
            include_hidden=False,
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data is not None

        # Check by_extension data
        by_ext = result.data["by_extension"]
        assert ".txt" in by_ext
        assert by_ext[".txt"]["count"] == 3  # file1, file2, nested.txt

        assert ".png" in by_ext
        assert by_ext[".png"]["count"] == 1

        assert ".py" in by_ext
        assert by_ext[".py"]["count"] == 1

        assert ".jpg" in by_ext
        assert by_ext[".jpg"]["count"] == 1

    @pytest.mark.asyncio
    async def test_analyze_largest_files(self, sample_directory):
        """Test largest files list."""
        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(
            path=str(sample_directory),
            recursive=True,
            include_hidden=False,
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data is not None

        largest = result.data["largest_files"]
        assert len(largest) > 0

        # Largest file should be large_file.dat (5000 bytes)
        assert largest[0]["name"] == "large_file.dat"
        assert largest[0]["size_bytes"] == 5000

        # Check that files are sorted by size (descending)
        sizes = [f["size_bytes"] for f in largest]
        assert sizes == sorted(sizes, reverse=True)

    @pytest.mark.asyncio
    async def test_analyze_potential_duplicates(self, tmp_path: Path):
        """Test duplicate detection."""
        # Create files with same name and size
        (tmp_path / "duplicate.txt").write_text("same content")

        subdir1 = tmp_path / "dir1"
        subdir1.mkdir()
        (subdir1 / "duplicate.txt").write_text("same content")

        subdir2 = tmp_path / "dir2"
        subdir2.mkdir()
        (subdir2 / "duplicate.txt").write_text("same content")

        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(
            path=str(tmp_path),
            recursive=True,
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data is not None

        # Should detect duplicates
        duplicates = result.data["potential_duplicates"]
        assert len(duplicates) > 0

        # Find the duplicate.txt entry
        dup_entry = next(
            (d for d in duplicates if d["name"] == "duplicate.txt"),
            None
        )
        assert dup_entry is not None
        assert dup_entry["count"] == 3
        assert len(dup_entry["locations"]) == 3

    @pytest.mark.asyncio
    async def test_analyze_empty_directory(self, tmp_path: Path):
        """Test analyzing an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(path=str(empty_dir))

        result = await tool.execute(params)

        assert result.success
        assert result.data is not None
        assert result.data["total_files"] == 0
        assert result.data["total_size_bytes"] == 0
        assert result.data["categories"] == {}

    @pytest.mark.asyncio
    async def test_analyze_nonexistent_directory(self):
        """Test analyzing a directory that doesn't exist."""
        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(path="/nonexistent/path/directory")

        result = await tool.execute(params)

        assert not result.success
        assert "does not exist" in result.error.lower()

    @pytest.mark.asyncio
    async def test_analyze_file_not_directory(self, tmp_path: Path):
        """Test analyzing a file instead of directory."""
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(path=str(test_file))

        result = await tool.execute(params)

        assert not result.success
        assert "not a directory" in result.error.lower()

    @pytest.mark.asyncio
    async def test_analyze_human_readable_sizes(self, sample_directory):
        """Test that sizes are formatted in human-readable format."""
        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(
            path=str(sample_directory),
            recursive=False,
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data is not None

        # Check that human-readable size is present
        assert "total_size_human" in result.data
        assert isinstance(result.data["total_size_human"], str)

        # Check largest files have human-readable sizes
        largest = result.data["largest_files"]
        if largest:
            assert "size_human" in largest[0]
            assert isinstance(largest[0]["size_human"], str)

    @pytest.mark.asyncio
    async def test_confirmation_message(self):
        """Test confirmation message formatting."""
        tool = AnalyzeDirectoryTool()

        # Non-recursive
        params = AnalyzeDirectoryParams(path="/test/path", recursive=False)
        msg = tool.get_confirmation_message(params)
        assert "non-recursively" in msg
        assert "/test/path" in msg

        # Recursive
        params = AnalyzeDirectoryParams(path="/test/path", recursive=True)
        msg = tool.get_confirmation_message(params)
        assert "recursively" in msg
        assert "/test/path" in msg

    @pytest.mark.asyncio
    async def test_analyze_oldest_newest_files(self, tmp_path: Path):
        """Test oldest and newest files detection."""
        import time

        # Create files with different modification times
        old_file = tmp_path / "old.txt"
        old_file.write_text("old")
        old_time = time.time() - 86400  # 1 day ago
        Path(old_file).touch(old_time)

        time.sleep(0.1)  # Ensure different timestamps

        medium_file = tmp_path / "medium.txt"
        medium_file.write_text("medium")
        medium_time = time.time() - 3600  # 1 hour ago
        Path(medium_file).touch(medium_time)

        time.sleep(0.1)

        new_file = tmp_path / "new.txt"
        new_file.write_text("new")
        # Current time

        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(path=str(tmp_path))

        result = await tool.execute(params)

        assert result.success
        assert result.data is not None

        oldest = result.data["oldest_files"]
        newest = result.data["newest_files"]

        # Oldest should have old.txt first
        assert len(oldest) >= 1
        assert oldest[0]["name"] == "old.txt"

        # Newest should have new.txt first
        assert len(newest) >= 1
        assert newest[0]["name"] == "new.txt"
