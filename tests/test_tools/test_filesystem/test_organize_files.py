"""Tests for organize_files tool."""

from pathlib import Path

import pytest

from aria.tools.filesystem.organize_files import (
    OrganizeFilesTool,
    OrganizeFilesParams,
    FileOperation,
)
from aria.tools import ToolResult


@pytest.fixture
def messy_directory(tmp_path: Path):
    """Create a messy directory with various file types.

    Structure:
    tmp_path/
        ├── report.pdf
        ├── photo.jpg
        ├── script.py
        ├── data.xlsx
        ├── video.mp4
        ├── archive.zip
        ├── random.txt
        └── .hidden_config
    """
    # Create various file types
    (tmp_path / "report.pdf").write_bytes(b"PDF content")
    (tmp_path / "photo.jpg").write_bytes(b"JPG content")
    (tmp_path / "script.py").write_text("print('hello')")
    (tmp_path / "data.xlsx").write_bytes(b"Excel content")
    (tmp_path / "video.mp4").write_bytes(b"Video content")
    (tmp_path / "archive.zip").write_bytes(b"ZIP content")
    (tmp_path / "random.txt").write_text("Random text")
    (tmp_path / ".hidden_config").write_text("config")

    return tmp_path


@pytest.fixture
def dated_directory(tmp_path: Path):
    """Create a directory with files having different modification times."""
    import time

    files = [
        ("old_file.txt", time.time() - 86400 * 60),  # 60 days ago
        ("recent_file.pdf", time.time() - 86400 * 5),  # 5 days ago
        ("new_file.jpg", time.time()),  # Now
    ]

    for filename, mtime in files:
        file_path = tmp_path / filename
        file_path.write_text(f"Content of {filename}")
        Path(file_path).touch(mtime)

    return tmp_path


class TestOrganizeFilesTool:
    """Test OrganizeFilesTool class."""

    @pytest.mark.asyncio
    async def test_organize_by_category_dry_run(self, messy_directory):
        """Test organizing by category in dry run mode."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(messy_directory),
            organization_scheme="category",
            dry_run=True,
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data is not None
        assert result.data["dry_run"] is True

        # Check operations were planned
        operations = result.data["operations"]
        assert len(operations) > 0

        # Verify categorization
        operation_map = {op["source"]: op for op in operations}

        # PDF should go to Documents
        pdf_ops = [op for op in operations if "report.pdf" in op["source"]]
        assert len(pdf_ops) == 1
        assert "Documents" in pdf_ops[0]["destination"]

        # JPG should go to Images
        jpg_ops = [op for op in operations if "photo.jpg" in op["source"]]
        assert len(jpg_ops) == 1
        assert "Images" in jpg_ops[0]["destination"]

        # Python should go to Code
        py_ops = [op for op in operations if "script.py" in op["source"]]
        assert len(py_ops) == 1
        assert "Code" in py_ops[0]["destination"]

        # XLSX should go to Spreadsheets
        xlsx_ops = [op for op in operations if "data.xlsx" in op["source"]]
        assert len(xlsx_ops) == 1
        assert "Spreadsheets" in xlsx_ops[0]["destination"]

        # Check summary
        summary = result.data["summary"]
        assert summary["total_files"] > 0
        assert "by_destination" in summary
        assert "Documents/" in summary["by_destination"]
        assert "Images/" in summary["by_destination"]
        assert "Code/" in summary["by_destination"]

    @pytest.mark.asyncio
    async def test_organize_by_category_skip_hidden(self, messy_directory):
        """Test that hidden files are skipped by default."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(messy_directory),
            organization_scheme="category",
            dry_run=True,
            skip_hidden=True,
        )

        result = await tool.execute(params)

        assert result.success

        # Hidden file should not be in operations
        operations = result.data["operations"]
        hidden_ops = [op for op in operations if ".hidden_config" in op["source"]]
        assert len(hidden_ops) == 0

    @pytest.mark.asyncio
    async def test_organize_by_category_include_hidden(self, messy_directory):
        """Test organizing including hidden files."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(messy_directory),
            organization_scheme="category",
            dry_run=True,
            skip_hidden=False,
        )

        result = await tool.execute(params)

        assert result.success

        # Hidden file should be in operations
        operations = result.data["operations"]
        hidden_ops = [op for op in operations if ".hidden_config" in op["source"]]
        assert len(hidden_ops) == 1

    @pytest.mark.asyncio
    async def test_organize_by_extension(self, messy_directory):
        """Test organizing by file extension."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(messy_directory),
            organization_scheme="extension",
            dry_run=True,
        )

        result = await tool.execute(params)

        assert result.success

        operations = result.data["operations"]

        # Check that files are organized by extension
        pdf_ops = [op for op in operations if "report.pdf" in op["source"]]
        assert len(pdf_ops) == 1
        assert "/pdf/" in pdf_ops[0]["destination"]

        jpg_ops = [op for op in operations if "photo.jpg" in op["source"]]
        assert len(jpg_ops) == 1
        assert "/jpg/" in jpg_ops[0]["destination"]

        py_ops = [op for op in operations if "script.py" in op["source"]]
        assert len(py_ops) == 1
        assert "/py/" in py_ops[0]["destination"]

    @pytest.mark.asyncio
    async def test_organize_by_date(self, dated_directory):
        """Test organizing by modification date."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(dated_directory),
            organization_scheme="date",
            dry_run=True,
        )

        result = await tool.execute(params)

        assert result.success

        operations = result.data["operations"]

        # All operations should have year/month in destination
        for op in operations:
            if op["action"] == "move":
                # Should have YYYY/MM pattern
                assert "/" in op["destination"]
                parts = Path(op["destination"]).parts
                # Find year/month parts (YYYY and MM)
                has_date_structure = any(
                    len(part) == 4 and part.isdigit() for part in parts
                )
                assert has_date_structure

    @pytest.mark.asyncio
    async def test_organize_by_date_category(self, messy_directory):
        """Test organizing by date and category combined."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(messy_directory),
            organization_scheme="date_category",
            dry_run=True,
        )

        result = await tool.execute(params)

        assert result.success

        operations = result.data["operations"]

        # Check that destinations have both date and category
        pdf_ops = [op for op in operations if "report.pdf" in op["source"]]
        assert len(pdf_ops) == 1
        # Should have year/month/category structure
        dest_parts = Path(pdf_ops[0]["destination"]).parts
        assert any(part.isdigit() and len(part) <= 4 for part in dest_parts)
        assert "Documents" in pdf_ops[0]["destination"]

    @pytest.mark.asyncio
    async def test_organize_actual_execution(self, messy_directory):
        """Test actual file organization (not dry run)."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(messy_directory),
            organization_scheme="category",
            dry_run=False,
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data["dry_run"] is False

        # Check that files were actually moved
        operations = result.data["operations"]
        completed = [op for op in operations if op["status"] == "completed"]
        assert len(completed) > 0

        # Verify directories were created
        assert (messy_directory / "Documents").exists()
        assert (messy_directory / "Images").exists()
        assert (messy_directory / "Code").exists()

        # Verify files are in correct locations
        assert (messy_directory / "Documents" / "report.pdf").exists()
        assert (messy_directory / "Images" / "photo.jpg").exists()
        assert (messy_directory / "Code" / "script.py").exists()

        # Original files should not exist in root
        assert not (messy_directory / "report.pdf").exists()
        assert not (messy_directory / "photo.jpg").exists()
        assert not (messy_directory / "script.py").exists()

    @pytest.mark.asyncio
    async def test_organize_to_different_destination(self, messy_directory, tmp_path):
        """Test organizing to a different destination directory."""
        dest_dir = tmp_path / "organized"

        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(messy_directory),
            destination_path=str(dest_dir),
            organization_scheme="category",
            dry_run=False,
        )

        result = await tool.execute(params)

        assert result.success

        # Check that destination directory was created
        assert dest_dir.exists()

        # Check that files are in destination
        assert (dest_dir / "Documents" / "report.pdf").exists()
        assert (dest_dir / "Images" / "photo.jpg").exists()

        # Original files should not exist
        assert not (messy_directory / "report.pdf").exists()
        assert not (messy_directory / "photo.jpg").exists()

    @pytest.mark.asyncio
    async def test_conflict_resolution_skip(self, messy_directory):
        """Test conflict resolution with skip strategy."""
        # Create existing destination file
        docs_dir = messy_directory / "Documents"
        docs_dir.mkdir()
        (docs_dir / "report.pdf").write_text("existing content")

        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(messy_directory),
            organization_scheme="category",
            conflict_resolution="skip",
            dry_run=False,
        )

        result = await tool.execute(params)

        assert result.success

        # Original report.pdf should still exist (skipped)
        assert (messy_directory / "report.pdf").exists()

        # Existing file should be unchanged
        assert (docs_dir / "report.pdf").read_text() == "existing content"

        # Check that operation was skipped
        operations = result.data["operations"]
        pdf_ops = [op for op in operations if "report.pdf" in op["source"]]
        assert len(pdf_ops) == 1
        assert pdf_ops[0]["action"] == "skip"
        assert pdf_ops[0]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_conflict_resolution_rename(self, messy_directory):
        """Test conflict resolution with rename strategy."""
        # Create existing destination file
        docs_dir = messy_directory / "Documents"
        docs_dir.mkdir()
        (docs_dir / "report.pdf").write_text("existing content")

        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(messy_directory),
            organization_scheme="category",
            conflict_resolution="rename",
            dry_run=False,
        )

        result = await tool.execute(params)

        assert result.success

        # Original should be moved to renamed file
        assert not (messy_directory / "report.pdf").exists()

        # Existing file should be unchanged
        assert (docs_dir / "report.pdf").read_text() == "existing content"

        # Renamed file should exist (report_1.pdf)
        assert (docs_dir / "report_1.pdf").exists()

    @pytest.mark.asyncio
    async def test_conflict_resolution_overwrite_blocked(self, messy_directory):
        """Test that overwrite mode is blocked without HIGH risk approval."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(messy_directory),
            organization_scheme="category",
            conflict_resolution="overwrite",
            dry_run=False,
        )

        result = await tool.execute(params)

        # Should fail with error about HIGH risk approval
        assert not result.success
        assert "high risk approval" in result.error.lower()

    @pytest.mark.asyncio
    async def test_organize_recursive(self, tmp_path):
        """Test recursive organization."""
        # Create nested structure
        (tmp_path / "file1.pdf").write_text("file1")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.pdf").write_text("file2")
        nested = subdir / "nested"
        nested.mkdir()
        (nested / "file3.pdf").write_text("file3")

        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(tmp_path),
            organization_scheme="category",
            recursive=True,
            dry_run=True,
        )

        result = await tool.execute(params)

        assert result.success

        # Should find all PDF files
        operations = result.data["operations"]
        pdf_ops = [op for op in operations if ".pdf" in op["source"]]
        assert len(pdf_ops) == 3

    @pytest.mark.asyncio
    async def test_organize_non_recursive(self, tmp_path):
        """Test non-recursive organization (default)."""
        # Create nested structure
        (tmp_path / "file1.pdf").write_text("file1")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file2.pdf").write_text("file2")

        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(tmp_path),
            organization_scheme="category",
            recursive=False,
            dry_run=True,
        )

        result = await tool.execute(params)

        assert result.success

        # Should only find top-level PDF
        operations = result.data["operations"]
        pdf_ops = [op for op in operations if ".pdf" in op["source"]]
        assert len(pdf_ops) == 1

    @pytest.mark.asyncio
    async def test_organize_empty_directory(self, tmp_path):
        """Test organizing an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(empty_dir),
            organization_scheme="category",
            dry_run=True,
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data["summary"]["total_files"] == 0

    @pytest.mark.asyncio
    async def test_organize_nonexistent_directory(self):
        """Test organizing a nonexistent directory."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path="/nonexistent/directory",
            organization_scheme="category",
            dry_run=True,
        )

        result = await tool.execute(params)

        assert not result.success
        assert "does not exist" in result.error.lower()

    @pytest.mark.asyncio
    async def test_organize_file_not_directory(self, tmp_path):
        """Test organizing a file instead of directory."""
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(test_file),
            organization_scheme="category",
            dry_run=True,
        )

        result = await tool.execute(params)

        assert not result.success
        assert "not a directory" in result.error.lower()

    @pytest.mark.asyncio
    async def test_organize_skip_already_organized(self, tmp_path):
        """Test that already organized files are skipped."""
        # Create pre-organized structure
        docs_dir = tmp_path / "Documents"
        docs_dir.mkdir()
        (docs_dir / "report.pdf").write_text("content")

        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(tmp_path),
            organization_scheme="category",
            recursive=True,
            dry_run=True,
        )

        result = await tool.execute(params)

        assert result.success

        # File in Documents folder should be skipped (already organized)
        operations = result.data["operations"]
        # Should have no operations or only skip operations
        move_ops = [op for op in operations if op["action"] == "move"]
        assert len(move_ops) == 0

    @pytest.mark.asyncio
    async def test_confirmation_message_dry_run(self):
        """Test confirmation message for dry run."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path="/test/path",
            organization_scheme="category",
            dry_run=True,
        )

        msg = tool.get_confirmation_message(params)
        assert "dry run" in msg.lower()
        assert "no changes" in msg.lower()
        assert "/test/path" in msg

    @pytest.mark.asyncio
    async def test_confirmation_message_actual_run(self):
        """Test confirmation message for actual execution."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path="/test/path",
            destination_path="/output/path",
            organization_scheme="date",
            conflict_resolution="rename",
            dry_run=False,
        )

        msg = tool.get_confirmation_message(params)
        assert "/test/path" in msg
        assert "/output/path" in msg
        assert "date" in msg.lower()
        assert "rename" in msg.lower()

    @pytest.mark.asyncio
    async def test_organize_with_errors(self, tmp_path, monkeypatch):
        """Test that errors are handled gracefully."""
        # Create files
        (tmp_path / "file1.pdf").write_text("content1")
        (tmp_path / "file2.pdf").write_text("content2")

        # Mock Path.rename to fail for one file
        original_rename = Path.rename

        def mock_rename(self, target):
            if "file1" in str(self):
                raise PermissionError("Mock permission error")
            return original_rename(self, target)

        monkeypatch.setattr(Path, "rename", mock_rename)

        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(tmp_path),
            organization_scheme="category",
            dry_run=False,
        )

        result = await tool.execute(params)

        # Should still succeed overall
        assert result.success

        # Check that one operation failed and one succeeded
        operations = result.data["operations"]
        error_ops = [op for op in operations if op["status"] == "error"]
        completed_ops = [op for op in operations if op["status"] == "completed"]

        assert len(error_ops) == 1
        assert len(completed_ops) == 1

        # Summary should show errors
        assert result.data["summary"]["errors"] == 1
        assert result.data["summary"]["completed"] == 1
