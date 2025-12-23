"""Comprehensive integration tests for file organization workflow."""

import time
from pathlib import Path
from datetime import datetime, timedelta

import pytest

from aria.tools.filesystem import (
    AnalyzeDirectoryTool,
    AnalyzeDirectoryParams,
    OrganizeFilesTool,
    OrganizeFilesParams,
)
from aria.tools.organization import (
    UndoOrganizationTool,
    UndoOrganizationParams,
    ListOrganizationLogsTool,
    ListOrganizationLogsParams,
)


@pytest.fixture
def temp_logs_dir(tmp_path, monkeypatch):
    """Create a temporary logs directory for testing."""
    logs_dir = tmp_path / "organization_logs"
    logs_dir.mkdir()

    # Monkeypatch get_logs_directory
    monkeypatch.setattr("aria.tools.organization.log_manager.get_logs_directory", lambda: logs_dir)
    monkeypatch.setattr(
        "aria.tools.filesystem.organize_files.save_organization_log",
        lambda **kwargs: __import__(
            "aria.tools.organization.log_manager", fromlist=["save_organization_log"]
        ).save_organization_log(**kwargs),
    )

    return logs_dir


@pytest.fixture
def empty_directory(tmp_path):
    """Create an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    return empty_dir


@pytest.fixture
def mixed_files_directory(tmp_path):
    """Create directory with mixed file types.

    Contains:
    - 5 PDFs
    - 10 images (jpg/png mix)
    - 3 Python files
    - 2 unknown file types
    """
    mixed_dir = tmp_path / "mixed"
    mixed_dir.mkdir()

    # PDFs
    for i in range(5):
        (mixed_dir / f"document{i}.pdf").write_text(f"PDF content {i}")

    # Images
    for i in range(6):
        (mixed_dir / f"photo{i}.jpg").write_bytes(f"JPG {i}".encode())
    for i in range(4):
        (mixed_dir / f"image{i}.png").write_bytes(f"PNG {i}".encode())

    # Python files
    for i in range(3):
        (mixed_dir / f"script{i}.py").write_text(f"print('hello {i}')")

    # Unknown types
    (mixed_dir / "data.dat").write_bytes(b"binary data")
    (mixed_dir / "config.conf").write_text("config=value")

    return mixed_dir


@pytest.fixture
def dated_files_directory(tmp_path):
    """Create directory with files having different modification dates."""
    dated_dir = tmp_path / "dated"
    dated_dir.mkdir()

    # Current time
    now = time.time()

    # Files from different months
    files_with_dates = [
        ("old_file1.txt", now - 86400 * 60),  # 2 months ago
        ("old_file2.pdf", now - 86400 * 45),  # 1.5 months ago
        ("recent_file1.jpg", now - 86400 * 5),  # 5 days ago
        ("recent_file2.doc", now - 86400 * 3),  # 3 days ago
        ("today_file.txt", now),  # Now
    ]

    for filename, mtime in files_with_dates:
        file_path = dated_dir / filename
        file_path.write_text(f"Content of {filename}")
        Path(file_path).touch(mtime)

    return dated_dir


@pytest.fixture
def large_directory(tmp_path):
    """Create directory with many files for performance testing."""
    large_dir = tmp_path / "large"
    large_dir.mkdir()

    # Create 500 small files with various types
    for i in range(200):
        (large_dir / f"doc{i}.pdf").write_text(f"doc{i}")
    for i in range(150):
        (large_dir / f"img{i}.jpg").write_bytes(f"img{i}".encode())
    for i in range(100):
        (large_dir / f"code{i}.py").write_text(f"code{i}")
    for i in range(50):
        (large_dir / f"data{i}.csv").write_text(f"data{i}")

    return large_dir


class TestFileOrganizationIntegration:
    """Integration tests for file organization workflow."""

    @pytest.mark.asyncio
    async def test_analyze_empty_directory(self, empty_directory):
        """Test analyzing an empty directory."""
        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(
            path=str(empty_directory),
            recursive=False,
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data["total_files"] == 0
        assert result.data["total_size_bytes"] == 0
        assert result.data["categories"] == {}
        assert result.data["by_extension"] == {}

    @pytest.mark.asyncio
    async def test_analyze_mixed_files(self, mixed_files_directory):
        """Test analyzing directory with mixed file types."""
        tool = AnalyzeDirectoryTool()
        params = AnalyzeDirectoryParams(
            path=str(mixed_files_directory),
            recursive=False,
        )

        result = await tool.execute(params)

        assert result.success
        data = result.data

        # Verify total count
        assert data["total_files"] == 20  # 5 PDF + 10 images + 3 Python + 2 unknown

        # Verify categories
        categories = data["categories"]
        assert categories["document"]["count"] == 5  # PDFs
        assert categories["image"]["count"] == 10  # JPG + PNG
        assert categories["code"]["count"] == 3  # Python files
        assert categories["other"]["count"] == 2  # Unknown types

        # Verify extensions
        by_ext = data["by_extension"]
        assert by_ext[".pdf"]["count"] == 5
        assert by_ext[".jpg"]["count"] == 6
        assert by_ext[".png"]["count"] == 4
        assert by_ext[".py"]["count"] == 3

        # Verify largest files list is populated
        assert len(data["largest_files"]) > 0

    @pytest.mark.asyncio
    async def test_organize_dry_run(self, temp_logs_dir, mixed_files_directory):
        """Test organizing with dry run - no actual changes."""
        # Get initial file list
        initial_files = set(mixed_files_directory.iterdir())

        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(mixed_files_directory),
            organization_scheme="category",
            dry_run=True,
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data["dry_run"] is True

        # Verify operations were planned
        operations = result.data["operations"]
        assert len(operations) == 20

        # Verify summary
        summary = result.data["summary"]
        assert summary["to_move"] == 20
        assert "Documents/" in summary["by_destination"]
        assert "Images/" in summary["by_destination"]
        assert "Code/" in summary["by_destination"]

        # Verify NO files were actually moved
        current_files = set(mixed_files_directory.iterdir())
        assert initial_files == current_files

        # Verify no subdirectories were created
        subdirs = [d for d in mixed_files_directory.iterdir() if d.is_dir()]
        assert len(subdirs) == 0

    @pytest.mark.asyncio
    async def test_organize_by_category(self, temp_logs_dir, mixed_files_directory):
        """Test organizing files by category."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(mixed_files_directory),
            organization_scheme="category",
            dry_run=False,
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data["dry_run"] is False

        # Verify operation log was created
        assert "log_file" in result.data
        log_file = Path(result.data["log_file"])
        assert log_file.exists()

        # Verify summary
        summary = result.data["summary"]
        assert summary["completed"] == 20
        assert summary["errors"] == 0

        # Verify folders were created
        docs_dir = mixed_files_directory / "Documents"
        images_dir = mixed_files_directory / "Images"
        code_dir = mixed_files_directory / "Code"
        other_dir = mixed_files_directory / "Other"

        assert docs_dir.exists() and docs_dir.is_dir()
        assert images_dir.exists() and images_dir.is_dir()
        assert code_dir.exists() and code_dir.is_dir()
        assert other_dir.exists() and other_dir.is_dir()

        # Verify files were moved correctly
        assert len(list(docs_dir.glob("*.pdf"))) == 5
        assert len(list(images_dir.glob("*.jpg"))) == 6
        assert len(list(images_dir.glob("*.png"))) == 4
        assert len(list(code_dir.glob("*.py"))) == 3
        assert len(list(other_dir.glob("*"))) == 2

        # Verify original files are gone from root
        root_files = [f for f in mixed_files_directory.iterdir() if f.is_file()]
        assert len(root_files) == 0

    @pytest.mark.asyncio
    async def test_organize_by_date(self, temp_logs_dir, dated_files_directory):
        """Test organizing files by modification date."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(dated_files_directory),
            organization_scheme="date",
            dry_run=False,
        )

        result = await tool.execute(params)

        assert result.success

        # Verify YYYY/MM folder structure was created
        year_dirs = [d for d in dated_files_directory.iterdir() if d.is_dir() and d.name.isdigit()]
        assert len(year_dirs) > 0

        # Verify files are organized by date
        all_organized_files = []
        for year_dir in year_dirs:
            for month_dir in year_dir.iterdir():
                if month_dir.is_dir():
                    all_organized_files.extend(month_dir.glob("*"))

        assert len(all_organized_files) == 5

    @pytest.mark.asyncio
    async def test_organize_by_extension(self, temp_logs_dir, mixed_files_directory):
        """Test organizing files by extension."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(mixed_files_directory),
            organization_scheme="extension",
            dry_run=False,
        )

        result = await tool.execute(params)

        assert result.success

        # Verify extension-based folders were created
        assert (mixed_files_directory / "pdf").exists()
        assert (mixed_files_directory / "jpg").exists()
        assert (mixed_files_directory / "png").exists()
        assert (mixed_files_directory / "py").exists()

        # Verify correct file counts
        assert len(list((mixed_files_directory / "pdf").iterdir())) == 5
        assert len(list((mixed_files_directory / "jpg").iterdir())) == 6
        assert len(list((mixed_files_directory / "png").iterdir())) == 4
        assert len(list((mixed_files_directory / "py").iterdir())) == 3

    @pytest.mark.asyncio
    async def test_organize_by_date_category(self, temp_logs_dir, dated_files_directory):
        """Test organizing files by date and category combined."""
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(dated_files_directory),
            organization_scheme="date_category",
            dry_run=False,
        )

        result = await tool.execute(params)

        assert result.success

        # Verify nested structure: YYYY/MM/Category/
        year_dirs = [d for d in dated_files_directory.iterdir() if d.is_dir() and d.name.isdigit()]
        assert len(year_dirs) > 0

        # Verify category folders exist within date folders
        found_categories = set()
        for year_dir in year_dirs:
            for month_dir in year_dir.iterdir():
                if month_dir.is_dir():
                    for category_dir in month_dir.iterdir():
                        if category_dir.is_dir():
                            found_categories.add(category_dir.name)

        # Should have at least Documents and Images
        assert "Documents" in found_categories or "Images" in found_categories

    @pytest.mark.asyncio
    async def test_undo_organization(self, temp_logs_dir, mixed_files_directory):
        """Test complete organize-undo workflow."""
        # Step 1: Organize
        organize_tool = OrganizeFilesTool()
        organize_params = OrganizeFilesParams(
            source_path=str(mixed_files_directory),
            organization_scheme="category",
            dry_run=False,
        )

        organize_result = await organize_tool.execute(organize_params)
        assert organize_result.success

        # Verify files were organized
        assert (mixed_files_directory / "Documents").exists()
        assert (mixed_files_directory / "Images").exists()

        # Step 2: Undo
        undo_tool = UndoOrganizationTool()
        undo_params = UndoOrganizationParams(dry_run=False)

        undo_result = await undo_tool.execute(undo_params)
        assert undo_result.success

        # Verify files are back in original locations
        summary = undo_result.data["summary"]
        assert summary["undone"] == 20
        assert summary["failed"] == 0

        # Verify all files are back in root
        root_files = [f for f in mixed_files_directory.iterdir() if f.is_file()]
        assert len(root_files) == 20

        # Verify empty directories were cleaned up
        subdirs = [d for d in mixed_files_directory.iterdir() if d.is_dir()]
        assert len(subdirs) == 0

    @pytest.mark.asyncio
    async def test_conflict_resolution_skip(self, temp_logs_dir, tmp_path):
        """Test conflict resolution with skip strategy."""
        # Create source directory
        source = tmp_path / "source"
        source.mkdir()
        (source / "file.txt").write_text("original")

        # Create destination with existing file
        docs = source / "Documents"
        docs.mkdir()
        (docs / "file.txt").write_text("existing")

        # Organize with skip
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(source),
            organization_scheme="category",
            conflict_resolution="skip",
            dry_run=False,
        )

        result = await tool.execute(params)
        assert result.success

        # Verify original file still in source (skipped)
        assert (source / "file.txt").exists()

        # Verify existing file unchanged
        assert (docs / "file.txt").read_text() == "existing"

    @pytest.mark.asyncio
    async def test_conflict_resolution_rename(self, temp_logs_dir, tmp_path):
        """Test conflict resolution with rename strategy."""
        # Create source directory
        source = tmp_path / "source"
        source.mkdir()
        (source / "file.txt").write_text("new")

        # Create destination with existing file
        docs = source / "Documents"
        docs.mkdir()
        (docs / "file.txt").write_text("existing")

        # Organize with rename
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(source),
            organization_scheme="category",
            conflict_resolution="rename",
            dry_run=False,
        )

        result = await tool.execute(params)
        assert result.success

        # Verify original moved to renamed file
        assert not (source / "file.txt").exists()

        # Verify existing file unchanged
        assert (docs / "file.txt").read_text() == "existing"

        # Verify renamed file exists
        assert (docs / "file_1.txt").exists()
        assert (docs / "file_1.txt").read_text() == "new"

    @pytest.mark.asyncio
    async def test_large_directory_performance(self, temp_logs_dir, large_directory):
        """Test performance with large directory (500 files)."""
        # Test analysis performance
        analyze_tool = AnalyzeDirectoryTool()
        analyze_params = AnalyzeDirectoryParams(
            path=str(large_directory),
            recursive=False,
        )

        start_time = time.time()
        analyze_result = await analyze_tool.execute(analyze_params)
        analyze_time = time.time() - start_time

        assert analyze_result.success
        assert analyze_result.data["total_files"] == 500
        assert analyze_time < 5.0, f"Analysis took {analyze_time:.2f}s, expected < 5s"

        # Test organization performance
        organize_tool = OrganizeFilesTool()
        organize_params = OrganizeFilesParams(
            source_path=str(large_directory),
            organization_scheme="category",
            dry_run=False,
        )

        start_time = time.time()
        organize_result = await organize_tool.execute(organize_params)
        organize_time = time.time() - start_time

        assert organize_result.success
        assert organize_result.data["summary"]["completed"] == 500
        assert organize_time < 30.0, f"Organization took {organize_time:.2f}s, expected < 30s"

    @pytest.mark.asyncio
    async def test_list_organization_logs(self, temp_logs_dir, mixed_files_directory):
        """Test listing organization logs."""
        # Create multiple organization operations
        tool = OrganizeFilesTool()

        for i in range(3):
            params = OrganizeFilesParams(
                source_path=str(mixed_files_directory),
                organization_scheme="category",
                dry_run=False,
            )
            await tool.execute(params)

            # Undo for next iteration
            undo_tool = UndoOrganizationTool()
            undo_params = UndoOrganizationParams(dry_run=False)
            await undo_tool.execute(undo_params)

        # List logs
        list_tool = ListOrganizationLogsTool()
        list_params = ListOrganizationLogsParams(limit=10)

        result = await list_tool.execute(list_params)

        assert result.success
        assert result.data["total_logs"] >= 3

        # Verify log structure
        for log in result.data["logs"]:
            assert "timestamp" in log
            assert "log_file" in log
            assert "source_path" in log
            assert "scheme" in log
            assert "total_files" in log

    @pytest.mark.asyncio
    async def test_recursive_organization(self, temp_logs_dir, tmp_path):
        """Test recursive directory organization."""
        # Create nested structure
        root = tmp_path / "root"
        root.mkdir()

        (root / "file1.pdf").write_text("file1")

        subdir1 = root / "sub1"
        subdir1.mkdir()
        (subdir1 / "file2.pdf").write_text("file2")

        subdir2 = subdir1 / "sub2"
        subdir2.mkdir()
        (subdir2 / "file3.pdf").write_text("file3")

        # Organize recursively
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(root),
            organization_scheme="category",
            recursive=True,
            dry_run=False,
        )

        result = await tool.execute(params)

        assert result.success
        assert result.data["summary"]["completed"] == 3

        # Verify all PDFs in Documents folder
        docs_dir = root / "Documents"
        assert docs_dir.exists()
        assert len(list(docs_dir.glob("*.pdf"))) == 3

    @pytest.mark.asyncio
    async def test_organize_to_different_destination(self, temp_logs_dir, tmp_path):
        """Test organizing to a different destination directory."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "file1.pdf").write_text("content1")
        (source / "file2.jpg").write_bytes(b"content2")

        dest = tmp_path / "destination"

        # Organize to different location
        tool = OrganizeFilesTool()
        params = OrganizeFilesParams(
            source_path=str(source),
            destination_path=str(dest),
            organization_scheme="category",
            dry_run=False,
        )

        result = await tool.execute(params)

        assert result.success

        # Verify destination structure
        assert (dest / "Documents" / "file1.pdf").exists()
        assert (dest / "Images" / "file2.jpg").exists()

        # Verify source is empty
        source_files = list(source.glob("*"))
        assert len(source_files) == 0

    @pytest.mark.asyncio
    async def test_workflow_end_to_end(self, temp_logs_dir, mixed_files_directory):
        """Test complete end-to-end workflow: analyze → organize → list → undo."""
        # Step 1: Analyze
        analyze_tool = AnalyzeDirectoryTool()
        analyze_params = AnalyzeDirectoryParams(
            path=str(mixed_files_directory),
            recursive=False,
        )

        analyze_result = await analyze_tool.execute(analyze_params)
        assert analyze_result.success
        initial_file_count = analyze_result.data["total_files"]

        # Step 2: Organize
        organize_tool = OrganizeFilesTool()
        organize_params = OrganizeFilesParams(
            source_path=str(mixed_files_directory),
            organization_scheme="category",
            dry_run=False,
        )

        organize_result = await organize_tool.execute(organize_params)
        assert organize_result.success
        assert "log_file" in organize_result.data

        # Step 3: List logs
        list_tool = ListOrganizationLogsTool()
        list_params = ListOrganizationLogsParams(limit=5)

        list_result = await list_tool.execute(list_params)
        assert list_result.success
        assert list_result.data["total_logs"] >= 1

        # Step 4: Undo
        undo_tool = UndoOrganizationTool()
        undo_params = UndoOrganizationParams(dry_run=False)

        undo_result = await undo_tool.execute(undo_params)
        assert undo_result.success

        # Step 5: Verify restoration
        final_analyze_result = await analyze_tool.execute(analyze_params)
        assert final_analyze_result.success
        assert final_analyze_result.data["total_files"] == initial_file_count

        # Verify directory structure is clean
        subdirs = [d for d in mixed_files_directory.iterdir() if d.is_dir()]
        assert len(subdirs) == 0
