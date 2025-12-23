"""Integration tests for organize-undo workflow."""

from pathlib import Path

import pytest

from aria.tools.filesystem.organize_files import OrganizeFilesTool, OrganizeFilesParams
from aria.tools.organization import (
    ListOrganizationLogsTool,
    ListOrganizationLogsParams,
    UndoOrganizationTool,
    UndoOrganizationParams,
)


@pytest.fixture
def temp_logs_dir(tmp_path, monkeypatch):
    """Create a temporary logs directory."""
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
def messy_directory(tmp_path):
    """Create a messy directory for testing."""
    messy = tmp_path / "messy"
    messy.mkdir()

    # Create various file types
    (messy / "report.pdf").write_text("PDF content")
    (messy / "photo.jpg").write_text("JPG content")
    (messy / "script.py").write_text("Python code")
    (messy / "data.xlsx").write_text("Excel data")

    return messy


class TestOrganizeUndoIntegration:
    """Test the complete organize-undo workflow."""

    @pytest.mark.asyncio
    async def test_complete_organize_undo_workflow(self, temp_logs_dir, messy_directory):
        """Test the complete workflow: organize → list logs → undo."""

        # Step 1: Organize files
        organize_tool = OrganizeFilesTool()
        organize_params = OrganizeFilesParams(
            source_path=str(messy_directory),
            organization_scheme="category",
            dry_run=False,
        )

        organize_result = await organize_tool.execute(organize_params)

        assert organize_result.success
        assert organize_result.data["dry_run"] is False
        assert "log_file" in organize_result.data

        # Verify files were moved
        assert (messy_directory / "Documents" / "report.pdf").exists()
        assert (messy_directory / "Images" / "photo.jpg").exists()
        assert (messy_directory / "Code" / "script.py").exists()
        assert (messy_directory / "Spreadsheets" / "data.xlsx").exists()

        # Original files should not exist
        assert not (messy_directory / "report.pdf").exists()
        assert not (messy_directory / "photo.jpg").exists()

        # Step 2: List organization logs
        list_tool = ListOrganizationLogsTool()
        list_params = ListOrganizationLogsParams(limit=10)

        list_result = await list_tool.execute(list_params)

        assert list_result.success
        assert list_result.data["total_logs"] >= 1

        # Find our log
        logs = list_result.data["logs"]
        our_log = next((log for log in logs if log["source_path"] == str(messy_directory)), None)
        assert our_log is not None
        assert our_log["scheme"] == "category"
        assert our_log["total_files"] == 4
        assert our_log["completed"] == 4

        # Step 3: Undo the organization (dry run first)
        undo_tool = UndoOrganizationTool()
        undo_dry_params = UndoOrganizationParams(dry_run=True)

        undo_dry_result = await undo_tool.execute(undo_dry_params)

        assert undo_dry_result.success
        assert undo_dry_result.data["dry_run"] is True
        assert undo_dry_result.data["summary"]["can_undo"] == 4
        assert undo_dry_result.data["summary"]["cannot_undo"] == 0

        # Files should still be organized
        assert (messy_directory / "Documents" / "report.pdf").exists()

        # Step 4: Actually undo the organization
        undo_params = UndoOrganizationParams(dry_run=False)

        undo_result = await undo_tool.execute(undo_params)

        assert undo_result.success
        assert undo_result.data["dry_run"] is False
        assert undo_result.data["summary"]["undone"] == 4
        assert undo_result.data["summary"]["failed"] == 0

        # Verify files are back in original location
        assert (messy_directory / "report.pdf").exists()
        assert (messy_directory / "photo.jpg").exists()
        assert (messy_directory / "script.py").exists()
        assert (messy_directory / "data.xlsx").exists()

        # Organized directories should be cleaned up
        assert not (messy_directory / "Documents" / "report.pdf").exists()
        assert not (messy_directory / "Images" / "photo.jpg").exists()

    @pytest.mark.asyncio
    async def test_organize_multiple_times_undo_specific(self, temp_logs_dir, tmp_path):
        """Test organizing multiple times and undoing a specific operation."""

        # Create first messy directory
        messy1 = tmp_path / "messy1"
        messy1.mkdir()
        (messy1 / "file1.pdf").write_text("content1")

        # Create second messy directory
        messy2 = tmp_path / "messy2"
        messy2.mkdir()
        (messy2 / "file2.jpg").write_text("content2")

        # Organize first directory
        organize_tool = OrganizeFilesTool()
        params1 = OrganizeFilesParams(
            source_path=str(messy1),
            organization_scheme="category",
            dry_run=False,
        )

        result1 = await organize_tool.execute(params1)
        assert result1.success
        log1_path = result1.data["log_file"]

        # Organize second directory
        params2 = OrganizeFilesParams(
            source_path=str(messy2),
            organization_scheme="category",
            dry_run=False,
        )

        result2 = await organize_tool.execute(params2)
        assert result2.success
        log2_path = result2.data["log_file"]

        # List logs - should have both
        list_tool = ListOrganizationLogsTool()
        list_result = await list_tool.execute(ListOrganizationLogsParams())

        assert list_result.success
        assert list_result.data["total_logs"] == 2

        # Undo only the first organization
        undo_tool = UndoOrganizationTool()
        undo_params = UndoOrganizationParams(
            log_file=log1_path,
            dry_run=False,
        )

        undo_result = await undo_tool.execute(undo_params)

        assert undo_result.success

        # First directory should be undone
        assert (messy1 / "file1.pdf").exists()
        assert not (messy1 / "Documents" / "file1.pdf").exists()

        # Second directory should still be organized
        assert (messy2 / "Images" / "file2.jpg").exists()
        assert not (messy2 / "file2.jpg").exists()

    @pytest.mark.asyncio
    async def test_partial_undo_when_files_changed(self, temp_logs_dir, messy_directory):
        """Test undo when some files have been deleted or moved."""

        # Organize files
        organize_tool = OrganizeFilesTool()
        organize_params = OrganizeFilesParams(
            source_path=str(messy_directory),
            organization_scheme="category",
            dry_run=False,
        )

        organize_result = await organize_tool.execute(organize_params)
        assert organize_result.success

        # Delete one organized file
        (messy_directory / "Documents" / "report.pdf").unlink()

        # Move another file manually
        (messy_directory / "script.py").write_text("new file")

        # Try to undo
        undo_tool = UndoOrganizationTool()
        undo_params = UndoOrganizationParams(dry_run=False)

        undo_result = await undo_tool.execute(undo_params)

        assert undo_result.success

        # Summary should show partial undo
        summary = undo_result.data["summary"]
        assert summary["undone"] > 0  # Some files undone
        assert summary["cannot_undo"] > 0  # Some files cannot be undone

        # Check specific results
        # report.pdf cannot be undone (deleted)
        # script.py cannot be undone (source occupied)
        # photo.jpg and data.xlsx should be undone
        assert (messy_directory / "photo.jpg").exists()
        assert (messy_directory / "data.xlsx").exists()

    @pytest.mark.asyncio
    async def test_organize_with_log_saved(self, temp_logs_dir, messy_directory):
        """Test that organize_files saves a log when dry_run=False."""

        # Initially no logs
        list_tool = ListOrganizationLogsTool()
        initial_result = await list_tool.execute(ListOrganizationLogsParams())
        initial_count = initial_result.data["total_logs"]

        # Organize files
        organize_tool = OrganizeFilesTool()
        organize_params = OrganizeFilesParams(
            source_path=str(messy_directory),
            organization_scheme="extension",
            dry_run=False,
        )

        organize_result = await organize_tool.execute(organize_params)

        assert organize_result.success
        assert "log_file" in organize_result.data

        # Verify log was saved
        log_file_path = Path(organize_result.data["log_file"])
        assert log_file_path.exists()

        # List logs should show the new log
        after_result = await list_tool.execute(ListOrganizationLogsParams())
        assert after_result.data["total_logs"] == initial_count + 1

        # Verify log content
        new_log = after_result.data["logs"][0]
        assert new_log["source_path"] == str(messy_directory)
        assert new_log["scheme"] == "extension"

    @pytest.mark.asyncio
    async def test_organize_dry_run_no_log_saved(self, temp_logs_dir, messy_directory):
        """Test that organize_files doesn't save a log when dry_run=True."""

        # Get initial log count
        list_tool = ListOrganizationLogsTool()
        initial_result = await list_tool.execute(ListOrganizationLogsParams())
        initial_count = initial_result.data["total_logs"]

        # Organize in dry run mode
        organize_tool = OrganizeFilesTool()
        organize_params = OrganizeFilesParams(
            source_path=str(messy_directory),
            organization_scheme="category",
            dry_run=True,
        )

        organize_result = await organize_tool.execute(organize_params)

        assert organize_result.success
        assert organize_result.data["dry_run"] is True
        assert "log_file" not in organize_result.data

        # Log count should be unchanged
        after_result = await list_tool.execute(ListOrganizationLogsParams())
        assert after_result.data["total_logs"] == initial_count
