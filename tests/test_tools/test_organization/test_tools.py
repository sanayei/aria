"""Tests for organization tools (list_logs and undo)."""

from pathlib import Path

import pytest

from aria.tools.organization import (
    ListOrganizationLogsTool,
    ListOrganizationLogsParams,
    UndoOrganizationTool,
    UndoOrganizationParams,
)
from aria.tools.organization.log_manager import save_organization_log


@pytest.fixture
def temp_logs_dir(tmp_path, monkeypatch):
    """Create a temporary logs directory."""
    logs_dir = tmp_path / "organization_logs"
    logs_dir.mkdir()

    # Monkeypatch get_logs_directory to use temp directory
    monkeypatch.setattr("aria.tools.organization.log_manager.get_logs_directory", lambda: logs_dir)
    monkeypatch.setattr(
        "aria.tools.organization.list_logs.list_organization_logs",
        lambda: __import__(
            "aria.tools.organization.log_manager", fromlist=["list_organization_logs"]
        ).list_organization_logs(),
    )
    monkeypatch.setattr(
        "aria.tools.organization.undo.load_organization_log",
        lambda path: __import__(
            "aria.tools.organization.log_manager", fromlist=["load_organization_log"]
        ).load_organization_log(path),
    )
    monkeypatch.setattr(
        "aria.tools.organization.undo.get_most_recent_log",
        lambda: __import__(
            "aria.tools.organization.log_manager", fromlist=["get_most_recent_log"]
        ).get_most_recent_log(),
    )

    return logs_dir


@pytest.fixture
def organized_directory(tmp_path):
    """Create a directory with organized files."""
    source = tmp_path / "source"
    source.mkdir()

    # Create organized structure
    docs = source / "Documents"
    docs.mkdir()
    (docs / "report.pdf").write_text("PDF content")

    images = source / "Images"
    images.mkdir()
    (images / "photo.jpg").write_text("JPG content")

    return source


class TestListOrganizationLogsTool:
    """Test ListOrganizationLogsTool."""

    @pytest.mark.asyncio
    async def test_list_logs_empty(self, temp_logs_dir):
        """Test listing logs when no logs exist."""
        tool = ListOrganizationLogsTool()
        params = ListOrganizationLogsParams()

        result = await tool.execute(params)

        assert result.success
        assert result.data["total_logs"] == 0
        assert len(result.data["logs"]) == 0
        assert "No organization logs found" in result.data["message"]

    @pytest.mark.asyncio
    async def test_list_logs_with_logs(self, temp_logs_dir):
        """Test listing logs when logs exist."""
        # Create test logs
        for i in range(3):
            operations = [
                {
                    "source": f"/test/file{i}.txt",
                    "destination": f"/test/Documents/file{i}.txt",
                    "action": "move",
                    "status": "completed",
                }
            ]

            save_organization_log(
                source_path=f"/test{i}",
                destination_path=f"/test{i}",
                scheme="category",
                operations=operations,
            )

        tool = ListOrganizationLogsTool()
        params = ListOrganizationLogsParams(limit=10)

        result = await tool.execute(params)

        assert result.success
        assert result.data["total_logs"] == 3
        assert len(result.data["logs"]) == 3

        # Check log data structure
        for log in result.data["logs"]:
            assert "timestamp" in log
            assert "log_file" in log
            assert "source_path" in log
            assert "destination_path" in log
            assert "scheme" in log
            assert "total_files" in log
            assert "completed" in log
            assert "failed" in log

    @pytest.mark.asyncio
    async def test_list_logs_with_limit(self, temp_logs_dir):
        """Test limiting number of logs returned."""
        # Create 5 logs
        for i in range(5):
            operations = [
                {
                    "source": f"/test/file{i}.txt",
                    "destination": f"/test/Documents/file{i}.txt",
                    "action": "move",
                    "status": "completed",
                }
            ]

            save_organization_log(
                source_path="/test",
                destination_path="/test",
                scheme="category",
                operations=operations,
            )

        # Request only 2
        tool = ListOrganizationLogsTool()
        params = ListOrganizationLogsParams(limit=2)

        result = await tool.execute(params)

        assert result.success
        assert result.data["showing"] == 2
        assert len(result.data["logs"]) == 2

    @pytest.mark.asyncio
    async def test_confirmation_message(self):
        """Test confirmation message."""
        tool = ListOrganizationLogsTool()
        params = ListOrganizationLogsParams(limit=5)

        msg = tool.get_confirmation_message(params)
        assert "5" in msg


class TestUndoOrganizationTool:
    """Test UndoOrganizationTool."""

    @pytest.mark.asyncio
    async def test_undo_no_logs(self, temp_logs_dir):
        """Test undo when no logs exist."""
        tool = UndoOrganizationTool()
        params = UndoOrganizationParams(dry_run=True)

        result = await tool.execute(params)

        assert not result.success
        assert "No organization logs found" in result.error

    @pytest.mark.asyncio
    async def test_undo_dry_run(self, temp_logs_dir, organized_directory):
        """Test undo in dry run mode."""
        # Create a log
        operations = [
            {
                "source": str(organized_directory.parent / "report.pdf"),
                "destination": str(organized_directory / "Documents" / "report.pdf"),
                "action": "move",
                "status": "completed",
            },
            {
                "source": str(organized_directory.parent / "photo.jpg"),
                "destination": str(organized_directory / "Images" / "photo.jpg"),
                "action": "move",
                "status": "completed",
            },
        ]

        save_organization_log(
            source_path=str(organized_directory.parent),
            destination_path=str(organized_directory),
            scheme="category",
            operations=operations,
        )

        tool = UndoOrganizationTool()
        params = UndoOrganizationParams(dry_run=True)

        result = await tool.execute(params)

        assert result.success
        assert result.data["dry_run"] is True
        assert result.data["original_scheme"] == "category"

        # Check that files can be undone
        summary = result.data["summary"]
        assert summary["can_undo"] == 2
        assert summary["cannot_undo"] == 0

        # Verify files weren't actually moved
        assert (organized_directory / "Documents" / "report.pdf").exists()
        assert (organized_directory / "Images" / "photo.jpg").exists()

    @pytest.mark.asyncio
    async def test_undo_actual_execution(self, temp_logs_dir, organized_directory):
        """Test actually undoing organization."""
        source_dir = organized_directory.parent

        # Create a log
        operations = [
            {
                "source": str(source_dir / "report.pdf"),
                "destination": str(organized_directory / "Documents" / "report.pdf"),
                "action": "move",
                "status": "completed",
            },
            {
                "source": str(source_dir / "photo.jpg"),
                "destination": str(organized_directory / "Images" / "photo.jpg"),
                "action": "move",
                "status": "completed",
            },
        ]

        save_organization_log(
            source_path=str(source_dir),
            destination_path=str(organized_directory),
            scheme="category",
            operations=operations,
        )

        tool = UndoOrganizationTool()
        params = UndoOrganizationParams(dry_run=False)

        result = await tool.execute(params)

        assert result.success
        assert result.data["dry_run"] is False

        # Check that files were moved back
        assert (source_dir / "report.pdf").exists()
        assert (source_dir / "photo.jpg").exists()

        # Original locations should be empty
        assert not (organized_directory / "Documents" / "report.pdf").exists()
        assert not (organized_directory / "Images" / "photo.jpg").exists()

        # Summary should show successful undo
        summary = result.data["summary"]
        assert summary["undone"] == 2
        assert summary["failed"] == 0

    @pytest.mark.asyncio
    async def test_undo_file_no_longer_exists(self, temp_logs_dir, organized_directory):
        """Test undo when destination file no longer exists."""
        source_dir = organized_directory.parent

        # Create a log
        operations = [
            {
                "source": str(source_dir / "report.pdf"),
                "destination": str(organized_directory / "Documents" / "report.pdf"),
                "action": "move",
                "status": "completed",
            },
        ]

        save_organization_log(
            source_path=str(source_dir),
            destination_path=str(organized_directory),
            scheme="category",
            operations=operations,
        )

        # Delete the destination file
        (organized_directory / "Documents" / "report.pdf").unlink()

        tool = UndoOrganizationTool()
        params = UndoOrganizationParams(dry_run=True)

        result = await tool.execute(params)

        assert result.success

        # Should report that file cannot be undone
        summary = result.data["summary"]
        assert summary["can_undo"] == 0
        assert summary["cannot_undo"] == 1

        cannot_undo = result.data["cannot_undo"]
        assert len(cannot_undo) == 1
        assert "no longer exists" in cannot_undo[0]["reason"].lower()

    @pytest.mark.asyncio
    async def test_undo_source_already_occupied(self, temp_logs_dir, organized_directory):
        """Test undo when source location is already occupied."""
        source_dir = organized_directory.parent

        # Create a log
        operations = [
            {
                "source": str(source_dir / "report.pdf"),
                "destination": str(organized_directory / "Documents" / "report.pdf"),
                "action": "move",
                "status": "completed",
            },
        ]

        save_organization_log(
            source_path=str(source_dir),
            destination_path=str(organized_directory),
            scheme="category",
            operations=operations,
        )

        # Create a file at source location
        (source_dir / "report.pdf").write_text("different file")

        tool = UndoOrganizationTool()
        params = UndoOrganizationParams(dry_run=True)

        result = await tool.execute(params)

        assert result.success

        # Should report that file cannot be undone
        summary = result.data["summary"]
        assert summary["can_undo"] == 0
        assert summary["cannot_undo"] == 1

        cannot_undo = result.data["cannot_undo"]
        assert len(cannot_undo) == 1
        assert "already occupied" in cannot_undo[0]["reason"].lower()

    @pytest.mark.asyncio
    async def test_undo_specific_log_file(self, temp_logs_dir, organized_directory):
        """Test undoing a specific log file."""
        source_dir = organized_directory.parent

        # Create multiple logs
        log1_ops = [
            {
                "source": str(source_dir / "report.pdf"),
                "destination": str(organized_directory / "Documents" / "report.pdf"),
                "action": "move",
                "status": "completed",
            }
        ]

        log1_path = save_organization_log(
            source_path=str(source_dir),
            destination_path=str(organized_directory),
            scheme="category",
            operations=log1_ops,
        )

        log2_ops = [
            {
                "source": str(source_dir / "photo.jpg"),
                "destination": str(organized_directory / "Images" / "photo.jpg"),
                "action": "move",
                "status": "completed",
            }
        ]

        save_organization_log(
            source_path=str(source_dir),
            destination_path=str(organized_directory),
            scheme="category",
            operations=log2_ops,
        )

        # Undo specific log (log1)
        tool = UndoOrganizationTool()
        params = UndoOrganizationParams(
            log_file=str(log1_path),
            dry_run=True,
        )

        result = await tool.execute(params)

        assert result.success

        # Should only undo operations from log1
        assert len(result.data["can_undo"]) == 1
        assert "report.pdf" in result.data["can_undo"][0]["destination"]

    @pytest.mark.asyncio
    async def test_undo_nonexistent_log_file(self, temp_logs_dir):
        """Test undoing a non-existent log file."""
        tool = UndoOrganizationTool()
        params = UndoOrganizationParams(
            log_file="/nonexistent/log.json",
            dry_run=True,
        )

        result = await tool.execute(params)

        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_undo_no_move_operations(self, temp_logs_dir):
        """Test undo when log has no move operations."""
        # Create a log with only skip operations
        operations = [
            {
                "source": "/test/file.pdf",
                "destination": "/test/Documents/file.pdf",
                "action": "skip",
                "status": "skipped",
            },
        ]

        save_organization_log(
            source_path="/test",
            destination_path="/test",
            scheme="category",
            operations=operations,
        )

        tool = UndoOrganizationTool()
        params = UndoOrganizationParams(dry_run=True)

        result = await tool.execute(params)

        assert result.success
        assert "No completed move operations" in result.data["message"]

    @pytest.mark.asyncio
    async def test_confirmation_message(self):
        """Test confirmation messages."""
        tool = UndoOrganizationTool()

        # Dry run, no log file
        params = UndoOrganizationParams(dry_run=True)
        msg = tool.get_confirmation_message(params)
        assert "dry run" in msg.lower()
        assert "most recent" in msg.lower()

        # Actual run, specific log file
        params = UndoOrganizationParams(
            log_file="/path/to/log.json",
            dry_run=False,
        )
        msg = tool.get_confirmation_message(params)
        assert "log.json" in msg
        assert "dry run" not in msg.lower()
