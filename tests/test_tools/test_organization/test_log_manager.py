"""Tests for organization log manager."""

import json
from pathlib import Path

import pytest

from aria.tools.organization.log_manager import (
    get_logs_directory,
    save_organization_log,
    load_organization_log,
    list_organization_logs,
    get_most_recent_log,
    OrganizationLog,
)


@pytest.fixture
def temp_logs_dir(tmp_path, monkeypatch):
    """Create a temporary logs directory."""
    logs_dir = tmp_path / "organization_logs"
    logs_dir.mkdir()

    # Monkeypatch get_logs_directory to use temp directory
    monkeypatch.setattr("aria.tools.organization.log_manager.get_logs_directory", lambda: logs_dir)

    return logs_dir


class TestLogManager:
    """Test log manager functionality."""

    def test_get_logs_directory_creates_if_not_exists(self, tmp_path, monkeypatch):
        """Test that get_logs_directory creates directory if it doesn't exist."""
        test_dir = tmp_path / "test_logs"

        # Monkeypatch to use test directory
        monkeypatch.setattr("aria.tools.organization.log_manager.Path.home", lambda: tmp_path)

        # Directory shouldn't exist yet
        assert not (tmp_path / ".aria" / "organization_logs").exists()

        # Call get_logs_directory
        from aria.tools.organization.log_manager import get_logs_directory

        logs_dir = get_logs_directory()

        # Directory should now exist
        assert logs_dir.exists()
        assert logs_dir.is_dir()

    def test_save_organization_log(self, temp_logs_dir):
        """Test saving an organization log."""
        operations = [
            {
                "source": "/test/file1.pdf",
                "destination": "/test/Documents/file1.pdf",
                "action": "move",
                "status": "completed",
            },
            {
                "source": "/test/file2.jpg",
                "destination": "/test/Images/file2.jpg",
                "action": "move",
                "status": "completed",
            },
        ]

        log_path = save_organization_log(
            source_path="/test",
            destination_path="/test",
            scheme="category",
            operations=operations,
        )

        # Verify log file was created
        assert log_path.exists()
        assert log_path.suffix == ".json"
        assert "organize" in log_path.name

        # Verify log content
        with open(log_path) as f:
            data = json.load(f)

        assert data["source_path"] == "/test"
        assert data["destination_path"] == "/test"
        assert data["scheme"] == "category"
        assert data["total_operations"] == 2
        assert data["completed_operations"] == 2
        assert data["failed_operations"] == 0
        assert len(data["operations"]) == 2

    def test_load_organization_log(self, temp_logs_dir):
        """Test loading an organization log."""
        # Create a log
        operations = [
            {
                "source": "/test/file.pdf",
                "destination": "/test/Documents/file.pdf",
                "action": "move",
                "status": "completed",
            }
        ]

        log_path = save_organization_log(
            source_path="/test",
            destination_path="/test/organized",
            scheme="date",
            operations=operations,
        )

        # Load the log
        log = load_organization_log(log_path)

        assert isinstance(log, OrganizationLog)
        assert log.source_path == "/test"
        assert log.destination_path == "/test/organized"
        assert log.scheme == "date"
        assert log.total_operations == 1
        assert log.completed_operations == 1
        assert len(log.operations) == 1

    def test_load_organization_log_file_not_found(self, temp_logs_dir):
        """Test loading a non-existent log file."""
        with pytest.raises(FileNotFoundError):
            load_organization_log(temp_logs_dir / "nonexistent.json")

    def test_load_organization_log_invalid_file(self, temp_logs_dir):
        """Test loading an invalid log file."""
        invalid_log = temp_logs_dir / "invalid.json"
        invalid_log.write_text("not valid json")

        with pytest.raises(ValueError):
            load_organization_log(invalid_log)

    def test_list_organization_logs(self, temp_logs_dir):
        """Test listing organization logs."""
        # Create multiple logs
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
                source_path="/test",
                destination_path="/test",
                scheme="category",
                operations=operations,
            )

        # List logs
        logs = list_organization_logs()

        assert len(logs) == 3
        assert all(isinstance(log, OrganizationLog) for log in logs)

        # Logs should be sorted by timestamp (newest first)
        timestamps = [log.timestamp for log in logs]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_organization_logs_empty(self, temp_logs_dir):
        """Test listing logs when directory is empty."""
        logs = list_organization_logs()
        assert logs == []

    def test_list_organization_logs_skips_invalid(self, temp_logs_dir):
        """Test that invalid logs are skipped."""
        # Create a valid log
        operations = [
            {
                "source": "/test/file.txt",
                "destination": "/test/Documents/file.txt",
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

        # Create an invalid log file
        invalid_log = temp_logs_dir / "2024-01-01_organize.json"
        invalid_log.write_text("invalid json")

        # List logs - should return only the valid one
        logs = list_organization_logs()
        assert len(logs) == 1

    def test_get_most_recent_log(self, temp_logs_dir):
        """Test getting the most recent log."""
        # Create multiple logs
        log_paths = []
        for i in range(3):
            operations = [
                {
                    "source": f"/test/file{i}.txt",
                    "destination": f"/test/Documents/file{i}.txt",
                    "action": "move",
                    "status": "completed",
                }
            ]

            log_path = save_organization_log(
                source_path=f"/test{i}",
                destination_path=f"/test{i}",
                scheme="category",
                operations=operations,
            )
            log_paths.append(log_path)

        # Get most recent
        recent = get_most_recent_log()

        assert recent is not None
        assert isinstance(recent, OrganizationLog)

        # Should be the last created log
        all_logs = list_organization_logs()
        assert recent.timestamp == all_logs[0].timestamp

    def test_get_most_recent_log_empty(self, temp_logs_dir):
        """Test getting most recent log when none exist."""
        recent = get_most_recent_log()
        assert recent is None

    def test_save_log_with_errors(self, temp_logs_dir):
        """Test saving a log with failed operations."""
        operations = [
            {
                "source": "/test/file1.pdf",
                "destination": "/test/Documents/file1.pdf",
                "action": "move",
                "status": "completed",
            },
            {
                "source": "/test/file2.pdf",
                "destination": "/test/Documents/file2.pdf",
                "action": "move",
                "status": "error",
            },
            {
                "source": "/test/file3.pdf",
                "destination": "/test/Documents/file3.pdf",
                "action": "skip",
                "status": "skipped",
            },
        ]

        log_path = save_organization_log(
            source_path="/test",
            destination_path="/test",
            scheme="category",
            operations=operations,
        )

        # Load and verify
        log = load_organization_log(log_path)

        assert log.total_operations == 3
        assert log.completed_operations == 1
        assert log.failed_operations == 1
