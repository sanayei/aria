"""Log management utilities for organization operations."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OrganizationLog(BaseModel):
    """Represents a single organization operation log."""

    timestamp: str
    source_path: str
    destination_path: str
    scheme: str
    operations: List[Dict[str, Any]]
    total_operations: int
    completed_operations: int
    failed_operations: int


def get_logs_directory() -> Path:
    """Get the directory where organization logs are stored.

    Creates the directory if it doesn't exist.

    Returns:
        Path: The logs directory path
    """
    logs_dir = Path.home() / ".aria" / "organization_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def save_organization_log(
    source_path: str,
    destination_path: str,
    scheme: str,
    operations: List[Dict[str, Any]],
) -> Path:
    """Save an organization operation log to disk.

    Args:
        source_path: Source directory that was organized
        destination_path: Destination directory
        scheme: Organization scheme used
        operations: List of file operations performed

    Returns:
        Path: Path to the saved log file
    """
    timestamp = datetime.now().isoformat()

    # Count operation statuses
    completed = sum(1 for op in operations if op.get("status") == "completed")
    failed = sum(1 for op in operations if op.get("status") == "error")

    log_data = {
        "timestamp": timestamp,
        "source_path": source_path,
        "destination_path": destination_path,
        "scheme": scheme,
        "operations": operations,
        "total_operations": len(operations),
        "completed_operations": completed,
        "failed_operations": failed,
    }

    # Generate filename from timestamp
    safe_timestamp = timestamp.replace(":", "-").replace(".", "-")
    log_filename = f"{safe_timestamp}_organize.json"
    log_path = get_logs_directory() / log_filename

    # Save to file
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    logger.info(f"Saved organization log to: {log_path}")
    return log_path


def load_organization_log(log_path: Path) -> OrganizationLog:
    """Load an organization log from disk.

    Args:
        log_path: Path to the log file

    Returns:
        OrganizationLog: The loaded log

    Raises:
        FileNotFoundError: If log file doesn't exist
        ValueError: If log file is invalid
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    try:
        with open(log_path, "r") as f:
            data = json.load(f)

        return OrganizationLog(**data)
    except Exception as e:
        raise ValueError(f"Invalid log file: {e}")


def list_organization_logs() -> List[OrganizationLog]:
    """List all organization logs.

    Returns:
        List[OrganizationLog]: List of logs sorted by timestamp (newest first)
    """
    logs_dir = get_logs_directory()
    log_files = sorted(logs_dir.glob("*_organize.json"), reverse=True)

    logs = []
    for log_file in log_files:
        try:
            log = load_organization_log(log_file)
            logs.append(log)
        except Exception as e:
            logger.warning(f"Failed to load log {log_file}: {e}")
            continue

    return logs


def get_most_recent_log() -> OrganizationLog | None:
    """Get the most recent organization log.

    Returns:
        OrganizationLog | None: The most recent log, or None if no logs exist
    """
    logs = list_organization_logs()
    return logs[0] if logs else None
