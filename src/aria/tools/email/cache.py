"""Email metadata cache for ARIA.

This module provides SQLite-based caching for email metadata to reduce
Gmail API calls and improve performance.
"""

import json
from datetime import datetime, UTC, timedelta
from pathlib import Path

import aiosqlite

from aria.logging import get_logger
from aria.tools.email.models import EmailSummary

logger = get_logger("aria.tools.email.cache")


class EmailCacheError(Exception):
    """Exception raised when email cache operations fail."""
    pass


class EmailCache:
    """SQLite-based cache for email metadata.

    This cache stores EmailSummary objects to reduce API calls when
    repeatedly accessing the same emails.
    """

    def __init__(self, db_path: Path, ttl_seconds: int = 300):
        """Initialize email cache.

        Args:
            db_path: Path to SQLite database file
            ttl_seconds: Cache TTL in seconds (default: 5 minutes)
        """
        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_seconds

        # Create database directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize the cache database and create tables."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")

                # Create email_cache table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS email_cache (
                        email_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        subject TEXT NOT NULL,
                        sender TEXT NOT NULL,
                        snippet TEXT NOT NULL,
                        date TEXT NOT NULL,
                        labels TEXT NOT NULL,
                        is_unread INTEGER NOT NULL,
                        has_attachments INTEGER NOT NULL,
                        cached_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL
                    )
                """)

                # Create index on expiration for cleanup
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_email_cache_expires
                    ON email_cache(expires_at)
                """)

                # Create index on thread_id for thread-based queries
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_email_cache_thread
                    ON email_cache(thread_id)
                """)

                await db.commit()
                logger.debug(f"Email cache initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize email cache: {e}")
            raise EmailCacheError(f"Failed to initialize cache: {e}") from e

    async def get(self, email_id: str) -> EmailSummary | None:
        """Get cached email summary.

        Args:
            email_id: Gmail message ID

        Returns:
            EmailSummary | None: Cached email summary or None if not found/expired
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                async with db.execute(
                    """
                    SELECT * FROM email_cache
                    WHERE email_id = ? AND expires_at > ?
                    """,
                    (email_id, datetime.now(UTC).isoformat())
                ) as cursor:
                    row = await cursor.fetchone()

                    if row is None:
                        return None

                    return self._row_to_summary(row)

        except Exception as e:
            logger.warning(f"Failed to get cached email {email_id}: {e}")
            return None

    async def put(self, email: EmailSummary) -> None:
        """Cache an email summary.

        Args:
            email: Email summary to cache
        """
        try:
            now = datetime.now(UTC)
            expires_at = now + timedelta(seconds=self.ttl_seconds)

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO email_cache
                    (email_id, thread_id, subject, sender, snippet, date,
                     labels, is_unread, has_attachments, cached_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    email.id,
                    email.thread_id,
                    email.subject,
                    email.sender,
                    email.snippet,
                    email.date.isoformat(),
                    json.dumps(email.labels),
                    1 if email.is_unread else 0,
                    1 if email.has_attachments else 0,
                    now.isoformat(),
                    expires_at.isoformat(),
                ))
                await db.commit()

            logger.debug(f"Cached email {email.id}")

        except Exception as e:
            logger.warning(f"Failed to cache email {email.id}: {e}")

    async def put_many(self, emails: list[EmailSummary]) -> None:
        """Cache multiple email summaries.

        Args:
            emails: List of email summaries to cache
        """
        if not emails:
            return

        try:
            now = datetime.now(UTC)
            expires_at = now + timedelta(seconds=self.ttl_seconds)

            async with aiosqlite.connect(self.db_path) as db:
                rows = [
                    (
                        email.id,
                        email.thread_id,
                        email.subject,
                        email.sender,
                        email.snippet,
                        email.date.isoformat(),
                        json.dumps(email.labels),
                        1 if email.is_unread else 0,
                        1 if email.has_attachments else 0,
                        now.isoformat(),
                        expires_at.isoformat(),
                    )
                    for email in emails
                ]

                await db.executemany("""
                    INSERT OR REPLACE INTO email_cache
                    (email_id, thread_id, subject, sender, snippet, date,
                     labels, is_unread, has_attachments, cached_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows)
                await db.commit()

            logger.debug(f"Cached {len(emails)} emails")

        except Exception as e:
            logger.warning(f"Failed to cache emails: {e}")

    async def invalidate(self, email_id: str) -> None:
        """Invalidate a cached email.

        Args:
            email_id: Gmail message ID to invalidate
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "DELETE FROM email_cache WHERE email_id = ?",
                    (email_id,)
                )
                await db.commit()

            logger.debug(f"Invalidated cache for email {email_id}")

        except Exception as e:
            logger.warning(f"Failed to invalidate cache for {email_id}: {e}")

    async def invalidate_all(self) -> None:
        """Invalidate all cached emails."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM email_cache")
                await db.commit()

            logger.info("Invalidated entire email cache")

        except Exception as e:
            logger.warning(f"Failed to invalidate cache: {e}")

    async def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            int: Number of entries removed
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM email_cache WHERE expires_at <= ?",
                    (datetime.now(UTC).isoformat(),)
                )
                await db.commit()
                count = cursor.rowcount

            if count > 0:
                logger.debug(f"Cleaned up {count} expired cache entries")

            return count

        except Exception as e:
            logger.warning(f"Failed to cleanup expired entries: {e}")
            return 0

    async def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            dict: Cache statistics (total, expired, valid)
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Total entries
                async with db.execute("SELECT COUNT(*) FROM email_cache") as cursor:
                    row = await cursor.fetchone()
                    total = row[0] if row else 0

                # Expired entries
                async with db.execute(
                    "SELECT COUNT(*) FROM email_cache WHERE expires_at <= ?",
                    (datetime.now(UTC).isoformat(),)
                ) as cursor:
                    row = await cursor.fetchone()
                    expired = row[0] if row else 0

            return {
                "total": total,
                "expired": expired,
                "valid": total - expired,
            }

        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {"total": 0, "expired": 0, "valid": 0}

    async def close(self) -> None:
        """Close the cache (cleanup if needed)."""
        # No persistent connections to close, but we can cleanup expired entries
        await self.cleanup_expired()

    def _row_to_summary(self, row: aiosqlite.Row) -> EmailSummary:
        """Convert database row to EmailSummary.

        Args:
            row: Database row

        Returns:
            EmailSummary: Email summary object
        """
        return EmailSummary(
            id=row["email_id"],
            thread_id=row["thread_id"],
            subject=row["subject"],
            sender=row["sender"],
            snippet=row["snippet"],
            date=datetime.fromisoformat(row["date"]),
            labels=json.loads(row["labels"]),
            is_unread=bool(row["is_unread"]),
            has_attachments=bool(row["has_attachments"]),
        )
