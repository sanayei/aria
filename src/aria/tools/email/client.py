"""Gmail API client wrapper for ARIA.

This module provides a high-level interface to the Gmail API, handling
message parsing, pagination, rate limiting, and data conversion.
"""

import base64
import email
from datetime import datetime, UTC
from email.mime.text import MIMEText
from typing import Any

from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError

from aria.logging import get_logger
from aria.tools.email.auth import GmailAuth, GmailAuthError
from aria.tools.email.models import (
    EmailSummary,
    EmailDetail,
    AttachmentInfo,
    EmailThread,
    GmailQuery,
)

logger = get_logger("aria.tools.email.client")


class GmailClientError(Exception):
    """Exception raised when Gmail client operations fail."""
    pass


class GmailClient:
    """High-level Gmail API client.

    This client wraps the Gmail API and provides convenient methods for
    reading, searching, and managing emails. It handles authentication,
    MIME parsing, and data conversion to Pydantic models.
    """

    def __init__(self, gmail_auth: GmailAuth):
        """Initialize Gmail client.

        Args:
            gmail_auth: GmailAuth instance for authentication
        """
        self.gmail_auth = gmail_auth
        self._service: Resource | None = None

    async def _get_service(self) -> Resource:
        """Get authenticated Gmail service, creating if needed.

        Returns:
            Resource: Authenticated Gmail API service

        Raises:
            GmailClientError: If authentication fails
        """
        if self._service is None:
            try:
                self._service = await self.gmail_auth.get_service()
            except GmailAuthError as e:
                raise GmailClientError(f"Failed to authenticate with Gmail: {e}") from e
        return self._service

    async def list_messages(
        self,
        label_ids: list[str] | None = None,
        max_results: int = 10,
        include_spam_trash: bool = False,
    ) -> list[EmailSummary]:
        """List messages from Gmail.

        Args:
            label_ids: List of label IDs to filter by (e.g., ["INBOX", "UNREAD"])
            max_results: Maximum number of messages to return
            include_spam_trash: Include messages from spam and trash

        Returns:
            list[EmailSummary]: List of email summaries

        Raises:
            GmailClientError: If listing fails
        """
        try:
            service = await self._get_service()

            # Build query parameters
            query_params: dict[str, Any] = {
                "userId": "me",
                "maxResults": max_results,
            }
            if label_ids:
                query_params["labelIds"] = label_ids
            if include_spam_trash:
                query_params["includeSpamTrash"] = True

            # Get message list
            result = service.users().messages().list(**query_params).execute()
            messages = result.get("messages", [])

            if not messages:
                return []

            # Fetch summaries for each message
            summaries = []
            for msg in messages:
                try:
                    summary = await self._get_message_summary(msg["id"])
                    summaries.append(summary)
                except Exception as e:
                    logger.warning(f"Failed to fetch message {msg['id']}: {e}")
                    continue

            return summaries

        except HttpError as e:
            logger.error(f"Gmail API error in list_messages: {e}")
            raise GmailClientError(f"Failed to list messages: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in list_messages: {e}")
            raise GmailClientError(f"Failed to list messages: {e}") from e

    async def search_messages(self, query: GmailQuery) -> list[EmailSummary]:
        """Search messages using Gmail query syntax.

        Args:
            query: Gmail search query

        Returns:
            list[EmailSummary]: List of matching email summaries

        Raises:
            GmailClientError: If search fails
        """
        try:
            service = await self._get_service()

            # Execute search
            query_params = query.to_gmail_params()
            query_params["userId"] = "me"

            result = service.users().messages().list(**query_params).execute()
            messages = result.get("messages", [])

            if not messages:
                return []

            # Fetch summaries for each message
            summaries = []
            for msg in messages:
                try:
                    summary = await self._get_message_summary(msg["id"])
                    summaries.append(summary)
                except Exception as e:
                    logger.warning(f"Failed to fetch message {msg['id']}: {e}")
                    continue

            return summaries

        except HttpError as e:
            logger.error(f"Gmail API error in search_messages: {e}")
            raise GmailClientError(f"Failed to search messages: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in search_messages: {e}")
            raise GmailClientError(f"Failed to search messages: {e}") from e

    async def get_message(self, message_id: str) -> EmailDetail:
        """Get full message details.

        Args:
            message_id: Gmail message ID

        Returns:
            EmailDetail: Full email details

        Raises:
            GmailClientError: If fetching fails
        """
        try:
            service = await self._get_service()

            # Get full message with metadata
            msg = service.users().messages().get(
                userId="me",
                id=message_id,
                format="full"
            ).execute()

            return self._parse_message_detail(msg)

        except HttpError as e:
            logger.error(f"Gmail API error in get_message: {e}")
            raise GmailClientError(f"Failed to get message {message_id}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in get_message: {e}")
            raise GmailClientError(f"Failed to get message {message_id}: {e}") from e

    async def get_thread(self, thread_id: str) -> EmailThread:
        """Get full thread with all messages.

        Args:
            thread_id: Gmail thread ID

        Returns:
            EmailThread: Complete thread with all messages

        Raises:
            GmailClientError: If fetching fails
        """
        try:
            service = await self._get_service()

            # Get thread
            thread = service.users().threads().get(
                userId="me",
                id=thread_id,
                format="full"
            ).execute()

            messages = []
            for msg_data in thread.get("messages", []):
                msg_detail = self._parse_message_detail(msg_data)
                messages.append(msg_detail)

            # Sort by date
            messages.sort(key=lambda m: m.date)

            return EmailThread(thread_id=thread_id, messages=messages)

        except HttpError as e:
            logger.error(f"Gmail API error in get_thread: {e}")
            raise GmailClientError(f"Failed to get thread {thread_id}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in get_thread: {e}")
            raise GmailClientError(f"Failed to get thread {thread_id}: {e}") from e

    async def mark_as_read(self, message_id: str) -> bool:
        """Mark a message as read.

        Args:
            message_id: Gmail message ID

        Returns:
            bool: True if successful

        Raises:
            GmailClientError: If operation fails
        """
        try:
            service = await self._get_service()

            service.users().messages().modify(
                userId="me",
                id=message_id,
                body={"removeLabelIds": ["UNREAD"]}
            ).execute()

            logger.info(f"Marked message {message_id} as read")
            return True

        except HttpError as e:
            logger.error(f"Gmail API error in mark_as_read: {e}")
            raise GmailClientError(f"Failed to mark message as read: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in mark_as_read: {e}")
            raise GmailClientError(f"Failed to mark message as read: {e}") from e

    async def _get_message_summary(self, message_id: str) -> EmailSummary:
        """Get message summary (metadata only, no body).

        Args:
            message_id: Gmail message ID

        Returns:
            EmailSummary: Email summary

        Raises:
            GmailClientError: If fetching fails
        """
        service = await self._get_service()

        # Get message with metadata format (faster than full)
        msg = service.users().messages().get(
            userId="me",
            id=message_id,
            format="metadata",
            metadataHeaders=["From", "Subject", "Date"]
        ).execute()

        return self._parse_message_summary(msg)

    def _parse_message_summary(self, msg_data: dict[str, Any]) -> EmailSummary:
        """Parse Gmail API message data into EmailSummary.

        Args:
            msg_data: Raw message data from Gmail API

        Returns:
            EmailSummary: Parsed email summary
        """
        headers = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}

        # Parse date
        date_str = headers.get("Date", "")
        date = self._parse_date(date_str) if date_str else datetime.now(UTC)

        # Get labels
        labels = msg_data.get("labelIds", [])

        # Check for attachments
        has_attachments = self._check_has_attachments(msg_data.get("payload", {}))

        return EmailSummary(
            id=msg_data["id"],
            thread_id=msg_data["threadId"],
            subject=headers.get("Subject", "(No Subject)"),
            sender=headers.get("From", "Unknown"),
            snippet=msg_data.get("snippet", ""),
            date=date,
            labels=labels,
            is_unread="UNREAD" in labels,
            has_attachments=has_attachments,
        )

    def _parse_message_detail(self, msg_data: dict[str, Any]) -> EmailDetail:
        """Parse Gmail API message data into EmailDetail.

        Args:
            msg_data: Raw message data from Gmail API

        Returns:
            EmailDetail: Parsed email details
        """
        headers = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}

        # Parse date
        date_str = headers.get("Date", "")
        date = self._parse_date(date_str) if date_str else datetime.now(UTC)

        # Parse internal date
        internal_date = None
        if "internalDate" in msg_data:
            internal_date = datetime.fromtimestamp(
                int(msg_data["internalDate"]) / 1000,
                tz=UTC
            )

        # Get labels
        labels = msg_data.get("labelIds", [])

        # Parse recipients
        to_list = self._parse_address_list(headers.get("To", ""))
        cc_list = self._parse_address_list(headers.get("Cc", ""))
        bcc_list = self._parse_address_list(headers.get("Bcc", ""))

        # Extract body content
        payload = msg_data.get("payload", {})
        body_text, body_html = self._extract_body(payload)

        # Extract attachments
        attachments = self._extract_attachments(payload)

        return EmailDetail(
            id=msg_data["id"],
            thread_id=msg_data["threadId"],
            subject=headers.get("Subject", "(No Subject)"),
            sender=headers.get("From", "Unknown"),
            to=to_list,
            cc=cc_list,
            bcc=bcc_list,
            reply_to=headers.get("Reply-To"),
            body_text=body_text,
            body_html=body_html,
            snippet=msg_data.get("snippet", ""),
            date=date,
            labels=labels,
            is_unread="UNREAD" in labels,
            attachments=attachments,
            internal_date=internal_date,
            size_estimate=msg_data.get("sizeEstimate"),
        )

    def _extract_body(self, payload: dict[str, Any]) -> tuple[str | None, str | None]:
        """Extract plain text and HTML body from message payload.

        Args:
            payload: Message payload from Gmail API

        Returns:
            tuple: (plain_text, html) - either may be None
        """
        body_text = None
        body_html = None

        def extract_parts(part: dict[str, Any]) -> None:
            """Recursively extract body parts."""
            nonlocal body_text, body_html

            mime_type = part.get("mimeType", "")

            # Handle multipart
            if mime_type.startswith("multipart/"):
                for subpart in part.get("parts", []):
                    extract_parts(subpart)
                return

            # Extract body data
            body_data = part.get("body", {}).get("data")
            if not body_data:
                return

            # Decode base64
            try:
                decoded = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"Failed to decode body part: {e}")
                return

            # Store based on MIME type
            if mime_type == "text/plain" and body_text is None:
                body_text = decoded
            elif mime_type == "text/html" and body_html is None:
                body_html = decoded

        # Start extraction
        extract_parts(payload)

        return body_text, body_html

    def _extract_attachments(self, payload: dict[str, Any]) -> list[AttachmentInfo]:
        """Extract attachment metadata from message payload.

        Args:
            payload: Message payload from Gmail API

        Returns:
            list[AttachmentInfo]: List of attachment information
        """
        attachments = []

        def extract_parts(part: dict[str, Any]) -> None:
            """Recursively extract attachment parts."""
            # Check if this part has a filename (indicates attachment)
            filename = part.get("filename")
            if filename and filename.strip():
                body = part.get("body", {})
                attachment_id = body.get("attachmentId")
                size = body.get("size", 0)
                mime_type = part.get("mimeType", "application/octet-stream")

                if attachment_id:
                    attachments.append(AttachmentInfo(
                        filename=filename,
                        mime_type=mime_type,
                        size_bytes=size,
                        attachment_id=attachment_id,
                    ))

            # Recurse into multipart
            if part.get("mimeType", "").startswith("multipart/"):
                for subpart in part.get("parts", []):
                    extract_parts(subpart)

        extract_parts(payload)
        return attachments

    def _check_has_attachments(self, payload: dict[str, Any]) -> bool:
        """Check if payload has attachments (quick check).

        Args:
            payload: Message payload from Gmail API

        Returns:
            bool: True if message has attachments
        """
        def check_parts(part: dict[str, Any]) -> bool:
            """Recursively check for attachments."""
            # Check if this part has a filename
            if part.get("filename"):
                return True

            # Check subparts
            if part.get("mimeType", "").startswith("multipart/"):
                for subpart in part.get("parts", []):
                    if check_parts(subpart):
                        return True

            return False

        return check_parts(payload)

    def _parse_address_list(self, address_str: str) -> list[str]:
        """Parse comma-separated email addresses.

        Args:
            address_str: Comma-separated email addresses

        Returns:
            list[str]: List of email addresses
        """
        if not address_str:
            return []

        # Split by comma and clean up
        addresses = []
        for addr in address_str.split(","):
            addr = addr.strip()
            if addr:
                addresses.append(addr)

        return addresses

    def _parse_date(self, date_str: str) -> datetime:
        """Parse email date header.

        Args:
            date_str: Date string from email header

        Returns:
            datetime: Parsed datetime (UTC)
        """
        try:
            # Parse using email.utils
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            # Convert to UTC if timezone-aware
            if dt.tzinfo is not None:
                dt = dt.astimezone(UTC)
            else:
                # Assume UTC if no timezone
                dt = dt.replace(tzinfo=UTC)
            return dt
        except Exception as e:
            logger.warning(f"Failed to parse date '{date_str}': {e}")
            return datetime.now(UTC)
