"""Data models for email representation in ARIA.

This module defines Pydantic models for emails, attachments, and related data
structures used throughout the email tools.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AttachmentInfo(BaseModel):
    """Information about an email attachment.

    This is metadata only - actual attachment data must be fetched separately.
    """

    filename: str = Field(description="Name of the attachment file")
    mime_type: str = Field(description="MIME type of the attachment")
    size_bytes: int = Field(description="Size of attachment in bytes", ge=0)
    attachment_id: str = Field(description="Gmail attachment ID for fetching")


class EmailSummary(BaseModel):
    """Brief email representation for lists and search results.

    This contains the minimal information needed to display emails in a list,
    without fetching the full email body.
    """

    id: str = Field(description="Gmail message ID")
    thread_id: str = Field(description="Gmail thread ID")
    subject: str = Field(description="Email subject line")
    sender: str = Field(description="Sender email address (or name <email>)")
    snippet: str = Field(description="Short preview of email content (~200 chars)")
    date: datetime = Field(description="Email sent/received date")
    labels: list[str] = Field(
        default_factory=list, description="Gmail labels (INBOX, UNREAD, etc.)"
    )
    is_unread: bool = Field(default=False, description="Whether email is unread")
    has_attachments: bool = Field(default=False, description="Whether email has attachments")

    def __str__(self) -> str:
        """Human-readable string representation."""
        unread_marker = "●" if self.is_unread else " "
        attachment_marker = "📎" if self.has_attachments else " "
        return f"{unread_marker}{attachment_marker} {self.date.strftime('%Y-%m-%d %H:%M')} | {self.sender:30.30} | {self.subject}"


class EmailDetail(BaseModel):
    """Full email with complete body content and metadata.

    This is the complete email representation including body, recipients,
    and all headers. Use for displaying or analyzing individual emails.
    """

    # Core identification (same as EmailSummary)
    id: str = Field(description="Gmail message ID")
    thread_id: str = Field(description="Gmail thread ID")

    # Headers
    subject: str = Field(description="Email subject line")
    sender: str = Field(description="Sender email address (or name <email>)")
    to: list[str] = Field(default_factory=list, description="List of recipient email addresses")
    cc: list[str] = Field(default_factory=list, description="List of CC recipient email addresses")
    bcc: list[str] = Field(
        default_factory=list, description="List of BCC recipient email addresses (if available)"
    )
    reply_to: str | None = Field(
        default=None, description="Reply-To address if different from sender"
    )

    # Content
    body_text: str | None = Field(default=None, description="Plain text body content")
    body_html: str | None = Field(default=None, description="HTML body content")
    snippet: str = Field(description="Short preview of email content")

    # Metadata
    date: datetime = Field(description="Email sent/received date")
    labels: list[str] = Field(
        default_factory=list, description="Gmail labels (INBOX, UNREAD, etc.)"
    )
    is_unread: bool = Field(default=False, description="Whether email is unread")

    # Attachments
    attachments: list[AttachmentInfo] = Field(
        default_factory=list, description="List of attachment metadata"
    )

    # Additional metadata
    internal_date: datetime | None = Field(
        default=None, description="Gmail internal date (when received by Gmail servers)"
    )
    size_estimate: int | None = Field(default=None, description="Estimated size in bytes", ge=0)

    @property
    def has_attachments(self) -> bool:
        """Check if email has attachments."""
        return len(self.attachments) > 0

    def get_display_body(self, prefer_html: bool = False) -> str:
        """Get the best available body content.

        Args:
            prefer_html: If True, prefer HTML body over plain text

        Returns:
            str: Email body content (HTML or plain text)
        """
        if prefer_html and self.body_html:
            return self.body_html
        if self.body_text:
            return self.body_text
        if self.body_html:
            return self.body_html
        return self.snippet


class EmailThread(BaseModel):
    """A Gmail conversation thread containing multiple emails.

    Gmail groups related emails into threads. This model represents
    a complete thread with all its messages.
    """

    thread_id: str = Field(description="Gmail thread ID")
    messages: list[EmailDetail] = Field(description="List of emails in the thread, ordered by date")

    @property
    def subject(self) -> str:
        """Get the thread subject (from first message)."""
        return self.messages[0].subject if self.messages else ""

    @property
    def message_count(self) -> int:
        """Number of messages in thread."""
        return len(self.messages)

    @property
    def participants(self) -> list[str]:
        """Get unique list of all participants in thread."""
        participants = set()
        for msg in self.messages:
            participants.add(msg.sender)
            participants.update(msg.to)
            participants.update(msg.cc)
        return sorted(participants)


class GmailQuery(BaseModel):
    """Gmail search query parameters.

    This model represents a Gmail search query, supporting Gmail's
    search operators and syntax.

    Examples:
        - "from:john@example.com"
        - "subject:invoice after:2024/01/01"
        - "has:attachment is:unread"
    """

    query: str = Field(description="Gmail search query string")
    max_results: int = Field(
        default=10, description="Maximum number of results to return", ge=1, le=500
    )
    include_spam_trash: bool = Field(
        default=False, description="Include emails from spam and trash"
    )

    def to_gmail_params(self) -> dict[str, Any]:
        """Convert to Gmail API query parameters.

        Returns:
            dict: Parameters suitable for Gmail API calls
        """
        params: dict[str, Any] = {
            "q": self.query,
            "maxResults": self.max_results,
        }
        if self.include_spam_trash:
            params["includeSpamTrash"] = True
        return params


class EmailStats(BaseModel):
    """Statistics about a user's Gmail account or query results.

    Useful for summarizing email counts, unread messages, etc.
    """

    total_messages: int = Field(default=0, ge=0)
    unread_messages: int = Field(default=0, ge=0)
    threads: int = Field(default=0, ge=0)
    labels: dict[str, int] = Field(default_factory=dict, description="Count of messages per label")
    size_estimate: int = Field(default=0, ge=0, description="Total estimated size in bytes")
