"""Gmail email tools for ARIA agent.

This module provides tools for listing, searching, and reading emails from Gmail.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from aria.config import get_settings
from aria.logging import get_logger
from aria.tools.base import BaseTool, ToolResult, RiskLevel
from aria.tools.email.auth import GmailAuth
from aria.tools.email.client import GmailClient
from aria.tools.email.cache import EmailCache
from aria.tools.email.models import GmailQuery

logger = get_logger("aria.tools.email.tools")


class ListEmailsParams(BaseModel):
    """Input for ListEmailsTool."""

    folder: str = Field(
        default="INBOX",
        description="Gmail label/folder to list emails from (e.g., INBOX, SENT, SPAM)",
    )
    max_results: int = Field(
        default=10, ge=1, le=50, description="Maximum number of emails to return"
    )
    unread_only: bool = Field(default=False, description="If True, only return unread emails")


class ListEmailsTool(BaseTool[ListEmailsParams]):
    """List recent emails from Gmail.

    This tool fetches a list of recent emails from a specified folder/label.
    Results are cached to reduce API calls.
    """

    name = "list_emails"
    description = (
        "List recent emails from Gmail inbox or other folders. "
        "Returns email summaries with subject, sender, date, and preview. "
        "Use this to see what emails the user has received. "
        "This tool is SUFFICIENT for showing the user their emails - no additional tools needed."
    )
    risk_level = RiskLevel.LOW
    parameters_schema = ListEmailsParams

    def __init__(self):
        """Initialize the list emails tool."""
        super().__init__()
        self._gmail_client: GmailClient | None = None
        self._email_cache: EmailCache | None = None

    async def _get_gmail_client(self) -> GmailClient:
        """Get or create Gmail client."""
        if self._gmail_client is None:
            settings = get_settings()
            gmail_auth = GmailAuth(settings.gmail_credentials_dir)
            self._gmail_client = GmailClient(gmail_auth)
        return self._gmail_client

    async def _get_email_cache(self) -> EmailCache:
        """Get or create email cache."""
        if self._email_cache is None:
            settings = get_settings()
            cache_path = settings.aria_data_dir / "cache" / "email_cache.db"
            self._email_cache = EmailCache(cache_path, ttl_seconds=settings.gmail_cache_ttl)
            await self._email_cache.initialize()
        return self._email_cache

    def get_confirmation_message(self, params: ListEmailsParams) -> str:
        """Get confirmation message.

        Args:
            params: Validated parameters

        Returns:
            str: Confirmation message
        """
        unread_text = " (unread only)" if params.unread_only else ""
        return f"List {params.max_results} emails from {params.folder}{unread_text}"

    async def execute(self, params: ListEmailsParams) -> ToolResult:
        """Execute the list emails tool.

        Args:
            params: Validated input parameters

        Returns:
            ToolResult: Result containing list of email summaries
        """
        try:
            gmail_client = await self._get_gmail_client()
            cache = await self._get_email_cache()

            # Build label IDs
            label_ids = [params.folder.upper()]
            if params.unread_only:
                label_ids.append("UNREAD")

            # Fetch emails
            logger.info(f"Listing emails from {params.folder} (max: {params.max_results})")
            emails = await gmail_client.list_messages(
                label_ids=label_ids,
                max_results=params.max_results,
                include_spam_trash=params.folder.upper() in ["SPAM", "TRASH"],
            )

            # Cache results
            await cache.put_many(emails)

            # Format results
            if not emails:
                return ToolResult.success_result(
                    data={
                        "count": 0,
                        "emails": [],
                        "message": f"No emails found in {params.folder}",
                    }
                )

            # Convert to dict for JSON serialization
            email_dicts = [
                {
                    "id": email.id,
                    "subject": email.subject,
                    "sender": email.sender,
                    "date": email.date.isoformat(),
                    "snippet": email.snippet,
                    "is_unread": email.is_unread,
                    "has_attachments": email.has_attachments,
                }
                for email in emails
            ]

            return ToolResult.success_result(
                data={
                    "count": len(emails),
                    "emails": email_dicts,
                    "folder": params.folder,
                }
            )

        except Exception as e:
            logger.error(f"Failed to list emails: {e}")
            return ToolResult.error_result(error=f"Failed to list emails: {e}")


class SearchEmailsParams(BaseModel):
    """Input for SearchEmailsTool."""

    query: str = Field(
        description="Gmail search query (supports Gmail search operators like 'from:', 'subject:', 'after:', etc.)"
    )
    max_results: int = Field(
        default=10, ge=1, le=50, description="Maximum number of emails to return"
    )


class SearchEmailsTool(BaseTool[SearchEmailsParams]):
    """Search emails using Gmail query syntax.

    This tool allows searching emails using Gmail's powerful search operators.
    Examples:
    - "from:john@example.com"
    - "subject:invoice after:2024/01/01"
    - "has:attachment is:unread"
    - "newer_than:2d" (emails from last 2 days)
    """

    name = "search_emails"
    description = (
        "Search emails using Gmail query syntax. "
        "Supports operators like 'from:', 'to:', 'subject:', 'after:', 'before:', "
        "'has:attachment', 'is:unread', 'newer_than:', etc. "
        "Use this to find specific emails based on criteria."
    )
    risk_level = RiskLevel.LOW
    parameters_schema = SearchEmailsParams

    def __init__(self):
        """Initialize the search emails tool."""
        super().__init__()
        self._gmail_client: GmailClient | None = None
        self._email_cache: EmailCache | None = None

    async def _get_gmail_client(self) -> GmailClient:
        """Get or create Gmail client."""
        if self._gmail_client is None:
            settings = get_settings()
            gmail_auth = GmailAuth(settings.gmail_credentials_dir)
            self._gmail_client = GmailClient(gmail_auth)
        return self._gmail_client

    async def _get_email_cache(self) -> EmailCache:
        """Get or create email cache."""
        if self._email_cache is None:
            settings = get_settings()
            cache_path = settings.aria_data_dir / "cache" / "email_cache.db"
            self._email_cache = EmailCache(cache_path, ttl_seconds=settings.gmail_cache_ttl)
            await self._email_cache.initialize()
        return self._email_cache

    def get_confirmation_message(self, params: SearchEmailsParams) -> str:
        """Get confirmation message.

        Args:
            params: Validated parameters

        Returns:
            str: Confirmation message
        """
        return f"Search emails with query: '{params.query}' (max {params.max_results} results)"

    async def execute(self, params: SearchEmailsParams) -> ToolResult:
        """Execute the search emails tool.

        Args:
            params: Validated input parameters

        Returns:
            ToolResult: Result containing matching email summaries
        """
        try:
            gmail_client = await self._get_gmail_client()
            cache = await self._get_email_cache()

            # Create query
            query = GmailQuery(query=params.query, max_results=params.max_results)

            # Search emails
            logger.info(f"Searching emails with query: {params.query}")
            emails = await gmail_client.search_messages(query)

            # Cache results
            await cache.put_many(emails)

            # Format results
            if not emails:
                return ToolResult.success_result(
                    data={
                        "count": 0,
                        "emails": [],
                        "query": params.query,
                        "message": f"No emails found matching query: {params.query}",
                    }
                )

            # Convert to dict for JSON serialization
            email_dicts = [
                {
                    "id": email.id,
                    "subject": email.subject,
                    "sender": email.sender,
                    "date": email.date.isoformat(),
                    "snippet": email.snippet,
                    "is_unread": email.is_unread,
                    "has_attachments": email.has_attachments,
                }
                for email in emails
            ]

            return ToolResult.success_result(
                data={
                    "count": len(emails),
                    "emails": email_dicts,
                    "query": params.query,
                }
            )

        except Exception as e:
            logger.error(f"Failed to search emails: {e}")
            return ToolResult.error_result(error=f"Failed to search emails: {e}")


class ReadEmailParams(BaseModel):
    """Input for ReadEmailTool."""

    email_id: str = Field(
        description="Gmail message ID to read (obtained from list_emails or search_emails)"
    )
    mark_as_read: bool = Field(
        default=False, description="If True, mark the email as read after reading"
    )


class ReadEmailTool(BaseTool[ReadEmailParams]):
    """Read full email content by ID.

    This tool fetches the complete email including full body text,
    all recipients, and attachment information.
    """

    name = "read_email"
    description = (
        "Read the full content of an email by its ID. "
        "Returns complete email with body text, all recipients, and attachments. "
        "Optionally mark the email as read."
    )
    parameters_schema = ReadEmailParams

    def __init__(self):
        """Initialize the read email tool."""
        super().__init__()
        self._gmail_client: GmailClient | None = None

    @property
    def risk_level(self) -> RiskLevel:
        """Dynamic risk level based on whether we're modifying state.

        Returns:
            RiskLevel: LOW if just reading, MEDIUM if marking as read
        """
        # This is a simplified approach - in practice, we'd check the actual input
        # For now, we'll mark it as MEDIUM to be safe since it can modify state
        return RiskLevel.MEDIUM

    async def _get_gmail_client(self) -> GmailClient:
        """Get or create Gmail client."""
        if self._gmail_client is None:
            settings = get_settings()
            gmail_auth = GmailAuth(settings.gmail_credentials_dir)
            self._gmail_client = GmailClient(gmail_auth)
        return self._gmail_client

    def get_confirmation_message(self, params: ReadEmailParams) -> str:
        """Get confirmation message.

        Args:
            params: Validated parameters

        Returns:
            str: Confirmation message
        """
        mark_text = " and mark as read" if params.mark_as_read else ""
        return f"Read email with ID: {params.email_id}{mark_text}"

    async def execute(self, params: ReadEmailParams) -> ToolResult:
        """Execute the read email tool.

        Args:
            params: Validated input parameters

        Returns:
            ToolResult: Result containing full email details
        """
        try:
            gmail_client = await self._get_gmail_client()

            # Fetch email
            logger.info(f"Reading email {params.email_id}")
            email = await gmail_client.get_message(params.email_id)

            # Mark as read if requested
            if params.mark_as_read and email.is_unread:
                logger.info(f"Marking email {params.email_id} as read")
                await gmail_client.mark_as_read(params.email_id)

            # Get body content
            body = email.get_display_body(prefer_html=False)

            # Convert to dict for JSON serialization
            email_dict = {
                "id": email.id,
                "subject": email.subject,
                "sender": email.sender,
                "to": email.to,
                "cc": email.cc,
                "date": email.date.isoformat(),
                "body": body,
                "snippet": email.snippet,
                "is_unread": email.is_unread,
                "labels": email.labels,
                "attachments": [
                    {
                        "filename": att.filename,
                        "mime_type": att.mime_type,
                        "size_bytes": att.size_bytes,
                    }
                    for att in email.attachments
                ],
            }

            return ToolResult.success_result(data=email_dict)

        except Exception as e:
            logger.error(f"Failed to read email: {e}")
            return ToolResult.error_result(error=f"Failed to read email: {e}")


class LabelEmailParams(BaseModel):
    """Input for LabelEmailTool."""

    email_id: str = Field(description="Gmail message ID to label")
    add_labels: list[str] = Field(
        default_factory=list, description="List of label names to add (e.g., ['IMPORTANT', 'Work'])"
    )
    remove_labels: list[str] = Field(
        default_factory=list, description="List of label names to remove"
    )


class LabelEmailTool(BaseTool[LabelEmailParams]):
    """Add or remove labels from an email.

    This tool modifies email labels, allowing organization and categorization.
    Common labels: INBOX, SENT, TRASH, SPAM, UNREAD, IMPORTANT, STARRED.
    """

    name = "label_email"
    description = (
        "Add or remove labels from an email message. "
        "Use this to organize emails by adding labels like IMPORTANT, STARRED, etc. "
        "Can also remove labels. Provide email_id and lists of labels to add/remove."
    )
    risk_level = RiskLevel.MEDIUM
    parameters_schema = LabelEmailParams

    def __init__(self):
        """Initialize the label email tool."""
        super().__init__()
        self._gmail_client: GmailClient | None = None

    async def _get_gmail_client(self) -> GmailClient:
        """Get or create Gmail client."""
        if self._gmail_client is None:
            settings = get_settings()
            gmail_auth = GmailAuth(settings.gmail_credentials_dir)
            self._gmail_client = GmailClient(gmail_auth)
        return self._gmail_client

    def get_confirmation_message(self, params: LabelEmailParams) -> str:
        """Get confirmation message.

        Args:
            params: Validated parameters

        Returns:
            str: Confirmation message
        """
        parts = [f"Modify labels for email {params.email_id}"]
        if params.add_labels:
            parts.append(f"Add: {', '.join(params.add_labels)}")
        if params.remove_labels:
            parts.append(f"Remove: {', '.join(params.remove_labels)}")
        return " | ".join(parts)

    async def execute(self, params: LabelEmailParams) -> ToolResult:
        """Execute the label email tool.

        Args:
            params: Validated input parameters

        Returns:
            ToolResult: Result of labeling operation
        """
        try:
            gmail_client = await self._get_gmail_client()

            # Add labels if specified
            if params.add_labels:
                logger.info(f"Adding labels {params.add_labels} to {params.email_id}")
                await gmail_client.add_labels(params.email_id, params.add_labels)

            # Remove labels if specified
            if params.remove_labels:
                logger.info(f"Removing labels {params.remove_labels} from {params.email_id}")
                await gmail_client.remove_labels(params.email_id, params.remove_labels)

            return ToolResult.success_result(
                data={
                    "email_id": params.email_id,
                    "added_labels": params.add_labels,
                    "removed_labels": params.remove_labels,
                    "message": "Labels updated successfully",
                }
            )

        except Exception as e:
            logger.error(f"Failed to update labels: {e}")
            return ToolResult.error_result(error=f"Failed to update labels: {e}")


class ArchiveEmailParams(BaseModel):
    """Input for ArchiveEmailTool."""

    email_id: str = Field(description="Gmail message ID to archive")


class ArchiveEmailTool(BaseTool[ArchiveEmailParams]):
    """Archive an email (remove from INBOX).

    Archiving removes the INBOX label, moving the email out of the inbox
    while keeping it accessible via search and other labels.
    """

    name = "archive_email"
    description = (
        "Archive an email by removing it from the inbox. "
        "The email will still be accessible via search and other labels. "
        "Use this to clean up the inbox without deleting emails."
    )
    risk_level = RiskLevel.MEDIUM
    parameters_schema = ArchiveEmailParams

    def __init__(self):
        """Initialize the archive email tool."""
        super().__init__()
        self._gmail_client: GmailClient | None = None

    async def _get_gmail_client(self) -> GmailClient:
        """Get or create Gmail client."""
        if self._gmail_client is None:
            settings = get_settings()
            gmail_auth = GmailAuth(settings.gmail_credentials_dir)
            self._gmail_client = GmailClient(gmail_auth)
        return self._gmail_client

    def get_confirmation_message(self, params: ArchiveEmailParams) -> str:
        """Get confirmation message.

        Args:
            params: Validated parameters

        Returns:
            str: Confirmation message
        """
        return f"Archive email {params.email_id} (remove from inbox)"

    async def execute(self, params: ArchiveEmailParams) -> ToolResult:
        """Execute the archive email tool.

        Args:
            params: Validated input parameters

        Returns:
            ToolResult: Result of archiving operation
        """
        try:
            gmail_client = await self._get_gmail_client()

            logger.info(f"Archiving email {params.email_id}")
            await gmail_client.archive_message(params.email_id)

            return ToolResult.success_result(
                data={"email_id": params.email_id, "message": "Email archived successfully"}
            )

        except Exception as e:
            logger.error(f"Failed to archive email: {e}")
            return ToolResult.error_result(error=f"Failed to archive email: {e}")


class CreateDraftParams(BaseModel):
    """Input for CreateDraftTool."""

    to: list[str] = Field(description="List of recipient email addresses")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body (plain text)")
    cc: list[str] = Field(default_factory=list, description="List of CC recipient email addresses")
    reply_to_id: str | None = Field(
        default=None, description="Optional message ID to reply to (for threading)"
    )


class CreateDraftTool(BaseTool[CreateDraftParams]):
    """Create an email draft.

    This tool creates a draft email that can be reviewed and sent later.
    Drafts are saved to Gmail and can be edited in any email client.
    """

    name = "create_draft"
    description = (
        "Create an email draft. The draft will be saved to Gmail but not sent. "
        "Use this to compose emails for later review and sending. "
        "Provide recipient(s), subject, and body text."
    )
    risk_level = RiskLevel.LOW
    parameters_schema = CreateDraftParams

    def __init__(self):
        """Initialize the create draft tool."""
        super().__init__()
        self._gmail_client: GmailClient | None = None

    async def _get_gmail_client(self) -> GmailClient:
        """Get or create Gmail client."""
        if self._gmail_client is None:
            settings = get_settings()
            gmail_auth = GmailAuth(settings.gmail_credentials_dir)
            self._gmail_client = GmailClient(gmail_auth)
        return self._gmail_client

    def get_confirmation_message(self, params: CreateDraftParams) -> str:
        """Get confirmation message.

        Args:
            params: Validated parameters

        Returns:
            str: Confirmation message
        """
        to_str = ", ".join(params.to)
        body_preview = params.body[:100] + "..." if len(params.body) > 100 else params.body
        return f"Create draft to {to_str} | Subject: {params.subject} | Body: {body_preview}"

    async def execute(self, params: CreateDraftParams) -> ToolResult:
        """Execute the create draft tool.

        Args:
            params: Validated input parameters

        Returns:
            ToolResult: Result containing draft ID
        """
        try:
            gmail_client = await self._get_gmail_client()

            logger.info(f"Creating draft to {params.to}")
            draft_id = await gmail_client.create_draft(
                to=params.to,
                subject=params.subject,
                body=params.body,
                cc=params.cc if params.cc else None,
                reply_to_id=params.reply_to_id,
            )

            return ToolResult.success_result(
                data={
                    "draft_id": draft_id,
                    "to": params.to,
                    "subject": params.subject,
                    "message": "Draft created successfully",
                }
            )

        except Exception as e:
            logger.error(f"Failed to create draft: {e}")
            return ToolResult.error_result(error=f"Failed to create draft: {e}")


class SendEmailParams(BaseModel):
    """Input for SendEmailTool."""

    to: list[str] = Field(description="List of recipient email addresses")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body (plain text)")
    cc: list[str] = Field(default_factory=list, description="List of CC recipient email addresses")
    reply_to_id: str | None = Field(
        default=None, description="Optional message ID to reply to (for threading)"
    )


class SendEmailTool(BaseTool[SendEmailParams]):
    """Send an email message.

    This tool sends email directly through Gmail. It requires user confirmation
    before sending to prevent accidental sends.
    """

    name = "send_email"
    description = (
        "Send an email message through Gmail. "
        "Requires user confirmation before sending. "
        "Provide recipient(s), subject, and body text. "
        "Optionally include CC recipients and reply-to threading."
    )
    risk_level = RiskLevel.MEDIUM
    parameters_schema = SendEmailParams

    def __init__(self):
        """Initialize the send email tool."""
        super().__init__()
        self._gmail_client: GmailClient | None = None

    async def _get_gmail_client(self) -> GmailClient:
        """Get or create Gmail client."""
        if self._gmail_client is None:
            settings = get_settings()
            gmail_auth = GmailAuth(settings.gmail_credentials_dir)
            self._gmail_client = GmailClient(gmail_auth)
        return self._gmail_client

    def get_confirmation_message(self, params: SendEmailParams) -> str:
        """Get confirmation message.

        Args:
            params: Validated parameters

        Returns:
            str: Confirmation message with full email preview
        """
        lines = [
            "Send email?",
            f"To: {', '.join(params.to)}",
        ]
        if params.cc:
            lines.append(f"CC: {', '.join(params.cc)}")
        lines.append(f"Subject: {params.subject}")
        lines.append("")  # Blank line

        # Include body preview (first 500 chars)
        body_preview = params.body[:500]
        if len(params.body) > 500:
            body_preview += "..."
        lines.append(body_preview)

        return "\n".join(lines)

    async def execute(self, params: SendEmailParams) -> ToolResult:
        """Execute the send email tool.

        Args:
            params: Validated input parameters

        Returns:
            ToolResult: Result containing sent message ID
        """
        try:
            gmail_client = await self._get_gmail_client()

            logger.info(f"Sending email to {params.to}")
            message_id = await gmail_client.send_message(
                to=params.to,
                subject=params.subject,
                body=params.body,
                cc=params.cc if params.cc else None,
                reply_to_id=params.reply_to_id,
            )

            return ToolResult.success_result(
                data={
                    "message_id": message_id,
                    "to": params.to,
                    "subject": params.subject,
                    "message": "Email sent successfully",
                }
            )

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return ToolResult.error_result(error=f"Failed to send email: {e}")
