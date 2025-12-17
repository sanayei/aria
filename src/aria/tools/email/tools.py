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
        description="Gmail label/folder to list emails from (e.g., INBOX, SENT, SPAM)"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of emails to return"
    )
    unread_only: bool = Field(
        default=False,
        description="If True, only return unread emails"
    )


class ListEmailsTool(BaseTool[ListEmailsParams]):
    """List recent emails from Gmail.

    This tool fetches a list of recent emails from a specified folder/label.
    Results are cached to reduce API calls.
    """

    name = "list_emails"
    description = (
        "List recent emails from Gmail inbox or other folders. "
        "Returns email summaries with subject, sender, date, and preview. "
        "Use this to see what emails the user has received."
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
                include_spam_trash=params.folder.upper() in ["SPAM", "TRASH"]
            )

            # Cache results
            await cache.put_many(emails)

            # Format results
            if not emails:
                return ToolResult.success_result(
                    data={
                        "count": 0,
                        "emails": [],
                        "message": f"No emails found in {params.folder}"
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
        default=10,
        ge=1,
        le=50,
        description="Maximum number of emails to return"
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
            query = GmailQuery(
                query=params.query,
                max_results=params.max_results
            )

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
                        "message": f"No emails found matching query: {params.query}"
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
        default=False,
        description="If True, mark the email as read after reading"
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
