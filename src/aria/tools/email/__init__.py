"""Email tools for ARIA - Gmail integration and email management."""

from aria.tools.email.auth import GmailAuth, GmailAuthError
from aria.tools.email.client import GmailClient, GmailClientError
from aria.tools.email.cache import EmailCache, EmailCacheError
from aria.tools.email.tools import (
    ListEmailsTool,
    SearchEmailsTool,
    ReadEmailTool,
)
from aria.tools.email.models import (
    EmailSummary,
    EmailDetail,
    AttachmentInfo,
    EmailThread,
    GmailQuery,
)

__all__ = [
    # Auth
    "GmailAuth",
    "GmailAuthError",
    # Client
    "GmailClient",
    "GmailClientError",
    # Cache
    "EmailCache",
    "EmailCacheError",
    # Tools
    "ListEmailsTool",
    "SearchEmailsTool",
    "ReadEmailTool",
    # Models
    "EmailSummary",
    "EmailDetail",
    "AttachmentInfo",
    "EmailThread",
    "GmailQuery",
]
