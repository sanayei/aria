"""Gmail OAuth authentication handler for ARIA.

This module provides secure OAuth2 authentication for Gmail API access,
including token storage, refresh, and revocation.
"""

import json
import os
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError

from aria.logging import get_logger

logger = get_logger("aria.tools.email.auth")


class GmailAuthError(Exception):
    """Exception raised when Gmail authentication fails."""

    pass


class GmailAuth:
    """Handle Gmail OAuth authentication.

    This class manages the OAuth2 flow for Gmail API access, including:
    - Initial authentication via browser consent
    - Token storage and retrieval
    - Automatic token refresh
    - Token revocation
    """

    # Gmail API scopes - defines what permissions we request
    SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",  # Read emails
        "https://www.googleapis.com/auth/gmail.modify",  # Modify emails (labels, etc.)
        "https://www.googleapis.com/auth/gmail.send",  # Send emails
    ]

    def __init__(self, credentials_dir: Path):
        """Initialize Gmail authentication.

        Args:
            credentials_dir: Directory to store credentials and tokens
        """
        self.credentials_dir = Path(credentials_dir)
        self.token_path = self.credentials_dir / "gmail_token.json"
        self.credentials_path = self.credentials_dir / "gmail_credentials.json"

        # Create credentials directory if it doesn't exist
        self.credentials_dir.mkdir(parents=True, exist_ok=True)

        # Set restrictive permissions on credentials directory (owner only)
        try:
            os.chmod(self.credentials_dir, 0o700)
        except OSError as e:
            logger.warning(f"Failed to set permissions on {self.credentials_dir}: {e}")

    def is_authenticated(self) -> bool:
        """Check if valid credentials exist.

        Returns:
            bool: True if valid credentials are available, False otherwise
        """
        if not self.token_path.exists():
            return False

        try:
            creds = Credentials.from_authorized_user_file(str(self.token_path), self.SCOPES)
            return creds.valid or (creds.expired and creds.refresh_token)
        except Exception as e:
            logger.debug(f"Failed to load credentials: {e}")
            return False

    async def authenticate(self, force_reauth: bool = False) -> Credentials:
        """Get authenticated credentials, running OAuth flow if needed.

        Args:
            force_reauth: If True, force re-authentication even if token exists

        Returns:
            Credentials: Authenticated credentials

        Raises:
            GmailAuthError: If authentication fails
        """
        creds = None

        # Load existing token if not forcing re-auth
        if not force_reauth and self.token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_path), self.SCOPES)
                logger.debug("Loaded existing credentials")
            except Exception as e:
                logger.warning(f"Failed to load existing credentials: {e}")

        # If no valid credentials available, run OAuth flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                # Token expired but we have refresh token - refresh it
                try:
                    logger.info("Refreshing expired Gmail token")
                    creds.refresh(Request())
                    logger.info("Token refreshed successfully")
                except Exception as e:
                    logger.error(f"Failed to refresh token: {e}")
                    raise GmailAuthError(f"Failed to refresh Gmail token: {e}") from e
            else:
                # No valid credentials - need to run OAuth flow
                if not self.credentials_path.exists():
                    raise GmailAuthError(
                        f"Gmail credentials file not found at {self.credentials_path}. "
                        "Please run 'python scripts/setup_gmail.py' to set up Gmail authentication."
                    )

                try:
                    logger.info("Starting Gmail OAuth flow")
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_path), self.SCOPES
                    )
                    # Run local server for OAuth callback
                    creds = flow.run_local_server(port=0)
                    logger.info("Gmail OAuth flow completed successfully")
                except Exception as e:
                    logger.error(f"OAuth flow failed: {e}")
                    raise GmailAuthError(f"Gmail OAuth authentication failed: {e}") from e

            # Save the credentials for next time
            try:
                self._save_credentials(creds)
            except Exception as e:
                logger.error(f"Failed to save credentials: {e}")
                raise GmailAuthError(f"Failed to save Gmail credentials: {e}") from e

        return creds

    def _save_credentials(self, creds: Credentials) -> None:
        """Save credentials to token file with restricted permissions.

        Args:
            creds: Credentials to save

        Raises:
            GmailAuthError: If saving fails
        """
        try:
            # Write credentials to file
            token_data = {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes,
            }

            with open(self.token_path, "w") as f:
                json.dump(token_data, f)

            # Set restrictive permissions (owner read/write only)
            os.chmod(self.token_path, 0o600)

            logger.info(f"Credentials saved to {self.token_path}")
        except Exception as e:
            raise GmailAuthError(f"Failed to save credentials: {e}") from e

    async def get_service(self, force_reauth: bool = False) -> Resource:
        """Get authenticated Gmail API service.

        Args:
            force_reauth: If True, force re-authentication

        Returns:
            Resource: Authenticated Gmail API service

        Raises:
            GmailAuthError: If authentication or service creation fails
        """
        try:
            creds = await self.authenticate(force_reauth=force_reauth)
            service = build("gmail", "v1", credentials=creds)
            logger.debug("Gmail service created successfully")
            return service
        except HttpError as e:
            logger.error(f"Gmail API error: {e}")
            raise GmailAuthError(f"Failed to create Gmail service: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error creating Gmail service: {e}")
            raise GmailAuthError(f"Failed to create Gmail service: {e}") from e

    async def test_connection(self) -> bool:
        """Test Gmail API connection by fetching user profile.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            service = await self.get_service()
            # Try to get user profile as a simple test
            profile = service.users().getProfile(userId="me").execute()
            email = profile.get("emailAddress", "unknown")
            logger.info(f"Gmail connection test successful for {email}")
            return True
        except Exception as e:
            logger.error(f"Gmail connection test failed: {e}")
            return False

    def revoke_access(self) -> bool:
        """Revoke Gmail access and delete stored credentials.

        Returns:
            bool: True if successfully revoked, False otherwise
        """
        try:
            # Delete token file
            if self.token_path.exists():
                self.token_path.unlink()
                logger.info("Gmail token deleted")

            # Note: We don't delete credentials file as user may want to re-authenticate
            logger.info("Gmail access revoked")
            return True
        except Exception as e:
            logger.error(f"Failed to revoke Gmail access: {e}")
            return False

    def get_auth_status(self) -> dict[str, str | bool]:
        """Get current authentication status.

        Returns:
            dict: Authentication status information
        """
        status = {
            "authenticated": self.is_authenticated(),
            "credentials_file_exists": self.credentials_path.exists(),
            "token_file_exists": self.token_path.exists(),
            "credentials_dir": str(self.credentials_dir),
        }

        if self.is_authenticated():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_path), self.SCOPES)
                status["token_valid"] = creds.valid
                status["token_expired"] = creds.expired
                status["has_refresh_token"] = bool(creds.refresh_token)
            except Exception as e:
                logger.debug(f"Failed to get detailed auth status: {e}")
                status["error"] = str(e)

        return status
