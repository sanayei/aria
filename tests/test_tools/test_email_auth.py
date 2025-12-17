"""Tests for Gmail OAuth authentication."""

import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from aria.tools.email.auth import GmailAuth, GmailAuthError


@pytest.fixture
def temp_credentials_dir(tmp_path: Path):
    """Create a temporary credentials directory for testing."""
    creds_dir = tmp_path / "credentials"
    creds_dir.mkdir(parents=True, exist_ok=True)
    return creds_dir


@pytest.fixture
def gmail_auth(temp_credentials_dir: Path):
    """Create a GmailAuth instance with temporary directory."""
    return GmailAuth(temp_credentials_dir)


@pytest.fixture
def mock_credentials_file(temp_credentials_dir: Path):
    """Create a mock credentials file."""
    creds_file = temp_credentials_dir / "gmail_credentials.json"
    creds_data = {
        "installed": {
            "client_id": "test_client_id.apps.googleusercontent.com",
            "project_id": "test_project",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": "test_secret",
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
        }
    }
    with open(creds_file, "w") as f:
        json.dump(creds_data, f)
    return creds_file


@pytest.fixture
def mock_token_file(temp_credentials_dir: Path):
    """Create a mock token file."""
    token_file = temp_credentials_dir / "gmail_token.json"
    token_data = {
        "token": "test_access_token",
        "refresh_token": "test_refresh_token",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "test_client_id.apps.googleusercontent.com",
        "client_secret": "test_secret",
        "scopes": GmailAuth.SCOPES,
    }
    with open(token_file, "w") as f:
        json.dump(token_data, f)
    return token_file


class TestGmailAuth:
    """Tests for GmailAuth class."""

    def test_init(self, temp_credentials_dir: Path):
        """Test GmailAuth initialization."""
        auth = GmailAuth(temp_credentials_dir)

        assert auth.credentials_dir == temp_credentials_dir
        assert auth.token_path == temp_credentials_dir / "gmail_token.json"
        assert auth.credentials_path == temp_credentials_dir / "gmail_credentials.json"
        assert temp_credentials_dir.exists()

    def test_is_authenticated_no_token(self, gmail_auth: GmailAuth):
        """Test is_authenticated when no token file exists."""
        assert not gmail_auth.is_authenticated()

    def test_is_authenticated_invalid_token(self, gmail_auth: GmailAuth, tmp_path: Path):
        """Test is_authenticated with invalid token file."""
        # Create invalid token file
        with open(gmail_auth.token_path, "w") as f:
            f.write("invalid json")

        assert not gmail_auth.is_authenticated()

    def test_is_authenticated_valid_token(self, gmail_auth: GmailAuth, mock_token_file: Path):
        """Test is_authenticated with valid token."""
        with patch("aria.tools.email.auth.Credentials") as mock_creds_class:
            # Mock a valid credentials object
            mock_creds = Mock()
            mock_creds.valid = True
            mock_creds.expired = False
            mock_creds.refresh_token = "test_refresh_token"
            mock_creds_class.from_authorized_user_file.return_value = mock_creds

            assert gmail_auth.is_authenticated()

    def test_is_authenticated_expired_with_refresh(self, gmail_auth: GmailAuth, mock_token_file: Path):
        """Test is_authenticated with expired token but valid refresh token."""
        with patch("aria.tools.email.auth.Credentials") as mock_creds_class:
            # Mock an expired credentials object with refresh token
            mock_creds = Mock()
            mock_creds.valid = False
            mock_creds.expired = True
            mock_creds.refresh_token = "test_refresh_token"
            mock_creds_class.from_authorized_user_file.return_value = mock_creds

            # Should return True because we can refresh
            assert gmail_auth.is_authenticated()

    @pytest.mark.asyncio
    async def test_authenticate_no_credentials_file(self, gmail_auth: GmailAuth):
        """Test authenticate when credentials file doesn't exist."""
        with pytest.raises(GmailAuthError, match="credentials file not found"):
            await gmail_auth.authenticate()

    @pytest.mark.asyncio
    async def test_authenticate_with_valid_token(self, gmail_auth: GmailAuth, mock_token_file: Path):
        """Test authenticate with valid existing token."""
        with patch("aria.tools.email.auth.Credentials") as mock_creds_class:
            # Mock valid credentials
            mock_creds = Mock()
            mock_creds.valid = True
            mock_creds_class.from_authorized_user_file.return_value = mock_creds

            result = await gmail_auth.authenticate()

            assert result == mock_creds
            mock_creds_class.from_authorized_user_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_authenticate_refresh_token(self, gmail_auth: GmailAuth, mock_token_file: Path):
        """Test authenticate with expired token that needs refresh."""
        with patch("aria.tools.email.auth.Credentials") as mock_creds_class, \
             patch("aria.tools.email.auth.Request") as mock_request, \
             patch.object(gmail_auth, "_save_credentials") as mock_save:

            # Mock expired credentials with refresh token
            mock_creds = Mock()
            mock_creds.valid = False
            mock_creds.expired = True
            mock_creds.refresh_token = "test_refresh_token"
            mock_creds_class.from_authorized_user_file.return_value = mock_creds

            result = await gmail_auth.authenticate()

            # Verify refresh was called
            mock_creds.refresh.assert_called_once_with(mock_request.return_value)
            # Verify credentials were saved
            mock_save.assert_called_once_with(mock_creds)
            assert result == mock_creds

    @pytest.mark.asyncio
    async def test_authenticate_refresh_fails(self, gmail_auth: GmailAuth, mock_token_file: Path):
        """Test authenticate when token refresh fails."""
        with patch("aria.tools.email.auth.Credentials") as mock_creds_class, \
             patch("aria.tools.email.auth.Request"):

            # Mock expired credentials that fail to refresh
            mock_creds = Mock()
            mock_creds.valid = False
            mock_creds.expired = True
            mock_creds.refresh_token = "test_refresh_token"
            mock_creds.refresh.side_effect = Exception("Refresh failed")
            mock_creds_class.from_authorized_user_file.return_value = mock_creds

            with pytest.raises(GmailAuthError, match="Failed to refresh Gmail token"):
                await gmail_auth.authenticate()

    def test_save_credentials(self, gmail_auth: GmailAuth):
        """Test saving credentials to file."""
        # Mock credentials object
        mock_creds = Mock()
        mock_creds.token = "test_token"
        mock_creds.refresh_token = "test_refresh"
        mock_creds.token_uri = "https://oauth2.googleapis.com/token"
        mock_creds.client_id = "test_client_id"
        mock_creds.client_secret = "test_secret"
        mock_creds.scopes = GmailAuth.SCOPES

        gmail_auth._save_credentials(mock_creds)

        # Verify file was created
        assert gmail_auth.token_path.exists()

        # Verify content
        with open(gmail_auth.token_path) as f:
            data = json.load(f)

        assert data["token"] == "test_token"
        assert data["refresh_token"] == "test_refresh"
        assert data["scopes"] == GmailAuth.SCOPES

    @pytest.mark.asyncio
    async def test_get_service_success(self, gmail_auth: GmailAuth, mock_token_file: Path):
        """Test getting Gmail service successfully."""
        with patch("aria.tools.email.auth.Credentials") as mock_creds_class, \
             patch("aria.tools.email.auth.build") as mock_build:

            # Mock valid credentials
            mock_creds = Mock()
            mock_creds.valid = True
            mock_creds_class.from_authorized_user_file.return_value = mock_creds

            # Mock service
            mock_service = Mock()
            mock_build.return_value = mock_service

            result = await gmail_auth.get_service()

            assert result == mock_service
            mock_build.assert_called_once_with("gmail", "v1", credentials=mock_creds)

    @pytest.mark.asyncio
    async def test_test_connection_success(self, gmail_auth: GmailAuth, mock_token_file: Path):
        """Test connection test with successful API call."""
        with patch("aria.tools.email.auth.Credentials") as mock_creds_class, \
             patch("aria.tools.email.auth.build") as mock_build:

            # Mock valid credentials
            mock_creds = Mock()
            mock_creds.valid = True
            mock_creds_class.from_authorized_user_file.return_value = mock_creds

            # Mock service and profile
            mock_service = MagicMock()
            mock_profile = {"emailAddress": "test@example.com"}
            mock_service.users().getProfile().execute.return_value = mock_profile
            mock_build.return_value = mock_service

            result = await gmail_auth.test_connection()

            assert result is True

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, gmail_auth: GmailAuth, mock_token_file: Path):
        """Test connection test with API failure."""
        with patch("aria.tools.email.auth.Credentials") as mock_creds_class, \
             patch("aria.tools.email.auth.build") as mock_build:

            # Mock valid credentials
            mock_creds = Mock()
            mock_creds.valid = True
            mock_creds_class.from_authorized_user_file.return_value = mock_creds

            # Mock service that raises error
            mock_service = MagicMock()
            mock_service.users().getProfile().execute.side_effect = Exception("API Error")
            mock_build.return_value = mock_service

            result = await gmail_auth.test_connection()

            assert result is False

    def test_revoke_access(self, gmail_auth: GmailAuth, mock_token_file: Path):
        """Test revoking Gmail access."""
        assert gmail_auth.token_path.exists()

        result = gmail_auth.revoke_access()

        assert result is True
        assert not gmail_auth.token_path.exists()

    def test_get_auth_status_authenticated(self, gmail_auth: GmailAuth, mock_token_file: Path):
        """Test get_auth_status when authenticated."""
        with patch("aria.tools.email.auth.Credentials") as mock_creds_class:
            # Mock valid credentials
            mock_creds = Mock()
            mock_creds.valid = True
            mock_creds.expired = False
            mock_creds.refresh_token = "test_refresh"
            mock_creds_class.from_authorized_user_file.return_value = mock_creds

            status = gmail_auth.get_auth_status()

            assert status["authenticated"] is True
            assert status["token_file_exists"] is True
            assert status["token_valid"] is True
            assert status["token_expired"] is False
            assert status["has_refresh_token"] is True

    def test_get_auth_status_not_authenticated(self, gmail_auth: GmailAuth):
        """Test get_auth_status when not authenticated."""
        status = gmail_auth.get_auth_status()

        assert status["authenticated"] is False
        assert status["token_file_exists"] is False
        assert status["credentials_file_exists"] is False
