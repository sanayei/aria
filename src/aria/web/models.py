"""User and permission models for web application."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class UserRole:
    """User role constants."""

    ADMIN = "admin"
    POWER_USER = "power_user"
    FAMILY_MEMBER = "family_member"
    VIEWER = "viewer"

    ALL_ROLES = [ADMIN, POWER_USER, FAMILY_MEMBER, VIEWER]


class UserPermissions(BaseModel):
    """User permission configuration."""

    # Person access (None = all access)
    allowed_persons: list[str] | None = None
    denied_persons: list[str] = Field(default_factory=list)

    # Category access (None = all access)
    allowed_categories: list[str] | None = None
    denied_categories: list[str] = Field(default_factory=list)

    # Action permissions
    can_view: bool = True
    can_download: bool = True
    can_search_all: bool = True
    can_manage_users: bool = False
    can_rescan: bool = False
    can_delete: bool = False
    can_create_tags: bool = False


class User(BaseModel):
    """User model."""

    id: str
    username: str
    full_name: str
    role: str
    permissions: UserPermissions
    created_at: datetime
    last_login: datetime | None = None
    is_active: bool = True

    @classmethod
    def get_default_permissions(cls, role: str) -> UserPermissions:
        """Get default permissions for a role."""
        if role == UserRole.ADMIN:
            return UserPermissions(
                allowed_persons=None,  # All
                allowed_categories=None,  # All
                can_manage_users=True,
                can_rescan=True,
                can_delete=True,
                can_create_tags=True,
            )
        elif role == UserRole.POWER_USER:
            return UserPermissions(
                allowed_persons=None,  # All
                allowed_categories=None,  # All
                can_manage_users=False,
                can_rescan=False,
                can_delete=False,
                can_create_tags=True,
            )
        elif role == UserRole.FAMILY_MEMBER:
            # Will be customized per user
            return UserPermissions(
                allowed_persons=[],  # Set to user's own person
                allowed_categories=None,  # All categories for their docs
                can_search_all=False,
            )
        else:  # VIEWER
            return UserPermissions(
                allowed_persons=[],  # Set manually
                allowed_categories=[],  # Set manually
                can_search_all=False,
                can_download=True,
            )


class UserInDB(User):
    """User model with hashed password (stored in database)."""

    password_hash: str


class UserCreate(BaseModel):
    """User creation request."""

    username: str
    password: str
    full_name: str
    role: str
    permissions: UserPermissions | None = None  # Optional, uses role defaults if not provided


class UserUpdate(BaseModel):
    """User update request."""

    full_name: str | None = None
    role: str | None = None
    permissions: UserPermissions | None = None
    is_active: bool | None = None
    password: str | None = None  # Optional password change


class Token(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """JWT token payload data."""

    username: str | None = None


class LoginRequest(BaseModel):
    """Login request."""

    username: str
    password: str
