"""Authentication and authorization logic."""

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from aria.config import get_settings
from aria.logging import get_logger
from aria.web.database import UserDatabase
from aria.web.models import TokenData, User, UserInDB

logger = get_logger("aria.web.auth")

# Import will be done lazily to avoid circular imports
def get_user_db_dependency():
    """Get user database dependency (lazy import)."""
    from aria.web.dependencies import get_user_db
    return get_user_db

# JWT settings
import os
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
if not SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY environment variable not set. "
        "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create JWT access token.

    Args:
        data: Data to encode in token
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


async def authenticate_user(
    user_db: UserDatabase, username: str, password: str
) -> UserInDB | None:
    """Authenticate user with username and password.

    Args:
        user_db: User database
        username: Username
        password: Plain password

    Returns:
        User if authenticated, None otherwise
    """
    user = await user_db.get_user_by_username(username)

    if not user:
        return None

    if not user.is_active:
        return None

    if not user_db.verify_password(password, user.password_hash):
        return None

    return user


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    user_db: Annotated[UserDatabase, Depends(get_user_db_dependency())],
) -> User:
    """Get current user from JWT token.

    Args:
        token: JWT token
        user_db: User database

    Returns:
        Current user

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")

        if username is None:
            raise credentials_exception

        token_data = TokenData(username=username)

    except JWTError:
        raise credentials_exception

    user = await user_db.get_user_by_username(token_data.username)

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="User account is disabled"
        )

    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """Get current active user (dependency).

    Args:
        current_user: Current user from token

    Returns:
        Active user

    Raises:
        HTTPException: If user is not active
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    return current_user


def require_admin(current_user: Annotated[User, Depends(get_current_active_user)]):
    """Require admin role (dependency).

    Args:
        current_user: Current user

    Raises:
        HTTPException: If user is not admin
    """
    if not current_user.permissions.can_manage_users:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )

    return current_user


def check_document_access(user: User, person: str, category: str) -> bool:
    """Check if user has access to a document.

    Args:
        user: User to check
        person: Document person
        category: Document category

    Returns:
        True if user has access
    """
    # Check person access
    if user.permissions.allowed_persons is not None:
        if person not in user.permissions.allowed_persons:
            return False

    if person in user.permissions.denied_persons:
        return False

    # Check category access
    if user.permissions.allowed_categories is not None:
        if category not in user.permissions.allowed_categories:
            return False

    if category in user.permissions.denied_categories:
        return False

    return True


def build_document_filters(user: User, additional_filters: dict | None = None) -> dict:
    """Build ChromaDB filters based on user permissions.

    Args:
        user: User to build filters for
        additional_filters: Additional filters to apply

    Returns:
        Filter dictionary for ChromaDB query
    """
    filter_conditions = []

    # Person filtering
    if user.permissions.allowed_persons is not None:
        if len(user.permissions.allowed_persons) == 1:
            filter_conditions.append({"person": user.permissions.allowed_persons[0]})
        elif len(user.permissions.allowed_persons) > 1:
            filter_conditions.append(
                {"person": {"$in": user.permissions.allowed_persons}}
            )
        # If empty list, no access - will return empty results

    # Category filtering
    if user.permissions.allowed_categories is not None:
        if len(user.permissions.allowed_categories) == 1:
            filter_conditions.append(
                {"category": user.permissions.allowed_categories[0]}
            )
        elif len(user.permissions.allowed_categories) > 1:
            filter_conditions.append(
                {"category": {"$in": user.permissions.allowed_categories}}
            )

    # Add user's requested filters
    if additional_filters:
        if "person" in additional_filters:
            # Verify user has access to this person
            if user.permissions.allowed_persons is not None:
                if additional_filters["person"] not in user.permissions.allowed_persons:
                    # User doesn't have access - return filter that matches nothing
                    return {"person": "__no_access__"}
            filter_conditions.append({"person": additional_filters["person"]})

        if "category" in additional_filters:
            # Verify user has access to this category
            if user.permissions.allowed_categories is not None:
                if (
                    additional_filters["category"]
                    not in user.permissions.allowed_categories
                ):
                    return {"category": "__no_access__"}
            filter_conditions.append({"category": additional_filters["category"]})

    # Combine filters with $and
    if len(filter_conditions) == 0:
        return {}
    elif len(filter_conditions) == 1:
        return filter_conditions[0]
    else:
        return {"$and": filter_conditions}
