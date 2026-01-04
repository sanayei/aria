"""Authentication API endpoints."""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from aria.logging import get_logger
from aria.web.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
    get_current_active_user,
)
from aria.web.database import UserDatabase
from aria.web.dependencies import get_user_db
from aria.web.models import Token, User

logger = get_logger("aria.web.api.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    user_db: Annotated[UserDatabase, Depends(get_user_db)],
):
    """Login endpoint.

    Args:
        form_data: OAuth2 password form (username + password)
        user_db: User database

    Returns:
        JWT token

    Raises:
        HTTPException: If authentication fails
    """
    user = await authenticate_user(user_db, form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login
    await user_db.update_last_login(form_data.username)

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    logger.info(f"User logged in: {user.username}")

    return Token(access_token=access_token)


@router.get("/me", response_model=User)
async def get_me(current_user: Annotated[User, Depends(get_current_active_user)]):
    """Get current user information.

    Args:
        current_user: Current authenticated user

    Returns:
        Current user
    """
    return current_user


@router.post("/logout")
async def logout(current_user: Annotated[User, Depends(get_current_active_user)]):
    """Logout endpoint (client should discard token).

    Args:
        current_user: Current authenticated user

    Returns:
        Success message
    """
    logger.info(f"User logged out: {current_user.username}")
    return {"message": "Successfully logged out"}
