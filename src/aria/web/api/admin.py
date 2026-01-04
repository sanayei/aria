"""Admin API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from aria.logging import get_logger
from aria.web.auth import require_admin
from aria.web.database import UserDatabase
from aria.web.dependencies import get_user_db
from aria.web.models import User, UserCreate, UserUpdate

logger = get_logger("aria.web.api.admin")

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/users", response_model=list[User])
async def list_users(
    admin_user: Annotated[User, Depends(require_admin)],
    user_db: Annotated[UserDatabase, Depends(get_user_db)],
):
    """List all users (admin only).

    Args:
        admin_user: Admin user (from dependency)
        user_db: User database

    Returns:
        List of all users
    """
    users = await user_db.list_users()
    return users


@router.post("/users", response_model=User, status_code=201)
async def create_user(
    user_create: UserCreate,
    admin_user: Annotated[User, Depends(require_admin)],
    user_db: Annotated[UserDatabase, Depends(get_user_db)],
):
    """Create a new user (admin only).

    Args:
        user_create: User creation data
        admin_user: Admin user (from dependency)
        user_db: User database

    Returns:
        Created user

    Raises:
        HTTPException: If username already exists
    """
    # Check if username already exists
    existing_user = await user_db.get_user_by_username(user_create.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    user = await user_db.create_user(user_create)
    logger.info(f"Admin {admin_user.username} created user: {user.username}")

    return user


@router.put("/users/{user_id}", response_model=User)
async def update_user(
    user_id: str,
    user_update: UserUpdate,
    admin_user: Annotated[User, Depends(require_admin)],
    user_db: Annotated[UserDatabase, Depends(get_user_db)],
):
    """Update a user (admin only).

    Args:
        user_id: User ID to update
        user_update: Update data
        admin_user: Admin user (from dependency)
        user_db: User database

    Returns:
        Updated user

    Raises:
        HTTPException: If user not found
    """
    user = await user_db.update_user(user_id, user_update)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    logger.info(f"Admin {admin_user.username} updated user: {user.username}")

    return user


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    admin_user: Annotated[User, Depends(require_admin)],
    user_db: Annotated[UserDatabase, Depends(get_user_db)],
):
    """Delete a user (admin only).

    Args:
        user_id: User ID to delete
        admin_user: Admin user (from dependency)
        user_db: User database

    Returns:
        Success message

    Raises:
        HTTPException: If user not found or trying to delete self
    """
    # Prevent admin from deleting themselves
    if user_id == admin_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    deleted = await user_db.delete_user(user_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")

    logger.info(f"Admin {admin_user.username} deleted user: {user_id}")

    return {"message": "User deleted successfully"}
