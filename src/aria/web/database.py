"""User database management."""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path

import aiosqlite
from passlib.context import CryptContext

from aria.logging import get_logger
from aria.web.models import User, UserInDB, UserCreate, UserUpdate, UserPermissions

logger = get_logger("aria.web.database")

# Password hashing - using argon2 (more modern, no 72-byte limit)
pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")


class UserDatabase:
    """User database manager using SQLite."""

    def __init__(self, db_path: Path):
        """Initialize user database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize database and create tables if needed."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row

        # Create users table
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL,
                permissions TEXT NOT NULL,  -- JSON
                created_at TIMESTAMP NOT NULL,
                last_login TIMESTAMP,
                is_active BOOLEAN NOT NULL DEFAULT 1
            )
            """
        )

        # Create indexes
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)"
        )

        await self._db.commit()

        logger.info(f"User database initialized at {self.db_path}")

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    def _hash_password(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash."""
        return pwd_context.verify(plain_password, hashed_password)

    async def create_user(self, user_create: UserCreate) -> User:
        """Create a new user.

        Args:
            user_create: User creation data

        Returns:
            Created user (without password hash)
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        user_id = str(uuid.uuid4())
        password_hash = self._hash_password(user_create.password)
        created_at = datetime.utcnow()

        # Use provided permissions or role defaults
        permissions = user_create.permissions or User.get_default_permissions(
            user_create.role
        )

        await self._db.execute(
            """
            INSERT INTO users (
                id, username, password_hash, full_name, role,
                permissions, created_at, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                user_create.username,
                password_hash,
                user_create.full_name,
                user_create.role,
                permissions.model_dump_json(),
                created_at,
                True,
            ),
        )
        await self._db.commit()

        logger.info(f"Created user: {user_create.username} (role: {user_create.role})")

        return User(
            id=user_id,
            username=user_create.username,
            full_name=user_create.full_name,
            role=user_create.role,
            permissions=permissions,
            created_at=created_at,
            is_active=True,
        )

    async def get_user_by_username(self, username: str) -> UserInDB | None:
        """Get user by username (includes password hash).

        Args:
            username: Username to look up

        Returns:
            User with password hash or None if not found
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        cursor = await self._db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return UserInDB(
            id=row["id"],
            username=row["username"],
            password_hash=row["password_hash"],
            full_name=row["full_name"],
            role=row["role"],
            permissions=UserPermissions.model_validate_json(row["permissions"]),
            created_at=row["created_at"],
            last_login=row["last_login"],
            is_active=bool(row["is_active"]),
        )

    async def get_user_by_id(self, user_id: str) -> User | None:
        """Get user by ID (without password hash).

        Args:
            user_id: User ID

        Returns:
            User or None if not found
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        cursor = await self._db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = await cursor.fetchone()

        if not row:
            return None

        return User(
            id=row["id"],
            username=row["username"],
            full_name=row["full_name"],
            role=row["role"],
            permissions=UserPermissions.model_validate_json(row["permissions"]),
            created_at=row["created_at"],
            last_login=row["last_login"],
            is_active=bool(row["is_active"]),
        )

    async def update_last_login(self, username: str) -> None:
        """Update user's last login timestamp.

        Args:
            username: Username
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        await self._db.execute(
            "UPDATE users SET last_login = ? WHERE username = ?",
            (datetime.utcnow(), username),
        )
        await self._db.commit()

    async def list_users(self) -> list[User]:
        """List all users (without password hashes).

        Returns:
            List of users
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        cursor = await self._db.execute("SELECT * FROM users ORDER BY created_at")
        rows = await cursor.fetchall()

        return [
            User(
                id=row["id"],
                username=row["username"],
                full_name=row["full_name"],
                role=row["role"],
                permissions=UserPermissions.model_validate_json(row["permissions"]),
                created_at=row["created_at"],
                last_login=row["last_login"],
                is_active=bool(row["is_active"]),
            )
            for row in rows
        ]

    async def update_user(self, user_id: str, user_update: UserUpdate) -> User | None:
        """Update user.

        Args:
            user_id: User ID
            user_update: Update data

        Returns:
            Updated user or None if not found
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        # Get current user
        user = await self.get_user_by_id(user_id)
        if not user:
            return None

        # Build update query
        updates = []
        params = []

        if user_update.full_name is not None:
            updates.append("full_name = ?")
            params.append(user_update.full_name)

        if user_update.role is not None:
            updates.append("role = ?")
            params.append(user_update.role)

        if user_update.permissions is not None:
            updates.append("permissions = ?")
            params.append(user_update.permissions.model_dump_json())

        if user_update.is_active is not None:
            updates.append("is_active = ?")
            params.append(user_update.is_active)

        if user_update.password is not None:
            updates.append("password_hash = ?")
            params.append(self._hash_password(user_update.password))

        if not updates:
            return user  # No changes

        params.append(user_id)

        await self._db.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = ?", tuple(params)
        )
        await self._db.commit()

        logger.info(f"Updated user: {user.username}")

        # Return updated user
        return await self.get_user_by_id(user_id)

    async def delete_user(self, user_id: str) -> bool:
        """Delete user.

        Args:
            user_id: User ID

        Returns:
            True if deleted, False if not found
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        cursor = await self._db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        await self._db.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted user: {user_id}")

        return deleted

    async def user_exists(self) -> bool:
        """Check if any users exist.

        Returns:
            True if at least one user exists
        """
        if not self._db:
            raise RuntimeError("Database not initialized")

        cursor = await self._db.execute("SELECT COUNT(*) FROM users")
        row = await cursor.fetchone()
        return row[0] > 0
