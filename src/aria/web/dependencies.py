"""Dependency injection for FastAPI endpoints."""

from aria.memory import ArchiveIndex, VectorStore
from aria.web.database import UserDatabase

# Global instances (set during app startup)
_user_db: UserDatabase | None = None
_vector_store: VectorStore | None = None
_archive_index: ArchiveIndex | None = None


def set_user_db(db: UserDatabase):
    """Set global user database instance."""
    global _user_db
    _user_db = db


def set_vector_store(store: VectorStore):
    """Set global vector store instance."""
    global _vector_store
    _vector_store = store


def set_archive_index(index: ArchiveIndex):
    """Set global archive index instance."""
    global _archive_index
    _archive_index = index


async def get_user_db() -> UserDatabase:
    """Get user database dependency."""
    if _user_db is None:
        raise RuntimeError("User database not initialized")
    return _user_db


async def get_vector_store() -> VectorStore:
    """Get vector store dependency."""
    if _vector_store is None:
        raise RuntimeError("Vector store not initialized")
    return _vector_store


async def get_archive_index() -> ArchiveIndex:
    """Get archive index dependency."""
    if _archive_index is None:
        raise RuntimeError("Archive index not initialized")
    return _archive_index
