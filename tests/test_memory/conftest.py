"""Pytest fixtures for memory tests."""

from pathlib import Path

import pytest

from aria.memory.conversation import ConversationStore


@pytest.fixture
async def temp_db(tmp_path: Path):
    """Create a temporary database for testing.

    Args:
        tmp_path: Pytest temporary path fixture

    Yields:
        Path: Path to temporary database file
    """
    db_path = tmp_path / "test_conversations.db"
    yield db_path

    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
async def conversation_store(temp_db: Path):
    """Create an initialized ConversationStore for testing.

    Args:
        temp_db: Temporary database path

    Yields:
        ConversationStore: Initialized conversation store
    """
    store = ConversationStore(db_path=temp_db)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def store_with_session(conversation_store: ConversationStore):
    """Create a ConversationStore with a test session.

    Args:
        conversation_store: Initialized conversation store

    Yields:
        tuple[ConversationStore, Session]: Store and created session
    """
    session = await conversation_store.create_session("Test Session")
    yield conversation_store, session


@pytest.fixture
async def store_with_messages(store_with_session):
    """Create a ConversationStore with a session and messages.

    Args:
        store_with_session: Store with test session

    Yields:
        tuple[ConversationStore, Session, list[Message]]: Store, session, and messages
    """
    store, session = store_with_session

    messages = []
    messages.append(await store.add_message(session.id, "user", "Hello, ARIA!"))
    messages.append(
        await store.add_message(session.id, "assistant", "Hello! How can I help you?")
    )
    messages.append(await store.add_message(session.id, "user", "What's the weather?"))

    yield store, session, messages
