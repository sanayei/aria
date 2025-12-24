"""Reset the entire archive system - ChromaDB, SQLite index, and folders.

WARNING: This will delete all indexed documents and embeddings!
Only the original scanned files in /mnt/scan will remain.
"""

import asyncio
import shutil
from pathlib import Path

import chromadb
import click

from aria.config import Settings


async def main():
    """Reset all archive data."""
    settings = Settings()

    print("=" * 70)
    print("ARCHIVE RESET - This will delete:")
    print("  1. ChromaDB vector store (all embeddings)")
    print("  2. Archive index database (SQLite)")
    print("  3. Organized archive folders")
    print("=" * 70)
    print()

    # Show what will be deleted
    items_to_delete = []

    # ChromaDB path
    chroma_path = settings.aria_data_dir / "chroma"
    if chroma_path.exists():
        items_to_delete.append(f"ChromaDB: {chroma_path}")

    # Archive index
    archive_db = settings.archive_db_path or (settings.aria_data_dir / "cache" / "archive.db")
    if archive_db.exists():
        items_to_delete.append(f"Archive DB: {archive_db}")

    # Archive directory
    archive_dir = settings.archive_directory
    if archive_dir.exists():
        items_to_delete.append(f"Archive folder: {archive_dir}")

    if not items_to_delete:
        print("Nothing to delete - archive is already clean!")
        return

    print("Will delete:")
    for item in items_to_delete:
        print(f"  - {item}")
    print()

    # Confirm
    if not click.confirm("Are you sure you want to delete all archive data?", default=False):
        print("Cancelled.")
        return

    print()
    print("Deleting...")

    # Delete ChromaDB
    if chroma_path.exists():
        print(f"  Deleting ChromaDB: {chroma_path}")
        shutil.rmtree(chroma_path)

    # Delete archive index
    if archive_db.exists():
        print(f"  Deleting archive DB: {archive_db}")
        archive_db.unlink()

    # Delete archive directory
    if archive_dir.exists():
        print(f"  Deleting archive folder: {archive_dir}")
        shutil.rmtree(archive_dir)

    print()
    print("✓ Archive reset complete!")
    print()
    print("Next steps:")
    print("  1. Run: uv run aria scan process /mnt/scan --execute")
    print("  2. This will re-process all documents with consistent naming")


if __name__ == "__main__":
    asyncio.run(main())
