"""Migrate person names in ChromaDB to lowercase for consistency.

This script normalizes all person metadata fields to lowercase first names
to match the current classification system.
"""

import asyncio
from pathlib import Path

import chromadb

from aria.config import Settings


async def main():
    """Migrate all person fields to lowercase."""
    settings = Settings()

    # Connect directly to ChromaDB
    persist_dir = settings.aria_data_dir / "chroma"
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection("archived_documents")

    print(f"Collection has {collection.count()} documents")

    # Get all documents in batches
    batch_size = 1000
    offset = 0
    total_updated = 0

    while True:
        # Fetch batch
        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )

        if not results["ids"]:
            break

        print(f"\nProcessing batch at offset {offset} ({len(results['ids'])} documents)...")

        # Update metadata for each document
        for doc_id, metadata in zip(results["ids"], results["metadatas"]):
            if not metadata:
                continue

            person = metadata.get("person")
            if not person:
                continue

            # Normalize to lowercase first name
            normalized = person.strip().split()[0].lower()

            if normalized != person:
                # Update the document
                metadata["person"] = normalized
                collection.update(
                    ids=[doc_id],
                    metadatas=[metadata],
                )
                total_updated += 1
                print(f"  Updated: {person} → {normalized}")

        offset += len(results["ids"])

        # Break if we got fewer results than batch size
        if len(results["ids"]) < batch_size:
            break

    print(f"\n✓ Migration complete!")
    print(f"  Total documents checked: {offset}")
    print(f"  Documents updated: {total_updated}")
    print(f"  Documents unchanged: {offset - total_updated}")


if __name__ == "__main__":
    asyncio.run(main())
