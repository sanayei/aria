"""View the full OCR text from a document in the archive.

This script retrieves all chunks for a document and displays the full text.
"""

import asyncio
from pathlib import Path

import click

from aria.config import Settings
from aria.memory.embeddings import OllamaEmbeddings
from aria.memory.vectors import VectorStore


async def view_document(document_path: str):
    """View full text of a document."""
    settings = Settings()

    # Initialize vector store
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        host=settings.ollama_host,
    )
    vector_store = VectorStore(
        persist_directory=settings.aria_data_dir / "chroma",
        embedding_provider=embeddings,
        collection_name="archived_documents",
    )
    await vector_store.initialize()

    # Search for chunks from this document
    results = await vector_store.search(
        query="document",  # Generic query
        limit=1000,
        filter={"archived_path": document_path},
    )

    if not results:
        print(f"No chunks found for: {document_path}")
        return

    print(f"\n{'=' * 80}")
    print(f"Document: {Path(document_path).name}")
    print(f"Path: {document_path}")
    print(f"Total chunks: {len(results)}")
    print(f"{'=' * 80}\n")

    # Sort chunks by chunk index if available
    sorted_results = sorted(
        results,
        key=lambda r: r.doc_id,  # Document IDs contain chunk index
    )

    for i, result in enumerate(sorted_results, 1):
        print(f"--- Chunk {i} of {len(results)} ---")
        print(result.content)
        print()


@click.command()
@click.argument("document_path")
def main(document_path: str):
    """View full OCR text from an archived document.

    Example:
        python view_document_text.py "/home/amir/Documents/Archive/2025/amir/medical/2025-11-24_meritain_health_medical.pdf"
    """
    asyncio.run(view_document(document_path))


if __name__ == "__main__":
    main()
