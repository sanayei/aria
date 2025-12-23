"""Document chunking for semantic search."""

from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

from aria.logging import get_logger

if TYPE_CHECKING:
    from aria.tools.documents.pdf_extractor import PDFExtractor

logger = get_logger("aria.memory.chunker")


class DocumentChunk(BaseModel):
    """A chunk of text from a document."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_index: int
    total_chunks: int
    source_file: str
    page_number: int | None = None  # For PDFs


class DocumentChunker:
    """Split documents into overlapping chunks for embedding.

    This class handles intelligent text splitting that preserves context
    while staying within token limits for embeddings.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n\n",
    ):
        """Initialize the document chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            separator: Preferred split point (e.g., paragraph breaks)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

        logger.debug(f"Initialized DocumentChunker: size={chunk_size}, overlap={chunk_overlap}")

    def chunk_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        source_file: str = "unknown",
    ) -> list[DocumentChunk]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            source_file: Source file path

        Returns:
            list[DocumentChunk]: List of text chunks
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}

        # Split on preferred separator first
        paragraphs = text.split(self.separator)

        chunks = []
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())

                # Start new chunk with overlap from previous chunk
                if self.chunk_overlap > 0:
                    # Take last chunk_overlap characters for context
                    overlap_text = current_chunk[-self.chunk_overlap :]
                    current_chunk = overlap_text + self.separator + para
                else:
                    current_chunk = para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += self.separator + para
                else:
                    current_chunk = para

            # If current chunk itself is too large, split it
            if len(current_chunk) > self.chunk_size * 1.5:
                # Split large paragraph into smaller chunks
                words = current_chunk.split()
                temp_chunk = ""

                for word in words:
                    if len(temp_chunk) + len(word) + 1 > self.chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                            # Overlap: take last N words
                            overlap_words = temp_chunk.split()[-10:]
                            temp_chunk = " ".join(overlap_words) + " " + word
                        else:
                            temp_chunk = word
                    else:
                        temp_chunk += " " + word if temp_chunk else word

                current_chunk = temp_chunk

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Convert to DocumentChunk objects
        total_chunks = len(chunks)

        return [
            DocumentChunk(
                content=chunk,
                metadata=metadata.copy(),
                chunk_index=idx,
                total_chunks=total_chunks,
                source_file=source_file,
            )
            for idx, chunk in enumerate(chunks)
        ]

    def chunk_pdf(
        self,
        pdf_path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Chunk PDF with page-aware splitting.

        Args:
            pdf_path: Path to PDF file
            metadata: Optional metadata

        Returns:
            list[DocumentChunk]: List of chunks with page numbers
        """
        pdf_path = Path(pdf_path)
        metadata = metadata or {}

        try:
            # Lazy import to avoid circular dependency
            from aria.tools.documents.pdf_extractor import PDFExtractor

            # Extract pages with text
            pages = PDFExtractor.extract_pages(pdf_path)

            all_chunks = []
            chunk_index = 0

            for page in pages:
                # Chunk each page's text
                page_chunks = self.chunk_text(
                    text=page.text,
                    metadata=metadata,
                    source_file=str(pdf_path),
                )

                # Add page number to each chunk
                for chunk in page_chunks:
                    chunk.page_number = page.page_number
                    chunk.chunk_index = chunk_index
                    chunk_index += 1
                    all_chunks.append(chunk)

            # Update total_chunks for all chunks
            total = len(all_chunks)
            for chunk in all_chunks:
                chunk.total_chunks = total

            logger.info(f"Chunked PDF {pdf_path.name} into {total} chunks")
            return all_chunks

        except Exception as e:
            logger.error(f"Failed to chunk PDF {pdf_path}: {e}")
            return []

    def chunk_file(
        self,
        file_path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Chunk any supported file type.

        Args:
            file_path: Path to file
            metadata: Optional metadata

        Returns:
            list[DocumentChunk]: List of chunks
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        # Determine file type and chunk accordingly
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self.chunk_pdf(file_path, metadata)

        elif suffix in [".txt", ".md", ".markdown"]:
            # Read text file
            try:
                text = file_path.read_text(encoding="utf-8")
                return self.chunk_text(text, metadata, str(file_path))
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                return []

        elif suffix in [".html", ".htm"]:
            # Read HTML and extract text
            try:
                from bs4 import BeautifulSoup

                html = file_path.read_text(encoding="utf-8")
                soup = BeautifulSoup(html, "html.parser")

                # Remove script and style elements
                for element in soup(["script", "style", "nav", "header", "footer"]):
                    element.decompose()

                text = soup.get_text(separator="\n\n", strip=True)
                return self.chunk_text(text, metadata, str(file_path))
            except Exception as e:
                logger.error(f"Failed to parse HTML {file_path}: {e}")
                return []

        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return []
