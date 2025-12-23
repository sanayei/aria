"""PDF extraction utilities using PyMuPDF."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf  # PyMuPDF

from aria.logging import get_logger
from aria.tools.documents.models import (
    PDFComparison,
    PDFDocument,
    PDFMetadata,
    PDFPage,
    PDFSearchMatch,
    PDFTable,
)

logger = get_logger("aria.tools.documents.pdf")


class PDFError(Exception):
    """Exception raised when PDF operations fail."""

    pass


class PDFExtractor:
    """Extract content and metadata from PDF documents."""

    @staticmethod
    def extract_metadata(pdf_path: Path | str) -> PDFMetadata:
        """Extract metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDFMetadata: Extracted metadata

        Raises:
            PDFError: If PDF cannot be opened or read
        """
        try:
            pdf_path = Path(pdf_path)
            doc = pymupdf.open(pdf_path)

            metadata = doc.metadata or {}

            # Parse dates
            creation_date = None
            if metadata.get("creationDate"):
                try:
                    # PyMuPDF date format: D:YYYYMMDDHHmmSS
                    date_str = metadata["creationDate"]
                    if date_str.startswith("D:"):
                        date_str = date_str[2:16]  # YYYYMMDDHHmmSS
                        creation_date = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                except Exception:
                    pass

            mod_date = None
            if metadata.get("modDate"):
                try:
                    date_str = metadata["modDate"]
                    if date_str.startswith("D:"):
                        date_str = date_str[2:16]
                        mod_date = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                except Exception:
                    pass

            result = PDFMetadata(
                title=metadata.get("title") or None,
                author=metadata.get("author") or None,
                subject=metadata.get("subject") or None,
                keywords=metadata.get("keywords") or None,
                creator=metadata.get("creator") or None,
                producer=metadata.get("producer") or None,
                creation_date=creation_date,
                modification_date=mod_date,
                format=metadata.get("format") or None,
                encrypted=doc.is_encrypted,
            )

            doc.close()
            return result

        except Exception as e:
            logger.error(f"Failed to extract metadata from {pdf_path}: {e}")
            raise PDFError(f"Failed to extract metadata: {e}") from e

    @staticmethod
    def extract_text(
        pdf_path: Path | str,
        pages: list[int] | None = None,
    ) -> str:
        """Extract all text from PDF.

        Args:
            pdf_path: Path to PDF file
            pages: Specific page numbers to extract (0-indexed), None for all

        Returns:
            str: Extracted text

        Raises:
            PDFError: If PDF cannot be read
        """
        try:
            pdf_path = Path(pdf_path)
            doc = pymupdf.open(pdf_path)

            text_parts = []

            page_range = pages if pages else range(len(doc))

            for page_num in page_range:
                if page_num >= len(doc):
                    logger.warning(f"Page {page_num} doesn't exist in PDF")
                    continue

                page = doc[page_num]
                text = page.get_text()
                text_parts.append(text)

            doc.close()

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise PDFError(f"Failed to extract text: {e}") from e

    @staticmethod
    def extract_pages(pdf_path: Path | str) -> list[PDFPage]:
        """Extract detailed information about each page.

        Args:
            pdf_path: Path to PDF file

        Returns:
            list[PDFPage]: List of page information

        Raises:
            PDFError: If PDF cannot be read
        """
        try:
            pdf_path = Path(pdf_path)
            doc = pymupdf.open(pdf_path)

            pages = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                rect = page.rect

                # Count images
                image_list = page.get_images()

                pages.append(
                    PDFPage(
                        page_number=page_num + 1,  # 1-indexed for user display
                        text=page.get_text(),
                        width=rect.width,
                        height=rect.height,
                        rotation=page.rotation,
                        images_count=len(image_list),
                    )
                )

            doc.close()
            return pages

        except Exception as e:
            logger.error(f"Failed to extract pages from {pdf_path}: {e}")
            raise PDFError(f"Failed to extract pages: {e}") from e

    @staticmethod
    def extract_tables(
        pdf_path: Path | str,
        pages: list[int] | None = None,
    ) -> list[PDFTable]:
        """Extract tables from PDF.

        Note: Basic table extraction using text positioning.
        For complex tables, consider using specialized libraries like pdfplumber.

        Args:
            pdf_path: Path to PDF file
            pages: Specific page numbers (0-indexed), None for all

        Returns:
            list[PDFTable]: Extracted tables

        Raises:
            PDFError: If PDF cannot be read
        """
        try:
            pdf_path = Path(pdf_path)
            doc = pymupdf.open(pdf_path)

            tables = []
            page_range = pages if pages else range(len(doc))

            for page_num in page_range:
                if page_num >= len(doc):
                    continue

                page = doc[page_num]

                # Extract tables using PyMuPDF's table detection
                # Note: This is basic - for better results use pdfplumber
                tabs = page.find_tables()

                for tab_idx, tab in enumerate(tabs):
                    # Extract table data
                    table_data = []
                    for row in tab.extract():
                        table_data.append([str(cell) if cell else "" for cell in row])

                    if not table_data:
                        continue

                    # Try to detect headers (first row)
                    headers = None
                    if len(table_data) > 1:
                        # Simple heuristic: if first row cells are short and distinct
                        first_row = table_data[0]
                        if all(len(cell) < 50 for cell in first_row):
                            headers = first_row
                            table_data = table_data[1:]

                    tables.append(
                        PDFTable(
                            page_number=page_num + 1,
                            table_number=tab_idx + 1,
                            rows=len(table_data),
                            columns=len(table_data[0]) if table_data else 0,
                            headers=headers,
                            data=table_data,
                            bbox=tab.bbox if hasattr(tab, "bbox") else None,
                        )
                    )

            doc.close()
            return tables

        except Exception as e:
            logger.error(f"Failed to extract tables from {pdf_path}: {e}")
            raise PDFError(f"Failed to extract tables: {e}") from e

    @staticmethod
    def search_text(
        pdf_path: Path | str,
        query: str,
        case_sensitive: bool = False,
        context_chars: int = 50,
    ) -> list[PDFSearchMatch]:
        """Search for text in PDF.

        Args:
            pdf_path: Path to PDF file
            query: Text to search for
            case_sensitive: Whether search should be case-sensitive
            context_chars: Number of characters to include before/after match

        Returns:
            list[PDFSearchMatch]: List of matches

        Raises:
            PDFError: If PDF cannot be read
        """
        try:
            pdf_path = Path(pdf_path)
            doc = pymupdf.open(pdf_path)

            matches = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                # Search for matches
                if case_sensitive:
                    pattern = re.escape(query)
                else:
                    pattern = re.escape(query)
                    text_to_search = text
                    text = text.lower()
                    query_lower = query.lower()
                    pattern = re.escape(query_lower)

                for match in re.finditer(pattern, text if not case_sensitive else text):
                    start = match.start()
                    end = match.end()

                    # Get context
                    context_start = max(0, start - context_chars)
                    context_end = min(len(text), end + context_chars)

                    context_before = text[context_start:start].strip()
                    context_after = text[end:context_end].strip()

                    # Get matched text from original (case-preserved)
                    if not case_sensitive:
                        matched_text = text_to_search[start:end]
                    else:
                        matched_text = text[start:end]

                    matches.append(
                        PDFSearchMatch(
                            page_number=page_num + 1,
                            text=matched_text,
                            context_before=context_before,
                            context_after=context_after,
                        )
                    )

            doc.close()
            return matches

        except Exception as e:
            logger.error(f"Failed to search in {pdf_path}: {e}")
            raise PDFError(f"Failed to search PDF: {e}") from e

    @staticmethod
    def analyze_pdf(
        pdf_path: Path | str,
        extract_pages_detail: bool = False,
        extract_tables: bool = False,
    ) -> PDFDocument:
        """Perform complete PDF analysis.

        Args:
            pdf_path: Path to PDF file
            extract_pages_detail: Whether to extract detailed page info
            extract_tables: Whether to extract tables

        Returns:
            PDFDocument: Complete document analysis

        Raises:
            PDFError: If PDF cannot be analyzed
        """
        try:
            pdf_path = Path(pdf_path)

            if not pdf_path.exists():
                raise PDFError(f"PDF file not found: {pdf_path}")

            # Get file size
            file_size = pdf_path.stat().st_size

            # Extract metadata
            metadata = PDFExtractor.extract_metadata(pdf_path)

            # Open document to get page count
            doc = pymupdf.open(pdf_path)
            page_count = len(doc)
            is_encrypted = doc.is_encrypted
            doc.close()

            # Extract full text
            full_text = PDFExtractor.extract_text(pdf_path)

            # Extract pages detail if requested
            pages = []
            if extract_pages_detail:
                pages = PDFExtractor.extract_pages(pdf_path)

            # Extract tables if requested
            tables = []
            if extract_tables:
                tables = PDFExtractor.extract_tables(pdf_path)

            return PDFDocument(
                file_path=str(pdf_path),
                metadata=metadata,
                page_count=page_count,
                pages=pages,
                full_text=full_text,
                tables=tables,
                file_size_bytes=file_size,
                is_encrypted=is_encrypted,
            )

        except PDFError:
            raise
        except Exception as e:
            logger.error(f"Failed to analyze PDF {pdf_path}: {e}")
            raise PDFError(f"Failed to analyze PDF: {e}") from e

    @staticmethod
    def compare_pdfs(
        pdf_path_1: Path | str,
        pdf_path_2: Path | str,
        comparison_type: str = "text",
    ) -> PDFComparison:
        """Compare two PDF documents.

        Args:
            pdf_path_1: Path to first PDF
            pdf_path_2: Path to second PDF
            comparison_type: Type of comparison ("text", "structure", or "full")

        Returns:
            PDFComparison: Comparison results

        Raises:
            PDFError: If PDFs cannot be compared
        """
        try:
            pdf_path_1 = Path(pdf_path_1)
            pdf_path_2 = Path(pdf_path_2)

            doc1 = pymupdf.open(pdf_path_1)
            doc2 = pymupdf.open(pdf_path_2)

            pages_differ = []
            structural_differences = []

            # Check page counts
            if len(doc1) != len(doc2):
                structural_differences.append(f"Page count differs: {len(doc1)} vs {len(doc2)}")

            # Determine added/removed pages
            added_pages = []
            removed_pages = []

            if len(doc2) > len(doc1):
                added_pages = list(range(len(doc1) + 1, len(doc2) + 1))
            elif len(doc1) > len(doc2):
                removed_pages = list(range(len(doc2) + 1, len(doc1) + 1))

            # Compare common pages
            common_pages = min(len(doc1), len(doc2))

            matching_chars = 0
            total_chars = 0

            for page_num in range(common_pages):
                page1 = doc1[page_num]
                page2 = doc2[page_num]

                text1 = page1.get_text()
                text2 = page2.get_text()

                # Text comparison
                if text1 != text2:
                    pages_differ.append(page_num + 1)

                # Calculate similarity for this page
                max_len = max(len(text1), len(text2))
                if max_len > 0:
                    # Simple character-level similarity
                    matches = sum(c1 == c2 for c1, c2 in zip(text1, text2))
                    matching_chars += matches
                    total_chars += max_len

                # Structure comparison
                if comparison_type in ["structure", "full"]:
                    if page1.rect != page2.rect:
                        structural_differences.append(f"Page {page_num + 1}: Different dimensions")

            # Calculate overall text similarity
            text_similarity = matching_chars / total_chars if total_chars > 0 else 1.0

            doc1.close()
            doc2.close()

            return PDFComparison(
                file1=str(pdf_path_1),
                file2=str(pdf_path_2),
                pages_differ=pages_differ,
                text_similarity=text_similarity,
                structural_differences=structural_differences,
                added_pages=added_pages,
                removed_pages=removed_pages,
            )

        except Exception as e:
            logger.error(f"Failed to compare PDFs: {e}")
            raise PDFError(f"Failed to compare PDFs: {e}") from e
