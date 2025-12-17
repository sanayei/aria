"""Document processor orchestrating the full processing pipeline."""

import re
from datetime import date
from pathlib import Path

from aria.config import Settings
from aria.tools.documents.models import ProcessedDocument, ClassificationResult
from aria.tools.documents.classifier import DocumentClassifier
from aria.tools.documents.ocr import OCRTool


class DocumentProcessor:
    """Orchestrates the document processing pipeline.

    This class coordinates OCR extraction, LLM classification,
    filename generation, and destination path construction.
    """

    def __init__(
        self,
        ocr_tool: OCRTool,
        classifier: DocumentClassifier,
        settings: Settings,
    ):
        """Initialize the processor.

        Args:
            ocr_tool: OCR tool for text extraction
            classifier: Document classifier using LLM
            settings: ARIA settings for output paths
        """
        self.ocr_tool = ocr_tool
        self.classifier = classifier
        self.settings = settings

    async def process_document(self, source_path: Path) -> ProcessedDocument:
        """Process a scanned document through the full pipeline.

        Pipeline steps:
        1. Run OCR to extract text
        2. Use LLM to classify (person, category, date, sender, summary)
        3. Generate suggested filename
        4. Construct destination path

        Args:
            source_path: Path to the PDF or image file

        Returns:
            ProcessedDocument with all extracted information

        Raises:
            RuntimeError: If any step of the pipeline fails
        """
        # Step 1: OCR extraction
        from aria.tools.documents.ocr import OCRParams

        ocr_params = OCRParams(file_path=str(source_path))
        ocr_result = await self.ocr_tool.execute(ocr_params)

        if not ocr_result.success:
            raise RuntimeError(f"OCR failed: {ocr_result.error}")

        ocr_data = ocr_result.data
        text_content = ocr_data["text"]
        confidence = ocr_data["confidence"]

        # Step 2: LLM classification
        classification = await self.classifier.classify(text_content)

        # Step 3: Generate filename
        filename = self.generate_filename(classification, source_path.name)

        # Step 4: Get destination path
        destination = self.get_destination_path(
            classification.person,
            classification.category,
            filename,
        )

        return ProcessedDocument(
            source_path=source_path,
            text_content=text_content,
            person=classification.person,
            category=classification.category,
            document_date=classification.document_date,
            sender=classification.sender,
            summary=classification.summary,
            confidence=confidence,
            suggested_filename=filename,
            suggested_destination=destination,
            metadata={
                "page_count": ocr_data.get("page_count", 1),
                "char_count": ocr_data.get("char_count", 0),
            },
        )

    def generate_filename(
        self,
        classification: ClassificationResult,
        original_name: str,
    ) -> str:
        """Generate a descriptive filename for the document.

        Format: {YYYY-MM-DD}_{sender_slug}_{category}.pdf

        Examples:
        - 2025-01-15_kaiser_permanente_medical.pdf
        - 2025-01-10_lincoln_elementary_education.pdf
        - 2025-01-20_pg_and_e_utilities.pdf

        If no date found, use today's date.
        Sender name is slugified (lowercase, underscores, no special chars).

        Args:
            classification: Classification result with metadata
            original_name: Original filename (for extension)

        Returns:
            Generated filename
        """
        # Use document date if available, otherwise use today
        doc_date = classification.document_date or date.today()
        date_str = doc_date.isoformat()

        # Slugify sender name
        if classification.sender:
            sender_slug = self._slugify(classification.sender)
        else:
            sender_slug = "unknown"

        # Get file extension from original
        ext = Path(original_name).suffix or ".pdf"

        return f"{date_str}_{sender_slug}_{classification.category}{ext}"

    def get_destination_path(
        self,
        person: str,
        category: str,
        filename: str,
    ) -> Path:
        """Construct the full destination path.

        Format: {output_dir}/{person_lower}/{category}/{filename}

        Example: ~/documents/amir/medical/2025-01-15_kaiser_permanente_medical.pdf

        Args:
            person: Family member name
            category: Document category
            filename: Generated filename

        Returns:
            Full destination path
        """
        return (
            self.settings.documents_output_dir
            / person.lower()
            / category
            / filename
        )

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to a filename-safe slug.

        Examples:
        - "Kaiser Permanente" -> "kaiser_permanente"
        - "PG&E" -> "pg_and_e"
        - "Bank of America" -> "bank_of_america"

        Args:
            text: Text to slugify

        Returns:
            Slugified text
        """
        # Convert to lowercase
        slug = text.lower()

        # Replace & with "_and_" to preserve word boundaries
        slug = slug.replace("&", "_and_")

        # Replace spaces and special chars with underscores
        slug = re.sub(r"[^a-z0-9]+", "_", slug)

        # Remove leading/trailing underscores
        slug = slug.strip("_")

        # Collapse multiple underscores
        slug = re.sub(r"_+", "_", slug)

        return slug
