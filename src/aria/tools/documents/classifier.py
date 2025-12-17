"""Document classifier using LLM for intelligent categorization."""

import json
from datetime import date

from aria.config import Settings
from aria.llm import OllamaClient, ChatMessage
from aria.tools.documents.models import ClassificationResult
from aria.tools.documents.prompts import format_classification_prompt


class DocumentClassifier:
    """Uses LLM to classify documents based on OCR text.

    This classifier sends the extracted text to the LLM along with
    the list of family members and categories, and receives back
    structured classification data.
    """

    def __init__(self, client: OllamaClient, settings: Settings):
        """Initialize the classifier.

        Args:
            client: Ollama client for LLM inference
            settings: ARIA settings (for family members and categories)
        """
        self.client = client
        self.settings = settings

    async def classify(self, ocr_text: str) -> ClassificationResult:
        """Classify a document based on OCR text.

        Args:
            ocr_text: Extracted text from the document

        Returns:
            ClassificationResult with person, category, date, sender, summary

        Raises:
            ValueError: If LLM returns invalid response
            RuntimeError: If LLM call fails
        """
        # Format the prompt
        prompt = format_classification_prompt(
            ocr_text=ocr_text,
            family_members=self.settings.family_members,
            categories=self.settings.document_categories,
        )

        # Create messages for the LLM
        messages = [
            ChatMessage(
                role="user",
                content=prompt,
            )
        ]

        try:
            # Call LLM with low temperature for consistent output
            response = await self.client.chat(
                messages=messages,
                temperature=0.1,  # Low temperature for structured output
                options={"num_predict": 500},  # Limit response length
            )

            # Extract and parse JSON from response
            content = response.message.content.strip()

            # Sometimes LLM wraps JSON in markdown code blocks
            if content.startswith("```"):
                # Extract JSON from code block
                lines = content.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                content = "\n".join(json_lines)

            # Parse JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"LLM returned invalid JSON: {content[:200]}"
                ) from e

            # Validate required fields
            if "person" not in data or "category" not in data or "summary" not in data:
                raise ValueError(
                    f"LLM response missing required fields: {data}"
                )

            # Parse document_date if present
            doc_date = None
            if data.get("document_date"):
                try:
                    doc_date = date.fromisoformat(data["document_date"])
                except (ValueError, TypeError):
                    # Invalid date format, leave as None
                    pass

            return ClassificationResult(
                person=data["person"],
                category=data["category"],
                document_date=doc_date,
                sender=data.get("sender"),
                summary=data["summary"],
            )

        except Exception as e:
            raise RuntimeError(f"Document classification failed: {e}") from e
