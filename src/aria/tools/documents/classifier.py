"""Document classifier using LLM for intelligent categorization."""

import asyncio
import json
import re
from datetime import date

from aria.config import Settings
from aria.llm import OllamaClient, ChatMessage
from aria.logging import get_logger
from aria.tools.documents.models import ClassificationResult
from aria.tools.documents.prompts import format_classification_prompt

logger = get_logger("aria.tools.documents.classifier")


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
        self.classification_model = settings.classification_model

    def _prepare_classification_input(self, ocr_text: str, max_chars: int = 3000) -> str:
        """Prepare OCR text for classification by intelligently truncating.

        Strategy:
        - First page usually has: sender, recipient, subject, date
        - Last page usually has: signatures, totals, summary
        - Middle pages often repetitive

        Args:
            ocr_text: Full OCR extracted text
            max_chars: Maximum characters to send to LLM

        Returns:
            Truncated text optimized for classification
        """
        if len(ocr_text) <= max_chars:
            return ocr_text

        # Calculate section sizes
        first_size = int(max_chars * 0.5)  # 50% from start
        last_size = int(max_chars * 0.35)  # 35% from end
        middle_size = max_chars - first_size - last_size  # 15% from middle

        # Extract sections
        first_part = ocr_text[:first_size]
        last_part = ocr_text[-last_size:]

        # Middle sample (for continuity check)
        middle_start = len(ocr_text) // 2 - middle_size // 2
        middle_part = ocr_text[middle_start : middle_start + middle_size]

        # Combine with markers
        truncated = f"""{first_part}

[... content truncated for classification ...]

{middle_part}

[... content truncated ...]

{last_part}"""

        logger.debug(
            "Prepared classification input",
            original_length=len(ocr_text),
            truncated_length=len(truncated),
            reduction_percent=round((1 - len(truncated) / len(ocr_text)) * 100, 1),
        )

        return truncated

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response, handling various formats.

        Handles:
        - Plain JSON
        - Markdown code blocks (```json ... ```)
        - JSON embedded in explanatory text

        Args:
            response: Raw LLM response

        Returns:
            Extracted JSON string
        """
        # Try direct parsing first
        response = response.strip()
        if response.startswith("{"):
            return response

        # Try extracting from markdown code block
        json_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        matches = re.findall(json_block_pattern, response, re.DOTALL)
        if matches:
            return matches[0]

        # Try finding any JSON object
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            # Return the largest JSON object found
            return max(matches, key=len)

        # If all else fails, return original
        return response

    async def classify(self, ocr_text: str, max_retries: int = 3) -> ClassificationResult:
        """Classify a document based on OCR text with retry logic.

        Args:
            ocr_text: Extracted text from the document
            max_retries: Maximum number of retry attempts

        Returns:
            ClassificationResult with person, category, date, sender, summary

        Raises:
            RuntimeError: If all retries fail
        """
        # Prepare input (with smart truncation)
        classification_input = self._prepare_classification_input(ocr_text)

        last_error = None

        for attempt in range(max_retries):
            try:
                # Adjust temperature on retries for variety
                temperature = 0.1 if attempt == 0 else 0.3

                # Format the prompt
                prompt = format_classification_prompt(
                    ocr_text=classification_input,
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

                logger.debug(
                    "Calling LLM for classification",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    model=self.classification_model,
                    temperature=temperature,
                )

                # Call LLM with JSON format enforcement and fast model
                response = await self.client.chat(
                    messages=messages,
                    model=self.classification_model,  # Use fast model
                    temperature=temperature,
                    format="json",  # Force JSON-only output
                    options={"num_predict": 500},  # Limit response length
                )

                # Extract and parse JSON from response
                content = response.message.content.strip()

                # Use robust JSON extraction
                json_text = self._extract_json_from_response(content)

                # Parse JSON
                try:
                    data = json.loads(json_text)
                except json.JSONDecodeError as e:
                    raise ValueError(f"LLM returned invalid JSON: {json_text[:200]}") from e

                # Validate required fields
                required_fields = {"person", "category", "summary"}
                if not required_fields.issubset(data.keys()):
                    missing = required_fields - data.keys()
                    raise ValueError(f"Missing required fields: {missing}")

                # Parse document_date if present
                doc_date = None
                if data.get("document_date"):
                    try:
                        doc_date = date.fromisoformat(data["document_date"])
                    except (ValueError, TypeError):
                        # Invalid date format, leave as None
                        pass

                # Parse tags (ensure it's a list)
                tags = data.get("tags", [])
                if not isinstance(tags, list):
                    tags = []
                # Normalize tags: lowercase, strip whitespace
                tags = [tag.strip().lower() for tag in tags if isinstance(tag, str)]

                # Success!
                logger.info(
                    "Document classified successfully",
                    person=data["person"],
                    category=data["category"],
                    attempt=attempt + 1,
                )

                return ClassificationResult(
                    person=data["person"],
                    category=data["category"],
                    document_date=doc_date,
                    sender=data.get("sender"),
                    summary=data["summary"],
                    tags=tags,
                )

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = e
                logger.warning(
                    "Classification attempt failed",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                )

                if attempt < max_retries - 1:
                    # Brief pause before retry
                    await asyncio.sleep(1)
                    continue

        # All retries failed
        raise RuntimeError(
            f"Document classification failed after {max_retries} attempts: {last_error}"
        )
