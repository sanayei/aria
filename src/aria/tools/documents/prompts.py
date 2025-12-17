"""Prompt templates for document classification."""

CLASSIFICATION_PROMPT = """You are classifying a scanned document. Based on the OCR text below, extract the following information.

**Family Members**: {family_members}
**Document Categories**: {categories}

Extract:
1. **person**: Which family member does this document belong to? Look for names, addresses, or account holders mentioned. If unclear, use "unknown".

2. **category**: What type of document is this? Choose from the categories above.

3. **document_date**: The date ON the document (not today's date). Format: YYYY-MM-DD. Use null if not clearly visible.

4. **sender**: The organization/company that sent this document. Examples: "Kaiser Permanente", "IRS", "PG&E", "Bank of America". Use null if unclear.

5. **summary**: One concise sentence describing what this document is. Examples:
   - "Electricity bill for January 2025 showing $142.50 due by Feb 15"
   - "Lab results from annual checkup showing normal values"
   - "Report card for Fall 2024 semester"

Respond with ONLY valid JSON in this exact format:
{{
    "person": "string",
    "category": "string",
    "document_date": "YYYY-MM-DD or null",
    "sender": "string or null",
    "summary": "string"
}}

---
DOCUMENT TEXT:
{ocr_text}
---"""


def format_classification_prompt(
    ocr_text: str,
    family_members: list[str],
    categories: list[str],
) -> str:
    """Format the classification prompt with document text and options.

    Args:
        ocr_text: Extracted text from OCR
        family_members: List of family member names
        categories: List of document categories

    Returns:
        Formatted prompt string
    """
    return CLASSIFICATION_PROMPT.format(
        family_members=", ".join(family_members),
        categories=", ".join(categories),
        ocr_text=ocr_text[:10000],  # Limit to prevent token overflow
    )
