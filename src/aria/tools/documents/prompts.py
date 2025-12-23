"""Prompt templates for document classification."""

CLASSIFICATION_PROMPT = """Classify this document. Return ONLY valid JSON (no text before or after).

Family members: {family_members}
Categories: {categories}

Extract and return JSON:
- person: Which family member (name/address match). Use "unknown" if unclear.
- category: Type from categories list above.
- document_date: Date on document (YYYY-MM-DD) or null.
- sender: Organization/company name or null.
- summary: One sentence describing the document.
- tags: 5-10 lowercase hyphenated tags (type, sender, subject, person, year).

DOCUMENT TEXT:
{ocr_text}"""


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
