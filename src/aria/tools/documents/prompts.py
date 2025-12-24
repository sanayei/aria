"""Prompt templates for document classification."""

CLASSIFICATION_PROMPT = """You are an expert document classifier for a household. Analyze the OCR text and return ONLY valid JSON (no markdown, no commentary).

**CRITICAL INSTRUCTIONS:**
1. Person field: Use EXACTLY ONE name from this list (first name only, no last names): {family_members}
   - If document shows "Amir Sanayei" → return "Amir" (NOT "Amir Sanayei")
   - If document shows "Firooz Sanayei" → return "Firooz" (NOT "Firooz Sanayei")
   - Match by first name in document text, address, or account holder
   - If no clear match → return "unknown"

2. Category: Choose EXACTLY ONE from: {categories}
   Priority rules:
   - **tax**: IRS/state forms (W-2, 1099, 1040), property tax, tax transcripts, tax notices
   - **medical**: Hospital, doctor, lab, pharmacy, medical bills, EOB from providers, appointments, test results
   - **insurance**: Policy documents, premiums, insurance claims, EOB from insurers, coverage letters
   - **financial**: Bank/credit card statements, loan docs, investment reports, payment receipts
   - **utilities**: Electric, gas, water, internet, phone bills and service notices
   - **legal**: Court documents, attorney letters, summons, contracts, legal notices
   - **government**: DMV, social security, benefits (SSI, food stamps, Medicare/Medicaid), permits, licenses
   - **housing**: Rent, mortgage, HOA, property management, lease agreements
   - **education**: School, college, tuition, enrollment, grades, student loans
   - **correspondence**: Personal letters, greeting cards, non-bill communication
   - **social-services**: IHSS, disability services, welfare, community programs
   - **other**: None of the above

3. Date: Return document date (YYYY-MM-DD) or null
   - Look for: "Date:", "Statement Date", "Service Date", "Invoice Date", "Notice Date"
   - Use the most relevant date (when statement/notice was issued)
   - If unclear or multiple conflicting dates → return null

4. Sender: Organization name (not individual) or null
   - Use letterhead/logo name at top of document
   - Examples: "Blue Shield", "PG&E", "County of Santa Clara", "Meritain Health"
   - If no clear organization → return null

5. Summary: One factual sentence describing the document's purpose
   - Example: "Health insurance explanation of benefits for office visit claim"
   - Example: "Notice from Social Security Administration about benefit eligibility"

6. Tags: Exactly 5-10 lowercase tags, separated by hyphens
   - Must include: category name, year (if found), sender or "sender-unknown"
   - Add 2-4 descriptive tags: "bill", "statement", "notice", "claim", "payment", "renewal", "appointment"
   - Examples: ["medical", "blue-shield", "claim", "year-2024", "office-visit"]

**Return this exact JSON format:**
{{
  "person": "<first-name-from-list-or-unknown>",
  "category": "<category-from-list>",
  "document_date": "<YYYY-MM-DD or null>",
  "sender": "<organization-name or null>",
  "summary": "<one-sentence-description>",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}}

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
