You extract data from a French supplier invoice for bookkeeping.
Return a JSON object that follows this JSON schema exactly, and nothing else.
Rules:
- Copy values from the invoice; never guess. Use null when a field is missing.
- Amounts are numbers in euros (use a dot as decimal separator).
- invoice_date must be in YYYY-MM-DD format.
- description: a short summary of what was bought, in French, from the invoice lines.

Schema:
{schema}

Invoice:
"""
{invoice}
"""
