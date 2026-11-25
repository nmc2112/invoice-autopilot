"""LLM extraction of invoice fields into a validated Pydantic model."""

from __future__ import annotations

import json
from datetime import date

from pydantic import BaseModel, Field, ValidationError, field_validator

from .config import Settings
from .llm import ChatModel, LLMError


class InvoiceData(BaseModel):
    supplier: str = Field(min_length=2)
    invoice_number: str = Field(min_length=1)
    invoice_date: date
    amount_ht: float = Field(ge=0, description="Total before VAT, EUR")
    vat_rate: float = Field(ge=0, le=100, description="VAT rate in percent, e.g. 20")
    vat_amount: float = Field(ge=0, description="VAT amount, EUR")
    amount_ttc: float = Field(ge=0, description="Total including VAT, EUR")
    description: str = Field(min_length=2, description="What was bought, short, in French")

    @field_validator("supplier", "invoice_number", "description")
    @classmethod
    def strip(cls, v: str) -> str:
        return v.strip()


class ExtractionResult(BaseModel):
    data: InvoiceData | None = None
    attempts: int = 0
    error: str | None = None


def extract_invoice(text: str, llm: ChatModel, settings: Settings, max_attempts: int = 2) -> ExtractionResult:
    schema = json.dumps(InvoiceData.model_json_schema(), ensure_ascii=False)
    prompt = settings.prompt("extract").replace("{schema}", schema).replace("{invoice}", text)
    error = None
    for attempt in range(1, max_attempts + 1):
        try:
            retry_note = f"\n\nYour previous answer was rejected: {error}\nReturn corrected JSON only."
            raw = llm.complete_json(prompt if error is None else prompt + retry_note)
        except LLMError as exc:
            return ExtractionResult(attempts=attempt, error=f"LLM unavailable: {exc}")
        try:
            return ExtractionResult(data=InvoiceData.model_validate_json(raw), attempts=attempt)
        except ValidationError as exc:
            error = str(exc)[:500]
    return ExtractionResult(attempts=max_attempts, error=error)
