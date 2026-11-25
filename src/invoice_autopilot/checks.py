"""Deterministic control checks run before anything is posted automatically."""

from __future__ import annotations

from .config import Settings
from .extract import InvoiceData


def run_checks(data: InvoiceData, settings: Settings, already_posted: set[tuple[str, str]]) -> list[str]:
    """Return a list of issues; an empty list means the invoice is consistent."""
    c = settings.checks
    issues: list[str] = []
    if abs(data.amount_ht + data.vat_amount - data.amount_ttc) > c.amount_tolerance:
        issues.append("amount_mismatch: HT + TVA != TTC")
    if not any(abs(data.vat_rate - r) < 1e-6 for r in c.vat_rates):
        issues.append(f"invalid_vat_rate: {data.vat_rate}")
    elif abs(data.amount_ht * data.vat_rate / 100 - data.vat_amount) > c.amount_tolerance + 0.01:
        issues.append("vat_mismatch: TVA amount does not match rate")
    if data.amount_ttc > c.max_auto_post_amount:
        issues.append(f"above_limit: {data.amount_ttc:.2f} EUR > {c.max_auto_post_amount:.0f} EUR")
    if (data.supplier.lower(), data.invoice_number.lower()) in already_posted:
        issues.append("duplicate: same supplier and invoice number already recorded")
    return issues
