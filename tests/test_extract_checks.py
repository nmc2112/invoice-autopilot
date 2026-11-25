import json

from conftest import truth_json

from invoice_autopilot.checks import run_checks
from invoice_autopilot.extract import InvoiceData, extract_invoice
from invoice_autopilot.llm import FakeLLM
from invoice_autopilot.synth import generate_invoices

INV = next(i for i in generate_invoices(20, seed=5) if i.expected_issue is None)


def test_extraction_valid_first_try(settings):
    res = extract_invoice(INV.text, FakeLLM([truth_json(INV)]), settings)
    assert res.data is not None and res.attempts == 1
    assert res.data.amount_ttc == INV.amount_ttc


def test_extraction_retries_after_invalid_json(settings):
    llm = FakeLLM(['{"supplier": "X"}', truth_json(INV)])
    res = extract_invoice(INV.text, llm, settings)
    assert res.data is not None and res.attempts == 2
    assert "rejected" in llm.prompts[1]


def test_extraction_gives_up(settings):
    res = extract_invoice(INV.text, FakeLLM(["nope", "still nope"]), settings)
    assert res.data is None and res.error


def test_prompt_contains_invoice_and_schema(settings):
    llm = FakeLLM([truth_json(INV)])
    extract_invoice(INV.text, llm, settings)
    assert INV.invoice_number in llm.prompts[0] and "amount_ttc" in llm.prompts[0]


def data(**over) -> InvoiceData:
    base = json.loads(truth_json(INV))
    base.update(over)
    return InvoiceData.model_validate(base)


def test_consistent_invoice_passes(settings):
    assert run_checks(data(), settings, set()) == []


def test_checks_detect_problems(settings):
    wrong_total = data(amount_ttc=INV.amount_ttc + 10)
    assert any(i.startswith("amount_mismatch") for i in run_checks(wrong_total, settings, set()))
    assert any(i.startswith("invalid_vat_rate") for i in run_checks(data(vat_rate=7.0), settings, set()))
    big = data(amount_ht=20000.0, vat_amount=4000.0, amount_ttc=24000.0, vat_rate=20.0)
    assert any(i.startswith("above_limit") for i in run_checks(big, settings, set()))
    key = {(INV.supplier.lower(), INV.invoice_number.lower())}
    assert any(i.startswith("duplicate") for i in run_checks(data(), settings, key))
