import pytest

from invoice_autopilot.classify import AccountClassifier, example_text
from invoice_autopilot.config import LedgerConfig
from invoice_autopilot.extract import InvoiceData
from invoice_autopilot.ledger import FEC_COLUMNS, entry_lines, export_fec
from invoice_autopilot.llm import HashingEmbedder
from invoice_autopilot.synth import generate_invoices


def test_classifier_accuracy_on_synthetic(history):
    clf = AccountClassifier(HashingEmbedder(), k=5)
    clf.add([example_text(i.supplier, i.description) for i in history], [i.account for i in history])
    test = [i for i in generate_invoices(80, seed=11) if i.expected_issue is None]
    acc = sum(clf.predict(example_text(i.supplier, i.description)).account == i.account for i in test) / len(test)
    assert acc > 0.85


def test_empty_classifier_raises():
    with pytest.raises(ValueError):
        AccountClassifier(HashingEmbedder()).predict("anything")


def test_learning_adds_examples():
    clf = AccountClassifier(HashingEmbedder(), k=1)
    clf.add(["Fournisseur A | loyer bureaux"], ["613200"])
    clf.add(["Fournisseur B | maintenance ascenseur"], ["615600"])
    assert clf.predict("Fournisseur B | maintenance ascenseur").account == "615600"
    assert len(clf) == 2


def test_entry_is_balanced_and_exports_fec():
    data = InvoiceData(supplier="TechMaint", invoice_number="T-1", invoice_date="2025-03-01", amount_ht=100.0,
                       vat_rate=20.0, vat_amount=20.0, amount_ttc=120.0, description="Maintenance")
    lines = entry_lines(data, "615600", "000001", LedgerConfig(), "20250302")
    assert sum(line["Debit"] for line in lines) == pytest.approx(sum(line["Credit"] for line in lines))
    assert [line["CompteNum"] for line in lines] == ["615600", "445660", "401000"]
    text = export_fec(lines)
    assert text.splitlines()[0].split("\t") == FEC_COLUMNS
    assert "120,00" in text
