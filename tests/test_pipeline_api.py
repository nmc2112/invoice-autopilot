from conftest import oracle
from fastapi.testclient import TestClient

from invoice_autopilot.api import create_app
from invoice_autopilot.llm import FakeLLM
from invoice_autopilot.synth import generate_invoices

STREAM = generate_invoices(40, seed=21)


def test_most_clean_invoices_are_auto_posted_correctly(make_autopilot):
    ap = make_autopilot(oracle(STREAM))
    clean = [i for i in STREAM if i.expected_issue is None]
    decisions = [(i, ap.process(i.invoice_id, i.text)) for i in clean]
    auto = [(i, d) for i, d in decisions if d.status == "auto_posted"]
    assert len(auto) / len(clean) > 0.6
    assert all(d.predicted_account == i.account for i, d in auto)
    lines = ap.store.entry_lines()
    assert sum(x["Debit"] for x in lines) == sum(x["Credit"] for x in lines)


def test_problem_invoices_go_to_review(make_autopilot):
    ap = make_autopilot(oracle(STREAM))
    statuses = {i.expected_issue: ap.process(i.invoice_id, i.text).status for i in STREAM}
    for issue, status in statuses.items():
        if issue:
            assert status == "needs_review", issue


def test_extraction_failure_goes_to_review(make_autopilot):
    ap = make_autopilot(FakeLLM(["bad", "bad"]))
    d = ap.process("X1", STREAM[0].text)
    assert d.status == "needs_review" and d.issues[0].startswith("extraction_failed")


def test_empty_index_needs_review(make_autopilot):
    ap = make_autopilot(oracle(STREAM), seed_history=False)
    inv = STREAM[0]
    assert "no_examples" in ap.process(inv.invoice_id, inv.text).issues[0]


def test_review_posts_and_learns(make_autopilot):
    ap = make_autopilot(oracle(STREAM), seed_history=False)
    inv = next(i for i in STREAM if i.expected_issue is None)
    ap.process(inv.invoice_id, inv.text)
    before = len(ap.classifier)
    assert ap.review(inv.invoice_id, approve=True, account=inv.account) == "posted_after_review"
    assert len(ap.classifier) == before + 1
    assert ap.store.get_invoice(inv.invoice_id)["final_account"] == inv.account


def test_review_twice_is_refused(make_autopilot):
    ap = make_autopilot(oracle(STREAM), seed_history=False)
    inv = STREAM[0]
    ap.process(inv.invoice_id, inv.text)
    ap.review(inv.invoice_id, approve=False)
    import pytest
    with pytest.raises(ValueError):
        ap.review(inv.invoice_id, approve=True, account=inv.account)


def test_api_flow(make_autopilot):
    ap = make_autopilot(oracle(STREAM))
    client = TestClient(create_app(ap))
    assert client.get("/health").json()["status"] == "ok"
    for inv in STREAM[:15]:
        r = client.post("/invoices", json={"text": inv.text, "invoice_id": inv.invoice_id})
        assert r.status_code == 200
    queue = client.get("/review").json()
    if queue:
        assert client.post(f"/review/{queue[0]['id']}", json={"approve": True, "account": "999999"}).status_code == 422
        assert client.post(f"/review/{queue[0]['id']}", json={"approve": True}).status_code == 200
    assert "EcritureNum" in client.get("/export/fec").text
    assert client.get("/metrics").json()["processed"] == 15
