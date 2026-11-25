"""End-to-end automation: extract -> check -> classify -> auto-post or send to review."""

from __future__ import annotations

import logging
import time
from datetime import date

from pydantic import BaseModel

from .checks import run_checks
from .classify import AccountClassifier, example_text
from .config import Settings
from .extract import InvoiceData, extract_invoice
from .ledger import entry_lines, export_fec
from .llm import ChatModel, Embedder
from .store import Store

log = logging.getLogger("invoice_autopilot.pipeline")


class Decision(BaseModel):
    invoice_id: str
    status: str
    data: InvoiceData | None
    predicted_account: str | None
    confidence: float | None
    issues: list[str]
    latency_s: float


class Autopilot:
    def __init__(self, settings: Settings, llm: ChatModel, embedder: Embedder, store: Store):
        self.settings, self.llm, self.store = settings, llm, store
        self.classifier = AccountClassifier(embedder, k=settings.classifier.k)
        stored = store.examples()
        if stored:
            self.classifier.add([t for t, _ in stored], [a for _, a in stored])

    def seed_history(self, examples: list[tuple[str, str]]) -> None:
        """Load labelled history (e.g. last year's booked invoices) into the index."""
        for text, account in examples:
            self.store.add_example(text, account, "history")
        self.classifier.add([t for t, _ in examples], [a for _, a in examples])

    def _post(self, invoice_id: str, data: InvoiceData, account: str) -> None:
        num = self.store.next_entry_num()
        lines = entry_lines(data, account, num, self.settings.ledger, date.today().strftime("%Y%m%d"))
        self.store.add_entry_lines(invoice_id, lines)

    def process(self, invoice_id: str, text: str) -> Decision:
        start = time.perf_counter()
        extraction = extract_invoice(text, self.llm, self.settings)
        issues: list[str] = []
        account, confidence = None, None

        if extraction.data is None:
            issues.append(f"extraction_failed: {extraction.error}")
        else:
            data = extraction.data
            issues += run_checks(data, self.settings, self.store.recorded_keys())
            if len(self.classifier):
                pred = self.classifier.predict(example_text(data.supplier, data.description))
                account, confidence = pred.account, pred.confidence
                if confidence < self.settings.classifier.auto_post_threshold:
                    issues.append(f"low_confidence: {confidence:.2f} for account {account}")
            else:
                issues.append("no_examples: classifier is empty")

        status = "auto_posted" if not issues else "needs_review"
        self.store.save_invoice({
            "id": invoice_id, "text": text, "status": status,
            "data": extraction.data.model_dump_json() if extraction.data else None,
            "predicted_account": account, "confidence": confidence,
            "final_account": account if status == "auto_posted" else None, "issues": issues,
        })
        if status == "auto_posted" and extraction.data and account:
            self._post(invoice_id, extraction.data, account)
        decision = Decision(invoice_id=invoice_id, status=status, data=extraction.data,
                            predicted_account=account, confidence=confidence, issues=issues,
                            latency_s=round(time.perf_counter() - start, 3))
        log.info(decision.model_dump_json(exclude={"data"}))
        return decision

    def review(self, invoice_id: str, approve: bool, account: str | None = None,
               corrected: InvoiceData | None = None) -> str:
        """Human decision. Approved invoices are posted and become new training examples."""
        inv = self.store.get_invoice(invoice_id)
        if inv is None or inv["status"] != "needs_review":
            raise ValueError(f"Invoice {invoice_id} is not waiting for review")
        if not approve:
            self.store.update_status(invoice_id, "rejected", None)
            return "rejected"
        data = corrected or (InvoiceData.model_validate(inv["data"]) if inv["data"] else None)
        final_account = account or inv["predicted_account"]
        if data is None or final_account is None:
            raise ValueError("Approving requires extracted data and an account")
        self._post(invoice_id, data, final_account)
        self.store.update_status(invoice_id, "posted_after_review", final_account)
        text = example_text(data.supplier, data.description)
        self.store.add_example(text, final_account, "review")
        self.classifier.add([text], [final_account])   # learning loop
        return "posted_after_review"

    def export_fec(self) -> str:
        return export_fec(self.store.entry_lines())

    def metrics(self) -> dict:
        counts = self.store.counts()
        processed = sum(counts.values()) or 1
        return {"processed": sum(counts.values()), **counts,
                "automation_rate": round(counts.get("auto_posted", 0) / processed, 3),
                "examples_in_index": len(self.classifier)}
