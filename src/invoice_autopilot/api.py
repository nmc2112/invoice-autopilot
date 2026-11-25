"""FastAPI service for the invoice automation MVP."""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .pipeline import Autopilot, Decision
from .synth import CATALOG

logging.basicConfig(level=logging.INFO, format="%(message)s")


class InvoiceIn(BaseModel):
    text: str = Field(min_length=20, max_length=20000)
    invoice_id: str | None = None


class ReviewIn(BaseModel):
    approve: bool
    account: str | None = Field(None, description="Corrected expense account, e.g. 615600")


def create_app(autopilot: Autopilot | None = None) -> FastAPI:
    if autopilot is None:
        from .factory import build_autopilot
        autopilot = build_autopilot()
    ap = autopilot
    app = FastAPI(title="Invoice Autopilot", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "examples_in_index": len(ap.classifier)}

    @app.post("/invoices", response_model=Decision)
    def process(inv: InvoiceIn) -> Decision:
        return ap.process(inv.invoice_id or f"API-{uuid.uuid4().hex[:8]}", inv.text)

    @app.get("/review")
    def review_queue() -> list[dict]:
        return [{k: v for k, v in i.items() if k != "text"} for i in ap.store.list_invoices("needs_review")]

    @app.post("/review/{invoice_id}")
    def review(invoice_id: str, body: ReviewIn) -> dict:
        if body.account and body.account not in CATALOG:
            raise HTTPException(422, f"Unknown account {body.account}")
        try:
            return {"status": ap.review(invoice_id, body.approve, body.account)}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/export/fec", response_class=PlainTextResponse)
    def export() -> str:
        return ap.export_fec()

    @app.get("/metrics")
    def metrics() -> dict:
        return ap.metrics()

    @app.get("/accounts")
    def accounts() -> dict:
        return {acc: label for acc, (label, _, _) in CATALOG.items()}

    return app
