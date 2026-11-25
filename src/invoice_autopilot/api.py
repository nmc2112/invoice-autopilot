"""FastAPI service for the invoice automation MVP."""

from __future__ import annotations

import logging
import os
import random
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .classify import example_text
from .pipeline import Autopilot, Decision
from .ratelimit import RateLimitMiddleware, rate_limit_from_env
from .synth import CATALOG, generate_invoices

logging.basicConfig(level=logging.INFO, format="%(message)s")


class InvoiceIn(BaseModel):
    text: str = Field(min_length=20, max_length=20000)
    invoice_id: str | None = None


class ReviewIn(BaseModel):
    approve: bool
    account: str | None = Field(None, description="Corrected expense account, e.g. 615600")


logger = logging.getLogger("invoice_autopilot")


def autoseed(ap: Autopilot) -> None:
    """Demo deployments start empty: load synthetic labelled history if AUTOSEED_HISTORY is set.

    Never fatal: if the embedding API fails, the service still starts (invoices go to review)."""
    n = int(os.environ.get("AUTOSEED_HISTORY", "0") or 0)
    if n > 0 and len(ap.classifier) == 0:
        history = [i for i in generate_invoices(n, seed=999) if i.expected_issue is None]
        try:
            ap.seed_history([(example_text(i.supplier, i.description), i.account) for i in history])
            logger.info("autoseed: %d examples indexed", len(ap.classifier))
        except Exception:
            logger.exception("autoseed failed; starting with an empty index")


def create_app(autopilot: Autopilot | None = None) -> FastAPI:
    if autopilot is None:
        from .factory import build_autopilot
        autopilot = build_autopilot()
    ap = autopilot
    autoseed(ap)
    app = FastAPI(title="Invoice Autopilot", version="0.1.0",
                  description="Automated bookkeeping of supplier invoices. Get a synthetic invoice from "
                              "/sample-invoice, then POST its text to /invoices.")
    if (limit := rate_limit_from_env()) > 0:
        app.add_middleware(RateLimitMiddleware, limit_per_hour=limit)
    samples = [i for i in generate_invoices(40, seed=2024)]

    @app.get("/sample-invoice")
    def sample_invoice() -> dict:
        inv = random.choice(samples)
        return {"text": inv.text, "expected_account": inv.account, "expected_issue": inv.expected_issue}

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
