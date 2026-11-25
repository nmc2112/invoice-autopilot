"""Evaluate the automation pipeline on synthetic invoices with ground truth.

A simulated reviewer handles the review queue (approves with the true account,
rejects duplicates), so the eval also measures the learning loop: does the
automation rate increase as corrections are added to the embedding index?

Usage:
    python evals/run_eval.py                  # models from config/settings.yaml
    python evals/run_eval.py --oracle         # scripted extraction + hashing embeddings (no network)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_autopilot.classify import example_text  # noqa: E402
from invoice_autopilot.config import load_settings  # noqa: E402
from invoice_autopilot.factory import build_chat  # noqa: E402
from invoice_autopilot.llm import HashingEmbedder, build_embedder  # noqa: E402
from invoice_autopilot.pipeline import Autopilot  # noqa: E402
from invoice_autopilot.store import Store  # noqa: E402
from invoice_autopilot.synth import generate_invoices  # noqa: E402

FIELDS = ["supplier", "invoice_number", "invoice_date", "amount_ht", "vat_amount", "amount_ttc"]


class OracleLLM:
    """Returns the ground-truth fields: isolates classification and checks from extraction quality."""

    def __init__(self, invoices):
        self.by_text = {i.text: i for i in invoices}

    def complete_json(self, prompt: str) -> str:
        inv = next(v for t, v in self.by_text.items() if t in prompt)
        return json.dumps({k: getattr(inv, k) for k in FIELDS + ["vat_rate", "description"]})


def field_ok(name, got, want) -> bool:
    if name in ("amount_ht", "vat_amount", "amount_ttc"):
        return abs(float(got) - float(want)) < 0.011
    return str(got).strip().lower() == str(want).strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("-n", type=int, default=60, help="invoices in the stream")
    parser.add_argument("--history", type=int, default=40, help="labelled invoices seeded before the stream")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    settings = load_settings()
    history = [i for i in generate_invoices(args.history, seed=999) if i.expected_issue is None]
    stream = generate_invoices(args.n, seed=args.seed)
    llm = OracleLLM(stream) if args.oracle else build_chat(settings)
    embedder = HashingEmbedder() if args.oracle else build_embedder(settings.embeddings)
    ap = Autopilot(settings, llm, embedder, Store(":memory:"))
    ap.seed_history([(example_text(i.supplier, i.description), i.account) for i in history])

    rows, start = [], time.perf_counter()
    for inv in stream:
        misses_before = getattr(llm, "misses", None)
        d = ap.process(inv.invoice_id, inv.text)
        tokens = 0 if args.oracle else (getattr(llm, "last_usage", {}).get("total_tokens") or 0)
        if misses_before is not None and llm.misses == misses_before:
            tokens = 0  # answered from cache, no tokens spent
        row = {"id": inv.invoice_id, "status": d.status, "expected_issue": inv.expected_issue,
               "account_ok": d.predicted_account == inv.account, "latency_s": d.latency_s, "tokens": tokens,
               "confidence": d.confidence,
               "hard_issue": any(not i.startswith("low_confidence") for i in d.issues),
               "fields_ok": sum(field_ok(f, getattr(d.data, f), getattr(inv, f)) for f in FIELDS) if d.data else 0}
        if d.status == "needs_review":   # simulated accountant
            if inv.expected_issue == "duplicate":
                ap.review(inv.invoice_id, approve=False)
            elif d.data is not None:
                ap.review(inv.invoice_id, approve=True, account=inv.account)
        rows.append(row)
        print(f"[{len(rows)}/{len(stream)}] {inv.invoice_id} {d.status} "
              f"account_ok={row['account_ok']} {d.latency_s}s", flush=True)

    n = len(rows)
    half = n // 2
    auto = [r for r in rows if r["status"] == "auto_posted"]
    issues = [r for r in rows if r["expected_issue"]]
    summary = {
        "mode": "oracle" if args.oracle else f"{settings.llm.model} + {settings.embeddings.model}",
        "invoices": n,
        "history_examples": len(history),
        "extraction_field_accuracy": round(sum(r["fields_ok"] for r in rows) / (n * len(FIELDS)), 3),
        "account_accuracy": round(sum(r["account_ok"] for r in rows) / n, 3),
        "automation_rate": round(len(auto) / n, 3),
        "automation_rate_first_half": round(sum(r["status"] == "auto_posted" for r in rows[:half]) / half, 3),
        "automation_rate_second_half": round(sum(r["status"] == "auto_posted" for r in rows[half:]) / (n - half), 3),
        "auto_posted_account_precision": round(sum(r["account_ok"] for r in auto) / len(auto), 3) if auto else None,
        "issues_caught": f"{sum(r['status'] == 'needs_review' for r in issues)}/{len(issues)}",
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
        "avg_tokens": round(sum(r["tokens"] for r in rows) / n),
        "total_time_s": round(time.perf_counter() - start, 1),
    }
    if hasattr(llm, "hits"):
        summary["llm_cache_hits"] = llm.hits
    print(json.dumps(summary, indent=2))

    # Threshold sweep (approximate: the learning loop ran with the configured threshold).
    print("\nthreshold  automation  auto-post precision")
    for t in [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]:
        auto_t = [r for r in rows if not r["hard_issue"] and r["confidence"] is not None and r["confidence"] >= t]
        prec = sum(r["account_ok"] for r in auto_t) / len(auto_t) if auto_t else float("nan")
        print(f"   {t:<7.2f} {len(auto_t) / n:>8.1%}   {prec:>10.1%}")
    out = ROOT / "evals" / "results"
    out.mkdir(exist_ok=True)
    (out / f"{summary['mode'].replace(':', '_').replace(' ', '')}_{int(time.time())}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
