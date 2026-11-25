"""SQLite persistence: processed invoices, review queue, posted entries, examples."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS invoices (
    id TEXT PRIMARY KEY,
    received_at TEXT DEFAULT CURRENT_TIMESTAMP,
    text TEXT NOT NULL,
    status TEXT NOT NULL,              -- auto_posted | needs_review | posted_after_review | rejected
    data TEXT,                         -- extracted fields (JSON)
    predicted_account TEXT,
    confidence REAL,
    final_account TEXT,
    issues TEXT                        -- JSON list
);
CREATE TABLE IF NOT EXISTS entries (
    invoice_id TEXT NOT NULL,
    line TEXT NOT NULL                 -- FEC line (JSON)
);
CREATE TABLE IF NOT EXISTS examples (
    text TEXT NOT NULL,
    account TEXT NOT NULL,
    source TEXT NOT NULL               -- history | review
);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def save_invoice(self, row: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO invoices (id, text, status, data, predicted_account, confidence, "
            "final_account, issues) VALUES (:id, :text, :status, :data, :predicted_account, :confidence, "
            ":final_account, :issues)",
            {**row, "issues": json.dumps(row.get("issues", []))})
        self.conn.commit()

    def get_invoice(self, invoice_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM invoices WHERE id = ?", (invoice_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["issues"] = json.loads(out["issues"] or "[]")
        out["data"] = json.loads(out["data"]) if out["data"] else None
        return out

    def list_invoices(self, status: str | None = None) -> list[dict]:
        query: str = "SELECT id FROM invoices"
        args: tuple[str, ...] = ()
        if status:
            query, args = query + " WHERE status = ?", (status,)
        return [i for r in self.conn.execute(query + " ORDER BY received_at, id", args).fetchall()
                if (i := self.get_invoice(r["id"])) is not None]

    def update_status(self, invoice_id: str, status: str, final_account: str | None) -> None:
        self.conn.execute("UPDATE invoices SET status = ?, final_account = ? WHERE id = ?",
                          (status, final_account, invoice_id))
        self.conn.commit()

    def add_entry_lines(self, invoice_id: str, lines: list[dict]) -> None:
        self.conn.executemany("INSERT INTO entries (invoice_id, line) VALUES (?, ?)",
                              [(invoice_id, json.dumps(line, ensure_ascii=False)) for line in lines])
        self.conn.commit()

    def next_entry_num(self) -> str:
        n = self.conn.execute("SELECT COUNT(DISTINCT invoice_id) FROM entries").fetchone()[0]
        return f"{n + 1:06d}"

    def entry_lines(self) -> list[dict]:
        return [json.loads(r["line"]) for r in self.conn.execute("SELECT line FROM entries").fetchall()]

    def recorded_keys(self) -> set[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT data FROM invoices WHERE data IS NOT NULL AND status != 'rejected'").fetchall()
        keys = set()
        for r in rows:
            d = json.loads(r["data"])
            keys.add((d["supplier"].lower(), d["invoice_number"].lower()))
        return keys

    def add_example(self, text: str, account: str, source: str) -> None:
        self.conn.execute("INSERT INTO examples VALUES (?, ?, ?)", (text, account, source))
        self.conn.commit()

    def examples(self) -> list[tuple[str, str]]:
        return [(r["text"], r["account"]) for r in self.conn.execute("SELECT text, account FROM examples")]

    def counts(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT status, COUNT(*) AS n FROM invoices GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}
