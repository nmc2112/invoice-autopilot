"""Turn a validated invoice into balanced journal-entry lines and export them as FEC."""

from __future__ import annotations

import io

import pandas as pd

from .config import LedgerConfig
from .extract import InvoiceData

FEC_COLUMNS = ["JournalCode", "JournalLib", "EcritureNum", "EcritureDate", "CompteNum", "CompteLib",
               "CompAuxNum", "CompAuxLib", "PieceRef", "PieceDate", "EcritureLib", "Debit", "Credit",
               "EcritureLet", "DateLet", "ValidDate", "Montantdevise", "Idevise"]


def entry_lines(data: InvoiceData, account: str, entry_num: str, cfg: LedgerConfig,
                valid_date: str) -> list[dict]:
    """Purchase entry: debit expense (HT) and deductible VAT, credit supplier (TTC)."""
    d = data.invoice_date.strftime("%Y%m%d")
    common = {"JournalCode": cfg.journal_code, "JournalLib": "Achats", "EcritureNum": entry_num,
              "EcritureDate": d, "CompAuxNum": "", "CompAuxLib": "", "PieceRef": data.invoice_number,
              "PieceDate": d, "EcritureLib": f"{data.supplier} - {data.description}"[:100],
              "EcritureLet": "", "DateLet": "", "ValidDate": valid_date, "Montantdevise": 0.0, "Idevise": "EUR"}
    lines = [{**common, "CompteNum": account, "CompteLib": "Charge", "Debit": data.amount_ht, "Credit": 0.0}]
    if data.vat_amount > 0:
        lines.append({**common, "CompteNum": cfg.vat_account, "CompteLib": "TVA déductible",
                      "Debit": data.vat_amount, "Credit": 0.0})
    lines.append({**common, "CompteNum": cfg.supplier_account, "CompteLib": "Fournisseurs",
                  "CompAuxLib": data.supplier, "Debit": 0.0, "Credit": round(data.amount_ht + data.vat_amount, 2)})
    return lines


def export_fec(lines: list[dict]) -> str:
    df = pd.DataFrame(lines, columns=FEC_COLUMNS)
    for col in ["Debit", "Credit", "Montantdevise"]:
        df[col] = df[col].map(lambda x: f"{float(x):.2f}".replace(".", ","))
    buf = io.StringIO()
    df.to_csv(buf, sep="\t", index=False)
    return buf.getvalue()
