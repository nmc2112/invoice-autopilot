"""Streamlit UI for the MVP: submit invoices, review the queue, follow automation, export FEC.

Run: streamlit run app.py
"""

from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st

from invoice_autopilot.factory import build_autopilot
from invoice_autopilot.synth import CATALOG

st.set_page_config(page_title="Invoice Autopilot", layout="wide")


@st.cache_resource
def get_autopilot():
    return build_autopilot()


ap = get_autopilot()
accounts = {acc: f"{acc} - {label}" for acc, (label, _, _) in CATALOG.items()}

st.title("Invoice Autopilot")
st.caption("Automated bookkeeping of supplier invoices: LLM extraction, embedding-based account "
           "classification, control checks and human review. Synthetic data only.")

m = ap.metrics()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Processed", m["processed"])
c2.metric("Automation rate", f"{m['automation_rate']:.0%}")
c3.metric("Waiting for review", m.get("needs_review", 0))
c4.metric("Examples in index", m["examples_in_index"])

tab_new, tab_review, tab_all = st.tabs(["New invoice", "Review queue", "All invoices & export"])

with tab_new:
    text = st.text_area("Paste the invoice text (OCR output)", height=260)
    if st.button("Process", type="primary", disabled=len(text) < 20):
        with st.spinner("Extracting and classifying..."):
            d = ap.process(f"UI-{uuid.uuid4().hex[:8]}", text)
        (st.success if d.status == "auto_posted" else st.warning)(f"Status: {d.status}")
        st.json(d.model_dump(mode="json"))

with tab_review:
    queue = ap.store.list_invoices("needs_review")
    if not queue:
        st.info("Nothing to review.")
    for inv in queue:
        with st.expander(f"{inv['id']} - {', '.join(i.split(':')[0] for i in inv['issues'])}"):
            left, right = st.columns(2)
            left.text(inv["text"])
            right.write("**Issues**")
            for issue in inv["issues"]:
                right.write(f"- {issue}")
            right.write("**Extracted data**")
            right.json(inv["data"] or {})
            keys: list[str] = list(accounts)
            default = keys.index(inv["predicted_account"]) if inv["predicted_account"] in keys else 0
            account = right.selectbox("Expense account", keys, index=default,
                                      format_func=lambda a: accounts[a], key=f"acc-{inv['id']}")
            b1, b2 = right.columns(2)
            if b1.button("Approve and post", key=f"ok-{inv['id']}", disabled=inv["data"] is None):
                ap.review(inv["id"], True, account)
                st.rerun()
            if b2.button("Reject", key=f"no-{inv['id']}"):
                ap.review(inv["id"], False)
                st.rerun()

with tab_all:
    rows = [{k: v for k, v in i.items() if k not in ("text", "data")} for i in ap.store.list_invoices()]
    st.dataframe(pd.DataFrame(rows), width="stretch")
    st.download_button("Download FEC export", ap.export_fec(), file_name="export_fec.txt")
