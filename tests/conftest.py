import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_autopilot.classify import example_text  # noqa: E402
from invoice_autopilot.config import load_settings  # noqa: E402
from invoice_autopilot.llm import FakeLLM, HashingEmbedder  # noqa: E402
from invoice_autopilot.pipeline import Autopilot  # noqa: E402
from invoice_autopilot.store import Store  # noqa: E402
from invoice_autopilot.synth import generate_invoices  # noqa: E402

FIELDS = ["supplier", "invoice_number", "invoice_date", "amount_ht", "vat_rate", "vat_amount",
          "amount_ttc", "description"]


def truth_json(inv) -> str:
    return json.dumps({k: getattr(inv, k) for k in FIELDS})


def oracle(invoices):
    by_text = {i.text: i for i in invoices}
    return FakeLLM(lambda prompt: truth_json(next(v for t, v in by_text.items() if t in prompt)))


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def history():
    return [i for i in generate_invoices(140, seed=999) if i.expected_issue is None]


@pytest.fixture
def make_autopilot(settings, history):
    def _make(llm, seed_history=True):
        ap = Autopilot(settings, llm, HashingEmbedder(), Store(":memory:"))
        if seed_history:
            ap.seed_history([(example_text(i.supplier, i.description), i.account) for i in history])
        return ap
    return _make
