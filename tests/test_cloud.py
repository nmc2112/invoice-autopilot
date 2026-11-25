import httpx
from conftest import oracle
from fastapi.testclient import TestClient

from invoice_autopilot.api import create_app
from invoice_autopilot.config import ROOT, EmbeddingConfig, load_settings
from invoice_autopilot.llm import HashingEmbedder, OpenAIEmbedder
from invoice_autopilot.pipeline import Autopilot
from invoice_autopilot.store import Store
from invoice_autopilot.synth import generate_invoices


def test_autoseed_and_sample_invoice(monkeypatch, settings):
    monkeypatch.setenv("AUTOSEED_HISTORY", "50")
    sample_pool = generate_invoices(40, seed=2024)
    ap = Autopilot(settings, oracle(sample_pool), HashingEmbedder(), Store(":memory:"))
    client = TestClient(create_app(ap))
    assert client.get("/health").json()["examples_in_index"] > 0
    sample = client.get("/sample-invoice").json()
    resp = client.post("/invoices", json={"text": sample["text"]})
    assert resp.status_code == 200 and resp.json()["status"] in {"auto_posted", "needs_review"}


def test_rate_limit(monkeypatch, settings):
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "1")
    client = TestClient(create_app(Autopilot(settings, oracle([]), HashingEmbedder(), Store(":memory:"))))
    first = client.post("/invoices", json={"text": "short"})            # 422, but counted
    second = client.post("/invoices", json={"text": "short"})
    assert first.status_code == 422 and second.status_code == 429


def test_embedder_sends_api_key(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret-value")
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    cfg = EmbeddingConfig(provider="openai", base_url="http://test/v1", model="m", api_key_env="TEST_KEY")
    OpenAIEmbedder(cfg, transport=httpx.MockTransport(handler)).embed(["x"])
    assert seen["auth"] == "Bearer secret-value"


def test_cloud_config():
    cfg = load_settings(ROOT / "config" / "settings.cloud.yaml")
    assert cfg.llm.api_key_env == "LLM_API_KEY" and cfg.embeddings.api_key_env == "LLM_API_KEY"
    assert cfg.embeddings.provider == "openai"


def test_openai_embedder_accepts_responses_without_index():
    import httpx

    from invoice_autopilot.config import EmbeddingConfig
    from invoice_autopilot.llm import OpenAIEmbedder

    def handler(request: httpx.Request) -> httpx.Response:  # Gemini style: no "index" field
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]})

    cfg = EmbeddingConfig(provider="openai", base_url="https://example.test/v1", model="m")
    vectors = OpenAIEmbedder(cfg, transport=httpx.MockTransport(handler)).embed(["a", "b"])
    assert vectors.shape == (2, 2) and vectors[0, 0] == 1.0 and vectors[1, 1] == 1.0
