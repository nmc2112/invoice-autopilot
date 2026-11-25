import httpx
import numpy as np
import pytest

from invoice_autopilot.config import EmbeddingConfig, LLMConfig
from invoice_autopilot.llm import ChatClient, LLMError, OpenAIEmbedder


def test_embedder_batches_and_normalises():
    seen = []

    def handler(request):
        body = request.read().decode()
        import json
        inputs = json.loads(body)["input"]
        seen.append(len(inputs))
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [3.0, 4.0]} for i in range(len(inputs))]})

    emb = OpenAIEmbedder(EmbeddingConfig(base_url="http://test/v1"), transport=httpx.MockTransport(handler),
                         batch_size=2, sleep=lambda s: None)
    vectors = emb.embed(["a", "b", "c"])
    assert seen == [2, 1]
    assert vectors.shape == (3, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_chat_retries_then_succeeds():
    statuses = iter([503, 200])

    def handler(request):
        s = next(statuses)
        body = {"choices": [{"message": {"content": "{}"}}], "usage": {"total_tokens": 7}} if s == 200 else {}
        return httpx.Response(s, json=body)

    client = ChatClient(LLMConfig(base_url="http://test/v1", model="m", backoff_s=0.0),
                        transport=httpx.MockTransport(handler), sleep=lambda s: None)
    assert client.complete_json("hi") == "{}"
    assert client.last_usage["total_tokens"] == 7


def test_chat_client_error_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={})

    client = ChatClient(LLMConfig(base_url="http://test/v1", model="m"), transport=httpx.MockTransport(handler),
                        sleep=lambda s: None)
    with pytest.raises(LLMError):
        client.complete_json("hi")
    assert len(calls) == 1


def test_cache_answers_repeated_prompts(tmp_path):
    from invoice_autopilot.llm import CachedChatModel, FakeLLM
    inner = FakeLLM(['{"a": 1}'])
    cached = CachedChatModel(inner, tmp_path / "c.db", namespace="m|v1")
    assert cached.complete_json("same prompt") == '{"a": 1}'
    assert cached.complete_json("same prompt") == '{"a": 1}'   # FakeLLM script is empty now: must hit cache
    assert (cached.hits, cached.misses) == (1, 1)
    other = CachedChatModel(FakeLLM(['{"b": 2}']), tmp_path / "c.db", namespace="other-model|v1")
    assert other.complete_json("same prompt") == '{"b": 2}'     # different model: no stale answer


def test_factory_wraps_client_with_cache(settings):
    from invoice_autopilot.factory import build_chat
    from invoice_autopilot.llm import CachedChatModel
    assert isinstance(build_chat(settings), CachedChatModel)
    no_cache = settings.model_copy(update={"llm": settings.llm.model_copy(update={"cache_path": None})})
    assert not isinstance(build_chat(no_cache), CachedChatModel)
