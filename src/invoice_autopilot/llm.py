"""LLM and embedding clients (OpenAI-compatible), with retries and offline fakes.

- ChatClient: /chat/completions in JSON mode, retries with exponential backoff.
- OpenAIEmbedder: /embeddings (works with Ollama: nomic-embed-text, bge-m3...).
- HashingEmbedder: deterministic bag-of-words vectors, no network (tests, CI).
- FakeLLM: scripted chat responses for tests.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np

from .config import EmbeddingConfig, LLMConfig

log = logging.getLogger("invoice_autopilot.llm")
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


class ChatModel(Protocol):
    def complete_json(self, prompt: str) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def _post_with_retries(client: httpx.Client, path: str, payload: dict, max_retries: int,
                       backoff_s: float, sleep: Callable[[float], None]) -> dict:
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.post(path, json=payload)
            if resp.status_code in RETRYABLE_STATUS:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json()
        except (httpx.TransportError, LLMError) as exc:
            last = exc
            log.warning("attempt %d on %s failed: %s", attempt, path, exc)
            if attempt < max_retries:
                sleep(backoff_s * 2 ** (attempt - 1))
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Non-retryable error on {path}: {exc}") from exc
    raise LLMError(f"{path} failed after {max_retries} attempts: {last}")


class ChatClient:
    def __init__(self, cfg: LLMConfig, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg, self._sleep = cfg, sleep
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        self._http = httpx.Client(base_url=cfg.base_url, timeout=cfg.timeout_s, headers=headers,
                                  transport=transport)
        self.last_usage: dict = {}

    def complete_json(self, prompt: str) -> str:
        payload = {"model": self.cfg.model, "temperature": 0.0,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "user", "content": prompt}]}
        data = _post_with_retries(self._http, "/chat/completions", payload, self.cfg.max_retries,
                                  self.cfg.backoff_s, self._sleep)
        self.last_usage = data.get("usage") or {}
        return data["choices"][0]["message"].get("content") or ""


class OpenAIEmbedder:
    def __init__(self, cfg: EmbeddingConfig, transport: httpx.BaseTransport | None = None,
                 batch_size: int = 64, sleep: Callable[[float], None] = time.sleep):
        self.cfg, self.batch_size, self._sleep = cfg, batch_size, sleep
        self._http = httpx.Client(base_url=cfg.base_url, timeout=120, transport=transport)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            data = _post_with_retries(self._http, "/embeddings", {"model": self.cfg.model, "input": batch},
                                      3, 1.0, self._sleep)
            vectors += [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]
        return normalize(np.asarray(vectors, dtype=np.float32))


class HashingEmbedder:
    """Deterministic offline embedder: hashed word and character-trigram counts."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ]+", text.lower())
        grams = [w[i:i + 3] for w in words for i in range(max(1, len(w) - 2))]
        return words + grams

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        return normalize(out)


class CachedChatModel:
    """Caches LLM answers on disk, keyed by model + prompt hash.

    Re-processing the same document costs no tokens and no latency (useful for
    re-runs, retries of a batch, and evaluation experiments).
    """

    def __init__(self, inner: ChatModel, path: str | Path, namespace: str):
        import sqlite3
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.inner, self.namespace = inner, namespace
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self.hits = self.misses = 0

    def complete_json(self, prompt: str) -> str:
        key = hashlib.sha256(f"{self.namespace}\n{prompt}".encode()).hexdigest()
        row = self.conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            self.hits += 1
            return str(row[0])
        self.misses += 1
        value = self.inner.complete_json(prompt)
        self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, value))
        self.conn.commit()
        return value

    @property
    def last_usage(self) -> dict:
        return getattr(self.inner, "last_usage", {}) if self.misses else {}


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    return HashingEmbedder() if cfg.provider == "hashing" else OpenAIEmbedder(cfg)


class FakeLLM:
    """Scripted chat model for tests: returns responses from a list or a function."""

    def __init__(self, responder: list[str] | Callable[[str], str]):
        self.responder = responder
        self.prompts: list[str] = []

    def complete_json(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if callable(self.responder):
            return self.responder(prompt)
        if not self.responder:
            raise LLMError("FakeLLM script exhausted")
        return self.responder.pop(0)
