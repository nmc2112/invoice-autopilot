"""Expense-account classification with embeddings and a k-nearest-neighbour vote.

The index starts from labelled history and grows with every human correction,
so the share of invoices that can be posted automatically increases over time.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from pydantic import BaseModel

from .llm import Embedder


class Prediction(BaseModel):
    account: str
    confidence: float
    neighbours: list[tuple[str, float]]


class AccountClassifier:
    def __init__(self, embedder: Embedder, k: int = 5):
        self.embedder, self.k = embedder, k
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.labels: list[str] = []

    def __len__(self) -> int:
        return len(self.labels)

    def add(self, texts: list[str], labels: list[str]) -> None:
        if not texts:
            return
        new = self.embedder.embed(texts)
        self.vectors = new if not self.labels else np.vstack([self.vectors, new])
        self.labels += labels

    def predict(self, text: str) -> Prediction:
        if not self.labels:
            raise ValueError("The classifier has no labelled examples yet")
        query = self.embedder.embed([text])[0]
        sims = self.vectors @ query
        top = np.argsort(-sims)[: self.k]
        votes: dict[str, float] = defaultdict(float)
        for i in top:
            votes[self.labels[i]] += max(float(sims[i]), 0.0)
        total = sum(votes.values()) or 1.0
        account, score = max(votes.items(), key=lambda kv: kv[1])
        return Prediction(account=account, confidence=round(score / total, 3),
                          neighbours=[(self.labels[i], round(float(sims[i]), 3)) for i in top])


def example_text(supplier: str, description: str) -> str:
    """Text that represents an invoice for classification."""
    return f"{supplier} | {description}"
