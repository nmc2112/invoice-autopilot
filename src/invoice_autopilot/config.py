"""Typed loading of the versioned YAML configuration and prompts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


def _find_root() -> Path:
    """Project root holding config/ and prompts/ (works installed or from source)."""
    if os.environ.get("INVOICE_AUTOPILOT_HOME"):
        return Path(os.environ["INVOICE_AUTOPILOT_HOME"])
    if (Path.cwd() / "config" / "settings.yaml").exists():
        return Path.cwd()
    return Path(__file__).resolve().parents[2]


ROOT = _find_root()


class LLMConfig(BaseModel):
    base_url: str
    model: str
    api_key_env: str | None = None
    timeout_s: float = 120
    max_retries: int = 3
    backoff_s: float = 1.0
    cache_path: str | None = None

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class EmbeddingConfig(BaseModel):
    provider: Literal["ollama", "hashing"] = "ollama"
    base_url: str = "http://localhost:11434/v1"
    model: str = "nomic-embed-text"


class ClassifierConfig(BaseModel):
    k: int = 5
    auto_post_threshold: float = 0.75


class ChecksConfig(BaseModel):
    vat_rates: list[float] = [0.0, 5.5, 10.0, 20.0]
    amount_tolerance: float = 0.02
    max_auto_post_amount: float = 10000


class LedgerConfig(BaseModel):
    journal_code: str = "AC"
    supplier_account: str = "401000"
    vat_account: str = "445660"
    db_path: str = "data/autopilot.db"


class Settings(BaseModel):
    prompt_version: str
    llm: LLMConfig
    embeddings: EmbeddingConfig = EmbeddingConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    checks: ChecksConfig = ChecksConfig()
    ledger: LedgerConfig = LedgerConfig()

    def prompt(self, name: str) -> str:
        return (ROOT / "prompts" / self.prompt_version / f"{name}.md").read_text(encoding="utf-8")

    @property
    def db_file(self) -> Path:
        path = Path(self.ledger.db_path)
        return path if path.is_absolute() else ROOT / path


def load_settings(path: str | Path | None = None) -> Settings:
    if path is None:
        path = os.environ.get("INVOICE_AUTOPILOT_CONFIG") or ROOT / "config" / "settings.yaml"
    return Settings.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
