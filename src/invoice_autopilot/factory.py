"""Wire the pipeline from configuration (used by the API, the UI and the CLI)."""

from __future__ import annotations

from .config import ROOT, Settings, load_settings
from .llm import CachedChatModel, ChatClient, ChatModel, build_embedder
from .pipeline import Autopilot
from .store import Store


def build_autopilot(settings: Settings | None = None) -> Autopilot:
    settings = settings or load_settings()
    return Autopilot(settings, build_chat(settings), build_embedder(settings.embeddings), Store(settings.db_file))


def build_chat(settings: Settings) -> ChatModel:
    client: ChatModel = ChatClient(settings.llm)
    if settings.llm.cache_path:
        namespace = f"{settings.llm.model}|{settings.prompt_version}"
        client = CachedChatModel(client, ROOT / settings.llm.cache_path, namespace)
    return client
