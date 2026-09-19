import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings


EMBEDDING_PROFILE_VERSION = 1


@dataclass(frozen=True)
class EmbeddingProfile:
    model: str
    query_prefix: str
    document_prefix: str
    version: int = EMBEDDING_PROFILE_VERSION

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "query_prefix": self.query_prefix,
                "document_prefix": self.document_prefix,
                "version": self.version,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "version": self.version,
            "fingerprint": self.fingerprint,
        }


def embedding_profile(model: str | None = None) -> EmbeddingProfile:
    model = (model or os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:8b-q8_0")).strip()
    normalized = model.lower().split(":", 1)[0]

    if "embeddinggemma" in normalized:
        query_prefix = "task: search result | query: "
        document_prefix = "title: none | text: "
    elif "nomic-embed-text" in normalized:
        query_prefix = "search_query: "
        document_prefix = "search_document: "
    elif "qwen3-embedding" in normalized:
        query_prefix = (
            "Instruct: Given a user question, retrieve relevant passages that answer the question\n"
            "Query: "
        )
        document_prefix = ""
    else:
        query_prefix = ""
        document_prefix = ""

    return EmbeddingProfile(
        model=model,
        query_prefix=os.getenv("EMBED_QUERY_PREFIX", query_prefix),
        document_prefix=os.getenv("EMBED_DOCUMENT_PREFIX", document_prefix),
    )


class PromptedOllamaEmbeddings(Embeddings):
    """Ollama embeddings with asymmetric query/document retrieval prompts."""

    def __init__(self, profile: EmbeddingProfile, base_url: str, keep_alive: int):
        self.profile = profile
        self._client = OllamaEmbeddings(
            model=profile.model,
            base_url=base_url,
            keep_alive=keep_alive,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefix = self.profile.document_prefix
        prepared = [f"{prefix}{text}" if prefix else text for text in texts]
        return self._client.embed_documents(prepared)

    def embed_query(self, text: str) -> list[float]:
        prefix = self.profile.query_prefix
        return self._client.embed_query(f"{prefix}{text}" if prefix else text)


def ollama_model_names(base_url: str, timeout_seconds: float = 3.0) -> set[str]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama is unavailable at {base_url}: {exc}") from exc

    return {str(model.get("name", "")) for model in payload.get("models", []) if model.get("name")}


def model_is_available(configured_model: str, installed_models: set[str]) -> bool:
    configured_model = configured_model.strip()
    if configured_model in installed_models:
        return True
    if ":" not in configured_model and f"{configured_model}:latest" in installed_models:
        return True
    return False
