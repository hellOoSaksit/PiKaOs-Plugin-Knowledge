"""Embedders for the knowledge RAG index (phase E/M2, knowledge-rag.md §3).

Turns text (a markdown chunk, or a search query) into a fixed-dimension vector. Two implementations
behind one tiny interface:

  * **StubEmbedder** (default) — deterministic, offline, free. The same text always maps to the
    same unit vector, so tests and dev can exercise the whole ingest→search pipeline without a
    model. It carries no semantics (it's a hash), so it only matches text that is byte-identical.
  * **OllamaEmbedder** — a real local embedding model (bge-m3) over Ollama's `/api/embed`, reusing
    httpx like the LLM adapters (no new dependency — tech-stack §3.2).

`get_embedder()` picks one from config (`embed_provider`); both web (search) and worker (ingest)
go through it, so the embedding model is chosen in one place. Dimension is fixed platform-wide
(`embed_dim`) and baked into the doc_chunks column — changing it means re-embedding (E1).
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

import httpx

from ...core.config import settings


class Embedder(Protocol):
    model_name: str
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def _stub_vector(text: str, dim: int) -> list[float]:
    """A deterministic unit vector derived from the text via SHA-256 (no semantics — for tests)."""
    out: list[float] = []
    counter = 0
    while len(out) < dim:
        digest = hashlib.sha256(f"{text}#{counter}".encode("utf-8")).digest()  # 32 bytes
        for i in range(0, len(digest), 4):
            val = int.from_bytes(digest[i:i + 4], "big") / 0xFFFFFFFF  # 0..1
            out.append(val * 2.0 - 1.0)                                 # -1..1
        counter += 1
    out = out[:dim]
    norm = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / norm for v in out]


class StubEmbedder:
    """Deterministic offline embedder (see module docstring)."""

    model_name = "stub"

    def __init__(self, dim: int | None = None):
        self.dim = dim or settings.embed_dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_stub_vector(t or "", self.dim) for t in texts]


class OllamaEmbedder:
    """Real embeddings from a local Ollama model (bge-m3) via `POST {base}/api/embed`."""

    def __init__(self, *, base_url: str | None = None, model: str | None = None,
                 timeout: float | None = None, transport: httpx.BaseTransport | None = None):
        self.base_url = (base_url or settings.embed_base_url).rstrip("/")
        self.model_name = model or settings.embed_model
        self.timeout = timeout or settings.embed_request_timeout_s
        self._transport = transport  # tests inject httpx.MockTransport; None → real network

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        body = {"model": self.model_name, "input": texts}
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout,
                                     transport=self._transport) as client:
            resp = await client.post("/api/embed", json=body)
            resp.raise_for_status()
            data = resp.json()
        embeddings = data.get("embeddings") or []
        return [[float(x) for x in vec] for vec in embeddings]


def get_embedder() -> Embedder:
    """The configured embedder — `embed_provider` selects ollama, else the stub default."""
    if settings.embed_provider == "ollama":
        return OllamaEmbedder()
    return StubEmbedder()
