"""Tests for the RAG embedders (phase E/M2).

StubEmbedder is pure/offline → driven directly. OllamaEmbedder's `complete`-style call runs
against an httpx MockTransport, so the real request/response path is covered without a model.

    docker compose exec backend pytest tests/test_embeddings.py
"""
from __future__ import annotations

import asyncio
import json
import math

import httpx

from app.core.config import settings
from app.plugins.knowledge.embeddings import OllamaEmbedder, StubEmbedder, get_embedder


# --- StubEmbedder -----------------------------------------------------------


def test_stub_is_deterministic_and_right_dim():
    a = asyncio.run(StubEmbedder(dim=8).embed(["hello", "world"]))
    b = asyncio.run(StubEmbedder(dim=8).embed(["hello", "world"]))
    assert a == b                                   # same text → same vector
    assert len(a) == 2 and len(a[0]) == 8


def test_stub_vectors_are_unit_length():
    v = asyncio.run(StubEmbedder(dim=16).embed(["something"]))[0]
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)


def test_stub_different_text_differs():
    [va, vb] = asyncio.run(StubEmbedder(dim=32).embed(["alpha", "beta"]))
    assert va != vb


def test_stub_default_dim_matches_config():
    v = asyncio.run(StubEmbedder().embed(["x"]))[0]
    assert len(v) == settings.embed_dim


def test_get_embedder_defaults_to_stub():
    assert isinstance(get_embedder(), StubEmbedder)   # embed_provider default = "stub"


# --- OllamaEmbedder over a mocked transport ---------------------------------


def test_ollama_posts_to_api_embed_and_parses():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]})

    prov = OllamaEmbedder(base_url="http://ollama:11434", model="bge-m3",
                          transport=httpx.MockTransport(handler))
    out = asyncio.run(prov.embed(["a", "b"]))
    assert out == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert captured["url"].endswith("/api/embed")
    assert captured["body"] == {"model": "bge-m3", "input": ["a", "b"]}


def test_ollama_empty_input_skips_request():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be called
        raise AssertionError("should not hit the network for empty input")

    prov = OllamaEmbedder(transport=httpx.MockTransport(handler))
    assert asyncio.run(prov.embed([])) == []
