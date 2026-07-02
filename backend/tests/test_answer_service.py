"""Tests for answer_service — RAG answer with citations (E8).

The LLM provider is faked and `knowledge_service.search_documents` is monkeypatched, so the
rewrite → retrieve → synthesize → cite flow runs without a DB, model, or network.

    docker compose exec backend pytest tests/test_answer_service.py
"""
from __future__ import annotations

import asyncio
import uuid

from app.plugins.knowledge import answer_service as asvc


class _Result:
    def __init__(self, text):
        self.text = text


class _Provider:
    """Returns a different reply for the rewrite vs the answer call (told apart by the system
    prompt), and records every call so tests can assert what was asked."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    async def complete(self, *, model, messages, tools):
        self.calls.append(messages)
        system = messages[0]["content"]
        if system == asvc.REWRITE_SYSTEM:
            return _Result("rewritten query")
        return _Result("The answer is 42 [1].")


def _rows(*names):
    return [
        {"document_id": uuid.uuid4(), "document_name": n, "document_kind": "md",
         "seq": 0, "heading": f"h-{n}", "content": f"body-{n}", "score": 0.9}
        for n in names
    ]


def _patch_search(monkeypatch, rows, sink):
    async def fake_search(db, *, embedder, user, query, k):
        sink["query"] = query
        sink["k"] = k
        return rows
    monkeypatch.setattr(asvc.knowledge_service, "search_documents", fake_search)


def _run(coro):
    return asyncio.run(coro)


def test_build_sources_numbering_and_fields():
    rows = _rows("a.md", "b.md")
    src = asvc.build_sources(rows)
    assert [s["n"] for s in src] == [1, 2]
    assert src[0]["document_name"] == "a.md" and src[0]["heading"] == "h-a.md"
    assert src[0]["score"] == 0.9 and src[0]["document_id"] == rows[0]["document_id"]


def test_empty_question_short_circuits(monkeypatch):
    called = {"n": 0}

    async def fake_search(*a, **k):
        called["n"] += 1
        return []
    monkeypatch.setattr(asvc.knowledge_service, "search_documents", fake_search)

    out = _run(asvc.answer_question(
        None, embedder=None, provider=_Provider(), user=None, question="   ", k=5))
    assert out == {"answer": "", "sources": [], "rewritten_query": "", "used_chunks": 0}
    assert called["n"] == 0                       # never searched


def test_no_matching_chunks_is_graceful(monkeypatch):
    sink: dict = {}
    _patch_search(monkeypatch, [], sink)
    out = _run(asvc.answer_question(
        None, embedder=None, provider=_Provider(), user=None, question="anything?", k=5))
    assert out["sources"] == [] and out["used_chunks"] == 0
    assert out["rewritten_query"] == "rewritten query"     # rewrite still ran
    assert "don't have" in out["answer"].lower()


def test_answer_with_rewrite_cites_sources(monkeypatch):
    sink: dict = {}
    rows = _rows("a.md", "b.md")
    _patch_search(monkeypatch, rows, sink)
    provider = _Provider()
    out = _run(asvc.answer_question(
        None, embedder=None, provider=provider, user=None, question="what is the answer?", k=4))

    assert out["answer"] == "The answer is 42 [1]."
    assert out["rewritten_query"] == "rewritten query"
    assert sink["query"] == "rewritten query" and sink["k"] == 4   # searched the rewrite
    assert out["used_chunks"] == 2 and [s["n"] for s in out["sources"]] == [1, 2]
    # the answer turn carried the numbered context the model cites
    answer_user = provider.calls[-1][1]["content"]
    assert "[1]" in answer_user and "[2]" in answer_user


def test_rewrite_disabled_searches_original(monkeypatch):
    sink: dict = {}
    _patch_search(monkeypatch, _rows("a.md"), sink)
    provider = _Provider()
    out = _run(asvc.answer_question(
        None, embedder=None, provider=provider, user=None,
        question="original question", k=3, rewrite=False))
    assert sink["query"] == "original question"
    assert out["rewritten_query"] == "original question"
    assert len(provider.calls) == 1               # only the answer call, no rewrite
