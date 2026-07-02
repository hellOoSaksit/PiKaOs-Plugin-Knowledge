"""Tests for summarize_service — RAG ingest enrich B (E7).

Pure: the LLM provider is a fake (anything with `complete(model, messages, tools) -> obj.text`),
so the prompt-build / trim / best-effort behaviour is exercised without a model or network.

    docker compose exec backend pytest tests/test_summarize_service.py
"""
from __future__ import annotations

import asyncio

from app.plugins.knowledge import summarize_service as ss


class _Result:
    def __init__(self, text):
        self.text = text


class _Provider:
    """Records the messages it was called with and returns a scripted summary."""

    def __init__(self, text="a concise summary"):
        self.text = text
        self.seen: list[dict] | None = None

    async def complete(self, *, model, messages, tools):
        self.seen = messages
        return _Result(self.text)


class _Boom:
    async def complete(self, *, model, messages, tools):
        raise RuntimeError("provider down")


def _run(coro):
    return asyncio.run(coro)


def test_summarize_returns_trimmed_text():
    p = _Provider("  Summary body.  ")
    out = _run(ss.summarize_document(p, title="Notes", markdown="# H\nbody", max_input_chars=1000))
    assert out == "Summary body."
    # the doc title + body reached the provider as a user turn
    user = next(m for m in p.seen if m["role"] == "user")
    assert "Notes" in user["content"] and "body" in user["content"]


def test_summarize_truncates_input():
    p = _Provider("ok")
    long_md = "x" * 5000
    _run(ss.summarize_document(p, title="T", markdown=long_md, max_input_chars=100))
    user = next(m for m in p.seen if m["role"] == "user")
    # only the first 100 chars of the body are sent (plus the short "Title: ..." preamble)
    assert user["content"].count("x") == 100


def test_summarize_empty_markdown_is_none():
    p = _Provider("should not be used")
    assert _run(ss.summarize_document(p, title="T", markdown="   ", max_input_chars=100)) is None
    assert p.seen is None                      # provider never called for empty input


def test_summarize_blank_result_is_none():
    p = _Provider("   ")                        # provider returns only whitespace
    assert _run(ss.summarize_document(p, title="T", markdown="real body", max_input_chars=100)) is None


def test_summarize_provider_failure_is_none():
    # best-effort: a provider error must not raise (ingest continues without a summary)
    assert _run(ss.summarize_document(_Boom(), title="T", markdown="body", max_input_chars=100)) is None
