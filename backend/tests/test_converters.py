"""Tests for the document → markdown converters (phase E / E6, knowledge-rag.md §6.4).

Pure + in-memory: the text paths (md/log) run as-is; the pdf/docx paths only route to the right
extractor here (the extractors themselves wrap pypdf/mammoth, exercised live in docker). No DB.

    docker compose exec backend pytest tests/test_converters.py
"""
from __future__ import annotations

from app.plugins.knowledge import converters


def test_text_kinds_decode_as_is():
    assert converters.to_markdown("md", b"# Title\nbody") == "# Title\nbody"
    assert converters.to_markdown("log", b"plain log line") == "plain log line"


def test_blank_text_is_none():
    """Whitespace-only / empty content embeds nothing → None (caller marks it skipped)."""
    assert converters.to_markdown("md", b"   \n  \t") is None
    assert converters.to_markdown("log", b"") is None


def test_unknown_kind_is_none():
    assert converters.to_markdown("image", b"\x89PNG\r\n") is None
    assert converters.to_markdown("other", b"anything") is None


def test_pdf_routes_to_pdf_extractor(monkeypatch):
    monkeypatch.setattr(converters, "_from_pdf", lambda data: "extracted pdf md")
    assert converters.to_markdown("pdf", b"%PDF-1.4 ...") == "extracted pdf md"


def test_docx_routes_to_docx_extractor(monkeypatch):
    monkeypatch.setattr(converters, "_from_docx", lambda data: "extracted docx md")
    assert converters.to_markdown("docx", b"PK\x03\x04 ...") == "extracted docx md"


def test_converted_is_a_subset_of_convertible():
    """pdf/docx are converted (original kept as Ref); md/log are already text, not converted."""
    assert converters.CONVERTED_KINDS <= converters.CONVERTIBLE_KINDS
    assert {"pdf", "docx"} <= converters.CONVERTED_KINDS
    assert "md" in converters.CONVERTIBLE_KINDS and "md" not in converters.CONVERTED_KINDS
