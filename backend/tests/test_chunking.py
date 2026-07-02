"""Tests for the markdown chunker (RAG ingestion, phase E/M2).

Pure functions — driven directly, no I/O.

    docker compose exec backend pytest tests/test_chunking.py
"""
from __future__ import annotations

from app.plugins.knowledge.chunking import chunk_markdown


def test_splits_along_headings():
    md = "# Title\nintro line\n\n## Section A\nbody a\n\n## Section B\nbody b"
    chunks = chunk_markdown(md)
    headings = [h for h, _ in chunks]
    assert headings == ["Title", "Section A", "Section B"]
    assert chunks[1] == ("Section A", "body a")


def test_preamble_before_first_heading_has_empty_heading():
    chunks = chunk_markdown("loose text before any heading\n\n# H\nx")
    assert chunks[0][0] == "" and "loose text" in chunks[0][1]
    assert chunks[1] == ("H", "x")


def test_heading_without_body_is_dropped():
    # a heading carrying no content isn't a retrievable chunk
    chunks = chunk_markdown("# Empty\n\n# Real\nhas body")
    assert [h for h, _ in chunks] == ["Real"]


def test_long_section_is_split_keeping_heading():
    big = "\n\n".join(f"paragraph number {i} with some words" for i in range(50))
    chunks = chunk_markdown(f"# Big\n{big}", max_chars=120)
    assert len(chunks) > 1
    assert all(h == "Big" for h, _ in chunks)          # every piece keeps the section heading
    assert all(len(c) <= 120 for _, c in chunks)        # no piece exceeds the cap


def test_hard_split_of_single_oversized_paragraph():
    chunks = chunk_markdown("# H\n" + "x" * 500, max_chars=100)
    assert len(chunks) == 5 and all(len(c) <= 100 for _, c in chunks)


def test_empty_input_is_no_chunks():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []


def test_crlf_normalized():
    chunks = chunk_markdown("# H\r\nline1\r\nline2")
    assert chunks == [("H", "line1\nline2")]
