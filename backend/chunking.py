"""Markdown chunking for RAG ingestion (phase E/M2, knowledge-rag.md §3).

Notes are written as headings already, so we chunk **along markdown headings** rather than at a
blind character offset — each chunk is one heading-bounded section, keeping a coherent topic
together. A section longer than `max_chars` is split into several chunks that all keep the same
heading, so one giant section can't swallow a single embedding.

Pure functions, no I/O — unit-tested directly. Returns `(heading, content)` pairs in document
order; the caller assigns sequence numbers and builds the text it actually embeds.
"""
from __future__ import annotations

import re

# ATX heading: 1–6 '#', a space, then the heading text (the leading hashes are stripped).
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Paragraph break: one or more blank lines.
_PARA = re.compile(r"\n\s*\n")


def _split_long(content: str, max_chars: int) -> list[str]:
    """Break a too-long section into <= max_chars pieces, preferring paragraph boundaries.
    A single paragraph longer than max_chars is hard-split (rare; over-long table/code block)."""
    if len(content) <= max_chars:
        return [content]
    pieces: list[str] = []
    buf = ""
    for para in _PARA.split(content):
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            # flush what we have, then hard-split the oversized paragraph
            if buf:
                pieces.append(buf)
                buf = ""
            for i in range(0, len(para), max_chars):
                pieces.append(para[i:i + max_chars])
            continue
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) > max_chars:
            pieces.append(buf)
            buf = para
        else:
            buf = candidate
    if buf:
        pieces.append(buf)
    return pieces


def chunk_markdown(text: str, *, max_chars: int = 1500) -> list[tuple[str, str]]:
    """Split markdown into `(heading, content)` chunks along its headings.

    Content before the first heading is emitted under an empty heading. Sections with no body are
    dropped (a heading alone carries no retrievable content). Headings are kept verbatim (without
    the leading '#'s) so the caller can prepend them to the embedded text for context."""
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")

    sections: list[tuple[str, list[str]]] = []
    heading = ""
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append((heading, content))

    for line in lines:
        m = _HEADING.match(line)
        if m:
            flush()
            heading = m.group(2).strip()
            body = []
        else:
            body.append(line)
    flush()

    out: list[tuple[str, str]] = []
    for h, content in sections:
        for piece in _split_long(content, max_chars):
            out.append((h, piece))
    return out
