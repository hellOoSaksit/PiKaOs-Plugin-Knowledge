"""Document → Markdown converters (phase E / E6, knowledge-rag.md §6.4).

Turns an uploaded file's bytes into markdown, which becomes the RAG source of truth (the original
is kept as a Ref — §0). Pure + in-memory (operates on bytes, no I/O) so it's testable directly and
runs inside the worker's ingest job off the event loop.

  * md / log / txt → decoded as-is (already markdown / plain text).
  * pdf           → text extracted with pypdf (text PDFs only; a scanned/image PDF yields no text →
                    returns None so the caller marks it "skipped" until OCR lands — §6.4).
  * docx          → mammoth → semantic markdown (headings/lists/bold preserved).

Returns None when there is nothing embeddable (unknown kind, or a valid file with no extractable
text). A genuinely corrupt file lets the underlying library raise, so the ingest job marks it
"failed" rather than silently empty.
"""
from __future__ import annotations

import io

# Kinds we attempt to convert; anything else (image/other) is skipped without reading bytes.
CONVERTIBLE_KINDS = {"md", "log", "pdf", "docx"}
# Kinds whose original is NOT markdown → conversion produces a new markdown object and the original
# is kept as a Ref (source_object_key). md/log are already text, so they are not "converted".
CONVERTED_KINDS = {"pdf", "docx"}


def _from_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _from_pdf(data: bytes) -> str | None:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(p for p in pages if p)
    return text or None  # image-only / scanned PDF → no text → skip (OCR deferred)


def _from_docx(data: bytes) -> str | None:
    import mammoth

    result = mammoth.convert_to_markdown(io.BytesIO(data))
    md = (result.value or "").strip()
    return md or None


def to_markdown(kind: str, data: bytes, filename: str | None = None) -> str | None:
    """Convert `data` of the given `documents.kind` to markdown, or None if nothing is embeddable."""
    if kind in ("md", "log"):
        text = _from_text(data)
        return text if text.strip() else None
    if kind == "pdf":
        return _from_pdf(data)
    if kind == "docx":
        return _from_docx(data)
    return None
