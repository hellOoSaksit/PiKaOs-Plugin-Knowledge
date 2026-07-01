"""RAG ingestion — turn a stored document into searchable chunks (phase E/M2, knowledge-rag.md §3).

The heavy half of the knowledge pipeline, run by the arq worker so embedding a big file can't block
the API (E2). One document at a time: read its bytes from MinIO → chunk along markdown headings →
embed each chunk → replace its rows in `doc_chunks`. Markdown stays the source of truth; these
chunks are a rebuildable cache, so this is safe to re-run any time (it replaces, never appends).

`ingest_document` takes the embedder + db as arguments (not globals) so it's testable directly with
a StubEmbedder against a real DB, exactly like the agent_runner loop. The worker job is a thin shim
that resolves the configured embedder and calls it.

Text-native documents (md / log) are chunked directly; pdf / docx are converted to markdown first
(E6, `converters.py`) — the markdown becomes the truth and the original is kept as a Ref. image /
other are marked `skipped`. No SQL here (repositories) and no FastAPI types (routers) — §2.1.
"""
from __future__ import annotations

import asyncio
import logging

import uuid

from .storage_ref import get_storage
from ...core.config import settings
from . import doc_chunks as chunks_repo
from . import documents as docs_repo
from . import chunking, converters, summarize_service
from ...core.services.embeddings import Embedder

log = logging.getLogger("pikaos.engine.ingest")

# Kinds we can turn into embeddable markdown: text-native (md/log) + converted (pdf/docx). Anything
# else (image/other) is marked "skipped" so the UI shows it wasn't indexed.
_EMBEDDABLE_KINDS = converters.CONVERTIBLE_KINDS

# Heading for the doc-level summary chunk (enrich B) — marks it apart from real section chunks so
# retrieval/answer can label it as the document overview.
_SUMMARY_HEADING = "(document summary)"


def _embed_text(title: str, heading: str, content: str) -> str:
    """What we actually embed for a chunk: the document title + section heading prepended to the
    body (enrich A, knowledge-rag.md §6.2), so each chunk carries which document/section it came
    from — high-level queries match the context, not just the bare section text."""
    prefix = " — ".join(p for p in (title, heading) if p)
    return f"{prefix}\n{content}" if prefix else content


async def _markdown_body(db, doc) -> str | None:
    """The markdown text to chunk for `doc`. A pdf/docx is converted on its first ingest: the
    generated markdown is stored as the new truth (`object_key`) and the original is kept as a Ref
    (`source_object_key`) — knowledge-rag.md §6.4. md/log (and already-converted docs, where
    `source_object_key` is set) are read back as text. Returns None when a converted file yields no
    embeddable text (e.g. a scanned PDF → OCR deferred), so the caller marks it `skipped`."""
    raw = await asyncio.to_thread(get_storage().get_object, doc.object_key)
    # object_key already points at markdown: a text-native kind, or a pdf/docx converted before.
    if doc.source_object_key is not None or doc.kind not in converters.CONVERTED_KINDS:
        return raw.decode("utf-8", errors="replace")

    md = converters.to_markdown(doc.kind, raw, doc.name)
    if md is None:
        return None
    md_key = f"{doc.object_key}.md"
    await asyncio.to_thread(get_storage().put_object, md_key, md.encode("utf-8"), "text/markdown")
    await docs_repo.set_converted_markdown(db, doc.id, markdown_key=md_key, source_key=doc.object_key)
    return md


async def ingest_document(db, embedder: Embedder, doc_id: uuid.UUID, *, summarizer=None) -> dict:
    """Chunk + embed one document into `doc_chunks`. Returns `{status, chunks}`.

    Records ingest state on the document throughout so the result is observable without the
    worker logs. Never raises for an expected condition (missing/non-text doc); on an unexpected
    failure it marks the doc `failed` and re-raises so the job is visibly failed.

    `summarizer` (optional, enrich B) is an injected LLM provider: when given, the whole markdown
    is summarized once → stored on `documents.summary` and embedded as an extra summary-chunk so
    high-level queries match the document. None (default) skips B entirely — keeping ingest free
    and offline. Summarization is best-effort: a failure leaves the section chunks intact."""
    doc = await docs_repo.get_document(db, doc_id)
    if doc is None:
        return {"status": "missing", "chunks": 0}

    if doc.kind not in _EMBEDDABLE_KINDS:
        await chunks_repo.delete_for_document(db, doc_id)
        await docs_repo.set_ingest_status(db, doc_id, status="skipped", embedding_model="")
        return {"status": "skipped", "chunks": 0}

    try:
        body = await _markdown_body(db, doc)
        if body is None:
            # A convertible file with no extractable text (e.g. a scanned PDF) → skip until OCR.
            await chunks_repo.delete_for_document(db, doc_id)
            await docs_repo.set_ingest_status(db, doc_id, status="skipped", embedding_model="")
            return {"status": "skipped", "chunks": 0}
        pairs = chunking.chunk_markdown(body, max_chars=settings.embed_chunk_max_chars)

        if not pairs:
            await chunks_repo.delete_for_document(db, doc_id)
            await docs_repo.set_ingest_status(db, doc_id, status="done", embedding_model=embedder.model_name)
            return {"status": "done", "chunks": 0}

        # (heading, content) for each chunk to store + the text we embed for it (enrich A).
        chunk_specs = [(h, c, _embed_text(doc.name, h, c)) for h, c in pairs]

        # Enrich B: a doc-level summary stored on the document and appended as one more chunk.
        summary = await _summarize(summarizer, title=doc.name, markdown=body)
        if summary:
            chunk_specs.append(
                (_SUMMARY_HEADING, summary, _embed_text(doc.name, _SUMMARY_HEADING, summary))
            )

        vectors = await embedder.embed([t for _, _, t in chunk_specs])
        rows = [
            {"seq": i, "heading": h, "content": c, "embedding": vec}
            for i, ((h, c, _), vec) in enumerate(zip(chunk_specs, vectors))
        ]
        n = await chunks_repo.replace_chunks(
            db, document_id=doc.id, owner_id=doc.owner_id, department_id=doc.department_id,
            embedding_model=embedder.model_name, chunks=rows,
        )
        await docs_repo.set_summary(db, doc_id, summary=summary)
        await docs_repo.set_ingest_status(db, doc_id, status="done", embedding_model=embedder.model_name)
        log.info("ingested document %s — %d chunks (model=%s, summary=%s)",
                 doc_id, n, embedder.model_name, bool(summary))
        return {"status": "done", "chunks": n}
    except Exception:
        await docs_repo.set_ingest_status(db, doc_id, status="failed")
        log.exception("ingest failed for document %s", doc_id)
        raise


async def _summarize(summarizer, *, title: str, markdown: str) -> str | None:
    """Enrich B helper: the doc summary via the injected provider, or None when B is off
    (no summarizer) — isolated so the main path reads cleanly."""
    if summarizer is None:
        return None
    return await summarize_service.summarize_document(
        summarizer, title=title, markdown=markdown,
        max_input_chars=settings.ingest_summary_max_input_chars,
    )
