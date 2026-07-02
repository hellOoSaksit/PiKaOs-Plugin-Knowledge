"""Tests for the RAG ingestion service (phase E/M2).

Runs `ingest_document` against the real DB (needs pgvector — migration 0005) with a StubEmbedder
and MinIO stubbed out (monkeypatched), so the full chunk→embed→store path is exercised offline.

    docker compose exec backend pytest tests/test_ingestion.py
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import delete as sql_delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.plugins.minio import storage
from app.core.config import settings
from app.plugins.postgres.engine import register_pgvector
from app.plugins.knowledge.models import Document
from app.plugins.knowledge import doc_chunks as chunks_repo
from app.plugins.knowledge import documents as docs_repo
from app.plugins.knowledge import ingestion_service
from app.plugins.knowledge import storage_ref
from app.plugins.knowledge.embeddings import StubEmbedder

_MD = "# Intro\nhello world\n\n## Details\nmore body text here\n\n## More\nand even more"


@pytest.fixture(autouse=True)
def _wire_storage_ref():
    """Ingestion resolves storage via `storage_ref.get_storage()`; point it at the same
    `minio.storage` module these tests monkeypatch, so the patched functions are what
    the service actually calls (see storage_ref.py — raises if never wired)."""
    storage_ref.set_storage(storage)
    yield
    storage_ref.set_storage(None)


def test_ingest_markdown_creates_chunks(monkeypatch):
    did = uuid.uuid4()
    monkeypatch.setattr(storage, "get_object", lambda key: _MD.encode("utf-8"))

    async def main():
        eng = register_pgvector(create_async_engine(settings.database_url))
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            async with Session() as db:
                await docs_repo.insert_document(
                    db, doc_id=did, owner_id=None, department_id=None, kind="md",
                    name="notes.md", object_key=f"k/{did}", content_type="text/markdown", size=len(_MD),
                )
                result = await ingestion_service.ingest_document(db, StubEmbedder(), did)
                n = await chunks_repo.count_for_document(db, did)
                doc = await docs_repo.get_document(db, did)
                return result, n, doc.ingest_status, doc.embedding_model
        finally:
            async with Session() as c:
                await c.execute(sql_delete(Document).where(Document.id == did))
                await c.commit()
            await eng.dispose()

    result, n, status, model = asyncio.run(main())
    assert result == {"status": "done", "chunks": 3}     # 3 headings with bodies
    assert n == 3
    assert status == "done" and model == "stub"


def test_ingest_is_idempotent_replace(monkeypatch):
    """Re-ingesting replaces chunks rather than appending (rebuildable cache)."""
    did = uuid.uuid4()
    monkeypatch.setattr(storage, "get_object", lambda key: _MD.encode("utf-8"))

    async def main():
        eng = register_pgvector(create_async_engine(settings.database_url))
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            async with Session() as db:
                await docs_repo.insert_document(
                    db, doc_id=did, owner_id=None, department_id=None, kind="md",
                    name="n.md", object_key=f"k/{did}", content_type="text/markdown", size=1,
                )
                await ingestion_service.ingest_document(db, StubEmbedder(), did)
                await ingestion_service.ingest_document(db, StubEmbedder(), did)   # twice
                return await chunks_repo.count_for_document(db, did)
        finally:
            async with Session() as c:
                await c.execute(sql_delete(Document).where(Document.id == did))
                await c.commit()
            await eng.dispose()

    assert asyncio.run(main()) == 3      # not 6 — replaced, not appended


def test_ingest_pdf_converts_to_markdown_and_binds_ref(monkeypatch):
    """A pdf is converted to markdown (the new truth) on ingest; the original is kept as a Ref
    (knowledge-rag.md §6.4). Conversion itself (pypdf) is stubbed — this asserts the wiring."""
    did = uuid.uuid4()
    original_key = f"documents/{did}/report.pdf"
    puts: dict[str, bytes] = {}
    monkeypatch.setattr(storage, "get_object", lambda key: b"%PDF-fake-bytes")
    monkeypatch.setattr(
        storage, "put_object",
        lambda key, data, content_type="application/octet-stream": puts.update({key: data}),
    )
    monkeypatch.setattr(ingestion_service.converters, "to_markdown",
                        lambda kind, data, name=None: _MD)

    async def main():
        eng = register_pgvector(create_async_engine(settings.database_url))
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            async with Session() as db:
                await docs_repo.insert_document(
                    db, doc_id=did, owner_id=None, department_id=None, kind="pdf",
                    name="report.pdf", object_key=original_key,
                    content_type="application/pdf", size=16,
                )
                result = await ingestion_service.ingest_document(db, StubEmbedder(), did)
                doc = await docs_repo.get_document(db, did)
                return result, doc.object_key, doc.source_object_key
        finally:
            async with Session() as c:
                await c.execute(sql_delete(Document).where(Document.id == did))
                await c.commit()
            await eng.dispose()

    result, object_key, source = asyncio.run(main())
    assert result == {"status": "done", "chunks": 3}
    assert source == original_key                   # original kept as a Ref
    assert object_key == f"{original_key}.md"       # truth is now the generated markdown
    assert object_key in puts                       # markdown was written to storage


def test_ingest_pdf_without_text_is_skipped(monkeypatch):
    """A scanned PDF (no extractable text) → converter returns None → skipped until OCR."""
    did = uuid.uuid4()
    monkeypatch.setattr(storage, "get_object", lambda key: b"%PDF-scanned")
    monkeypatch.setattr(storage, "put_object",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no md to write")))
    monkeypatch.setattr(ingestion_service.converters, "to_markdown",
                        lambda kind, data, name=None: None)

    async def main():
        eng = register_pgvector(create_async_engine(settings.database_url))
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            async with Session() as db:
                await docs_repo.insert_document(
                    db, doc_id=did, owner_id=None, department_id=None, kind="pdf",
                    name="scan.pdf", object_key=f"documents/{did}/scan.pdf",
                    content_type="application/pdf", size=8,
                )
                result = await ingestion_service.ingest_document(db, StubEmbedder(), did)
                doc = await docs_repo.get_document(db, did)
                return result, doc.ingest_status, doc.source_object_key
        finally:
            async with Session() as c:
                await c.execute(sql_delete(Document).where(Document.id == did))
                await c.commit()
            await eng.dispose()

    result, status, source = asyncio.run(main())
    assert result == {"status": "skipped", "chunks": 0} and status == "skipped"
    assert source is None                            # nothing converted → no Ref bound


def test_ingest_skips_non_text_kind(monkeypatch):
    did = uuid.uuid4()
    # get_object must not even be called for a skipped kind
    monkeypatch.setattr(storage, "get_object",
                        lambda key: (_ for _ in ()).throw(AssertionError("should skip")))

    async def main():
        eng = register_pgvector(create_async_engine(settings.database_url))
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            async with Session() as db:
                await docs_repo.insert_document(
                    db, doc_id=did, owner_id=None, department_id=None, kind="image",
                    name="p.png", object_key=f"k/{did}", content_type="image/png", size=1,
                )
                result = await ingestion_service.ingest_document(db, StubEmbedder(), did)
                doc = await docs_repo.get_document(db, did)
                return result, doc.ingest_status
        finally:
            async with Session() as c:
                await c.execute(sql_delete(Document).where(Document.id == did))
                await c.commit()
            await eng.dispose()

    result, status = asyncio.run(main())
    assert result == {"status": "skipped", "chunks": 0} and status == "skipped"


class _SummaryResult:
    def __init__(self, text):
        self.text = text


class _FakeSummarizer:
    """An injected summarize provider (enrich B) — returns a fixed summary."""

    def __init__(self, text="This document is about greetings and details."):
        self.text = text

    async def complete(self, *, model, messages, tools):
        return _SummaryResult(self.text)


class _BoomSummarizer:
    async def complete(self, *, model, messages, tools):
        raise RuntimeError("summarize provider down")


def test_ingest_with_summarizer_adds_summary_chunk(monkeypatch):
    """Enrich B: a summarizer → one extra summary-chunk + documents.summary stored."""
    did = uuid.uuid4()
    monkeypatch.setattr(storage, "get_object", lambda key: _MD.encode("utf-8"))

    async def main():
        eng = register_pgvector(create_async_engine(settings.database_url))
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            async with Session() as db:
                await docs_repo.insert_document(
                    db, doc_id=did, owner_id=None, department_id=None, kind="md",
                    name="notes.md", object_key=f"k/{did}", content_type="text/markdown", size=len(_MD),
                )
                result = await ingestion_service.ingest_document(
                    db, StubEmbedder(), did, summarizer=_FakeSummarizer())
                n = await chunks_repo.count_for_document(db, did)
                doc = await docs_repo.get_document(db, did)
                return result, n, doc.summary
        finally:
            async with Session() as c:
                await c.execute(sql_delete(Document).where(Document.id == did))
                await c.commit()
            await eng.dispose()

    result, n, summary = asyncio.run(main())
    assert result == {"status": "done", "chunks": 4}     # 3 section chunks + 1 summary chunk
    assert n == 4
    assert summary == "This document is about greetings and details."


def test_ingest_summarizer_failure_still_ingests(monkeypatch):
    """Best-effort B: a failing summarizer leaves the section chunks intact, summary unset."""
    did = uuid.uuid4()
    monkeypatch.setattr(storage, "get_object", lambda key: _MD.encode("utf-8"))

    async def main():
        eng = register_pgvector(create_async_engine(settings.database_url))
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            async with Session() as db:
                await docs_repo.insert_document(
                    db, doc_id=did, owner_id=None, department_id=None, kind="md",
                    name="n.md", object_key=f"k/{did}", content_type="text/markdown", size=1,
                )
                result = await ingestion_service.ingest_document(
                    db, StubEmbedder(), did, summarizer=_BoomSummarizer())
                doc = await docs_repo.get_document(db, did)
                return result, doc.summary
        finally:
            async with Session() as c:
                await c.execute(sql_delete(Document).where(Document.id == did))
                await c.commit()
            await eng.dispose()

    result, summary = asyncio.run(main())
    assert result == {"status": "done", "chunks": 3}     # summary skipped → 3, not 4
    assert summary is None


def test_ingest_missing_document_is_noop(monkeypatch):
    monkeypatch.setattr(storage, "get_object", lambda key: b"")

    async def main():
        eng = register_pgvector(create_async_engine(settings.database_url))
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            async with Session() as db:
                return await ingestion_service.ingest_document(db, StubEmbedder(), uuid.uuid4())
        finally:
            await eng.dispose()

    assert asyncio.run(main()) == {"status": "missing", "chunks": 0}
