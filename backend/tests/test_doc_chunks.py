"""Tests for the doc_chunks repo — pgvector search, scoping, and cascade (phase E/M2).

Hits the real DB (needs the `vector` extension from migration 0005) via a fresh engine inside
asyncio.run — same pattern as test_knowledge. Proves the three things the RAG index must get right:
retrieval is scoped to what the caller may read, and deleting a document removes its chunks (no
orphan vectors).

    docker compose exec backend pytest tests/test_doc_chunks.py
"""
from __future__ import annotations

import asyncio
import uuid

from sqlalchemy import delete as sql_delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.plugins.postgres.engine import register_pgvector
from app.plugins.knowledge.models import Document
from app.plugins.auth.models import Department, User
from app.plugins.knowledge import doc_chunks as chunks_repo
from app.plugins.knowledge import documents as docs_repo
from app.plugins.auth.security import hash_password
from app.plugins.knowledge.embeddings import StubEmbedder


def test_search_is_scoped_and_delete_cascades():
    dept_a, dept_b = uuid.uuid4(), uuid.uuid4()
    uid = uuid.uuid4()
    d_org, d_a, d_b, d_own = (uuid.uuid4() for _ in range(4))
    # (doc_id, department_id, owner_id)
    specs = [(d_org, None, None), (d_a, dept_a, None), (d_b, dept_b, None), (d_own, dept_b, uid)]

    async def main():
        eng = register_pgvector(create_async_engine(settings.database_url))
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        emb = StubEmbedder()
        try:
            async with Session() as s:
                s.add_all([
                    Department(id=dept_a, name_th="A", name_en="A"),
                    Department(id=dept_b, name_th="B", name_en="B"),
                    User(id=uid, username=f"rag_{uid.hex[:8]}", email=f"{uid.hex[:8]}@t.io",
                         display="t", password_hash=hash_password("x")),
                ])
                await s.commit()

            async with Session() as db:
                for did, dept, owner in specs:
                    await docs_repo.insert_document(
                        db, doc_id=did, owner_id=owner, department_id=dept, kind="md",
                        name=str(did), object_key=f"k/{did}", content_type="text/markdown", size=1,
                    )
                vecs = await emb.embed([f"c-{did}" for did, _, _ in specs])
                for (did, dept, owner), vec in zip(specs, vecs):
                    await chunks_repo.replace_chunks(
                        db, document_id=did, owner_id=owner, department_id=dept,
                        embedding_model="stub",
                        chunks=[{"seq": 0, "heading": "h", "content": f"c-{did}", "embedding": vec}],
                    )

                qvec = (await emb.embed(["anything"]))[0]
                seen_a = {r["document_id"] for r in await chunks_repo.search(
                    db, embedding=qvec, dept_ids=[dept_a], owner_id=uid, k=10)}
                seen_admin = {r["document_id"] for r in await chunks_repo.search(
                    db, embedding=qvec, dept_ids=None, owner_id=None, k=10)}

                # cascade: removing the document removes its chunk (FK ON DELETE CASCADE)
                await db.execute(sql_delete(Document).where(Document.id == d_a))
                await db.commit()
                n_after = await chunks_repo.count_for_document(db, d_a)
                return seen_a, seen_admin, n_after
        finally:
            async with Session() as c:
                await c.execute(sql_delete(Document).where(Document.id.in_([d_org, d_a, d_b, d_own])))
                await c.execute(sql_delete(Department).where(Department.id.in_([dept_a, dept_b])))
                await c.execute(sql_delete(User).where(User.id == uid))
                await c.commit()
            await eng.dispose()

    seen_a, seen_admin, n_after = asyncio.run(main())
    # dept_a member who owns d_own: sees org-wide + dept_a + their own doc, never dept_b's
    assert {d_org, d_a, d_own} <= seen_a
    assert d_b not in seen_a
    # admin (no scope) sees every chunk
    assert {d_org, d_a, d_b, d_own} <= seen_admin
    # deleting the document cascaded its chunk away
    assert n_after == 0


def test_search_ranks_exact_match_first():
    did = uuid.uuid4()

    async def main():
        eng = register_pgvector(create_async_engine(settings.database_url))
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        emb = StubEmbedder()
        try:
            async with Session() as db:
                await docs_repo.insert_document(
                    db, doc_id=did, owner_id=None, department_id=None, kind="md",
                    name="n", object_key=f"k/{did}", content_type="text/markdown", size=1,
                )
                contents = ["apples and oranges", "the quick brown fox", "lorem ipsum dolor"]
                vecs = await emb.embed(contents)
                await chunks_repo.replace_chunks(
                    db, document_id=did, owner_id=None, department_id=None, embedding_model="stub",
                    chunks=[{"seq": i, "heading": "", "content": c, "embedding": v}
                            for i, (c, v) in enumerate(zip(contents, vecs))],
                )
                # query identical to chunk[1]'s text → its (identical) vector ranks first
                qvec = (await emb.embed(["the quick brown fox"]))[0]
                top = await chunks_repo.search(db, embedding=qvec, dept_ids=None, owner_id=None, k=3)
                return top[0]["content"], top[0]["score"]
        finally:
            async with Session() as c:
                await c.execute(sql_delete(Document).where(Document.id == did))
                await c.commit()
            await eng.dispose()

    content, score = asyncio.run(main())
    assert content == "the quick brown fox"
    assert score > 0.99   # identical vectors → cosine similarity ≈ 1
