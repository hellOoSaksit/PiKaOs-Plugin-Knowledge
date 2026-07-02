"""Tests for retrieval_service — RAG context for the agent loop (E3).

Pure helpers (query/format) need no DB. The scoping test hits the real DB + pgvector (like
test_doc_chunks): retrieval must only surface chunks the run owner may read, and must stay off
when disabled (k<=0) or there's nothing to ask.

    docker compose exec backend pytest tests/test_retrieval.py
"""
from __future__ import annotations

import asyncio
import uuid

from sqlalchemy import delete as sql_delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.plugins.postgres.engine import register_pgvector
from app.plugins.knowledge.models import Document
from app.plugins.auth.models import Department, User, UserDepartment
from app.plugins.knowledge import doc_chunks as chunks_repo
from app.plugins.knowledge import documents as docs_repo
from app.plugins.auth.security import hash_password
from app.plugins.knowledge import retrieval_service as rs
from app.plugins.knowledge.embeddings import StubEmbedder


# --- pure helpers ---------------------------------------------------------

def test_query_from_input():
    assert rs.query_from_input({"task": "fix the bug"}) == "fix the bug"
    # falls back to the first user message when there's no task
    assert rs.query_from_input({"messages": [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]}) == "hello"
    assert rs.query_from_input({}) == ""
    assert rs.query_from_input(None) == ""


def test_format_context_numbers_and_labels():
    out = rs.format_context([
        {"document_name": "notes.md", "heading": "Setup", "content": "run start.bat"},
        {"document_name": "faq.md", "heading": "", "content": "no heading here"},
    ])
    assert "[1] notes.md — Setup" in out
    assert "run start.bat" in out
    assert "[2] faq.md" in out and "no heading here" in out


# --- scoped retrieval against the real DB ---------------------------------

def test_context_for_run_is_owner_scoped_and_toggleable():
    dept_a, dept_b = uuid.uuid4(), uuid.uuid4()
    member, admin = uuid.uuid4(), uuid.uuid4()
    d_org, d_a, d_b, d_own = (uuid.uuid4() for _ in range(4))
    # (doc_id, department_id, owner_id) — member is in dept_a and owns d_own (which sits in dept_b)
    specs = [(d_org, None, None), (d_a, dept_a, None), (d_b, dept_b, None), (d_own, dept_b, member)]

    async def main():
        eng = register_pgvector(create_async_engine(settings.database_url))
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        emb = StubEmbedder()
        try:
            async with Session() as s:
                s.add_all([
                    Department(id=dept_a, name_th="A", name_en="A"),
                    Department(id=dept_b, name_th="B", name_en="B"),
                    User(id=member, username=f"rg_{member.hex[:8]}", email=f"{member.hex[:8]}@t.io",
                         display="m", role="member", password_hash=hash_password("x")),
                    User(id=admin, username=f"rg_{admin.hex[:8]}", email=f"{admin.hex[:8]}@t.io",
                         display="a", role="admin", password_hash=hash_password("x")),
                    UserDepartment(user_id=member, department_id=dept_a, is_primary=True),
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
                        db, document_id=did, owner_id=owner, department_id=dept, embedding_model="stub",
                        chunks=[{"seq": 0, "heading": "h", "content": f"c-{did}", "embedding": vec}],
                    )

                member_ctx = await rs.context_for_run(db, owner_id=member, query="anything", k=10, embedder=emb)
                admin_ctx = await rs.context_for_run(db, owner_id=admin, query="anything", k=10, embedder=emb)
                noowner_ctx = await rs.context_for_run(db, owner_id=None, query="anything", k=10, embedder=emb)
                disabled = await rs.context_for_run(db, owner_id=admin, query="anything", k=0, embedder=emb)
                empty_q = await rs.context_for_run(db, owner_id=admin, query="   ", k=10, embedder=emb)
                return member_ctx, admin_ctx, noowner_ctx, disabled, empty_q
        finally:
            async with Session() as c:
                await c.execute(sql_delete(Document).where(Document.id.in_([d_org, d_a, d_b, d_own])))
                await c.execute(sql_delete(UserDepartment).where(UserDepartment.user_id == member))
                await c.execute(sql_delete(Department).where(Department.id.in_([dept_a, dept_b])))
                await c.execute(sql_delete(User).where(User.id.in_([member, admin])))
                await c.commit()
            await eng.dispose()

    member_ctx, admin_ctx, noowner_ctx, disabled, empty_q = asyncio.run(main())

    # member (dept_a, owns d_own): sees org-wide + dept_a + own; never dept_b's d_b
    assert f"c-{d_org}" in member_ctx and f"c-{d_a}" in member_ctx and f"c-{d_own}" in member_ctx
    assert f"c-{d_b}" not in member_ctx
    # admin: sees every chunk
    for did in (d_org, d_a, d_b, d_own):
        assert f"c-{did}" in admin_ctx
    # no owner → org-wide only
    assert f"c-{d_org}" in noowner_ctx
    assert f"c-{d_a}" not in noowner_ctx and f"c-{d_b}" not in noowner_ctx
    # disabled (k<=0) and blank query → no retrieval
    assert disabled is None and empty_q is None
