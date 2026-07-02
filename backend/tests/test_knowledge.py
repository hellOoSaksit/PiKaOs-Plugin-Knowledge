"""Tests for the M1 knowledge / document store (phase E storage layer).

* Pure helpers (object-key/kind/scoping) → driven directly, no I/O.
* Department scoping of `list_documents` hits the real DB via a fresh engine inside
  asyncio.run (same pattern as test_engine_stubs — sidesteps the module-level-engine
  event-loop issue). MinIO/router live-path is exercised end-to-end by hand / later.

    docker compose exec backend pytest tests/test_knowledge.py
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

from sqlalchemy import delete as sql_delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.plugins.knowledge.models import Document
from app.plugins.auth.models import Department, User
from app.plugins.knowledge import documents as docs_repo
from app.plugins.knowledge import knowledge_service as ks


# --- pure helpers -----------------------------------------------------------


def test_safe_name_strips_unsafe_and_never_empty():
    assert ks.safe_name("My Notes (v2).md") == "My_Notes_v2.md"  # space→_, () stripped
    assert ks.safe_name("  ") == "file"
    assert ks.safe_name(None) == "file"


def test_build_object_key_namespaced_by_id():
    did = uuid.uuid4()
    assert ks.build_object_key(did, "a b.md") == f"documents/{did}/a_b.md"


def test_infer_kind():
    assert ks.infer_kind("text/markdown", "x") == "md"
    assert ks.infer_kind(None, "NOTES.MD") == "md"
    assert ks.infer_kind("image/png", "p.png") == "image"
    assert ks.infer_kind("application/pdf", "r.pdf") == "pdf"
    assert ks.infer_kind(None, "run.log") == "log"
    assert ks.infer_kind("application/octet-stream", "blob.bin") == "other"


# --- scope helpers (can_view / can_manage) ----------------------------------


def _user(role="member", uid=None):
    return SimpleNamespace(role=role, id=uid or uuid.uuid4())


def _doc(owner_id=None, department_id=None):
    return SimpleNamespace(owner_id=owner_id, department_id=department_id)


def test_can_view_admin_sees_all():
    assert ks.can_view(_user("admin"), _doc(department_id=uuid.uuid4()), []) is True


def test_can_view_owner_sees_own_even_other_dept():
    u = _user()
    assert ks.can_view(u, _doc(owner_id=u.id, department_id=uuid.uuid4()), []) is True


def test_can_view_org_wide_doc():
    assert ks.can_view(_user(), _doc(department_id=None), []) is True


def test_can_view_dept_member_only():
    dept = uuid.uuid4()
    assert ks.can_view(_user(), _doc(department_id=dept), [dept]) is True
    assert ks.can_view(_user(), _doc(department_id=dept), []) is False  # not a member


def test_can_manage_owner_or_admin():
    u = _user()
    assert ks.can_manage(u, _doc(owner_id=u.id)) is True
    assert ks.can_manage(u, _doc(owner_id=uuid.uuid4())) is False
    assert ks.can_manage(_user("admin"), _doc(owner_id=uuid.uuid4())) is True


# --- department scoping of list_documents (real DB) -------------------------


def test_list_documents_scopes_by_department():
    dept_a, dept_b = uuid.uuid4(), uuid.uuid4()
    d_org, d_a, d_b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    async def main():
        eng = create_async_engine(settings.database_url)
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            async with Session() as s:
                s.add_all([
                    Department(id=dept_a, name_th="A", name_en="A"),
                    Department(id=dept_b, name_th="B", name_en="B"),
                ])
                await s.commit()
            async with Session() as db:
                for did, dept in ((d_org, None), (d_a, dept_a), (d_b, dept_b)):
                    await docs_repo.insert_document(
                        db, doc_id=did, owner_id=None, department_id=dept, kind="md",
                        name="n", object_key=f"k/{did}", content_type="text/markdown", size=1,
                    )
                in_a = {d.id for d in await docs_repo.list_documents(db, dept_ids=[dept_a])}
                all_ids = {d.id for d in await docs_repo.list_documents(db, dept_ids=None)}
                n_a = await docs_repo.count_documents(db, dept_ids=[dept_a])
                return in_a, all_ids, n_a
        finally:
            async with Session() as c:
                await c.execute(sql_delete(Document).where(Document.id.in_([d_org, d_a, d_b])))
                await c.execute(sql_delete(Department).where(Department.id.in_([dept_a, dept_b])))
                await c.commit()
            await eng.dispose()

    in_a, all_ids, n_a = asyncio.run(main())
    # scope [dept_a] sees org-wide + own dept, never another dept's doc
    assert d_org in in_a and d_a in in_a and d_b not in in_a
    assert n_a == 2
    # admin scope (dept_ids=None) sees everything
    assert {d_org, d_a, d_b} <= all_ids


# --- RAG reindex targets — the 'single rebuild command' (knowledge-rag.md §3, E5) -------


def test_reindex_targets_scope_and_stale_filter():
    ua, ub = uuid.uuid4(), uuid.uuid4()
    # A owns three docs on different embedding models; B owns one (never embedded).
    a_blank, a_stub, a_bge, b_blank = (uuid.uuid4() for _ in range(4))

    async def main():
        eng = create_async_engine(settings.database_url)
        Session = async_sessionmaker(eng, expire_on_commit=False, class_=AsyncSession)
        try:
            async with Session() as s:
                s.add_all([
                    User(id=ua, username=f"ra_{ua.hex[:8]}", email=f"{ua.hex[:8]}@t", password_hash="x"),
                    User(id=ub, username=f"rb_{ub.hex[:8]}", email=f"{ub.hex[:8]}@t", password_hash="x"),
                ])
                await s.commit()
            async with Session() as db:
                for did, owner, model in (
                    (a_blank, ua, ""), (a_stub, ua, "stub"), (a_bge, ua, "bge-m3"), (b_blank, ub, ""),
                ):
                    await docs_repo.insert_document(
                        db, doc_id=did, owner_id=owner, department_id=None, kind="md",
                        name="n", object_key=f"k/{did}", content_type="text/markdown", size=1,
                    )
                    if model:
                        await docs_repo.set_ingest_status(db, did, status="done", embedding_model=model)

                # repo: exclude the current model → only stale docs (whole corpus)
                stale_all = set(await docs_repo.ids_for_reindex(db, exclude_model="bge-m3"))
                # repo: scope to one owner + exclude current model
                stale_a = set(await docs_repo.ids_for_reindex(db, owner_id=ua, exclude_model="bge-m3"))
                # repo: full rebuild for one owner (no exclude)
                all_a = set(await docs_repo.ids_for_reindex(db, owner_id=ua))

                # service: admin rebuilds the corpus; a member only their own
                admin = SimpleNamespace(role="admin", id=uuid.uuid4())
                member = SimpleNamespace(role="member", id=ua)
                svc_admin = set(await ks.reindex_targets(db, user=admin, only_stale=True, current_model="bge-m3"))
                svc_member = set(await ks.reindex_targets(db, user=member, only_stale=True, current_model="bge-m3"))
                svc_member_full = set(await ks.reindex_targets(db, user=member, only_stale=False, current_model="bge-m3"))
                return stale_all, stale_a, all_a, svc_admin, svc_member, svc_member_full
        finally:
            async with Session() as c:
                await c.execute(sql_delete(Document).where(Document.id.in_([a_blank, a_stub, a_bge, b_blank])))
                await c.execute(sql_delete(User).where(User.id.in_([ua, ub])))
                await c.commit()
            await eng.dispose()

    stale_all, stale_a, all_a, svc_admin, svc_member, svc_member_full = asyncio.run(main())
    # stale filter drops only docs already on the current model (a_bge)
    assert {a_blank, a_stub, b_blank} <= stale_all and a_bge not in stale_all
    # owner scope keeps only A's docs, still minus the current-model one
    assert stale_a == {a_blank, a_stub}
    # full rebuild for A = all of A's docs regardless of model
    assert all_a == {a_blank, a_stub, a_bge}
    # service: admin sees every stale doc (incl. B's); member only their own stale ones
    assert {a_blank, a_stub, b_blank} <= svc_admin and a_bge not in svc_admin
    assert svc_member == {a_blank, a_stub}
    assert svc_member_full == {a_blank, a_stub, a_bge}
