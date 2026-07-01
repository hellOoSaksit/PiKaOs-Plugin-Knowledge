"""All SQL for the document / knowledge store (layering §2.1).

The knowledge base is markdown-as-truth (docs/architecture/knowledge-rag.md): the bytes
live in MinIO, this `documents` table is their metadata + owner/department scoping. RAG
embeddings (phase E) get their own table + migration; this layer stays storage-only.
"""
from __future__ import annotations

import uuid

from sqlalchemy import String, column, delete as sql_delete, table
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.models import Document

# Logical cross-plugin reads: `users` + `user_departments` are owned by the auth plugin (Phase C). We
# reference them by table name — no model import, no FK across the plugin boundary (logical-UUID rule).
# Typed columns so asyncpg binds the UUID param correctly.
_users = table("users", column("id", UUID(as_uuid=True)), column("role", String))
_user_departments = table(
    "user_departments", column("user_id", UUID(as_uuid=True)), column("department_id", UUID(as_uuid=True))
)


async def user_role(db: AsyncSession, user_id: uuid.UUID) -> str | None:
    """The user's role by id, or None if unknown (logical read of the auth-owned `users` table)."""
    return (await db.execute(select(_users.c.role).where(_users.c.id == user_id))).scalar_one_or_none()


async def user_department_ids(db: AsyncSession, user_id: uuid.UUID) -> list[uuid.UUID]:
    """The departments a user belongs to — the scope used to filter visible documents."""
    stmt = select(_user_departments.c.department_id).where(_user_departments.c.user_id == user_id)
    return list((await db.execute(stmt)).scalars().all())


def _scope(stmt, dept_ids: list[uuid.UUID] | None):
    """Restrict to documents the scope may see: org-wide (department_id IS NULL) + own depts.
    `dept_ids = None` means no restriction (admin sees everything)."""
    if dept_ids is None:
        return stmt
    return stmt.where(Document.department_id.is_(None) | Document.department_id.in_(dept_ids))


async def insert_document(
    db: AsyncSession, *, doc_id: uuid.UUID, owner_id: uuid.UUID | None,
    department_id: uuid.UUID | None, kind: str, name: str, object_key: str,
    content_type: str, size: int,
) -> Document:
    doc = Document(
        id=doc_id, owner_id=owner_id, department_id=department_id, kind=kind,
        name=name, object_key=object_key, content_type=content_type, size=size,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def get_document(db: AsyncSession, doc_id: uuid.UUID) -> Document | None:
    return await db.get(Document, doc_id)


async def list_documents(
    db: AsyncSession, *, dept_ids: list[uuid.UUID] | None = None,
    kind: str | None = None, limit: int = 50, offset: int = 0,
) -> list[Document]:
    stmt = _scope(select(Document), dept_ids)
    if kind:
        stmt = stmt.where(Document.kind == kind)
    stmt = stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset)
    return list((await db.execute(stmt)).scalars().all())


async def count_documents(
    db: AsyncSession, *, dept_ids: list[uuid.UUID] | None = None, kind: str | None = None,
) -> int:
    stmt = _scope(select(func.count(Document.id)), dept_ids)
    if kind:
        stmt = stmt.where(Document.kind == kind)
    return int((await db.execute(stmt)).scalar_one())


async def ids_for_reindex(
    db: AsyncSession, *, owner_id: uuid.UUID | None = None, exclude_model: str | None = None,
) -> list[uuid.UUID]:
    """Document ids to (re)build the RAG index for — the 'single rebuild command'
    (knowledge-rag.md §3). `owner_id` limits to one owner (a non-admin rebuilding only their
    own); None = the whole corpus (admin). `exclude_model` skips docs already embedded with that
    model — pass the current embedder's model to re-embed only the stale rest after switching
    `embed_provider`; None = rebuild everything. Oldest first for a stable rebuild order."""
    stmt = select(Document.id)
    if owner_id is not None:
        stmt = stmt.where(Document.owner_id == owner_id)
    if exclude_model is not None:
        stmt = stmt.where(Document.embedding_model != exclude_model)
    stmt = stmt.order_by(Document.created_at.asc())
    return list((await db.execute(stmt)).scalars().all())


async def delete_document(db: AsyncSession, doc_id: uuid.UUID) -> bool:
    res = await db.execute(sql_delete(Document).where(Document.id == doc_id))
    await db.commit()
    return res.rowcount > 0


async def set_converted_markdown(
    db: AsyncSession, doc_id: uuid.UUID, *, markdown_key: str, source_key: str
) -> None:
    """Bind a converted pdf/docx (knowledge-rag.md §6.4): point `object_key` at the generated
    markdown (the new RAG truth) and keep the original upload as a Ref in `source_object_key`.
    Tolerates a missing row (the doc may have been deleted mid-ingest)."""
    doc = await db.get(Document, doc_id)
    if doc is None:
        return
    doc.object_key = markdown_key
    doc.source_object_key = source_key
    await db.commit()


async def set_summary(db: AsyncSession, doc_id: uuid.UUID, *, summary: str | None) -> None:
    """Store the doc-level summary produced at ingest (enrich B, knowledge-rag.md §6.2). Derived
    metadata — rebuilt on every ingest. Tolerates a missing row (deleted mid-ingest)."""
    doc = await db.get(Document, doc_id)
    if doc is None:
        return
    doc.summary = summary
    await db.commit()


async def set_ingest_status(
    db: AsyncSession, doc_id: uuid.UUID, *, status: str, embedding_model: str | None = None
) -> None:
    """Record RAG ingest state on the document (pending|done|failed|skipped). Tolerates a
    missing row (the doc may have been deleted mid-ingest)."""
    doc = await db.get(Document, doc_id)
    if doc is None:
        return
    doc.ingest_status = status
    if embedding_model is not None:
        doc.embedding_model = embedding_model
    await db.commit()
