"""SQL for `doc_chunks` — the RAG semantic index (phase E/M2, knowledge-rag.md §3).

All embedding reads/writes are **raw SQL** here (layering §2.1 — SQL lives in repositories). The
embedding itself is bound as a plain `list[float]`: the official **pgvector** asyncpg codec is
registered on every connection (`db.register_pgvector`), so the driver encodes/decodes the `vector`
type for us — no `'[..]'::vector` string literal, no manual float formatting.

Chunks are a rebuildable cache: `replace_chunks` deletes+reinserts a document's chunks atomically
(re-ingest after an edit), and the FK cascade removes them when the document is deleted — so there
are never orphan vectors (knowledge-rag.md §3 / phase-E acceptance criterion).
"""
from __future__ import annotations

import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def replace_chunks(
    db: AsyncSession, *, document_id: uuid.UUID, owner_id: uuid.UUID | None,
    department_id: uuid.UUID | None, embedding_model: str, chunks: list[dict],
) -> int:
    """Replace all chunks of a document (delete old, insert new) in one transaction.
    Each chunk is `{seq, heading, content, embedding}`. Returns the number inserted."""
    await db.execute(text("DELETE FROM doc_chunks WHERE document_id = :doc"), {"doc": document_id})
    insert = text(
        "INSERT INTO doc_chunks "
        "(id, document_id, owner_id, department_id, seq, heading, content, embedding, embedding_model) "
        "VALUES (:id, :doc, :owner, :dept, :seq, :heading, :content, :emb, :model)"
    )
    for ch in chunks:
        await db.execute(insert, {
            "id": uuid.uuid4(), "doc": document_id, "owner": owner_id, "dept": department_id,
            "seq": ch["seq"], "heading": ch.get("heading", ""), "content": ch.get("content", ""),
            "emb": ch["embedding"], "model": embedding_model,
        })
    await db.commit()
    return len(chunks)


async def delete_for_document(db: AsyncSession, document_id: uuid.UUID) -> int:
    res = await db.execute(text("DELETE FROM doc_chunks WHERE document_id = :doc"), {"doc": document_id})
    await db.commit()
    return res.rowcount


async def count_for_document(db: AsyncSession, document_id: uuid.UUID) -> int:
    row = await db.execute(
        text("SELECT count(*) FROM doc_chunks WHERE document_id = :doc"), {"doc": document_id}
    )
    return int(row.scalar_one())


async def search(
    db: AsyncSession, *, embedding: list[float], dept_ids: list[uuid.UUID] | None,
    owner_id: uuid.UUID | None, k: int,
) -> list[dict]:
    """Top-k chunks by cosine similarity, scoped to what the caller may read.

    `dept_ids=None` means no scope filter (admin sees all). Otherwise a chunk is visible when it
    is org-wide (department_id IS NULL), in one of the caller's departments, or owned by them —
    the same rule as knowledge_service.can_view, enforced in SQL so retrieval can't leak scope."""
    params: dict = {"q": embedding, "k": int(k)}
    where = ""
    if dept_ids is not None:
        conds = ["c.department_id IS NULL"]
        if dept_ids:
            conds.append("c.department_id = ANY(:dept_ids)")
            params["dept_ids"] = dept_ids
        if owner_id is not None:
            conds.append("c.owner_id = :owner_id")
            params["owner_id"] = owner_id
        where = "WHERE (" + " OR ".join(conds) + ")"
    sql = text(
        f"""
        SELECT c.id, c.document_id, c.seq, c.heading, c.content,
               d.name AS document_name, d.kind AS document_kind,
               1 - (c.embedding <=> :q) AS score
        FROM doc_chunks c
        JOIN documents d ON d.id = c.document_id
        {where}
        ORDER BY c.embedding <=> :q
        LIMIT :k
        """
    )
    rows = (await db.execute(sql, params)).mappings().all()
    return [dict(r) for r in rows]
