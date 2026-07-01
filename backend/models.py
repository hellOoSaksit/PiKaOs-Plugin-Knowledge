"""Knowledge plugin models — the document store + RAG index this plugin OWNS.

`documents` and `doc_chunks` left Core when the knowledge extraction finished: they live on this
plugin's OWN declarative `Base` (separate metadata from the kernel), created by the plugin's migrate()
step (CREATE EXTENSION vector → create_all → HNSW index), never by Core's Alembic baseline.

Cross-plugin refs are logical UUIDs, NOT foreign keys: `owner_id`/`department_id` point at the auth
plugin's users/departments by id with no DB-level FK across the plugin boundary. The FK BETWEEN this
plugin's own tables (doc_chunks.document_id → documents.id) is kept.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import UserDefinedType

from ...core.config import settings


class Base(DeclarativeBase):
    """This plugin's declarative base — metadata independent of the kernel's `app.core.db.Base`."""


class Vector(UserDefinedType):
    """Minimal mapping for pgvector's `vector(N)` column — DDL/metadata only.

    Reads and writes of embeddings go through raw SQL in doc_chunks.py, binding the embedding as a
    `list[float]` via the official pgvector asyncpg codec (`app.core.db.register_pgvector`). This type
    exists so the ORM/migrations know the column shape; it deliberately has no bind/result processors,
    so the embedding column is never round-tripped through the ORM."""

    cache_ok = True

    def __init__(self, dim: int):
        self.dim = dim

    def get_col_spec(self, **_kw) -> str:
        return f"vector({self.dim})"


class Document(Base):
    """File metadata for MinIO-stored documents. The knowledge store is markdown-as-truth
    (docs/architecture/knowledge-rag.md)."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Logical ref → auth.users.id (no cross-plugin FK): deleting a user leaves docs, ownership goes stale.
    owner_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True, nullable=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="md")  # md|image|log|pdf|other
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    object_key: Mapped[str] = mapped_column(String(512), nullable=False)  # MinIO object path (markdown truth once ingested)
    # The original uploaded file (pdf/word) kept as a Ref after conversion to markdown (E6,
    # knowledge-rag.md §6.4). NULL = the upload was already markdown/text (object_key IS the truth).
    source_object_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False, default="application/octet-stream")
    size: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    # department scoping (logical ref → auth.departments.id, no FK) — system-design §7.1
    department_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True, nullable=True)
    # RAG ingest state — markdown stays the truth; records whether the file has been chunked+embedded
    # into doc_chunks, and with which model (knowledge-rag.md §3).
    ingest_status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")  # pending|done|failed|skipped
    embedding_model: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    # Doc-level summary produced at ingest (enrich B, knowledge-rag.md §6.2). Derived/rebuildable; NULL
    # until summarized (off by default, or no summarize provider configured).
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class DocChunk(Base):
    """RAG semantic index — one heading-bounded slice of a document + its embedding (knowledge-rag.md
    §3). A **derived, rebuildable cache**: chunks are deleted+recreated on re-ingest and cascade-deleted
    with their document (no orphan vectors). owner_id/department_id are denormalized from the document so
    retrieval can scope by permission without a join.

    The `embedding` column is read/written only via raw SQL (doc_chunks.py) — see the Vector docstring.
    The HNSW cosine index on `embedding` is created by migrate.py (raw), not by create_all."""

    __tablename__ = "doc_chunks"
    __table_args__ = (UniqueConstraint("document_id", "seq", name="uq_doc_chunks_document_seq"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    owner_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True, nullable=True)
    department_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True, nullable=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    heading: Mapped[str] = mapped_column(Text, nullable=False, default="")
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embed_dim), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
