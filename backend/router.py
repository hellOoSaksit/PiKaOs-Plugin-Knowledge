"""Knowledge / codex HTTP routes — the document store (markdown-as-truth).

Thin edge over services/knowledge_service (§2.1): parse the request → call the service →
shape the response / map domain errors to HTTP. Permission split: reads require `codex.view`
(then department-scoped in the service), upload/reindex require `codex.manage`, and deleting a
document requires `codex.delete`. RAG search lands here later (phase E) as `GET /search`.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...core import queue
from ...core.config import settings
from ...core.db import get_db
from ...core.deps import require_perm
from ...core.models import User
from ...core.schemas import (
    DocumentListOut,
    DocumentOut,
    KnowledgeAnswerIn,
    KnowledgeAnswerOut,
    KnowledgeReindexOut,
    KnowledgeSearchOut,
    KnowledgeSearchResult,
)
from . import answer_service, knowledge_service
from ...core.services.embeddings import get_embedder
from ...core.services.llm_config_service import ConfiguredLLMProvider

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

# Refuse files large enough to blow up memory / MinIO on a single dev box. Tune later.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


@router.post("/docs", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    department_id: uuid.UUID | None = Form(default=None),
    user: User = Depends(require_perm("codex.manage")),
    db: AsyncSession = Depends(get_db),
) -> DocumentOut:
    """Store an uploaded file in the codex (MinIO + metadata row)."""
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"file too large (> {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
    doc = await knowledge_service.create_document(
        db, user=user, data=data, name=file.filename,
        content_type=file.content_type, department_id=department_id,
    )
    # Index it for RAG in the background (E2). Best-effort: a Redis outage leaves the file
    # stored with ingest_status="pending" — it just isn't searchable until re-ingested.
    await queue.enqueue("ingest_document", str(doc.id))
    return DocumentOut.model_validate(doc)


@router.get("/search", response_model=KnowledgeSearchOut)
async def search_documents(
    q: str,
    k: int = 0,
    user: User = Depends(require_perm("codex.view")),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeSearchOut:
    """Semantic search over the codex (RAG retrieval). Returns the top-k chunks the caller may
    read, ranked by similarity. Any authenticated user; scope is enforced in the service."""
    k = settings.embed_search_top_k if k <= 0 else max(1, min(k, 50))
    results = await knowledge_service.search_documents(
        db, embedder=get_embedder(), user=user, query=q, k=k
    )
    return KnowledgeSearchOut(items=[KnowledgeSearchResult(**r) for r in results])


@router.post("/answer", response_model=KnowledgeAnswerOut)
async def answer_question(
    body: KnowledgeAnswerIn,
    user: User = Depends(require_perm("codex.view")),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeAnswerOut:
    """Ask a question and get an answer synthesized from the codex, with citations (E8). Retrieves
    the top-k chunks the caller may read (same scope as /search), then the 'answer'-role LLM writes
    the reply citing them as [n]. Any authenticated user; the answer model is config-driven (falls
    back to the stub offline). `k<=0` uses the server default (`rag_answer_top_k`)."""
    k = settings.rag_answer_top_k if body.k <= 0 else max(1, min(body.k, 50))
    result = await answer_service.answer_question(
        db, embedder=get_embedder(), provider=ConfiguredLLMProvider(role="answer"),
        user=user, question=body.question, k=k, rewrite=settings.rag_answer_rewrite,
    )
    return KnowledgeAnswerOut(**result)


@router.post("/reindex", response_model=KnowledgeReindexOut)
async def reindex_documents(
    only_stale: bool = True,
    user: User = Depends(require_perm("codex.manage")),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeReindexOut:
    """Rebuild the RAG index from the markdown source (knowledge-rag.md §3 'single rebuild
    command'). Re-enqueues ingest for each in-scope document — use after switching the embedder
    (`embed_provider`) so existing docs get re-embedded with the new model. Admin rebuilds the
    whole corpus; otherwise only the caller's own docs. `only_stale=true` (default) skips docs
    already on the current model; `false` forces a full rebuild. Idempotent — ingest replaces a
    document's chunks, never appends, so re-running is safe."""
    model = get_embedder().model_name
    ids = await knowledge_service.reindex_targets(
        db, user=user, only_stale=only_stale, current_model=model
    )
    queued = 0
    for doc_id in ids:
        if await queue.enqueue("ingest_document", str(doc_id)):
            queued += 1
    return KnowledgeReindexOut(queued=queued, matched=len(ids), model=model)


@router.get("/docs", response_model=DocumentListOut)
async def list_documents(
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
    user: User = Depends(require_perm("codex.view")),
    db: AsyncSession = Depends(get_db),
) -> DocumentListOut:
    """Documents visible to the caller (own + department + org-wide), newest first."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    items, total = await knowledge_service.list_documents(
        db, user=user, kind=kind, limit=limit, offset=offset
    )
    return DocumentListOut(items=[DocumentOut.model_validate(d) for d in items], total=total)


@router.get("/docs/{doc_id}", response_model=DocumentOut)
async def get_document(
    doc_id: uuid.UUID,
    user: User = Depends(require_perm("codex.view")),
    db: AsyncSession = Depends(get_db),
) -> DocumentOut:
    """Document metadata + a presigned download URL."""
    try:
        doc, url = await knowledge_service.get_document_with_url(db, user=user, doc_id=doc_id)
    except knowledge_service.NotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    except knowledge_service.Forbidden:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "forbidden")
    out = DocumentOut.model_validate(doc)
    out.url = url
    return out


@router.delete("/docs/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: uuid.UUID,
    user: User = Depends(require_perm("codex.delete")),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a document (owner or admin)."""
    try:
        await knowledge_service.delete_document(db, user=user, doc_id=doc_id)
    except knowledge_service.NotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    except knowledge_service.Forbidden:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "forbidden")
