"""Knowledge plugin's arq jobs — contributed to the worker via the package's `jobs` list, so the worker
(Core/engine) runs them when this plugin is enabled WITHOUT importing the plugin (plugin-architecture.md
§5/§10; the worker collects jobs through the Loader's dynamic import). Moved here from app/worker.py in
Phase 3 — a plugin owns its own background work, the engine just provides the runtime to run it on.
"""
from __future__ import annotations

import uuid

from ...core.config import settings
from .embeddings import get_embedder
from . import db_ref, ingestion_service, llm_ref


async def ingest_document(ctx, doc_id: str) -> str:
    """arq job: chunk + embed one document into the RAG index (E2). The embedder is resolved from config
    per job (stub by default), so flipping `embed_provider` needs no code change. When `ingest_summary_enabled`
    (E7 enrich B) the doc is also summarized via the 'summarize' role — best-effort, off by default so
    ingest stays free/offline. On success emits `knowledge.ingested` on the Event Bus (§5) when the worker
    wired one into the arq context — a no-op today (no listeners) but the published contract is live."""
    embedder = get_embedder()
    summarizer = llm_ref.provider_for("summarize") if settings.ingest_summary_enabled else None
    async with db_ref.new_session() as db:
        result = await ingestion_service.ingest_document(
            db, embedder, uuid.UUID(doc_id), summarizer=summarizer
        )
    bus = ctx.get("bus") if isinstance(ctx, dict) else None
    if bus is not None and result.get("status") == "done":
        await bus.emit("knowledge.ingested", {"doc_id": doc_id, "chunks": result.get("chunks")})
    return result["status"]
