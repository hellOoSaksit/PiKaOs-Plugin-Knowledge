"""RAG retrieval for the agent loop (E3, knowledge-rag.md §3).

Fetch the top-k codex chunks relevant to a run and format them as a context block the LLM can
cite. Reuses the knowledge vector index ([`repositories/doc_chunks.search`]) and the config-driven
embedder ([`services/embeddings.get_embedder`]), scoped to what the run's owner may read — exactly
like `knowledge_service.search_documents`, but keyed by `owner_id` (the agent loop has no `User`).

Read-only and side-effect-free: retrieval adds no `run_step` and consumes no token quota, so it
stays outside the engine's replay/quota guarantees and is safe to re-derive on every resume.
No FastAPI types (called from the worker loop) and no raw SQL here — repositories own that (§2.1).
"""
from __future__ import annotations

import uuid

from . import doc_chunks as chunks_repo
from . import documents as docs_repo
from .embeddings import Embedder, get_embedder


def query_from_input(run_input: dict | None) -> str:
    """The text to retrieve against — the run's `task`, else its first user message."""
    run_input = run_input or {}
    if run_input.get("task"):
        return str(run_input["task"])
    for m in run_input.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def format_context(rows: list[dict]) -> str:
    """Render retrieved chunks as a numbered context block the LLM can cite by [n]."""
    lines = ["Relevant knowledge from the codex (cite as [n] when used):"]
    for i, r in enumerate(rows, start=1):
        head = (r.get("heading") or "").strip()
        doc = r.get("document_name") or ""
        label = f"{doc} — {head}" if head else doc
        lines.append(f"\n[{i}] {label}\n{(r.get('content') or '').strip()}")
    return "\n".join(lines)


async def context_for_run(
    db, *, owner_id: uuid.UUID | None, query: str, k: int, embedder: Embedder | None = None,
) -> str | None:
    """Top-k codex chunks relevant to `query`, scoped to what `owner_id` may read, formatted as a
    context string — or `None` when disabled (`k<=0`) / no query / nothing matches.

    Scope mirrors `knowledge_service.can_view`: admin sees all; everyone else sees org-wide docs
    (`department_id IS NULL`) ∪ their departments ∪ their own. A run with no owner sees org-wide only.
    """
    query = (query or "").strip()
    if k <= 0 or not query:
        return None
    role = await docs_repo.user_role(db, owner_id) if owner_id else None
    if role == "admin":
        dept_ids, owner = None, None            # admin → no scope filter
    else:
        dept_ids = await docs_repo.user_department_ids(db, owner_id) if owner_id else []
        owner = owner_id
    embedder = embedder or get_embedder()
    vector = (await embedder.embed([query]))[0]
    rows = await chunks_repo.search(db, embedding=vector, dept_ids=dept_ids, owner_id=owner, k=k)
    return format_context(rows) if rows else None
