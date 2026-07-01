"""RAG answer service (E8, knowledge-rag.md §6.5): search → answer with citations.

The query side of knowledge-base v1 ("upload a file → ask → answer with sources"). Take a question,
(best-effort) rewrite it into a search query, retrieve the top-k codex chunks the user may read
(reusing `knowledge_service.search_documents` — the same embed + permission scope as `/search`),
feed them to the answer LLM as numbered context, and return the synthesized answer plus the sources
it can cite. Citations are nearly free: search already returns document_name/heading/score per chunk.

The answer + rewrite models resolve via `llm_connections` (role 'answer') so they're config-driven
(no-hardcode); with no real provider the stub answers, so the endpoint still works offline (just
without real synthesis). Runs in the web process like `/search` — it embeds the query here.

No SQL (repositories) and no FastAPI types (routers) — §2.1.
"""
from __future__ import annotations

import logging

from . import knowledge_service
from .embeddings import Embedder
from .retrieval_service import format_context

log = logging.getLogger("pikaos.knowledge.answer")

ANSWER_SYSTEM = (
    "You answer the user's question using ONLY the provided codex context. Cite the sources you "
    "use inline as [n], matching the numbered context blocks. If the context does not contain the "
    "answer, say you don't have enough information — never invent facts."
)
REWRITE_SYSTEM = (
    "Rewrite the user's question into a single concise search query that captures its intent and "
    "key terms. Output only the rewritten query — no quotes, no explanation."
)


async def _rewrite(provider, question: str) -> str:
    """Best-effort query rewrite; falls back to the original question on any failure/empty result."""
    try:
        result = await provider.complete(
            model="", tools=[],
            messages=[{"role": "system", "content": REWRITE_SYSTEM},
                      {"role": "user", "content": question}],
        )
    except Exception:
        log.exception("query rewrite failed — using the original question")
        return question
    rewritten = (getattr(result, "text", "") or "").strip()
    return rewritten or question


def build_sources(rows: list[dict]) -> list[dict]:
    """Citation list parallel to the [n] markers `format_context` emits (1-based, same order)."""
    return [
        {"n": i, "document_id": r["document_id"], "document_name": r.get("document_name", ""),
         "heading": r.get("heading", ""), "score": float(r.get("score", 0.0))}
        for i, r in enumerate(rows, start=1)
    ]


async def answer_question(
    db, *, embedder: Embedder, provider, user, question: str, k: int, rewrite: bool = True,
) -> dict:
    """Answer `question` from the codex, scoped to what `user` may read. Returns
    `{answer, sources, rewritten_query, used_chunks}`. An empty question, or no matching chunk,
    yields a graceful 'no information' answer with no sources — this never raises."""
    question = (question or "").strip()
    if not question:
        return {"answer": "", "sources": [], "rewritten_query": "", "used_chunks": 0}

    search_query = await _rewrite(provider, question) if rewrite else question
    rows = await knowledge_service.search_documents(
        db, embedder=embedder, user=user, query=search_query, k=k
    )
    if not rows:
        return {"answer": "I don't have any indexed knowledge that answers that.",
                "sources": [], "rewritten_query": search_query, "used_chunks": 0}

    context = format_context(rows)
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
    ]
    try:
        result = await provider.complete(model="", messages=messages, tools=[])
        answer = (getattr(result, "text", "") or "").strip()
    except Exception:
        log.exception("answer synthesis failed for question %r", question)
        answer = ""
    return {"answer": answer, "sources": build_sources(rows),
            "rewritten_query": search_query, "used_chunks": len(rows)}
