"""Doc summarization for RAG ingest enrich B (E7, knowledge-rag.md §6.2).

At ingest, summarize the whole markdown once so a high-level query can match the *document* (not
just a low-level chunk). The summary does three jobs at once (§6.3): better high-level retrieval,
a coarse "which file" layer instead of a graph, and context for the answer LLM (E8). It is stored
on `documents.summary` **and** embedded as a summary-chunk.

Best-effort by design: the summary is derived metadata, rebuildable from the markdown (the §0
rule), so a missing provider / error / timeout returns None and ingest still stores the chunks —
summarization must never fail an ingest. The summarizing model is **injected** (any object with
`complete(model, messages, tools) -> LLMResult`, resolved by the worker from `llm_connections`
role 'summarize'), keeping this testable with a fake and the model config-driven (no-hardcode).

No SQL here (repositories) and no FastAPI types (routers) — §2.1.
"""
from __future__ import annotations

import logging

log = logging.getLogger("pikaos.engine.summarize")

# A short, retrieval-friendly summary — enough to match a high-level query and brief the answer
# LLM, not a full rewrite. Kept provider-agnostic (plain system+user turns like a real adapter).
SUMMARY_SYSTEM = (
    "You summarize a document for a search index. Write a concise summary (3-6 sentences) that "
    "captures what the document is about, its key topics, and any names/terms someone might search "
    "for. Output only the summary text — no preamble, headings, or markdown."
)


async def summarize_document(
    provider, *, title: str, markdown: str, max_input_chars: int
) -> str | None:
    """Summarize `markdown` (a document titled `title`) via the injected LLM `provider`.

    Only the first `max_input_chars` are sent (bounds cost/latency on a long doc). Returns the
    trimmed summary text, or None when there's nothing to summarize or the call fails — the caller
    treats None as "no summary" and proceeds, never raising."""
    text = (markdown or "").strip()
    if not text:
        return None
    excerpt = text[: max(0, max_input_chars)] if max_input_chars else text
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": f"Title: {title or '(untitled)'}\n\n{excerpt}"},
    ]
    try:
        result = await provider.complete(model="", messages=messages, tools=[])
    except Exception:
        # A summarize failure is non-fatal: the chunks still embed without it.
        log.exception("summarize failed for %r — ingesting without a summary", title)
        return None
    summary = (getattr(result, "text", "") or "").strip()
    return summary or None
