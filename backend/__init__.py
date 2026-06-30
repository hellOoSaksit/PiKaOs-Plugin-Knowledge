"""Knowledge plugin — codex/documents + RAG (ingest · chunk · embed · search · answer).

A **plugin** (off unless ENABLED_MODULES lists `knowledge`). Was flat under app/routers/knowledge.py +
app/services/{ingestion,retrieval,summarize,answer,knowledge}_service.py + chunking/converters +
app/repositories/{doc_chunks,documents}.py; moved here per [extraction-plan.md]. The shared
`embeddings` + the `vector(N)` column type stay in the **Base** (used by db/models/config), so this
plugin depends on the Base — never the reverse. The engine consumes RAG only through the `Retriever`
contract this plugin `register()`s into the DI container (Phase 3), so the Base never imports this plugin.

Package surface the Loader looks for (plugin-architecture.md §5/§10):
  router    — mounted by modules.register_routers when this plugin is enabled
  register  — binds the `knowledge.Retriever` contract into the container
  jobs      — the arq job(s) the worker runs when this plugin is enabled
"""
from .jobs import ingest_document
from .router import router

jobs = [ingest_document]


def register(ctx) -> None:
    """Bind this plugin's provided contract — `knowledge.Retriever` (manifest `provides`) — into the DI
    container. The engine resolves it at worker startup; neither side imports the other (§5). Imported
    lazily so merely listing the plugin never drags the impl in until it's actually wired."""
    from ...core.contracts import RETRIEVER
    from .retriever import KnowledgeRetriever

    ctx.container.bind(RETRIEVER, KnowledgeRetriever())


__all__ = ["router", "jobs", "register"]
