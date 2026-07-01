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
    """Bind `knowledge.Retriever` into the DI container, and stash the `minio.Storage` facade resolved
    from the container so this plugin's services reach storage through the contract — never importing the
    sibling `minio` plugin (§2.3)."""
    from ...core.contracts import RETRIEVER, STORAGE
    from .retriever import KnowledgeRetriever
    from . import storage_ref

    ctx.container.bind(RETRIEVER, KnowledgeRetriever())
    storage_ref.set_storage(ctx.container.resolve(STORAGE))


__all__ = ["router", "jobs", "register"]
