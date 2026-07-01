"""Knowledge plugin — codex/documents + RAG (ingest · chunk · embed · search · answer).

A **plugin** (off unless ENABLED_MODULES lists `knowledge`). Data layer (knowledge extraction): this
plugin OWNS `documents` + `doc_chunks` + the pgvector `Vector` column type + the `embeddings` service —
all on its own `Base` metadata (models.py), created by `migrate.migrate()` (CREATE EXTENSION vector →
create_all → HNSW index), run per enabled plugin by the kernel's `scripts.migrate_plugins`. Core's
Alembic no longer owns these tables. Cross-plugin refs (owner_id/department_id → auth.users/departments)
are logical UUIDs, no FK. The engine consumes RAG only through the `Retriever` contract this plugin
`register()`s into the DI container, so Core never imports this plugin.

Package surface the Loader looks for (plugin-architecture.md §5/§10):
  router    — mounted by modules.register_routers when this plugin is enabled
  register  — binds the `knowledge.Retriever` contract into the container
  jobs      — the arq job(s) the worker runs when this plugin is enabled
  migrate   — install-time schema step (extension + create_all + HNSW), run by scripts.migrate_plugins
"""
from .jobs import ingest_document
from .router import router

jobs = [ingest_document]


def register(ctx) -> None:
    """Bind `knowledge.Retriever` into the DI container, and stash the `minio.Storage` facade + `ai.LLM`
    factory resolved from the container so this plugin's services reach storage + the LLM through the
    contracts — never importing the sibling `minio`/`ai` plugins (§2.3). Knowledge declares
    `dependencies: ["ai", "minio"]`, so both are registered before this runs."""
    from ...core.contracts import AI_LLM, RETRIEVER, STORAGE
    from .retriever import KnowledgeRetriever
    from . import llm_ref, storage_ref

    ctx.container.bind(RETRIEVER, KnowledgeRetriever())
    storage_ref.set_storage(ctx.container.resolve(STORAGE))
    llm_ref.set_factory(ctx.container.resolve(AI_LLM))


__all__ = ["router", "jobs", "register"]
