"""KnowledgeRetriever — this plugin's implementation of the engine's `Retriever` protocol.

The engine (Base) injects RAG through a runtime slot (agent_runner.set_engine_runtime) instead of
importing knowledge, so the Base never depends on this plugin (modularity §2). The worker wires this
in ONLY when the knowledge plugin is active; a build without knowledge passes `retriever=None` and the
agent simply runs without retrieved context. Thin wrapper over retrieval_service.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from . import retrieval_service


class KnowledgeRetriever:
    """Structural match for agent_runner.Retriever — no import the other way."""

    async def retrieve_context(
        self, db: AsyncSession, *, owner_id, run_input: dict | None, k: int
    ) -> str:
        query = retrieval_service.query_from_input(run_input)
        return await retrieval_service.context_for_run(db, owner_id=owner_id, query=query, k=k)
