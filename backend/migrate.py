"""Install-time schema step for the knowledge plugin.

The kernel migration runner (`scripts.migrate_plugins`) calls `migrate(engine, session_factory)` for
each enabled plugin after Core's Alembic baseline. Knowledge owns `documents` + `doc_chunks` on its own
`Base` metadata (models.py), so here we enable pgvector, create the tables, and build the HNSW index.

We deliberately use a DEDICATED, codec-free engine (not the passed-in app engine): `app.core.db.engine`
registers pgvector's asyncpg codec on every connect, which requires the `vector` type to already exist —
a chicken/egg when WE are the one creating the extension. The extension is created in its own committed
transaction so the follow-up connection sees the type before `create_all` emits the `vector(N)` column.

No seed: the knowledge store starts empty. Functional/fresh-DB model — plain create_all, not a versioned
Alembic history yet.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from ...core.config import settings
from .models import Base


async def migrate(engine, session_factory) -> None:
    # NullPool + no pgvector codec: each begin() gets a FRESH connection. asyncpg caches its type
    # registry per connection at connect time, so `create_all` (which emits the `vector(N)` column) must
    # run on a connection opened AFTER the extension is committed — otherwise a reused, pre-extension
    # connection reports "unknown type: public.vector".
    ddl_engine = create_async_engine(settings.database_url, poolclass=NullPool)
    try:
        # Enable the extension first, in its own transaction, so the next connection knows `vector`.
        async with ddl_engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        async with ddl_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Approximate-nearest-neighbour index for cosine search (`<=>`). HNSW builds fine on an empty
            # table; raw + IF NOT EXISTS keeps the whole step idempotent on every boot.
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_doc_chunks_embedding ON doc_chunks "
                "USING hnsw (embedding vector_cosine_ops)"
            ))
    finally:
        await ddl_engine.dispose()
