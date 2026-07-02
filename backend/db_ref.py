"""DB session factory for the knowledge plugin — resolved from the `postgres.Connection` DI contract.

The zero-datastore kernel owns no engine (no `app.core.db.SessionLocal`): the postgres Tool creates the
engine + session factory and binds them under `postgres.Connection`. This plugin's register() resolves
that contract and stashes the factory here; the arq ingest job opens a session via `new_session()` at call
time. `postgres` is a declared dependency, so it is always bound when knowledge is enabled — mirrors the
sibling `storage_ref` / `llm_ref` / `queue_ref` holders.
"""
from __future__ import annotations

_sf = None  # async_sessionmaker, bound from postgres.Connection at register()


def set_factory(conn) -> None:
    """Wire the session factory from the postgres.Connection contract (called by the plugin's register())."""
    global _sf
    _sf = conn["session_factory"] if conn else None


def new_session():
    """Open a new AsyncSession (async context manager) from the bound factory. Raises if postgres is
    unbound — a declared dependency, so this only fires on a genuine misconfiguration."""
    if _sf is None:
        raise RuntimeError("knowledge: postgres.Connection not bound — enable the postgres tool")
    return _sf()
