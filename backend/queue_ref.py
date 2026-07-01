"""Access the arq job queue through the Core `redis.Queue` DI contract — NEVER import a kernel queue
module or the `redis` sibling tool directly (§2.3). The enqueue handle is resolved from the container at
this plugin's `register()` and stashed here; the router enqueues ingestion via `enqueue()`.
"""
from __future__ import annotations

_queue = None


def set_queue(handle) -> None:
    """Called from register() with the container-resolved `redis.Queue` handle (or None if the redis tool
    is not enabled)."""
    global _queue
    _queue = handle


async def enqueue(job: str, *args) -> bool:
    """Best-effort enqueue. Returns False when the redis tool is disabled (queue unbound) or Redis is
    down — the caller degrades (e.g. the document is stored, just not indexed yet)."""
    if _queue is None:
        return False
    return await _queue.enqueue(job, *args)
