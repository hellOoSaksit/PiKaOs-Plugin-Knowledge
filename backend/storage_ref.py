"""Access the object-storage facade through the Core `minio.Storage` DI contract — NEVER import the
`minio` sibling plugin directly (§2.3). The facade is resolved from the container at this plugin's
`register()` and stashed here; services read it via `get_storage()`.
"""
from __future__ import annotations

_storage = None


def set_storage(facade) -> None:
    """Called from register() with the container-resolved `minio.Storage` facade (or None if the minio
    tool is not enabled)."""
    global _storage
    _storage = facade


def get_storage():
    """The object-storage facade. Raises if knowledge was enabled without the `minio` tool (a
    misconfiguration the install dependency-request normally prevents)."""
    if _storage is None:
        raise RuntimeError(
            "object storage unavailable — knowledge depends on the 'minio' tool; enable it")
    return _storage
