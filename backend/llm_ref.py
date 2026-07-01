"""Access the configured LLM through the Core `ai.LLM` DI contract — NEVER import the `ai` sibling
plugin directly (§2.3). The factory is resolved from the container at this plugin's `register()` and
stashed here; services build a role-specific provider via `provider_for(role)`.
"""
from __future__ import annotations

_factory = None


def set_factory(factory) -> None:
    """Called from register() with the container-resolved `ai.LLM` factory (or None if the ai plugin is
    not enabled)."""
    global _factory
    _factory = factory


def provider_for(role: str):
    """A provider for a system role ("answer" | "summarize" | …) exposing
    `async complete(*, model, messages, tools)`. Raises if knowledge was enabled without the `ai` plugin
    (a misconfiguration the install dependency-request — `dependencies: ["ai"]` — normally prevents)."""
    if _factory is None:
        raise RuntimeError("LLM unavailable — knowledge depends on the 'ai' plugin; enable it")
    return _factory(role=role)
